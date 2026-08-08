"""No-reference (production-shape) evaluation — the operating mode we ship into.

Every published number in this project (docs/REPORT.md §3, §4) is measured on
traces that carry BOTH a ground-truth reference and gold-chunk annotations. A
production trace carries neither: nobody labels the answer, and nobody marks
which retrieved chunk holds it. The architecture is *designed* for that case —
``FeatureVector.missingness_signature()`` appends ``|noref``,
``RetrievalExtractor._gold_index`` falls back to NLI-style support matching, and
``gold_recall_in_context`` / ``is_correct`` become missing rather than 0.0 — but
"supported" and "measured" are different claims. This module measures it.

Two experiments are reported, mirroring the (a)/(b) split in
:mod:`tokentrace.eval.ablations`, because they answer different questions:

**(a) shipped-engine transfer** — train the engine exactly as shipped, then
evaluate on stripped traces. Nothing about the engine changes; only the input
loses its reference. *This is the real production scenario* and is the headline
number. Note what "as shipped" now includes: ``train_engine`` fits calibration
from the cal split's reference-bearing rows AND a reference-stripped pass over
them (``fit_noref=True``, the default everywhere), so the shipped calibrator has
genuine ``|noref`` maps and a stripped trace is served one instead of falling
through the admission guard to the raw sigmoid. The heads are still fitted on
reference-bearing training rows only.

**(b) retrained-on-stripped** — train and calibrate on stripped data too. This is
the ceiling: how much of the diagnosis is recoverable if you accept up front that
you will never have references.

Both are reported against the reference-bearing baseline, so the delta is
explicit rather than inferred.

**(c) frozen-mock transfer** — the same strip as (a), but with the simulated
model's decision *pinned* to the one it made on the reference-bearing original.
This is the artifact-controlled figure and the one the report should quote as the
production estimate; see :func:`freeze_mock_decisions` for why (a) is not.

**(d) black-box x no-reference** — (a) and (c) re-scored at :attr:`Tier.BLACK`.
(a)-(c) vary only the reference dimension while keeping full white-box mechanistic
capture, but a trace arriving through :mod:`tokentrace.ingest` has *neither* a
reference *nor* activations. Calling (a) "the operating mode the product will
actually run in" was therefore an assumption about one axis and a claim about two;
this condition measures the second axis instead of assuming it.

Metrics that cannot be computed without a reference are returned as the explicit
:data:`UNAVAILABLE` marker, never as 0.0 — see
:data:`GROUND_TRUTH_DEPENDENT_METRICS` for which ones and why.

One confound is measured rather than hoped away. ``MockModel._decide`` reads
``Chunk.gold`` as its OWN oracle (it derives ``gold_pos_frac`` from the flag, and
its dilution trigger depends on that), so removing the annotation changes the
*simulated model's* behaviour — which cannot happen in production, where a real
model never sees an annotation in the first place. ``confound_controls`` in
:func:`run_noref` therefore reports two half-strips that remove one thing each,
plus a count of how many rows the mock re-decides; the ``frozen_mock`` block then
*scores* the strip against an unchanged simulated model, so the artifact's cost is
measured rather than inferred from the half-strips.

Scope caveat, stated up front: the *labels* survive stripping because they come
from the injection harness, not from the reference. But those labels were only
obtainable because the harness's verification gates could read GT-derived
features at corpus-construction time. So this measures "how well would the
shipped engine have diagnosed these traces had it not seen their references" —
which is the right counterfactual — and NOT "you can evaluate diagnosis quality
in production". In production you have neither the reference nor the label.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Optional, Union

from tokentrace.core.types import (
    ALL_MODES,
    DiagnosisReport,
    FailureMode,
    Inference,
    LabeledInference,
    Tier,
)
from tokentrace.data.strip import strip_inference_in_place
from tokentrace.data.synthetic import build_dataset
from tokentrace.eval.ablations import _ece_full
from tokentrace.eval.benchmark import (
    family_dropout_schedule,
    run_reports,
    split_dataset,
    train_engine,
)
from tokentrace.eval.metrics import compute_metrics, gold_primary_set, predicted_primary
from tokentrace.models.base import ModelHandle
from tokentrace.signals.registry import SignalPipeline

#: Explicit marker for a metric that is *undefined* without a reference, as
#: opposed to a metric that legitimately measured zero. It is a string on purpose:
#: it is JSON-serializable, greppable, and it will loudly break any caller that
#: tries to format it as a float — which is exactly the failure we want, rather
#: than a silent 0.0 that reads as "the system scored nothing".
UNAVAILABLE = "UNAVAILABLE(no_ground_truth)"

#: Keys of :meth:`Metrics.as_dict` that carry a HIDDEN ground-truth dependency.
#:
#: Found by reading ``eval/metrics.py`` together with ``recommend/recommender.py``:
#: ``compute_metrics`` itself never touches ``inference.ground_truth``, so every
#: label-derived metric (accuracy, Top-3, conformal, per-mode P/R/F1, abstention,
#: debugging-time) stays computable. The recommendation family is the exception,
#: and the dependency is one call deep:
#:
#:   ``compute_metrics`` only scores a recommendation when ``r.validated is not
#:   None`` -> ``Recommender._maybe_validate`` only assigns ``validated`` inside
#:   ``if inference.has_ground_truth:`` -> with no reference the simulated
#:   intervention still runs, but its result cannot be checked for correctness, so
#:   ``validated`` stays None and every recommendation is skipped.
#:
#: The consequence is the reason this constant exists: ``rec_total`` falls to 0
#: and ``recommendation_precision`` is computed as ``0.0`` by the
#: ``if rec_total else 0.0`` guard. A reader sees 0.0 — a catastrophic-looking
#: score — where the truth is "not measurable". ``recommendation_n`` and
#: ``recommendation_negatives`` are also listed: they are structurally 0, so
#: reporting them as counts would imply zero interventions were *worth* scoring
#: rather than zero being *scoreable*.
GROUND_TRUTH_DEPENDENT_METRICS: tuple[str, ...] = (
    "recommendation_precision",
    "recommendation_n",
    "recommendation_negatives",
)

_StripTarget = Union[LabeledInference, Inference]


# --------------------------------------------------------------------------- #
# Stripping
# --------------------------------------------------------------------------- #
def strip_reference(obj: _StripTarget, *, drop_gold_flags: bool = True,
                    drop_ground_truth: bool = True) -> _StripTarget:
    """Return a deep copy of ``obj`` in the shape a production trace arrives in.

    Removes ``ground_truth`` and (by default) every ``Chunk.gold`` annotation.
    A copy, not an in-place edit, because the reference-bearing originals are the
    baseline the stripped run is measured against — mutating them would quietly
    delete the control group.

    **Labels are preserved deliberately.** ``LabeledInference.labels`` come from
    the injection harness's recipe, not from the reference, so diagnosis accuracy
    remains measurable after stripping. Stripping them too would not model
    production more faithfully; it would just make the experiment unscoreable.

    **``meta["_sim"]`` is preserved deliberately, and this is NOT leakage.**
    It looks like leakage and a reviewer will ask, so: ``_sim`` carries the mock
    model's world-model hints (the answer, the grounding fact, the distractor).
    The mock needs them to generate, resample and produce mechanistic captures at
    all — a real backend ignores ``_sim`` entirely and reads the same information
    out of its own weights and activations. Removing it would not simulate "no
    reference", it would simulate "no model".

    **Be precise about what that does and does not guarantee.** No module under
    ``signals/`` reads ``_sim`` directly, but that is not the same as "no leakage
    path", which an earlier revision of this docstring claimed. The extractors call
    ``model.capture()`` and ``model.sample()``, and ``MockModel._decide`` reads
    ``_sim["gold_fact"]`` — so ``gold_present`` (and through it
    ``gold_attention_ratio``, ``external_context_score``,
    ``parametric_knowledge_score`` and the logit-lens features) is reconstructed
    from ``_sim`` after ``Chunk.gold`` is gone. Measured: ``gold_present`` is True on
    the same 51 of 86 test rows before and after stripping. The reference string also
    survives verbatim in the rendered prompt on 54/86 rows and in the context on
    53/86 — necessarily so, since a production trace really does contain whatever
    text the model was shown.

    The honest framing is that this measures **"how well would the shipped engine
    have diagnosed these traces had it not been given their reference *metadata*"**,
    with a mock whose internal states stay as informative as a real model's. It is
    not a claim that every channel carrying answer-related information was severed.

    ``drop_gold_flags=False`` keeps the gold annotations and removes only the
    reference; ``drop_ground_truth=False`` does the mirror image. Neither is a
    shippable configuration — they exist as the two controls in :func:`run_noref`
    that separate "the diagnosis lost its reference" from "the *mock* lost an
    oracle it reads out of ``Chunk.gold``" (see ``confound_controls`` there).

    Gold flags are set to ``False`` rather than ``None``. ``None`` ("unknown
    whether this chunk bears the answer") is arguably the more faithful
    production value, but every consumer in the codebase tests ``gold`` for plain
    truthiness (``if c.gold``, ``[c for c in chunks if c.gold]``), so the two are
    behaviourally identical and ``False`` states the intent unambiguously.

    ``LabeledInference.verification`` is cleared alongside the reference. It holds
    the injection gate's readings — ``{'gate_passed': True, 'is_correct': 1.0,
    'gold_recall': ...}`` — which are GT-derived by construction, so leaving them on
    an object this function documents as "the shape a production trace arrives in"
    was a contradiction. Nothing on the signal or metrics path reads the field
    today, so this changes no number; it removes residue that a future reader (or a
    future feature) could mistake for legitimately available evidence. The
    *labels* are kept for the reason above — they are supervision, not reference —
    and the recipe name is kept because ``run_noref`` reports the mock's re-decisions
    per recipe.

    The trace-level mechanics live in :func:`tokentrace.data.strip.strip_inference_in_place`
    — one definition shared with ``eval/benchmark.train_engine``'s reference-stripped
    calibration pass, so "stripped" cannot mean two different things in the fit
    path and the eval path. This wrapper adds the ``LabeledInference``-aware part
    (the ``verification`` clearing) on top.
    """
    stripped = copy.deepcopy(obj)
    inference = stripped.inference if isinstance(stripped, LabeledInference) else stripped
    strip_inference_in_place(inference, drop_gold_flags=drop_gold_flags,
                             drop_ground_truth=drop_ground_truth)
    if drop_ground_truth and isinstance(stripped, LabeledInference):
        stripped.verification = {}
    return stripped


def strip_dataset(dataset: list[LabeledInference], **kwargs) -> list[LabeledInference]:
    """:func:`strip_reference` over a whole split, preserving order."""
    return [strip_reference(li, **kwargs) for li in dataset]


# --------------------------------------------------------------------------- #
# The mock-artifact control
# --------------------------------------------------------------------------- #
def _decision_key(inference: Inference) -> tuple:
    """Identity of a trace *as the simulated model sees it*, minus the reference.

    Everything ``MockModel._decide`` branches on except ``Chunk.gold`` and
    ``ground_truth``: the rendered prompt, the chunk texts in order, and the
    ``_sim`` world-model hints. Those two exclusions are the point — they are what
    stripping removes, so including them would give a trace and its stripped copy
    different keys and make the freeze a silent no-op.

    Everything else is included so a *counterfactual* stays a different trace. The
    recommender's interventions rebuild the prompt, the chunk list and ``_sim``
    (``clarify_prompt`` and ``decompose_question`` change ``_sim`` alone and leave
    the prompt byte-identical), so a key that ignored ``_sim`` would pin the
    simulated fix to the unfixed decision and quietly neuter every intervention.
    """
    chunks = tuple(c.text for c in (inference.retrieved_context or ()))
    sim = tuple(sorted((str(k), repr(v)) for k, v in inference.meta.get("_sim", {}).items()))
    return (inference.prompt, chunks, sim)


def freeze_mock_decisions(model: ModelHandle,
                          originals: list[_StripTarget]) -> Optional[ModelHandle]:
    """A handle whose simulated decision is pinned to the reference-bearing run.

    **Why this control is necessary, and why the half-strips cannot replace it.**
    ``MockModel._decide`` reads ``Chunk.gold``: it derives ``gold_pos_frac`` from the
    flag's position, and the dilution trigger requires ``0.2 <= gold_pos_frac <=
    0.8``. Delete the flags and ``gold_pos_frac`` collapses to 0.0, the trigger stops
    firing, and the mock *answers a dilution row correctly*. The diagnosis is then
    scored against a different simulated model, not against the same model observed
    with less metadata — and no real model can experience that, because no real model
    is shown your eval metadata in the first place. The "annotations removed,
    reference kept" half-strip does not isolate it either: with the reference intact
    ``dil.present_wrong`` still fires and re-diagnoses the re-decided rows, so that
    condition *masks* the artifact rather than measuring it.

    This function removes the artifact directly. It replays, for each trace, the
    decision the mock made on the reference-bearing original, so the stripped run
    scores the *same* simulated behaviour under strictly less metadata — the
    counterfactual the production claim is actually about.

    Returns ``None`` for any backend without a ``_decide`` (i.e. every real one).
    There is nothing to control for there: an HF or GGUF model never saw the
    annotation, so stripping it cannot change what the model did.
    """
    decide = getattr(model, "_decide", None)
    if decide is None:
        return None

    table = {}
    for obj in originals:
        inference = obj.inference if isinstance(obj, LabeledInference) else obj
        table[_decision_key(inference)] = decide(inference)

    # A shared mutable counter, exposed as `_replay_hits` on the handle. It exists
    # so a caller can PROVE a scoring pass actually went through the pinned handle:
    # adversarial verification showed the black-tier frozen block could be
    # relabelled from the transfer metrics with nothing going red, because on this
    # corpus the two conditions coincide metric-for-metric. Provenance has to be
    # observable, not inferred from numbers that happen to match.
    hits = {"replayed": 0}

    def replay(inference: Inference):
        pinned = table.get(_decision_key(inference))
        if pinned is not None:
            hits["replayed"] += 1
            return pinned
        return decide(inference)

    frozen = copy.copy(model)
    # An INSTANCE attribute, deliberately: every call site inside the mock goes
    # through ``self._decide(...)``, which finds the instance attribute before the
    # class method, and both ``bound_to`` and ``with_tier`` are ``copy.copy`` — so
    # the pin survives the two wrappers the pipeline puts between us and the model.
    frozen._decide = replay
    frozen._replay_hits = hits
    return frozen


def mark_unavailable(metrics: dict) -> dict:
    """Replace the ground-truth-dependent metrics with :data:`UNAVAILABLE`.

    Applied to any metrics dict computed on stripped traces. See
    :data:`GROUND_TRUTH_DEPENDENT_METRICS` for the derivation of the list.
    """
    out = dict(metrics)
    for key in GROUND_TRUTH_DEPENDENT_METRICS:
        if key in out:
            out[key] = UNAVAILABLE
    return out


# --------------------------------------------------------------------------- #
# Diagnostics used by the report
# --------------------------------------------------------------------------- #
def _confusion(reports: list[DiagnosisReport], data: list[LabeledInference]) -> dict[str, int]:
    """``"gold_root -> predicted_primary": count``, JSON-safe.

    Aggregate accuracy cannot answer the question this module is testing (*which*
    mode absorbs the collapsed one), so the confusion is reported directly rather
    than left to be inferred from per-mode recall.
    """
    counts: Counter = Counter()
    for report, li in zip(reports, data):
        gold_set = gold_primary_set(li)
        gold = "/".join(m.value for m in gold_set) or "(clean)"
        pred = predicted_primary(report)
        counts[f"{gold} -> {pred.value if pred else '(abstain)'}"] += 1
    return dict(sorted(counts.items()))


def _flow_from(confusion: dict[str, int], gold: str) -> dict[str, int]:
    """Where rows whose true root is ``gold`` actually landed."""
    return {k.split(" -> ")[1]: v for k, v in confusion.items() if k.split(" -> ")[0] == gold}


def _calibration_path(engine, data: list[LabeledInference], model: ModelHandle,
                      pipeline: SignalPipeline, tier: Tier = Tier.WHITE) -> dict:
    """Which calibration map actually serves these traces.

    ``Calibrator.transform`` walks ``exact signature -> coarse ref/noref ->
    global -> raw sigmoid`` and records how far it fell in ``last_fallback``.
    That counter is the only observable, so we replay the transform per
    (row, unmasked mode) and tally it. For a stripped trace, levels 0 and 1 are
    by construction |noref-fitted maps; level 2 (``__global__``) is a pooled map
    that is only legitimate when the pool was genuinely shaped by reference-free
    records (``global_admits_noref``) — served without that, it is a map fitted
    on reference-bearing records under another name, precisely the silent
    pooling the ``|noref`` signature was introduced to prevent.

    The levels are tallied **per reference-availability class**, not just in
    aggregate. An aggregate cannot answer the question: on a mixed split the
    level-2 hits could all belong to reference-*bearing* rows while every stripped
    row fell through to the raw sigmoid, and the flag below would still read True.
    """
    levels: Counter = Counter()
    noref_levels: Counter = Counter()
    signatures: Counter = Counter()
    calibrator = getattr(engine, "calibrator", None)
    for li in data:
        fv = pipeline.run(li.inference, model.with_tier(tier))
        signature = fv.missingness_signature()
        signatures[signature] += 1
        if calibrator is None:
            continue
        is_noref = signature.endswith("|noref")
        z = engine.raw_logits(fv, li.inference)
        for mode in ALL_MODES:
            if z[mode] <= -49.0:          # hard-masked (non-RAG); nothing to calibrate
                continue
            calibrator.transform(mode, signature, z[mode])
            levels[calibrator.last_fallback] += 1
            if is_noref:
                noref_levels[calibrator.last_fallback] += 1

    fitted = sorted({sig for _, sig in calibrator.maps}) if calibrator is not None else []
    named = {0: "exact_signature", 1: "coarse_ref_or_noref", 2: "global_pooled", 3: "raw_sigmoid"}
    # For a |noref row, level 0 is by construction the exact `...|noref` map and
    # level 1 the coarse `__noref__` bucket — both fitted from reference-free
    # records only. This is the count the "stripped traces get real |noref
    # calibration" claim rests on, so it is reported directly rather than left to
    # be inferred from the level tally.
    noref_served_by_noref_maps = int(noref_levels.get(0, 0) + noref_levels.get(1, 0))
    # `global_admits_noref` is the calibrator's own admission predicate (it replaced
    # the older presence test `saw_noref`). Consulting it is what separates the BUG
    # -- a reference-free trace served a pool that is reference-bearing data under
    # another name -- from the LEGITIMATE case, where the pool really was fitted from
    # enough reference-free records to have been shaped by them. Without it a
    # correctly-mixed calibrator, e.g. the (b) retrained engine, is reported as
    # buggy for doing exactly what the guard is supposed to allow.
    admits = bool(getattr(calibrator, "global_admits_noref", False)) if calibrator else False
    return {
        "observed_signatures": dict(signatures),
        "signatures_with_a_fitted_map": fitted,
        "fallback_levels": {named[k]: v for k, v in sorted(levels.items())},
        "fallback_levels_noref_rows_only": {named[k]: v for k, v in sorted(noref_levels.items())},
        "noref_rows_served_by_noref_fitted_maps": noref_served_by_noref_maps,
        "global_pool_admits_noref": admits,
        "n_noref_calibration_records": int(getattr(calibrator, "n_noref", 0)) if calibrator else 0,
        # The bug condition: a |noref trace served by the pooled global map while
        # that pool is (near-)entirely reference-bearing.
        #
        # Note honestly what this can and cannot be: against the CURRENT Calibrator
        # it is structurally unreachable, because `transform` withholds the
        # __global__ rung under exactly the same predicate. It is a regression
        # detector for that guard, not an observation about the run — so a False here
        # is evidence the guard is in place, NOT evidence that pooling was measured
        # and found absent. (Its own test therefore drives it with a stub calibrator;
        # a flag that cannot come out True against any input is the kind of
        # unfalsifiable metric this project keeps having to retract.)
        "noref_served_by_pooled_map": bool(noref_levels.get(2, 0) > 0 and not admits),
        # Reported separately because it is not a bug: the pool contains enough
        # reference-free records that serving it to a reference-free trace is the
        # guard working as designed.
        "noref_served_by_admitted_pooled_map": bool(noref_levels.get(2, 0) > 0 and admits),
    }


def _delta(baseline: dict, other: dict, keys: tuple[str, ...]) -> dict:
    """Signed change vs the reference-bearing baseline for comparable metrics."""
    out = {}
    for k in keys:
        b, o = baseline.get(k), other.get(k)
        if isinstance(b, (int, float)) and isinstance(o, (int, float)):
            out[k] = round(o - b, 4)
    return out


_COMPARABLE = (
    "diagnosis_accuracy", "top3_accuracy", "conformal_coverage", "conformal_set_size",
    "abstention_rate", "healthy_abstention_rate", "declined_rate",
    "debugging_time_reduction", "mean_root_rank",
)


def _per_mode_table(configs: dict[str, dict]) -> dict:
    """mode -> config -> {precision, recall, f1, support}.

    Pivoted this way because the question under test is per-mode ("does
    retrieval-failure vs context-dilution collapse?"), and a config-major layout
    makes that comparison a cross-dict lookup.
    """
    table: dict[str, dict] = {}
    for mode in ALL_MODES:
        table[mode.value] = {
            name: metrics["per_mode"][mode.value] for name, metrics in configs.items()
        }
    return table


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #
def run_noref(
    model: ModelHandle,
    dataset: Optional[list[LabeledInference]] = None,
    pipeline: Optional[SignalPipeline] = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    tier: Tier = Tier.WHITE,
    ingest_tier: Tier = Tier.BLACK,
    fit_noref: bool = True,
) -> dict:
    """Measure the engine on production-shape (no-reference) traces.

    Same seeds and same family-dropout tier schedule as
    :func:`tokentrace.eval.ablations.run_ablations`, so the numbers sit alongside
    the published table rather than beside it.

    ``tier`` is the observability tier every headline condition is scored at, and it
    defaults to WHITE so (a)/(b) remain comparable with the published tables.
    ``ingest_tier`` is the *second* condition: the tier a trace arriving through
    :mod:`tokentrace.ingest` actually has. Both are scored, because varying only the
    reference dimension and then calling the result "production" asserts the other
    dimension is free — and §4.2(b) measures that dropping the retrieval family alone
    costs more than half the accuracy, so freeness is not a safe assumption.
    """
    pipeline = pipeline or SignalPipeline()
    dataset = dataset if dataset is not None else build_dataset(model, pipeline, seeds=seeds)
    train, cal, test = split_dataset(dataset)

    # Identical family-dropout schedule to run_ablations / train_and_evaluate /
    # TokenTrace.default — now literally the same function. The entry points
    # previously trained differently while the docs presented their tables as
    # interchangeable; this experiment must not reintroduce that.
    train_tiers = family_dropout_schedule(len(train))

    # (a) the SHIPPED engine: heads trained on reference-bearing data; calibration
    # fitted from the cal rows plus their reference-stripped twins (train_engine's
    # fit_noref default), so it carries real |noref maps — this is the engine
    # TokenTrace.default() builds, which is the point of the condition.
    # fit_noref is a passthrough so the guard-cost machinery below stays
    # FALSIFIABLE: with the default the guard is inert and every delta is a
    # measured zero, which is indistinguishable from a broken measurement.
    # run_noref(fit_noref=False) must reproduce the nonzero cost table — a
    # test pins that, so neutering the force-admission now goes red.
    shipped = train_engine(train, cal, model, pipeline, train_tiers=train_tiers,
                           fit_noref=fit_noref)

    stripped_train = strip_dataset(train)
    stripped_cal = strip_dataset(cal)
    stripped_test = strip_dataset(test)

    # (b) the ceiling: everything refit on stripped data, including calibration.
    refit = train_engine(stripped_train, stripped_cal, model, pipeline, train_tiers=train_tiers)

    def score(engine, data, handle: Optional[ModelHandle] = None,
              at: Optional[Tier] = None) -> tuple[dict, dict]:
        # `is None`, not `or`: Tier.BLACK == 0 is FALSY, so `at or tier` silently
        # scored the black-box condition at the default white tier and reported it
        # as black. Exactly the kind of quiet substitution this module exists to
        # catch, so it is spelled out rather than left to truthiness.
        handle = model if handle is None else handle
        at = tier if at is None else at
        reports = run_reports(engine, data, handle, at, pipeline)
        return compute_metrics(reports, data).as_dict(), _confusion(reports, data)

    baseline, baseline_cm = score(shipped, test)
    transfer, transfer_cm = score(shipped, stripped_test)
    retrained, retrained_cm = score(refit, stripped_test)

    # (c) the artifact-controlled transfer: same strip as (a), but the simulated
    # model's decision is pinned to the one it made on the reference-bearing
    # original, so the diagnosis is scored against an UNCHANGED simulated model.
    # See freeze_mock_decisions for why (a) alone cannot be read as production loss.
    frozen_model = freeze_mock_decisions(model, test)
    frozen: Optional[dict] = None
    frozen_cm: dict[str, int] = {}
    frozen_replays: Optional[int] = None
    if frozen_model is not None:
        _before = frozen_model._replay_hits["replayed"]
        frozen, frozen_cm = score(shipped, stripped_test, handle=frozen_model)
        # Provenance, not inference: proves this block scored through the PINNED
        # handle. The frozen and transfer conditions can coincide metric-for-metric
        # on this corpus, so without this count a relabel is undetectable.
        frozen_replays = frozen_model._replay_hits["replayed"] - _before

    # Controls that make the headline interpretable. The mock reads Chunk.gold as
    # its OWN oracle (MockModel._decide derives gold_pos_frac from the flag, and
    # the dilution trigger depends on it), so dropping the flag perturbs the
    # simulated model's behaviour — something that cannot happen in production,
    # where a real model never sees an annotation. These two isolates separate
    # that artifact from the genuine loss of diagnostic information.
    # mark_unavailable applies to ANY metrics dict computed on reference-stripped
    # traces, this control included: it drops the reference, so every recommendation
    # goes unscored and `recommendation_precision` fell out of compute_metrics as a
    # bare 0.0 — the "catastrophic-looking score where the truth is 'not measurable'"
    # that the UNAVAILABLE marker exists to prevent, emitted in the same output that
    # explains why it must not be.
    reference_only = mark_unavailable(score(shipped, strip_dataset(test, drop_gold_flags=False))[0])
    # This one KEEPS the reference, so its recommendation metrics are real.
    annotations_only = score(shipped, strip_dataset(test, drop_ground_truth=False))[0]

    mock_decisions_changed: Counter = Counter()
    if hasattr(model, "_decide"):        # mock backend only; real handles have no world model
        fields = ("answer", "correct", "grounded", "parametric", "ambiguous",
                  "reasoning_failed", "gold_present", "overrode")
        for original, s in zip(test, stripped_test):
            before, after = model._decide(original.inference), model._decide(s.inference)
            if any(getattr(before, f) != getattr(after, f) for f in fields):
                mock_decisions_changed[original.injection_recipe or "(unknown)"] += 1

    # (d) the OTHER production axis. A trace from tokentrace.ingest carries neither a
    # reference nor activations, so the reference-only sweep above bounds one axis
    # and asserts the other. Scored on the same shipped engine (it was trained across
    # the mixed-tier schedule, so black-box input is in-distribution for it) with the
    # handle down-capped: no mechanistic capture, no logprob confidence.
    ingest: dict[str, object] = {}
    if ingest_tier != tier:
        bb_baseline, bb_baseline_cm = score(shipped, test, at=ingest_tier)
        bb_transfer, bb_transfer_cm = score(shipped, stripped_test, at=ingest_tier)
        bb_frozen, bb_frozen_cm = (None, {})
        bb_frozen_replays: Optional[int] = None
        if frozen_model is not None:
            _bb_before = frozen_model._replay_hits["replayed"]
            bb_frozen, bb_frozen_cm = score(shipped, stripped_test, handle=frozen_model,
                                            at=ingest_tier)
            bb_frozen_replays = frozen_model._replay_hits["replayed"] - _bb_before
        ingest = {
            "tier": ingest_tier.label,
            "note": "The condition tokentrace.ingest actually delivers: no reference AND "
                    "no mechanistic capture. Reported next to the same-tier baseline so "
                    "the two losses can be separated instead of summed.",
            "baseline": bb_baseline,
            "transfer": mark_unavailable(bb_transfer),
            "frozen_mock": mark_unavailable(bb_frozen) if bb_frozen is not None else None,
            "delta_vs_same_tier_baseline": {
                "transfer": _delta(bb_baseline, bb_transfer, _COMPARABLE),
                "frozen_mock": _delta(bb_baseline, bb_frozen, _COMPARABLE)
                if bb_frozen is not None else None,
            },
            "delta_vs_full_capture": {
                "baseline": _delta(baseline, bb_baseline, _COMPARABLE),
                "transfer": _delta(transfer, bb_transfer, _COMPARABLE),
                "frozen_mock": _delta(frozen, bb_frozen, _COMPARABLE)
                if (frozen is not None and bb_frozen is not None) else None,
            },
            "confusion": {"baseline": bb_baseline_cm, "transfer": bb_transfer_cm,
                          "frozen_mock": bb_frozen_cm},
            # > 0 proves the frozen condition at THIS tier was genuinely scored
            # through the pinned handle rather than relabelled from transfer.
            "frozen_decisions_replayed": bb_frozen_replays,
            "calibration_path": _calibration_path(shipped, stripped_test, model, pipeline,
                                                  ingest_tier),
        }

    # What the __global__ admission guard costs, regenerated rather than remembered.
    # The guard withholds the pooled map from reference-free traces when that pool
    # was not shaped by enough reference-free records; forcing `n_noref` over the
    # admission threshold reproduces the unguarded ladder exactly, so the cost is a
    # measurement instead of a historical note nobody can re-run. Now that
    # train_engine fits real |noref maps by default, the shipped calibrator's pool
    # IS legitimately mixed, so on the default path the guard has nothing left to
    # withhold and this measurement should come out ~0 on every metric — which is
    # measured (both ladders are scored), not asserted: revert `fit_noref` and the
    # deltas reappear. It is left unconditional for exactly that reason.
    calibrator = getattr(shipped, "calibrator", None)
    guard_cost: dict[str, object] = {}
    if calibrator is not None:
        saved = calibrator.n_noref
        try:
            calibrator.n_noref = max(saved, calibrator.min_samples)
            unguarded, unguarded_cm = score(shipped, stripped_test)
            unguarded_ece = {
                "uncalibrated": _ece_full(shipped, stripped_test, model, pipeline, tier,
                                          calibrated=False),
                "calibrated": _ece_full(shipped, stripped_test, model, pipeline, tier,
                                        calibrated=True),
            }
            # How many production-shape probabilities the pooled reference-bearing map
            # actually served before the guard. Regenerated here rather than quoted
            # from memory: with the guard live the guarded path reports 0, so the
            # published count has no other way to stay checkable.
            unguarded_path = _calibration_path(shipped, stripped_test, model, pipeline, tier)
        finally:
            calibrator.n_noref = saved
        guard_cost = {
            "note": "Transfer condition with the __global__ rung force-admitted vs the "
                    "live ladder. When the live calibrator already admits reference-free "
                    "traces (guard_live_admits_noref=True — the default now that "
                    "train_engine fits real |noref maps), forcing admission changes "
                    "nothing and every delta below is a measured zero: the guard is "
                    "inert. When it does not (fit_noref=False, the pre-fix shipped "
                    "engine), the deltas are the guard's cost — the pooled map was "
                    "validated on reference-BEARING held-out data and never on "
                    "reference-free traces, so refusing it is right, but not free.",
            "guard_live_admits_noref": bool(calibrator.global_admits_noref),
            "before_guard": mark_unavailable(unguarded),
            "after_guard": mark_unavailable(transfer),
            "delta_after_minus_before": _delta(unguarded, transfer, _COMPARABLE),
            "before_guard_ece": unguarded_ece,
            "before_guard_confusion": unguarded_cm,
            "before_guard_per_mode": unguarded["per_mode"],
            "before_guard_calibration_path": unguarded_path,
            # How many of the metrics this module calls comparable actually moved.
            # Counted rather than asserted: "it moves at least eight metrics" was a
            # prose claim with nothing behind it.
            "n_comparable_metrics_moved": sum(
                1 for v in _delta(unguarded, transfer, _COMPARABLE).values() if v != 0),
            "comparable_metrics_moved": sorted(
                k for k, v in _delta(unguarded, transfer, _COMPARABLE).items() if v != 0),
        }

    ret, dil = FailureMode.RETRIEVAL_FAILURE.value, FailureMode.CONTEXT_DILUTION.value
    retrieval_to_dilution = _flow_from(transfer_cm, ret).get(dil, 0)
    dilution_to_retrieval = _flow_from(transfer_cm, dil).get(ret, 0)

    configs = {"baseline": baseline, "transfer": transfer, "retrained": retrained}

    return {
        "sizes": {"train": len(train), "cal": len(cal), "test": len(test)},
        "tier": tier.label,
        # Reference-bearing control = the currently published numbers.
        "baseline": baseline,
        # (a) THE HEADLINE: shipped engine, production-shape input.
        "transfer": mark_unavailable(transfer),
        # (b) the ceiling: refit end-to-end on stripped data.
        "retrained": mark_unavailable(retrained),
        # (c) THE HONEST HEADLINE: (a) with the mock's decision pinned to the
        # reference-bearing run, so nothing but the available metadata changed.
        # ``None`` on a real backend, where the artifact cannot arise.
        "frozen_mock": {
            "note": "Both reference and gold flags stripped, but MockModel._decide is "
                    "replayed from the reference-bearing original. Quote THIS, not "
                    "`transfer`, as the production estimate: the gap between them is "
                    "the simulator losing an oracle no real model ever had.",
            "metrics": mark_unavailable(frozen) if frozen is not None else None,
            "confusion": frozen_cm,
            # Same provenance count the black-tier block carries: > 0 proves this
            # condition was scored through the pinned handle, not relabelled.
            "decisions_replayed": frozen_replays,
            "delta_vs_baseline": _delta(baseline, frozen, _COMPARABLE)
            if frozen is not None else None,
            "delta_vs_transfer": _delta(transfer, frozen, _COMPARABLE)
            if frozen is not None else None,
            "available": frozen is not None,
        },
        "delta_vs_baseline": {
            "transfer": _delta(baseline, transfer, _COMPARABLE),
            "retrained": _delta(baseline, retrained, _COMPARABLE),
        },
        "per_mode": _per_mode_table(configs),
        "confusion": {
            "baseline": baseline_cm,
            "transfer": transfer_cm,
            "retrained": retrained_cm,
        },
        # The specific prediction under test (docs/ARCHITECTURE.md: gold_recall_in_context
        # is the stated retrieval-failure vs context-dilution discriminator, and it is
        # exactly what goes missing). Reported whether or not it holds.
        "retrieval_vs_dilution": {
            "prediction": "retrieval_failure and context_dilution collapse into each other "
                          "once gold_recall_in_context is unavailable",
            "retrieval_predicted_as_dilution": retrieval_to_dilution,
            "dilution_predicted_as_retrieval": dilution_to_retrieval,
            "held": bool(retrieval_to_dilution or dilution_to_retrieval),
            "retrieval_actually_went_to": _flow_from(transfer_cm, ret),
            "dilution_actually_went_to": _flow_from(transfer_cm, dil),
        },
        "unavailable_metrics": {
            k: "no reference -> Recommender._maybe_validate never sets Recommendation."
               "validated, so compute_metrics scores zero interventions and would "
               "otherwise emit a misleading 0.0"
            for k in GROUND_TRUTH_DEPENDENT_METRICS
        },
        "calibration_path": {
            "baseline": _calibration_path(shipped, test, model, pipeline, tier),
            "transfer": _calibration_path(shipped, stripped_test, model, pipeline, tier),
            "retrained": _calibration_path(refit, stripped_test, model, pipeline, tier),
        },
        # ECE is reported with its interior-mass fraction for the same reason
        # ablations does: on a saturated corpus it is a rescaled error rate and
        # carries no calibration information on its own.
        "ece": {
            "baseline": {"uncalibrated": _ece_full(shipped, test, model, pipeline, tier,
                                                   calibrated=False),
                         "calibrated": _ece_full(shipped, test, model, pipeline, tier,
                                                 calibrated=True)},
            "transfer": {"uncalibrated": _ece_full(shipped, stripped_test, model, pipeline, tier,
                                                   calibrated=False),
                         "calibrated": _ece_full(shipped, stripped_test, model, pipeline, tier,
                                                 calibrated=True)},
            "retrained": {"uncalibrated": _ece_full(refit, stripped_test, model, pipeline, tier,
                                                    calibrated=False),
                          "calibrated": _ece_full(refit, stripped_test, model, pipeline, tier,
                                                  calibrated=True)},
            "frozen_mock": {"uncalibrated": _ece_full(shipped, stripped_test, frozen_model,
                                                      pipeline, tier, calibrated=False),
                            "calibrated": _ece_full(shipped, stripped_test, frozen_model,
                                                    pipeline, tier, calibrated=True)}
            if frozen_model is not None else None,
        },
        "confound_controls": {
            "note": "Not shippable configurations. They decompose the headline drop into "
                    "(i) losing the reference, (ii) losing the gold-chunk annotation, and "
                    "(iii) the mock re-deciding because MockModel._decide reads Chunk.gold "
                    "as its own oracle — an artifact with no production analogue. (iii) is "
                    "SCORED in the top-level `frozen_mock` block; these two half-strips only "
                    "bound it, and the 'annotations removed, reference kept' row cannot "
                    "isolate it at all (the reference keeps dil.present_wrong firing, which "
                    "masks the re-decided rows rather than exposing them).",
            "reference_only_removed": reference_only,
            "annotations_only_removed": annotations_only,
            "mock_decisions_changed_by_recipe": dict(mock_decisions_changed),
        },
        # (d) the second production axis, measured rather than assumed.
        "ingest_shape": ingest,
        # What refusing the pooled map costs, on every comparable metric.
        "guard_cost": guard_cost,
    }
