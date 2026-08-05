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

**(a) shipped-engine transfer** — train and calibrate on reference-bearing data
exactly as today, then evaluate on stripped traces. Nothing about the engine
changes; only the input loses its reference. *This is the real production
scenario* and is the headline number.

**(b) retrained-on-stripped** — train and calibrate on stripped data too. This is
the ceiling: how much of the diagnosis is recoverable if you accept up front that
you will never have references.

Both are reported against the reference-bearing baseline, so the delta is
explicit rather than inferred.

Metrics that cannot be computed without a reference are returned as the explicit
:data:`UNAVAILABLE` marker, never as 0.0 — see
:data:`GROUND_TRUTH_DEPENDENT_METRICS` for which ones and why.

One confound is measured rather than hoped away. ``MockModel._decide`` reads
``Chunk.gold`` as its OWN oracle (it derives ``gold_pos_frac`` from the flag, and
its dilution trigger depends on that), so removing the annotation changes the
*simulated model's* behaviour — which cannot happen in production, where a real
model never sees an annotation in the first place. ``confound_controls`` in
:func:`run_noref` therefore reports two half-strips that remove one thing each,
plus a count of how many rows the mock re-decides, so the headline drop can be
attributed rather than assumed.

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
from tokentrace.data.synthetic import build_dataset
from tokentrace.eval.ablations import _ece_full
from tokentrace.eval.benchmark import run_reports, split_dataset, train_engine
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
    """
    stripped = copy.deepcopy(obj)
    inference = stripped.inference if isinstance(stripped, LabeledInference) else stripped
    if drop_ground_truth:
        inference.ground_truth = None
    if drop_gold_flags:
        for chunk in inference.retrieved_context or ():
            chunk.gold = False
    return stripped


def strip_dataset(dataset: list[LabeledInference], **kwargs) -> list[LabeledInference]:
    """:func:`strip_reference` over a whole split, preserving order."""
    return [strip_reference(li, **kwargs) for li in dataset]


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
    (row, unmasked mode) and tally it. A stripped trace landing on level 2
    (``__global__``) is being served a map fitted on reference-bearing records —
    precisely the silent pooling the ``|noref`` signature was introduced to
    prevent.
    """
    levels: Counter = Counter()
    signatures: Counter = Counter()
    calibrator = getattr(engine, "calibrator", None)
    for li in data:
        fv = pipeline.run(li.inference, model.with_tier(tier))
        signature = fv.missingness_signature()
        signatures[signature] += 1
        if calibrator is None:
            continue
        z = engine.raw_logits(fv, li.inference)
        for mode in ALL_MODES:
            if z[mode] <= -49.0:          # hard-masked (non-RAG); nothing to calibrate
                continue
            calibrator.transform(mode, signature, z[mode])
            levels[calibrator.last_fallback] += 1

    fitted = sorted({sig for _, sig in calibrator.maps}) if calibrator is not None else []
    named = {0: "exact_signature", 1: "coarse_ref_or_noref", 2: "global_pooled", 3: "raw_sigmoid"}
    return {
        "observed_signatures": dict(signatures),
        "signatures_with_a_fitted_map": fitted,
        "fallback_levels": {named[k]: v for k, v in sorted(levels.items())},
        # The bug condition: a |noref trace served by the pooled global map, which
        # on a reference-bearing calibration split is 100% reference-bearing data.
        "noref_served_by_pooled_map": bool(
            any(s.endswith("|noref") for s in signatures) and levels.get(2, 0) > 0
        ),
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
) -> dict:
    """Measure the engine on production-shape (no-reference) traces.

    Same seeds and same family-dropout tier schedule as
    :func:`tokentrace.eval.ablations.run_ablations`, so the numbers sit alongside
    the published table rather than beside it.
    """
    pipeline = pipeline or SignalPipeline()
    dataset = dataset if dataset is not None else build_dataset(model, pipeline, seeds=seeds)
    train, cal, test = split_dataset(dataset)

    # Identical family-dropout schedule to run_ablations / train_and_evaluate. The
    # three entry points previously trained differently while the docs presented
    # their tables as interchangeable; this experiment must not reintroduce that.
    cycle = [Tier.WHITE, Tier.GREY, Tier.BLACK]
    train_tiers = [cycle[i % 3] for i in range(len(train))]

    # (a) the SHIPPED engine: trained and calibrated on reference-bearing data.
    shipped = train_engine(train, cal, model, pipeline, train_tiers=train_tiers)

    stripped_train = strip_dataset(train)
    stripped_cal = strip_dataset(cal)
    stripped_test = strip_dataset(test)

    # (b) the ceiling: everything refit on stripped data, including calibration.
    refit = train_engine(stripped_train, stripped_cal, model, pipeline, train_tiers=train_tiers)

    def score(engine, data) -> tuple[dict, dict]:
        reports = run_reports(engine, data, model, tier, pipeline)
        return compute_metrics(reports, data).as_dict(), _confusion(reports, data)

    baseline, baseline_cm = score(shipped, test)
    transfer, transfer_cm = score(shipped, stripped_test)
    retrained, retrained_cm = score(refit, stripped_test)

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
        },
        "confound_controls": {
            "note": "Not shippable configurations. They decompose the headline drop into "
                    "(i) losing the reference, (ii) losing the gold-chunk annotation, and "
                    "(iii) the mock re-deciding because MockModel._decide reads Chunk.gold "
                    "as its own oracle — an artifact with no production analogue.",
            "reference_only_removed": reference_only,
            "annotations_only_removed": annotations_only,
            "mock_decisions_changed_by_recipe": dict(mock_decisions_changed),
        },
    }
