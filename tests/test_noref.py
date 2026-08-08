"""No-reference (production-shape) evaluation.

Guards the claim that TokenTrace's ``|noref`` path is *measured*, not merely
supported: that stripping a trace really removes the reference, that the
GT-dependent features go MISSING rather than to 0.0, that metrics which cannot be
computed without a reference say so explicitly, and that the two experiments in
:func:`tokentrace.eval.noref.run_noref` are genuinely different runs.
"""

from __future__ import annotations

import pathlib

import pytest

from tokentrace.core.types import ALL_MODES, FailureMode, Inference, LabeledInference, Tier
from tokentrace.eval.metrics import compute_metrics
from tokentrace.eval.noref import (
    GROUND_TRUTH_DEPENDENT_METRICS,
    UNAVAILABLE,
    mark_unavailable,
    run_noref,
    strip_dataset,
    strip_reference,
)

from _helpers import dilution, observe, retrieval_failure


@pytest.fixture(scope="module")
def labeled(model) -> LabeledInference:
    """A reference-bearing multi-chunk RAG example with a flagged gold chunk."""
    inf = observe(model, dilution())
    return LabeledInference(
        inference=inf,
        labels=[FailureMode.CONTEXT_DILUTION, FailureMode.HALLUCINATION],
        causal_edges=[(FailureMode.CONTEXT_DILUTION, FailureMode.HALLUCINATION)],
    )


@pytest.fixture(scope="module")
def labeled_retrieval(model) -> LabeledInference:
    """A retrieval failure, labelled the way the injection harness labels it.

    Used for the recommendation tests specifically because its diagnosis survives
    stripping (the reference-free ``ret.irrelevant`` / ``hal.confab`` rules still
    fire), so recommendations are still *attached* and the question becomes purely
    whether they can be *validated* — which is the dependency under test.
    """
    inf = observe(model, retrieval_failure())
    return LabeledInference(
        inference=inf,
        labels=[FailureMode.RETRIEVAL_FAILURE, FailureMode.HALLUCINATION],
        causal_edges=[(FailureMode.RETRIEVAL_FAILURE, FailureMode.HALLUCINATION)],
    )


# --------------------------------------------------------------------------- #
# strip_reference
# --------------------------------------------------------------------------- #
def test_strip_reference_removes_reference_and_preserves_labels(labeled):
    """The reference must be gone and the supervision must survive.

    Labels come from the injection recipe, not from ``ground_truth``, so accuracy
    stays measurable after stripping. If that stopped being true the whole
    experiment would be unscoreable, so it is asserted rather than assumed.
    """
    # preconditions: the fixture really is reference-bearing (else the test is vacuous)
    assert labeled.inference.has_ground_truth is True
    assert any(c.gold for c in labeled.inference.retrieved_context)

    stripped = strip_reference(labeled)

    assert stripped.inference.ground_truth is None
    assert stripped.inference.has_ground_truth is False
    assert not any(c.gold for c in stripped.inference.retrieved_context)
    # labels, causal edges and provenance are supervision, not reference
    assert stripped.labels == labeled.labels
    assert stripped.causal_edges == labeled.causal_edges
    assert stripped.provenance == labeled.provenance
    # the chunks themselves must survive — only the annotation is removed
    assert [c.text for c in stripped.inference.retrieved_context] == \
           [c.text for c in labeled.inference.retrieved_context]


def test_strip_reference_does_not_mutate_the_original(labeled):
    """The reference-bearing originals ARE the baseline control group; stripping
    in place would silently delete it and make every delta read as zero."""
    strip_reference(labeled)

    assert labeled.inference.has_ground_truth is True
    assert any(c.gold for c in labeled.inference.retrieved_context)
    # a deep copy, so mutating the stripped copy cannot reach back
    stripped = strip_reference(labeled)
    stripped.labels.append(FailureMode.PROMPT_AMBIGUITY)
    assert FailureMode.PROMPT_AMBIGUITY not in labeled.labels


def test_strip_reference_accepts_a_bare_inference():
    inf = Inference(prompt="q", generated_answer="a", ground_truth=["a"])
    stripped = strip_reference(inf)
    assert isinstance(stripped, Inference)
    assert stripped.has_ground_truth is False
    assert inf.has_ground_truth is True


def test_the_two_confound_controls_remove_one_thing_each(labeled):
    """The isolates that make the headline interpretable.

    ``MockModel._decide`` reads ``Chunk.gold`` as its own oracle, so dropping the
    flag perturbs the simulated model — something that cannot happen in
    production. These two half-strips separate that artifact from the genuine
    loss of diagnostic information, so each must remove exactly one thing.
    """
    reference_only = strip_reference(labeled, drop_gold_flags=False)
    assert reference_only.inference.has_ground_truth is False
    assert any(c.gold for c in reference_only.inference.retrieved_context)

    annotations_only = strip_reference(labeled, drop_ground_truth=False)
    assert annotations_only.inference.has_ground_truth is True
    assert not any(c.gold for c in annotations_only.inference.retrieved_context)


def test_sim_hints_are_preserved_and_no_signal_extractor_reads_them(labeled):
    """``meta['_sim']`` stays because the mock needs it to generate at all.

    That is only defensible while nothing on the signal path can see it, so the
    claim is enforced here: if an extractor ever starts reading ``_sim``,
    stripping stops being a genuine removal of the reference and this fails.
    """
    stripped = strip_reference(labeled)
    assert stripped.inference.meta.get("_sim") == labeled.inference.meta.get("_sim")
    assert stripped.inference.meta["_sim"], "fixture must carry sim hints"

    signals_dir = pathlib.Path(__file__).resolve().parents[1] / "tokentrace" / "signals"
    offenders = [p.name for p in sorted(signals_dir.glob("*.py")) if "_sim" in p.read_text()]
    assert offenders == [], f"signal modules must not read _sim: {offenders}"


# --------------------------------------------------------------------------- #
# What the signal pipeline sees
# --------------------------------------------------------------------------- #
def test_stripped_trace_selects_a_noref_signature(labeled, model, pipeline):
    """The missingness signature must record the lost reference, because that is
    what selects the calibration map."""
    ref_sig = pipeline.run(labeled.inference, model).missingness_signature()
    noref_sig = pipeline.run(strip_reference(labeled).inference, model).missingness_signature()

    assert not ref_sig.endswith("|noref")
    assert noref_sig.endswith("|noref")
    assert noref_sig != ref_sig
    # the families themselves are unchanged — only reference-availability differs
    assert noref_sig == f"{ref_sig}|noref"


def test_gt_dependent_features_are_missing_not_zero(labeled, model, pipeline):
    """``gold_recall_in_context`` and ``is_correct`` must be MISSING.

    A real 0.0 would be read by the rules as "the answer is absent from context"
    and "the answer is wrong" — confident, wrong evidence. ``None`` makes the
    dependent rules abstain instead.
    """
    ref = pipeline.run(labeled.inference, model)
    noref = pipeline.run(strip_reference(labeled).inference, model)

    for name in ("gold_recall_in_context", "is_correct"):
        assert ref.has(name), f"{name} must be present when a reference exists"
        assert not noref.has(name)
        assert noref.get(name) is None
        assert name in noref.missing
        assert name not in noref.values          # not stored as 0.0 either
        assert noref.get(name) != 0.0


# --------------------------------------------------------------------------- #
# Metrics that cannot be computed without a reference
# --------------------------------------------------------------------------- #
def test_recommendation_metrics_are_unscoreable_and_marked_unavailable(
    labeled_retrieval, model, pipeline
):
    """The trap this marker exists for.

    Without a reference ``Recommender._maybe_validate`` never assigns
    ``validated``, so ``compute_metrics`` scores zero interventions and its
    ``if rec_total else 0.0`` guard emits ``recommendation_precision = 0.0`` — a
    catastrophic-looking number where the truth is "not measurable".
    """
    from tokentrace.engine.diagnosis import DiagnosisEngine
    from tokentrace.recommend.recommender import Recommender

    engine = DiagnosisEngine(recommender=Recommender(model=model))
    stripped = strip_reference(labeled_retrieval)
    fv = pipeline.run(stripped.inference, model)
    report = engine.diagnose(stripped.inference, fv, Tier.WHITE)

    recs = [r for d in report.diagnoses for r in d.recommendations]
    assert recs, "the engine must still emit recommendations without a reference"
    assert all(r.validated is None for r in recs), \
        "no recommendation can be validated without a reference to check against"

    raw = compute_metrics([report], [stripped]).as_dict()
    assert raw["recommendation_n"] == 0
    assert raw["recommendation_precision"] == 0.0        # the misleading value

    marked = mark_unavailable(raw)
    for key in GROUND_TRUTH_DEPENDENT_METRICS:
        assert marked[key] == UNAVAILABLE
        assert not isinstance(marked[key], (int, float))
    # everything else must still be a real number
    assert isinstance(marked["diagnosis_accuracy"], float)
    assert marked["diagnosis_accuracy"] == raw["diagnosis_accuracy"]


def test_the_same_trace_IS_scoreable_while_it_still_has_its_reference(
    labeled_retrieval, model, pipeline
):
    """Control for the test above: same scenario, same engine, reference intact.

    Without this the previous test would be consistent with "this trace simply
    never produces a scoreable recommendation", rather than with "the reference is
    what makes it scoreable".
    """
    from tokentrace.engine.diagnosis import DiagnosisEngine
    from tokentrace.recommend.recommender import Recommender

    engine = DiagnosisEngine(recommender=Recommender(model=model))
    fv = pipeline.run(labeled_retrieval.inference, model)
    report = engine.diagnose(labeled_retrieval.inference, fv, Tier.WHITE)
    recs = [r for d in report.diagnoses for r in d.recommendations]
    assert any(r.validated is not None for r in recs)

    raw = compute_metrics([report], [labeled_retrieval]).as_dict()
    assert raw["recommendation_n"] >= 1
    assert isinstance(raw["recommendation_precision"], float)


def test_strip_dataset_preserves_order_and_length(labeled):
    ds = [labeled, labeled]
    out = strip_dataset(ds)
    assert len(out) == 2
    assert all(li.inference.has_ground_truth is False for li in out)
    assert [li.labels for li in out] == [li.labels for li in ds]


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def noref_result(model):
    # Two seeds: enough calibration data for the retrained engine to actually fit
    # a no-reference map (a single seed leaves the cal split under min_samples),
    # which is what makes (a) and (b) structurally distinguishable.
    return run_noref(model, seeds=(0, 1))


def test_run_noref_reports_both_experiments_against_the_baseline(noref_result):
    r = noref_result
    for key in ("sizes", "baseline", "transfer", "retrained", "delta_vs_baseline",
                "per_mode", "confusion", "retrieval_vs_dilution", "unavailable_metrics",
                "calibration_path", "confound_controls"):
        assert key in r, f"missing section: {key}"

    assert r["sizes"]["test"] > 0
    # the baseline must reproduce the shipped behaviour, or the deltas mean nothing
    assert r["baseline"]["diagnosis_accuracy"] >= 0.80
    assert isinstance(r["baseline"]["recommendation_precision"], float)

    for cfg in ("transfer", "retrained"):
        for key in GROUND_TRUTH_DEPENDENT_METRICS:
            assert r[cfg][key] == UNAVAILABLE
        assert isinstance(r[cfg]["diagnosis_accuracy"], float)

    # per-mode P/R/F1 for every mode in every configuration, not just aggregates
    for mode in ALL_MODES:
        per_mode = r["per_mode"][mode.value]
        assert set(per_mode) == {"baseline", "transfer", "retrained"}
        for stats in per_mode.values():
            assert set(stats) == {"precision", "recall", "f1", "support"}


def test_transfer_and_retrained_are_genuinely_different_runs(noref_result):
    """(a) and (b) must not be the same experiment relabelled.

    They share the stripped test set, so the difference has to come from the
    engine. REWRITTEN with the |noref calibration fix: this test used to assert
    that the shipped engine has NO no-reference calibration map — which was the
    §4.5.1 bug being pinned as a property. `train_engine` now fits calibration
    from the cal rows AND their reference-stripped twins by default
    (`fit_noref=True`), so the shipped engine legitimately carries |noref maps.
    What still separates (a) from (b): the shipped engine is calibrated on BOTH
    populations (reference-bearing and stripped), while the refit engine's cal
    split is all-stripped, so it can have no reference-bearing map at all — and
    its heads were fitted on stripped features.
    """
    r = noref_result
    transfer_maps = r["calibration_path"]["transfer"]["signatures_with_a_fitted_map"]
    retrained_maps = r["calibration_path"]["retrained"]["signatures_with_a_fitted_map"]

    def noref_sigs(maps):
        return [s for s in maps if s.endswith("|noref") or s == "__noref__"]

    def ref_sigs(maps):
        # __global__ is pooled over both populations, so it names neither.
        return [s for s in maps
                if s != "__global__" and not (s.endswith("|noref") or s == "__noref__")]

    # Changed from `not has_noref(transfer_maps)`: the shipped engine now fits
    # real |noref maps (mutation check: drop the stripped pass in train_engine
    # and this goes red).
    assert noref_sigs(transfer_maps), \
        "the shipped engine must fit |noref calibration maps (fit_noref default)"
    assert ref_sigs(transfer_maps), \
        "the shipped engine is still calibrated on reference-bearing data too"
    assert noref_sigs(retrained_maps), \
        "the refit engine must fit a no-reference calibration map"
    assert ref_sigs(retrained_maps) == [], \
        "the refit engine never saw a reference-bearing record, so it cannot " \
        "have a reference-bearing map"
    assert transfer_maps != retrained_maps
    assert r["confusion"]["transfer"] != r["confusion"]["retrained"]
    assert r["delta_vs_baseline"]["transfer"] != r["delta_vs_baseline"]["retrained"]


def test_stripped_traces_are_served_real_noref_calibration_maps(noref_result):
    """REWRITTEN: this test used to pin the *finding* ("no |noref map exists, so
    no stripped trace can be served an exact-signature map"). The fix it
    anticipated has landed — ``train_engine`` fits calibration from
    reference-stripped passes by default — so the honest property inverts:

    a stripped trace is served an exact or coarse |noref map (fallback level 0
    or 1, both fitted from reference-free records only), and is NEVER served the
    reference-only pooled map without admission. A mode whose |noref buckets
    failed the beats-the-sigmoid adoption bar may still legitimately fall to the
    (now admitted, genuinely mixed) __global__ pool or the raw sigmoid — that is
    the validation gate working, not the old bug — so what is pinned is that the
    noref-fitted maps exist, clear the data-sufficiency bar, and actually serve.

    Mutation check: reverting `fit_noref` (dropping the stripped calibration
    pass in train_engine) turns every assertion about noref maps below red.
    """
    r = noref_result
    transfer = r["calibration_path"]["transfer"]
    baseline = r["calibration_path"]["baseline"]

    assert transfer["observed_signatures"], "no traces observed"
    assert all(s.endswith("|noref") for s in transfer["observed_signatures"])
    assert not any(s.endswith("|noref") for s in baseline["observed_signatures"])

    # The coarse __noref__ bucket clears Calibrator.min_samples (40): one stripped
    # pass per cal row per tier is hundreds of records on two seeds.
    assert transfer["n_noref_calibration_records"] >= 40
    assert transfer["global_pool_admits_noref"] is True

    # The served count must BE the sum of the level-0/1 noref tallies, not a
    # separately-maintained number: adversarial verification inflated it with
    # level-2 (pooled) fallbacks — 35 became 160 — and nothing went red. The
    # tallies and the count are reported side by side precisely so they can be
    # cross-checked; do it here rather than leaving it to the reader.
    tallies = transfer["fallback_levels_noref_rows_only"]
    assert transfer["noref_rows_served_by_noref_fitted_maps"] == (
        tallies.get("exact_signature", 0) + tallies.get("coarse_ref_or_noref", 0))

    # Real |noref maps were fitted AND serve stripped traces (levels 0/1).
    assert any(s.endswith("|noref") or s == "__noref__"
               for s in transfer["signatures_with_a_fitted_map"])
    assert transfer["noref_rows_served_by_noref_fitted_maps"] > 0

    # Never the reference-only pool without admission — the §4.5.1 bug.
    assert transfer["noref_served_by_pooled_map"] is False

    # the reference-bearing baseline is still served reference-fitted maps — so
    # none of the above is a symptom of the calibrator having collapsed the two
    # populations into one
    assert baseline["fallback_levels"].get("exact_signature", 0) > 0 or \
        baseline["fallback_levels"].get("coarse_ref_or_noref", 0) > 0
    assert baseline["fallback_levels_noref_rows_only"] == {}


def test_retrieval_vs_dilution_prediction_is_reported_either_way(noref_result):
    """The stated prediction is that these two modes collapse into each other.

    The result is recorded whether or not it holds; what must not happen is the
    question going unanswered.
    """
    rvd = noref_result["retrieval_vs_dilution"]
    ret, dil = FailureMode.RETRIEVAL_FAILURE.value, FailureMode.CONTEXT_DILUTION.value
    confusion = noref_result["confusion"]["transfer"]

    # Re-derive the cross-confusions straight from the confusion matrix so the
    # verdict is checked against independent evidence rather than restated.
    r_as_d = confusion.get(f"{ret} -> {dil}", 0)
    d_as_r = confusion.get(f"{dil} -> {ret}", 0)
    assert rvd["retrieval_predicted_as_dilution"] == r_as_d
    assert rvd["dilution_predicted_as_retrieval"] == d_as_r
    # `held` must be a function of those counts, not a hard-coded verdict
    assert isinstance(rvd["held"], bool)
    assert rvd["held"] == bool(r_as_d + d_as_r)
    # where the rows actually landed is the finding, so it must be reported
    assert rvd["retrieval_actually_went_to"] == \
        {k.split(" -> ")[1]: v for k, v in confusion.items() if k.startswith(f"{ret} -> ")}
    assert sum(rvd["retrieval_actually_went_to"].values()) > 0


def test_losing_the_reference_costs_accuracy(noref_result):
    """The headline. If this ever stops being true the corpus has changed enough
    that the published numbers need re-deriving, not that the problem went away."""
    r = noref_result
    assert r["transfer"]["diagnosis_accuracy"] < r["baseline"]["diagnosis_accuracy"]
    assert r["delta_vs_baseline"]["transfer"]["diagnosis_accuracy"] < 0
