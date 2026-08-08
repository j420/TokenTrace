"""The two controls that bound §4.5's production claim.

``run_noref``'s headline (a) "shipped engine on stripped traces" was for a long
time reported as *the* production number while two things stood between it and
that claim, neither of them measured:

1. **A simulator artifact.** ``MockModel._decide`` reads ``Chunk.gold`` as its own
   oracle, so stripping the annotation changes what the *simulated model does* —
   impossible for a real model, which never sees eval metadata. The drop was
   therefore partly the diagnosis losing evidence and partly the corpus changing
   under it. :func:`~tokentrace.eval.noref.freeze_mock_decisions` removes the
   second half by pinning the decision to the reference-bearing run, and these
   tests pin *that*: the pin must hold on the traces under test, and must NOT hold
   on counterfactuals (a freeze that also froze the simulated interventions would
   silently make every recommended fix look inert).

2. **A second axis nobody varied.** (a) is measured at ``Tier.WHITE`` with full
   mechanistic capture, while a trace from :mod:`tokentrace.ingest` has neither a
   reference nor activations. ``run_noref``'s ``ingest_shape`` block scores that
   condition; these tests check it is genuinely scored at the black-box tier and
   not the white-tier numbers relabelled — which is not paranoia, it is the bug
   this file caught: ``Tier.BLACK == 0`` is falsy, so an ``at or tier`` default
   ran the whole "black-box" condition at white.

Also covered: the two internal-consistency fixes in the same module —
``strip_reference`` clearing the GT-derived verification residue, and
``_calibration_path`` consulting the calibrator's own admission predicate instead
of guessing from an aggregate fallback count.
"""

from __future__ import annotations

import pytest

from tokentrace.core.types import FailureMode, Inference, LabeledInference, Tier
from tokentrace.eval.noref import (
    UNAVAILABLE,
    _calibration_path,
    _decision_key,
    freeze_mock_decisions,
    run_noref,
    strip_reference,
)
from tokentrace.recommend.recommender import clarify_prompt

from _helpers import dilution, observe

_DECISION_FIELDS = ("answer", "correct", "grounded", "parametric", "ambiguous",
                    "reasoning_failed", "gold_present", "gold_pos_frac", "overrode")


def _same_decision(a, b) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in _DECISION_FIELDS)


@pytest.fixture(scope="module")
def diluted(model) -> LabeledInference:
    """A dilution row: gold flagged, buried mid-list, long context.

    Chosen because it is exactly the row the artifact bites on — the mock's
    dilution trigger needs ``gold_pos_frac``, which is derived from the flag.
    """
    return LabeledInference(inference=observe(model, dilution()),
                            labels=[FailureMode.CONTEXT_DILUTION],
                            injection_recipe="context_dilution",
                            verification={"gate_passed": True, "is_correct": 0.0})


@pytest.fixture(scope="module")
def ambiguous(model) -> Inference:
    """A non-RAG ambiguous row, so ``clarify_prompt`` changes ONLY ``_sim``.

    That makes it the sharpest possible probe of the counterfactual requirement:
    prompt and context are byte-identical before and after the fix, so if
    ``_decision_key`` ignored ``_sim`` the frozen model would replay the unfixed
    decision and the intervention would appear to do nothing.
    """
    inf = Inference("Where is the bank?", "", None, ["the river bank"],
                    question="Where is the bank?",
                    meta={"_sim": {"answer": "the river bank", "ambiguous": True,
                                   "readings": ["the river bank", "the savings bank"]}})
    return observe(model, inf)


# --------------------------------------------------------------------------- #
# 1. The frozen-mock control
# --------------------------------------------------------------------------- #
def test_the_artifact_this_control_exists_for_is_real(model, diluted):
    """Precondition. Without this the freeze test could pass vacuously.

    Stripping the gold annotation must actually change what the simulated model
    decides — from "diluted, answered wrong" to "grounded, answered right" — since
    that is the whole reason the raw transfer number cannot be read as production
    loss.
    """
    before = model._decide(diluted.inference)
    after = model._decide(strip_reference(diluted).inference)

    assert not _same_decision(before, after), \
        "MockModel._decide no longer reads Chunk.gold; the control may be obsolete"
    assert before.correct is False and after.correct is True


def test_freezing_pins_the_stripped_decision_to_the_reference_bearing_one(model, diluted):
    """The control itself: same simulated model, strictly less metadata."""
    frozen = freeze_mock_decisions(model, [diluted])
    stripped = strip_reference(diluted)

    original = model._decide(diluted.inference)
    replayed = frozen._decide(stripped.inference)

    assert _same_decision(replayed, original)
    # and it is genuinely the pin doing the work, not a coincidence
    assert not _same_decision(replayed, model._decide(stripped.inference))


def test_freezing_does_not_pin_counterfactuals(model, ambiguous):
    """A freeze that caught simulated interventions would neuter every fix.

    ``clarify_prompt`` on a non-RAG row leaves the prompt and the (absent) context
    byte-identical and changes only the ``_sim`` hints, so this fails the moment
    ``_decision_key`` stops distinguishing them.
    """
    frozen = freeze_mock_decisions(model, [ambiguous])
    fixed = clarify_prompt(ambiguous)
    assert fixed is not None

    pinned = frozen._decide(ambiguous)
    after_fix = frozen._decide(fixed)

    assert pinned.ambiguous is True
    assert after_fix.ambiguous is False, "the simulated fix was swallowed by the freeze"
    assert after_fix.answer == model._decide(fixed).answer


def test_decision_key_ignores_exactly_the_reference_metadata(diluted):
    """The key must equate a trace with its stripped copy and nothing else."""
    stripped = strip_reference(diluted)
    assert _decision_key(diluted.inference) == _decision_key(stripped.inference)

    other = strip_reference(diluted)
    other.inference.retrieved_context[0].text += " extra"
    assert _decision_key(other.inference) != _decision_key(diluted.inference)


def test_the_pin_survives_bound_to_and_with_tier(model, diluted):
    """The pipeline puts two ``copy.copy`` wrappers between us and the model.

    ``SignalPipeline.run`` calls ``bound_to`` and ``run_reports`` calls
    ``with_tier``; if either dropped the instance attribute the freeze would be a
    no-op in exactly the code path it is used from.
    """
    frozen = freeze_mock_decisions(model, [diluted])
    stripped = strip_reference(diluted).inference
    expected = model._decide(diluted.inference)

    assert _same_decision(frozen.bound_to(stripped)._decide(stripped), expected)
    assert _same_decision(frozen.with_tier(Tier.BLACK)._decide(stripped), expected)
    assert _same_decision(
        frozen.with_tier(Tier.GREY).bound_to(stripped)._decide(stripped), expected)


def test_freezing_is_unavailable_for_a_backend_with_no_world_model():
    """Real backends have no ``_decide``, and no artifact to control for either."""

    class _Real:
        profile = None
        tier = Tier.WHITE

    assert freeze_mock_decisions(_Real(), []) is None


# --------------------------------------------------------------------------- #
# 2. strip_reference leaves no ground-truth residue
# --------------------------------------------------------------------------- #
def test_stripping_clears_the_gate_verification_residue(diluted):
    """``verification`` holds the injection gate's GT-derived readings.

    Inert on today's signal path, but the object is documented as being in
    production shape and those readings are ground truth by construction.
    """
    assert diluted.verification, "fixture must carry gate readings"

    assert strip_reference(diluted).verification == {}
    # the half-strip that KEEPS the reference must keep them: it is a control, and
    # its whole point is to remove one thing only
    assert strip_reference(diluted, drop_ground_truth=False).verification == diluted.verification
    # and the original is untouched, as for every other field
    assert diluted.verification["gate_passed"] is True


# --------------------------------------------------------------------------- #
# 3. End-to-end: both new conditions land in run_noref's output
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def result(model):
    # Two seeds, matching tests/test_noref.py: enough calibration data for the
    # retrained engine to fit a real no-reference map, while keeping the run short.
    return run_noref(model, seeds=(0, 1))


def test_frozen_mock_condition_is_reported_and_not_worse_than_transfer(result):
    """(c) must be present, scored, and never worse than (a) at white tier.

    REWRITTEN (was: strictly better). With real |noref calibration maps
    (`train_engine`'s ``fit_noref`` default) the transfer condition recovers the
    re-decided dilution rows on its own — calibrated probabilities keep them
    above the primary threshold whether or not the mock re-decided — so on this
    corpus (c) and (a) now coincide at the metric level and the strict `>` went
    red for the right reason. The pin itself is still proven to take effect at
    the DECISION level (`test_frozen_run_leaves_the_simulated_model_unchanged`);
    what this test now guards is that the control is scored and that removing
    the artifact never scores WORSE than leaving it in at white tier — if that
    direction ever flips, the §4.5 decomposition needs re-deriving.
    Mutation check: `fit_noref=False` on the shipped engine restores the strict
    gap (0.674 vs 0.593 shape), and `freeze_mock_decisions` returning None for a
    mock backend turns `available` red.
    """
    block = result["frozen_mock"]
    assert block["available"] is True
    metrics = block["metrics"]

    assert isinstance(metrics["diagnosis_accuracy"], float)
    assert metrics["diagnosis_accuracy"] >= result["transfer"]["diagnosis_accuracy"]
    assert metrics["declined_rate"] <= result["transfer"]["declined_rate"]
    assert block["delta_vs_transfer"]["diagnosis_accuracy"] >= 0
    assert block["delta_vs_baseline"]["diagnosis_accuracy"] < 0, \
        "the reference still has to cost something once the artifact is removed"

    # it is a reference-free run, so the same metrics are unmeasurable here as in (a)
    assert metrics["recommendation_precision"] == UNAVAILABLE
    assert block["confusion"], "the confusion matrix is the finding, not an extra"


def test_noref_calibration_absorbs_the_dilution_artifact(result):
    """REWRITTEN. The old assertion — freezing the mock must *recover* dilution
    F1 — pinned the §4.5 asymmetry as it stood when the shipped calibrator had
    no |noref maps: the artifact (the mock re-deciding stripped dilution rows)
    cost the transfer condition its dilution mode, and only the freeze got it
    back. With `fit_noref` calibration the transfer condition diagnoses those
    rows correctly on its own — the |noref-calibrated dilution probability stays
    above the primary threshold whether or not the simulated model re-decided —
    so the artifact's measurable cost is now zero and freezing changes neither
    mode's F1 on this corpus. That equality is asserted (it is the new §4.5
    result), not narrated.
    Mutation check: `fit_noref=False` on the shipped engine restores the old
    asymmetry (frozen dilution F1 1.0 vs transfer 0.667 on these seeds) and the
    dilution equality below goes red.
    """
    frozen = result["frozen_mock"]["metrics"]["per_mode"]

    assert frozen["context_dilution"]["f1"] == \
        result["per_mode"]["context_dilution"]["transfer"]["f1"], \
        "with |noref calibration the dilution artifact must cost nothing measurable"
    assert frozen["retrieval_failure"]["f1"] == \
        result["per_mode"]["retrieval_failure"]["transfer"]["f1"], \
        "retrieval must survive the control unchanged, as it always did"


def test_frozen_run_leaves_the_simulated_model_unchanged(model, result):
    """The control's defining property, checked against the run's own bookkeeping.

    ``mock_decisions_changed_by_recipe`` counts the rows the artifact moves in (a).
    Under the freeze that count must be zero for the very same rows — otherwise
    "unchanged simulated model" is a claim, not a fact.
    """
    from tokentrace.data.synthetic import build_dataset
    from tokentrace.eval.benchmark import split_dataset
    from tokentrace.eval.noref import strip_dataset
    from tokentrace.signals import SignalPipeline

    changed = result["confound_controls"]["mock_decisions_changed_by_recipe"]
    assert sum(changed.values()) > 0, "no artifact in this corpus; the control is vacuous"

    _tr, _cal, test = split_dataset(build_dataset(model, SignalPipeline(), seeds=(0, 1)))
    frozen = freeze_mock_decisions(model, test)
    still_changed = sum(
        1 for orig, s in zip(test, strip_dataset(test))
        if not _same_decision(model._decide(orig.inference), frozen._decide(s.inference))
    )
    assert still_changed == 0


def test_black_box_no_reference_condition_is_scored_at_the_black_box_tier(result):
    """(d) must be a different experiment, not (a) under a different label.

    ``Tier.BLACK == 0`` is falsy, so an ``or``-style default silently ran this at
    white and reported it as black. The check is structural: at black tier no
    mechanistic family can be present in any observed missingness signature.
    """
    ing = result["ingest_shape"]
    assert ing["tier"] == Tier.BLACK.label

    signatures = ing["calibration_path"]["observed_signatures"]
    assert signatures
    assert all("mechanistic" not in s for s in signatures)
    assert all(s.endswith("|noref") for s in signatures)

    # ...and behaviourally: black-box costs something, on some metric, in every
    # condition. Which metric moves is left unpinned so the finding can change.
    for name in ("baseline", "transfer", "frozen_mock"):
        deltas = ing["delta_vs_full_capture"][name]
        assert any(v != 0 for v in deltas.values()), \
            f"{name} at black tier is bit-identical to white — check the tier plumbing"


def test_black_box_condition_reports_both_losses_separately(result):
    """Reference loss and capture loss must be attributable, not summed.

    Note what is deliberately NOT asserted: that the frozen-mock control *improves*
    the black-box number. An earlier version of this test required it and went red —
    on the two-seed corpus, freezing scores 0.644 against the unfrozen 0.667. That is
    the correct behaviour, not a bug: the control removes an artifact, and an artifact
    can push a score either way. Requiring a direction would have made the control
    look like an optimization, which is exactly the framing §4.5 exists to avoid. What
    IS required is that it *does something* — a freeze that changed nothing would mean
    the pin never took.
    """
    ing = result["ingest_shape"]

    assert isinstance(ing["baseline"]["diagnosis_accuracy"], float)
    for cfg in ("transfer", "frozen_mock"):
        assert ing[cfg]["recommendation_precision"] == UNAVAILABLE
        assert ing[cfg]["diagnosis_accuracy"] < ing["baseline"]["diagnosis_accuracy"]
    # CHANGED: `frozen_mock != transfer` used to prove the pin took effect, but
    # that conflated two claims. With |noref calibration the two conditions
    # legitimately coincide at the metric level (the calibrated maps absorb the
    # re-decided rows), while the pin's effect on the SIMULATED MODEL is proven
    # directly by `test_frozen_run_leaves_the_simulated_model_unchanged`. What
    # must hold here is that the frozen condition was genuinely scored at this
    # tier, not relabelled from another one.
    assert isinstance(ing["frozen_mock"]["diagnosis_accuracy"], float)
    assert ing["delta_vs_same_tier_baseline"]["frozen_mock"] is not None
    assert set(ing["delta_vs_same_tier_baseline"]) == {"transfer", "frozen_mock"}
    # PROVENANCE, not inference. Adversarial verification relabelled this block
    # from the transfer metrics (bb_frozen = dict(bb_transfer)) and nothing went
    # red, because the two conditions coincide metric-for-metric here. The replay
    # counter on the pinned handle is incremented ONLY when a scoring pass routes
    # a decision through the pin, so a relabel leaves it at zero.
    # ...and the white-tier frozen block carries the same provenance count.
    assert isinstance(result["frozen_mock"]["decisions_replayed"], int)
    assert result["frozen_mock"]["decisions_replayed"] > 0
    assert isinstance(ing["frozen_decisions_replayed"], int)
    assert ing["frozen_decisions_replayed"] > 0, (
        "the black-tier frozen condition was never scored through the pinned "
        "handle — its metrics were relabelled from another condition")


# --------------------------------------------------------------------------- #
# 4. The __global__ admission guard is inert now that real |noref maps exist
# --------------------------------------------------------------------------- #
def test_the_global_admission_guard_is_now_inert(result):
    """The §4.5.1 guard-cost table, driven to zero rather than deleted.

    The guard withheld the pooled reference-bearing ``__global__`` map from
    reference-free traces, and REPORT.md 4.5.1 itemized seven metrics that
    refusal cost on the transfer condition. The better fix was always to fit
    real |noref maps so the guard has nothing left to withhold; ``fit_noref``
    does that, and this test asserts the consequence: forcing the admission
    predicate open changes NOTHING — every comparable delta is a measured zero,
    because a stripped trace is served a |noref-fitted map before the ladder
    ever reaches the pooled rung.

    These zeros CAN fail: the deltas are computed from two genuinely scored
    ladders, and reverting `fit_noref` (the mutation check) un-fits the |noref
    maps, flips `guard_live_admits_noref` to False, and reopens every gap the
    old guard-cost table recorded.
    """
    gc = result["guard_cost"]
    assert gc, "guard_cost must be measured on the shipped engine, not skipped"
    assert gc["guard_live_admits_noref"] is True

    deltas = gc["delta_after_minus_before"]
    assert deltas, "the comparable deltas must be reported"
    assert all(v == 0 for v in deltas.values()), \
        f"the admission guard still moves metrics: {deltas}"
    assert gc["n_comparable_metrics_moved"] == 0
    assert gc["comparable_metrics_moved"] == []
    # both ladders were really scored — and scored identically, ECE included
    assert gc["before_guard"] == gc["after_guard"]
    assert gc["before_guard_confusion"] == result["confusion"]["transfer"]
    assert gc["before_guard_ece"]["calibrated"]["ece"] == \
        result["ece"]["transfer"]["calibrated"]["ece"]


# --------------------------------------------------------------------------- #
# 5. The pooled-map flag consults the calibrator's admission predicate
# --------------------------------------------------------------------------- #
def test_pooled_map_flag_distinguishes_the_bug_from_the_legitimate_case(result):
    """``noref_served_by_pooled_map`` must not fire on a legitimately mixed pool.

    Both engines are now legitimately mixed: the (b) retrained engine is
    calibrated on reference-free data, and — CHANGED with the ``fit_noref``
    default — the (a) shipped engine's calibration also sees a reference-stripped
    pass over the cal split, so its pool really was shaped by reference-free
    records too. This test used to assert the shipped engine's pool was
    UNMIXED (``admits is False``, ``n_noref == 0``); that was the pre-fix state,
    and asserting it kept the §4.5.1 guard cost alive as a pinned property.
    Serving an admitted pool to a reference-free trace is the guard working, and
    a flag that only counted level-2 fallbacks would report it as the bug.
    Mutation check: dropping the stripped calibration pass in train_engine turns
    the two `transfer` assertions red.
    """
    transfer = result["calibration_path"]["transfer"]
    retrained = result["calibration_path"]["retrained"]

    assert transfer["global_pool_admits_noref"] is True
    assert transfer["n_noref_calibration_records"] >= 40      # Calibrator.min_samples
    assert retrained["global_pool_admits_noref"] is True
    assert retrained["n_noref_calibration_records"] > 0
    for cp in (transfer, retrained):
        assert cp["noref_served_by_pooled_map"] is False, \
            "an admitted pool is not the bug"
        # the noref-only tally must be a real subset view, not a copy of the aggregate
        for level, n in cp["fallback_levels_noref_rows_only"].items():
            assert n <= cp["fallback_levels"][level]


def test_a_mixed_pooled_bucket_is_validated_per_population_not_only_on_the_mixture():
    """The ``__global__`` pool now genuinely mixes reference-bearing and
    reference-free records (train_engine's ``fit_noref``), but it is served to
    one population at a time — so a candidate map that models the population
    CONTRAST can beat the raw sigmoid on the pooled holdout while making one
    population's probabilities strictly worse. Measured origin: under
    observation noise 0.75 (``run_robustness``) the pooled map was adopted on
    the mixture and pushed reference-bearing calibrated ECE to 0.0683 against a
    0.0077 raw sigmoid. ``Calibrator._fit_validated`` therefore also requires
    the candidate not to be worse than the sigmoid on each population that
    meaningfully shaped the bucket (>= min_samples records of it).

    The construction: the two populations have INVERTED label structure at the
    same z, so the pooled fit lands near 0.5 — a big Brier win over the sigmoid
    on the pool (the noref half is far off), and a strict loss on the
    reference-bearing half (whose sigmoid was nearly perfect).
    Mutation check: disabling the population gate in ``_fit_validated`` adopts
    the ``__global__`` bucket here and the first assertion goes red.
    """
    from tokentrace.engine.calibration import Calibrator

    recs = []
    for _ in range(50):
        recs += [
            {"mode": FailureMode.HALLUCINATION, "signature": "prompt+retrieval",
             "z": 2.0, "label": 1},
            {"mode": FailureMode.HALLUCINATION, "signature": "prompt+retrieval",
             "z": -2.0, "label": 0},
            {"mode": FailureMode.HALLUCINATION, "signature": "prompt+retrieval|noref",
             "z": 2.0, "label": 0},
            {"mode": FailureMode.HALLUCINATION, "signature": "prompt+retrieval|noref",
             "z": -2.0, "label": 1},
        ]
    cal = Calibrator().fit(recs)

    key = (FailureMode.HALLUCINATION.value, "__global__")
    assert key not in cal.maps, \
        "a pooled map that hurts the reference-bearing population was adopted"
    # the per-population rungs stay free to adopt on their own merits, so the
    # gate is a refusal to cross populations, not a refusal to calibrate: a
    # reference-free trace is still served its own coarse map...
    assert (FailureMode.HALLUCINATION.value, "__noref__") in cal.maps
    cal.transform(FailureMode.HALLUCINATION, "prompt+retrieval|noref", 2.0)
    assert cal.last_fallback <= 1
    # ...and the (already well-calibrated) reference-bearing trace falls through
    # to the raw sigmoid rather than being served the contrast-fitting pool.
    cal.maps.pop((FailureMode.HALLUCINATION.value, "prompt+retrieval"), None)
    cal.maps.pop((FailureMode.HALLUCINATION.value, "__ref__"), None)
    cal.transform(FailureMode.HALLUCINATION, "prompt+retrieval", 2.0)
    assert cal.last_fallback == 3


class _StubCalibrator:
    """A calibrator that serves the pooled map to whoever asks, guard or no guard.

    Needed because the real :class:`~tokentrace.engine.calibration.Calibrator` now
    refuses the ``__global__`` rung to reference-free traces inside ``transform``
    itself — which makes ``noref_served_by_pooled_map`` **structurally unreachable
    against a correct calibrator**. That is the right outcome for the flag (it is a
    regression detector for the guard, not an observation about the run), but it
    means the only way to test the flag's own logic is to hand it a calibrator that
    behaves the way the pre-guard one did.
    """

    maps: dict = {("hallucination", "__global__"): None}
    min_samples = 40

    def __init__(self, *, n_noref: int, pooled_for_noref: bool):
        self.n_noref = n_noref
        self._pooled_for_noref = pooled_for_noref
        self.last_fallback = 3

    @property
    def global_admits_noref(self) -> bool:
        return self.n_noref >= self.min_samples

    def transform(self, mode, signature, z):
        noref = signature.endswith("|noref")
        # level 2 = pooled __global__, level 3 = raw sigmoid
        self.last_fallback = 2 if (self._pooled_for_noref or not noref) else 3
        return 0.5


def test_pooled_map_flag_counts_noref_rows_only_and_defers_to_the_admission_predicate(
    model, pipeline, diluted
):
    """The two ways the old one-line flag misreported, exercised directly.

    (i) It ORed "some ``|noref`` signature was observed" with "some *row* fell to the
    pooled map", so on a mixed split the pooled hits could belong entirely to
    reference-BEARING rows and the flag would still cry bug. (ii) It never asked the
    calibrator whether the pool was entitled to serve reference-free traces, so a
    legitimately mixed calibrator was reported as buggy for working correctly.
    """
    mixed = [diluted, strip_reference(diluted)]

    class _Engine:
        def __init__(self, cal):
            self.calibrator = cal

        def raw_logits(self, fv, inference):
            from tokentrace.core.types import ALL_MODES
            return {m: 1.0 for m in ALL_MODES}

    # (i) only the reference-BEARING row is pooled -> not the bug
    ref_only_pooled = _calibration_path(
        _Engine(_StubCalibrator(n_noref=0, pooled_for_noref=False)),
        mixed, model, pipeline)
    assert ref_only_pooled["fallback_levels"]["global_pooled"] > 0
    assert "global_pooled" not in ref_only_pooled["fallback_levels_noref_rows_only"]
    assert ref_only_pooled["noref_served_by_pooled_map"] is False

    # the bug itself: a reference-free row served a pool that is not entitled to it
    bug = _calibration_path(
        _Engine(_StubCalibrator(n_noref=0, pooled_for_noref=True)),
        mixed, model, pipeline)
    assert bug["fallback_levels_noref_rows_only"]["global_pooled"] > 0
    assert bug["noref_served_by_pooled_map"] is True

    # (ii) same fallbacks, but the pool WAS shaped by reference-free records
    legit = _calibration_path(
        _Engine(_StubCalibrator(n_noref=40, pooled_for_noref=True)),
        mixed, model, pipeline)
    assert legit["fallback_levels_noref_rows_only"]["global_pooled"] > 0
    assert legit["noref_served_by_pooled_map"] is False
    assert legit["noref_served_by_admitted_pooled_map"] is True


# --------------------------------------------------------------------------- #
# 5. The guard-cost zeros must be FALSIFIABLE
# --------------------------------------------------------------------------- #
def test_guard_cost_machinery_reproduces_the_nonzero_table_without_noref_maps(model):
    """The inert-guard zeros are a measurement only if the same machinery can
    still produce the nonzero table.

    On the default path (fit_noref=True) the force-admission inside run_noref's
    guard-cost block is a no-op — the calibrator already admits — so every delta
    is a deterministic run compared with itself. Adversarial verification neutered
    the force-admission entirely and 31/31 tests stayed green: the zeros could
    not fail. This companion runs the SAME machinery on an engine trained without
    the stripped calibration pass, where the guard still has something to
    withhold; the deltas must reappear. Neutering the force-admission now zeroes
    THIS table too, and this test goes red.
    """
    res = run_noref(model, seeds=(0, 1), fit_noref=False)
    gc = res["guard_cost"]
    assert gc["guard_live_admits_noref"] is False
    moved = gc["n_comparable_metrics_moved"]
    assert isinstance(moved, int) and moved > 0, (
        "with no |noref maps the admission guard must have a measurable cost; "
        "zero moved metrics here means the before/after ladders were not "
        "genuinely scored under different admission states")
    deltas = gc["delta_after_minus_before"]
    assert any(abs(v) > 0 for v in deltas.values() if isinstance(v, (int, float)))
