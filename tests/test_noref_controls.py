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


def test_frozen_mock_condition_is_reported_and_recovers_the_artifact(result):
    """(c) must be present, scored, and better than (a).

    "Better" is the measurable content of the claim that part of the raw drop was
    the simulator changing rather than the diagnosis failing. If it ever stops
    holding, the report's decomposition is wrong and needs re-deriving.
    """
    block = result["frozen_mock"]
    assert block["available"] is True
    metrics = block["metrics"]

    assert isinstance(metrics["diagnosis_accuracy"], float)
    assert metrics["diagnosis_accuracy"] > result["transfer"]["diagnosis_accuracy"]
    assert metrics["declined_rate"] < result["transfer"]["declined_rate"]
    assert block["delta_vs_transfer"]["diagnosis_accuracy"] > 0
    assert block["delta_vs_baseline"]["diagnosis_accuracy"] < 0, \
        "the reference still has to cost something once the artifact is removed"

    # it is a reference-free run, so the same metrics are unmeasurable here as in (a)
    assert metrics["recommendation_precision"] == UNAVAILABLE
    assert block["confusion"], "the confusion matrix is the finding, not an extra"


def test_the_dilution_collapse_is_the_artifact_and_the_retrieval_one_is_not(result):
    """The asymmetry is the actual §4.5 result, so it is asserted, not narrated."""
    transfer = result["per_mode"]["context_dilution"]["transfer"]["f1"]
    frozen = result["frozen_mock"]["metrics"]["per_mode"]

    assert frozen["context_dilution"]["f1"] > transfer, \
        "freezing the mock must recover the dilution mode; that collapse is an artifact"
    assert frozen["retrieval_failure"]["f1"] == \
        result["per_mode"]["retrieval_failure"]["transfer"]["f1"], \
        "the retrieval collapse is a genuine loss and must survive the control unchanged"


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
    assert ing["frozen_mock"] != ing["transfer"], "the pin had no effect at this tier"
    assert set(ing["delta_vs_same_tier_baseline"]) == {"transfer", "frozen_mock"}


# --------------------------------------------------------------------------- #
# 4. The pooled-map flag consults the calibrator's admission predicate
# --------------------------------------------------------------------------- #
def test_pooled_map_flag_distinguishes_the_bug_from_the_legitimate_case(result):
    """``noref_served_by_pooled_map`` must not fire on a legitimately mixed pool.

    The (b) retrained engine is calibrated on reference-free data, so its
    ``__global__`` pool really was shaped by such records and serving it to a
    reference-free trace is the guard working. A flag that only counted level-2
    fallbacks would report that as the bug it was written to detect.
    """
    transfer = result["calibration_path"]["transfer"]
    retrained = result["calibration_path"]["retrained"]

    assert transfer["global_pool_admits_noref"] is False
    assert transfer["n_noref_calibration_records"] == 0
    assert retrained["global_pool_admits_noref"] is True
    assert retrained["n_noref_calibration_records"] > 0
    assert retrained["noref_served_by_pooled_map"] is False, \
        "an admitted pool is not the bug"
    # the noref-only tally must be a real subset view, not a copy of the aggregate
    for cp in (transfer, retrained):
        for level, n in cp["fallback_levels_noref_rows_only"].items():
            assert n <= cp["fallback_levels"][level]


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
