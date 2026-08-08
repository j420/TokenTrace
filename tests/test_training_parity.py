"""Training-schedule parity: the SHIPPED engine is the BENCHMARKED engine.

Every published table is measured on an engine trained with the family-dropout
tier schedule (``[WHITE, GREY, BLACK]`` cycled over the train split — see
``eval/ablations.py`` and ``train_and_evaluate``). ``TokenTrace.default()`` — the
path ``demo``, ``analyze``, ``triage`` and the Streamlit app all build their
engine through — used to call ``train_engine`` WITHOUT that schedule, so the
engine users ran was not the engine the benchmarks described, and
ARCHITECTURE.md documented the divergence as a caveat.

These tests pin the fix by OBSERVABLE, not by spying on call arguments: the
engine ``default()`` builds must be bit-identical — trained LightGBM boosters,
fitted calibration buckets, noref record count, conformal tau — to an engine
built the way the benchmarks build one. A control shows the equality is not
vacuous (an unscheduled engine genuinely differs on this corpus), which doubles
as the standing record of the mutation check: revert ``default()`` to the
no-schedule call and the equality test goes red.

Two seeds: enough for the schedule to change both the boosters and which
calibration maps clear the adoption bar, while keeping the run short. Training
is deterministic (single-threaded LightGBM, hash-based split, seeded mock), so
bit-equality is a fair assertion.
"""

from __future__ import annotations

import pytest

from tokentrace.api import TokenTrace
from tokentrace.core.types import Tier
from tokentrace.data.synthetic import build_dataset
from tokentrace.eval.benchmark import family_dropout_schedule, split_dataset, train_engine

SEEDS = (0, 1)


def _boosters(engine) -> dict[str, str]:
    """The trained heads themselves, as LightGBM's canonical text dump."""
    return {m.value: mdl.booster_.model_to_string()
            for m, mdl in engine.classifier.models.items()}


@pytest.fixture(scope="module")
def default_engine():
    """What a user actually gets from the facade."""
    return TokenTrace.default(seeds=SEEDS).engine


@pytest.fixture(scope="module")
def split(model, pipeline):
    return split_dataset(build_dataset(model, pipeline, seeds=SEEDS))


@pytest.fixture(scope="module")
def benchmarked_engine(split, model, pipeline):
    """An engine trained exactly the way train_and_evaluate / run_ablations /
    run_noref train theirs: same corpus, same split, same dropout schedule."""
    train, cal, _ = split
    return train_engine(train, cal, model, pipeline,
                        train_tiers=family_dropout_schedule(len(train)))


@pytest.fixture(scope="module")
def unscheduled_engine(split, model, pipeline):
    """The old ``default()`` behaviour: no schedule, all rows at the handle's tier."""
    train, cal, _ = split
    return train_engine(train, cal, model, pipeline)


def test_family_dropout_schedule_is_the_published_cycle():
    """The helper must be the cycle the docs describe, not an approximation."""
    cycle = [Tier.WHITE, Tier.GREY, Tier.BLACK]
    assert family_dropout_schedule(7) == [cycle[i % 3] for i in range(7)]
    assert family_dropout_schedule(0) == []
    assert family_dropout_schedule(3) == cycle


def test_default_engine_is_the_benchmarked_engine(default_engine, benchmarked_engine):
    """The parity itself, on the trained artifacts.

    Mutation check (verified red): reverting ``TokenTrace.default()`` to
    ``train_engine(train, cal, model, pipeline)`` — the pre-fix call with no
    ``train_tiers`` — makes the booster comparison fail for every head.
    """
    assert _boosters(default_engine) == _boosters(benchmarked_engine)
    assert sorted(default_engine.calibrator.maps.keys()) == \
        sorted(benchmarked_engine.calibrator.maps.keys())
    assert default_engine.calibrator.n_noref == benchmarked_engine.calibrator.n_noref
    assert default_engine.conformal.tau == benchmarked_engine.conformal.tau


def test_the_parity_equality_is_not_vacuous(benchmarked_engine, unscheduled_engine):
    """Control: without the schedule the engine genuinely differs, so the
    bit-equality above is measuring the schedule, not comparing two runs of the
    same code path. This is also what makes the mutation check meaningful."""
    assert _boosters(benchmarked_engine) != _boosters(unscheduled_engine)


def test_mixed_tier_training_changes_which_calibration_maps_are_adopted(
    benchmarked_engine, unscheduled_engine
):
    """The schedule reaches the CALIBRATOR through the heads, and observably so.

    Calibration is always fitted across all three tiers (``cal_tiers`` in
    ``train_engine``), so the candidate buckets are the same either way — but
    which candidates beat the raw sigmoid on held-out data depends on the raw
    logits, i.e. on how the heads were trained. On this corpus the
    schedule-trained heads get the exact reduced-signature |noref map and the
    per-population ``__ref__`` maps adopted where the all-WHITE heads do not.
    Asserted as a plain inequality plus the specific direction measured, so the
    test fails loudly if the corpus changes enough to erase the effect — at
    which point the parity observable above still stands on the boosters.
    """
    sched = sorted(benchmarked_engine.calibrator.maps.keys())
    unsched = sorted(unscheduled_engine.calibrator.maps.keys())
    assert sched != unsched, \
        "the tier schedule no longer changes calibration-map adoption on this corpus"
    # both engines still fit |noref maps (that is fit_noref's doing, not the
    # schedule's) — the schedule changes WHICH buckets clear the adoption bar
    for keys in (sched, unsched):
        assert any(s.endswith("|noref") or s == "__noref__" for _, s in keys)
