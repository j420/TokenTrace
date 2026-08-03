"""End-to-end benchmark: train + calibrate + evaluate should meet the proposal's
targets on the offline synthetic corpus, and degrade gracefully at black-box."""

from __future__ import annotations

import pytest

from tokentrace.data import build_dataset
from tokentrace.eval import train_and_evaluate


@pytest.fixture(scope="module")
def results(model):
    ds = build_dataset(model, seeds=(0, 1, 2))
    return train_and_evaluate(model, ds)


def test_white_box_meets_targets(results):
    m = results["tiers"]["white"]
    assert m["diagnosis_accuracy"] >= 0.80    # target >= 0.80
    assert m["top3_accuracy"] >= 0.90         # target >= 0.90
    assert m["recommendation_precision"] >= 0.75  # target >= 0.75


def test_grey_box_meets_targets(results):
    m = results["tiers"]["grey"]
    assert m["diagnosis_accuracy"] >= 0.80
    assert m["top3_accuracy"] >= 0.90


def test_black_box_degrades_gracefully(results):
    """Black-box loses mechanistic evidence, so accuracy may dip but must stay
    usable and not crash."""
    m = results["tiers"]["black"]
    assert m["diagnosis_accuracy"] >= 0.70
    # reasoning/dilution separation is the hardest without mechanistic evidence
    assert m["per_mode"]["retrieval_failure"]["recall"] >= 0.8


def test_per_mode_precision_reasonable(results):
    white = results["tiers"]["white"]["per_mode"]
    for mode, s in white.items():
        if s["support"] >= 3:
            assert s["f1"] >= 0.6, f"{mode} F1 too low: {s}"


def test_debugging_time_reduction_is_positive_and_below_ceiling(results):
    """Ranking proxy: must beat unaided search but cannot exceed the structural
    ceiling 1 - 1/((K+1)/2) = 0.667 at K=5. Abstained rows are charged the full
    unaided cost, so unlike the old version this CAN fall when the engine declines."""
    dt = results["tiers"]["white"]["debugging_time_reduction"]
    assert 0.0 < dt <= 2 / 3 + 1e-9


def test_declined_rate_is_reported_separately(results):
    """Combined abstention mixes correct silence on clean rows with refusal to
    diagnose a real failure; both must be visible."""
    w = results["tiers"]["white"]
    assert "healthy_abstention_rate" in w and "declined_rate" in w
    assert 0.0 <= w["declined_rate"] <= 1.0


def test_noise_desaturates_and_calibration_helps():
    from tokentrace.eval.ablations import run_robustness

    r = run_robustness(noise_levels=(0.0, 0.75), seeds=(0,))
    lo, hi = r["0.00"], r["0.75"]
    # heavy observation noise must not IMPROVE accuracy (de-saturation)
    assert hi["diagnosis_accuracy"] <= lo["diagnosis_accuracy"] + 1e-9
    # calibration must never be worse than raw scores by a meaningful margin
    assert hi["ece_calibrated"] <= hi["ece_uncalibrated"] + 0.01


def test_ablations_separate_retrained_from_robustness(model):
    """The family ablation must report BOTH experiments.

    Evaluating a fully-trained engine with one family NaN'd out measures
    inference-time robustness, NOT information contribution — reporting only that
    made the published "which family is load-bearing" claim an artifact.
    """
    from tokentrace.eval.ablations import run_ablations

    res = run_ablations(model, seeds=(0,))
    al = res["ablation_learned_head"]
    assert al["rules_plus_gbt"]["diagnosis_accuracy"] >= al["rules_only"]["diagnosis_accuracy"]
    assert "per_family_retrained" in res and "per_family_robustness" in res
    for block in (res["per_family_retrained"], res["per_family_robustness"]):
        assert set(block) == {"prompt", "retrieval", "mechanistic", "confidence"}
    # the ECE figure must carry its interior-mass caveat so it cannot be quoted bare
    assert "interior_mass_fraction" in res["calibration"]
