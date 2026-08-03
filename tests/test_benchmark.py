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


def test_ablations_structure_and_findings(model):
    from tokentrace.eval.ablations import run_ablations

    res = run_ablations(model, seeds=(0, 1))
    al = res["ablation_learned_head"]
    # the learned residual never hurts vs pure rules
    assert al["rules_plus_gbt"]["diagnosis_accuracy"] >= al["rules_only"]["diagnosis_accuracy"]
    # retrieval family is more load-bearing than confidence (it carries 2 modes)
    pf = res["per_family_dropped"]
    assert pf["retrieval"]["diagnosis_accuracy"] <= pf["confidence"]["diagnosis_accuracy"]
    # dropping the prompt family destroys ambiguity detection specifically
    assert pf["prompt"]["per_mode"]["prompt_ambiguity"]["f1"] < 0.5
