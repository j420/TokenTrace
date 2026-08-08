"""Tests for the mechanistic-value experiment: the white-exclusive rule, the
override-prevalence knob (byte-parity sacred), and the sweep instrument.

Every assertion here was mutation-verified: the specific break it guards against
was applied, the test observed red, the break was reverted, green confirmed.
"""

from __future__ import annotations

import hashlib

import pytest

from tokentrace.core.types import FailureMode as M, Tier
from tokentrace.data.injection import InjectionHarness
from tokentrace.data.samples import FACTS
from tokentrace.data.synthetic import build_dataset, dataset_summary
from tokentrace.engine.rules import evaluate_rules
from tokentrace.eval.mechanistic_value import run_mechanistic_value

RULE_ID = "hal.override_causal"


def _corpus_hash(ds) -> str:
    """sha256 over (prompt, sorted label values) per row, in corpus order.

    Text and enum values only — no floats, no repr() — so the digest is stable
    across Python versions and platforms.
    """
    h = hashlib.sha256()
    for li in ds:
        h.update(li.inference.prompt.encode())
        h.update(b"\x00")
        h.update("|".join(sorted(m.value for m in li.labels)).encode())
        h.update(b"\x01")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Task 2 — the default path is byte-identical to the pre-knob corpus
# --------------------------------------------------------------------------- #
def test_default_corpus_byte_parity(model, pipeline):
    """The override_weight knob must not perturb the default corpus AT ALL.

    The pinned digests were computed by importing the PRE-CHANGE module (the
    tree as of commit 0feb099, before ``override_weight`` existed) and hashing
    prompts + labels with ``_corpus_hash`` above — not by re-running the
    post-change code. Row order, row content, and labels are all covered.
    """
    ds = build_dataset(model, pipeline, seeds=(0, 1))
    assert len(ds) == 221
    assert _corpus_hash(ds) == (
        "55f779c3e858c584cc4da4f6a17e967177c0bcfef1578a004971af05a2a53d07")

    ds4 = build_dataset(model, pipeline, seeds=(0, 1, 2, 3))
    assert len(ds4) == 426
    assert _corpus_hash(ds4) == (
        "3ff7bbb1d15f9150c22b83396dd0ed1a0f80b14f1c10d3394f52ddb6d571e6e5")
    # Readable failure alongside the opaque digest: the composition pin.
    assert dataset_summary(ds4)["by_recipe"] == {
        "clean": 60, "retrieval_failure": 64, "context_dilution": 64,
        "prompt_ambiguity": 56, "hallucination": 46, "hallucination_override": 64,
        "retrieval_failure+parametric": 48, "reasoning_failure": 24,
    }


def test_override_weight_scales_only_the_override_class(model, pipeline):
    """override_weight=k must contribute k DISTINCT override rows per (fact,
    seed) — genuinely new rows, not duplicates for the dedup pass to delete —
    and must leave every other recipe untouched."""
    base = dataset_summary(build_dataset(model, pipeline, seeds=(0, 1)))["by_recipe"]
    doubled = dataset_summary(
        build_dataset(model, pipeline, seeds=(0, 1), override_weight=2))["by_recipe"]
    removed = dataset_summary(
        build_dataset(model, pipeline, seeds=(0, 1), override_weight=0))["by_recipe"]

    assert doubled["hallucination_override"] == 2 * base["hallucination_override"]
    assert "hallucination_override" not in removed
    for recipe, count in base.items():
        if recipe == "hallucination_override":
            continue
        assert doubled[recipe] == count, f"{recipe} changed under override_weight=2"
        assert removed[recipe] == count, f"{recipe} changed under override_weight=0"

    with pytest.raises(ValueError):
        build_dataset(model, pipeline, seeds=(0,), override_weight=-1)


# --------------------------------------------------------------------------- #
# Task 1 — the white-exclusive rule: fires at WHITE, abstains below
# --------------------------------------------------------------------------- #
def test_rule_fires_at_white_and_abstains_at_grey(model, pipeline):
    h = InjectionHarness(model, pipeline)
    li = h.hallucination_override(FACTS[0], seed=0)
    assert li is not None, "override recipe gate rejected its own row"

    fv_white = pipeline.run(li.inference, model.with_tier(Tier.WHITE))
    logits_w, fired_w = evaluate_rules(fv_white)
    contribs = {r.id: c for r, c, _ in fired_w}
    assert RULE_ID in contribs, "white-exclusive rule did not fire at WHITE"
    assert contribs[RULE_ID] > 0.5, (
        f"contribution {contribs[RULE_ID]:.3f} too weak to be tier-load-bearing")

    for tier in (Tier.GREY, Tier.BLACK):
        fv = pipeline.run(li.inference, model.with_tier(tier))
        logits, fired = evaluate_rules(fv)
        assert RULE_ID not in {r.id for r, _, _ in fired}, (
            f"rule must abstain at {tier.label}: gold_patch_effect is WHITE-only")
        # the abstention IS the tier degradation: less hallucination evidence below white
        assert logits_w[M.HALLUCINATION] > logits[M.HALLUCINATION]


def test_rule_stays_silent_on_dilution_and_clean(model, pipeline):
    """The rule's job is the override discrimination: on the behaviourally
    identical dilution row (gold present, wrong, but UNattended) and on a clean
    grounded row (HIGH patch effect) it must contribute nothing, even at WHITE."""
    h = InjectionHarness(model, pipeline)
    for maker in (h.context_dilution, h.clean):
        li = maker(FACTS[0], seed=0)
        assert li is not None
        fv = pipeline.run(li.inference, model.with_tier(Tier.WHITE))
        _, fired = evaluate_rules(fv)
        assert RULE_ID not in {r.id for r, _, _ in fired}, (
            f"rule misfires on {li.injection_recipe}")


def test_white_and_grey_are_distinct_experiments(model, pipeline):
    """Corpus-level distinctness observable: the WHITE and GREY rule priors must
    differ on every override row (previously they were bit-identical on the
    whole corpus, so the tier comparison compared an experiment to itself)."""
    ds = build_dataset(model, pipeline, seeds=(0,))
    differing, override_rows = set(), set()
    for i, li in enumerate(ds):
        lw, _ = evaluate_rules(pipeline.run(li.inference, model.with_tier(Tier.WHITE)))
        lg, _ = evaluate_rules(pipeline.run(li.inference, model.with_tier(Tier.GREY)))
        if abs(lw[M.HALLUCINATION] - lg[M.HALLUCINATION]) > 1e-9:
            differing.add(i)
        if li.injection_recipe == "hallucination_override":
            override_rows.add(i)
    assert override_rows, "corpus lost its override class"
    assert differing == override_rows, (
        f"white/grey hallucination priors should differ exactly on the override "
        f"rows: differing={sorted(differing)} override={sorted(override_rows)}")


# --------------------------------------------------------------------------- #
# Task 3 — sweep structure sanity (tiny configuration)
# --------------------------------------------------------------------------- #
def test_sweep_structure_and_internal_consistency(model, pipeline):
    res = run_mechanistic_value(levels=(1, 2), seeds=(0,), model=model,
                                pipeline=pipeline)
    assert res["config"]["levels"] == [1, 2]
    assert res["config"]["seeds"] == [0]
    assert set(res["levels"]) == {"1", "2"}

    for lvl, d in res["levels"].items():
        assert d["override_weight"] == int(lvl)
        pt = d["per_tier_accuracy"]
        assert set(pt) == {"white", "grey", "black"}
        assert all(0.0 <= v <= 1.0 for v in pt.values())
        assert d["white_minus_grey"] == pytest.approx(pt["white"] - pt["grey"], abs=1e-4)
        loo = d["leave_mechanistic_out"]
        assert loo["delta"] == pytest.approx(
            loo["white_full"] - loo["white_without_mechanistic"], abs=1e-4)
        for tier in ("white", "grey", "black"):
            sl = d["override_slice"][tier]
            # the slice must be non-vacuous on BOTH sides, or its precision/F1
            # degenerates into a can't-fail metric
            assert sl["n_override"] > 0 and sl["n_dilution"] > 0
            assert sl["tp"] + sl["fn"] == sl["n_override"]
            assert sl["fp"] + sl["tn"] == sl["n_dilution"]
            assert 0.0 <= sl["f1"] <= 1.0

    # the knob must actually move prevalence inside the experiment
    assert (res["levels"]["2"]["by_recipe"]["hallucination_override"]
            == 2 * res["levels"]["1"]["by_recipe"]["hallucination_override"])

    # On the NOISE-FREE deterministic mock the trained engine separates override
    # from dilution perfectly (measured, and consistent with the corpus's known
    # saturation) — so tp == n_override and fp == 0 at WHITE. This is not a
    # can't-fail metric: it goes red if the engine stops calling override rows
    # hallucination, or if the slice accounting miscounts either class.
    top = res["levels"]["2"]["override_slice"]["white"]
    assert top["tp"] == top["n_override"] and top["fp"] == 0
