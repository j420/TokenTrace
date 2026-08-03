"""Build a labeled dataset from the injection harness + featurize it for training.

This is the offline, no-download path: it turns the bundled seed pool into a
balanced multi-label corpus (all gates enforced), and can featurize each example
at a chosen tier — including a per-row tier schedule that simulates black/grey/
white, which is the family-dropout regularizer that hardens the learned heads
against missing signal families.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from tokentrace.core.types import ALL_MODES, LabeledInference, Tier
from tokentrace.data.injection import InjectionHarness
from tokentrace.data.samples import FACTS, MULTIHOP
from tokentrace.engine.rules import evaluate_rules
from tokentrace.models.base import ModelHandle
from tokentrace.signals.features import ALL_FEATURES
from tokentrace.signals.registry import SignalPipeline


def build_dataset(
    model: ModelHandle,
    pipeline: Optional[SignalPipeline] = None,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[LabeledInference]:
    harness = InjectionHarness(model, pipeline)
    out: list[LabeledInference] = []
    for s in seeds:
        for fact in FACTS:
            for recipe in (harness.clean, harness.retrieval_failure, harness.context_dilution,
                           harness.prompt_ambiguity, harness.hallucination):
                li = recipe(fact, seed=s)
                if li is not None:
                    out.append(li)
            if fact.parametric:
                li = harness.retrieval_failure(fact, seed=s, parametric_recovery=True)
                if li is not None:
                    out.append(li)
        for mh in MULTIHOP:
            li = harness.reasoning_failure(mh, seed=s)
            if li is not None:
                out.append(li)

    # Drop exact-duplicate examples: some recipes (e.g. the non-RAG hallucination
    # prompt) are seed-invariant and would otherwise contribute identical rows for
    # every seed, tripling their support and shrinking eval diversity.
    seen: set = set()
    uniq: list[LabeledInference] = []
    for li in out:
        key = (li.inference.prompt, li.inference.generated_answer,
               tuple(sorted(m.value for m in li.labels)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(li)
    return uniq


def dataset_summary(dataset: list[LabeledInference]) -> dict:
    recipes = Counter(li.injection_recipe for li in dataset)
    modes: Counter = Counter()
    for li in dataset:
        if not li.labels:
            modes["(none)"] += 1
        for m in li.labels:
            modes[m.value] += 1
    return {"n": len(dataset), "by_recipe": dict(recipes), "by_mode": dict(modes)}


def featurize(
    dataset: list[LabeledInference],
    model: ModelHandle,
    pipeline: Optional[SignalPipeline] = None,
    tiers: Optional[list[Tier]] = None,
) -> dict:
    """Return arrays for training / calibration.

    ``tiers``: optional per-row tier (family dropout). Defaults to the model's tier
    for every row.
    """
    pipeline = pipeline or SignalPipeline()
    n = len(dataset)
    X = np.full((n, len(ALL_FEATURES)), np.nan)
    rule_logits = np.zeros((n, len(ALL_MODES)))
    Y = np.zeros((n, len(ALL_MODES)), dtype=int)
    signatures: list[str] = []
    weights = np.ones(n)

    for i, li in enumerate(dataset):
        tier = tiers[i] if tiers else model.tier
        fv = pipeline.run(li.inference, model.with_tier(tier))
        X[i] = fv.to_array(ALL_FEATURES)
        logits, _ = evaluate_rules(fv)
        rule_logits[i] = [logits[m] for m in ALL_MODES]
        Y[i] = li.label_vector()
        signatures.append(fv.missingness_signature())
        weights[i] = li.weight

    return {
        "X": X, "rule_logits": rule_logits, "Y": Y,
        "signatures": signatures, "weights": weights,
        "feature_names": ALL_FEATURES,
    }
