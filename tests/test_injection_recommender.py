"""Injection-harness gates and simulated-intervention recommendation validation."""

from __future__ import annotations

from tokentrace.core.types import FailureMode as M
from tokentrace.data.injection import InjectionHarness
from tokentrace.data.samples import FACTS, MULTIHOP
from tokentrace.data.synthetic import build_dataset, dataset_summary
from tokentrace.engine import DiagnosisEngine
from tokentrace.recommend import Recommender

from _helpers import dilution, observe, reasoning_failure, retrieval_failure


def test_every_recipe_passes_its_gate(model):
    h = InjectionHarness(model)
    f = FACTS[0]
    assert h.clean(f) is not None
    assert h.retrieval_failure(f) is not None
    assert h.context_dilution(f) is not None
    assert h.prompt_ambiguity(f) is not None
    assert h.hallucination(f) is not None
    assert h.reasoning_failure(MULTIHOP[0]) is not None


def test_injected_labels_match_recipe(model):
    h = InjectionHarness(model)
    li = h.retrieval_failure(FACTS[0])
    assert M.RETRIEVAL_FAILURE in li.labels
    assert (M.RETRIEVAL_FAILURE, M.HALLUCINATION) in li.causal_edges  # multi-label + causal edge
    assert li.verification["gate_passed"]


def test_dataset_is_balanced_multilabel(model):
    ds = build_dataset(model, seeds=(0, 1))
    summ = dataset_summary(ds)
    assert summ["n"] > 100
    for mode in ("retrieval_failure", "context_dilution", "prompt_ambiguity",
                 "hallucination", "reasoning_failure", "(none)"):
        assert summ["by_mode"].get(mode, 0) > 0


def test_correct_fix_corrects_the_answer(model, pipeline):
    """The primary diagnosis's recommendation, when simulated, corrects the answer."""
    engine = DiagnosisEngine(recommender=Recommender(model=model))
    for builder in (retrieval_failure, dilution, reasoning_failure):
        inf = observe(model, builder())
        report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
        recs = report.primary.recommendations
        assert recs, f"no recommendation for {builder.__name__}"
        assert any(r.validated for r in recs), f"no working fix for {builder.__name__}"
