"""Signal pipeline, tier degradation, and cold-start engine diagnosis."""

from __future__ import annotations

import pytest

from tokentrace.core.types import DiagnosisRole, FailureMode as M, SignalFamily, Tier
from tokentrace.engine import DiagnosisEngine

from _helpers import (
    dilution,
    grounded,
    nonrag_hallucination,
    observe,
    parametric_fragile,
    reasoning_failure,
    retrieval_failure,
)

FM = M


def test_retrieval_vs_dilution_discriminator(model, pipeline):
    """gold_recall_in_context is the decisive separator: 0 for retrieval failure,
    high for dilution."""
    rf = pipeline.run(observe(model, retrieval_failure()), model)
    dl = pipeline.run(observe(model, dilution()), model)
    assert rf.get("gold_recall_in_context") < 0.3
    assert dl.get("gold_recall_in_context") >= 0.6


def test_tier_degradation_drops_mechanistic(model, pipeline):
    inf = observe(model, grounded())
    black = pipeline.run(inf, model.with_tier(Tier.BLACK))
    white = pipeline.run(inf, model.with_tier(Tier.WHITE))
    assert black.family_present.get(SignalFamily.MECHANISTIC) is None
    assert white.family_present.get(SignalFamily.MECHANISTIC) is True
    assert len(white.values) > len(black.values)


@pytest.mark.parametrize("builder,expected", [
    (retrieval_failure, FM.RETRIEVAL_FAILURE),
    (dilution, FM.CONTEXT_DILUTION),
    (reasoning_failure, FM.REASONING_FAILURE),
    (nonrag_hallucination, FM.HALLUCINATION),
])
def test_cold_start_primary_diagnosis(model, pipeline, builder, expected):
    engine = DiagnosisEngine()
    inf = observe(model, builder())
    fv = pipeline.run(inf, model)
    report = engine.diagnose(inf, fv, model.tier)
    assert report.primary.mode == expected


def test_grounded_abstains(model, pipeline):
    engine = DiagnosisEngine()
    inf = observe(model, grounded())
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    assert report.abstained


def test_nonrag_masks_retrieval_modes(model, pipeline):
    engine = DiagnosisEngine()
    inf = observe(model, nonrag_hallucination())
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    probs = {d.mode: d.probability for d in report.diagnoses}
    assert probs[FM.RETRIEVAL_FAILURE] == 0.0
    assert probs[FM.CONTEXT_DILUTION] == 0.0


def test_causal_chain_retrieval_to_hallucination(model, pipeline):
    engine = DiagnosisEngine()
    inf = observe(model, retrieval_failure())
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    roles = {d.mode: d.role for d in report.diagnoses}
    assert roles[FM.RETRIEVAL_FAILURE] == DiagnosisRole.PRIMARY_ROOT
    assert roles[FM.HALLUCINATION] == DiagnosisRole.SEQUELA
    hall = next(d for d in report.diagnoses if d.mode == FM.HALLUCINATION)
    assert FM.RETRIEVAL_FAILURE in hall.causal_parents


def test_reasoning_not_demoted_below_weak_dilution(model, pipeline):
    """Regression: a dominant downstream mode stays primary when its parent is a
    weak co-fire."""
    engine = DiagnosisEngine()
    inf = observe(model, reasoning_failure())
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    assert report.primary.mode == FM.REASONING_FAILURE


def test_parametric_fragile_note(model, pipeline):
    engine = DiagnosisEngine()
    inf = observe(model, parametric_fragile())
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    assert any("unsupported" in n.lower() or "fragile" in n.lower() for n in report.notes)


def test_evidence_is_additive_logodds(model, pipeline):
    engine = DiagnosisEngine()
    inf = observe(model, retrieval_failure())
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    primary = report.primary
    assert primary.evidence
    assert all(hasattr(e, "contribution_logodds") for e in primary.evidence)
    assert primary.evidence[0].rendered  # human-readable
