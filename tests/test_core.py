"""Core contracts, serialization, and mock-simulator coherence."""

from __future__ import annotations

import json

from tokentrace.core.serialize import inference_from_dict, inference_to_dict
from tokentrace.core.types import Chunk, FeatureVector, Inference, SignalFamily, Tier

from _helpers import (
    dilution,
    grounded,
    parametric_fragile,
    reasoning_failure,
    retrieval_failure,
)


def test_tier_ordering():
    assert Tier.WHITE >= Tier.GREY >= Tier.BLACK
    assert Tier.BLACK.label == "black"


def test_inference_json_roundtrip():
    inf = Inference("Q?", "A", [Chunk("ctx", 0.5, "s1", gold=True)], ["A"], question="Q?")
    inf2 = inference_from_dict(json.loads(json.dumps(inference_to_dict(inf))))
    assert inf2.generated_answer == "A"
    assert inf2.is_rag and inf2.has_ground_truth
    assert inf2.retrieved_context[0].gold is True


def test_feature_missingness_signature():
    fv = FeatureVector()
    fv.set("answer_supported_by_context", 0.1, SignalFamily.RETRIEVAL)
    fv.set("gold_attention_ratio", None, SignalFamily.MECHANISTIC)  # missing => absent
    assert fv.get("answer_supported_by_context") == 0.1
    assert fv.get("gold_attention_ratio") is None
    assert fv.family_present.get(SignalFamily.MECHANISTIC) is None
    assert fv.missingness_signature() == "retrieval"


def test_mock_answers_are_coherent(model):
    """Each scenario should produce the answer its failure mode implies."""
    assert model.bound_to(grounded()).generate(grounded().prompt).text == "1889"
    assert model.bound_to(retrieval_failure()).generate(retrieval_failure().prompt).text == "1920"
    assert model.bound_to(parametric_fragile()).generate(parametric_fragile().prompt).text == "1889"
    assert model.bound_to(dilution()).generate(dilution().prompt).text == "1850"
    assert model.bound_to(reasoning_failure()).generate(reasoning_failure().prompt).text == "Marie Curie"


def test_mock_mechanistic_signatures(model):
    """Grounded: high context attention; parametric/hallucination: low + high parametric."""
    cap_g = model.capture(grounded())
    cap_r = model.capture(retrieval_failure())
    assert cap_g.context_attention_ratio > cap_r.context_attention_ratio
    assert cap_r.parametric_knowledge_score > cap_g.parametric_knowledge_score
    # dilution: gold present but under-attended
    assert model.capture(dilution()).gold_attention_ratio < 0.2


def test_semantic_entropy_low_when_confident(model):
    from tokentrace.signals.scorers import HeuristicScorers

    s = HeuristicScorers()
    assert s.semantic_entropy(["1889", "1889", "1889"]) == 0.0
    assert s.semantic_entropy(["1889", "1887", "2000"]) > 0.5
