"""Regression tests locking in the four-eyes review fixes."""

from __future__ import annotations

from tokentrace.core.types import Chunk, FailureMode as M, Inference, Tier
from tokentrace.data.injection import InjectionHarness
from tokentrace.data.samples import FACTS
from tokentrace.engine import DiagnosisEngine
from tokentrace.engine.causal import CausalResolver
from tokentrace.engine.conformal import ConformalPredictor
from tokentrace.signals.scorers import HeuristicScorers

from _helpers import observe


def test_causal_walks_to_strongest_parent():
    # HALL strongest; CD (0.70) and RET (0.65) both qualify as parents -> pick the
    # STRONGER (context_dilution), not the first in dict order (MED-2).
    probs = {M.PROMPT_AMBIGUITY: 0.1, M.RETRIEVAL_FAILURE: 0.65, M.CONTEXT_DILUTION: 0.70,
             M.HALLUCINATION: 0.90, M.REASONING_FAILURE: 0.1}
    roles = CausalResolver().resolve(probs)
    primary = next(m for m, (r, _) in roles.items() if r.value == "primary_root")
    assert primary == M.CONTEXT_DILUTION


def test_conformal_excludes_masked_modes():
    # Non-RAG masks retrieval modes to 0; they must never pad the conformal set (MED-3).
    probs = {M.PROMPT_AMBIGUITY: 0.20, M.RETRIEVAL_FAILURE: 0.0, M.CONTEXT_DILUTION: 0.0,
             M.HALLUCINATION: 0.30, M.REASONING_FAILURE: 0.15}
    cs = ConformalPredictor(tau=0.9).predict(probs)
    assert M.RETRIEVAL_FAILURE not in cs and M.CONTEXT_DILUTION not in cs


def test_ambiguity_with_trailing_capital_is_detected():
    # "...in Celsius?" previously zeroed the ambiguity score (any capital = antecedent);
    # now a trailing capitalized unit is not treated as the pronoun's antecedent (S1).
    s = HeuristicScorers()
    assert s.ambiguity("What is its boiling point at sea level in Celsius?") >= 0.5
    # a genuinely resolved pronoun stays low
    assert s.ambiguity("When was the Eiffel Tower built, and when was it opened?") < 0.4


def test_previously_blind_ambiguity_example_now_injectable_and_diagnosed(model, pipeline):
    # FACTS[4] = boiling point of water; its ambiguous variant ends in "Celsius" and
    # was silently discarded by the injection gate before the fix.
    fact = FACTS[4]
    li = InjectionHarness(model).prompt_ambiguity(fact)
    assert li is not None and M.PROMPT_AMBIGUITY in li.labels
    report = DiagnosisEngine().diagnose(li.inference, pipeline.run(li.inference, model), model.tier)
    assert report.primary.mode == M.PROMPT_AMBIGUITY


def test_ambiguity_and_dilution_have_distinct_mock_signatures(model):
    # S2: the mock's capture() must not give ambiguity a dilution-identical signature.
    amb = Inference("When was it completed?\nThe Eiffel Tower was completed in 1889.", "",
                    [Chunk("The Eiffel Tower was completed in 1889.", 0.9, "g", gold=True)], ["1889"],
                    question="When was it completed?",
                    meta={"_sim": {"answer": "1889", "ambiguous": True,
                                   "readings": ["1889", "1887", "2000"], "gold_fact": "completed in 1889"}})
    cap = model.capture(amb)
    # ambiguity: model DID attend to gold (not the buried/unattended dilution signature)
    assert cap.gold_attention_ratio >= 0.3
    assert cap.logit_lens_stability >= 0.7          # stable, unlike reasoning failure


def test_evidence_ledger_includes_base_rate(model, pipeline):
    engine = DiagnosisEngine()
    inf = observe(model, Inference(
        "When was the Eiffel Tower completed?\nParis is in France.", "",
        [Chunk("Paris is in France.", 0.5, "d2")], ["1889"],
        question="When was the Eiffel Tower completed?",
        meta={"_sim": {"answer": "1889", "gold_fact": "1889", "distractor": "1920"}}))
    report = engine.diagnose(inf, pipeline.run(inf, model), model.tier)
    assert any(e.signal == "base_rate" for e in report.primary.evidence)


def test_recommendation_not_self_credited_on_abstain_path():
    # ground_or_abstain with no retrievable gold must be UNSCORED, not a guaranteed hit.
    from tokentrace.core.types import Recommendation
    from tokentrace.models import load_model
    from tokentrace.recommend.recommender import Recommender

    rec = Recommendation(action="ground_or_abstain", description="", targets_mode=M.HALLUCINATION)
    # Production-style: no ground truth and no gold_fact, so there is nothing to
    # ground to -> add_gold_context returns None -> the abstain path must NOT self-credit.
    inf = Inference("What is the capital of Zorbia?", "Zorbia City", None, None,
                    question="What is the capital of Zorbia?",
                    meta={"_sim": {"answer": "unknown", "distractor": "Zorbia City"}})
    Recommender(model=load_model("mock-4b", tier=Tier.WHITE))._maybe_validate(rec, inf)
    assert rec.validated is None   # no gold to ground to -> unscored, not True
