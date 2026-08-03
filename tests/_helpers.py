"""Shared scenario builders for the tests (not collected as tests)."""

from __future__ import annotations

from tokentrace.core.types import Chunk, Inference
from tokentrace.models.base import ModelHandle

GOLD = "The Eiffel Tower was completed in 1889."


def _sim(**k):
    return {"_sim": k}


def grounded() -> Inference:
    g = Chunk(GOLD, 0.9, "d1", gold=True)
    return Inference("When was the Eiffel Tower completed?\n" + GOLD, "", [g], ["1889"],
                     question="When was the Eiffel Tower completed?",
                     meta=_sim(answer="1889", gold_fact="completed in 1889"))


def retrieval_failure() -> Inference:
    return Inference("When was the Eiffel Tower completed?\nParis is in France.", "",
                     [Chunk("Paris is in France.", 0.5, "d2")], ["1889"],
                     question="When was the Eiffel Tower completed?",
                     meta=_sim(answer="1889", gold_fact="1889", distractor="1920"))


def parametric_fragile() -> Inference:
    return Inference("When was the Eiffel Tower completed?\nParis is in France.", "",
                     [Chunk("Paris is in France.", 0.5, "d2")], ["1889"],
                     question="When was the Eiffel Tower completed?",
                     meta=_sim(answer="1889", gold_fact="1889", parametric_known=True))


def dilution() -> Inference:
    many = [Chunk(f"Filler {i} about unrelated topics facts words here now today plus more." * 3,
                  0.4, f"f{i}") for i in range(8)]
    many.insert(4, Chunk(GOLD, 0.6, "g", gold=True))
    return Inference("When was the Eiffel Tower completed?\n" + " ".join(c.text for c in many),
                     "", many, ["1889"], question="When was the Eiffel Tower completed?",
                     meta=_sim(answer="1889", gold_fact="completed in 1889", distractor="1850"))


def reasoning_failure() -> Inference:
    g2 = Chunk("Marie Curie was born in 1867. Pierre Curie was born in 1859.", 0.9, "g2", gold=True)
    return Inference("Who was born first, Marie Curie or Pierre Curie?\n" + g2.text, "",
                     [g2], ["Pierre Curie"], question="Who was born first, Marie Curie or Pierre Curie?",
                     meta=_sim(answer="Pierre Curie", gold_fact="Pierre Curie was born in 1859",
                               requires_multihop=True, hard_composition=True, distractor="Marie Curie"))


def nonrag_hallucination() -> Inference:
    return Inference("What is the capital of the fictional country Zorbia?", "", None, ["unknown"],
                     question="What is the capital of the fictional country Zorbia?",
                     meta=_sim(answer="unknown", distractor="Zorbia City"))


def observe(model: ModelHandle, inf: Inference) -> Inference:
    """Populate the observed answer the way the real workflow would."""
    inf.generated_answer = model.bound_to(inf).generate(inf.prompt).text
    return inf
