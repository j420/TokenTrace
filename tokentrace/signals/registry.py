"""Signal extractors + the pipeline that assembles a :class:`FeatureVector`.

Each extractor declares its family, minimum tier, cost, and the features it
produces. The pipeline runs only applicable extractors (tier / RAG / GT gating)
and records everything absent in the feature vector's missingness mask — which is
what makes tiers and edge cases fall out of one mechanism.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Optional

from tokentrace.core.types import FeatureVector, Inference, SignalFamily, Tier
from tokentrace.models.base import ModelHandle, TierUnavailable
from tokentrace.signals.scorers import DEFAULT_SCORERS, HeuristicScorers, norm

_COMPARE_MARKERS = {
    "than", "more", "less", "before", "after", "both", "difference", "compare",
    "compared", "combined", "total", "older", "younger", "taller", "shorter",
    "first", "last", "between",
}


class Extractor(ABC):
    family: SignalFamily
    min_tier: Tier = Tier.BLACK
    cost: float = 1.0
    produces: list[str] = []
    requires_rag: bool = False

    def applicable(self, inference: Inference, tier: Tier) -> bool:
        if tier < self.min_tier:
            return False
        if self.requires_rag and not inference.is_rag:
            return False
        return True

    @abstractmethod
    def extract(
        self, inference: Inference, model: ModelHandle, scorers: HeuristicScorers
    ) -> dict[str, Optional[float]]:
        ...


# --------------------------------------------------------------------------- #
class PromptExtractor(Extractor):
    family = SignalFamily.PROMPT
    min_tier = Tier.BLACK
    cost = 0.1
    produces = ["prompt_ambiguity", "prompt_n_content_words", "multihop_query"]

    def extract(self, inference, model, scorers):
        q = inference.query
        toks = norm(q)
        # Group consecutive capitalized words into entities (so "Eiffel Tower" is
        # one entity, not two) and drop leading question/stop words.
        _QWORDS = {"When", "What", "Who", "Where", "Why", "How", "Which", "Is",
                   "Was", "Did", "Does", "Are", "The", "A", "An", "In", "On", "Whose"}
        entities = [
            e for e in re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", q)
            if e.split()[0] not in _QWORDS
        ]
        multihop = 1.0 if (any(m in toks for m in _COMPARE_MARKERS) or len(entities) >= 2) else 0.0
        return {
            "prompt_ambiguity": scorers.ambiguity(q),
            "prompt_n_content_words": float(len(toks)),
            "multihop_query": multihop,
        }


class RetrievalExtractor(Extractor):
    family = SignalFamily.RETRIEVAL
    min_tier = Tier.BLACK
    cost = 0.3
    requires_rag = True
    produces = [
        "answer_supported_by_context", "max_chunk_relevance", "mean_chunk_relevance",
        "n_chunks", "context_length_tokens", "gold_position_frac", "gold_recall_in_context",
    ]

    def extract(self, inference, model, scorers):
        chunks = inference.retrieved_context or []
        texts = [c.text for c in chunks]
        rels = [scorers.relevance(inference.query, t) for t in texts] or [0.0]
        ctx_tokens = sum(len(norm(t)) for t in texts)

        # Position of the answer-bearing chunk (prefer annotation, then GT, then answer).
        gold_idx = self._gold_index(inference, chunks, scorers)
        gold_pos = (gold_idx / (len(chunks) - 1)) if (gold_idx is not None and len(chunks) > 1) else None

        # GT-dependent discriminator: is the *correct* answer present in context at all?
        gold_recall = None
        if inference.has_ground_truth:
            gold_recall = max(
                (scorers.support(gt, texts) for gt in inference.ground_truth), default=0.0
            )
        return {
            "answer_supported_by_context": scorers.support(inference.generated_answer, texts),
            "max_chunk_relevance": max(rels),
            "mean_chunk_relevance": round(sum(rels) / len(rels), 4),
            "n_chunks": float(len(chunks)),
            "context_length_tokens": float(ctx_tokens),
            "gold_position_frac": gold_pos,
            "gold_recall_in_context": gold_recall,
        }

    @staticmethod
    def _gold_index(inference, chunks, scorers) -> Optional[int]:
        for i, c in enumerate(chunks):
            if c.gold:
                return i
        refs = inference.ground_truth or [inference.generated_answer]
        best_i, best_s = None, 0.35  # threshold: must be reasonably supported
        for i, c in enumerate(chunks):
            s = max((scorers.support(r, [c.text]) for r in refs), default=0.0)
            if s > best_s:
                best_i, best_s = i, s
        return best_i


class ConfidenceResampleExtractor(Extractor):
    """Reference-free confidence via resampling — works even black-box."""

    family = SignalFamily.CONFIDENCE
    min_tier = Tier.BLACK
    cost = 5.0  # K generations
    produces = ["semantic_entropy", "self_consistency"]

    def __init__(self, k: int = 8, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

    def extract(self, inference, model, scorers):
        samples = model.sample(inference.query if not inference.prompt else inference.prompt,
                               k=self.k, temperature=self.temperature)
        return {
            "semantic_entropy": scorers.semantic_entropy(samples),
            "self_consistency": scorers.self_consistency(samples, inference.generated_answer),
        }


class ConfidenceLogprobExtractor(Extractor):
    """Token-level confidence from logprobs — grey-box+."""

    family = SignalFamily.CONFIDENCE
    min_tier = Tier.GREY
    cost = 1.0
    produces = ["mean_token_entropy", "max_token_entropy", "answer_perplexity"]

    def extract(self, inference, model, scorers):
        gen = model.confidence(inference.prompt, answer=inference.generated_answer)
        if not gen.token_entropies:
            return {}
        ent = gen.token_entropies
        mean_e = sum(ent) / len(ent)
        ppl = math.exp(-sum(gen.token_logprobs) / len(gen.token_logprobs)) if gen.token_logprobs else None
        return {
            "mean_token_entropy": round(mean_e, 4),
            "max_token_entropy": round(max(ent), 4),
            "answer_perplexity": round(ppl, 4) if ppl is not None else None,
        }


class MechanisticExtractor(Extractor):
    family = SignalFamily.MECHANISTIC
    min_tier = Tier.GREY
    cost = 3.0
    produces = [
        "context_attention_ratio", "gold_attention_ratio", "logit_lens_answer_layer",
        "logit_lens_stability", "external_context_score", "parametric_knowledge_score",
        "gold_patch_effect",
    ]

    def extract(self, inference, model, scorers):
        try:
            cap = model.capture(inference)
        except TierUnavailable:
            return {}
        return {
            "context_attention_ratio": cap.context_attention_ratio,
            "gold_attention_ratio": cap.gold_attention_ratio,
            "logit_lens_answer_layer": cap.logit_lens_answer_layer,
            "logit_lens_stability": cap.logit_lens_stability,
            "external_context_score": cap.external_context_score,
            "parametric_knowledge_score": cap.parametric_knowledge_score,
            "gold_patch_effect": cap.gold_patch_effect,  # None unless white-box
        }


class MetaExtractor(Extractor):
    """Correctness + shape features. GT-dependent items become missing in
    production. ``family = None`` so these never flip a signal family to present."""

    family = None  # type: ignore[assignment]
    min_tier = Tier.BLACK
    cost = 0.1
    produces = ["is_correct", "answer_length_tokens"]

    def extract(self, inference, model, scorers):
        out: dict[str, Optional[float]] = {
            "answer_length_tokens": float(len(norm(inference.generated_answer))),
            "is_correct": None,
        }
        if inference.has_ground_truth:
            out["is_correct"] = 1.0 if scorers.match(
                inference.generated_answer, inference.ground_truth
            ) >= 0.6 else 0.0
        return out


def default_extractors() -> list[Extractor]:
    return [
        PromptExtractor(),
        RetrievalExtractor(),
        ConfidenceResampleExtractor(),
        ConfidenceLogprobExtractor(),
        MechanisticExtractor(),
        MetaExtractor(),
    ]


class SignalPipeline:
    """Runs applicable extractors and assembles a :class:`FeatureVector`."""

    def __init__(self, extractors: Optional[list[Extractor]] = None, scorers=DEFAULT_SCORERS):
        self.extractors = extractors if extractors is not None else default_extractors()
        self.scorers = scorers

    def run(self, inference: Inference, model: ModelHandle) -> FeatureVector:
        bound = model.bound_to(inference)  # keeps mock generation consistent; no-op for real
        fv = FeatureVector()
        # Recorded up front so it is part of the missingness signature even if no
        # GT-dependent extractor runs (production path).
        fv.reference_available = inference.has_ground_truth
        for ex in self.extractors:
            if not ex.applicable(inference, model.tier):
                continue
            try:
                feats = ex.extract(inference, bound, self.scorers)
            except TierUnavailable:
                continue
            for name, val in feats.items():
                fv.set(name, val, ex.family)
        return fv
