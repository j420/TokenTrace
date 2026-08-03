"""Canonical feature names and their families.

A single source of truth for the fixed feature ordering the learned heads consume
(so a saved model's columns are stable) and for which family each feature belongs
to (so the missingness mask is computed correctly).
"""

from __future__ import annotations

from tokentrace.core.types import SignalFamily as F

# name -> (family, requires_rag, requires_ground_truth, min_tier_label)
FEATURE_SPECS: dict[str, tuple[F, bool, bool, str]] = {
    # --- prompt (black-box) --- #
    "prompt_ambiguity": (F.PROMPT, False, False, "black"),
    "prompt_n_content_words": (F.PROMPT, False, False, "black"),
    "multihop_query": (F.PROMPT, False, False, "black"),
    # --- retrieval (black-box, RAG only) --- #
    "answer_supported_by_context": (F.RETRIEVAL, True, False, "black"),
    "max_chunk_relevance": (F.RETRIEVAL, True, False, "black"),
    "mean_chunk_relevance": (F.RETRIEVAL, True, False, "black"),
    "n_chunks": (F.RETRIEVAL, True, False, "black"),
    "context_length_tokens": (F.RETRIEVAL, True, False, "black"),
    "gold_position_frac": (F.RETRIEVAL, True, False, "black"),
    "gold_recall_in_context": (F.RETRIEVAL, True, True, "black"),  # GT-dependent discriminator
    # --- confidence: resampling (black-box) --- #
    "semantic_entropy": (F.CONFIDENCE, False, False, "black"),
    "self_consistency": (F.CONFIDENCE, False, False, "black"),
    # --- confidence: logprobs (grey-box) --- #
    "mean_token_entropy": (F.CONFIDENCE, False, False, "grey"),
    "max_token_entropy": (F.CONFIDENCE, False, False, "grey"),
    "answer_perplexity": (F.CONFIDENCE, False, False, "grey"),
    # --- mechanistic (grey-box) --- #
    "context_attention_ratio": (F.MECHANISTIC, False, False, "grey"),
    "gold_attention_ratio": (F.MECHANISTIC, False, False, "grey"),
    "logit_lens_answer_layer": (F.MECHANISTIC, False, False, "grey"),
    "logit_lens_stability": (F.MECHANISTIC, False, False, "grey"),
    "external_context_score": (F.MECHANISTIC, False, False, "grey"),
    "parametric_knowledge_score": (F.MECHANISTIC, False, False, "grey"),
    # --- mechanistic (white-box, causal) --- #
    "gold_patch_effect": (F.MECHANISTIC, False, False, "white"),
    # --- meta / correctness (GT-dependent, always-tier) --- #
    "is_correct": (F.PROMPT, False, True, "black"),   # family arbitrary; used by rules/gates
    "answer_length_tokens": (F.PROMPT, False, False, "black"),
}

#: Fixed ordering for array construction / model columns.
ALL_FEATURES: list[str] = list(FEATURE_SPECS.keys())


def family_of(name: str) -> F:
    return FEATURE_SPECS[name][0]
