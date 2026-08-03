"""Scorer interface parity (offline — no model downloads)."""

from __future__ import annotations

import pytest

from tokentrace.signals import HeuristicScorers, ModelScorers

_METHODS = ["support", "relevance", "ambiguity", "match", "semantic_entropy", "self_consistency"]


def test_heuristic_scorers_expose_full_interface():
    s = HeuristicScorers()
    for m in _METHODS:
        assert callable(getattr(s, m, None)), f"missing scorer method: {m}"


def test_model_scorers_share_interface():
    # ModelScorers composes with the heuristics so ambiguity/match are inherited.
    assert issubclass(ModelScorers, HeuristicScorers)
    for m in _METHODS:
        assert callable(getattr(ModelScorers, m))


def test_model_scorers_clean_error_without_extra():
    # sentence-transformers is not in the CPU core; construction must fail loudly.
    with pytest.raises(ImportError, match="retrieval"):
        ModelScorers()
