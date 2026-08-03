"""Signal extraction: scorers, feature registry, extractors, pipeline."""

from __future__ import annotations

from tokentrace.signals.features import ALL_FEATURES, FEATURE_SPECS, family_of
from tokentrace.signals.registry import (
    ConfidenceLogprobExtractor,
    ConfidenceResampleExtractor,
    Extractor,
    MechanisticExtractor,
    MetaExtractor,
    PromptExtractor,
    RetrievalExtractor,
    SignalPipeline,
    default_extractors,
)
from tokentrace.signals.scorers import DEFAULT_SCORERS, HeuristicScorers, Scorers

__all__ = [
    "ALL_FEATURES",
    "FEATURE_SPECS",
    "family_of",
    "Extractor",
    "SignalPipeline",
    "default_extractors",
    "PromptExtractor",
    "RetrievalExtractor",
    "ConfidenceResampleExtractor",
    "ConfidenceLogprobExtractor",
    "MechanisticExtractor",
    "MetaExtractor",
    "DEFAULT_SCORERS",
    "HeuristicScorers",
    "Scorers",
]
