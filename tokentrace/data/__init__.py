"""Datasets: bundled seed pool, injection harness, synthetic dataset builder.

Real dataset loaders (RAGTruth, HotpotQA, NQ, TruthfulQA, BEIR) live in
``loaders.py`` and require the ``data`` extra; the rest of this package is
download-free.
"""

from __future__ import annotations

from tokentrace.data.injection import InjectionHarness
from tokentrace.data.samples import FACTS, MULTIHOP, Fact, MultiHopFact
from tokentrace.data.synthetic import build_dataset, dataset_summary, featurize

__all__ = [
    "InjectionHarness",
    "FACTS",
    "MULTIHOP",
    "Fact",
    "MultiHopFact",
    "build_dataset",
    "dataset_summary",
    "featurize",
]
