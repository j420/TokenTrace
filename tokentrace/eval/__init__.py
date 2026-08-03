"""Evaluation harness + metrics."""

from __future__ import annotations

from tokentrace.eval.benchmark import (
    evaluate,
    run_reports,
    split_dataset,
    train_and_evaluate,
    train_engine,
)
from tokentrace.eval.metrics import Metrics, compute_metrics, gold_primary, predicted_primary

__all__ = [
    "train_and_evaluate",
    "train_engine",
    "evaluate",
    "run_reports",
    "split_dataset",
    "Metrics",
    "compute_metrics",
    "gold_primary",
    "predicted_primary",
]
