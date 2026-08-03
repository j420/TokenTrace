"""Diagnosis engine: rules, learned residual, calibration, causal resolution,
evidence attribution, conformal Top-3, abstention."""

from __future__ import annotations

from tokentrace.engine.calibration import Calibrator, sigmoid
from tokentrace.engine.causal import CausalResolver, EvidenceAttributor, parents_of
from tokentrace.engine.classifier import ResidualClassifier
from tokentrace.engine.conformal import ConformalPredictor
from tokentrace.engine.diagnosis import DiagnosisEngine
from tokentrace.engine.rules import RULES, evaluate_rules

__all__ = [
    "DiagnosisEngine",
    "ResidualClassifier",
    "Calibrator",
    "sigmoid",
    "CausalResolver",
    "EvidenceAttributor",
    "parents_of",
    "ConformalPredictor",
    "RULES",
    "evaluate_rules",
]
