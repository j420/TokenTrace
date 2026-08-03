"""TokenTrace — evidence-based root-cause analysis for LLM applications.

Top-level imports are kept dependency-light: importing :mod:`tokentrace` pulls in
only the core contracts, never torch/transformers/faiss. Heavier subsystems are
imported from their submodules on demand.
"""

from __future__ import annotations

__version__ = "0.1.0"

from tokentrace.core.types import (
    ALL_MODES,
    Chunk,
    Diagnosis,
    DiagnosisReport,
    DiagnosisRole,
    EvidenceItem,
    FailureMode,
    FeatureVector,
    Inference,
    LabeledInference,
    ModelProfile,
    Provenance,
    Recommendation,
    SignalFamily,
    Tier,
)

def __getattr__(name: str):
    # Lazy top-level access to the facade so `import tokentrace` stays light.
    if name == "TokenTrace":
        from tokentrace.api import TokenTrace

        return TokenTrace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "TokenTrace",
    "ALL_MODES",
    "Chunk",
    "Diagnosis",
    "DiagnosisReport",
    "DiagnosisRole",
    "EvidenceItem",
    "FailureMode",
    "FeatureVector",
    "Inference",
    "LabeledInference",
    "ModelProfile",
    "Provenance",
    "Recommendation",
    "SignalFamily",
    "Tier",
]
