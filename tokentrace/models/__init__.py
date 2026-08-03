"""Model backends. Import stays light — only the mock is eagerly importable;
GGUF/HF are imported lazily by :func:`load_model`."""

from __future__ import annotations

from tokentrace.models.base import (
    CaptureResult,
    GenerationResult,
    ModelHandle,
    TierUnavailable,
)
from tokentrace.models.mock import MockModel
from tokentrace.models.registry import (
    PROFILES,
    available_backends,
    get_profile,
    load_model,
)

__all__ = [
    "CaptureResult",
    "GenerationResult",
    "ModelHandle",
    "TierUnavailable",
    "MockModel",
    "PROFILES",
    "available_backends",
    "get_profile",
    "load_model",
]
