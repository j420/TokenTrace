from __future__ import annotations

import pytest

from tokentrace.core.types import Tier
from tokentrace.models import load_model


@pytest.fixture(scope="session")
def model():
    return load_model("mock-4b", backend="mock", tier=Tier.WHITE)


@pytest.fixture(scope="session")
def pipeline():
    from tokentrace.signals import SignalPipeline

    return SignalPipeline()
