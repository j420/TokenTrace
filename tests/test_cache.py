"""Trace cache round-trip (offline)."""

from __future__ import annotations

from tokentrace.cache import TraceStore
from tokentrace.core.types import Tier

from _helpers import observe, retrieval_failure


def test_trace_store_roundtrip(tmp_path, model, pipeline):
    from tokentrace.engine import DiagnosisEngine

    store = TraceStore(root=tmp_path / "cache")
    engine = DiagnosisEngine()
    inf = observe(model, retrieval_failure())
    fv = pipeline.run(inf, model)
    report = engine.diagnose(inf, fv, model.tier)

    key = store.save(inf, "mock-4b", model.tier, fv, report)
    assert store.has(key)
    assert key in store.list_keys()

    loaded = store.load(inf, "mock-4b", model.tier)
    assert loaded is not None
    assert loaded["report"].primary.mode == report.primary.mode
    assert loaded["features"].missingness_signature() == fv.missingness_signature()
    assert loaded["inference"].generated_answer == inf.generated_answer

    # miss on a different tier
    assert store.load(inf, "mock-4b", Tier.BLACK) is None
    assert store.clear() == 1
