"""Content-addressed on-disk trace cache.

The CPU-first workflow is *pre-compute once, read many times*: mechanistic capture
and K-sample semantic entropy are the slow parts, so we cache the whole computed
trace (inference + feature vector + diagnosis report) keyed by a hash of
``(model, tier, prompt, context, ground truth, config)``. The Streamlit app and
eval loops then read cached traces instead of recomputing, which is what makes the
tool responsive on CPU.

Depends only on the standard library + the core contracts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from tokentrace.core.serialize import (
    inference_from_dict,
    inference_to_dict,
    report_from_dict,
    report_to_dict,
)
from tokentrace.core.types import (
    DiagnosisReport,
    FeatureVector,
    Inference,
    SignalFamily,
    Tier,
)


# --- FeatureVector (de)serialization (kept here to avoid touching core.serialize) --- #
def feature_vector_to_dict(fv: FeatureVector) -> dict[str, Any]:
    return {
        "values": fv.values,
        "missing": sorted(fv.missing),
        "family_present": [f.value for f in SignalFamily if fv.family_present.get(f)],
    }


def feature_vector_from_dict(d: dict[str, Any]) -> FeatureVector:
    fv = FeatureVector(values=dict(d.get("values", {})), missing=set(d.get("missing", [])))
    fv.family_present = {SignalFamily(v): True for v in d.get("family_present", [])}
    return fv


class TraceStore:
    def __init__(self, root: str | Path = ".tokentrace_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def make_key(self, inference: Inference, model_name: str, tier: Tier, extra: str = "") -> str:
        # Include EVERY field the captured signals depend on — notably per-chunk
        # gold/source_id/retriever_score and the question — so two inferences with
        # the same prompt text but different gold-marking don't collide to a stale
        # trace (rag.make_inference toggles exactly that).
        ctx = (
            [[c.text, c.gold, c.source_id, c.retriever_score] for c in inference.retrieved_context]
            if inference.retrieved_context is not None else None
        )
        payload = json.dumps({
            "model": model_name, "tier": tier.label, "prompt": inference.prompt,
            "question": inference.question, "answer": inference.generated_answer,
            "context": ctx, "gt": inference.ground_truth, "extra": extra,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    # ------------------------------------------------------------------ #
    def save(self, inference: Inference, model_name: str, tier: Tier,
             features: FeatureVector, report: DiagnosisReport, extra: str = "") -> str:
        key = self.make_key(inference, model_name, tier, extra)
        obj = {
            "key": key, "model": model_name, "tier": tier.label,
            "inference": inference_to_dict(inference),
            "features": feature_vector_to_dict(features),
            "report": report_to_dict(report),
        }
        self._path(key).write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        return key

    def load(self, inference: Inference, model_name: str, tier: Tier,
             extra: str = "") -> Optional[dict[str, Any]]:
        """Return the reconstructed trace dict (with Inference/FeatureVector/
        DiagnosisReport objects) or None on a miss."""
        key = self.make_key(inference, model_name, tier, extra)
        return self.load_key(key)

    def load_key(self, key: str) -> Optional[dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        return {
            "key": raw["key"], "model": raw["model"], "tier": raw["tier"],
            "inference": inference_from_dict(raw["inference"]),
            "features": feature_vector_from_dict(raw["features"]),
            "report": report_from_dict(raw["report"]),
        }

    def list_keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def clear(self) -> int:
        n = 0
        for p in self.root.glob("*.json"):
            p.unlink()
            n += 1
        return n
