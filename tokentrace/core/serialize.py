"""JSON (de)serialization for the core contracts.

Used by the trace cache and the Streamlit app so that traces computed offline
(the CPU-first workflow: pre-compute once, read many times) round-trip cleanly.
Enums serialize to their value/name; sets and tuples to lists.
"""

from __future__ import annotations

from typing import Any

from tokentrace.core.types import (
    Chunk,
    Diagnosis,
    DiagnosisReport,
    DiagnosisRole,
    EvidenceItem,
    FailureMode,
    Inference,
    LabeledInference,
    Provenance,
    Recommendation,
    SignalFamily,
    Tier,
)


# --------------------------- to dict --------------------------- #
def chunk_to_dict(c: Chunk) -> dict[str, Any]:
    return {
        "text": c.text,
        # Scrubbed like every other float we emit: a retriever that returns NaN
        # (a cosine over a zero vector will) otherwise wrote a bare NaN token into
        # cached traces and `analyze --json`, which _fin exists to prevent. This
        # was the one float serializer that skipped it.
        "retriever_score": _fin(c.retriever_score),
        "source_id": c.source_id,
        "gold": c.gold,
        "char_span": list(c.char_span) if c.char_span else None,
    }


def inference_to_dict(inf: Inference) -> dict[str, Any]:
    return {
        "id": inf.id,
        "prompt": inf.prompt,
        "generated_answer": inf.generated_answer,
        "retrieved_context": (
            [chunk_to_dict(c) for c in inf.retrieved_context]
            if inf.retrieved_context is not None
            else None
        ),
        "ground_truth": inf.ground_truth,
        "question": inf.question,
        "meta": inf.meta,
    }


def _fin(x: Any) -> Any:
    """NaN/inf -> None.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens, which are NOT valid
    RFC 8259 JSON: they break ``tokentrace analyze --json`` for any strict consumer
    and make every cached trace file unparseable by non-Python readers.
    """
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def evidence_to_dict(e: EvidenceItem) -> dict[str, Any]:
    return {
        "signal": e.signal,
        "family": e.family.value,
        "value": _fin(e.value),
        "contribution_logodds": _fin(e.contribution_logodds),
        "direction": e.direction,
        "source": e.source,
        "provenance": e.provenance,
        "rendered": e.rendered,
    }


def recommendation_to_dict(r: Recommendation) -> dict[str, Any]:
    return {
        "action": r.action,
        "description": r.description,
        "targets_mode": r.targets_mode.value,
        "priority": r.priority,
        "validated": r.validated,
        "validation_detail": r.validation_detail,
    }


def diagnosis_to_dict(d: Diagnosis) -> dict[str, Any]:
    return {
        "mode": d.mode.value,
        "probability": d.probability,
        "rank": d.rank,
        "role": d.role.value if d.role else None,
        "causal_parents": [m.value for m in d.causal_parents],
        "evidence": [evidence_to_dict(e) for e in d.evidence],
        "confidence": d.confidence,
        "recommendations": [recommendation_to_dict(r) for r in d.recommendations],
    }


def report_to_dict(rep: DiagnosisReport) -> dict[str, Any]:
    return {
        "inference_id": rep.inference_id,
        "tier": rep.tier.label,
        "diagnoses": [diagnosis_to_dict(d) for d in rep.diagnoses],
        "conformal_set": [m.value for m in rep.conformal_set],
        "diagnostic_confidence": rep.diagnostic_confidence,
        "abstained": rep.abstained,
        "notes": rep.notes,
    }


def labeled_to_dict(li: LabeledInference) -> dict[str, Any]:
    return {
        "inference": inference_to_dict(li.inference),
        "labels": [m.value for m in li.labels],
        "causal_edges": [[a.value, b.value] for a, b in li.causal_edges],
        "severity": {m.value: v for m, v in li.severity.items()},
        "provenance": li.provenance.value,
        "injection_recipe": li.injection_recipe,
        "verification": li.verification,
        "weight": li.weight,
    }


# --------------------------- from dict --------------------------- #
def chunk_from_dict(d: dict[str, Any]) -> Chunk:
    span = d.get("char_span")
    # `.get(k, default)` does NOT cover an explicit null, and `chunk_to_dict` now
    # writes one whenever _fin scrubs a non-finite score. Without this the round trip
    # returned retriever_score=None -- violating the float annotation, and worse,
    # changing TraceStore.make_key's hash, so a reloaded trace could never be found
    # in the cache again and every re-save minted a new file. Every other _fin-scrubbed
    # field already had this guard on the way back in; this one was missed.
    score = d.get("retriever_score")
    return Chunk(
        text=d["text"],
        retriever_score=0.0 if score is None else score,
        source_id=d.get("source_id", ""),
        gold=d.get("gold"),
        char_span=tuple(span) if span else None,
    )


def inference_from_dict(d: dict[str, Any]) -> Inference:
    ctx = d.get("retrieved_context")
    return Inference(
        prompt=d["prompt"],
        generated_answer=d["generated_answer"],
        retrieved_context=[chunk_from_dict(c) for c in ctx] if ctx is not None else None,
        ground_truth=d.get("ground_truth"),
        question=d.get("question"),
        meta=d.get("meta", {}),
        id=d.get("id"),
    )


def evidence_from_dict(d: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        signal=d["signal"],
        family=SignalFamily(d["family"]),
        value=d["value"] if d.get("value") is not None else 0.0,
        contribution_logodds=d["contribution_logodds"] or 0.0,
        direction=d.get("direction", "supports"),
        source=d.get("source", "rule"),
        provenance=d.get("provenance", {}),
        rendered=d.get("rendered", ""),
    )


def recommendation_from_dict(d: dict[str, Any]) -> Recommendation:
    return Recommendation(
        action=d["action"],
        description=d["description"],
        targets_mode=FailureMode(d["targets_mode"]),
        priority=d.get("priority", 0),
        validated=d.get("validated"),
        validation_detail=d.get("validation_detail", ""),
    )


def diagnosis_from_dict(d: dict[str, Any]) -> Diagnosis:
    return Diagnosis(
        mode=FailureMode(d["mode"]),
        probability=d["probability"],
        rank=d.get("rank", 0),
        role=DiagnosisRole(d["role"]) if d.get("role") else None,
        causal_parents=[FailureMode(m) for m in d.get("causal_parents", [])],
        evidence=[evidence_from_dict(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
        recommendations=[recommendation_from_dict(r) for r in d.get("recommendations", [])],
    )


def report_from_dict(d: dict[str, Any]) -> DiagnosisReport:
    tier_map = {t.label: t for t in Tier}
    return DiagnosisReport(
        diagnoses=[diagnosis_from_dict(x) for x in d.get("diagnoses", [])],
        tier=tier_map.get(d.get("tier", "black"), Tier.BLACK),
        inference_id=d.get("inference_id"),
        conformal_set=[FailureMode(m) for m in d.get("conformal_set", [])],
        diagnostic_confidence=d.get("diagnostic_confidence", 0.0),
        abstained=d.get("abstained", False),
        notes=d.get("notes", []),
    )


def labeled_from_dict(d: dict[str, Any]) -> LabeledInference:
    return LabeledInference(
        inference=inference_from_dict(d["inference"]),
        labels=[FailureMode(m) for m in d.get("labels", [])],
        causal_edges=[(FailureMode(a), FailureMode(b)) for a, b in d.get("causal_edges", [])],
        severity={FailureMode(m): v for m, v in d.get("severity", {}).items()},
        provenance=Provenance(d.get("provenance", "synthetic")),
        injection_recipe=d.get("injection_recipe"),
        verification=d.get("verification", {}),
        weight=d.get("weight", 1.0),
    )
