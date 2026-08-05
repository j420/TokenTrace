"""Adapter for arbitrary payloads described by a user-supplied field map.

The other four adapters cover the frameworks; this one covers everybody else —
the in-house logging table, the vendor export, the Postgres row dumped to JSON.
Rather than adding a bespoke module per shape, the caller states where the fields
live::

    from tokentrace.ingest import load_any

    inf = load_any(payload, field_map={
        "question": "input.text",
        "answer":   "output.text",
        "chunks":   "docs[].body",
        "chunk_scores":     "docs[].relevance",
        "chunk_source_ids": "docs[].id",
    })

Recognized map keys (all optional except ``answer`` and one of
``prompt``/``question``):

===================  =======================================================
``prompt``           full rendered text the model saw
``question``         the user's actual query
``answer``           the generated answer            **required**
``chunks``           retrieved chunk texts (usually a ``[]`` path)
``chunk_scores``     retriever scores, positionally aligned with ``chunks``
``chunk_source_ids`` chunk identifiers, positionally aligned
``chunk_gold``       explicit gold flags, positionally aligned
``ground_truth``     reference answer(s), if the source has any
``id``               trace id
``model``            model name (recorded in ``meta``)
===================  =======================================================

Path syntax is :func:`tokentrace.ingest.base.resolve_path`: dotted keys, ``[i]``
for an index, ``[]`` (or ``[*]``) to map over a list, and ``['a.b']`` for keys
that contain dots.

Design notes
------------
* **Parallel paths, not a nested sub-map.** ``chunks``/``chunk_scores``/... each
  resolve to their own list and are zipped positionally. This is what makes
  ``"docs[].body"`` — the shape the caller naturally writes — work directly. The
  cost is that a ragged payload (some documents lacking a score) silently
  misaligns, so a length mismatch is a hard error rather than a truncation.
* **Only ``chunks`` decides RAG-ness.** If the map omits ``chunks``,
  ``retrieved_context`` is ``None`` and the retrieval modes are hard-masked. If it
  names a path that resolves to an empty list, that is a RAG run that retrieved
  nothing, and stays ``[]``.
* **A path that does not resolve is missing, not empty.** A typo'd path for a
  required field raises naming the path, instead of quietly producing an
  ``Inference`` with an empty prompt.
* **Auto-detection requires the map to be embedded.** A payload has no intrinsic
  "generic" signature, so :func:`detect` claims a payload only when it carries a
  ``tokentrace_field_map`` object — which is how a pipeline can self-describe its
  own log format. Otherwise pass ``field_map=`` to ``load_any``/``load_traces``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from tokentrace.core.types import Inference
from tokentrace.ingest.base import (
    ChunkBuilder,
    IngestError,
    compose_prompt,
    content_text,
    gold_flag,
    ingest_meta,
    normalize_reference,
    register,
    resolve_path,
)

SOURCE = "generic"

#: Key under which a payload may carry its own field map (enables auto-detection).
EMBEDDED_MAP_KEY = "tokentrace_field_map"

_KNOWN_KEYS = frozenset({
    "prompt", "question", "answer", "chunks", "chunk_scores", "chunk_source_ids",
    "chunk_gold", "ground_truth", "id", "model",
})

_MISSING = object()


def _get(payload: Any, field_map: Mapping[str, str], key: str) -> Any:
    path = field_map.get(key)
    if not path:
        return _MISSING
    value = resolve_path(payload, path, default=_MISSING)
    return value


def _as_list(value: Any) -> list[Any]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _aligned(name: str, values: Sequence[Any], n: int, path: Optional[str]) -> list[Any]:
    """Positionally align a per-chunk list, erroring loudly on a length mismatch."""
    if not values:
        return [None] * n
    if len(values) != n:
        raise IngestError(
            f"{SOURCE}: field map '{name}' -> {path!r} resolved to {len(values)} values "
            f"but 'chunks' resolved to {n}. Per-chunk paths are zipped positionally, so "
            f"they must be the same length; a ragged source needs one entry per chunk "
            f"(use null for the missing ones)."
        )
    return list(values)


# --------------------------------------------------------------------------- #
def detect(payload: Any) -> bool:
    """True only when the payload embeds its own field map."""
    return isinstance(payload, Mapping) and isinstance(payload.get(EMBEDDED_MAP_KEY), Mapping)


def to_inference(payload: Any) -> Inference:
    if not detect(payload):
        raise IngestError(
            f"{SOURCE}: no field map. Either embed one under {EMBEDDED_MAP_KEY!r} or call "
            f"load_any(payload, field_map={{...}})."
        )
    return from_field_map(payload, payload[EMBEDDED_MAP_KEY])


def from_field_map(payload: Any, field_map: Mapping[str, str]) -> Inference:
    """Convert ``payload`` using an explicit field map."""
    if not isinstance(field_map, Mapping) or not field_map:
        raise IngestError(f"{SOURCE}: field_map must be a non-empty mapping of field -> path")
    unknown = sorted(set(field_map) - _KNOWN_KEYS)
    if unknown:
        raise IngestError(
            f"{SOURCE}: unknown field-map key(s) {unknown}. "
            f"Supported: {', '.join(sorted(_KNOWN_KEYS))}."
        )

    answer_value = _get(payload, field_map, "answer")
    if answer_value is _MISSING:
        path = field_map.get("answer")
        raise IngestError(
            f"{SOURCE}: field map is missing the required 'answer' field"
            if not path
            else f"{SOURCE}: field map 'answer' -> {path!r} did not resolve in this payload"
        )
    answer = content_text(answer_value).strip()
    if not answer:
        raise IngestError(
            f"{SOURCE}: field map 'answer' -> {field_map['answer']!r} resolved to empty text"
        )

    # Strict for the structural fields (answer / question / prompt / chunks): if the
    # map names a path, every row is expected to have it, and a typo'd path is the
    # single most common field-map mistake — silently dropping the question would
    # leave `Inference.query` falling back to the whole prompt with no sign why.
    # Lenient for the rest (ground_truth / id / model / the per-chunk paths), which
    # legitimately vary row to row in a mixed eval-and-production log.
    question_value = _get(payload, field_map, "question")
    if "question" in field_map and question_value is _MISSING:
        raise IngestError(
            f"{SOURCE}: field map 'question' -> {field_map['question']!r} did not resolve in this "
            f"payload. Drop the key if this source has no separate user question."
        )
    question = content_text(question_value).strip() if question_value is not _MISSING else ""

    # ---- chunks -------------------------------------------------------- #
    chunks_value = _get(payload, field_map, "chunks")
    builder = ChunkBuilder()
    has_chunk_field = "chunks" in field_map
    if has_chunk_field and chunks_value is _MISSING:
        raise IngestError(
            f"{SOURCE}: field map 'chunks' -> {field_map['chunks']!r} did not resolve in this "
            f"payload. Drop the key for a non-RAG trace, or point it at the document list."
        )
    texts = _as_list(chunks_value)
    scores = _aligned("chunk_scores", _as_list(_get(payload, field_map, "chunk_scores")),
                      len(texts), field_map.get("chunk_scores"))
    ids = _aligned("chunk_source_ids", _as_list(_get(payload, field_map, "chunk_source_ids")),
                   len(texts), field_map.get("chunk_source_ids"))
    golds = _aligned("chunk_gold", _as_list(_get(payload, field_map, "chunk_gold")),
                     len(texts), field_map.get("chunk_gold"))
    for i, text in enumerate(texts):
        builder.add(
            text,
            score=scores[i],
            source_id=ids[i] if ids[i] is not None else f"c{i}",
            gold=gold_flag({"gold": golds[i]}) if golds[i] is not None else None,
        )

    # ---- prompt -------------------------------------------------------- #
    prompt_value = _get(payload, field_map, "prompt")
    if "prompt" in field_map and prompt_value is _MISSING:
        raise IngestError(
            f"{SOURCE}: field map 'prompt' -> {field_map['prompt']!r} did not resolve in this payload"
        )
    prompt = content_text(prompt_value).strip() if prompt_value is not _MISSING else ""
    prompt_origin = "logged" if prompt else "rebuilt"
    if not prompt:
        prompt = compose_prompt(question, [c.text for c in builder.chunks])
    if not prompt:
        raise IngestError(
            f"{SOURCE}: field map produced no prompt text — give it a 'prompt' path, "
            f"a 'question' path, or a 'chunks' path to rebuild one from."
        )

    reference_value = _get(payload, field_map, "ground_truth")
    reference = normalize_reference(None if reference_value is _MISSING else reference_value)

    id_value = _get(payload, field_map, "id")
    model_value = _get(payload, field_map, "model")

    return Inference(
        prompt=prompt,
        generated_answer=answer,
        retrieved_context=builder.chunks if has_chunk_field else None,
        ground_truth=reference,
        question=question or None,
        meta=ingest_meta(
            SOURCE,
            model_name=(content_text(model_value).strip() or None) if model_value is not _MISSING else None,
            context_origin=f"field_map:{field_map['chunks']}" if has_chunk_field else "none",
            prompt_origin=prompt_origin,
            question_origin=f"field_map:{field_map['question']}" if question else "absent",
            builder=builder,
            reference=reference,
            extra={"field_map": dict(field_map)},
        ),
        id=(content_text(id_value).strip() or None) if id_value is not _MISSING else None,
    )


# Checked first: an explicit, embedded field map is a statement of intent and must
# beat any structural sniffing.
register(SOURCE, detect, to_inference, priority=10)
