"""Trace ingestion — the front door.

TokenTrace's analysis contract (:class:`~tokentrace.core.types.Inference`) asks for
things no production system logs in that shape: per-chunk retriever scores, gold
annotations, a reference answer. This package converts what people *do* already
have — an OpenAI request/response pair, a LangChain chain output, a LlamaIndex
response, an OpenTelemetry GenAI span, or an arbitrary JSON row plus a field map —
into that contract, honestly::

    from tokentrace import TokenTrace
    from tokentrace.ingest import load_traces

    tt = TokenTrace.default(train=False)
    for inference in load_traces("prod_traces.jsonl"):
        report = tt.analyze(inference)

The degraded case is the *primary* case. Real logs have no reference answer and no
gold flags, so an ingested trace normally has ``ground_truth=None`` and
``Chunk.gold=None``; the reference-dependent features drop out of the feature
vector, the missingness signature gains its ``|noref`` suffix, the rules that need
them abstain, and the report comes back as risk rather than confirmed error. That
path is the one the adapters are designed and tested for — anything a source
genuinely carries (a logged score, an eval harness's reference, an explicit gold
annotation) is mapped, and nothing else is invented. See
:mod:`tokentrace.ingest.base` for the exact missing-value policy.

Each adapter module exposes ``detect(payload) -> bool`` and
``to_inference(payload) -> Inference`` and self-registers, so :func:`load_any`
routes an unknown payload automatically and third-party adapters can join via
:func:`register`.

Note: :mod:`tokentrace.ingest.langchain` shadows the real ``langchain`` package
*only* for absolute imports of that exact dotted path; ``import langchain``
elsewhere is unaffected. This package imports nothing outside the standard library
and :mod:`tokentrace.core`.
"""

from __future__ import annotations

from tokentrace.ingest import generic, langchain, llamaindex, openai_chat, otel
from tokentrace.ingest.base import (
    Adapter,
    ChunkBuilder,
    IngestError,
    UnknownSourceError,
    adapters,
    compose_prompt,
    detect_all,
    detect_source,
    find_reference,
    first_present,
    first_text,
    gold_flag,
    load_any,
    load_traces,
    make_chunk,
    register,
    render_messages,
    resolve_path,
)

__all__ = [
    # routing
    "load_any",
    "load_traces",
    "detect_source",
    "detect_all",
    "adapters",
    "register",
    "Adapter",
    # errors
    "IngestError",
    "UnknownSourceError",
    # helpers (useful when writing an adapter of your own)
    "ChunkBuilder",
    "make_chunk",
    "resolve_path",
    "first_present",
    "first_text",
    "find_reference",
    "gold_flag",
    "render_messages",
    "compose_prompt",
    # adapter modules
    "openai_chat",
    "langchain",
    "llamaindex",
    "otel",
    "generic",
]
