"""Adapter for LangChain-style RAG runs.

The payoff over :mod:`~tokentrace.ingest.openai_chat` is that the retrieval
structure survives: ``source_documents`` is a list of real ``Document`` objects,
so chunk boundaries are the retriever's own, ``metadata['score']`` is the actual
retriever score when the chain kept it, and ``n_chunks`` /
``gold_position_frac`` are measurements rather than guesses. That is the
difference between the dilution rules firing on evidence and firing on a
paragraph-splitting artifact.

Accepted shape (all keys optional except a question and an answer)::

    {"question": "...",                        # or query / input / input_documents
     "result":   "...",                        # or answer / output_text / output / text
     "source_documents": [                     # or context / documents / contexts
        {"page_content": "...",
         "metadata": {"source": "...", "score": 0.83}}
     ],
     "ground_truth": "...",                    # only if an eval harness logged one
     "run_id": "..."}

Notes on the mapping choices
----------------------------
* ``answer`` is read as the *generated* answer, never as the reference. In
  RAGAS-shaped rows ``answer`` is the generation and ``ground_truth`` is the
  reference; getting that backwards would score every trace correct.
* ``context`` may be a plain string (LCEL chains that stuff the docs before the
  prompt). One string is one chunk — we do not re-split it, because unlike the
  chat adapter we have no reason to believe the retriever's boundaries are
  recoverable, and inventing extra chunks would inflate ``n_chunks``.
* ``metadata['gold']`` (and ``is_gold`` / ``is_ground_truth``) is honoured when
  present, because a labelled eval set genuinely knows which document is the
  answer-bearing one. Nothing else is read as a gold flag.
* A present-but-empty ``source_documents`` stays ``[]`` rather than becoming
  ``None``: a RAG chain that retrieved nothing is a retrieval failure, and
  collapsing it to non-RAG would hard-mask the very mode it exhibits.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from tokentrace.core.types import Inference
from tokentrace.ingest.base import (
    ChunkBuilder,
    compose_prompt,
    content_text,
    find_reference,
    first_present,
    first_text,
    gold_flag,
    ingest_meta,
    message_role,
    message_text,
    missing_field,
    register,
)

SOURCE = "langchain"

_QUESTION_PATHS = ("question", "query", "input", "inputs.question", "inputs.query", "inputs.input")
# `response` is intentionally absent: that is the LlamaIndex answer key, and
# claiming it here would make the two detectors overlap.
_ANSWER_PATHS = (
    "result", "answer", "output_text", "output", "text",
    "outputs.result", "outputs.answer", "outputs.output_text", "generated_answer",
)
_DOC_PATHS = ("source_documents", "context", "documents", "contexts", "docs",
              "outputs.source_documents")
_PROMPT_PATHS = ("prompt", "rendered_prompt", "full_prompt", "formatted_prompt",
                 "inputs.prompt", "llm_prompt")
_MODEL_PATHS = ("model", "llm", "model_name", "metadata.model", "metadata.ls_model_name")
_ID_PATHS = ("run_id", "id", "trace_id", "session_id")

# Similarity-shaped keys only. A *distance* (``distance_to_query``) is monotonically
# inverted relative to a score, so copying one into ``retriever_score`` would record
# the most relevant chunk as the least relevant one.
_SCORE_KEYS = ("score", "relevance_score", "similarity", "similarity_score",
               "retriever_score", "_score")
_SOURCE_KEYS = ("source", "file_path", "file_name", "doc_id", "document_id", "id", "url")


def _looks_like_document(value: Any) -> bool:
    """Anything ``_build_chunks`` can actually turn into a chunk.

    This deliberately mirrors what the builder accepts rather than demanding
    ``page_content``. When it demanded that key, ``detect`` rejected shapes that
    ``to_inference`` parses perfectly — a RAGAS row (``contexts: ["...", "..."]``,
    the single most common eval-log format) and a ``{"text": ...}``-keyed document
    both parsed but could never be auto-routed.
    """
    if isinstance(value, str):
        return bool(value.strip())
    # Exactly the keys _build_chunks reads below, and no more. Adding a key the
    # builder does not read (`body`) made this claim payloads it cannot parse —
    # notably a generic field-map trace, which must stay unclaimed so the explicit
    # map routes it.
    return isinstance(value, Mapping) and any(
        k in value for k in ("page_content", "pageContent", "text", "content")
    )


def _documents(payload: Any) -> Any:
    """Raw documents value, or ``None`` when the payload has no documents key."""
    return first_present(payload, _DOC_PATHS)


def _build_chunks(documents: Any) -> tuple[ChunkBuilder, str]:
    builder = ChunkBuilder()
    if isinstance(documents, str):
        builder.add(documents, source_id="context")
        return builder, "context_string"
    if not isinstance(documents, (list, tuple)):
        return builder, "none"
    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            builder.add(doc, source_id=f"doc{i}")
            continue
        if not isinstance(doc, Mapping):
            continue
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), Mapping) else {}
        text = first_present(doc, ("page_content", "pageContent", "text", "content"), "")
        # The score lives on the document in some chains and in metadata in others;
        # neither is canonical, so both are checked and absence stays absence.
        score = first_present(doc, tuple(f"['{k}']" for k in _SCORE_KEYS))
        if score is None:
            score = first_present(metadata, tuple(f"['{k}']" for k in _SCORE_KEYS))
        source_id = first_present(metadata, tuple(f"['{k}']" for k in _SOURCE_KEYS))
        if source_id is None:
            source_id = first_present(doc, ("id", "doc_id"))
        builder.add(
            text,
            score=score,
            source_id=source_id if source_id is not None else f"doc{i}",
            gold=gold_flag(metadata) if metadata else gold_flag(doc),
        )
    return builder, "source_documents"


# --------------------------------------------------------------------------- #
def detect(payload: Any) -> bool:
    """True for a LangChain chain output: a question + an answer, LangChain-shaped.

    The discriminating marker is ``page_content`` on the documents, and it is
    strong enough to win even when the payload ALSO carries the raw chat call —
    a callback handler that logs ``messages`` alongside ``source_documents`` is
    common, and this adapter is strictly better for it (structured retrieval beats
    a re-parsed prompt). ``openai_chat`` defers on the same signature.

    Without documents there is nothing LangChain-specific left, so we fall back to
    the ``question`` + ``result``-family key shape and step aside for anything that
    looks like a raw chat log.
    """
    if not isinstance(payload, Mapping):
        return False
    if any(str(k).startswith("gen_ai.") for k in payload):
        return False
    for foreign in ("source_nodes", "tokentrace_field_map"):
        if foreign in payload:
            return False
    if isinstance(payload.get("attributes"), (Mapping, list)):
        return False
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else None
    has_answer = first_text(payload, _ANSWER_PATHS) is not None
    if not has_answer and messages:
        # The answer may be the trailing assistant turn rather than a `result` key.
        # Requiring the key left a real shape unclaimed by EVERY adapter: openai_chat
        # defers on the documents, and this one refused for want of `result`.
        has_answer = _last_message_text(messages, "assistant") is not None
    if not has_answer:
        return False

    documents = _documents(payload)
    if isinstance(documents, (list, tuple)) and documents:
        # Documents present: they must actually look like retrieved documents.
        return any(_looks_like_document(d) for d in documents)
    if "messages" in payload or "choices" in payload:
        return False  # a chat log with no retrieval structure -> openai_chat's job
    # No documents (or an empty list): fall back to the chain-output key shape.
    return first_text(payload, _QUESTION_PATHS) is not None


def _last_message_text(messages: list, role: str) -> Optional[str]:
    """Text of the LAST message with this role — the live turn, not the first."""
    for message in reversed(messages):
        if message_role(message) == role:
            text = message_text(message)
            if text:
                return text
    return None


def to_inference(payload: Any) -> Inference:
    if not isinstance(payload, Mapping):
        raise missing_field(SOURCE, "chain output object", ("<root>",), payload)

    # A callback log may carry the chat transcript instead of a `question` field.
    # Since `openai_chat` correctly defers whenever structured retrieval is present,
    # this adapter has to be able to read that transcript, or such a payload routes
    # here and then fails — parseable by one adapter, claimed by another.
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else None
    answer = first_text(payload, _ANSWER_PATHS)
    if answer is None and messages:
        answer = _last_message_text(messages, "assistant")
    if answer is None:
        raise missing_field(SOURCE, "generated answer", _ANSWER_PATHS, payload)
    question = first_text(payload, _QUESTION_PATHS)
    question_origin = "question_field"
    if question is None and messages:
        question = _last_message_text(messages, "user")
        question_origin = "last_user_message"
    if question is None:
        raise missing_field(SOURCE, "user question", _QUESTION_PATHS, payload)

    documents = _documents(payload)
    builder, context_origin = _build_chunks(documents)
    # Absent key -> non-RAG (None); present-but-empty -> RAG that retrieved nothing.
    if documents is None:
        retrieved: Optional[list] = None
        context_origin = "none"
    else:
        retrieved = builder.chunks

    logged_prompt = first_text(payload, _PROMPT_PATHS)
    prompt_origin = "logged" if logged_prompt else "rebuilt"
    prompt = logged_prompt or compose_prompt(question, [c.text for c in builder.chunks])

    reference = find_reference(payload, "outputs.ground_truth", "metadata.ground_truth")
    return Inference(
        prompt=prompt,
        generated_answer=answer,
        retrieved_context=retrieved,
        ground_truth=reference,
        question=question,
        meta=ingest_meta(
            SOURCE,
            model_name=content_text(first_present(payload, _MODEL_PATHS)) or None,
            context_origin=context_origin,
            prompt_origin=prompt_origin,
            question_origin=question_origin,
            builder=builder,
            reference=reference,
            extra={"chain": content_text(first_present(payload, ("chain", "chain_type"))) or None},
        ),
        id=content_text(first_present(payload, _ID_PATHS)) or None,
    )


register(SOURCE, detect, to_inference, priority=40)
