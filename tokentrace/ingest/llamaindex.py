"""Adapter for LlamaIndex query-engine responses.

A serialized ``Response`` is the friendliest RAG payload TokenTrace can receive:
``source_nodes`` is a list of ``NodeWithScore``, so both the chunk boundary *and*
the similarity score are the retriever's own. Unlike LangChain, LlamaIndex puts
the score on the wrapper rather than in the node metadata, which is the one thing
that needs care here.

Accepted shape::

    {"response": "...",
     "query": "...",                            # or query_str / question
     "source_nodes": [
        {"node": {"id_": "...", "text": "...", "metadata": {"file_name": "..."}},
         "score": 0.81}
     ],
     "metadata": {...}}

Flat nodes (``{"text": ..., "score": ..., "id_": ...}``, which is what several
tracing integrations emit after dropping the wrapper) are handled too.

Mapping notes
-------------
* A ``Response`` object does not carry the query — LlamaIndex answers and forgets.
  When the log did not add one, ``question`` is ``None`` and
  :attr:`Inference.query` falls back to the prompt, which for this adapter is the
  concatenated context. Prompt-family signals are correspondingly weak, and
  ``meta['ingest']['question_origin'] == 'absent'`` says so.
* ``score`` on a ``NodeWithScore`` is optional in LlamaIndex itself (it is ``None``
  for nodes that came from a keyword or summary index). An unscored node keeps the
  default ``retriever_score`` and is reported through the ``partial``/``absent``
  provenance rather than being written down as 0.0.
* Only similarity-shaped keys are read as a score (:data:`_SCORE_KEYS`). A
  *distance* is inverted relative to a similarity and would rank the retrieval
  backwards; it stays unmeasured.
* ``node.metadata`` is *not* mined for a gold flag beyond the explicit gold keys —
  LlamaIndex metadata is user-defined and routinely contains ``relevance``,
  ``label``, ``score`` keys that mean something else entirely.
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
    missing_field,
    register,
)

SOURCE = "llamaindex"

_ANSWER_PATHS = ("response", "response.response", "answer", "result", "output",
                 "generated_answer", "text")
_QUESTION_PATHS = ("query_str", "query", "question", "user_query", "metadata.query_str",
                   "metadata.query")
_NODE_PATHS = ("source_nodes", "response.source_nodes", "sources", "nodes")
_PROMPT_PATHS = ("prompt", "formatted_prompt", "rendered_prompt", "metadata.prompt")
_MODEL_PATHS = ("model", "model_name", "llm", "metadata.model", "metadata.llm")
_ID_PATHS = ("id", "response_id", "trace_id", "query_id")

_TEXT_PATHS = ("text", "text_resource.text", "content", "get_content", "node_content")
_NODE_ID_PATHS = ("id_", "node_id", "id", "doc_id", "ref_doc_id")
_META_SOURCE_KEYS = ("file_name", "source", "file_path", "doc_id", "document_id", "url", "title")

#: Similarity-shaped keys only, and the same correctness invariant ``langchain``
#: states for ``langchain._SCORE_KEYS``: a *distance* (``distance``,
#: ``_distance``, ``distance_to_query``) is monotonically INVERTED relative to a
#: similarity, so copying one into ``retriever_score`` records the most relevant
#: node as the least relevant one. Vector stores that LlamaIndex wraps (Chroma,
#: FAISS with an L2 index) return exactly that key, so the temptation to add it
#: here is real — it was named only in a sibling module's comment, with nothing
#: enforcing it on this side. An unrecognized key leaves the score *unmeasured*,
#: which the ``absent``/``partial`` provenance reports honestly.
_SCORE_KEYS = ("score", "similarity")


def _looks_like_node(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if isinstance(value.get("node"), Mapping):
        return True
    # A flattened node still carries a LlamaIndex-shaped identifier or text+score.
    return ("id_" in value) or ("text" in value and "score" in value)


def _build_chunks(nodes: Any) -> ChunkBuilder:
    builder = ChunkBuilder()
    if not isinstance(nodes, (list, tuple)):
        return builder
    for i, item in enumerate(nodes):
        if isinstance(item, str):
            builder.add(item, source_id=f"node{i}")
            continue
        if not isinstance(item, Mapping):
            continue
        node = item.get("node") if isinstance(item.get("node"), Mapping) else item
        metadata = node.get("metadata") if isinstance(node.get("metadata"), Mapping) else {}

        text = first_text(node, _TEXT_PATHS)
        # The score sits on the NodeWithScore wrapper; fall back to the node itself
        # for integrations that flattened the pair.
        score = first_present(item, _SCORE_KEYS)
        if score is None:
            score = first_present(node, _SCORE_KEYS)

        node_id = first_present(node, _NODE_ID_PATHS)
        source_id = first_present(metadata, tuple(f"['{k}']" for k in _META_SOURCE_KEYS))
        builder.add(
            text,
            score=score,
            source_id=source_id if source_id is not None else (node_id or f"node{i}"),
            gold=gold_flag(metadata),
        )
    return builder


# --------------------------------------------------------------------------- #
def detect(payload: Any) -> bool:
    """True for a LlamaIndex response: an answer plus ``source_nodes``.

    ``source_nodes`` is the signature — no other source in this package uses it —
    but it must contain node-shaped entries, so an unrelated payload that happens
    to reuse the key is still rejected. A payload that also carries the raw chat
    call is still ours: structured nodes beat a re-parsed prompt, and
    ``openai_chat`` defers on ``source_nodes`` for exactly that reason.
    """
    if not isinstance(payload, Mapping):
        return False
    if any(str(k).startswith("gen_ai.") for k in payload):
        return False
    for foreign in ("page_content", "source_documents", "tokentrace_field_map"):
        if foreign in payload:
            return False
    if isinstance(payload.get("attributes"), (Mapping, list)):
        return False
    if first_text(payload, _ANSWER_PATHS) is None:
        return False
    nodes = first_present(payload, _NODE_PATHS)
    if isinstance(nodes, (list, tuple)):
        return not nodes or any(_looks_like_node(n) for n in nodes)
    return False


def to_inference(payload: Any) -> Inference:
    if not isinstance(payload, Mapping):
        raise missing_field(SOURCE, "response object", ("<root>",), payload)

    answer = first_text(payload, _ANSWER_PATHS)
    if answer is None:
        raise missing_field(SOURCE, "response text", _ANSWER_PATHS, payload)

    nodes = first_present(payload, _NODE_PATHS)
    builder = _build_chunks(nodes)
    # Absent key -> non-RAG; present-but-empty -> a query engine that retrieved nothing.
    retrieved: Optional[list] = None if nodes is None else builder.chunks

    question = first_text(payload, _QUESTION_PATHS)
    logged_prompt = first_text(payload, _PROMPT_PATHS)
    prompt = logged_prompt or compose_prompt(question, [c.text for c in builder.chunks])
    if not prompt:
        raise missing_field(SOURCE, "query text or source-node text to rebuild a prompt from",
                            _QUESTION_PATHS + _NODE_PATHS, payload)

    reference = find_reference(payload, "metadata.ground_truth", "metadata.reference")
    return Inference(
        prompt=prompt,
        generated_answer=answer,
        retrieved_context=retrieved,
        ground_truth=reference,
        question=question,
        meta=ingest_meta(
            SOURCE,
            model_name=content_text(first_present(payload, _MODEL_PATHS)) or None,
            context_origin="source_nodes" if nodes is not None else "none",
            prompt_origin="logged" if logged_prompt else "rebuilt",
            question_origin="query_field" if question else "absent",
            builder=builder,
            reference=reference,
        ),
        id=content_text(first_present(payload, _ID_PATHS)) or None,
    )


register(SOURCE, detect, to_inference, priority=30)
