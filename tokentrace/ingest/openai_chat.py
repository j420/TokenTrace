"""Adapter for OpenAI / Anthropic-style chat-completion request+response logs.

This is the lowest-common-denominator front door: whatever framework an app uses,
*something* eventually logs a messages array and a completion string. It is also
the hardest adapter, because a chat log is a rendered prompt — the retrieval
structure that TokenTrace's retrieval family needs has already been flattened into
prose by the time it reaches the log.

Accepted shapes (all auto-detected)::

    {"model": ..., "messages": [...], "choices": [{"message": {...}}]}
    {"request": {"messages": [...]}, "response": {"choices": [...]}}
    {"model": ..., "system": "...", "messages": [...],
     "content": [{"type": "text", "text": "..."}]}          # Anthropic
    {"messages": [..., {"role": "assistant", "content": "..."}]}  # answer appended

``prompt`` is the rendered transcript of the *input* messages;
``question`` is the last user turn with any detected context block removed.


The RAG-context heuristic and what it cannot do
-----------------------------------------------
:func:`extract_injected_context` looks for context that was *delimited* when it
was injected, in this order:

1. ``<context> ... </context>`` (or ``<documents>``/``<passages>``) wrappers;
2. repeated ``<document id="...">...</document>`` tags;
3. a ``Context:`` / ``Retrieved documents:`` / ``Sources:`` header line, with the
   block running to the next section header (``Question:``, ``Instructions:``, ...)
   or, when there is none, to the end of the message minus a trailing instruction
   *about* the block ("Answer using only the context above.", see
   :data:`_TRAILING_INSTRUCTION`) — which otherwise ended up inside the last chunk
   and was scored as retrieved evidence.

Inside a block, entries are split on explicit markers (``[1]``, ``1.``,
``Document 3:``), else on rule separators (``---``), else on blank lines.
:func:`extract_all_injected_context` applies that to **every** input message and
concatenates the results, because a RAG prompt routinely splits the block across
turns (retrieved facts in the system message, more retrieved facts plus the
question in the user message).

Limits, stated plainly — every one of these is a *silent* wrong answer, so the
strategy that was actually used is recorded in
``meta['ingest']['context_origin']`` for every trace:

* **Undelimited context is not extracted.** Facts pasted into a prompt as plain
  prose are indistinguishable from instructions, so the trace comes back with
  ``retrieved_context=None`` — a non-RAG inference — and the retrieval modes are
  hard-masked. That under-reports RAG; it does not fabricate retrieval evidence.
* **Chunk boundaries are guessed.** Blank-line splitting turns one multi-paragraph
  document into several chunks. ``n_chunks``, ``context_length_tokens`` and the
  dilution geometry (``gold_position_frac``) are therefore approximations, and
  ``dil.buried`` in particular keys on exactly those. Both known ways for that
  guess to go badly wrong are bounded rather than merely documented: bare
  markdown bullets are **not** treated as document boundaries (see
  :data:`_ENTRY_MARKER`) and no single block may exceed
  :data:`_MAX_BLOCK_CHUNKS` chunks.
* **No scores, ever.** A rendered prompt cannot carry retriever scores, so
  ``retriever_scores`` is always ``absent`` for this adapter.
* **No gold flags, ever.** Nothing in a chat log says which document bore the
  answer, so ``gold`` stays ``None`` and ``gold_recall_in_context`` is only
  available if the payload separately carries a reference answer (it usually
  does not).
* **False positives on quoted user content.** A summarization prompt whose user
  turn is ``Documents: <the user's own text>`` is read as RAG.
* **The prompt is a reconstruction.** The provider's chat template is not logged.
* **Only the first completion.** ``choices[0]`` is the answer; a payload logging
  ``n>1`` samples has the rest dropped rather than concatenated into a string no
  model produced.

If any of this matters for your pipeline, log the retrieval structure and use the
:mod:`~tokentrace.ingest.langchain` / :mod:`~tokentrace.ingest.llamaindex` /
:mod:`~tokentrace.ingest.generic` adapters instead — this one is the fallback for
when all you have is the prompt.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from tokentrace.core.types import Chunk, Inference
from tokentrace.ingest.base import (
    ChunkBuilder,
    content_text,
    defers_to_structured_retrieval,
    find_reference,
    first_present,
    first_text,
    ingest_meta,
    message_role,
    message_text,
    missing_field,
    register,
    render_messages,
)

SOURCE = "openai_chat"

# --------------------------------------------------------------------------- #
# Context-block detection
# --------------------------------------------------------------------------- #
_CONTEXT_TAG = re.compile(
    r"<\s*(context|contexts|retrieved_context|retrieved-context|documents|sources|passages)\s*>"
    r"(?P<body>.*?)"
    r"<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

_DOC_TAG = re.compile(
    r"<\s*(document|doc|passage|chunk|source)(?P<attrs>\s[^>]*?)?\s*>"
    r"(?P<body>.*?)"
    r"<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

_TAG_ATTR = re.compile(r"""([\w:-]+)\s*=\s*["']([^"']*)["']""")

# A header must end in a colon: bare "Documents" starting a sentence is far too
# common to treat as a section marker.
_HEADER = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?"
    r"(?P<label>context|contexts|retrieved context|retrieved documents|retrieved passages|"
    r"relevant (?:context|documents|passages|excerpts)|documents|sources|passages|"
    r"reference material|knowledge base|background)"
    r"(?:\*\*)?[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)

_NEXT_SECTION = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?"
    r"(?:question|query|user question|user query|instructions?|task|answer|rules|"
    r"guidelines|constraints|output format|format|examples?|notes?)"
    r"(?:\*\*)?[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)

_SEPARATOR = re.compile(r"^[ \t]*(?:-{3,}|={3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)

#: What counts as "a new retrieved document starts here" inside a context block.
#:
#: Bare markdown bullets (``- ``, ``* ``, ``• ``) are deliberately NOT in this set.
#: Unlike ``[1]`` / ``(1)`` / ``1.`` / ``Document 3:`` a bullet carries no document
#: *identity*, so it is indistinguishable from ordinary list formatting inside a
#: single retrieved document — and it was measurably not harmless: one document
#: containing a five-item bulleted list became six chunks with
#: ``gold_position_frac=0.2`` while the identical text written as prose stayed one
#: chunk, and ``rules.dil.buried`` keys on exactly ``n_chunks`` +
#: ``gold_position_frac``. That made the *formatting* of a log change the dilution
#: diagnosis. Merging a bulleted list back into one chunk loses boundary detail but
#: loses no text and cannot manufacture that geometry, which is the only error
#: direction that does not make the engine confidently wrong. Pipelines that really
#: do delimit their documents still split on tags, numbers and ``---`` rules.
_ENTRY_MARKER = re.compile(
    r"^[ \t]*(?:"
    r"\[(?P<bracket>[^\]\n]{1,60})\]"                                   # [1]  [doc-3]
    r"|\((?P<paren>\d{1,3})\)"                                          # (1)
    r"|(?P<num>\d{1,3})[.)]"                                            # 1.  1)
    r"|(?:document|doc|source|passage|chunk|excerpt)[ \t]+(?P<named>[^\s:.\n]{1,40})[ \t]*[:.]"
    r")[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)

#: Upper bound on the chunks ONE block may be split into. A 20,000-line numbered
#: list produced 20,000 ``Chunk`` objects — an O(n) blow-up through every retrieval
#: signal and a ``dil.buried`` geometry computed over noise. Over the cap the block
#: degrades to a SINGLE chunk rather than being truncated: coarser boundaries, but
#: no text dropped, because dropping retrieved text is what fabricates a retrieval
#: failure.
#:
#: Deliberately far above any plausible top-k. The cap exists to bound a log line
#: that is really a data dump, NOT to second-guess a large retrieval set: collapsing
#: a genuine 100-document context would drive ``n_chunks`` from 100 to 1 and switch
#: ``dil.buried``'s ``up(n_chunks, 4, 10)`` term from saturated to zero — suppressing
#: the dilution geometry in exactly the case most likely to be dilution.
_MAX_BLOCK_CHUNKS = 512

#: A trailing line that instructs the model *about* the block it follows
#: ("Answer using only the context above."). Matched only against the LAST line of
#: a block. Both halves are required — an imperative AND a reference to the context
#: itself — because the cost of a false positive here is deleting real retrieved
#: text, which is strictly worse than leaving one instruction line inside the final
#: chunk.
_TRAILING_INSTRUCTION = re.compile(
    r"^[ \t]*(?:answer|respond|reply|cite|quote|summari[sz]e|use|using|based|"
    r"refer|do not|don't|only use|only answer|if the)\b"
    r"[^\n]{0,200}?"
    r"\b(?:context|contexts|document|documents|passage|passages|source|sources|"
    r"excerpt|excerpts|above|below)\b"
    r"[^\n]{0,200}$",
    re.IGNORECASE,
)


def _marker_id(match: "re.Match[str]") -> str:
    for group in ("bracket", "paren", "num", "named"):
        value = match.group(group)
        if value:
            return value.strip()
    return ""


def _block_entries(body: str, prefix: str) -> tuple[list[tuple[str, str]], str]:
    """Candidate ``(text, source_id)`` entries for one block + the strategy used."""
    doc_tags = list(_DOC_TAG.finditer(body))
    if doc_tags:
        entries = []
        for i, match in enumerate(doc_tags):
            attrs = dict(_TAG_ATTR.findall(match.group("attrs") or ""))
            source_id = attrs.get("id") or attrs.get("source") or attrs.get("name") or f"{prefix}{i}"
            entries.append((match.group("body"), source_id))
        return entries, "document_tags"

    markers = list(_ENTRY_MARKER.finditer(body))
    if markers:
        bounds = [m.start() for m in markers] + [len(body)]
        # Text BEFORE the first marker is retrieved context too, and dropping it is
        # not a harmless omission: the layout "Context:\n<lead-in sentence>\n[1] fact"
        # is ordinary, and if the answer lives in that lead-in the engine sees a
        # confident gold_recall_in_context = 0.0 — a measured zero, not a missing
        # value — and reports a retrieval failure for a trace where retrieval
        # actually worked. Fabricating evidence is worse than losing it.
        entries = [(body[: markers[0].start()], f"{prefix}pre")]
        for i, match in enumerate(markers):
            entries.append((body[match.end() : bounds[i + 1]], _marker_id(match) or f"{prefix}{i}"))
        return entries, "entry_markers"

    if _SEPARATOR.search(body):
        return ([(part, f"{prefix}{i}") for i, part in enumerate(_SEPARATOR.split(body))],
                "rule_separators")

    paragraphs = [p for p in re.split(r"\n[ \t]*\n", body) if p.strip()]
    return ([(part, f"{prefix}{i}") for i, part in enumerate(paragraphs)],
            "blank_line_split" if len(paragraphs) > 1 else "whole_block")


def _split_block(body: str, builder: ChunkBuilder, prefix: str) -> str:
    """Split one context block into chunks. Returns the strategy used."""
    entries, strategy = _block_entries(body, prefix)
    if len(entries) > _MAX_BLOCK_CHUNKS:
        entries, strategy = [(body, f"{prefix}0")], f"{strategy}_capped"
    for text, source_id in entries:
        builder.add(text, source_id=source_id)
    return strategy


def _split_trailing_instruction(body: str) -> tuple[str, str]:
    """Split a trailing "Answer using only the context above." off a block body.

    Returns ``(body, trailing)``; ``trailing`` is ``""`` when nothing matched. The
    trailing line is handed back to the *residual* (and so to ``question`` and the
    prompt), never discarded — it is a real part of the message, just not a
    retrieved document.

    A single-line block is left alone, and a block that is *nothing but* the
    instruction ends up with an empty body — which is not a hole: ``_split_block``
    then produces no chunks, :func:`extract_injected_context` falls through to
    ``origin="none"``, and the whole message stays in the residual. A "Context:"
    header followed only by an instruction genuinely carries no documents, so
    reporting the trace as non-RAG is the correct answer rather than a lost one.
    """
    stripped = body.rstrip()
    cut = stripped.rfind("\n")
    if cut < 0:                       # a one-line block has no "trailing" line
        return body, ""
    last = stripped[cut + 1 :]
    if _ENTRY_MARKER.match(last):     # it is an entry of the block, not a note about it
        return body, ""
    if not _TRAILING_INSTRUCTION.match(last):
        return body, ""
    return stripped[:cut], last.strip()


def extract_injected_context(text: str, prefix: str = "ctx") -> tuple[list[Chunk], str, str]:
    """Pull a delimited RAG context block out of one message.

    Returns ``(chunks, remaining_text, origin)``. ``origin`` is ``"none"`` when
    nothing was detected, otherwise ``"<locator>/<split strategy>"`` — recorded in
    ``meta`` so the guess is auditable rather than invisible.
    """
    builder = ChunkBuilder()

    tag = _CONTEXT_TAG.search(text)
    if tag:
        strategy = _split_block(tag.group("body"), builder, prefix)
        if builder.chunks:
            residual = (text[: tag.start()] + "\n" + text[tag.end() :]).strip()
            return builder.chunks, residual, f"context_tag/{strategy}"

    doc_tags = list(_DOC_TAG.finditer(text))
    if doc_tags:
        # Via _split_block so the chunk cap applies here too. It is the only splitting
        # path that used to bypass it, and an unbounded path next to three bounded
        # ones is the kind of inconsistency that gets found the hard way.
        span = text[doc_tags[0].start() : doc_tags[-1].end()]
        strategy = _split_block(span, builder, prefix)
        if builder.chunks:
            residual = (text[: doc_tags[0].start()] + "\n" + text[doc_tags[-1].end() :]).strip()
            return builder.chunks, residual, f"document_tags/{strategy}"

    header = _HEADER.search(text)
    if header:
        start = header.end()
        following = _NEXT_SECTION.search(text, start)
        end = following.start() if following else len(text)
        # A header block with no following section header runs to end of message, so
        # "Answer using only the context above." was landing inside the last chunk.
        body, trailing = _split_trailing_instruction(text[start:end])
        strategy = _split_block(body, builder, prefix)
        if builder.chunks:
            residual = "\n".join(
                part for part in (text[: header.start()], trailing, text[end:]) if part.strip()
            ).strip()
            return builder.chunks, residual, f"header/{strategy}"

    return [], text, "none"


def extract_all_injected_context(texts: Sequence[str]) -> tuple[list[Chunk], list[str], str]:
    """Apply :func:`extract_injected_context` to EVERY message, not just the first.

    Returns ``(chunks, residual texts, origin)``.

    Stopping at the first message carrying a block was a silent data-destroying
    bug, not an optimization: for the ordinary layout "retrieved facts in the
    system turn, more retrieved facts plus the question in the user turn" it kept
    the system turn's chunks, **discarded the user turn's entirely** (including the
    answer-bearing one), and then left that undetected block sitting in
    ``question`` as raw context prose. The engine therefore saw a measured
    ``gold_recall_in_context`` of 0.0 over a context that did contain the answer,
    plus a question polluted with a context blob — a confident retrieval failure
    for a trace where retrieval worked.
    """
    chunks: list[Chunk] = []
    residuals = list(texts)
    origins: list[str] = []
    for i, text in enumerate(texts):
        found, residual, how = extract_injected_context(text, prefix=f"m{i}c")
        if not found:
            continue
        chunks.extend(found)
        residuals[i] = residual
        origins.append(f"message[{i}]:{how}")
    return chunks, residuals, "+".join(origins) or "none"


# --------------------------------------------------------------------------- #
# Payload navigation
# --------------------------------------------------------------------------- #
_MESSAGE_PATHS = ("messages", "request.messages", "body.messages", "input.messages")
_COMPLETION_PATHS = (
    "response.choices[0].message.content",
    "choices[0].message.content",
    "response.choices[0].text",
    "choices[0].text",
    "response.content",            # Anthropic: [{"type": "text", "text": ...}]
    "response.output_text",
    "output_text",
    "response.output[0].content",  # OpenAI Responses API
    "output[0].content",
    "completion",
    "response.completion",
    "generated_answer",
    # LAST, and deliberately so. A bare top-level `content` is the most ambiguous
    # key here — it is just as likely to be an article body or a template as a
    # completion. Ranked above the explicit keys it silently beat
    # `generated_answer`, putting the wrong string in the answer field and
    # computing every correctness and confidence signal against it.
    "content",
)
_MODEL_PATHS = ("model", "request.model", "response.model", "body.model")
_ID_PATHS = ("id", "response.id", "request_id", "trace_id", "run_id")


def _messages(payload: Any) -> Optional[list[Any]]:
    value = first_present(payload, _MESSAGE_PATHS)
    if isinstance(value, list) and all(isinstance(m, (Mapping, str)) for m in value):
        return list(value)
    return None


def _input_messages(payload: Any, messages: Sequence[Any]) -> list[Any]:
    """Messages the model actually saw, with a top-level Anthropic ``system`` folded in."""
    system = first_present(payload, ("system", "request.system", "body.system"))
    system_text = content_text(system).strip() if system is not None else ""
    if system_text:
        return [{"role": "system", "content": system_text}, *messages]
    return list(messages)


def _last_user_index(messages: Sequence[Any]) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if message_role(messages[i]) == "user":
            return i
    return None


# --------------------------------------------------------------------------- #
def detect(payload: Any) -> bool:
    """True for a chat-completion log: a messages array (or a ``choices`` block).

    Deliberately **defers**: a payload that also carries structured retrieval
    (``source_documents``/``page_content``/``source_nodes``) or GenAI span
    attributes is rejected here even though it does contain a messages array,
    because those adapters recover the retriever's real chunk boundaries and
    scores whereas this one can only re-parse the rendered prompt. Priority alone
    would pick the right adapter, but rejecting keeps ``detect`` honest — exactly
    one adapter claims any given payload.
    """
    if not isinstance(payload, Mapping):
        return False
    if any(str(k).startswith("gen_ai.") for k in payload):
        return False
    if "tokentrace_field_map" in payload:
        return False
    # Shared list rather than a hand-kept one: this deferral previously named only
    # 3 of the 6 document keys langchain reads, so `messages` + `documents` was
    # claimed here and the scored, gold-annotated documents were thrown away.
    if defers_to_structured_retrieval(payload):
        return False
    if isinstance(payload.get("attributes"), (Mapping, list)):
        return False  # an OTel-style span, not a chat log
    messages = _messages(payload)
    if messages is not None and messages:
        return True
    return isinstance(first_present(payload, ("choices", "response.choices")), list)


def to_inference(payload: Any) -> Inference:
    messages = _messages(payload)
    completion = first_text(payload, _COMPLETION_PATHS)

    if messages is None:
        if completion is None:
            raise missing_field(SOURCE, "messages array and completion",
                                _MESSAGE_PATHS + _COMPLETION_PATHS, payload)
        raise missing_field(SOURCE, "messages array", _MESSAGE_PATHS, payload)
    if not messages:
        raise missing_field(SOURCE, "non-empty messages array (the key is present but empty)",
                            _MESSAGE_PATHS, payload)

    inputs = _input_messages(payload, messages)

    # A log that appends the assistant turn to `messages` instead of recording a
    # separate response: treat the trailing assistant turn as the completion and
    # drop it from the prompt (the model never saw its own output as input).
    if completion is None and inputs and message_role(inputs[-1]) == "assistant":
        completion = message_text(inputs[-1]).strip() or None
        inputs = inputs[:-1]

    if completion is None:
        raise missing_field(SOURCE, "completion text", _COMPLETION_PATHS, payload)

    prompt = render_messages(inputs)
    if not prompt:
        raise missing_field(SOURCE, "non-empty prompt text in the messages array",
                            _MESSAGE_PATHS, payload)

    # Every input message contributes its delimited blocks (see the function's
    # docstring for why stopping at the first one destroyed retrieval evidence).
    chunks, texts, origin = extract_all_injected_context([message_text(m) for m in inputs])

    user_index = _last_user_index(inputs)
    question = texts[user_index].strip() if user_index is not None else ""
    question_origin = "last_user_message" if question else "absent"

    builder = ChunkBuilder()
    builder.extend(chunks)  # parsed from prose: never scored, never gold-annotated

    reference = find_reference(payload, "response.ground_truth", "metadata.ground_truth")
    return Inference(
        prompt=prompt,
        generated_answer=completion,
        # No block detected => genuinely non-RAG as far as the log can tell.
        retrieved_context=chunks or None,
        ground_truth=reference,
        question=question or None,
        meta=ingest_meta(
            SOURCE,
            model_name=content_text(first_present(payload, _MODEL_PATHS)) or None,
            context_origin=origin,
            prompt_origin="rebuilt",
            question_origin=question_origin,
            builder=builder,
            reference=reference,
            extra={"provider": content_text(first_present(payload, ("provider",))) or None},
        ),
        id=content_text(first_present(payload, _ID_PATHS)) or None,
    )


# Lowest priority: a bare messages array is the least specific signature there is,
# so every structured source gets first refusal.
register(SOURCE, detect, to_inference, priority=90)
