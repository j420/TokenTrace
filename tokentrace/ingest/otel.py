"""Adapter for OpenTelemetry spans using the GenAI semantic conventions.

OTel is the one place where a trace might already exist for reasons that have
nothing to do with LLM debugging — the app was instrumented for latency, and the
prompt/completion came along for free. That makes it the highest-leverage front
door and the messiest: the GenAI conventions have been through three incompatible
shapes, and the major instrumentation libraries each emit a different one. All
three are handled here.

**A. Flat attributes (original convention, still what most exporters emit)**::

    {"attributes": {"gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4o-mini",
                    "gen_ai.prompt": "<text, or a JSON messages array>",
                    "gen_ai.completion": "<text, or a JSON messages array>"}}

**B. Indexed flat attributes (OpenLLMetry / OpenInference style)**::

    "gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "...",
    "gen_ai.prompt.1.role": "user",   "gen_ai.prompt.1.content": "...",
    "gen_ai.completion.0.content": "..."

**C. Events / message form (current convention)**::

    {"attributes": {"gen_ai.input.messages": "[{\\"role\\": \\"user\\",
                                               \\"parts\\": [{\\"type\\": \\"text\\",
                                                             \\"content\\": \\"...\\"}]}]"},
     "events": [{"name": "gen_ai.user.message", "attributes": {"content": "..."}},
                {"name": "gen_ai.choice", "attributes": {"message": "{...}"}}]}

Attributes may be a plain dict or the OTLP/JSON key-value list
(``[{"key": ..., "value": {"stringValue": ...}}]``); both are normalized.

Retrieval context
-----------------
A GenAI span carries the *rendered* prompt, not the retrieval structure — the
retriever ran in a different span. So context extraction reuses the chat adapter's
delimited-block heuristic, and every caveat in
:func:`tokentrace.ingest.openai_chat.extract_injected_context` applies verbatim:
no scores, no gold flags, guessed chunk boundaries, and undelimited context is not
extracted at all (the trace comes back non-RAG rather than fabricated). A span
with no detectable block is treated as a plain chat completion, which is usually
exactly what it is.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from tokentrace.core.types import Chunk, Inference
from tokentrace.ingest.base import (
    ChunkBuilder,
    content_text,
    find_reference,
    first_present,
    ingest_meta,
    message_role,
    message_text,
    missing_field,
    register,
    render_messages,
)
from tokentrace.ingest.openai_chat import extract_injected_context

SOURCE = "otel"

GENAI_PREFIX = "gen_ai."

_PROMPT_KEYS = ("gen_ai.input.messages", "gen_ai.prompt", "llm.prompts", "llm.input_messages")
_COMPLETION_KEYS = ("gen_ai.output.messages", "gen_ai.completion", "llm.completions",
                    "llm.output_messages")
_MODEL_KEYS = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name", "llm.request.model")
_SYSTEM_KEYS = ("gen_ai.system", "gen_ai.provider.name", "llm.system")
_ID_KEYS = ("gen_ai.response.id", "span_id", "spanId", "trace_id", "traceId")

#: Event names in the current convention, mapped to the role they contribute.
_EVENT_ROLES = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}
_CHOICE_EVENT = "gen_ai.choice"


# --------------------------------------------------------------------------- #
# Attribute normalization
# --------------------------------------------------------------------------- #
def _anyvalue(value: Any) -> Any:
    """Unwrap one OTLP/JSON ``AnyValue``. Plain values pass straight through."""
    if not isinstance(value, Mapping):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:  # protobuf JSON renders int64 as a string
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        values = (value.get("arrayValue") or {}).get("values") or []
        return [_anyvalue(v) for v in values]
    if "kvlistValue" in value:
        values = (value.get("kvlistValue") or {}).get("values") or []
        return {kv.get("key"): _anyvalue(kv.get("value")) for kv in values if isinstance(kv, Mapping)}
    return value


def attributes(payload: Any) -> dict[str, Any]:
    """Flatten a span's attributes to ``{key: python value}``.

    Accepts a dict, the OTLP key-value list, or attributes hoisted to the top level
    (which is what a flattened log line looks like after a JSON exporter).
    """
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("attributes")
    if isinstance(raw, Mapping):
        out = {str(k): _anyvalue(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        out = {
            str(kv["key"]): _anyvalue(kv.get("value"))
            for kv in raw
            if isinstance(kv, Mapping) and "key" in kv
        }
    else:
        out = {}
    for key, value in payload.items():
        key = str(key)
        if key.startswith((GENAI_PREFIX, "llm.")) and key not in out:
            out[key] = _anyvalue(value)
    return out


def _events(payload: Any) -> list[Mapping[str, Any]]:
    raw = payload.get("events") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, Mapping)]


# --------------------------------------------------------------------------- #
# Message reconstruction
# --------------------------------------------------------------------------- #
def _maybe_json(value: Any) -> Any:
    """Attributes are strings on the wire; a messages array arrives JSON-encoded."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("[", "{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _as_messages(value: Any, default_role: str) -> list[dict[str, Any]]:
    """Coerce an attribute value into a message list.

    Handles: a JSON-encoded array of messages, an already-parsed array, a single
    message object, an array of bare strings (``llm.prompts``), and plain text.
    """
    value = _maybe_json(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": default_role, "content": value}] if value.strip() else []
    if isinstance(value, Mapping):
        text = message_text(value)
        return [{"role": message_role(value), "content": text}] if text else []
    if isinstance(value, (list, tuple)):
        out: list[dict[str, Any]] = []
        for item in value:
            item = _maybe_json(item)
            if isinstance(item, str):
                if item.strip():
                    out.append({"role": default_role, "content": item})
            elif isinstance(item, Mapping):
                text = message_text(item)
                if text:
                    out.append({"role": message_role(item), "content": text})
        return out
    return []


def _indexed_messages(attrs: Mapping[str, Any], prefix: str, default_role: str) -> list[dict[str, Any]]:
    """Reassemble ``<prefix>.<i>.role`` / ``<prefix>.<i>.content`` attribute pairs."""
    by_index: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix + "."):
            continue
        rest = key[len(prefix) + 1 :].split(".")
        if len(rest) != 2 or not rest[0].isdigit():
            continue
        index, field = int(rest[0]), rest[1]
        if field in ("role", "content"):
            by_index.setdefault(index, {})[field] = value
    out = []
    for index in sorted(by_index):
        entry = by_index[index]
        text = content_text(_maybe_json(entry.get("content"))).strip()
        if text:
            out.append({"role": str(entry.get("role") or default_role).lower(), "content": text})
    return out


def _messages_from_events(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(input messages, output messages)`` from GenAI span events."""
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for event in _events(payload):
        name = str(event.get("name") or "")
        attrs = event.get("attributes")
        attrs = {str(k): _anyvalue(v) for k, v in attrs.items()} if isinstance(attrs, Mapping) else (
            {str(kv["key"]): _anyvalue(kv.get("value")) for kv in attrs
             if isinstance(kv, Mapping) and "key" in kv} if isinstance(attrs, list) else {}
        )
        if name in _EVENT_ROLES:
            role = str(attrs.get("role") or _EVENT_ROLES[name]).lower()
            text = content_text(_maybe_json(attrs.get("content"))).strip()
            if text:
                inputs.append({"role": role, "content": text})
        elif name == _CHOICE_EVENT:
            message = _maybe_json(attrs.get("message"))
            text = content_text(message).strip() or content_text(
                _maybe_json(attrs.get("content"))).strip()
            if text:
                outputs.append({"role": "assistant", "content": text})
    return inputs, outputs


def _collect(attrs: Mapping[str, Any], payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Input messages, output messages, and which convention produced them."""
    event_inputs, event_outputs = _messages_from_events(payload)

    inputs: list[dict[str, Any]] = []
    form = ""
    for key in _PROMPT_KEYS:
        if key in attrs:
            inputs = _as_messages(attrs[key], default_role="user")
            if inputs:
                form = "input.messages" if key.endswith("input.messages") else "flat"
                break
    if not inputs:
        inputs = _indexed_messages(attrs, "gen_ai.prompt", "user") or \
            _indexed_messages(attrs, "llm.input_messages", "user")
        if inputs:
            form = "indexed"
    if not inputs and event_inputs:
        inputs, form = event_inputs, "events"

    outputs: list[dict[str, Any]] = []
    for key in _COMPLETION_KEYS:
        if key in attrs:
            outputs = _as_messages(attrs[key], default_role="assistant")
            if outputs:
                break
    if not outputs:
        outputs = _indexed_messages(attrs, "gen_ai.completion", "assistant") or \
            _indexed_messages(attrs, "llm.output_messages", "assistant")
    if not outputs and event_outputs:
        outputs = event_outputs
        form = form or "events"

    return inputs, outputs, form or "unknown"


# --------------------------------------------------------------------------- #
def detect(payload: Any) -> bool:
    """True for a span carrying GenAI semantic-convention attributes or events."""
    if not isinstance(payload, Mapping):
        return False
    if "tokentrace_field_map" in payload:
        return False
    if any(key.startswith(GENAI_PREFIX) for key in attributes(payload)):
        return True
    return any(
        str(e.get("name") or "").startswith(GENAI_PREFIX) for e in _events(payload)
    )


def to_inference(payload: Any) -> Inference:
    attrs = attributes(payload)
    inputs, outputs, form = _collect(attrs, payload)

    if not inputs:
        raise missing_field(
            SOURCE, "prompt / input messages",
            _PROMPT_KEYS + ("gen_ai.prompt.<i>.content", "gen_ai.user.message event"),
            attrs or payload,
        )
    if not outputs:
        raise missing_field(
            SOURCE, "completion / output messages",
            _COMPLETION_KEYS + ("gen_ai.completion.<i>.content", "gen_ai.choice event"),
            attrs or payload,
        )

    answer = "\n".join(m["content"] for m in outputs if m.get("content")).strip()
    if not answer:
        raise missing_field(SOURCE, "non-empty completion text", _COMPLETION_KEYS, attrs or payload)

    prompt = render_messages(inputs)

    texts = [m["content"] for m in inputs]
    chunks: list[Chunk] = []
    origin = "none"
    for i, text in enumerate(texts):
        found, residual, how = extract_injected_context(text, prefix=f"m{i}c")
        if found:
            chunks, origin = found, f"message[{i}]:{how}"
            texts[i] = residual
            break

    question = ""
    for i in range(len(inputs) - 1, -1, -1):
        if inputs[i]["role"] == "user":
            question = texts[i].strip()
            break

    builder = ChunkBuilder()
    builder.extend(chunks)  # parsed from a rendered prompt: no scores, no gold flags

    # A GenAI span has no place to put a reference answer, but a harness that wrote
    # one onto the span envelope should still be honoured. Never derived.
    reference = find_reference(payload)
    system = content_text(first_present(attrs, tuple(f"['{k}']" for k in _SYSTEM_KEYS)))
    span_id = content_text(first_present(payload, ("span_id", "spanId", "context.span_id",
                                                   "id", "trace_id", "traceId")))
    if not span_id:
        span_id = content_text(first_present(attrs, tuple(f"['{k}']" for k in _ID_KEYS)))

    return Inference(
        prompt=prompt,
        generated_answer=answer,
        retrieved_context=chunks or None,
        ground_truth=reference,
        question=question or None,
        meta=ingest_meta(
            SOURCE,
            model_name=content_text(first_present(attrs, tuple(f"['{k}']" for k in _MODEL_KEYS))) or None,
            context_origin=origin,
            prompt_origin="rebuilt",
            question_origin="last_user_message" if question else "absent",
            builder=builder,
            reference=reference,
            extra={
                "provider": system or None,
                "otel_form": form,
                "span_name": content_text(first_present(payload, ("name",))) or None,
            },
        ),
        id=span_id or None,
    )


# Highest specificity after the explicit field map: a `gen_ai.*` attribute is
# unambiguous, and an OTel span can otherwise look like a generic object.
register(SOURCE, detect, to_inference, priority=20)
