"""Shared machinery for the trace adapters + the adapter registry.

Why this module exists
----------------------
Every adapter in :mod:`tokentrace.ingest` hits the same four problems:

1. reach into a nested payload for a field whose exact path differs by SDK
   version (:func:`resolve_path`, :func:`first_present`, :func:`first_text`);
2. turn a logged document into a :class:`~tokentrace.core.types.Chunk` *without
   inventing* a relevance score or a gold flag (:class:`ChunkBuilder`);
3. render a chat-message list into the one prompt string the model saw
   (:func:`render_messages`);
4. be discoverable, so :func:`load_any` can auto-route an unrecognized payload.

Keeping all of that here means each adapter is only the ~60 lines that are
genuinely source-specific, and — more importantly — the "missing stays missing"
policy is enforced in ONE place instead of being re-decided five times.

Missing-value policy (read this before changing a default)
----------------------------------------------------------
**Reference answers.** Production logs have none. ``ground_truth`` is set only
from a field the source actually carries (``ground_truth`` / ``reference`` /
``expected_answer`` and friends) and is otherwise ``None``. It is never derived
from the generated answer — doing so would make ``is_correct`` trivially 1.0 and
turn every diagnosis into a rubber stamp.

**``Chunk.retriever_score``.** Declared ``float`` (default ``0.0``). No *rule* in
the signal pipeline keys on it today, but it is not invisible either: it feeds
``TraceStore.make_key`` (cache identity), it is set by the FAISS retriever, and
``serialize.chunk_to_dict`` writes it into every cached trace file and into
``tokentrace analyze --json``, so an unlogged 0.0 is externally visible and
indistinguishable from a measured one. Therefore:

* the score is assigned ONLY when the payload logged a number;
* otherwise the dataclass default is left untouched and
  ``meta['ingest']['retriever_scores']`` records ``'absent'`` / ``'partial'``,
  so a consumer can always tell a measured 0.0 from an unlogged one;
* NaN is deliberately *not* used as the sentinel. It is now safe to *serialize*
  one — ``serialize.chunk_to_dict`` routes the score through ``serialize._fin``
  like every other float, so no bare ``NaN`` token reaches the JSON — but the
  round trip normalizes it back to 0.0, so NaN would still be a lossy sentinel
  that silently becomes "measured 0.0" on reload.

The real fix is ``retriever_score: Optional[float] = None`` in
``core.types.Chunk``; that file is outside this package's remit.

**``Chunk.gold``.** Typed ``Optional[bool]``, and its own docstring says
"None = unknown whether it bears the answer". A production log tells us nothing
about which document bore the answer, so unannotated chunks get ``gold=None``,
not ``gold=False``: ``False`` is a positive claim ("this chunk does not contain
the answer") for which there is no evidence, and it is the boolean analogue of
writing 0.0 for an unlogged score. Every consumer tests truthiness
(``if c.gold`` in ``RetrievalExtractor._gold_index`` / ``MockModel._decide``,
``if not c.gold`` in the recommender), so ``None`` and ``False`` behave
identically today — only the semantics differ. ``gold=True`` is set solely from
an explicit gold annotation in the payload and is never inferred.

**RAG vs non-RAG.** ``retrieved_context is None`` hard-masks the retrieval modes,
so the distinction matters: a key that is *absent* yields ``None`` (non-RAG),
while a key that is present but empty (``"source_documents": []``) yields ``[]``
— a RAG pipeline that retrieved nothing is a retrieval failure, not a chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from tokentrace.core.types import Chunk, Inference


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class IngestError(ValueError):
    """A payload could not be converted into an :class:`Inference`.

    Always names the source, the field that was missing, and the paths that were
    tried — a ``KeyError`` raised four frames deep tells a user nothing about
    which of their log fields to rename.
    """


class UnknownSourceError(IngestError):
    """No registered adapter recognized the payload."""


def _keys_hint(payload: Any) -> str:
    if isinstance(payload, Mapping):
        keys = sorted(str(k) for k in payload)
        shown = ", ".join(keys[:12]) + (", ..." if len(keys) > 12 else "")
        return f"top-level keys present: [{shown}]"
    return f"payload is a {type(payload).__name__}, expected a JSON object"


def missing_field(source: str, what: str, tried: Sequence[str], payload: Any) -> IngestError:
    """Build the one error message shape every adapter raises."""
    return IngestError(
        f"{source}: could not find the {what}. "
        f"Tried: {', '.join(tried)}. {_keys_hint(payload)}"
    )


#: Keys meaning "this payload carries STRUCTURED retrieval" — the retriever's own
#: chunk boundaries and scores. An adapter that can only re-parse a rendered prompt
#: must defer on these, because discarding real chunk boundaries is not a cosmetic
#: loss: leaving ``retrieved_context is None`` makes the engine hard-mask
#: RETRIEVAL_FAILURE and CONTEXT_DILUTION to P=0, so the two modes the trace could
#: exhibit become undiagnosable rather than merely unsupported.
#:
#: This must stay a superset of every document/node path the structured adapters
#: actually read (``langchain._DOC_PATHS``, ``llamaindex._NODE_PATHS``). A test
#: enforces that, because the lists drifted apart once already: ``openai_chat``
#: deferred on 3 of langchain's 6 document keys and ``otel`` deferred on none, so a
#: scored, gold-annotated document set was silently discarded in favour of prose
#: re-parsing.
STRUCTURED_RETRIEVAL_KEYS: tuple[str, ...] = (
    "source_documents", "source_nodes", "page_content",
    "context", "contexts", "documents", "docs", "nodes", "sources",
)


def defers_to_structured_retrieval(payload: Any) -> bool:
    """True if some other adapter can recover real chunk boundaries from this payload."""
    if not isinstance(payload, Mapping):
        return False
    return any(k in payload for k in STRUCTURED_RETRIEVAL_KEYS)


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
#: Sentinel distinguishing "path did not resolve" from "resolved to None".
_MISSING = object()


class _Wildcard:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "[]"


WILDCARD = _Wildcard()

_PathSeg = Union[str, int, _Wildcard]


def parse_path(path: str) -> list[_PathSeg]:
    """``"docs[].meta['a.b']"`` -> ``['docs', WILDCARD, 'meta', 'a.b']``.

    Grammar: dotted keys, ``[n]`` for a list index (negatives allowed), ``[]`` or
    ``[*]`` to map over a list, and ``['literal key']`` for keys that themselves
    contain dots — which is not academic here: OpenTelemetry attribute names are
    ``gen_ai.request.model``.

    What is a hard error, and why (a field path is *code*, and a path that
    silently resolves to the wrong field is the worst outcome available here —
    the caller gets an ``Inference`` built from a neighbouring column with no
    sign that anything went wrong):

    * an **empty segment** — ``a..b``, ``.a``, ``a.`` — used to collapse to
      ``a.b`` / ``a``, so the single most likely typo in a hand-written field map
      resolved *successfully* against the wrong path;
    * an **unterminated ``[``** or an unterminated quoted key inside one;
    * ``[--1]`` and friends, which used to reach ``int()`` and surface a bare
      ``ValueError`` from four frames down instead of an :class:`IngestError`.

    A quoted key is scanned to its matching close quote *before* the ``]`` is
    looked for, so ``attrs['a]b']`` is the key ``a]b`` rather than being
    mis-segmented into ``a`` + ``b']``; and exactly one layer of quotes is
    removed, so ``attrs["'q'"]`` is the key ``'q'`` (both layers used to be
    stripped, yielding a key nobody asked for).

    Stated limit, deliberately not fixed: there is **no escape syntax**, so a key
    containing both a single and a double quote is not expressible. Adding
    backslash escapes would complicate the grammar of every path in the package
    for a key shape that neither OTel attribute names nor JSON log columns
    produce; a caller with one can reach the value with ``resolve_path`` on the
    parent and an ordinary ``[key]`` lookup.
    """
    segments: list[_PathSeg] = []
    buf = ""
    #: True once a '.' has been consumed and a key (or a '[') must follow, so an
    #: empty segment can be told apart from the legitimate '.' in "a[0].b".
    pending = False
    i, n = 0, len(path)
    while i < n:
        ch = path[i]
        if ch == ".":
            if buf:
                segments.append(buf)
                buf = ""
            elif pending or not segments:
                raise IngestError(
                    f"empty segment in field path {path!r} (at position {i}). "
                    f"'a..b', a leading '.' and a trailing '.' are typos, not 'a.b'."
                )
            pending = True
            i += 1
        elif ch == "[":
            if buf:
                segments.append(buf)
                buf = ""
            pending = False
            segment, i = _parse_bracket(path, i)
            segments.append(segment)
        else:
            buf += ch
            pending = False
            i += 1
    if buf:
        segments.append(buf)
    elif pending:
        raise IngestError(
            f"empty segment in field path {path!r}: it ends in '.', so the last key is missing"
        )
    if not segments:
        raise IngestError(f"empty field path {path!r}")
    return segments


def _parse_bracket(path: str, start: int) -> tuple[_PathSeg, int]:
    """Parse one ``[...]`` beginning at ``start``; return ``(segment, next index)``."""
    i = start + 1
    n = len(path)
    while i < n and path[i] in " \t":
        i += 1
    if i < n and path[i] in "'\"":
        # Quoted key: find the matching close quote FIRST, so a ']' inside the key
        # is part of the key rather than the end of the bracket.
        quote = path[i]
        close_quote = path.find(quote, i + 1)
        if close_quote < 0:
            raise IngestError(f"unterminated {quote} in field path {path!r}")
        key = path[i + 1 : close_quote]
        rest = path.find("]", close_quote + 1)
        if rest < 0:
            raise IngestError(f"unterminated '[' in field path {path!r}")
        if path[close_quote + 1 : rest].strip():
            raise IngestError(
                f"trailing text after the quoted key in field path {path!r}: "
                f"{path[close_quote + 1 : rest]!r}"
            )
        return key, rest + 1

    close = path.find("]", start)
    if close < 0:
        raise IngestError(f"unterminated '[' in field path {path!r}")
    inner = path[start + 1 : close].strip()
    if inner in ("", "*"):
        return WILDCARD, close + 1
    body = inner[1:] if inner.startswith("-") else inner
    if body.isdigit():
        return int(inner), close + 1
    if inner.lstrip("+-").isdigit():
        # "--1" / "+-1": numeric-looking but not a number. Left to int() this raised a
        # bare ValueError from inside the parser instead of a named IngestError.
        raise IngestError(f"malformed list index {inner!r} in field path {path!r}")
    return inner, close + 1


def _walk(obj: Any, segments: Sequence[_PathSeg]) -> Any:
    if not segments:
        return obj
    seg, rest = segments[0], segments[1:]
    if isinstance(seg, _Wildcard):
        if not isinstance(obj, (list, tuple)):
            return _MISSING
        out = []
        for item in obj:
            value = _walk(item, rest)
            if value is not _MISSING:
                out.append(value)
        return out
    if isinstance(seg, int):
        if not isinstance(obj, (list, tuple)) or not -len(obj) <= seg < len(obj):
            return _MISSING
        return _walk(obj[seg], rest)
    if isinstance(obj, Mapping):
        if seg not in obj:
            return _MISSING
        return _walk(obj[seg], rest)
    return _MISSING


def resolve_path(obj: Any, path: str, default: Any = None) -> Any:
    """Resolve a dotted/bracket path, returning ``default`` if it does not exist."""
    value = _walk(obj, parse_path(path))
    return default if value is _MISSING else value


def first_present(obj: Any, paths: Sequence[str], default: Any = None) -> Any:
    """First path that resolves to a non-``None`` value.

    ``[]``, ``""`` and ``0`` count as *present* — the caller decides whether an
    empty value is meaningful (for ``source_documents`` it very much is).
    """
    for path in paths:
        value = _walk(obj, parse_path(path))
        if value is not _MISSING and value is not None:
            return value
    return default


def first_text(obj: Any, paths: Sequence[str]) -> Optional[str]:
    """First path that yields a non-empty string (after text flattening)."""
    for path in paths:
        value = _walk(obj, parse_path(path))
        if value is _MISSING:
            continue
        text = content_text(value).strip()
        if text:
            return text
    return None


# --------------------------------------------------------------------------- #
# Scalar coercion
# --------------------------------------------------------------------------- #
def as_float(value: Any) -> Optional[float]:
    """Numeric coercion that refuses to guess. ``None`` means "not a number"."""
    if isinstance(value, bool):
        return None  # bool is an int subclass; a flag is never a score
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


#: Keys that unambiguously annotate "this document bears the answer". Deliberately
#: narrow: ``relevance``/``label`` are too overloaded to read as a gold flag, and a
#: false positive here fabricates the single most decisive retrieval feature
#: (``gold_recall_in_context``, the retrieval-failure vs dilution discriminator).
GOLD_KEYS = ("gold", "is_gold", "is_ground_truth", "ground_truth_relevant")


def gold_flag(metadata: Any) -> Optional[bool]:
    """Explicit gold annotation, or ``None`` when the source does not say.

    An explicit ``false`` is kept: a labelled negative is real information. Only
    *absence* — or a value that is not unambiguously a flag — becomes ``None``.

    A number is read as a flag only when it is exactly 0 or 1. Coercing every
    nonzero number with ``bool()`` re-opened the hole :data:`GOLD_KEYS` is narrow
    to avoid: ``{"gold": 0.3}`` is *graded relevance*, and reading it as
    ``gold=True`` fabricates the single most decisive retrieval feature
    (``gold_recall_in_context``, the retrieval-failure vs dilution discriminator)
    AND reports it as provenance ``'annotated'``, i.e. as something the source
    genuinely said. A graded scale is not a gold flag, so it stays unknown; the
    key is then skipped rather than short-circuiting, so an unambiguous
    ``is_gold`` alongside a graded ``gold`` is still honoured.
    """
    if not isinstance(metadata, Mapping):
        return None
    for key in GOLD_KEYS:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                if value == 1:
                    return True
                if value == 0:
                    return False
                continue  # graded relevance, not a flag
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "yes", "1"):
                    return True
                if low in ("false", "no", "0"):
                    return False
    return None


#: Where a reference answer hides when a source has one at all. ``answer`` is
#: pointedly absent: in RAGAS-style payloads ``answer`` is the GENERATED answer,
#: and reading it as the reference would score every trace perfectly correct.
REFERENCE_KEYS = (
    "ground_truth", "ground_truths", "reference", "references",
    "reference_answer", "expected_answer", "expected", "gold_answer", "gold_answers",
)


def normalize_reference(value: Any) -> Optional[list[str]]:
    """Coerce a reference answer (or a set of them) to the ``list[str]`` contract.

    ``Inference.has_ground_truth`` already treats ``[]``/``[""]`` as absent; we
    normalize them to ``None`` here so the two representations never diverge.
    """
    if value is None or isinstance(value, bool):
        # bool is an int subclass, so `{"expected": true}` would otherwise become the
        # reference answer "True". Keys like `expected` and `reference` hold flags in
        # ordinary eval logs, and a fabricated reference is not a cosmetic error: it
        # flips `reference_available`, DROPS the |noref calibration signature, and
        # scores `is_correct` / `gold_recall_in_context` against the literal "True".
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if isinstance(value, (list, tuple)):
        out = [str(v).strip() for v in value
               if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        out = [v for v in out if v]
        return out or None
    if isinstance(value, (int, float)):
        return [str(value)]
    return None


def find_reference(payload: Any, *extra_paths: str) -> Optional[list[str]]:
    """Reference answer if the source logged one, else ``None``. Never invents."""
    paths = list(extra_paths) + [f"['{k}']" for k in REFERENCE_KEYS]
    for path in paths:
        value = _walk(payload, parse_path(path))
        if value is _MISSING:
            continue
        refs = normalize_reference(value)
        if refs:
            return refs
    return None


# --------------------------------------------------------------------------- #
# Message / content flattening
# --------------------------------------------------------------------------- #
def content_text(content: Any) -> str:
    """Flatten a message ``content`` of any shape into plain text.

    Handles the four shapes in the wild: a bare string; OpenAI/Anthropic content
    blocks ``[{"type": "text", "text": ...}]``; the OTel GenAI part shape
    ``[{"type": "text", "content": ...}]``; and already-parsed nesting of those.
    Non-text parts (images, tool calls, audio) flatten to nothing — TokenTrace's
    signals are text-only, and silently stringifying an image URL into the prompt
    would corrupt every token-level feature.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bool):
        return ""
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, Mapping):
        for key in ("text", "content", "value", "parts"):
            if key in content:
                text = content_text(content[key])
                if text:
                    return text
        return ""
    if isinstance(content, (list, tuple)):
        parts = [content_text(c) for c in content]
        return "\n".join(p for p in parts if p)
    return ""


def message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        for key in ("role", "type", "author"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return "user"


def message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, Mapping):
        return ""
    for key in ("content", "parts", "text", "message", "value"):
        if key in message:
            text = content_text(message[key])
            if text:
                return text
    return ""


def render_messages(messages: Sequence[Any]) -> str:
    """Render a chat transcript into the single ``prompt`` string.

    This is a *reconstruction*, not the byte-exact input: the provider's chat
    template (``<|im_start|>`` &c.) is not in the log, so we emit a canonical
    ``role: content`` transcript instead. Every prompt-side signal in this repo is
    bag-of-words over the text (``signals.scorers.norm``), so template tokens do
    not change any feature — but a backend that re-tokenizes ``prompt`` will not
    reproduce the original token count. ``meta['ingest']['prompt_origin']`` records
    whether the prompt came from the log verbatim or was rebuilt here.
    """
    lines: list[str] = []
    for message in messages:
        text = message_text(message).strip()
        if not text:
            continue
        lines.append(f"{message_role(message)}: {text}")
    return "\n\n".join(lines)


def compose_prompt(question: Optional[str], chunk_texts: Sequence[str]) -> str:
    """Rebuild the prompt for sources that log the *pieces* but not the render.

    Mirrors the layout the rest of the repo uses when it materializes a prompt
    (``recommend.recommender._rebuild``, ``cli._canonical``): question first, then
    the context. Wrong in detail versus whatever template the app actually used;
    right in the only respect the signals care about (which text was visible).
    """
    body = "\n\n".join(t for t in chunk_texts if t)
    if question and body:
        return f"{question}\n\n{body}"
    return question or body


# --------------------------------------------------------------------------- #
# Chunk construction
# --------------------------------------------------------------------------- #
def make_chunk(
    text: Any,
    *,
    score: Any = None,
    source_id: Any = "",
    gold: Optional[bool] = None,
    char_span: Optional[tuple[int, int]] = None,
) -> Chunk:
    """Build a :class:`Chunk`, leaving unlogged fields at their defaults.

    ``score`` is applied only if it coerces to a number; a non-numeric or absent
    score leaves ``retriever_score`` at the dataclass default rather than being
    written as a measured 0.0 (see the module docstring).
    """
    chunk = Chunk(
        text=content_text(text).strip(),
        source_id=str(source_id) if source_id not in (None, "") else "",
        gold=gold,
        char_span=char_span,
    )
    value = as_float(score)
    if value is not None:
        chunk.retriever_score = value
    return chunk


class ChunkBuilder:
    """Accumulates chunks *and remembers what the source actually logged*.

    Needed because the information is destroyed by construction: once a chunk
    carries ``retriever_score == 0.0`` there is no way to tell an unlogged score
    from a measured one. The builder therefore counts as it goes and exposes the
    provenance strings that land in ``meta['ingest']``.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._scored = 0
        self._gold_annotated = 0

    def add(
        self,
        text: Any,
        *,
        score: Any = None,
        source_id: Any = "",
        gold: Optional[bool] = None,
        char_span: Optional[tuple[int, int]] = None,
    ) -> Optional[Chunk]:
        """Append a chunk. Empty text is skipped (an empty chunk is not evidence)."""
        chunk = make_chunk(text, score=score, source_id=source_id, gold=gold, char_span=char_span)
        if not chunk.text:
            return None
        if as_float(score) is not None:
            self._scored += 1
        if gold is not None:
            self._gold_annotated += 1
        self.chunks.append(chunk)
        return chunk

    def extend(self, chunks: Sequence[Chunk]) -> None:
        """Adopt pre-built chunks. They count as unscored and unannotated, which is
        correct for every path that produces chunks by parsing prose rather than by
        reading structured retrieval fields."""
        self.chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def score_provenance(self) -> str:
        if not self.chunks:
            return "n/a"
        if self._scored == 0:
            return "absent"
        return "present" if self._scored == len(self.chunks) else "partial"

    @property
    def gold_provenance(self) -> str:
        if not self.chunks:
            return "n/a"
        return "annotated" if self._gold_annotated else "unknown"


# --------------------------------------------------------------------------- #
# Provenance metadata
# --------------------------------------------------------------------------- #
def ingest_meta(
    source: str,
    *,
    model_name: Optional[str] = None,
    context_origin: str = "none",
    prompt_origin: str = "rebuilt",
    question_origin: str = "absent",
    builder: Optional[ChunkBuilder] = None,
    reference: Optional[list[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build ``Inference.meta`` with an audit trail of what was real vs derived.

    Downstream consumers (and reviewers) need to distinguish "the retriever scored
    this 0.0" from "we never saw a score", and "this prompt is what the model saw"
    from "we glued the question and the chunks together". Recording it as data
    rather than as a docstring is what makes those questions answerable at
    inspection time.
    """
    meta: dict[str, Any] = {
        "source": source,
        "ingest": {
            "adapter": source,
            "context_origin": context_origin,
            "prompt_origin": prompt_origin,
            "question_origin": question_origin,
            "retriever_scores": builder.score_provenance if builder else "n/a",
            "gold_flags": builder.gold_provenance if builder else "n/a",
            "reference": "logged" if reference else "absent",
        },
    }
    if model_name:
        meta["model_name"] = model_name
    if extra:
        meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Adapter:
    """One source's ``detect`` / ``to_inference`` pair.

    ``priority`` orders detection: lower runs first, so a specific signature (an
    embedded field map, a ``gen_ai.*`` attribute) is checked before a looser one
    (a bare ``messages`` array).
    """

    name: str
    detect: Callable[[Any], bool]
    to_inference: Callable[[Any], Inference]
    priority: int = 50


_REGISTRY: dict[str, Adapter] = {}
_BUILTINS_LOADED = False
#: Import order is irrelevant (priority decides detection order); this is only the
#: set of modules that must be imported for the built-ins to self-register.
_BUILTIN_MODULES = ("generic", "otel", "llamaindex", "langchain", "openai_chat")


def register(
    name: str,
    detect: Callable[[Any], bool],
    to_inference: Callable[[Any], Inference],
    priority: int = 50,
) -> Adapter:
    """Register an adapter (built-in or third-party). Re-registering replaces."""
    adapter = Adapter(name=name, detect=detect, to_inference=to_inference, priority=priority)
    _REGISTRY[name] = adapter
    return adapter


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True  # set first: the imports below re-enter via register()
    import importlib

    for module in _BUILTIN_MODULES:
        importlib.import_module(f"{__package__}.{module}")


def adapters() -> list[Adapter]:
    """All registered adapters in detection order."""
    _ensure_builtins()
    return sorted(_REGISTRY.values(), key=lambda a: (a.priority, a.name))


def detect_all(payload: Any) -> list[str]:
    """Names of every adapter that claims ``payload``.

    More than one is a bug in a detector, so this is exposed for tests rather
    than hidden inside :func:`load_any`.
    """
    return [a.name for a in adapters() if _safe_detect(a, payload)]


def _safe_detect(adapter: Adapter, payload: Any) -> bool:
    # A detector must never take down routing for the other adapters; a payload
    # shaped unlike anything it expects is a "no", not a crash.
    try:
        return bool(adapter.detect(payload))
    except Exception:  # noqa: BLE001 - deliberate: detection is best-effort
        return False


def detect_source(payload: Any) -> Optional[str]:
    """Name of the highest-priority adapter that claims ``payload``, else ``None``."""
    for adapter in adapters():
        if _safe_detect(adapter, payload):
            return adapter.name
    return None


def load_any(
    payload: Any,
    *,
    field_map: Optional[Mapping[str, str]] = None,
    source: Optional[str] = None,
) -> Inference:
    """Convert one logged trace payload into an :class:`Inference`.

    ``field_map`` forces the :mod:`~tokentrace.ingest.generic` adapter (an explicit
    mapping always beats sniffing); ``source`` forces a named adapter.
    """
    _ensure_builtins()
    if field_map is not None:
        from tokentrace.ingest.generic import from_field_map

        return from_field_map(payload, field_map)
    if source is not None:
        if source not in _REGISTRY:
            raise IngestError(
                f"unknown ingest source {source!r}; registered: "
                f"{', '.join(sorted(_REGISTRY))}"
            )
        return _REGISTRY[source].to_inference(payload)
    name = detect_source(payload)
    if name is None:
        raise UnknownSourceError(
            "no ingest adapter recognized this payload. "
            f"{_keys_hint(payload)}. Registered adapters: {', '.join(a.name for a in adapters())}. "
            "Pass field_map={...} to map the fields explicitly (see tokentrace.ingest.generic)."
        )
    return _REGISTRY[name].to_inference(payload)


# --------------------------------------------------------------------------- #
# File loading
# --------------------------------------------------------------------------- #
def load_traces(
    path: Union[str, Path],
    *,
    field_map: Optional[Mapping[str, str]] = None,
    source: Optional[str] = None,
) -> Iterator[Inference]:
    """Stream a ``.json`` array (or single object) / ``.jsonl`` file of traces.

    Format is decided by content, not by extension: a whole-file JSON parse is
    tried first and line mode is the fallback, because ``.json`` files holding
    line-delimited records are common enough in log exports to be worth handling.

    Genuinely incremental for ``.jsonl``/``.ndjson``: the file is iterated a line
    at a time and each record is yielded before the next is read, so peak memory
    is one line rather than the whole export. (It used to promise this while
    doing ``read_text()`` + ``splitlines()``, which allocated ~2x the file before
    the first yield.) A ``.json`` array cannot be streamed — the stdlib has no
    incremental array parser — so it is materialized first; prefer ``.jsonl`` for
    large exports, and note that a ``.json`` file that turns out to be
    line-delimited has already been read whole by the time line mode takes over.

    Read as ``utf-8-sig`` so a UTF-8 BOM — which Windows and several log
    exporters prepend, and which broke both formats — is consumed rather than
    parsed as content.

    Every failure is an :class:`IngestError` naming the file and the record: a
    path that is a directory, bytes that are not UTF-8, a syntactically bad line,
    and — the case this is actually about, because it is the batch path — an
    *adapter* error partway through a file, which used to surface as a bare
    ``openai_chat: could not find ...`` with no clue which of 40,000 records it
    came from.

    Traces whose payload carries no id get ``"<filename>#<locator>"``, where
    ``<locator>`` is exactly the number the error messages for that file use: the
    1-based **line number** in line mode (``bl.jsonl#5`` <-> ``bl.jsonl:5``) and
    the 0-based **array index** in array mode (``bl.json#1`` <-> ``bl.json[1]``).
    They were previously always 0-based while errors were always 1-based, so an
    id and the error for the same record named different numbers.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"trace file not found: {path}")
    if path.is_dir():
        raise IngestError(
            f"trace path is a directory, not a trace file: {path}. "
            f"Pass a .json / .jsonl file (or iterate the directory yourself)."
        )

    #: (is_array_index, number, payload). The flag only selects how the number is
    #: rendered — `file[3]` for an array index, `file:3` for a line — so an id and
    #: an error for the same record always carry the same number.
    records: Iterator[tuple[bool, int, Any]]
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        records = _iter_lines(_stream_lines(path), path)
    else:
        text = _read_text(path)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # A .json file that is really line-delimited: try that before giving up.
            if "\n" in text.strip():
                records = _iter_lines(text.splitlines(), path)
            else:
                raise IngestError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
        else:
            payloads = parsed if isinstance(parsed, list) else [parsed]
            records = ((True, i, p) for i, p in enumerate(payloads))

    for is_index, number, payload in records:
        where = f"{path}[{number}]" if is_index else f"{path}:{number}"
        try:
            inference = load_any(payload, field_map=field_map, source=source)
        except IngestError as exc:
            # The syntax-error path has always reported path:lineno; an adapter error
            # reported neither, which on the primary batch path means a user is told a
            # field is missing but not from which of their records.
            raise IngestError(f"{where}: {exc}") from exc
        if not inference.id:
            inference.id = f"{path.name}#{number}"
        yield inference


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestError(_not_utf8(path, exc)) from exc
    except OSError as exc:  # unreadable, a directory raced in, a broken symlink...
        raise IngestError(f"could not read trace file {path}: {exc}") from exc


def _not_utf8(path: Path, exc: UnicodeDecodeError) -> str:
    # No byte offset: for the streaming reader it would be an offset into whichever
    # buffer the decoder was filling, not into the file, and a wrong number is worse
    # than none. The offending byte and the decoder's own reason are unambiguous.
    return (
        f"{path}: not valid UTF-8 (undecodable byte "
        f"{exc.object[exc.start:exc.start + 1]!r}: {exc.reason}). Re-encode the export "
        f"as UTF-8; a BOM is fine, other encodings are not decoded by guesswork."
    )


def _stream_lines(path: Path) -> Iterator[str]:
    """Yield lines without materializing the file (the ``.jsonl`` promise)."""
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise IngestError(f"could not read trace file {path}: {exc}") from exc
    with handle:
        try:
            yield from handle
        except UnicodeDecodeError as exc:
            raise IngestError(_not_utf8(path, exc)) from exc


def _iter_lines(lines: Iterable[str], path: Path) -> Iterator[tuple[bool, int, Any]]:
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield False, lineno, json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from exc
