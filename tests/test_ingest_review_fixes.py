"""Regressions for the defects an adversarial review of `tokentrace/ingest/` found.

Each test here corresponds to a defect that shipped and that the existing suite did
NOT catch — the review proved it by mutation, breaking the implementation and
watching every test stay green. They are grouped by the failure they prevent rather
than by module, because the serious ones share a theme: an adapter that invents or
destroys retrieval evidence makes the engine *confidently wrong*, which is worse
than making it silent.
"""

from __future__ import annotations

import json
import math

import pytest

from tokentrace.api import TokenTrace
from tokentrace.ingest import IngestError, detect_all, load_any, load_traces, resolve_path
from tokentrace.ingest import langchain as lc
from tokentrace.ingest import llamaindex as li
from tokentrace.ingest import openai_chat, otel
from tokentrace.ingest.base import (
    STRUCTURED_RETRIEVAL_KEYS,
    ChunkBuilder,
    gold_flag,
    normalize_reference,
)


@pytest.fixture(scope="module")
def tt() -> TokenTrace:
    return TokenTrace.default(train=False)


# --------------------------------------------------------------------------- #
# C1: text before the first list marker was deleted from the context
_PREAMBLE_PAYLOAD = {
    "messages": [
        {"role": "system", "content": (
            "Context:\n"
            "Paris is the capital of France and the seat of government.\n"
            "[1] The Eiffel Tower was completed in 1889.\n"
            "[2] The population of the city proper is 2.1 million."
        )},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "choices": [{"message": {"role": "assistant", "content": "Paris."}}],
    "ground_truth": "Paris",
}


def test_context_before_the_first_list_marker_is_not_discarded():
    inf = load_any(_PREAMBLE_PAYLOAD)
    texts = [c.text for c in inf.retrieved_context or []]
    assert any("capital of France" in t for t in texts), texts
    # The lead-in must be its own chunk ahead of the marked entries, not merged
    # into one of them — that is what makes the assertion above discriminating.
    assert texts[0] == "Paris is the capital of France and the seat of government."
    assert texts[1] == "The Eiffel Tower was completed in 1889."


def test_discarded_preamble_does_not_fabricate_a_retrieval_failure(tt):
    """The consequence, not just the mechanism. Deleting the lead-in produced a
    *measured* gold_recall_in_context of 0.0 — not a missing value — and the engine
    reported a confident retrieval failure for a trace where retrieval worked."""
    inf = load_any(_PREAMBLE_PAYLOAD)
    assert tt.features(inf).values.get("gold_recall_in_context") == 1.0
    report = tt.analyze(inf)
    from tokentrace.core.types import FailureMode

    retrieval = next(d for d in report.diagnoses if d.mode is FailureMode.RETRIEVAL_FAILURE)
    assert retrieval.probability < 0.5, (
        f"gold was in the prompt, yet retrieval failure scored {retrieval.probability:.3f}")


# --------------------------------------------------------------------------- #
# C2: structured retrieval discarded in favour of re-parsing the rendered prompt
def test_deferral_list_covers_every_document_key_the_structured_adapters_read():
    """The two lists drifted apart once: openai_chat deferred on 3 of langchain's 6
    document keys. Anything readable by a structured adapter must appear here, or a
    prose re-parser will claim the payload and throw the real chunks away."""
    from tokentrace.ingest import langchain as lc
    from tokentrace.ingest import llamaindex as li

    read = {p for p in (*lc._DOC_PATHS, *li._NODE_PATHS) if "." not in p}
    assert read <= set(STRUCTURED_RETRIEVAL_KEYS), read - set(STRUCTURED_RETRIEVAL_KEYS)


@pytest.mark.parametrize("doc_key", ["source_documents", "documents", "context", "docs"])
def test_scored_gold_documents_survive_alongside_a_messages_array(doc_key):
    payload = {
        "messages": [{"role": "user", "content": "Where is the depot?"}],
        "result": "Antwerp.",
        doc_key: [
            {"page_content": "The depot is in Antwerp.", "metadata": {"score": 0.91, "gold": True}},
            {"page_content": "Rotterdam handles freight.", "metadata": {"score": 0.44}},
        ],
    }
    inf = load_any(payload)
    assert inf.is_rag, "retrieval structure was discarded -> retrieval modes get masked to P=0"
    assert len(inf.retrieved_context) == 2
    assert sum(1 for c in inf.retrieved_context if c.gold) == 1


def test_otel_span_uses_attached_documents_rather_than_re_parsing_the_prompt():
    """A GenAI span from an instrumented chain often carries the retriever's own
    documents. otel must NOT defer (no other adapter reads gen_ai.*) but must prefer
    those documents, which have real boundaries, scores and gold flags."""
    payload = {
        "attributes": {"gen_ai.prompt": "Where is the depot?", "gen_ai.completion": "Antwerp."},
        "source_documents": [
            {"page_content": "The depot is in Antwerp.", "metadata": {"score": 0.93, "gold": True}}],
    }
    assert detect_all(payload) == ["otel"], "a GenAI span must stay claimed by exactly one adapter"
    inf = load_any(payload)
    assert inf.is_rag and len(inf.retrieved_context) == 1
    assert inf.retrieved_context[0].gold is True


def test_a_plain_genai_span_is_still_non_rag():
    """Guard the fix above from over-reaching: no documents means no retrieval."""
    inf = load_any({"attributes": {"gen_ai.prompt": "What is 2+2?", "gen_ai.completion": "4"}})
    assert not inf.is_rag and inf.retrieved_context is None


# --------------------------------------------------------------------------- #
# M2/M3: payloads one adapter parses but no adapter claims
@pytest.mark.parametrize("payload,expected_chunks", [
    # A callback log whose answer is the trailing assistant turn, not a `result` key.
    ({"run_id": "r1",
      "messages": [{"role": "user", "content": "Where is the depot?"},
                   {"role": "assistant", "content": "Antwerp."}],
      "source_documents": [{"page_content": "The depot is in Antwerp.",
                            "metadata": {"score": 0.9, "gold": True}}]}, 1),
    # A RAGAS evaluation row: contexts are bare strings.
    ({"question": "Where is the depot?", "answer": "Antwerp.",
      "contexts": ["The depot is in Antwerp.", "Rotterdam handles freight."]}, 2),
    # Documents keyed on `text` rather than `page_content` (the builder reads it).
    ({"question": "Where is the depot?", "result": "Antwerp.",
      "source_documents": [{"text": "The depot is in Antwerp.", "metadata": {"score": 0.9}}]}, 1),
])
def test_parseable_payloads_are_routable(payload, expected_chunks):
    """detect() and to_inference() must agree. Where they disagreed, a payload that
    parses perfectly raised UnknownSourceError from load_any."""
    assert detect_all(payload), "no adapter claimed a payload that parses fine"
    inf = load_any(payload)
    assert len(inf.retrieved_context or []) == expected_chunks
    assert inf.query and inf.generated_answer


# --------------------------------------------------------------------------- #
# M1: a flag became a reference answer
@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_is_never_read_as_a_reference_answer(value):
    """`expected` and `reference` hold flags in ordinary eval logs. bool is an int
    subclass, so these became the reference strings "True"/"False" — flipping
    reference_available and dropping the |noref calibration signature."""
    assert normalize_reference(value) is None
    inf = load_any({"messages": [{"role": "user", "content": "Q?"}],
                    "choices": [{"message": {"content": "A."}}],
                    "expected": value})
    assert inf.has_ground_truth is False
    assert inf.ground_truth in (None, [])


def test_a_real_reference_still_survives():
    """Guard the fix above from over-reaching."""
    inf = load_any({"messages": [{"role": "user", "content": "Q?"}],
                    "choices": [{"message": {"content": "A."}}],
                    "expected": "Paris"})
    assert inf.has_ground_truth and inf.ground_truth == ["Paris"]


def test_a_boolean_inside_a_reference_list_is_dropped_not_stringified():
    assert normalize_reference(["Paris", True]) == ["Paris"]


# --------------------------------------------------------------------------- #
# M5: a bare `content` key outranked an explicit `generated_answer`
def test_explicit_generated_answer_beats_an_ambiguous_content_key():
    """A top-level `content` is as likely to be an article body as a completion.
    Ranked above the explicit keys it silently won, and every correctness and
    confidence signal was then computed against the wrong string."""
    inf = load_any({"messages": [{"role": "user", "content": "Summarise this."}],
                    "content": "the raw article body",
                    "generated_answer": "The article argues X."})
    assert inf.generated_answer == "The article argues X."


def test_content_is_still_used_when_it_is_the_only_completion_key():
    """Guard the reordering from removing a real fallback."""
    inf = load_any({"messages": [{"role": "user", "content": "Summarise this."}],
                    "content": "The article argues X."})
    assert inf.generated_answer == "The article argues X."


# --------------------------------------------------------------------------- #
# Round-trip / cache regressions introduced by the _fin scrub on retriever_score
def test_non_finite_retriever_score_round_trips_to_a_usable_float():
    """`chunk_to_dict` scrubs NaN to null, but `.get(k, default)` does NOT cover an
    explicit null, so the score came back as None — violating the float annotation
    and breaking any arithmetic over scores."""
    import json
    import math

    from tokentrace.core.serialize import chunk_from_dict, chunk_to_dict
    from tokentrace.core.types import Chunk

    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = json.loads(json.dumps(chunk_to_dict(Chunk("x", bad, "d1"))))
        back = chunk_from_dict(payload)
        assert isinstance(back.retriever_score, float)
        assert math.isfinite(back.retriever_score)
    assert chunk_from_dict(json.loads(json.dumps(
        chunk_to_dict(Chunk("x", 0.87, "d1"))))).retriever_score == 0.87


def test_cache_key_survives_a_save_reload_cycle(tmp_path):
    """TraceStore.make_key hashed the in-memory score while a reloaded trace carried
    the scrubbed one, so a saved trace could never be found again: a permanent miss
    plus a fresh file on every re-save, on the pre-compute-once workflow the cache
    exists for."""
    import json

    from tokentrace.cache.trace_store import TraceStore
    from tokentrace.core.serialize import chunk_from_dict, chunk_to_dict
    from tokentrace.core.types import Chunk, Inference, Tier

    store = TraceStore(tmp_path)

    def key_for(chunk):
        inf = Inference("p", "a", [chunk], ["gt"], question="q")
        return store.make_key(inf, "mock-4b", Tier.WHITE)

    for score in (float("nan"), float("inf"), 0.87):
        original = Chunk("x", score, "d1")
        reloaded = chunk_from_dict(json.loads(json.dumps(chunk_to_dict(original))))
        assert key_for(original) == key_for(reloaded), f"key unstable for score={score!r}"

    # The normalisation must not blind the key to real differences.
    assert key_for(Chunk("x", 0.87, "d1")) != key_for(Chunk("x", 0.42, "d1"))


# --------------------------------------------------------------------------- #
# D1: only the FIRST message carrying a context block was extracted, and the rest
# was left sitting in `question` as raw context prose.
# --------------------------------------------------------------------------- #
#: The ordinary two-turn RAG layout: retrieved facts in the system turn, MORE
#: retrieved facts plus the question in the user turn. The answer-bearing chunk is
#: deliberately the one in the SECOND message.
_SPLIT_CONTEXT_MESSAGES = [
    {"role": "system", "content": "Context:\n[1] Rotterdam handles freight."},
    {"role": "user", "content": (
        "Context:\n[2] Antwerp handles returns.\n\nQuestion: where do returns go?")},
]
_SPLIT_CONTEXT_PAYLOAD = {
    "messages": _SPLIT_CONTEXT_MESSAGES,
    "choices": [{"message": {"role": "assistant", "content": "Antwerp."}}],
    "ground_truth": "Antwerp",
}
_SPLIT_CONTEXT_SPAN = {
    "attributes": {
        "gen_ai.input.messages": json.dumps(_SPLIT_CONTEXT_MESSAGES),
        "gen_ai.completion": "Antwerp.",
    },
    "ground_truth": "Antwerp",
}


@pytest.mark.parametrize("payload", [_SPLIT_CONTEXT_PAYLOAD, _SPLIT_CONTEXT_SPAN],
                         ids=["openai_chat", "otel"])
def test_context_is_extracted_from_every_message_not_only_the_first(payload):
    """`break` after the first block silently DELETED the second one. The two
    adapters carried the identical loop, so both are pinned here."""
    inf = load_any(payload)
    assert [c.text for c in inf.retrieved_context or []] == [
        "Rotterdam handles freight.", "Antwerp handles returns."]


@pytest.mark.parametrize("payload", [_SPLIT_CONTEXT_PAYLOAD, _SPLIT_CONTEXT_SPAN],
                         ids=["openai_chat", "otel"])
def test_an_unextracted_context_block_does_not_pollute_the_question(payload):
    """The other half: a block that was never detected stayed in `question`, so
    every prompt-family signal scored the question as a context blob."""
    inf = load_any(payload)
    assert "Antwerp handles returns." not in inf.question
    assert "Context:" not in inf.question
    assert inf.meta["ingest"]["context_origin"] == (
        "message[0]:header/entry_markers+message[1]:header/entry_markers")


def test_dropping_the_second_block_fabricated_a_retrieval_failure(tt):
    """The consequence, not just the mechanism: with the answer-bearing chunk
    deleted the engine saw a *measured* gold_recall_in_context of 0.0 over a
    context that did contain the answer."""
    inf = load_any(_SPLIT_CONTEXT_PAYLOAD)
    assert tt.features(inf).values.get("gold_recall_in_context") == 1.0


# --------------------------------------------------------------------------- #
# D2: `question` is documented as the LAST user turn, but no fixture had two.
# --------------------------------------------------------------------------- #
_MULTI_TURN = [
    {"role": "system", "content": "You are a support assistant."},
    {"role": "user", "content": "Where is the depot?"},
    {"role": "assistant", "content": "In Antwerp."},
    {"role": "user", "content": "And what are its opening hours?"},
]
_LAST_TURN = "And what are its opening hours?"
_FIRST_TURN = "Where is the depot?"


def _multi_turn_inference(adapter: str):
    if adapter == "openai_chat":
        return load_any({"messages": _MULTI_TURN,
                         "choices": [{"message": {"content": "08:00 to 18:00."}}]})
    if adapter == "otel_messages":
        return load_any({"attributes": {"gen_ai.input.messages": json.dumps(_MULTI_TURN),
                                        "gen_ai.completion": "08:00 to 18:00."}})
    if adapter == "otel_indexed":
        attrs = {f"gen_ai.prompt.{i}.{k}": m[k] for i, m in enumerate(_MULTI_TURN)
                 for k in ("role", "content")}
        attrs["gen_ai.completion.0.content"] = "08:00 to 18:00."
        return load_any({"attributes": attrs})
    if adapter == "otel_events":
        events = [{"name": f"gen_ai.{m['role']}.message", "attributes": m} for m in _MULTI_TURN]
        events.append({"name": "gen_ai.choice", "attributes": {"content": "08:00 to 18:00."}})
        return load_any({"events": events})
    if adapter == "langchain":
        # No `question` key, so the adapter must fall back to the last user turn.
        return load_any({"messages": _MULTI_TURN, "result": "08:00 to 18:00.",
                         "source_documents": [{"page_content": "The depot opens at 08:00."}]})
    raise AssertionError(adapter)


@pytest.mark.parametrize("adapter", ["openai_chat", "otel_messages", "otel_indexed",
                                     "otel_events", "langchain"])
def test_question_is_the_last_user_turn_not_the_first(adapter):
    """Every adapter documents `question` as the LAST user turn. The reviewer
    changed each of them to return the FIRST and no test failed, because no fixture
    had more than one user turn — so the whole multi-turn contract was unpinned."""
    inf = _multi_turn_inference(adapter)
    assert inf.question == _LAST_TURN
    assert inf.question != _FIRST_TURN
    assert inf.meta["ingest"]["question_origin"] in ("last_user_message", "query_field")


@pytest.mark.parametrize("adapter", ["openai_chat", "otel_messages", "otel_indexed",
                                     "otel_events"])
def test_the_earlier_turns_stay_in_the_prompt(adapter):
    """The transcript the model saw is the whole conversation; only `question` is
    the live turn. Both halves must hold, or "last turn" could be satisfied by
    dropping history. (`langchain` is excluded on purpose: it rebuilds the prompt
    from question + chunks and records ``prompt_origin='rebuilt'``, so it has no
    transcript to preserve.)"""
    inf = _multi_turn_inference(adapter)
    assert _FIRST_TURN in inf.prompt and _LAST_TURN in inf.prompt


# --------------------------------------------------------------------------- #
# D3: a graded relevance stored under a `gold` key became gold=True
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0.3, 0.75, 2, 3, -1, 0.999])
def test_graded_relevance_under_a_gold_key_is_not_a_gold_flag(value):
    """GOLD_KEYS is narrow because "a false positive here fabricates the single most
    decisive retrieval feature". Coercing every nonzero number with bool() re-opened
    exactly that hole for graded-relevance eval sets."""
    assert gold_flag({"gold": value}) is None


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), (1, True), (0, False), (1.0, True), (0.0, False),
    ("true", True), ("no", False), ("1", True),
])
def test_an_unambiguous_gold_flag_is_still_honoured(value, expected):
    """Guard the narrowing from over-reaching: a real annotation must survive."""
    assert gold_flag({"gold": value}) is expected


def test_a_graded_gold_key_does_not_mask_an_explicit_one():
    assert gold_flag({"gold": 0.3, "is_gold": True}) is True


def test_a_graded_gold_is_reported_as_unknown_not_annotated():
    """End to end: the provenance must not claim the source annotated a gold flag."""
    inf = lc.to_inference({
        "question": "Where is the depot?", "result": "Antwerp.",
        "source_documents": [
            {"page_content": "The depot is in Antwerp.", "metadata": {"gold": 0.3}},
            {"page_content": "Rotterdam handles freight.", "metadata": {"gold": 0.1}},
        ],
    })
    assert [c.gold for c in inf.retrieved_context] == [None, None]
    assert inf.meta["ingest"]["gold_flags"] == "unknown"


def test_a_graded_gold_in_a_field_map_is_not_a_gold_flag_either():
    inf = load_any(
        {"a": "Antwerp.", "docs": [{"t": "The depot is in Antwerp.", "rel": 0.9}]},
        field_map={"answer": "a", "chunks": "docs[].t", "chunk_gold": "docs[].rel"},
    )
    assert [c.gold for c in inf.retrieved_context] == [None]
    assert inf.meta["ingest"]["gold_flags"] == "unknown"


# --------------------------------------------------------------------------- #
# D4: load_traces input handling
# --------------------------------------------------------------------------- #
_ROW = {"question": "Where is the depot?", "result": "Antwerp.",
        "source_documents": [{"page_content": "The depot is in Antwerp."}]}


@pytest.mark.parametrize("suffix,body", [
    (".json", json.dumps([_ROW, _ROW])),
    (".jsonl", json.dumps(_ROW) + "\n" + json.dumps(_ROW) + "\n"),
])
def test_a_utf8_bom_breaks_neither_format(tmp_path, suffix, body):
    """Windows tooling and several log exporters prepend a BOM. Read as plain
    utf-8 it became content and BOTH formats failed on the first record."""
    path = tmp_path / f"bom{suffix}"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert [t.generated_answer for t in load_traces(path)] == ["Antwerp.", "Antwerp."]


@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_non_utf8_bytes_raise_an_ingest_error_not_a_unicode_error(tmp_path, suffix):
    """The contract is that every load_traces failure is an IngestError naming the
    file; a raw UnicodeDecodeError names only a byte offset. Both readers are
    covered: `.jsonl` streams the file, `.json` reads it whole, and the wrap has to
    be on both or one format still leaks the raw error."""
    path = tmp_path / f"latin{suffix}"
    path.write_bytes(json.dumps(_ROW).encode("utf-8") + b"\n" + b'{"q": "caf\xe9"}\n')
    with pytest.raises(IngestError, match="not valid UTF-8"):
        list(load_traces(path))


def test_a_directory_raises_an_ingest_error_not_is_a_directory(tmp_path):
    target = tmp_path / "traces"
    target.mkdir()
    with pytest.raises(IngestError, match="is a directory"):
        list(load_traces(target))


def test_an_adapter_error_names_the_file_and_the_line(tmp_path):
    """This is the PRIMARY batch path: a field-mapping failure on record 5 of 40,000
    used to surface as a bare "openai_chat: could not find the completion text" with
    nothing identifying the record, even though the syntax-error path next to it
    carefully reported path:lineno."""
    path = tmp_path / "bl.jsonl"
    rows = [json.dumps(_ROW)] * 4 + [json.dumps({"messages": [{"role": "user", "content": "hi"}]})]
    path.write_text("\n".join(rows))
    with pytest.raises(IngestError) as exc:
        list(load_traces(path))
    message = str(exc.value)
    assert "bl.jsonl:5" in message
    assert "could not find the completion text" in message   # the cause is still there


def test_an_adapter_error_names_the_index_for_a_json_array(tmp_path):
    path = tmp_path / "bl.json"
    path.write_text(json.dumps([_ROW, {"messages": [{"role": "user", "content": "hi"}]}]))
    with pytest.raises(IngestError) as exc:
        list(load_traces(path))
    assert "bl.json[1]" in str(exc.value)


def test_a_positional_id_and_its_error_locator_name_the_same_record(tmp_path):
    """Ids were 0-based while error messages were 1-based, so `bl.jsonl#4` and
    `bl.jsonl:5` were the same line and nothing said so."""
    path = tmp_path / "bl.jsonl"
    path.write_text("\n".join([json.dumps(_ROW)] * 5))
    # Line mode: 1-based, the same numbering `path:lineno` errors use.
    assert [t.id for t in load_traces(path)] == [f"bl.jsonl#{n}" for n in (1, 2, 3, 4, 5)]

    # Break the FIFTH record and check the error names the number its id would have.
    broken = tmp_path / "bl.jsonl"
    broken.write_text("\n".join([json.dumps(_ROW)] * 4 + ["{not json}"]))
    with pytest.raises(IngestError) as exc:
        list(load_traces(broken))
    assert "bl.jsonl:5" in str(exc.value)

    # Array mode: 0-based, the same numbering `path[index]` errors use.
    array = tmp_path / "bl.json"
    array.write_text(json.dumps([_ROW] * 3))
    assert [t.id for t in load_traces(array)] == [f"bl.json#{n}" for n in (0, 1, 2)]


# --------------------------------------------------------------------------- #
# D5: `.jsonl` was documented as streaming while reading the whole file first
# --------------------------------------------------------------------------- #
def test_jsonl_is_genuinely_streamed_before_the_first_yield(tmp_path):
    """The docstring says "prefer .jsonl for large exports" and callers size their
    batches on that. `read_text()` + `splitlines()` allocated roughly TWICE the file
    before yielding record one."""
    import tracemalloc

    path = tmp_path / "big.jsonl"
    row = json.dumps({"question": "Where is the depot?" * 5, "result": "Antwerp." * 5,
                      "source_documents": [{"page_content": "x" * 600}]})
    with path.open("w") as handle:
        for _ in range(6000):
            handle.write(row + "\n")
    size = path.stat().st_size
    assert size > 4_000_000, "fixture too small to discriminate"

    tracemalloc.start()
    try:
        stream = load_traces(path)
        first = next(stream)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert first.generated_answer.startswith("Antwerp.")
    assert peak < size / 4, f"materialized {peak / 1e6:.1f} MB of a {size / 1e6:.1f} MB file"


# --------------------------------------------------------------------------- #
# D6: parse_path quirks. A field path is code; one that resolves to the WRONG
# field is worse than one that fails.
# --------------------------------------------------------------------------- #
def test_a_quoted_key_may_contain_a_closing_bracket():
    """`path.find("]")` stopped at the bracket INSIDE the quoted key, so the path
    was mis-segmented into two keys, neither of which existed."""
    assert resolve_path({"attrs": {"a]b": 7}}, "attrs['a]b']") == 7


def test_only_one_layer_of_quotes_is_stripped():
    """`strip("'\\"")` removed both layers, resolving the key `q` for a path that
    asked for the key `'q'`."""
    assert resolve_path({"attrs": {"'q'": 7}}, 'attrs["\'q\'"]') == 7
    assert resolve_path({"attrs": {"q": 7}}, 'attrs["\'q\'"]') is None


@pytest.mark.parametrize("path", ["a..b", ".a", "a.", "a.b..c"])
def test_an_empty_path_segment_is_an_error_not_a_silent_collapse(path):
    """`a..b` collapsed to `a.b`, so the most likely typo in a hand-written field
    map resolved SUCCESSFULLY against the path the user did not write."""
    with pytest.raises(IngestError, match="empty segment"):
        resolve_path({"a": {"b": {"c": 1}}}, path)


@pytest.mark.parametrize("path,match", [
    ("a[--1]", "malformed list index"),
    ("a['unterminated", "unterminated"),
    ("a[0", "unterminated"),
])
def test_malformed_bracket_syntax_raises_an_ingest_error(path, match):
    """`[--1]` reached `int()` and surfaced a bare ValueError from inside the parser
    rather than the named IngestError every other path error raises."""
    with pytest.raises(IngestError, match=match):
        resolve_path({"a": [1, 2]}, path)


def test_the_ordinary_path_forms_still_parse():
    """Guard the strictness from over-reaching: '.' after ']' is legitimate."""
    obj = {"a": {"b": [{"c": 1}, {"c": 2}]}, "d": {"gen_ai.request.model": "m1"}}
    assert resolve_path(obj, "a.b[1].c") == 2
    assert resolve_path(obj, "a.b[].c") == [1, 2]
    assert resolve_path(obj, "a.b[-1].c") == 2
    assert resolve_path(obj, "d['gen_ai.request.model']") == "m1"


# --------------------------------------------------------------------------- #
# D7: prose chunking — formatting must not decide the dilution diagnosis
# --------------------------------------------------------------------------- #
_FACTS = [
    "The warehouse in Rotterdam is open on weekdays.",
    "Returns are handled at the Antwerp depot.",
    "Refunds are issued within five business days.",
    "Shipping is free on orders above 50 EUR.",
    "Telephone support opens at 09:00 local time.",
]


def _chat(context_body: str):
    return load_any({
        "messages": [{"role": "user", "content": f"Context:\n{context_body}\n\nQuestion: where?"}],
        "choices": [{"message": {"content": "Rotterdam."}}],
    })


def _chat_no_question(content: str):
    """No trailing `Question:` header, so the block runs to end of message."""
    return load_any({"messages": [{"role": "user", "content": content}],
                     "choices": [{"message": {"content": "Antwerp."}}]})


def test_a_bulleted_list_is_chunked_like_the_same_text_as_prose():
    """One document containing a five-item bulleted list became six chunks with
    gold_position_frac=0.2 while the identical text as prose stayed one chunk, and
    `rules.dil.buried` keys on exactly n_chunks + gold_position_frac — so the
    FORMATTING of a log changed the dilution diagnosis."""
    prose = _chat(" ".join(_FACTS))
    bullets = _chat("\n".join(f"- {f}" for f in _FACTS))
    assert len(bullets.retrieved_context) == len(prose.retrieved_context) == 1
    # ...and no text was lost in the merge.
    for fact in _FACTS:
        assert fact in bullets.retrieved_context[0].text


def test_numbered_document_markers_still_split():
    """Guard the above from over-reaching: `[1]`/`1.` carry a document IDENTITY, so
    they remain real boundaries. Losing those would flatten every genuine
    multi-document prompt to n_chunks=1."""
    numbered = _chat("\n".join(f"[{i + 1}] {f}" for i, f in enumerate(_FACTS)))
    assert [c.source_id for c in numbered.retrieved_context] == ["1", "2", "3", "4", "5"]


_HUGE = 20_000


@pytest.mark.parametrize("body,builder", [
    ("numbered", lambda i: f"{i + 1}. Fact number {i} about logistics."),
    ("separators", lambda i: f"Fact number {i} about logistics.\n---"),
    ("paragraphs", lambda i: f"Fact number {i} about logistics.\n"),
    ("doc_tags", lambda i: f"<document id='d{i}'>Fact number {i} about logistics.</document>"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_one_block_cannot_produce_unbounded_chunks(body, builder):
    """20,000 list lines produced 20,000 Chunk objects. Over the cap the block
    degrades to a single chunk — coarser, but no text dropped, because dropping
    retrieved text is what fabricates a retrieval failure. Every splitting path is
    covered: an unbounded one next to three bounded ones is how this comes back."""
    inf = _chat("\n".join(builder(i) for i in range(_HUGE)))
    chunks = inf.retrieved_context
    assert 1 <= len(chunks) <= openai_chat._MAX_BLOCK_CHUNKS
    assert f"Fact number {_HUGE - 1}" in "".join(c.text for c in chunks)
    assert inf.meta["ingest"]["context_origin"].endswith("_capped")


def test_the_cap_is_far_above_any_plausible_retrieval_set():
    """The cap must not degrade a genuinely large context: collapsing 100 documents
    to one chunk would drive n_chunks from 100 to 1 and switch `dil.buried`'s
    up(n_chunks, 4, 10) term from saturated to zero, suppressing the dilution
    geometry in exactly the case most likely to BE dilution."""
    inf = _chat("\n".join(f"[{i + 1}] Fact number {i} about logistics." for i in range(100)))
    assert len(inf.retrieved_context) == 100
    assert not inf.meta["ingest"]["context_origin"].endswith("_capped")


def test_a_trailing_instruction_is_not_swallowed_into_the_last_chunk():
    """With no next-section header the block ran to end of message, so
    "Answer using only the context above." landed inside the final chunk and was
    scored as retrieved evidence."""
    inf = _chat_no_question(
        "Context:\n[1] Rotterdam handles freight.\n[2] Antwerp handles returns.\n"
        "Answer using only the context above.")
    texts = [c.text for c in inf.retrieved_context]
    assert texts == ["Rotterdam handles freight.", "Antwerp handles returns."]
    # Handed back to the RESIDUAL, not deleted — it is real text the model saw, just
    # not a retrieved document. (Asserting on `prompt` cannot fail: the prompt is
    # rendered from the original messages, so it holds the line either way.)
    assert inf.question == "Answer using only the context above."


def test_a_document_whose_last_line_is_prose_is_left_alone():
    """Guard the trailing-instruction cut from deleting real retrieved text: the
    pattern needs BOTH an imperative and a reference to the context itself."""
    inf = _chat_no_question(
        "Context:\n[1] Rotterdam handles freight.\n[2] Antwerp handles returns.\n"
        "The depot was rebuilt in 2019.")
    assert "The depot was rebuilt in 2019." in inf.retrieved_context[-1].text


def test_an_instruction_on_the_same_line_as_an_entry_is_kept():
    """The cut is line-granular by design. Trimming mid-line would mean guessing
    where a document ends, and a wrong guess deletes retrieved text — the failure
    this whole change is guarding against."""
    inf = _chat_no_question(
        "Context:\n[1] Rotterdam handles freight.\n"
        "[2] Antwerp handles returns. Answer using only the context above.")
    assert inf.retrieved_context[-1].text == (
        "Antwerp handles returns. Answer using only the context above.")


def test_a_context_block_that_is_only_an_instruction_is_not_retrieval():
    """The degenerate case: trimming can empty the block. Nothing is lost — no
    chunks are produced, the trace is reported non-RAG, and the text stays in the
    message. A `Context:` header followed only by an instruction genuinely carries
    no documents, so `retrieved_context=None` is the right answer, not a hole."""
    inf = _chat_no_question("Context:\nAnswer using only the context above.")
    assert inf.retrieved_context is None
    assert inf.meta["ingest"]["context_origin"] == "none"
    assert "Answer using only the context above." in inf.prompt


# --------------------------------------------------------------------------- #
# D8: otel concatenated multiple completions into a string no model produced
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("span", [
    {"attributes": {"gen_ai.prompt": "What is 3+4?"},
     "events": [{"name": "gen_ai.choice", "attributes": {"content": "7"}},
                {"name": "gen_ai.choice", "attributes": {"content": "11"}}]},
    {"attributes": {"gen_ai.prompt": "What is 3+4?",
                    "gen_ai.completion.0.content": "7",
                    "gen_ai.completion.1.content": "11"}},
    {"attributes": {"gen_ai.prompt": "What is 3+4?",
                    "gen_ai.completion": json.dumps([{"role": "assistant", "content": "7"},
                                                     {"role": "assistant", "content": "11"}])}},
], ids=["choice_events", "indexed", "json_array"])
def test_otel_takes_the_first_choice_like_openai_chat_does(span):
    """Two sampled completions became `generated_answer = "7\\n11"` — a string no
    model produced — and every correctness/support/confidence signal was computed
    against that composite. `openai_chat` takes `choices[0]` for the same payload."""
    inf = otel.to_inference(span)
    assert inf.generated_answer == "7"
    assert inf.meta["n_choices"] == 2      # the discard is recorded, not invisible


def test_a_single_completion_records_no_choice_count():
    inf = otel.to_inference({"attributes": {"gen_ai.prompt": "q", "gen_ai.completion": "a"}})
    assert inf.generated_answer == "a"
    assert "n_choices" not in inf.meta


# --------------------------------------------------------------------------- #
# D9: report_to_dict's _fin sweep skipped three floats
# --------------------------------------------------------------------------- #
def _report_with(field: str, value: float):
    from tokentrace.core.types import Diagnosis, DiagnosisReport, FailureMode

    diagnosis = Diagnosis(mode=FailureMode.HALLUCINATION, probability=0.4, confidence=0.7)
    report = DiagnosisReport(diagnoses=[diagnosis], diagnostic_confidence=0.5)
    target = report if field == "diagnostic_confidence" else diagnosis
    setattr(target, field, value)
    return report


@pytest.mark.parametrize("field", ["probability", "confidence", "diagnostic_confidence"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_every_report_float_is_scrubbed_before_json(field, value):
    """A NaN in ANY of these emitted a bare `NaN` token, which is not RFC 8259 and
    poisons the WHOLE `analyze --json` document. The standing guard builds a
    *healthy* report, so it can never observe a gap in the sweep — the value has to
    be injected field by field."""
    from tokentrace.core.serialize import report_to_dict

    blob = json.dumps(report_to_dict(_report_with(field, value)))
    json.loads(blob, parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(f"non-RFC-8259 token in output: {c}")))


@pytest.mark.parametrize("field", ["probability", "confidence", "diagnostic_confidence"])
def test_a_scrubbed_report_float_round_trips_to_a_usable_number(field):
    """`.get(key, default)` does not cover an explicit null, so a scrubbed field came
    back as None where the dataclass declares float — the same defect that had to be
    fixed for `chunk_from_dict`."""
    from tokentrace.core.serialize import report_from_dict, report_to_dict

    blob = json.loads(json.dumps(report_to_dict(_report_with(field, float("nan")))))
    back = report_from_dict(blob)
    holder = back if field == "diagnostic_confidence" else back.diagnoses[0]
    value = getattr(holder, field)
    assert isinstance(value, float) and math.isfinite(value)


def test_labeled_inference_floats_are_scrubbed_too():
    from tokentrace.core.serialize import labeled_from_dict, labeled_to_dict
    from tokentrace.core.types import FailureMode, Inference, LabeledInference

    li = LabeledInference(inference=Inference("p", "a"), labels=[FailureMode.HALLUCINATION],
                          severity={FailureMode.HALLUCINATION: float("nan")},
                          weight=float("inf"))
    blob = json.dumps(labeled_to_dict(li))
    json.loads(blob, parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(f"non-RFC-8259 token in output: {c}")))
    back = labeled_from_dict(json.loads(blob))
    assert math.isfinite(back.weight)
    assert math.isfinite(back.severity[FailureMode.HALLUCINATION])


# --------------------------------------------------------------------------- #
# D10: the mutations that survived the reviewer's run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["", "   ", "\n\t "])
def test_an_empty_reference_string_is_absent_not_a_reference(value):
    """`Inference.has_ground_truth` already treats `[""]` as absent; returning it
    here instead of None makes the two representations disagree, so `ground_truth`
    is truthy while `has_ground_truth` is False and callers branch inconsistently."""
    assert normalize_reference(value) is None
    inf = load_any({"messages": [{"role": "user", "content": "Q?"}],
                    "choices": [{"message": {"content": "A."}}],
                    "ground_truth": value})
    assert inf.ground_truth is None
    assert inf.meta["ingest"]["reference"] == "absent"


def test_a_distance_is_never_read_as_a_retriever_score():
    """langchain documents this as a correctness invariant — a distance is inverted
    relative to a similarity, so copying one in "would record the most relevant chunk
    as the least relevant one" — but llamaindex had no equivalent guard and no test,
    and the vector stores it wraps return exactly that key."""
    for keys in (lc._SCORE_KEYS, li._SCORE_KEYS):
        assert not any("distance" in key for key in keys), keys

    inf = li.to_inference({
        "query": "Where is the depot?", "response": "Antwerp.",
        "source_nodes": [
            {"node": {"id_": "n1", "text": "The depot is in Antwerp."}, "distance": 0.02},
            {"node": {"id_": "n2", "text": "Rotterdam handles freight."}, "distance": 0.91},
        ],
    })
    # Unmeasured, and reported as unmeasured — NOT written down as a (backwards) score.
    assert [c.retriever_score for c in inf.retrieved_context] == [0.0, 0.0]
    assert inf.meta["ingest"]["retriever_scores"] == "absent"


def test_a_real_similarity_score_still_survives():
    """Guard the above from over-reaching."""
    inf = li.to_inference({
        "query": "Where is the depot?", "response": "Antwerp.",
        "source_nodes": [{"node": {"id_": "n1", "text": "The depot is in Antwerp."},
                          "score": 0.81}],
    })
    assert [c.retriever_score for c in inf.retrieved_context] == [0.81]
    assert inf.meta["ingest"]["retriever_scores"] == "present"


def test_an_empty_chunk_is_not_evidence():
    """ChunkBuilder.add's docstring says so; nothing enforced it. An empty chunk
    inflates n_chunks and shifts gold_position_frac — the two features dil.buried
    keys on — and contributes a 0.0 to mean_chunk_relevance."""
    builder = ChunkBuilder()
    assert builder.add("The depot is in Antwerp.", source_id="d1") is not None
    for blank in ("", "   ", "\n\n", None):
        assert builder.add(blank, source_id="blank") is None
    assert len(builder) == 1
    assert [c.source_id for c in builder.chunks] == ["d1"]


def test_an_empty_document_does_not_become_a_chunk_end_to_end():
    inf = lc.to_inference({
        "question": "Where is the depot?", "result": "Antwerp.",
        "source_documents": [
            {"page_content": "The depot is in Antwerp.", "metadata": {"score": 0.9}},
            {"page_content": "   ", "metadata": {"score": 0.4}},
            {"page_content": "Rotterdam handles freight.", "metadata": {"score": 0.3}},
        ],
    })
    assert [c.text for c in inf.retrieved_context] == [
        "The depot is in Antwerp.", "Rotterdam handles freight."]
