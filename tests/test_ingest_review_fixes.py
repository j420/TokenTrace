"""Regressions for the defects an adversarial review of `tokentrace/ingest/` found.

Each test here corresponds to a defect that shipped and that the existing suite did
NOT catch — the review proved it by mutation, breaking the implementation and
watching every test stay green. They are grouped by the failure they prevent rather
than by module, because the serious ones share a theme: an adapter that invents or
destroys retrieval evidence makes the engine *confidently wrong*, which is worse
than making it silent.
"""

from __future__ import annotations

import pytest

from tokentrace.api import TokenTrace
from tokentrace.ingest import detect_all, load_any
from tokentrace.ingest.base import STRUCTURED_RETRIEVAL_KEYS, normalize_reference


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
            "- The Eiffel Tower was completed in 1889.\n"
            "- The population of the city proper is 2.1 million."
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
