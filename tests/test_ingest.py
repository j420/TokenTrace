"""Trace ingestion: adapter field mapping, detection, routing, and round-trip.

Two properties carry most of the weight here and are written so they can actually
fail:

* the **detection matrix** — each adapter must accept its own fixture and reject
  all five of the others. Detection is heuristic, so a loosened predicate in any
  adapter breaks a specific cell of the matrix rather than silently swallowing a
  neighbour's payload at routing time.
* the **degraded round-trip** — an ingested production trace (no reference answer,
  no gold flags) must reach a ``DiagnosisReport`` at every tier, with the
  reference-dependent features genuinely absent and the ``|noref`` calibration
  signature selected. Its reference-bearing counterpart must NOT get that
  signature, so the assertion discriminates in both directions.

Every field assertion compares against a concrete expected value taken from the
fixture, never ``is not None``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokentrace.core.types import DiagnosisReport, FailureMode, Tier
from tokentrace.ingest import (
    IngestError,
    UnknownSourceError,
    detect_all,
    detect_source,
    generic,
    langchain,
    llamaindex,
    load_any,
    load_traces,
    openai_chat,
    otel,
    resolve_path,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"

#: fixture file -> the adapter that must claim it (and only it).
EXPECTED_SOURCE = {
    "openai_chat.json": "openai_chat",
    "langchain.json": "langchain",
    "llamaindex.json": "llamaindex",
    "otel_flat.json": "otel",
    "otel_events.json": "otel",
    "generic.json": "generic",
}

ADAPTER_MODULES = {
    "openai_chat": openai_chat,
    "langchain": langchain,
    "llamaindex": llamaindex,
    "otel": otel,
    "generic": generic,
}


def payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def inference(name: str):
    return load_any(payload(name))


@pytest.fixture(scope="module")
def tt():
    from tokentrace import TokenTrace

    # Cold-start (rules only): the round-trip is about the ingest contract holding
    # up through the pipeline, not about learned-head quality.
    return TokenTrace.default(train=False)


# --------------------------------------------------------------------------- #
# 1. Field-by-field mapping
# --------------------------------------------------------------------------- #
def test_openai_chat_fields():
    inf = inference("openai_chat.json")
    assert inf.id == "chatcmpl-tt-openai-0001"
    assert inf.generated_answer == "You have 30 days from delivery to return an unopened item [kb-101]."
    # question = the user's query; prompt = everything the model saw. Different fields.
    assert inf.question == "How long do I have to return an unopened widget?"
    assert inf.prompt.startswith("system: You are a support assistant for the Widgetworks returns desk.")
    assert inf.prompt.endswith("user: How long do I have to return an unopened widget?")
    assert "[kb-102] Refunds are issued" in inf.prompt   # context stays in the prompt
    assert inf.question not in ("", inf.prompt)

    chunks = inf.retrieved_context
    assert [c.source_id for c in chunks] == ["kb-101", "kb-102", "kb-103"]
    assert chunks[0].text == "Standard returns are accepted within 30 days of delivery for unopened items."
    assert chunks[2].text == "The Widgetworks warehouse is located in Rotterdam and is open on weekdays."
    # A rendered prompt carries neither scores nor gold annotations, and we say so.
    assert [c.gold for c in chunks] == [None, None, None]
    assert inf.meta["ingest"]["retriever_scores"] == "absent"
    assert inf.meta["ingest"]["gold_flags"] == "unknown"
    assert inf.meta["ingest"]["context_origin"] == "message[0]:header/entry_markers"
    assert inf.meta["model_name"] == "gpt-4o-mini"
    assert inf.meta["source"] == "openai_chat"
    assert inf.ground_truth is None and inf.has_ground_truth is False


def test_langchain_fields():
    inf = inference("langchain.json")
    assert inf.id == "lc-run-7f2a"
    assert inf.question == "Which port does the Widgetworks warehouse ship from?"
    assert inf.generated_answer == "The Widgetworks warehouse ships from the port of Rotterdam."
    # A reference the source really logged IS mapped (and `result` is never read as one).
    assert inf.ground_truth == ["Rotterdam"]
    assert inf.has_ground_truth is True

    chunks = inf.retrieved_context
    assert [c.retriever_score for c in chunks] == [0.87, 0.51, 0.22]
    assert [c.source_id for c in chunks] == [
        "logistics_handbook.md", "logistics_handbook.md", "support_policy.md",
    ]
    # Explicit metadata gold flag honoured; unannotated docs stay UNKNOWN, not False.
    assert [c.gold for c in chunks] == [True, None, None]
    assert chunks[1].text.startswith("Widgetworks operates a small returns depot in Antwerp")

    assert inf.meta["ingest"]["retriever_scores"] == "present"
    assert inf.meta["ingest"]["gold_flags"] == "annotated"
    assert inf.meta["ingest"]["reference"] == "logged"
    assert inf.meta["ingest"]["prompt_origin"] == "rebuilt"
    assert inf.meta["model_name"] == "claude-haiku-4-5"
    assert inf.prompt.startswith("Which port does the Widgetworks warehouse ship from?\n\n")
    assert chunks[0].text in inf.prompt


def test_llamaindex_fields():
    inf = inference("llamaindex.json")
    assert inf.question == "What is the escalation path for a delayed shipment?"
    assert inf.generated_answer == (
        "Delayed shipments are escalated to the regional logistics lead after 48 hours."
    )
    chunks = inf.retrieved_context
    assert len(chunks) == 3
    assert [c.retriever_score for c in chunks] == [0.7421, 0.6013, 0.0]
    assert [c.source_id for c in chunks] == ["ops_runbook.md", "ops_runbook.md", "hr_onboarding.md"]
    assert chunks[0].text.startswith("If a shipment is delayed by more than 48 hours")
    assert [c.gold for c in chunks] == [None, None, None]
    # The third node logged NO score. It reads 0.0 only because Chunk.retriever_score
    # is a non-optional float; the provenance is what tells a consumer it is unmeasured.
    assert inf.meta["ingest"]["retriever_scores"] == "partial"
    assert inf.meta["model_name"] == "local-qwen3-4b"
    assert inf.ground_truth is None


def test_otel_flat_fields():
    inf = inference("otel_flat.json")
    assert inf.id == "00f067aa0ba902b7"
    assert inf.generated_answer == (
        "Michelangelo painted the ceiling of the Sistine Chapel between 1508 and 1512."
    )
    assert inf.question == "Who painted the ceiling of the Sistine Chapel?"
    assert inf.prompt == (
        "system: You are a concise encyclopaedia assistant.\n\n"
        "user: Who painted the ceiling of the Sistine Chapel?"
    )
    # No delimited context block => non-RAG, so the retrieval modes get hard-masked.
    # `[]` here would instead read as "a RAG pipeline that retrieved nothing".
    assert inf.retrieved_context is None and inf.is_rag is False
    assert inf.meta["ingest"]["context_origin"] == "none"
    assert inf.meta["otel_form"] == "flat"
    assert inf.meta["provider"] == "openai"
    assert inf.meta["model_name"] == "gpt-4o-mini"


def test_otel_events_fields():
    inf = inference("otel_events.json")
    assert inf.id == "b7ad6b7169203331"
    # The completion arrives as a JSON-encoded message inside a gen_ai.choice event.
    assert inf.generated_answer == (
        "Outbound freight leaves the Rotterdam warehouse every weekday at 06:00."
    )
    assert inf.question == "When does outbound freight leave?"
    chunks = inf.retrieved_context
    assert [c.source_id for c in chunks] == ["1", "2"]
    assert chunks[0].text == (
        "The Antwerp depot processes returns only and does not handle outbound freight."
    )
    assert chunks[1].text == "Outbound freight leaves the Rotterdam warehouse every weekday at 06:00."
    # The block must stop at the next section header, not swallow the instructions.
    assert "Instructions:" not in chunks[1].text
    assert inf.meta["otel_form"] == "events"
    assert inf.meta["model_name"] == "claude-haiku-4-5"


def test_otel_otlp_attribute_values_are_unwrapped():
    attrs = otel.attributes(payload("otel_flat.json"))
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["gen_ai.usage.input_tokens"] == 58        # protobuf JSON int64 -> str -> int


def test_otel_indexed_flat_attributes():
    """OpenLLMetry-style ``gen_ai.prompt.<i>.role/.content`` pairs."""
    span = {
        "attributes": {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4.1",
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.0.content": "You are terse.",
            "gen_ai.prompt.1.role": "user",
            "gen_ai.prompt.1.content": "What is the capital of France?",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "Paris.",
        }
    }
    inf = otel.to_inference(span)
    assert inf.prompt == "system: You are terse.\n\nuser: What is the capital of France?"
    assert inf.question == "What is the capital of France?"
    assert inf.generated_answer == "Paris."
    assert inf.meta["otel_form"] == "indexed"


def test_otel_new_message_convention_parts():
    """``gen_ai.input.messages`` with the ``parts`` content shape."""
    span = {
        "attributes": {
            "gen_ai.provider.name": "anthropic",
            "gen_ai.input.messages": json.dumps([
                {"role": "user", "parts": [{"type": "text", "content": "Define entropy."}]}
            ]),
            "gen_ai.output.messages": json.dumps([
                {"role": "assistant",
                 "parts": [{"type": "text", "content": "A measure of uncertainty."}]}
            ]),
        }
    }
    inf = otel.to_inference(span)
    assert inf.question == "Define entropy."
    assert inf.generated_answer == "A measure of uncertainty."
    assert inf.meta["otel_form"] == "input.messages"


def test_generic_field_map_fields():
    inf = inference("generic.json")
    assert inf.id == "row-90211"
    assert inf.question == "What is the response target for a priority-one incident?"
    assert inf.generated_answer == "A priority-one incident must get a first response within 15 minutes, 24/7."
    # The payload logged the real rendered prompt, so it is used verbatim.
    assert inf.meta["ingest"]["prompt_origin"] == "logged"
    assert inf.prompt.startswith("SYSTEM: Answer from the runbook excerpts.")
    chunks = inf.retrieved_context
    assert [c.source_id for c in chunks] == ["runbook#sla-p1", "runbook#sla-p2", "runbook#oncall"]
    assert [c.retriever_score for c in chunks] == [0.912, 0.478, 0.0]
    assert inf.meta["ingest"]["retriever_scores"] == "partial"   # third hit logged null
    assert inf.meta["model_name"] == "internal-llm-v3"
    assert inf.ground_truth is None


def test_generic_field_map_passed_explicitly():
    """The long-tail path: an unrecognizable payload plus a caller-supplied map."""
    raw = {
        "input": {"text": "Who signed the Treaty of Waitangi?"},
        "output": {"text": "Representatives of the British Crown and Maori chiefs."},
        "docs": [
            {"body": "The Treaty of Waitangi was signed in 1840.", "rank": 1},
            {"body": "Waitangi is in the Bay of Islands.", "rank": 2},
        ],
    }
    assert detect_source(raw) is None            # nothing claims it without a map
    inf = load_any(raw, field_map={
        "question": "input.text",
        "answer": "output.text",
        "chunks": "docs[].body",
    })
    assert inf.question == "Who signed the Treaty of Waitangi?"
    assert inf.generated_answer == "Representatives of the British Crown and Maori chiefs."
    assert [c.text for c in inf.retrieved_context] == [
        "The Treaty of Waitangi was signed in 1840.",
        "Waitangi is in the Bay of Islands.",
    ]
    assert inf.meta["ingest"]["retriever_scores"] == "absent"
    assert inf.meta["ingest"]["prompt_origin"] == "rebuilt"


# --------------------------------------------------------------------------- #
# 2. Detection matrix — the test that can genuinely fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture,expected", sorted(EXPECTED_SOURCE.items()))
@pytest.mark.parametrize("adapter", sorted(ADAPTER_MODULES))
def test_detect_matrix(fixture, expected, adapter):
    """Every adapter accepts its own fixture and rejects all the others."""
    claimed = ADAPTER_MODULES[adapter].detect(payload(fixture))
    assert claimed is (adapter == expected), (
        f"{adapter}.detect({fixture}) returned {claimed}; expected {adapter == expected}"
    )


@pytest.mark.parametrize("fixture,expected", sorted(EXPECTED_SOURCE.items()))
def test_exactly_one_adapter_claims_each_fixture(fixture, expected):
    assert detect_all(payload(fixture)) == [expected]


#: Realistic payloads that carry MORE than one source's markers — a callback
#: handler that logs the raw chat call next to the structured retrieval, or an
#: OTel span with the messages mirrored to the top level. Exactly one adapter must
#: claim each, and it must be the one that recovers real retrieval structure
#: rather than re-parsing the rendered prompt. Without these cases the matrix
#: above cannot detect a detector that has been loosened, because no single-source
#: fixture carries a competitor's markers.
HYBRID_PAYLOADS = {
    "langchain callback logging the raw chat call": (
        {
            "run_id": "lc-cb-1",
            "question": "Where is the depot?",
            "result": "In Antwerp.",
            "messages": [
                {"role": "system", "content": "Answer from the documents."},
                {"role": "user", "content": "Where is the depot?"},
            ],
            "source_documents": [
                {"page_content": "The returns depot is in Antwerp.",
                 "metadata": {"source": "logistics.md", "score": 0.91}}
            ],
        },
        "langchain",
    ),
    "llamaindex trace logging the raw chat call": (
        {
            "query": "Where is the depot?",
            "response": "In Antwerp.",
            "messages": [{"role": "user", "content": "Where is the depot?"}],
            "source_nodes": [
                {"score": 0.9, "node": {"id_": "n1", "text": "The returns depot is in Antwerp."}}
            ],
        },
        "llamaindex",
    ),
    "otel span with the messages mirrored to the top level": (
        {
            "name": "chat gpt-4o-mini",
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.prompt": "[{\"role\": \"user\", \"content\": \"Where is the depot?\"}]",
                "gen_ai.completion": "In Antwerp.",
            },
            "messages": [{"role": "user", "content": "Where is the depot?"}],
            "choices": [{"message": {"role": "assistant", "content": "In Antwerp."}}],
        },
        "otel",
    ),
}


@pytest.mark.parametrize("case", sorted(HYBRID_PAYLOADS))
def test_hybrid_payload_is_claimed_only_by_the_structured_adapter(case):
    raw, expected = HYBRID_PAYLOADS[case]
    assert detect_all(raw) == [expected]
    inf = load_any(raw)
    assert inf.meta["source"] == expected
    # The structured adapters must actually use the structure they won on.
    if expected == "langchain":
        assert [c.retriever_score for c in inf.retrieved_context] == [0.91]
    elif expected == "llamaindex":
        assert [c.retriever_score for c in inf.retrieved_context] == [0.9]


def test_detectors_reject_non_traces():
    for junk in ({}, {"foo": "bar"}, [], "a string", 42, None, {"messages": "not a list"}):
        assert detect_all(junk) == []
        assert detect_source(junk) is None


# --------------------------------------------------------------------------- #
# 3. Routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture,expected", sorted(EXPECTED_SOURCE.items()))
def test_load_any_routes_to_the_right_adapter(fixture, expected):
    inf = load_any(payload(fixture))
    assert inf.meta["source"] == expected
    assert inf.meta["ingest"]["adapter"] == expected
    assert inf.generated_answer == ADAPTER_MODULES[expected].to_inference(
        payload(fixture)
    ).generated_answer


def test_explicit_source_overrides_detection():
    # A LangChain payload forced through the generic adapter must fail loudly rather
    # than be quietly mis-parsed.
    with pytest.raises(IngestError):
        load_any(payload("langchain.json"), source="generic")
    with pytest.raises(IngestError, match="unknown ingest source"):
        load_any(payload("langchain.json"), source="not_a_real_adapter")


# --------------------------------------------------------------------------- #
# 4. Round-trip through the analysis pipeline, at every tier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", sorted(EXPECTED_SOURCE))
@pytest.mark.parametrize("tier", [Tier.BLACK, Tier.GREY, Tier.WHITE])
def test_ingested_trace_analyzes_at_every_tier(tt, fixture, tier):
    inf = inference(fixture)
    report = tt.analyze(inf, tier=tier)

    assert isinstance(report, DiagnosisReport)
    assert report.tier == tier
    assert report.inference_id == inf.id
    assert len(report.diagnoses) == 5                       # every mode is ranked
    assert all(0.0 <= d.probability <= 1.0 for d in report.diagnoses)
    assert report.primary is not None
    # Ingest must not have silently invented an answer.
    assert inf.generated_answer.strip() != ""

    probs = {d.mode: d.probability for d in report.diagnoses}
    if not inf.is_rag:
        # Non-RAG (otel_flat) must have the retrieval modes hard-masked; if the
        # adapter had emitted `[]` instead of None these would be non-zero.
        assert probs[FailureMode.RETRIEVAL_FAILURE] == 0.0
        assert probs[FailureMode.CONTEXT_DILUTION] == 0.0
        assert FailureMode.RETRIEVAL_FAILURE not in report.conformal_set


def test_analysis_does_not_mutate_the_ingested_inference(tt):
    inf = inference("langchain.json")
    before = (inf.prompt, inf.generated_answer, tuple(c.text for c in inf.retrieved_context))
    tt.analyze(inf, tier=Tier.WHITE)
    assert (inf.prompt, inf.generated_answer,
            tuple(c.text for c in inf.retrieved_context)) == before


# --------------------------------------------------------------------------- #
# 5. The degraded case is the primary case
# --------------------------------------------------------------------------- #
DEGRADED = ["openai_chat.json", "llamaindex.json", "otel_flat.json", "otel_events.json",
            "generic.json"]


@pytest.mark.parametrize("fixture", DEGRADED)
def test_degraded_trace_has_no_reference_and_no_gold(tt, fixture):
    inf = inference(fixture)
    assert inf.ground_truth is None
    assert inf.has_ground_truth is False
    # Never guessed: no chunk is marked gold, and none is marked "definitely not gold".
    for chunk in inf.retrieved_context or []:
        assert chunk.gold is None

    fv = tt.features(inf, tier=Tier.BLACK)
    assert fv.reference_available is False
    assert fv.missingness_signature().endswith("|noref")
    # The reference-dependent features must be MISSING, not zero — a real 0.0 here
    # diagnoses a healthy answer as a confident retrieval failure.
    assert fv.get("is_correct") is None
    if inf.is_rag:
        assert fv.get("gold_recall_in_context") is None


def test_reference_bearing_trace_does_not_get_the_noref_signature(tt):
    """The other half of the discrimination: a logged reference must be used."""
    inf = inference("langchain.json")
    fv = tt.features(inf, tier=Tier.BLACK)
    assert fv.reference_available is True
    assert fv.missingness_signature() == "confidence+prompt+retrieval"
    assert isinstance(fv.get("gold_recall_in_context"), float)
    assert isinstance(fv.get("is_correct"), float)


def test_reference_is_never_derived_from_the_generated_answer():
    """RAGAS-shaped rows use `answer` for the GENERATION; reading it as the
    reference would score every ingested trace correct."""
    inf = langchain.to_inference({
        "question": "What is the boiling point of water at sea level?",
        "answer": "100 degrees Celsius.",
        "source_documents": [{"page_content": "Water boils at 100 C at 1 atm.", "metadata": {}}],
    })
    assert inf.generated_answer == "100 degrees Celsius."
    assert inf.ground_truth is None and inf.has_ground_truth is False


def test_empty_retrieval_stays_rag_but_absent_retrieval_is_not():
    """`source_documents: []` is a retriever that returned nothing (a RAG failure);
    an absent key is a non-RAG chain. Collapsing the two masks the very mode the
    first one exhibits."""
    empty = langchain.to_inference({"question": "Q?", "result": "A.", "source_documents": []})
    assert empty.retrieved_context == [] and empty.is_rag is True

    absent = langchain.to_inference({"question": "Q?", "result": "A."})
    assert absent.retrieved_context is None and absent.is_rag is False


def test_undelimited_context_is_not_fabricated():
    """The chat heuristic under-reports rather than inventing retrieval evidence."""
    inf = openai_chat.to_inference({
        "messages": [
            {"role": "system", "content": "You are helpful. The Eiffel Tower opened in 1889."},
            {"role": "user", "content": "When did it open?"},
        ],
        "choices": [{"message": {"role": "assistant", "content": "1889."}}],
    })
    assert inf.retrieved_context is None
    assert inf.meta["ingest"]["context_origin"] == "none"


def test_xml_delimited_context_is_extracted_with_ids():
    inf = openai_chat.to_inference({
        "messages": [
            {"role": "user", "content":
                "<context>\n"
                "<document id=\"a1\">Mount Aspiring is in New Zealand.</document>\n"
                "<document id=\"b2\">Aoraki is the highest peak in New Zealand.</document>\n"
                "</context>\n"
                "Which is the highest peak?"},
        ],
        "choices": [{"message": {"content": "Aoraki."}}],
    })
    assert [c.source_id for c in inf.retrieved_context] == ["a1", "b2"]
    assert inf.retrieved_context[1].text == "Aoraki is the highest peak in New Zealand."
    assert inf.meta["ingest"]["context_origin"] == "message[0]:context_tag/document_tags"
    # The block is stripped from the QUESTION but kept in the PROMPT.
    assert inf.question == "Which is the highest peak?"
    assert "Mount Aspiring" in inf.prompt


def test_trailing_assistant_message_is_the_completion_not_the_prompt():
    inf = openai_chat.to_inference({
        "messages": [
            {"role": "user", "content": "Name a prime above 20."},
            {"role": "assistant", "content": "23."},
        ]
    })
    assert inf.generated_answer == "23."
    assert inf.prompt == "user: Name a prime above 20."
    assert "23." not in inf.prompt


def test_anthropic_shape_with_top_level_system_and_content_blocks():
    inf = openai_chat.to_inference({
        "model": "claude-haiku-4-5",
        "system": "Answer in one word.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Capital of Japan?"}]}],
        "content": [{"type": "text", "text": "Tokyo"}],
    })
    assert inf.generated_answer == "Tokyo"
    assert inf.question == "Capital of Japan?"
    assert inf.prompt == "system: Answer in one word.\n\nuser: Capital of Japan?"
    assert inf.meta["model_name"] == "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# 6. Malformed input -> a clear error naming the field
# --------------------------------------------------------------------------- #
def test_missing_completion_names_the_field():
    with pytest.raises(IngestError) as exc:
        openai_chat.to_inference({"messages": [{"role": "user", "content": "hi"}]})
    message = str(exc.value)
    assert "openai_chat" in message and "completion" in message
    assert "choices[0].message.content" in message      # tells the user what was tried


def test_missing_messages_names_the_field():
    with pytest.raises(IngestError) as exc:
        openai_chat.to_inference({"model": "gpt-4o-mini", "temperature": 0.0})
    assert "messages" in str(exc.value)
    assert "top-level keys present" in str(exc.value)


def test_langchain_missing_answer_names_the_field():
    with pytest.raises(IngestError) as exc:
        langchain.to_inference({"question": "Q?", "source_documents": [{"page_content": "x"}]})
    message = str(exc.value)
    assert "langchain" in message and "generated answer" in message and "result" in message


def test_langchain_missing_question_names_the_field():
    with pytest.raises(IngestError) as exc:
        langchain.to_inference({"result": "A."})
    assert "user question" in str(exc.value)


def test_llamaindex_missing_response_names_the_field():
    with pytest.raises(IngestError) as exc:
        llamaindex.to_inference({"source_nodes": [{"node": {"text": "t"}, "score": 0.5}]})
    assert "response" in str(exc.value)


def test_otel_missing_prompt_names_the_field():
    with pytest.raises(IngestError) as exc:
        otel.to_inference({"attributes": {"gen_ai.system": "openai",
                                          "gen_ai.request.model": "gpt-4o-mini"}})
    message = str(exc.value)
    assert "gen_ai.prompt" in message and "gen_ai.input.messages" in message


def test_generic_bad_path_names_the_path():
    with pytest.raises(IngestError) as exc:
        load_any({"input": {"text": "q"}, "output": {"text": "a"}},
                 field_map={"question": "input.text", "answer": "output.WRONG"})
    assert "output.WRONG" in str(exc.value)


def test_generic_unresolvable_structural_path_names_the_path():
    """A typo in a structural path fails loudly; an optional one is allowed to be
    absent row to row."""
    raw = {"input": {"text": "q"}, "output": {"text": "a"}}
    with pytest.raises(IngestError) as exc:
        load_any(raw, field_map={"question": "input.TYPO", "answer": "output.text"})
    assert "input.TYPO" in str(exc.value)

    # ground_truth is per-row optional: absent is absent, not an error.
    inf = load_any(raw, field_map={"question": "input.text", "answer": "output.text",
                                   "ground_truth": "eval.reference"})
    assert inf.ground_truth is None and inf.question == "q"


def test_generic_unknown_map_key_is_rejected():
    with pytest.raises(IngestError) as exc:
        load_any({"a": 1}, field_map={"answer": "a", "anwser": "a"})
    assert "anwser" in str(exc.value)


def test_generic_ragged_per_chunk_paths_error_rather_than_misalign():
    raw = {"a": "answer", "docs": [{"t": "one", "s": 0.5}, {"t": "two"}]}
    with pytest.raises(IngestError) as exc:
        load_any(raw, field_map={"answer": "a", "chunks": "docs[].t", "chunk_scores": "docs[].s"})
    message = str(exc.value)
    assert "chunk_scores" in message and "1 values" in message and "2" in message


def test_unrecognized_payload_raises_unknown_source():
    with pytest.raises(UnknownSourceError) as exc:
        load_any({"totally": "unrelated", "shape": [1, 2, 3]})
    message = str(exc.value)
    assert "no ingest adapter recognized" in message
    assert "field_map" in message                      # tells the user the way out


def test_malformed_paths_are_rejected():
    with pytest.raises(IngestError, match="unterminated"):
        resolve_path({"a": 1}, "a[0")
    with pytest.raises(IngestError, match="empty field path"):
        resolve_path({"a": 1}, "")


# --------------------------------------------------------------------------- #
# 7. Path resolution
# --------------------------------------------------------------------------- #
def test_resolve_path_forms():
    obj = {
        "a": {"b": [{"c": 1}, {"c": 2}]},
        "list": [10, 20, 30],
        "dotted": {"gen_ai.request.model": "m1"},
    }
    assert resolve_path(obj, "a.b[1].c") == 2
    assert resolve_path(obj, "a.b[].c") == [1, 2]
    assert resolve_path(obj, "a.b[*].c") == [1, 2]
    assert resolve_path(obj, "list[-1]") == 30
    assert resolve_path(obj, "dotted['gen_ai.request.model']") == "m1"
    # Absent paths resolve to the default, never a KeyError/IndexError.
    assert resolve_path(obj, "a.missing.deep") is None
    assert resolve_path(obj, "list[9]", default="fallback") == "fallback"
    # A wildcard skips elements that lack the key rather than padding with None.
    assert resolve_path({"xs": [{"k": 1}, {}, {"k": 3}]}, "xs[].k") == [1, 3]


# --------------------------------------------------------------------------- #
# 8. load_traces: .json array and .jsonl
# --------------------------------------------------------------------------- #
def _two_payloads():
    return [payload("langchain.json"), payload("llamaindex.json")]


def test_load_traces_json_array(tmp_path):
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(_two_payloads()))
    traces = list(load_traces(path))
    assert [t.meta["source"] for t in traces] == ["langchain", "llamaindex"]
    assert traces[0].id == "lc-run-7f2a"                 # id from the payload wins
    assert traces[1].id == "traces.json#1"               # positional fallback
    assert traces[1].question == "What is the escalation path for a delayed shipment?"


def test_load_traces_jsonl(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in _two_payloads()) + "\n\n")
    traces = list(load_traces(path))
    assert [t.meta["source"] for t in traces] == ["langchain", "llamaindex"]
    assert traces[1].id == "traces.jsonl#1"
    assert [len(t.retrieved_context) for t in traces] == [3, 3]


def test_load_traces_single_object(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps(payload("otel_events.json")))
    traces = list(load_traces(path))
    assert len(traces) == 1 and traces[0].meta["source"] == "otel"


def test_load_traces_json_extension_holding_jsonl(tmp_path):
    path = tmp_path / "mislabelled.json"
    path.write_text("\n".join(json.dumps(p) for p in _two_payloads()))
    assert [t.meta["source"] for t in load_traces(path)] == ["langchain", "llamaindex"]


def test_load_traces_with_field_map(tmp_path):
    rows = [{"q": "Q1?", "a": "A1", "d": ["c1"]}, {"q": "Q2?", "a": "A2", "d": ["c2"]}]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    traces = list(load_traces(path, field_map={"question": "q", "answer": "a", "chunks": "d[]"}))
    assert [t.question for t in traces] == ["Q1?", "Q2?"]
    assert [t.retrieved_context[0].text for t in traces] == ["c1", "c2"]


def test_load_traces_reports_the_bad_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(payload("langchain.json")) + "\n{not json}\n")
    with pytest.raises(IngestError) as exc:
        list(load_traces(path))
    assert "broken.jsonl:2" in str(exc.value)


def test_load_traces_missing_file():
    with pytest.raises(IngestError, match="trace file not found"):
        list(load_traces("/nonexistent/path/to/traces.jsonl"))


# --------------------------------------------------------------------------- #
# 9. Serialization round-trip (ingested traces must survive the trace cache)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", sorted(EXPECTED_SOURCE))
def test_ingested_inference_survives_json_serde(fixture):
    from tokentrace.core.serialize import inference_from_dict, inference_to_dict

    inf = inference(fixture)
    # Strict JSON: no NaN/Infinity tokens may leak in from an unlogged score.
    text = json.dumps(inference_to_dict(inf), allow_nan=False)
    back = inference_from_dict(json.loads(text))
    assert back.prompt == inf.prompt
    assert back.generated_answer == inf.generated_answer
    assert back.question == inf.question
    assert back.ground_truth == inf.ground_truth
    assert (back.retrieved_context is None) == (inf.retrieved_context is None)
    if inf.retrieved_context is not None:
        assert [c.text for c in back.retrieved_context] == [c.text for c in inf.retrieved_context]
        assert [c.gold for c in back.retrieved_context] == [c.gold for c in inf.retrieved_context]
