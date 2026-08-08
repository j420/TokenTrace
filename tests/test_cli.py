"""Command-line surface.

There were no CLI tests before this file, which is precisely how ``tokentrace
ablate`` shipped broken: ``run_ablations`` was refactored to return
``per_family_retrained``/``per_family_robustness``, ``test_benchmark`` was updated
to assert the new keys, and nothing checked that the *printer* in ``cli.py`` had
followed. The producer and the consumer were each tested; the seam between them
was not.

The first version of this file did not fix that. It asserted that the strings
``"retrained"`` and ``"robustness"`` appeared in the output — but those are
hardcoded literals in ``cli.py``, so the assertion held while the robustness block
printed the *retrained* numbers, and held again when the block was deleted
outright. A CLI test that can only see the printer's own literals is testing
nothing.

So every printer test here asserts on values that came from the DATA: the fixtures
give each metric a value that is unique across fields and across blocks, and the
tests compare the numbers on a row against the fixture. Reading the wrong key
raises, reading the right key from the wrong block prints a number no assertion
accepts, and dropping a section makes its lookup fail. The mutations that used to
survive — a typo'd key in the robustness printer, ``per_family_robustness``
printing the retrained data, deleting the sweep — are each covered by a named test
below.

The triage/ingest/error paths run for real (cold-start, tiny inputs, seconds); the
expensive research commands have their printers driven against fixtures shaped like
the real return values, whose schemas ``test_benchmark.py`` asserts against the
producers. The two together are what would have caught the regression.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from tokentrace.cli import cmd_ablate, cmd_eval, cmd_noref, main
from tokentrace.core.serialize import inference_to_dict
from tokentrace.core.types import (
    Chunk,
    Diagnosis,
    DiagnosisReport,
    DiagnosisRole,
    EvidenceItem,
    FailureMode,
    Inference,
    Recommendation,
    SignalFamily,
    Tier,
)

_ARGS = ("model", "backend", "tier", "seeds", "json", "fast", "top", "traces", "out",
         "trace", "source", "field_map")


class _Args:
    def __init__(self, **kw):
        for k in _ARGS:
            setattr(self, k, None)
        self.model, self.backend, self.tier = "mock-4b", "mock", "white"
        self.source, self.top = "auto", 10
        for k, v in kw.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------- #
# helpers: read the OUTPUT the way a user does, then compare it to the fixture
# --------------------------------------------------------------------------- #
def _after(out: str, marker: str) -> str:
    """Everything printed after ``marker``. Fails loudly if the section is gone."""
    assert marker in out, f"missing section {marker!r} in:\n{out}"
    return out.split(marker, 1)[1]


def _row(out: str, label: str) -> list[float]:
    """The numbers on the first row whose text starts with ``label``."""
    for line in out.splitlines():
        if line.strip().startswith(label):
            return [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
    raise AssertionError(f"no row starting with {label!r} in:\n{out}")


def _round(*values: float) -> list[float]:
    return [round(v, 3) for v in values]


@pytest.fixture(scope="module")
def trace_log(tmp_path_factory):
    """A small JSONL log shaped exactly like a captured production log."""
    from tokentrace.cli import _canonical

    path = tmp_path_factory.mktemp("cli") / "log.jsonl"
    with open(path, "w") as f:
        for _, inf in _canonical():
            f.write(json.dumps(inference_to_dict(inf)) + "\n")
    return path


# --------------------------------------------------------------------------- #
# triage
def test_triage_renders_a_ranked_report(trace_log, capsys):
    """The bucket COUNTS on the text row must be the ones the data produced.

    The first version asserted only that the words "diagnosed"/"healthy"/"declined"
    appeared — but those are literals in ``render_text``'s format string, so merging
    two buckets into one number left the assertion green. The counts are therefore
    read from the ``--json`` view of the SAME log and compared against the text row.
    """
    assert main(["triage", str(trace_log), "--fast", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    diagnosed, healthy, declined = (payload["n_diagnosed"], payload["n_healthy"],
                                    payload["n_abstained"])
    # Precondition: the canonical 5-trace log must populate more than one bucket,
    # or a merged-buckets regression would be invisible to the row assert below.
    assert diagnosed + healthy + declined == payload["n_analyzed"] == 5, payload
    assert diagnosed > 0 and healthy > 0, payload

    assert main(["triage", str(trace_log), "--fast"]) == 0
    out = capsys.readouterr().out
    assert "top clusters" in out
    # The counts must come from the input, not from a constant.
    assert "triage: 5 traces  analyzed=5" in out, out
    # Each bucket must carry ITS OWN count — merging any two moves a number here.
    assert f"diagnosed {diagnosed}  healthy {healthy}  declined {declined}" in out, out


def test_triage_json_is_strict_json(trace_log, capsys):
    assert main(["triage", str(trace_log), "--fast", "--json"]) == 0
    # json.loads is strict by default: NaN/Infinity would raise here, and the
    # engine does emit non-finite values that must be scrubbed before output.
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_analyzed"] == 5
    assert isinstance(payload["clusters"], list)


def test_triage_accepts_a_json_array_as_well_as_jsonl(trace_log, tmp_path, capsys):
    rows = [json.loads(line) for line in open(trace_log)]
    arr = tmp_path / "arr.json"
    arr.write_text(json.dumps(rows))
    assert main(["triage", str(arr), "--fast"]) == 0
    assert "analyzed=5" in capsys.readouterr().out


def test_triage_reports_a_malformed_line_by_number(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "x"}\n')
    with pytest.raises(SystemExit) as exc:
        main(["triage", str(bad), "--fast"])
    msg = str(exc.value)
    assert "bad.jsonl:1" in msg, msg          # names the file AND the line
    assert "generated_answer" in msg, msg     # names the field that is missing


def test_triage_top_limits_the_cluster_table(trace_log, capsys):
    """``--top`` must bound the table by the DATA's cluster count, not by luck."""
    assert main(["triage", str(trace_log), "--fast", "--json"]) == 0
    n_clusters = len(json.loads(capsys.readouterr().out)["clusters"])
    for top in (1, 10):
        assert main(["triage", str(trace_log), "--fast", "--top", str(top)]) == 0
        table = _after(capsys.readouterr().out, "top clusters")
        ranks = re.findall(r"^\s{2,3}(\d+)\s+\d+\s+\d", table, flags=re.M)
        assert len(ranks) == min(top, n_clusters), table


@pytest.mark.parametrize("top", ["0", "-1"])
def test_triage_rejects_a_top_that_would_silently_drop_clusters(trace_log, top):
    """``--top -1`` sliced ``clusters[:-1]`` and quietly dropped the last cluster;
    ``--top 0`` printed a header over an empty table."""
    with pytest.raises(SystemExit) as exc:
        main(["triage", str(trace_log), "--fast", "--top", top])
    assert "--top" in str(exc.value)


# --------------------------------------------------------------------------- #
# reading logs TokenTrace did not write (tokentrace.ingest <-> the CLI)
def _fixture_payload(name: str) -> dict:
    from pathlib import Path

    root = Path(__file__).parent / "fixtures" / "ingest"
    return json.loads((root / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def production_log(tmp_path_factory):
    """One log holding four different vendors' shapes, as an aggregator would."""
    path = tmp_path_factory.mktemp("ingest") / "prod.jsonl"
    with open(path, "w") as f:
        for name in ("openai_chat", "langchain", "llamaindex", "otel_flat"):
            f.write(json.dumps(_fixture_payload(name)) + "\n")
    return path


def test_triage_reads_a_real_production_log(production_log, capsys):
    """The README advertises `triage(load_traces(...))` in Python and `tokentrace
    triage log.jsonl` eight lines apart; the CLI parsed only TokenTrace's own dicts,
    so every one of these rows died with `not a valid trace — 'prompt'`."""
    assert main(["triage", str(production_log), "--fast"]) == 0
    out = capsys.readouterr().out
    assert "triage: 4 traces  analyzed=4  failed=0" in out, out


@pytest.mark.parametrize("name,source", [
    ("openai_chat", "openai_chat"),
    ("langchain", "langchain"),
    ("llamaindex", "llamaindex"),
    ("otel_flat", "otel"),
    ("generic", "generic"),
])
def test_analyze_reads_each_source_both_forced_and_auto_detected(name, source, tmp_path, capsys):
    payload = _fixture_payload(name)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    for argv in ([], ["--source", source]):
        assert main(["analyze", str(path), "--fast"] + argv) == 0
        out = capsys.readouterr().out
        # The answer printed must be the one in the payload, not a default.
        assert "  A: '" in out and "ranked diagnoses" in out, out


def test_analyze_maps_an_arbitrary_log_with_a_field_map(tmp_path, capsys):
    row = {"q": {"text": "Which port does the warehouse ship from?"},
           "out": {"text": "Rotterdam."},
           "docs": [{"body": "The warehouse ships from Rotterdam.", "rel": 0.9, "id": "d1"}],
           "expected": "Rotterdam"}
    path = tmp_path / "inhouse.json"
    path.write_text(json.dumps(row))
    field_map = {"question": "q.text", "answer": "out.text", "chunks": "docs[].body",
                 "chunk_scores": "docs[].rel", "chunk_source_ids": "docs[].id",
                 "ground_truth": "expected"}
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps(field_map))

    for spec in (json.dumps(field_map), str(map_file)):      # inline JSON or a path
        assert main(["analyze", str(path), "--fast", "--field-map", spec]) == 0
        out = capsys.readouterr().out
        assert "Q: Which port does the warehouse ship from?" in out, out
        assert "A: 'Rotterdam.'" in out, out
        assert "ground truth: ['Rotterdam']" in out, out     # the map's own key


def test_triage_round_trips_inject_output(tmp_path, capsys):
    """`tokentrace inject` writes LabeledInference rows and no ingest adapter claims
    them, so the CLI has to read its own output — this round trip was broken."""
    out_path = tmp_path / "corpus.jsonl"
    assert main(["inject", "--seeds", "0", "--out", str(out_path)]) == 0
    written = capsys.readouterr().out
    n_lines = sum(1 for line in open(out_path) if line.strip())
    summary = json.loads(written[:written.rindex("}") + 1])
    assert summary["n"] == n_lines, written                  # summary describes the file
    assert f"wrote {n_lines} labeled examples -> {out_path}" in written, written

    for argv in ([], ["--source", "tokentrace"]):            # auto must fall back
        assert main(["triage", str(out_path), "--fast", "--top", "3"] + argv) == 0
        text = capsys.readouterr().out
        assert f"triage: {n_lines} traces  analyzed={n_lines}  failed=0" in text, text
        # Traces with no id of their own get the identity ingest.load_traces assigns.
        assert "corpus.jsonl#" in text, text


def test_auto_prefers_our_own_shape_over_an_adapter_that_would_lose_the_chunks(tmp_path, capsys):
    """`langchain` claims a TokenTrace row (it reads `generated_answer`) but cannot
    see `retrieved_context`, so routing there would silently drop every chunk and
    hard-mask the retrieval modes."""
    from tokentrace.ingest import detect_source

    inf = Inference(prompt="Q?\nThe warehouse is in Rotterdam.",
                    generated_answer="Rotterdam.",
                    retrieved_context=[Chunk("The warehouse is in Rotterdam.", 0.9, "d1")],
                    question="Where is the warehouse?")
    payload = inference_to_dict(inf)
    assert detect_source(payload) == "langchain"             # the trap this guards

    path = tmp_path / "own.json"
    path.write_text(json.dumps(payload))
    assert main(["analyze", str(path), "--fast", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    modes = {d["mode"]: d["probability"] for d in report["diagnoses"]}
    # A non-RAG parse hard-masks these to exactly 0.0.
    assert modes["retrieval_failure"] > 0.0, report
    assert modes["context_dilution"] > 0.0, report


def test_a_wrong_source_names_the_adapter_that_does_recognize_the_row(tmp_path):
    path = tmp_path / "lc.json"
    path.write_text(json.dumps(_fixture_payload("langchain")))
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(path), "--fast", "--source", "otel"])
    assert "--source langchain" in str(exc.value), str(exc.value)


def test_field_map_and_a_conflicting_source_is_rejected(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(path), "--source", "otel", "--field-map", '{"answer": "a"}'])
    assert "--field-map" in str(exc.value)


# --------------------------------------------------------------------------- #
# user errors are messages, not stack traces
@pytest.mark.parametrize("argv,needle", [
    (["eval", "--seeds", "abc"], "not an integer"),
    (["eval", "--seeds", ""], "empty"),
    (["ablate", "--seeds", "0,,1"], "empty entry"),
    (["inject", "--seeds", "1.5"], "not an integer"),
    (["noref", "--seeds", "x,1"], "not an integer"),
])
def test_a_bad_seed_list_is_an_error_not_a_traceback(argv, needle):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert needle in str(exc.value), str(exc.value)
    assert exc.value.code != 0


def test_a_missing_file_names_the_file(tmp_path):
    for argv in (["analyze", str(tmp_path / "nope.json")], ["triage", str(tmp_path / "nope.jsonl")]):
        with pytest.raises(SystemExit) as exc:
            main(argv + ["--fast"])
        assert "nope" in str(exc.value) and "cannot read" in str(exc.value)


def test_a_jsonl_log_given_to_analyze_points_at_triage(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"prompt": "a", "generated_answer": "b"}\n'
                    '{"prompt": "c", "generated_answer": "d"}\n')
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(path), "--fast"])
    msg = str(exc.value)
    assert "not valid JSON" in msg and "triage" in msg, msg


def test_a_json_array_given_to_analyze_points_at_triage(tmp_path):
    path = tmp_path / "arr.json"
    path.write_text(json.dumps([{"prompt": "a", "generated_answer": "b"}]))
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(path), "--fast"])
    assert "triage" in str(exc.value)


def test_an_unwritable_output_path_is_an_error_not_a_traceback(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["inject", "--seeds", "0", "--out", str(tmp_path / "no" / "such" / "dir.jsonl")])
    assert "cannot write" in str(exc.value)


def test_an_unavailable_backend_is_an_error_not_a_traceback(monkeypatch):
    """`--backend hf` with no torch installed raised ImportError through argparse's
    caller. It reaches the user two different ways — the facade (demo/analyze/triage)
    and load_model (inject/eval/ablate) — and both were unguarded."""
    import tokentrace
    import tokentrace.models as models

    def boom(*a, **k):
        raise ImportError("HF backend needs torch+transformers. "
                          "Install: pip install 'tokentrace[mechanistic]'")

    monkeypatch.setattr(tokentrace, "TokenTrace", type("T", (), {"default": staticmethod(boom)}))
    monkeypatch.setattr(models, "load_model", boom)
    for argv in (["demo", "--fast"], ["eval", "--seeds", "0"], ["ablate", "--seeds", "0"]):
        with pytest.raises(SystemExit) as exc:
            main(argv + ["--backend", "hf"])
        msg = str(exc.value)
        assert "backend 'hf' is unavailable" in msg, msg
        assert "tokentrace[mechanistic]" in msg, msg          # keeps the fix instruction


def test_a_broken_field_map_is_an_error_not_a_traceback(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(path), "--field-map", "{oops}"])
    assert "--field-map" in str(exc.value) and "JSON" in str(exc.value)


# --------------------------------------------------------------------------- #
# --tier: honoured where it is offered, and not offered where it is meaningless
def _install_stub(monkeypatch, report):
    """Replace the TokenTrace facade so every printed value comes from ``report``."""
    import tokentrace

    seen: dict = {"default": [], "analyze": []}

    class _Stub:
        @staticmethod
        def default(**kw):
            seen["default"].append(kw)
            return _Stub()

        def analyze(self, inference, tier=None):
            seen["analyze"].append((inference, tier))
            return report

    monkeypatch.setattr(tokentrace, "TokenTrace", _Stub)
    return seen


@pytest.mark.parametrize("cmd", ["eval", "ablate", "noref"])
def test_tier_is_not_offered_where_it_would_be_ignored(cmd):
    """It used to sit in the shared parent parser: advertised by ``--help`` on both
    commands, and hard-coded to WHITE in both bodies. All of these sweep every tier
    they report and print the curve, so the flag had nothing to mean."""
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--tier", "black"])
    assert exc.value.code == 2


@pytest.mark.parametrize("cmd,extra", [
    ("demo", []),
    ("analyze", ["TRACE"]),
    ("triage", ["TRACE"]),
])
def test_model_and_backend_reach_the_facade(cmd, extra, tmp_path, monkeypatch, capsys):
    """``demo`` accepted ``--model``/``--backend`` from the shared parser and then
    built the facade with neither, so it always ran the mock."""
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(inference_to_dict(
        Inference(prompt="Q?", generated_answer="A", question="Q?"))))
    argv = [cmd] + [str(trace) if a == "TRACE" else a for a in extra]

    seen = _install_stub(monkeypatch, _report_fixture())
    assert main(argv + ["--fast", "--model", "qwen3-0.6b", "--backend", "mock"]) == 0
    capsys.readouterr()
    assert seen["default"][0]["model_name"] == "qwen3-0.6b", seen["default"]
    assert seen["default"][0]["backend"] == "mock", seen["default"]


@pytest.mark.parametrize("cmd,extra", [
    ("demo", []),
    ("analyze", ["TRACE"]),
    ("triage", ["TRACE"]),
])
def test_tier_is_honoured_where_it_is_offered(cmd, extra, tmp_path, monkeypatch, capsys):
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(inference_to_dict(
        Inference(prompt="Q?", generated_answer="A", question="Q?"))))
    argv = [cmd] + [str(trace) if a == "TRACE" else a for a in extra] + ["--fast", "--tier", "black"]

    seen = _install_stub(monkeypatch, _report_fixture())
    assert main(argv) == 0
    capsys.readouterr()
    assert seen["default"], "the facade was never built"
    assert seen["default"][0]["tier"] is Tier.BLACK, seen["default"]
    assert all(t is Tier.BLACK for _, t in seen["analyze"]), seen["analyze"]


# --------------------------------------------------------------------------- #
# printer: _print_report (demo / analyze)
def _report_fixture(abstained: bool = False) -> DiagnosisReport:
    prior = EvidenceItem(signal="prior", family=SignalFamily.RETRIEVAL, value=0.0,
                         contribution_logodds=-1.4, source="prior",
                         rendered="BOOKKEEPING-PRIOR-LINE")
    return DiagnosisReport(
        tier=Tier.GREY,                       # deliberately NOT the requested tier
        diagnostic_confidence=0.61,
        abstained=abstained,
        conformal_set=[FailureMode.RETRIEVAL_FAILURE, FailureMode.HALLUCINATION],
        notes=["NOTE-FROM-THE-REPORT"],
        diagnoses=[
            Diagnosis(mode=FailureMode.RETRIEVAL_FAILURE, probability=0.83, rank=1,
                      role=DiagnosisRole.PRIMARY_ROOT, confidence=0.37,
                      evidence=[prior,
                                EvidenceItem(signal="ret.gold_absent",
                                             family=SignalFamily.RETRIEVAL, value=0.05,
                                             contribution_logodds=2.4,
                                             rendered="HEADLINE-RETRIEVAL-LINE")],
                      recommendations=[Recommendation(
                          action="add_gold_context", description="d",
                          targets_mode=FailureMode.RETRIEVAL_FAILURE, validated=True)]),
            Diagnosis(mode=FailureMode.HALLUCINATION, probability=0.42, rank=2,
                      role=DiagnosisRole.SEQUELA, confidence=0.29,
                      causal_parents=[FailureMode.RETRIEVAL_FAILURE],
                      evidence=[EvidenceItem(signal="hal.override",
                                             family=SignalFamily.CONFIDENCE, value=0.9,
                                             contribution_logodds=1.1,
                                             rendered="HEADLINE-HALLUCINATION-LINE")],
                      recommendations=[Recommendation(
                          action="ground_or_abstain", description="d",
                          targets_mode=FailureMode.HALLUCINATION, validated=False)]),
            Diagnosis(mode=FailureMode.CONTEXT_DILUTION, probability=0.10, rank=3),
        ],
    )


@pytest.fixture
def one_trace(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(inference_to_dict(Inference(
        prompt="PROMPT-TEXT", generated_answer="ANSWER-FROM-THE-TRACE",
        question="QUESTION-FROM-THE-TRACE", ground_truth=["GT-FROM-THE-TRACE"]))))
    return path


def test_analyze_prints_the_report_it_was_given(one_trace, monkeypatch, capsys):
    """Every line of ``_print_report`` must be traceable to the report/inference —
    the whole printer was mutable with a green suite before this."""
    _install_stub(monkeypatch, _report_fixture())
    assert main(["analyze", str(one_trace), "--fast"]) == 0
    out = capsys.readouterr().out

    assert "Q: QUESTION-FROM-THE-TRACE" in out, out          # inf.query, not the prompt
    assert "A: 'ANSWER-FROM-THE-TRACE'" in out, out
    assert "ground truth: ['GT-FROM-THE-TRACE']" in out, out
    assert "tier=grey" in out, out                           # report.tier, not --tier
    assert "diagnostic_confidence=0.61" in out, out
    assert "[ABSTAINED]" not in out, out

    assert _row(out, "0.83") == [0.83], out                  # probability, not confidence
    assert "0.37" not in out and "0.29" not in out, out      # per-diagnosis meta-confidence
    assert "Retrieval Failure  [primary_root]" in out, out
    assert "[sequela] <- Retrieval Failure" in out, out
    assert "Context Dilution" not in out, out                # p=0.10 is below the floor

    assert "HEADLINE-RETRIEVAL-LINE" in out, out
    assert "HEADLINE-HALLUCINATION-LINE" in out, out
    assert "BOOKKEEPING-PRIOR-LINE" not in out, out          # headline_evidence, not [0]
    assert "fix: add_gold_context ✓ validated" in out, out
    assert "fix: ground_or_abstain ✗ no effect" in out, out
    assert "Top-k set: ['Retrieval Failure', 'Hallucination']" in out, out
    assert "⚠ NOTE-FROM-THE-REPORT" in out, out


def test_analyze_marks_an_abstention(one_trace, monkeypatch, capsys):
    _install_stub(monkeypatch, _report_fixture(abstained=True))
    assert main(["analyze", str(one_trace), "--fast"]) == 0
    assert "[ABSTAINED]" in capsys.readouterr().out


def test_analyze_json_is_strict_and_carries_the_report(one_trace, monkeypatch, capsys):
    _install_stub(monkeypatch, _report_fixture())
    assert main(["analyze", str(one_trace), "--fast", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)             # strict: NaN would raise
    assert payload["tier"] == "grey"
    assert payload["diagnostic_confidence"] == 0.61
    assert [d["probability"] for d in payload["diagnoses"]] == [0.83, 0.42, 0.10]


def test_demo_prints_one_report_per_canonical_scenario(monkeypatch, capsys):
    from tokentrace.cli import _canonical

    _install_stub(monkeypatch, _report_fixture())
    assert main(["demo", "--fast"]) == 0
    out = capsys.readouterr().out
    names = [name for name, _ in _canonical()]
    assert f"Diagnosing {len(names)} canonical scenarios" in out, out
    for name in names:
        assert f"=== {name} ===" in out, out
    assert out.count("diagnostic_confidence=0.61") == len(names), out


# --------------------------------------------------------------------------- #
# printer: cmd_eval
_METRIC_KEYS = ("diagnosis_accuracy", "top3_accuracy", "conformal_coverage",
                "conformal_set_size", "abstention_rate", "debugging_time_reduction",
                "recommendation_precision")


def _metrics(tag: int) -> dict:
    """One metric block whose every value is unique — across FIELDS and across
    BLOCKS. That is what makes "printer read the wrong key" and "printer read the
    right key from the wrong block" both visible in the output."""
    return {k: round(tag / 10 + i / 1000, 4) for i, k in enumerate(_METRIC_KEYS)}


def _eval_fixture() -> dict:
    """Shaped like ``train_and_evaluate`` output (see ``test_benchmark``)."""
    return {"sizes": {"train": 61, "cal": 62, "test": 63},
            "tiers": {"white": _metrics(1), "grey": _metrics(2), "black": _metrics(3)},
            "_engine": object()}


def _install_eval(monkeypatch, res: dict) -> None:
    import tokentrace.data.synthetic as syn
    import tokentrace.eval.benchmark as bench

    monkeypatch.setattr(syn, "build_dataset", lambda *a, **k: [])
    monkeypatch.setattr(bench, "train_and_evaluate", lambda *a, **k: res)


def test_eval_prints_the_metrics_it_was_given(monkeypatch, capsys):
    """The whole of ``cmd_eval``'s printer was mutable with a green suite."""
    _install_eval(monkeypatch, _eval_fixture())
    assert cmd_eval(_Args(seeds="0", json=False)) == 0
    out = capsys.readouterr().out

    assert "dataset sizes: {'train': 61, 'cal': 62, 'test': 63}" in out, out
    for tier, tag in (("white", 1), ("grey", 2), ("black", 3)):
        m = _metrics(tag)
        assert _row(out, tier) == _round(m["diagnosis_accuracy"], m["top3_accuracy"],
                                         m["recommendation_precision"],
                                         m["abstention_rate"]), out
    assert "targets: diagnosis>=0.80" in out, out


def test_eval_json_is_strict_json_even_with_non_finite_metrics(monkeypatch, capsys):
    """``eval --json`` called ``json.dumps`` with no scrub and no ``allow_nan=False``,
    so a degenerate run emitted a bare ``NaN`` token — not RFC 8259, and unreadable
    by every strict consumer."""
    res = _eval_fixture()
    res["tiers"]["white"]["recommendation_precision"] = float("nan")
    res["tiers"]["grey"]["conformal_set_size"] = float("inf")
    _install_eval(monkeypatch, res)
    assert cmd_eval(_Args(seeds="0", json=True)) == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw and "Infinity" not in raw, raw
    payload = json.loads(raw)                                  # strict by default
    assert payload["tiers"]["white"]["recommendation_precision"] is None
    assert payload["tiers"]["grey"]["conformal_set_size"] is None
    assert payload["tiers"]["black"]["diagnosis_accuracy"] == _metrics(3)["diagnosis_accuracy"]
    assert "_engine" not in payload


# --------------------------------------------------------------------------- #
# printer: cmd_ablate — the seam that actually broke
def _ablation_fixture() -> dict:
    """Shaped like ``run_ablations`` output. ``test_benchmark`` asserts the real
    thing has these keys; this asserts the printer reads only these keys — and the
    per-block tags assert it reads them from the right block."""
    families = ("prompt", "retrieval", "mechanistic", "confidence")
    return {
        "sizes": {"train": 71, "cal": 72, "test": 73},
        "ablation_learned_head": {"rules_only": _metrics(1), "rules_plus_gbt": _metrics(2)},
        "per_tier": {t: _metrics(3 + i) for i, t in enumerate(("white", "grey", "black"))},
        "per_family_retrained": {f: _metrics(10 + i) for i, f in enumerate(families)},
        "per_family_robustness": {f: _metrics(20 + i) for i, f in enumerate(families)},
        "calibration": {"ece_uncalibrated": 0.4321, "ece_calibrated": 0.8765,
                        "interior_mass_fraction": 0.5432},
        "conformal": {t: {"coverage": 1.0, "mean_set_size": 1.2}
                      for t in ("white", "grey", "black")},
    }


def _robustness_fixture() -> dict:
    """Shaped like ``run_robustness`` output, one row per noise level."""
    return {f"{nz:.2f}": {"diagnosis_accuracy": 0.9 - i / 100,
                          "abstention_rate": 0.11 + i / 100,
                          "debugging_time_reduction": 0.31 + i / 100,
                          "ece_uncalibrated": 0.0101 + i / 10000,
                          "ece_calibrated": 0.0505 + i / 10000,
                          "top3_accuracy": 0.77, "conformal_set_size": 1.5,
                          "mean_root_rank": 2.5}
            for i, nz in enumerate((0.0, 0.25, 0.5, 0.75))}


def _install_ablate(monkeypatch, res=None, robustness=None) -> dict:
    import tokentrace.eval.ablations as ab

    seen: dict = {}

    def fake_ablations(model, **kw):
        seen["ablations"] = {"model": model, **kw}
        return _ablation_fixture() if res is None else res

    def fake_robustness(**kw):
        seen["robustness"] = kw
        return _robustness_fixture() if robustness is None else robustness

    monkeypatch.setattr(ab, "run_ablations", fake_ablations)
    monkeypatch.setattr(ab, "run_robustness", fake_robustness)
    return seen


def test_ablate_prints_every_block_from_its_own_data(monkeypatch, capsys):
    """Regression: ``cmd_ablate`` read ``per_family_dropped``, a key ``run_ablations``
    had stopped emitting, and died with a KeyError mid-report."""
    _install_ablate(monkeypatch)
    assert cmd_ablate(_Args(seeds="0", json=False)) == 0
    out = capsys.readouterr().out

    assert "sizes: {'train': 71, 'cal': 72, 'test': 73}" in out, out
    head = _after(out, "learned-head ablation")
    for key, tag in (("rules_only", 1), ("rules_plus_gbt", 2)):
        m = _metrics(tag)
        assert _row(head, key) == _round(m["diagnosis_accuracy"], m["top3_accuracy"]), out

    tiers = _after(out, "per-tier:")
    for i, tier in enumerate(("white", "grey", "black")):
        m = _metrics(3 + i)
        assert _row(tiers, tier) == _round(m["diagnosis_accuracy"], m["top3_accuracy"],
                                           m["conformal_coverage"], m["abstention_rate"],
                                           m["debugging_time_reduction"]), out


def test_ablate_never_conflates_the_two_family_ablations(monkeypatch, capsys):
    """They answer different questions — the family's INFORMATION contribution vs how
    the SHIPPED engine copes — and collapsing them is what made the published
    "which family is load-bearing" claim an artifact. The old test asserted only that
    the words "retrained" and "robustness" appeared, which are literals in the
    printer: swapping the robustness block's data source for the retrained one
    survived it."""
    _install_ablate(monkeypatch)
    assert cmd_ablate(_Args(seeds="0", json=False)) == 0
    out = capsys.readouterr().out

    retrained = _after(out, "(a) retrained leave-one-out")
    robustness = _after(out, "(b) missing-signal robustness")
    for i, family in enumerate(("prompt", "retrieval", "mechanistic", "confidence")):
        assert _row(retrained, f"drop {family}") == _round(
            _metrics(10 + i)["diagnosis_accuracy"]), out
        assert _row(robustness, f"drop {family}") == _round(
            _metrics(20 + i)["diagnosis_accuracy"]), out


def test_ablate_prints_the_robustness_sweep(monkeypatch, capsys):
    """Deleting this entire section, or typo'ing a key inside its loop, used to leave
    the suite green: the old test stubbed ``run_robustness`` to ``{}``, so the loop
    body never ran."""
    _install_ablate(monkeypatch)
    assert cmd_ablate(_Args(seeds="0", json=False)) == 0
    sweep = _after(capsys.readouterr().out, "robustness under signal-observation noise")

    for level, m in _robustness_fixture().items():
        assert _row(sweep, level) == [float(level)] + _round(
            m["diagnosis_accuracy"], m["abstention_rate"],
            m["debugging_time_reduction"]) + [round(m["ece_uncalibrated"], 4),
                                              round(m["ece_calibrated"], 4)], sweep


def test_ablate_prints_calibration_without_promising_an_outcome(monkeypatch, capsys):
    """The sweep header used to read "(calibration should reduce ECE as noise grows)"
    over a table whose calibrated ECE was WORSE at three of four noise levels. A
    printed expectation reads as a finding."""
    _install_ablate(monkeypatch)
    assert cmd_ablate(_Args(seeds="0", json=False)) == 0
    out = capsys.readouterr().out
    assert "should reduce" not in out, out
    assert "uncalibrated=0.4321" in out and "calibrated=0.8765" in out, out
    assert "interior_mass=0.543" in out, out
    assert "saturated" not in out, out            # interior mass is high in this fixture


def test_ablate_flags_saturated_probabilities_when_the_data_says_so(monkeypatch, capsys):
    res = _ablation_fixture()
    res["calibration"]["interior_mass_fraction"] = 0.0
    _install_ablate(monkeypatch, res=res)
    assert cmd_ablate(_Args(seeds="0", json=False)) == 0
    assert "saturated" in capsys.readouterr().out


def test_ablate_json_carries_the_robustness_sweep(monkeypatch, capsys):
    """``--json`` returned before the sweep ran, so a consumer scripting it lost the
    whole section with no error at all."""
    _install_ablate(monkeypatch)
    assert cmd_ablate(_Args(seeds="0", json=True)) == 0
    payload = json.loads(capsys.readouterr().out)             # strict by default

    assert set(payload["robustness"]) == set(_robustness_fixture())
    assert payload["robustness"]["0.75"]["ece_calibrated"] == \
        _robustness_fixture()["0.75"]["ece_calibrated"]
    assert payload["per_family_retrained"]["prompt"] == _metrics(10)
    assert payload["per_family_robustness"]["prompt"] == _metrics(20)
    assert payload["noise_levels"] == [0.0, 0.25, 0.5, 0.75]


def test_ablate_json_is_strict_json_even_with_non_finite_metrics(monkeypatch, capsys):
    res = _ablation_fixture()
    res["per_tier"]["black"]["conformal_coverage"] = float("nan")
    rob = _robustness_fixture()
    rob["0.75"]["ece_calibrated"] = float("-inf")
    _install_ablate(monkeypatch, res=res, robustness=rob)
    assert cmd_ablate(_Args(seeds="0", json=True)) == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw and "Infinity" not in raw, raw
    payload = json.loads(raw)
    assert payload["per_tier"]["black"]["conformal_coverage"] is None
    assert payload["robustness"]["0.75"]["ece_calibrated"] is None


def test_ablate_runs_the_sweep_on_the_same_handle_as_the_ablations(monkeypatch, capsys):
    """``run_robustness`` hard-coded ``load_model("mock-4b", backend="mock")``, so
    ``ablate --backend hf --model qwen3-4b`` ran the ablations on the real handle and
    the sweep on the mock, and printed both under one report."""
    seen = _install_ablate(monkeypatch)
    assert cmd_ablate(_Args(seeds="0,2", json=False, model="qwen3-4b")) == 0
    out = capsys.readouterr().out

    assert seen["robustness"]["model"] is seen["ablations"]["model"]
    assert seen["robustness"]["seeds"] == seen["ablations"]["seeds"] == (0, 2)
    assert seen["ablations"]["model"].profile.name == "qwen3-4b"
    # ...and the report says which handle produced it.
    assert "model: qwen3-4b (backend=mock, tier=white)" in out, out


def test_run_robustness_stays_backward_compatible_for_regen_report():
    """``scripts/regen_report.py`` calls ``run_robustness(noise_levels=..., seeds=...)``.
    The new ``model`` parameter must be optional and last, and must default to the
    handle the function used to build for itself."""
    from tokentrace.eval.ablations import run_robustness

    params = list(inspect.signature(run_robustness).parameters.values())
    assert [p.name for p in params] == ["noise_levels", "seeds", "pipeline", "model"]
    assert all(p.default is not inspect.Parameter.empty for p in params)
    assert params[-1].default is None


def test_run_robustness_default_handle_matches_an_explicit_mock_handle():
    """The default is not just *a* model: it must reproduce today's exact numbers."""
    from tokentrace.eval.ablations import run_robustness
    from tokentrace.models import load_model

    implicit = run_robustness(noise_levels=(0.25,), seeds=(0,))
    explicit = run_robustness(noise_levels=(0.25,), seeds=(0,),
                              model=load_model("mock-4b", backend="mock"))
    assert implicit == explicit


# --------------------------------------------------------------------------- #
# printer: cmd_noref — the production-mode measurement, fixture-driven per the
# module docstring: every value unique across fields, modes, conditions and blocks.
_NOREF_KEYS = ("diagnosis_accuracy", "top3_accuracy", "conformal_coverage",
               "conformal_set_size", "declined_rate", "debugging_time_reduction",
               "mean_root_rank", "recommendation_precision")

_MODES = ("prompt_ambiguity", "retrieval_failure", "context_dilution",
          "hallucination", "reasoning_failure")


def _noref_metrics(tag: int, unavailable: bool = False) -> dict:
    """One condition block shaped like ``Metrics.as_dict()`` after ``run_noref``'s
    post-processing. ``unavailable=True`` gives it the UNAVAILABLE marker the real
    stripped conditions carry instead of a recommendation precision."""
    from tokentrace.eval.noref import UNAVAILABLE

    m = {k: round(tag / 10 + i / 1000, 4) for i, k in enumerate(_NOREF_KEYS)}
    if unavailable:
        m["recommendation_precision"] = UNAVAILABLE
    m["per_mode"] = {mode: {"precision": round(tag / 10 + j / 100 + 0.001, 3),
                            "recall": round(tag / 10 + j / 100 + 0.002, 3),
                            "f1": round(tag / 10 + j / 100, 2),
                            "support": 30 + 10 * tag + j}
                     for j, mode in enumerate(_MODES)}
    return m


def _noref_row(m: dict) -> list[float]:
    """The condition-table row the printer must produce for block ``m``."""
    row = _round(m["diagnosis_accuracy"], m["top3_accuracy"], m["conformal_coverage"])
    row += [round(m["conformal_set_size"], 2)]
    row += _round(m["declined_rate"], m["debugging_time_reduction"])
    row += [round(m["mean_root_rank"], 2)]
    if isinstance(m["recommendation_precision"], float):
        row += _round(m["recommendation_precision"])
    return row


def _noref_fixture() -> dict:
    """Shaped like ``run_noref`` output (asserted against the producer by
    ``test_noref.py``); condition tags: baseline=1, transfer=2, frozen=3,
    retrained=4, and 5/6/7 for the ingest-shape block."""
    conds = {"baseline": _noref_metrics(1),
             "transfer": _noref_metrics(2, unavailable=True),
             "retrained": _noref_metrics(4, unavailable=True)}
    return {
        **conds,
        "sizes": {"train": 81, "cal": 82, "test": 83},
        "tier": "white",
        "frozen_mock": {"note": "n", "metrics": _noref_metrics(3, unavailable=True),
                        "confusion": {}, "delta_vs_baseline": {},
                        "delta_vs_transfer": {}, "available": True},
        "delta_vs_baseline": {"transfer": {}, "retrained": {}},
        "per_mode": {mode: {name: m["per_mode"][mode] for name, m in conds.items()}
                     for mode in _MODES},
        "confusion": {name: {} for name in conds},
        "retrieval_vs_dilution": {"prediction": "p", "held": True},
        "unavailable_metrics": {},
        "calibration_path": {name: {} for name in conds},
        "ece": {name: {"uncalibrated": {}, "calibrated": {}} for name in conds},
        "confound_controls": {"note": "n"},
        "ingest_shape": {
            "tier": "black", "note": "n",
            "baseline": _noref_metrics(5),
            "transfer": _noref_metrics(6, unavailable=True),
            "frozen_mock": _noref_metrics(7, unavailable=True),
            "delta_vs_same_tier_baseline": {}, "delta_vs_full_capture": {},
            "confusion": {}, "calibration_path": {},
        },
        "guard_cost": {"note": "n"},
        # run_noref is actively growing; the printer must pass unknown keys through.
        "a_block_the_printer_has_never_heard_of": {"x": 1.0},
    }


def _install_noref(monkeypatch, res=None) -> dict:
    import tokentrace.eval.noref as noref_mod

    seen: dict = {}

    def fake(**kw):        # keyword-only on purpose: a positional call must break here
        seen.update(kw)
        return _noref_fixture() if res is None else res

    monkeypatch.setattr(noref_mod, "run_noref", fake)
    return seen


def test_noref_prints_the_condition_table_from_each_conditions_own_data(monkeypatch, capsys):
    seen = _install_noref(monkeypatch)
    assert cmd_noref(_Args(seeds="0,2", json=False, model="qwen3-0.6b")) == 0
    out = capsys.readouterr().out

    # the producer got the parsed seeds and the --model/--backend handle, at white
    assert seen["seeds"] == (0, 2), seen
    assert seen["model"].profile.name == "qwen3-0.6b", seen
    assert seen["model"].tier is Tier.WHITE, seen

    assert "model: qwen3-0.6b (backend=mock, tier=white)" in out, out
    table = _after(out, "no-reference")
    assert "tier=white" in table and "test n=83" in table, out
    for name, tag in (("baseline", 1), ("transfer", 2), ("frozen-mock", 3),
                      ("retrained", 4)):
        assert _row(table, name) == _noref_row(_noref_metrics(
            tag, unavailable=(tag != 1))), out
    # UNAVAILABLE renders as a dash — never raw, never coerced to a number.
    assert "UNAVAILABLE" not in out, out


def test_noref_prints_the_per_mode_block_from_each_condition(monkeypatch, capsys):
    _install_noref(monkeypatch)
    assert cmd_noref(_Args(seeds="0", json=False)) == 0
    pm = _after(capsys.readouterr().out, "per-mode F1")
    for j, mode in enumerate(_MODES):
        # one F1 per condition, each from ITS block (tags 1..4), baseline support
        assert _row(pm, mode) == [round(t / 10 + j / 100, 2) for t in (1, 2, 3, 4)], pm
        line = next(ln for ln in pm.splitlines() if ln.strip().startswith(mode))
        assert line.split()[-1] == str(30 + 10 * 1 + j), pm


def test_noref_prints_the_ingest_shape_block_when_present(monkeypatch, capsys):
    _install_noref(monkeypatch)
    assert cmd_noref(_Args(seeds="0", json=False)) == 0
    ing = _after(capsys.readouterr().out, "ingest shape")
    assert "tier=black" in ing, ing
    for name, tag in (("baseline", 5), ("transfer", 6), ("frozen-mock", 7)):
        m = _noref_metrics(tag, unavailable=(tag != 5))
        assert _row(ing, name) == _round(
            m["diagnosis_accuracy"], m["top3_accuracy"], m["declined_rate"],
            m["debugging_time_reduction"]) + [
            m["per_mode"]["hallucination"]["f1"],
            m["per_mode"]["retrieval_failure"]["f1"],
            m["per_mode"]["context_dilution"]["f1"]], ing


def test_noref_tolerates_a_result_without_the_optional_blocks(monkeypatch, capsys):
    """A real backend has no frozen-mock control (`metrics` is None), and
    `ingest_shape`/`guard_cost` may be absent entirely; unknown extra keys are
    already in every fixture above. None of that may crash or invent a row."""
    res = _noref_fixture()
    res["frozen_mock"] = {"note": "n", "metrics": None, "confusion": {},
                          "delta_vs_baseline": None, "delta_vs_transfer": None,
                          "available": False}
    del res["ingest_shape"]
    del res["guard_cost"]
    _install_noref(monkeypatch, res=res)
    assert cmd_noref(_Args(seeds="0", json=False)) == 0
    out = capsys.readouterr().out
    assert "frozen-mock" not in out, out
    assert "ingest shape" not in out, out
    # ...and the required blocks still render from their own data.
    table = _after(out, "no-reference")
    for name, tag in (("baseline", 1), ("transfer", 2), ("retrained", 4)):
        assert _row(table, name) == _noref_row(_noref_metrics(
            tag, unavailable=(tag != 1))), out


def test_noref_json_is_strict_json_and_keeps_the_unavailable_marker(monkeypatch, capsys):
    from tokentrace.eval.noref import UNAVAILABLE

    res = _noref_fixture()
    res["baseline"]["conformal_coverage"] = float("nan")
    res["frozen_mock"]["metrics"]["mean_root_rank"] = float("inf")
    _install_noref(monkeypatch, res=res)
    assert cmd_noref(_Args(seeds="0", json=True)) == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw and "Infinity" not in raw, raw
    payload = json.loads(raw)                                  # strict by default
    assert payload["baseline"]["conformal_coverage"] is None
    assert payload["frozen_mock"]["metrics"]["mean_root_rank"] is None
    # "not measurable" must survive to the JSON consumer as the marker, not as 0.0
    assert payload["transfer"]["recommendation_precision"] == UNAVAILABLE
    assert payload["transfer"]["diagnosis_accuracy"] == \
        _noref_metrics(2)["diagnosis_accuracy"]
    assert payload["ingest_shape"]["transfer"]["top3_accuracy"] == \
        _noref_metrics(6)["top3_accuracy"]
    assert payload["model"] == "mock-4b" and payload["backend"] == "mock"


def test_noref_smoke_really_runs_the_benchmark(capsys):
    """The one test here that runs the real producer (single seed, mock backend,
    ~5-15 s); everything above drives the printer against fixtures."""
    assert main(["noref", "--seeds", "0"]) == 0
    out = capsys.readouterr().out
    assert "no-reference" in out, out
    for name in ("baseline", "transfer", "frozen-mock", "retrained"):
        assert len(_row(out, name)) >= 7, out       # a full row of real metrics
    assert "per-mode F1" in out and "ingest shape" in out, out


# --------------------------------------------------------------------------- #
# printer: cmd_inject
def test_inject_summary_describes_the_file_it_wrote(tmp_path, capsys):
    out_path = tmp_path / "ds.jsonl"
    assert main(["inject", "--seeds", "0", "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[:out.rindex("}") + 1])           # strict: NaN would raise

    rows = [json.loads(line) for line in open(out_path) if line.strip()]
    assert summary["n"] == len(rows), out
    assert sum(summary["by_recipe"].values()) == len(rows), out
    assert f"wrote {len(rows)} labeled examples -> {out_path}" in out, out
    # The summary must describe THESE rows, not a constant: recount the labels.
    modes = [m for r in rows for m in (r["labels"] or ["(none)"])]
    assert summary["by_mode"] == {m: modes.count(m) for m in dict.fromkeys(modes)}, out


# --------------------------------------------------------------------------- #
def test_every_documented_subcommand_is_registered():
    import argparse

    for cmd in ("demo", "analyze", "triage", "inject", "eval", "ablate", "noref"):
        with pytest.raises((SystemExit, argparse.ArgumentError)) as exc:
            main([cmd, "--definitely-not-a-flag"])
        # exit code 2 == argparse rejected the flag, i.e. the subcommand exists.
        assert exc.value.code == 2, f"{cmd} is not a registered subcommand"


def test_the_top_level_help_lists_every_subcommand(capsys):
    """``--help``'s description listed five commands and omitted ``ablate``."""
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    described = out.split("positional arguments")[0]
    for cmd in ("demo", "analyze", "triage", "inject", "eval", "ablate", "noref"):
        assert f"tokentrace {cmd}" in described, described
