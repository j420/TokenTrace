"""Command-line surface.

There were no CLI tests before this file, which is precisely how ``tokentrace
ablate`` shipped broken: ``run_ablations`` was refactored to return
``per_family_retrained``/``per_family_robustness``, ``test_benchmark`` was updated
to assert the new keys, and nothing checked that the *printer* in ``cli.py`` had
followed. The producer and the consumer were each tested; the seam between them
was not.

So the tests here cover the seam. The triage command is exercised for real
(cold-start, tiny input, seconds), while the expensive research commands have
their printers driven against a fixture shaped like the real return value — the
producers' own schemas are asserted in ``test_benchmark.py``, and the two
together are what would have caught the regression.
"""

from __future__ import annotations

import json

import pytest

from tokentrace.cli import cmd_ablate, main
from tokentrace.core.serialize import inference_to_dict

_ARGS = ("model", "backend", "tier", "seeds", "json", "fast", "top", "traces", "out")


class _Args:
    def __init__(self, **kw):
        for k in _ARGS:
            setattr(self, k, None)
        self.model, self.backend, self.tier = "mock-4b", "mock", "white"
        for k, v in kw.items():
            setattr(self, k, v)


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
    assert main(["triage", str(trace_log), "--fast"]) == 0
    out = capsys.readouterr().out
    assert "top clusters" in out
    # The three buckets must be reported separately, never merged into one number.
    for word in ("diagnosed", "healthy", "declined"):
        assert word in out, out


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
    assert main(["triage", str(trace_log), "--fast", "--top", "1"]) == 0
    out = capsys.readouterr().out
    ranks = [ln for ln in out.splitlines() if ln.strip().startswith(("1 ", "2 "))]
    assert not any(ln.strip().startswith("2 ") for ln in ranks), out


# --------------------------------------------------------------------------- #
# the seam that broke: printers vs producer schemas
def _ablation_fixture() -> dict:
    """Shaped like ``run_ablations`` output. ``test_benchmark`` asserts the real
    thing has these keys; this asserts the printer reads only these keys."""
    met = {"diagnosis_accuracy": 0.9, "top3_accuracy": 0.95, "conformal_coverage": 1.0,
           "conformal_set_size": 1.2, "abstention_rate": 0.1,
           "debugging_time_reduction": 0.6, "recommendation_precision": 0.8}
    fam = {f: dict(met) for f in ("prompt", "retrieval", "mechanistic", "confidence")}
    return {
        "sizes": {"train": 1, "cal": 1, "test": 1},
        "ablation_learned_head": {"rules_only": dict(met), "rules_plus_gbt": dict(met)},
        "per_tier": {t: dict(met) for t in ("white", "grey", "black")},
        "per_family_retrained": fam,
        "per_family_robustness": dict(fam),
        "calibration": {"ece_uncalibrated": 0.01, "ece_calibrated": 0.005,
                        "interior_mass_fraction": 0.01},
        "conformal": {t: {"coverage": 1.0, "mean_set_size": 1.2}
                      for t in ("white", "grey", "black")},
    }


def test_ablate_printer_reads_only_keys_the_producer_emits(monkeypatch, capsys):
    """Regression: cmd_ablate read `per_family_dropped`, a key run_ablations had
    stopped emitting, and died with a KeyError mid-report."""
    import tokentrace.eval.ablations as ab

    monkeypatch.setattr(ab, "run_ablations", lambda *a, **k: _ablation_fixture())
    monkeypatch.setattr(ab, "run_robustness", lambda *a, **k: {})
    assert cmd_ablate(_Args(seeds="0", json=False)) == 0
    out = capsys.readouterr().out
    # Both family ablations must be shown; they answer different questions and
    # collapsing them is what made the published claim an artifact.
    assert "retrained" in out and "robustness" in out, out


def test_every_documented_subcommand_is_registered():
    import argparse

    for cmd in ("demo", "analyze", "triage", "inject", "eval", "ablate"):
        with pytest.raises((SystemExit, argparse.ArgumentError)) as exc:
            main([cmd, "--definitely-not-a-flag"])
        # exit code 2 == argparse rejected the flag, i.e. the subcommand exists.
        assert exc.value.code == 2, f"{cmd} is not a registered subcommand"
