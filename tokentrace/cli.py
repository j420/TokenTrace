"""TokenTrace command line.

    tokentrace demo                 # run the five canonical failure scenarios
    tokentrace analyze trace.json   # diagnose one inference (JSON, any logged shape)
    tokentrace triage traces.jsonl  # diagnose a whole log, ranked by cluster
    tokentrace inject --out d.jsonl # build the synthetic labeled corpus
    tokentrace eval                 # train + evaluate against the targets
    tokentrace ablate               # ablations + the observation-noise sweep
    tokentrace noref                # the no-reference (production-shape) benchmark

`analyze` and `triage` read TokenTrace's own trace JSON *and* real production logs:
`--source auto|openai_chat|langchain|llamaindex|otel|generic|tokentrace` routes
through `tokentrace.ingest`, and `--field-map` maps an arbitrary log shape.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator, NoReturn, Optional

from tokentrace.core.types import ALL_MODES, Chunk, Inference, Tier
from tokentrace.core.serialize import inference_from_dict, labeled_to_dict, report_to_dict

_TIER = {"black": Tier.BLACK, "grey": Tier.GREY, "white": Tier.WHITE}

#: Trace formats `analyze`/`triage` can read. Everything except "tokentrace" is an
#: adapter in :mod:`tokentrace.ingest`; "tokentrace" is this repo's own
#: `inference_to_dict` / `labeled_to_dict` output, which no ingest adapter claims.
_SOURCES = ("auto", "tokentrace", "openai_chat", "langchain", "llamaindex", "otel", "generic")

#: Noise levels for `ablate`'s robustness sweep (shared by the text and --json paths
#: so they cannot report different experiments).
_NOISE_LEVELS = (0.0, 0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- #
# User errors
# --------------------------------------------------------------------------- #
def _fail(msg: str) -> NoReturn:
    """Report a user error on stderr and exit non-zero — never a stack trace.

    A traceback out of `--seeds abc` or a missing file names the line of OUR code
    that raised, which is never the thing the user has to change. Everything a user
    can trigger by typing the wrong thing routes through here; genuine bugs are
    still allowed to raise.
    """
    raise SystemExit(f"tokentrace: error: {msg}")


def _reason(exc: BaseException) -> str:
    """One clause naming what was wrong with a payload."""
    if isinstance(exc, KeyError):
        return f"missing field {exc}"
    return str(exc) or type(exc).__name__


def _parse_seeds(raw: str) -> tuple[int, ...]:
    """`--seeds 0,1,2` -> `(0, 1, 2)`, or an actionable error."""
    parts = [s.strip() for s in str(raw).split(",")]
    if not any(parts):
        _fail(f"--seeds {raw!r} is empty — give a comma-separated list of integers, "
              "e.g. --seeds 0,1,2 (or --seeds 0 for a fast run)")
    out: list[int] = []
    for s in parts:
        if not s:
            _fail(f"--seeds {raw!r} has an empty entry — write e.g. --seeds 0,1,2")
        try:
            out.append(int(s))
        except ValueError:
            _fail(f"--seeds {raw!r}: {s!r} is not an integer — write e.g. --seeds 0,1,2")
    return tuple(out)


def _open_text(path: str):
    """`open()` that reports why it could not, and keeps the file streamable."""
    try:
        # utf-8-sig, matching ingest.load_traces. A BOM is routine in files written by
        # Windows tooling, and under plain utf-8 it survives as the first character --
        # so the one-byte format sniff below reads the BOM instead of the opening
        # bracket, misroutes a BOM-prefixed JSON *array* into the JSONL branch, and
        # then reports a parse error on a line that is perfectly valid.
        return open(path, encoding="utf-8-sig")
    except OSError as exc:
        _fail(f"cannot read {path}: {exc.strerror or exc}")


def _read_text(path: str, what: str = "file") -> str:
    try:
        with _open_text(path) as f:
            return f.read()
    except UnicodeDecodeError:
        _fail(f"{path} is not UTF-8 text (expected a JSON {what})")


def _parse_json(text: str, path: str) -> Any:
    """Whole-file JSON parse with a message that points at the offending line."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        hint = ""
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        try:
            json.loads(first)
        except ValueError:
            pass
        else:
            # The first line parses on its own: this is a JSONL log, not a broken file.
            hint = " — this looks like a JSONL log; `tokentrace triage` reads those"
        _fail(f"{path}:{exc.lineno}:{exc.colno}: not valid JSON ({exc.msg}){hint}")


def _load_model(name: str, backend: str, tier: Tier):
    """`load_model` whose optional-dependency failures are messages, not tracebacks."""
    from tokentrace.models import load_model

    try:
        return load_model(name, backend=backend, tier=tier)
    except ImportError as exc:
        # The backend's own message already names the extra to install.
        _fail(f"backend {backend!r} is unavailable: {exc} (or use --backend mock)")
    except (KeyError, ValueError) as exc:
        _fail(str(exc))


def _facade(args, train: bool):
    """Build the `TokenTrace` facade for the one-tier commands.

    Goes through the same error handling as :func:`_load_model` — the facade builds
    its own handle, so `--backend hf` with no torch installed came out of here as a
    raw ImportError traceback — and passes `--model`/`--backend` on every command
    that offers them (`demo` accepted both and quietly ran the mock regardless).
    """
    from tokentrace import TokenTrace

    try:
        return TokenTrace.default(backend=args.backend, model_name=args.model,
                                  tier=_TIER[args.tier], train=train)
    except ImportError as exc:
        _fail(f"backend {args.backend!r} is unavailable: {exc} (or use --backend mock)")
    except (KeyError, ValueError) as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #
def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    `json.dumps` emits bare `NaN`/`Infinity` tokens, which are not RFC 8259: they
    break `jq` and every strict parser. `analyze`/`triage` scrub at the boundary
    (`serialize._fin`, `triage._fin`); `eval`/`ablate` assemble plain metric dicts
    and had no scrub at all, so a degenerate run emitted an unparseable document.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _emit_json(payload: Any) -> None:
    """Print strict JSON. `allow_nan=False` makes anything the scrub missed an error
    rather than an invalid document."""
    print(json.dumps(_json_safe(payload), indent=2, allow_nan=False))


# --------------------------------------------------------------------------- #
# Trace loading (TokenTrace's own JSON + everything `tokentrace.ingest` reads)
# --------------------------------------------------------------------------- #
def _unwrap_native(payload: Any) -> Any:
    """`tokentrace inject` writes LabeledInference rows; the trace is one level in."""
    if isinstance(payload, dict) and isinstance(payload.get("inference"), dict):
        return payload["inference"]
    return payload


def _from_tokentrace(payload: Any) -> Inference:
    """This repo's own `inference_to_dict` / `labeled_to_dict` output.

    `ingest.load_traces` deliberately does not claim these rows — no adapter detects
    them — so round-tripping `tokentrace inject` output back through `triage` has to
    be handled here.
    """
    return inference_from_dict(_unwrap_native(payload))


def _is_tokentrace(payload: Any, strict: bool = True) -> bool:
    inner = _unwrap_native(payload)
    if not isinstance(inner, dict):
        return False
    keys = ("prompt", "generated_answer")
    return all(k in inner for k in keys) if strict else any(k in inner for k in keys)


def _load_auto(payload: Any) -> Inference:
    """Route one payload by sniffing, falling back to TokenTrace's own shape."""
    from tokentrace.ingest import detect_source, load_any

    detected = detect_source(payload)
    if detected == "generic":
        # An embedded field map is an explicit statement of intent: it outranks shape.
        return load_any(payload)
    if _is_tokentrace(payload):
        # Our own rows can be CLAIMED by `langchain` (it reads `generated_answer` as
        # an answer key) but only parsed lossily: `retrieved_context` is not a
        # document path it knows, so the chunks would vanish and the retrieval modes
        # would be hard-masked. Our own shape therefore wins outright.
        return _from_tokentrace(payload)
    if detected is not None:
        return load_any(payload)
    if _is_tokentrace(payload, strict=False):
        # Half a TokenTrace row: let `inference_from_dict` name the missing field
        # instead of reporting "no adapter recognized this payload".
        return _from_tokentrace(payload)
    return load_any(payload)      # -> UnknownSourceError, which lists the keys it saw


def _field_map(raw: Optional[str]) -> Optional[dict]:
    """`--field-map` is inline JSON or the path to a JSON file."""
    if raw is None:
        return None
    text = raw if raw.lstrip().startswith("{") else _read_text(raw, what="field map")
    try:
        fm = json.loads(text)
    except json.JSONDecodeError as exc:
        _fail(f"--field-map is not valid JSON ({exc.msg} at line {exc.lineno}, "
              f"column {exc.colno}). Pass inline JSON or a path to a .json file.")
    if not isinstance(fm, dict) or not fm:
        _fail("--field-map must be a non-empty JSON object of field -> path, e.g. "
              '\'{"question": "input.text", "answer": "output.text"}\'')
    return fm


def _trace_loader(source: str, field_map: Optional[dict]):
    """Build the `payload -> Inference` function selected by --source/--field-map."""
    from tokentrace.ingest import load_any

    if field_map is not None:
        if source not in ("auto", "generic"):
            _fail(f"--field-map forces the generic adapter, so --source {source} cannot "
                  "also apply — drop one of them.")
        return lambda payload: load_any(payload, field_map=field_map)
    if source == "tokentrace":
        return _from_tokentrace
    if source == "auto":
        return _load_auto
    return lambda payload: load_any(payload, source=source)


def _other_adapters(payload: Any, source: str) -> str:
    """Suggest a --source when some other adapter does recognize a failing row."""
    from tokentrace.ingest import detect_all

    names = [n for n in detect_all(payload) if n != source]
    if not names:
        return ""
    return f"  (recognized by: {', '.join(names)} — try --source {names[0]})"


def _load_trace(load, payload: Any, source: str, where: str, trace_id: str) -> Inference:
    try:
        inf = load(payload)
    except (ValueError, KeyError, TypeError) as exc:
        _fail(f"{where}: not a valid trace — {_reason(exc)}{_other_adapters(payload, source)}")
    if not inf.id:
        inf.id = trace_id            # the identity `ingest.load_traces` would assign
    return inf


def _stream_traces(path: str, load, source: str) -> Iterator[Inference]:
    """Yield traces lazily so a large log never has to fit in memory."""
    name = Path(path).name
    with _open_text(path) as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":                       # a single JSON array
            payloads = _parse_json(f.read(), path)
            for i, payload in enumerate(payloads):
                yield _load_trace(load, payload, source, f"{path}[{i}]", f"{name}#{i}")
            return
        for n, line in enumerate(f, 1):       # JSONL, one trace per line
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"{path}:{n}: not valid JSON ({exc.msg})")
            # 1-based, matching BOTH the error locator on this line and the id
            # ingest.load_traces assigns. It was 0-based, so a reported `file:5`
            # could not be matched to the trace it named, and the CLI and the
            # library disagreed about the identity of the same row.
            yield _load_trace(load, payload, source, f"{path}:{n}", f"{name}#{n}")


# --------------------------------------------------------------------------- #
# Printers — the seam between the producers and the terminal
# --------------------------------------------------------------------------- #
def _print_report(inf: Inference, report) -> None:
    print(f"\n  Q: {inf.query}")
    print(f"  A: {inf.generated_answer!r}"
          + (f"   (ground truth: {inf.ground_truth})" if inf.has_ground_truth else "  [no ground truth]"))
    print(f"  tier={report.tier.label}  diagnostic_confidence={report.diagnostic_confidence:.2f}"
          + ("  [ABSTAINED]" if report.abstained else ""))
    print("  ── ranked diagnoses ──")
    for d in report.diagnoses:
        if d.probability < 0.15:
            continue
        role = f" [{d.role.value}]" if d.role else ""
        parents = (" <- " + ", ".join(p.pretty for p in d.causal_parents)) if d.causal_parents else ""
        print(f"    {d.probability:5.2f}  {d.mode.pretty:18s}{role}{parents}")
        head = d.headline_evidence
        if head is not None:
            print(f"           evidence: {head.rendered}")
        for r in d.recommendations:
            tag = {True: "✓ validated", False: "✗ no effect", None: ""}[r.validated]
            print(f"           fix: {r.action} {tag}")
    if report.conformal_set:
        print(f"  Top-k set: {[m.pretty for m in report.conformal_set]}")
    for n in report.notes:
        print(f"  ⚠ {n}")


def _print_eval(res: dict) -> None:
    print(f"dataset sizes: {res['sizes']}")
    print(f"{'tier':6s} {'diagnosis':>10s} {'top3':>8s} {'rec_prec':>9s} {'abstain':>8s}")
    for tier, m in res["tiers"].items():
        print(f"{tier:6s} {m['diagnosis_accuracy']:10.3f} {m['top3_accuracy']:8.3f} "
              f"{m['recommendation_precision']:9.3f} {m['abstention_rate']:8.3f}")
    print("\ntargets: diagnosis>=0.80, top3>=0.90, rec_prec>=0.75")


def _print_ablations(res: dict, robustness: dict, model_label: str) -> None:
    # Both blocks below run on THIS handle. They used not to: the sweep hard-coded
    # the mock, so `--backend hf` printed real and mock numbers under one report with
    # nothing saying which was which.
    print(f"model: {model_label}")
    print(f"sizes: {res['sizes']}\n")
    print("learned-head ablation (white-box):")
    for k, v in res["ablation_learned_head"].items():
        print(f"  {k:16s} diagnosis={v['diagnosis_accuracy']:.3f}  top3={v['top3_accuracy']:.3f}")
    print("\nper-signal-family ablation — two distinct questions:")
    print("  (a) retrained leave-one-out = the family's INFORMATION contribution")
    for f, v in res["per_family_retrained"].items():
        print(f"      drop {f:12s} diagnosis={v['diagnosis_accuracy']:.3f}")
    print("  (b) missing-signal robustness = how the SHIPPED engine copes (no retrain)")
    for f, v in res["per_family_robustness"].items():
        print(f"      drop {f:12s} diagnosis={v['diagnosis_accuracy']:.3f}")

    c = res["calibration"]
    print(f"\ncalibration ECE: uncalibrated={c['ece_uncalibrated']:.4f} "
          f"calibrated={c['ece_calibrated']:.4f} "
          f"interior_mass={c['interior_mass_fraction']:.3f}")
    if c["interior_mass_fraction"] < 0.05:
        # Stated by the data, not about it: with no interior probability mass the ECE
        # figures are a rescaled error rate and carry no calibration information.
        print("  (interior mass ~0: the probabilities are saturated, so these ECEs are "
              "a rescaled error rate, not calibration evidence)")

    print("per-tier:")
    for t, v in res["per_tier"].items():
        print(f"  {t:6s} diagnosis={v['diagnosis_accuracy']:.3f} top3={v['top3_accuracy']:.3f} "
              f"conf_cov={v['conformal_coverage']:.3f} abstain={v['abstention_rate']:.3f} "
              f"dbg_time_reduction={v['debugging_time_reduction']:.3f}")

    # No editorial expectation here. This header used to promise that "calibration
    # should reduce ECE as noise grows" while the run underneath it showed the
    # calibrated ECE *worse* at three of four noise levels — a printed expectation
    # reads as a finding.
    print("\nrobustness under signal-observation noise "
          "(train + eval on the noisy pipeline; labels stay clean):")
    print(f"  {'noise':6s} {'diag':>6s} {'abstain':>8s} {'dbg_time':>9s} "
          f"{'ECE_uncal':>10s} {'ECE_cal':>8s}")
    for nz, mm in robustness.items():
        print(f"  {nz:6s} {mm['diagnosis_accuracy']:6.3f} {mm['abstention_rate']:8.3f} "
              f"{mm['debugging_time_reduction']:9.3f} {mm['ece_uncalibrated']:10.4f} "
              f"{mm['ece_calibrated']:8.4f}")


def _cell(x, fmt: str = "6.3f", width: int = 6) -> str:
    """Format a metric that may be ``noref``'s explicit UNAVAILABLE marker.

    ``run_noref`` replaces every ground-truth-dependent metric with a string rather
    than a 0.0, precisely so a reader cannot mistake "not measurable" for "measured
    zero". A printer that crashed on it — or worse, coerced it — would undo that, so
    the marker renders as an obvious dash (same convention as scripts/regen_report.py).
    """
    if isinstance(x, (int, float)):
        return format(x, fmt)
    return "-".rjust(width)


def _print_noref(nr: dict, model_label: str) -> None:
    """The §4.5 (no-reference / production split) tables, regen_report style.

    Reads only the keys it prints and looks optional blocks up with ``.get``:
    ``run_noref`` is an evolving result dict — ``frozen_mock.metrics`` is None on a
    real backend, ``ingest_shape`` is empty when the ingest tier equals the scoring
    tier, and new top-level keys must pass through without breaking the printer.
    """
    print(f"model: {model_label}")
    print(f"no-reference (production) split — tier={nr['tier']}, "
          f"test n={nr['sizes']['test']}:")

    cond = [("baseline", nr["baseline"]), ("transfer", nr["transfer"])]
    frozen = (nr.get("frozen_mock") or {}).get("metrics")
    if frozen is not None:
        cond.append(("frozen-mock", frozen))
    cond.append(("retrained", nr["retrained"]))

    print(f"  {'condition':12s} {'diag':>6s} {'top3':>6s} {'cov':>6s} {'set':>5s} "
          f"{'declnd':>6s} {'dbg':>6s} {'rank':>5s} {'recP':>6s}")
    for name, m in cond:
        print(f"  {name:12s} {m['diagnosis_accuracy']:6.3f} {m['top3_accuracy']:6.3f} "
              f"{m['conformal_coverage']:6.3f} {m['conformal_set_size']:5.2f} "
              f"{m['declined_rate']:6.3f} {m['debugging_time_reduction']:6.3f} "
              f"{m['mean_root_rank']:5.2f} {_cell(m['recommendation_precision'])}")

    print("\n  per-mode F1 (support from the baseline run):")
    names = [n for n, _ in cond]
    print("    " + f"{'mode':20s}" + "".join(f"{n[:6]:>7s}" for n in names) + f"{'supp':>6s}")
    for mode in ALL_MODES:
        cells = "".join(f"{m['per_mode'][mode.value]['f1']:7.2f}" for _, m in cond)
        supp = nr["baseline"]["per_mode"][mode.value]["support"]
        print(f"    {mode.value:20s}{cells}{supp:6d}")

    ing = nr.get("ingest_shape") or {}
    if ing:
        print(f"\n  ingest shape — no reference AND no mechanistic capture "
              f"(tier={ing['tier']}):")
        rows = [("baseline", ing["baseline"]), ("transfer", ing["transfer"])]
        if ing.get("frozen_mock"):
            rows.append(("frozen-mock", ing["frozen_mock"]))
        for name, m in rows:
            print(f"    {name:12s} diag={m['diagnosis_accuracy']:6.3f} "
                  f"top3={m['top3_accuracy']:6.3f} declined={m['declined_rate']:6.3f} "
                  f"dbg={m['debugging_time_reduction']:6.3f} "
                  f"halluc.F1={m['per_mode']['hallucination']['f1']:.2f} "
                  f"retr.F1={m['per_mode']['retrieval_failure']['f1']:.2f} "
                  f"dil.F1={m['per_mode']['context_dilution']['f1']:.2f}")


def _print_inject(summary: dict, n: int, out: str) -> None:
    _emit_json(summary)
    print(f"wrote {n} labeled examples -> {out}")


def _canonical() -> list[tuple[str, Inference]]:
    gold = Chunk("The Eiffel Tower was completed in 1889.", 0.9, "d1", gold=True)
    many = [Chunk(f"Filler {i} about unrelated topics facts words here now today plus more." * 3,
                  0.4, f"f{i}", gold=False) for i in range(8)]
    many.insert(4, Chunk("The Eiffel Tower was completed in 1889.", 0.6, "g", gold=True))
    g2 = Chunk("Marie Curie was born in 1867. Pierre Curie was born in 1859.", 0.9, "g2", gold=True)
    S = lambda **k: {"_sim": k}
    return [
        ("grounded (healthy)", Inference("When was the Eiffel Tower completed?\n" + gold.text, "",
            [gold], ["1889"], question="When was the Eiffel Tower completed?",
            meta=S(answer="1889", gold_fact="completed in 1889"))),
        ("retrieval failure", Inference("When was the Eiffel Tower completed?\nParis is in France.", "",
            [Chunk("Paris is in France.", 0.5, "d2")], ["1889"],
            question="When was the Eiffel Tower completed?",
            meta=S(answer="1889", gold_fact="1889", distractor="1920"))),
        ("context dilution", Inference("When was the Eiffel Tower completed?\n" + " ".join(c.text for c in many),
            "", many, ["1889"], question="When was the Eiffel Tower completed?",
            meta=S(answer="1889", gold_fact="completed in 1889", distractor="1850"))),
        ("prompt ambiguity", Inference("When was it completed?\n" + gold.text, "", [gold], ["1889"],
            question="When was it completed?",
            meta=S(answer="1889", ambiguous=True, readings=["1889", "1887", "2000"], gold_fact="completed in 1889"))),
        ("reasoning failure", Inference("Who was born first, Marie Curie or Pierre Curie?\n" + g2.text,
            "", [g2], ["Pierre Curie"], question="Who was born first, Marie Curie or Pierre Curie?",
            meta=S(answer="Pierre Curie", gold_fact="Pierre Curie was born in 1859",
                   requires_multihop=True, hard_composition=True, distractor="Marie Curie"))),
    ]


# --------------------------------------------------------------------------- #
def cmd_demo(args) -> int:
    print("Cold-start engine (interpretable rules only)..." if args.fast
          else f"Training engine on the offline synthetic corpus ({args.model}, CPU)...")
    tt = _facade(args, train=not args.fast)
    print(f"Ready. Diagnosing {len(_canonical())} canonical scenarios at tier={args.tier}:")
    for name, inf in _canonical():
        print(f"\n=== {name} ===", end="")
        _print_report(inf, tt.analyze(inf, tier=_TIER[args.tier]))
    return 0


def cmd_analyze(args) -> int:
    load = _trace_loader(args.source, _field_map(args.field_map))
    payload = _parse_json(_read_text(args.trace, what="trace"), args.trace)
    if isinstance(payload, list):
        _fail(f"{args.trace} holds a JSON array of {len(payload)} traces — `analyze` "
              "diagnoses one; use `tokentrace triage` for a whole log.")
    inf = _load_trace(load, payload, args.source, args.trace, f"{Path(args.trace).name}#0")
    tt = _facade(args, train=not args.fast)
    report = tt.analyze(inf, tier=_TIER[args.tier])
    if args.json:
        _emit_json(report_to_dict(report))
    else:
        _print_report(inf, report)
    return 0


def cmd_inject(args) -> int:
    from tokentrace.data.synthetic import build_dataset, dataset_summary

    seeds = _parse_seeds(args.seeds)
    model = _load_model(args.model, args.backend, _TIER[args.tier])
    ds = build_dataset(model, seeds=seeds)
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for li in ds:
                f.write(json.dumps(labeled_to_dict(li)) + "\n")
    except OSError as exc:
        _fail(f"cannot write {args.out}: {exc.strerror or exc}")
    _print_inject(dataset_summary(ds), len(ds), args.out)
    return 0


def cmd_ablate(args) -> int:
    from tokentrace.eval.ablations import run_ablations, run_robustness

    seeds = _parse_seeds(args.seeds)
    # `ablate` reports a per-tier degradation curve, so the handle is opened at its
    # ceiling and the curve is swept inside; there is deliberately no --tier flag.
    model = _load_model(args.model, args.backend, Tier.WHITE)
    res = run_ablations(model, seeds=seeds)
    robustness = run_robustness(noise_levels=_NOISE_LEVELS, seeds=seeds, model=model)
    if args.json:
        # The sweep used to be printed but omitted from --json (an early return), so a
        # consumer scripting the JSON silently lost the whole robustness section.
        _emit_json({**res, "model": args.model, "backend": args.backend,
                    "noise_levels": list(_NOISE_LEVELS), "robustness": robustness})
        return 0
    _print_ablations(res, robustness,
                     f"{args.model} (backend={args.backend}, tier={model.tier.label})")
    return 0


def cmd_noref(args) -> int:
    from tokentrace.eval.noref import run_noref

    seeds = _parse_seeds(args.seeds)
    # Like `eval`/`ablate`: the condition table IS the output — the white-box score
    # and the black-tier ingest condition are both swept inside — so the handle
    # opens at its ceiling and there is deliberately no --tier flag.
    model = _load_model(args.model, args.backend, Tier.WHITE)
    # Keywords only: run_noref grows keyword-compatibly, so a positional call is
    # the one shape that could silently rebind an argument.
    res = run_noref(model=model, seeds=seeds)
    if args.json:
        _emit_json({**res, "model": args.model, "backend": args.backend})
        return 0
    _print_noref(res, f"{args.model} (backend={args.backend}, tier={model.tier.label})")
    return 0


def cmd_triage(args) -> int:
    from tokentrace.triage import render_text, triage

    if args.top < 1:
        # `--top -1` used to slice clusters[:-1] and silently drop the last cluster;
        # `--top 0` printed a header over an empty table.
        _fail(f"--top must be >= 1 (got {args.top}) — it is how many clusters to show")
    load = _trace_loader(args.source, _field_map(args.field_map))
    tt = _facade(args, train=not args.fast)
    report = triage(_stream_traces(args.traces, load, args.source), tt, tier=_TIER[args.tier])
    if args.json:
        _emit_json(report.to_dict())
    else:
        print(render_text(report, top=args.top))
    return 0


def cmd_eval(args) -> int:
    from tokentrace.data.synthetic import build_dataset
    from tokentrace.eval.benchmark import train_and_evaluate

    seeds = _parse_seeds(args.seeds)
    # Like `ablate`: the per-tier table IS the output, so the handle opens at its
    # ceiling and the tiers are swept inside. No --tier flag to ignore.
    model = _load_model(args.model, args.backend, Tier.WHITE)
    ds = build_dataset(model, seeds=seeds)
    res = train_and_evaluate(model, ds)
    res.pop("_engine", None)
    if args.json:
        _emit_json(res)
        return 0
    _print_eval(res)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tokentrace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="mock-4b")
    common.add_argument("--backend", default="mock", choices=["mock", "gguf", "hf"])

    # Only for the commands that RUN at one tier. `eval`/`ablate` sweep white, grey
    # and black in a single run and print the curve, so a single --tier has nothing
    # to mean there; it used to be accepted, advertised in --help, and ignored.
    tiered = argparse.ArgumentParser(add_help=False)
    tiered.add_argument("--tier", default="white", choices=["black", "grey", "white"],
                        help="observation tier to run at (default: white)")

    # Reading logs that were not produced by TokenTrace.
    ingest = argparse.ArgumentParser(add_help=False)
    ingest.add_argument("--source", default="auto", choices=list(_SOURCES),
                        help="trace format (default: auto-detect, falling back to "
                             "TokenTrace's own trace JSON)")
    ingest.add_argument("--field-map", dest="field_map", metavar="JSON|PATH",
                        help="explicit field -> path map for an arbitrary log shape "
                             "(inline JSON or a .json file); forces --source generic")

    d = sub.add_parser("demo", parents=[common, tiered],
                       help="run the five canonical scenarios")
    d.add_argument("--fast", action="store_true", help="cold-start (rules only), skip training")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("analyze", parents=[common, tiered, ingest],
                       help="diagnose one inference JSON (TokenTrace's or a logged trace)")
    a.add_argument("trace")
    a.add_argument("--fast", action="store_true")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    inj = sub.add_parser("inject", parents=[common, tiered],
                         help="build the synthetic labeled corpus")
    inj.add_argument("--out", default="tokentrace_dataset.jsonl")
    inj.add_argument("--seeds", default="0,1,2")
    inj.set_defaults(func=cmd_inject)

    e = sub.add_parser("eval", parents=[common],
                       help="train + evaluate against targets (all three tiers)",
                       description="Trains once and evaluates at white, grey and black; "
                                   "the per-tier table is the output, so there is no "
                                   "--tier flag.")
    e.add_argument("--seeds", default="0,1,2,3")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval)

    tr = sub.add_parser("triage", parents=[common, tiered, ingest],
                        help="diagnose a whole log of traces and rank the clusters to fix")
    tr.add_argument("traces", help="JSONL (one trace per line) or a JSON array")
    tr.add_argument("--fast", action="store_true", help="cold-start (rules only), skip training")
    tr.add_argument("--top", type=int, default=10, help="clusters to show (default 10)")
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(func=cmd_triage)

    nr = sub.add_parser("noref", parents=[common],
                        help="the no-reference (production-shape) benchmark: strip "
                             "ground truth + gold annotations and re-score against "
                             "the reference-bearing baseline (runtime ~20-60 s)",
                        description="Measures the shipped engine on traces stripped of "
                                    "their reference metadata — the shape a production "
                                    "trace arrives in — next to the reference-bearing "
                                    "baseline, the frozen-mock artifact control and the "
                                    "black-tier ingest condition. Sweeps its own tiers, "
                                    "so there is no --tier flag. Runtime is ~20-60 s "
                                    "on the mock backend.")
    nr.add_argument("--seeds", default="0,1,2,3")
    nr.add_argument("--json", action="store_true")
    nr.set_defaults(func=cmd_noref)

    ab = sub.add_parser("ablate", parents=[common],
                        help="ablation studies (learned head, per-family, calibration) "
                             "+ the observation-noise sweep",
                        description="Sweeps the tiers and the signal families itself, "
                                    "so there is no --tier flag.")
    ab.add_argument("--seeds", default="0,1,2,3")
    ab.add_argument("--json", action="store_true")
    ab.set_defaults(func=cmd_ablate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
