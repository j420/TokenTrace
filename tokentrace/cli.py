"""TokenTrace command line.

    tokentrace demo                 # run the five canonical failure scenarios
    tokentrace analyze trace.json   # diagnose one inference (JSON)
    tokentrace inject --out d.jsonl # build the synthetic labeled corpus
    tokentrace eval                 # train + evaluate against the targets
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from tokentrace.core.types import Chunk, Inference, Tier
from tokentrace.core.serialize import inference_from_dict, labeled_to_dict, report_to_dict

_TIER = {"black": Tier.BLACK, "grey": Tier.GREY, "white": Tier.WHITE}


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
    from tokentrace import TokenTrace

    print("Cold-start engine (interpretable rules only)..." if args.fast
          else "Training engine on the offline synthetic corpus (mock model, CPU)...")
    tt = TokenTrace.default(train=not args.fast, tier=_TIER[args.tier])
    print(f"Ready. Diagnosing {len(_canonical())} canonical scenarios at tier={args.tier}:")
    for name, inf in _canonical():
        print(f"\n=== {name} ===", end="")
        _print_report(inf, tt.analyze(inf, tier=_TIER[args.tier]))
    return 0


def cmd_analyze(args) -> int:
    from tokentrace import TokenTrace

    with open(args.trace) as f:
        inf = inference_from_dict(json.load(f))
    tt = TokenTrace.default(backend=args.backend, model_name=args.model,
                            tier=_TIER[args.tier], train=not args.fast)
    report = tt.analyze(inf, tier=_TIER[args.tier])
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        _print_report(inf, report)
    return 0


def cmd_inject(args) -> int:
    from tokentrace.data.synthetic import build_dataset, dataset_summary
    from tokentrace.models import load_model

    model = load_model(args.model, backend=args.backend, tier=_TIER[args.tier])
    seeds = tuple(int(s) for s in args.seeds.split(","))
    ds = build_dataset(model, seeds=seeds)
    with open(args.out, "w") as f:
        for li in ds:
            f.write(json.dumps(labeled_to_dict(li)) + "\n")
    print(json.dumps(dataset_summary(ds), indent=2))
    print(f"wrote {len(ds)} labeled examples -> {args.out}")
    return 0


def cmd_ablate(args) -> int:
    from tokentrace.eval.ablations import run_ablations
    from tokentrace.models import load_model

    model = load_model(args.model, backend=args.backend, tier=Tier.WHITE)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    res = run_ablations(model, seeds=seeds)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"sizes: {res['sizes']}\n")
    al = res["ablation_learned_head"]
    print("learned-head ablation (white-box):")
    for k, v in al.items():
        print(f"  {k:16s} diagnosis={v['diagnosis_accuracy']:.3f}  top3={v['top3_accuracy']:.3f}")
    print("\nper-signal-family ablation — two distinct questions:")
    print("  (a) retrained leave-one-out = the family's INFORMATION contribution")
    for f, v in res["per_family_retrained"].items():
        print(f"      drop {f:12s} diagnosis={v['diagnosis_accuracy']:.3f}")
    print("  (b) missing-signal robustness = how the SHIPPED engine copes (no retrain)")
    for f, v in res["per_family_robustness"].items():
        print(f"      drop {f:12s} diagnosis={v['diagnosis_accuracy']:.3f}")
    print(f"\ncalibration ECE: {res['calibration']}")
    print("per-tier:")
    for t, v in res["per_tier"].items():
        print(f"  {t:6s} diagnosis={v['diagnosis_accuracy']:.3f} top3={v['top3_accuracy']:.3f} "
              f"conf_cov={v['conformal_coverage']:.3f} abstain={v['abstention_rate']:.3f} "
              f"dbg_time_reduction={v['debugging_time_reduction']:.3f}")

    from tokentrace.eval.ablations import run_robustness
    print("\nrobustness under signal-observation noise (calibration should reduce ECE as noise grows):")
    print(f"  {'noise':6s} {'diag':>6s} {'abstain':>8s} {'dbg_time':>9s} {'ECE_uncal':>10s} {'ECE_cal':>8s}")
    for nz, mm in run_robustness(noise_levels=(0.0, 0.25, 0.5, 0.75), seeds=seeds).items():
        print(f"  {nz:6s} {mm['diagnosis_accuracy']:6.3f} {mm['abstention_rate']:8.3f} "
              f"{mm['debugging_time_reduction']:9.3f} {mm['ece_uncalibrated']:10.4f} {mm['ece_calibrated']:8.4f}")
    return 0


def cmd_eval(args) -> int:
    from tokentrace.data.synthetic import build_dataset
    from tokentrace.eval.benchmark import train_and_evaluate
    from tokentrace.models import load_model

    model = load_model(args.model, backend=args.backend, tier=Tier.WHITE)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    ds = build_dataset(model, seeds=seeds)
    res = train_and_evaluate(model, ds)
    res.pop("_engine", None)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"dataset sizes: {res['sizes']}")
    print(f"{'tier':6s} {'diagnosis':>10s} {'top3':>8s} {'rec_prec':>9s} {'abstain':>8s}")
    for tier, m in res["tiers"].items():
        print(f"{tier:6s} {m['diagnosis_accuracy']:10.3f} {m['top3_accuracy']:8.3f} "
              f"{m['recommendation_precision']:9.3f} {m['abstention_rate']:8.3f}")
    print("\ntargets: diagnosis>=0.80, top3>=0.90, rec_prec>=0.75")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tokentrace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="mock-4b")
    common.add_argument("--backend", default="mock", choices=["mock", "gguf", "hf"])
    common.add_argument("--tier", default="white", choices=["black", "grey", "white"])

    d = sub.add_parser("demo", parents=[common], help="run the five canonical scenarios")
    d.add_argument("--fast", action="store_true", help="cold-start (rules only), skip training")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("analyze", parents=[common], help="diagnose one inference JSON")
    a.add_argument("trace")
    a.add_argument("--fast", action="store_true")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    inj = sub.add_parser("inject", parents=[common], help="build the synthetic labeled corpus")
    inj.add_argument("--out", default="tokentrace_dataset.jsonl")
    inj.add_argument("--seeds", default="0,1,2")
    inj.set_defaults(func=cmd_inject)

    e = sub.add_parser("eval", parents=[common], help="train + evaluate against targets")
    e.add_argument("--seeds", default="0,1,2,3")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablate", parents=[common], help="run ablation studies (learned head, per-family, calibration)")
    ab.add_argument("--seeds", default="0,1,2,3")
    ab.add_argument("--json", action="store_true")
    ab.set_defaults(func=cmd_ablate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
