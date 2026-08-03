#!/usr/bin/env python3
"""Regenerate every published number from a single reproducible run.

    python3 scripts/regen_report.py            # print the results block
    python3 scripts/regen_report.py --json     # machine-readable

Why this exists: an audit found that most headline figures in README.md /
docs/REPORT.md no longer corresponded to any code path — each round of fixes changed
the numbers and the prose was updated by hand. Every table in the docs is now
produced by THIS script, so a claim and the code that justifies it cannot drift
apart. Re-run it after any change that could move a metric and paste the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokentrace.core.types import Tier  # noqa: E402
from tokentrace.data import build_dataset, dataset_summary  # noqa: E402
from tokentrace.eval.ablations import run_ablations, run_robustness  # noqa: E402
from tokentrace.eval.benchmark import split_dataset  # noqa: E402
from tokentrace.eval.metrics import gold_primary  # noqa: E402
from tokentrace.models import load_model  # noqa: E402
from tokentrace.signals import SignalPipeline  # noqa: E402

SEEDS = (0, 1, 2, 3)
SHAPE = ["n_chunks", "context_length_tokens", "prompt_n_content_words",
         "answer_length_tokens"]


def shape_shortcut_probe(model, pipeline, dataset) -> dict:
    """How much of the diagnosis is recoverable from context SHAPE alone.

    A standing guard against the corpus leaking labels through geometry. Dilution is
    causally geometric so some signal is expected; the bar is that shape must not be
    sufficient.
    """
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    train, _cal, test = split_dataset(dataset)

    def mk(rows):
        X, y = [], []
        for li in rows:
            gp = gold_primary(li)
            if gp is None:
                continue
            fv = pipeline.run(li.inference, model)
            X.append([fv.get(f) if fv.get(f) is not None else np.nan for f in SHAPE])
            y.append(gp.value)
        return np.nan_to_num(np.array(X, dtype=float), nan=-1.0), np.array(y)

    Xtr, ytr = mk(train)
    Xte, yte = mk(test)
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    vals, counts = np.unique(yte, return_counts=True)
    return {"shape_only_accuracy": round(acc, 4),
            "class_prior": round(float(counts.max() / counts.sum()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = load_model("mock-4b", backend="mock", tier=Tier.WHITE)
    pipeline = SignalPipeline()
    dataset = build_dataset(model, pipeline, seeds=SEEDS)

    out = {
        "seeds": list(SEEDS),
        "corpus": dataset_summary(dataset),
        "shortcut_probe": shape_shortcut_probe(model, pipeline, dataset),
        "ablations": run_ablations(model, dataset=dataset, pipeline=pipeline, seeds=SEEDS),
        "robustness": run_robustness(noise_levels=(0.0, 0.25, 0.5, 0.75), seeds=SEEDS),
    }

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    ab = out["ablations"]
    print(f"corpus: {out['corpus']['n']} rows  sizes={ab['sizes']}")
    print(f"by mode: {out['corpus']['by_mode']}\n")

    p = out["shortcut_probe"]
    print(f"shape-only shortcut probe: acc={p['shape_only_accuracy']:.3f} "
          f"(class prior {p['class_prior']:.3f})\n")

    hdr = f"{'tier':6s} {'diag':>6s} {'top3':>6s} {'cov':>6s} {'set':>5s} " \
          f"{'recP':>6s} {'recN':>5s} {'neg':>4s} {'healthy':>8s} {'declined':>9s} {'dbg':>6s}"
    print("per tier:")
    print("  " + hdr)
    for tier, m in ab["per_tier"].items():
        print(f"  {tier:6s} {m['diagnosis_accuracy']:6.3f} {m['top3_accuracy']:6.3f} "
              f"{m['conformal_coverage']:6.3f} {m['conformal_set_size']:5.2f} "
              f"{m['recommendation_precision']:6.3f} {m['recommendation_n']:5d} "
              f"{m['recommendation_negatives']:4d} {m['healthy_abstention_rate']:8.3f} "
              f"{m['declined_rate']:9.3f} {m['debugging_time_reduction']:6.3f}")

    al = ab["ablation_learned_head"]
    print("\nlearned head (white):")
    for k, v in al.items():
        f1s = [s["f1"] for s in v["per_mode"].values() if s["support"]]
        macro = sum(f1s) / len(f1s) if f1s else 0.0
        print(f"  {k:16s} diag={v['diagnosis_accuracy']:.3f} macroF1={macro:.3f}")

    for label, block in (("retrained leave-one-family-out", ab["per_family_retrained"]),
                         ("missing-signal robustness", ab["per_family_robustness"])):
        print(f"\nper-family — {label}:")
        for fam, v in block.items():
            pm = v["per_mode"]
            cols = " ".join(f"{k[:4]}={pm[k]['f1']:.2f}" for k in sorted(pm))
            print(f"  drop {fam:12s} diag={v['diagnosis_accuracy']:.3f}  {cols}")

    c = ab["calibration"]
    print(f"\ncalibration: ece_uncal={c['ece_uncalibrated']:.4f} "
          f"ece_cal={c['ece_calibrated']:.4f} interior_mass={c['interior_mass_fraction']:.3f}")

    print("\nrobustness (observation noise):")
    print(f"  {'sigma':6s} {'diag':>6s} {'abst':>6s} {'dbg':>6s} {'ece_unc':>8s} {'ece_cal':>8s}")
    for nz, m in out["robustness"].items():
        print(f"  {nz:6s} {m['diagnosis_accuracy']:6.3f} {m['abstention_rate']:6.3f} "
              f"{m['debugging_time_reduction']:6.3f} {m['ece_uncalibrated']:8.4f} "
              f"{m['ece_calibrated']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
