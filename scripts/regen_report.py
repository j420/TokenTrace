#!/usr/bin/env python3
"""Regenerate every published number from a single reproducible run.

    python3 scripts/regen_report.py            # print the results block
    python3 scripts/regen_report.py --json     # machine-readable

Why this exists: an audit found that most headline figures in README.md /
docs/REPORT.md no longer corresponded to any code path — each round of fixes changed
the numbers and the prose was updated by hand. Every table in the docs is now
produced by THIS script, so a claim and the code that justifies it cannot drift
apart. Re-run it after any change that could move a metric and paste the output.

That guarantee had a hole until this revision: ``run_noref`` was never imported
here, so the whole of REPORT §4.5 — the no-reference (production) split, which the
report calls its most consequential result — was hand-transcribed. It is wired in
below, along with the two figures §4.5 quotes that ``Metrics`` does not carry
(primary-root precision on the rows that were given a root, and the per-class means
of the reference-free separators), and the shape probe is now run at BOTH the
published seed set and the one the standing test uses, since those differ.

Runtime: 57 s single-threaded (``OMP_NUM_THREADS=1``) on the reference machine, of
which ~20 s is ``run_noref``; it was ~50 s before §4.5 was wired in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokentrace.core.types import ALL_MODES, Tier  # noqa: E402
from tokentrace.data import build_dataset, dataset_summary  # noqa: E402
from tokentrace.eval.ablations import run_ablations, run_robustness  # noqa: E402
from tokentrace.eval.benchmark import split_dataset  # noqa: E402
from tokentrace.eval.metrics import gold_primary  # noqa: E402
from tokentrace.eval.noref import run_noref  # noqa: E402
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


#: The two reference-free features §4.5 cites as retrieval-failure's surviving
#: separators. Printed per class because the claim is a CONTRAST, and a contrast
#: quoted without its comparison class cannot be checked — which is exactly how
#: that sentence came to quote two different classes as if they were one.
SEPARATORS = ("max_chunk_relevance", "answer_supported_by_context")


def separator_means(model, pipeline, dataset) -> dict:
    """Per-gold-primary-class means of the reference-free separators, on the test split.

    ``n_rag`` is reported separately from ``n`` because both features come from
    ``RetrievalExtractor``, which does not run on a non-RAG inference at all — so the
    mean is over the RAG rows only. Printing one ``n`` next to a mean taken over a
    different, smaller set is how a table quietly misstates its own denominator.
    """
    _train, _cal, test = split_dataset(dataset)
    buckets: dict[str, dict] = {}
    for li in test:
        gp = gold_primary(li)
        cls = gp.value if gp is not None else "(clean)"
        fv = pipeline.run(li.inference, model)
        b = buckets.setdefault(cls, {"n": 0, "n_rag": 0, **{f: [] for f in SEPARATORS}})
        b["n"] += 1
        b["n_rag"] += int(li.inference.is_rag)
        for f in SEPARATORS:
            v = fv.get(f)
            if v is not None:
                b[f].append(float(v))
    out = {}
    for cls, b in sorted(buckets.items()):
        out[cls] = {"n": b["n"], "n_rag": b["n_rag"], **{
            f: (round(sum(b[f]) / len(b[f]), 4) if b[f] else None) for f in SEPARATORS}}
    return out


def primary_precision(confusion: dict) -> tuple[int, int, float]:
    """(correct, named, precision) over the rows that were GIVEN a primary root.

    Derived from the confusion matrix rather than from ``Metrics``, because
    ``diagnosis_accuracy`` charges an abstention as a miss — which is right for
    accuracy and wrong for the separate question §4.5 turns on: when the engine does
    speak without a reference, how often is it right? A clean row counts as wrong
    whenever a cause is named at all; abstaining is the correct answer there.
    """
    correct = named = 0
    for key, n in confusion.items():
        gold, pred = key.split(" -> ")
        if pred == "(abstain)":
            continue
        named += n
        if gold != "(clean)" and pred in gold.split("/"):
            correct += n
    return correct, named, (correct / named if named else 0.0)


def _num(x, fmt: str = "6.3f", width: int = 6) -> str:
    """Format a metric that may be the explicit UNAVAILABLE marker.

    ``noref`` replaces every ground-truth-dependent metric with a string rather than
    a 0.0, precisely so a reader cannot mistake "not measurable" for "measured zero".
    A printer that crashed on it — or worse, coerced it — would undo that, so the
    marker is rendered as an obvious dash instead.
    """
    if isinstance(x, (int, float)):
        return format(x, fmt)
    return "-".rjust(width)


def print_noref(nr: dict) -> None:
    """§4.5 of docs/REPORT.md, regenerated.

    Every figure §4.5 quotes is printed here — the condition table, per-mode F1,
    the confound controls, the calibration-path counts, the ECE chain and the
    confusion flows — because until this function existed that whole section was
    hand-transcribed with no regeneration path, which is how it drifted before.
    """
    cond = [("baseline", nr["baseline"]), ("transfer", nr["transfer"])]
    frozen = nr["frozen_mock"]["metrics"]
    if frozen is not None:
        cond.append(("frozen-mock", frozen))
    cond.append(("retrained", nr["retrained"]))

    print(f"\nno-reference (production) split — tier={nr['tier']}, "
          f"test n={nr['sizes']['test']}:")
    cms = {"baseline": nr["confusion"]["baseline"], "transfer": nr["confusion"]["transfer"],
           "frozen-mock": nr["frozen_mock"]["confusion"],
           "retrained": nr["confusion"]["retrained"]}
    print(f"  {'condition':12s} {'diag':>6s} {'top3':>6s} {'cov':>6s} {'set':>5s} "
          f"{'declnd':>6s} {'dbg':>6s} {'rank':>5s} {'recP':>6s}  primary-root precision")
    for name, m in cond:
        ok, named, prec = primary_precision(cms[name])
        print(f"  {name:12s} {m['diagnosis_accuracy']:6.3f} {m['top3_accuracy']:6.3f} "
              f"{m['conformal_coverage']:6.3f} {m['conformal_set_size']:5.2f} "
              f"{m['declined_rate']:6.3f} {m['debugging_time_reduction']:6.3f} "
              f"{m['mean_root_rank']:5.2f} {_num(m['recommendation_precision'])}  "
              f"{prec:.3f} ({ok}/{named} named)")

    print("\n  per-mode F1 (support from the baseline run):")
    names = [n for n, _ in cond]
    print("    " + f"{'mode':20s}" + "".join(f"{n[:6]:>7s}" for n in names) + f"{'supp':>6s}")
    for mode in ALL_MODES:
        cells = "".join(f"{m['per_mode'][mode.value]['f1']:7.2f}" for _, m in cond)
        supp = nr["baseline"]["per_mode"][mode.value]["support"]
        print(f"    {mode.value:20s}{cells}{supp:6d}")

    cc = nr["confound_controls"]
    print("\n  confound controls (not shippable; they attribute the drop):")
    print(f"    {'annotations removed, reference kept':44s} "
          f"diag={cc['annotations_only_removed']['diagnosis_accuracy']:.3f}")
    print(f"    {'reference removed, annotations kept':44s} "
          f"diag={cc['reference_only_removed']['diagnosis_accuracy']:.3f}")
    print(f"    {'both removed (= transfer)':44s} "
          f"diag={nr['transfer']['diagnosis_accuracy']:.3f}")
    if frozen is not None:
        print(f"    {'both removed, mock decision frozen':44s} "
              f"diag={frozen['diagnosis_accuracy']:.3f}  "
              f"declined={frozen['declined_rate']:.3f}  "
              f"dilutionF1={frozen['per_mode']['context_dilution']['f1']:.2f}")
    print(f"    mock re-decided rows, by recipe: {cc['mock_decisions_changed_by_recipe']}")

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

    print("\n  calibration path (which map served each probability):")
    for name in ("baseline", "transfer", "retrained"):
        cp = nr["calibration_path"][name]
        lv = cp["fallback_levels"]
        total = sum(lv.values())
        pooled = cp["fallback_levels_noref_rows_only"].get("global_pooled", 0)
        print(f"    {name:10s} {lv}  total={total}")
        print(f"               noref rows -> pooled global map: {pooled}/{total}  "
              f"pool_admits_noref={cp['global_pool_admits_noref']} "
              f"(n_noref_records={cp['n_noref_calibration_records']})  "
              f"BUG={cp['noref_served_by_pooled_map']}")

    gc = nr.get("guard_cost") or {}
    if gc:
        print("\n  cost of the __global__ guard on the transfer condition "
              "(before -> after refusing the pooled reference-bearing map):")
        before, after = gc["before_guard"], gc["after_guard"]
        for key, fmt in (("top3_accuracy", "6.3f"), ("conformal_coverage", "6.3f"),
                         ("conformal_set_size", "6.3f"), ("declined_rate", "6.3f"),
                         ("debugging_time_reduction", "6.3f"), ("mean_root_rank", "6.3f"),
                         ("diagnosis_accuracy", "6.3f")):
            print(f"    {key:26s} {before[key]:{fmt}} -> {after[key]:{fmt}}")
        for mode in ALL_MODES:
            b = gc["before_guard_per_mode"][mode.value]["f1"]
            a = after["per_mode"][mode.value]["f1"]
            if b != a:
                print(f"    {mode.value + ' F1':26s} {b:6.3f} -> {a:6.3f}")
        print(f"    {'ECE (calibrated)':26s} "
              f"{gc['before_guard_ece']['calibrated']['ece']:6.4f} -> "
              f"{nr['ece']['transfer']['calibrated']['ece']:6.4f}")
        bp = gc["before_guard_calibration_path"]
        pooled = bp["fallback_levels_noref_rows_only"].get("global_pooled", 0)
        total = sum(bp["fallback_levels"].values())
        print(f"    production-shape probabilities served by the pooled reference-bearing "
              f"map: {pooled}/{total} before the guard, "
              f"{nr['calibration_path']['transfer']['fallback_levels_noref_rows_only'].get('global_pooled', 0)}"
              f"/{total} after")
        print(f"    comparable metrics moved by the guard: {gc['n_comparable_metrics_moved']} "
              f"{gc['comparable_metrics_moved']}")

    print("\n  ECE (uncalibrated -> calibrated, interior mass in brackets):")
    for name, block in nr["ece"].items():
        if block is None:
            continue
        u, c = block["uncalibrated"], block["calibrated"]
        print(f"    {name:12s} {u['ece']:.4f} -> {c['ece']:.4f}  "
              f"[interior {c['interior_frac']:.3f}, n={c['n']}]")

    rvd = nr["retrieval_vs_dilution"]
    print("\n  where the collapsed modes actually went (transfer):")
    print(f"    retrieval_failure -> {rvd['retrieval_actually_went_to']}")
    print(f"    context_dilution  -> {rvd['dilution_actually_went_to']}")
    print(f"    cross-confusion retrieval<->dilution: "
          f"{rvd['retrieval_predicted_as_dilution']} / "
          f"{rvd['dilution_predicted_as_retrieval']}  (prediction held: {rvd['held']})")
    if frozen is not None:
        fz = nr["frozen_mock"]["confusion"]
        print("  where they went with the mock frozen:")
        for gold in ("retrieval_failure", "context_dilution"):
            flow = {k.split(" -> ")[1]: v for k, v in fz.items()
                    if k.split(" -> ")[0] == gold}
            print(f"    {gold:18s} -> {flow}")

    # The full matrix, because §4.5 narrates individual cells ("2 clean rows are
    # called reasoning failures") and a flow view cannot be checked against those.
    print("\n  full confusion, gold root -> predicted primary (misses only):")
    for name in ("baseline", "transfer", "frozen-mock", "retrained"):
        misses = {k: v for k, v in cms[name].items()
                  if k.split(" -> ")[1] not in k.split(" -> ")[0].split("/")}
        print(f"    {name:12s} {misses}")


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
        # The standing test (`test_labels_not_recoverable_from_context_shape`) now
        # builds the SAME four-seed corpus as this probe, so the single figure below
        # is both the published number and the tested one. A second three-seed probe
        # used to be printed here to expose the mismatch; the mismatch is gone, and
        # keeping the extra probe would re-create the confusion it existed to flag.
        "shortcut_probe": shape_shortcut_probe(model, pipeline, dataset),
        "ablations": run_ablations(model, dataset=dataset, pipeline=pipeline, seeds=SEEDS),
        # §4.5. Wired in here because it previously was not: the whole section was
        # hand-transcribed, with no regeneration path and nothing pinning it, while
        # line 3 of docs/REPORT.md promised every number came from this script.
        "noref": run_noref(model, dataset=dataset, pipeline=pipeline, seeds=SEEDS),
        "separator_means": separator_means(model, pipeline, dataset),
        "robustness": run_robustness(noise_levels=(0.0, 0.25, 0.5, 0.75), seeds=SEEDS),
    }

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    ab = out["ablations"]
    print(f"corpus: {out['corpus']['n']} rows  sizes={ab['sizes']}")
    print(f"by mode: {out['corpus']['by_mode']}\n")

    p = out["shortcut_probe"]
    print(f"shape-only shortcut probe, seeds={list(SEEDS)} (published AND tested): "
          f"acc={p['shape_only_accuracy']:.3f} (class prior {p['class_prior']:.3f})\n")

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

    print_noref(out["noref"])

    print("\n  reference-free separators, mean per gold-primary class (test split;"
          " means are over the n_rag rows, since both features need retrieval):")
    print(f"    {'class':20s} {'n':>4s} {'n_rag':>6s} "
          + " ".join(f"{f:>28s}" for f in SEPARATORS))
    for cls, row in out["separator_means"].items():
        cells = " ".join(f"{row[f]:28.3f}" if row[f] is not None else f"{'--':>28s}"
                         for f in SEPARATORS)
        print(f"    {cls:20s} {row['n']:4d} {row['n_rag']:6d} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
