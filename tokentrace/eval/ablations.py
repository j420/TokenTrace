"""Ablation studies on the offline synthetic corpus (no downloads).

Produces the analysis that turns the validated machinery into evidence:
  1. rules-only vs rules (+) learned GBT residual  — contribution of the learned head
  2. per-tier accuracy curves                       — graceful black/grey/white degradation
  3. per-signal-family ablation                     — which family carries which mode
  4. calibration ECE (calibrated vs uncalibrated)   — does isotonic calibration help
  5. conformal coverage vs mean set size            — Top-k adaptivity

All numbers are on a disjoint synthetic test split with the deterministic mock; they
validate the method end-to-end, not real-data performance (see docs/REPORT.md).
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import ALL_MODES, SignalFamily, Tier
from tokentrace.data.synthetic import build_dataset
from tokentrace.engine.calibration import sigmoid
from tokentrace.engine.diagnosis import DiagnosisEngine
from tokentrace.eval.benchmark import evaluate, split_dataset, train_engine
from tokentrace.models.base import ModelHandle
from tokentrace.recommend.recommender import Recommender
from tokentrace.signals.registry import SignalPipeline, default_extractors

_TIERS = (Tier.WHITE, Tier.GREY, Tier.BLACK)


def _ece(engine: DiagnosisEngine, test, model, pipeline, tier=Tier.WHITE,
         n_bins: int = 10, calibrated: bool = True) -> float:
    """Expected calibration error over all (example, mode) probabilities."""
    confs: list[float] = []
    labels: list[int] = []
    for li in test:
        fv = pipeline.run(li.inference, model.with_tier(tier))
        yv = dict(zip(ALL_MODES, li.label_vector()))
        if calibrated:
            probs = engine.probabilities(fv, li.inference)
        else:
            z = engine.raw_logits(fv, li.inference)
            probs = {m: (sigmoid(z[m]) if z[m] > -40 else 0.0) for m in ALL_MODES}
        for m in ALL_MODES:
            confs.append(probs[m])
            labels.append(yv[m])
    n = len(confs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c <= 0.0)]
        if not idx:
            continue
        acc = sum(labels[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return round(ece, 4)


def run_ablations(model: ModelHandle, dataset=None, pipeline: Optional[SignalPipeline] = None,
                  seeds: tuple[int, ...] = (0, 1, 2, 3)) -> dict:
    pipeline = pipeline or SignalPipeline()
    dataset = dataset if dataset is not None else build_dataset(model, pipeline, seeds=seeds)
    train, cal, test = split_dataset(dataset)

    full = train_engine(train, cal, model, pipeline)

    # 1. rules-only (cold-start: residual 0, no calibration/conformal fit) vs full
    rules_only = DiagnosisEngine(recommender=Recommender(model=model))
    ablation_learned = {
        "rules_only": evaluate(rules_only, test, model, Tier.WHITE, pipeline).as_dict(),
        "rules_plus_gbt": evaluate(full, test, model, Tier.WHITE, pipeline).as_dict(),
    }

    # 2. per-tier curves (full engine)
    per_tier = {t.label: evaluate(full, test, model, t, pipeline).as_dict() for t in _TIERS}

    # 3. per-family ablation — drop each family's extractors, evaluate at white-box
    per_family = {}
    for fam in (SignalFamily.PROMPT, SignalFamily.RETRIEVAL,
                SignalFamily.MECHANISTIC, SignalFamily.CONFIDENCE):
        kept = [ex for ex in default_extractors() if ex.family != fam]
        per_family[fam.value] = evaluate(full, test, model, Tier.WHITE,
                                         SignalPipeline(kept)).as_dict()

    # 4. calibration ECE
    calibration = {
        "ece_uncalibrated": _ece(full, test, model, pipeline, calibrated=False),
        "ece_calibrated": _ece(full, test, model, pipeline, calibrated=True),
    }

    # 5. conformal coverage vs set size, per tier (already in per_tier metrics)
    conformal = {t: {"coverage": per_tier[t]["conformal_coverage"],
                     "mean_set_size": per_tier[t]["conformal_set_size"]} for t in per_tier}

    return {
        "sizes": {"train": len(train), "cal": len(cal), "test": len(test)},
        "ablation_learned_head": ablation_learned,
        "per_tier": per_tier,
        "per_family_dropped": per_family,
        "calibration": calibration,
        "conformal": conformal,
    }
