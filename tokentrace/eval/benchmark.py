"""Train + evaluate the diagnosis engine end-to-end (offline path).

Trains the residual heads on synthetic + semi labels, fits per-missingness
calibration and the conformal set on a held-out split, then evaluates on a
disjoint split. In the real pipeline the *evaluation* split is RAGTruth + the
human-audited seed (provenance REAL); here it is a disjoint synthetic split so the
whole thing runs with no downloads.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from tokentrace.core.types import ALL_MODES, LabeledInference, Tier
from tokentrace.data.synthetic import featurize
from tokentrace.engine.calibration import Calibrator
from tokentrace.engine.classifier import ResidualClassifier
from tokentrace.engine.conformal import ConformalPredictor
from tokentrace.engine.diagnosis import DiagnosisEngine
from tokentrace.eval.metrics import Metrics, compute_metrics, gold_primary
from tokentrace.models.base import ModelHandle
from tokentrace.recommend.recommender import Recommender
from tokentrace.signals.registry import SignalPipeline

_MASK = -49.0


def split_dataset(
    dataset: list[LabeledInference], ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> tuple[list, list, list]:
    """Deterministic hash-shuffled train/cal/eval split (no RNG)."""
    keyed = sorted(dataset, key=lambda li: hashlib.sha1(li.inference.prompt.encode()).hexdigest())
    n = len(keyed)
    a = int(n * ratios[0])
    b = a + int(n * ratios[1])
    return keyed[:a], keyed[a:b], keyed[b:]


def train_engine(
    train: list[LabeledInference],
    cal: list[LabeledInference],
    model: ModelHandle,
    pipeline: Optional[SignalPipeline] = None,
    train_tiers: Optional[list[Tier]] = None,
    validate_recommendations: bool = True,
) -> DiagnosisEngine:
    pipeline = pipeline or SignalPipeline()

    # 1) fit residual heads on the featurized training set
    feats = featurize(train, model, pipeline, tiers=train_tiers)
    clf = ResidualClassifier(feats["feature_names"]).fit(
        feats["X"], feats["rule_logits"], feats["Y"], sample_weight=feats["weights"]
    )
    engine = DiagnosisEngine(classifier=clf)

    # 2) fit per-(mode, signature) calibration on the cal split, across EVERY tier
    # that will be evaluated — otherwise the black-box missingness signature is never
    # fit and transform() falls back to a WHITE-fit map (the miscalibration this is
    # meant to prevent).
    # `with_tier` can only DOWN-cap, so asking a black-box handle for WHITE silently
    # returns black — three identical passes whose duplicates then inflate counts
    # against Calibrator.min_samples. Derive the schedule from the handle's real
    # ceiling and de-duplicate the records.
    cal_tiers = [t for t in (Tier.WHITE, Tier.GREY, Tier.BLACK) if t <= model.tier] or [model.tier]
    records = []
    seen_rec: set = set()
    for li in cal:
        for t in cal_tiers:
            fv = pipeline.run(li.inference, model.with_tier(t))
            z = engine.raw_logits(fv, li.inference)
            sig = fv.missingness_signature()
            yv = dict(zip(ALL_MODES, li.label_vector()))
            for m in ALL_MODES:
                if z[m] <= _MASK:            # masked (non-RAG) — nothing to calibrate
                    continue
                key = (m.value, sig, round(z[m], 6), yv[m])
                if key in seen_rec:
                    continue
                seen_rec.add(key)
                records.append({"mode": m, "signature": sig, "z": z[m], "label": yv[m]})
    calibrator = Calibrator().fit(records)
    engine.calibrator = calibrator

    # 3) fit conformal on the cal split across the same tiers (tau must reflect the
    # flatter black-box probability distribution too, not just WHITE), de-duplicated
    # so identical tier passes cannot weight the quantile.
    prob_rows, gps = [], []
    seen_prob: set = set()
    for li in cal:
        gp = gold_primary(li)
        if gp is None:
            continue
        for t in cal_tiers:
            fv = pipeline.run(li.inference, model.with_tier(t))
            probs = engine.probabilities(fv, li.inference)
            key = (gp.value, tuple(round(probs[m], 6) for m in ALL_MODES))
            if key in seen_prob:
                continue
            seen_prob.add(key)
            prob_rows.append(probs)
            gps.append(gp)
    engine.conformal = ConformalPredictor().fit(prob_rows, gps)

    # 4) attach recommender for eval (simulated-intervention validation)
    if validate_recommendations:
        engine.recommender = Recommender(model=model)
    return engine


def run_reports(
    engine: DiagnosisEngine,
    dataset: list[LabeledInference],
    model: ModelHandle,
    tier: Tier,
    pipeline: Optional[SignalPipeline] = None,
) -> list:
    pipeline = pipeline or SignalPipeline()
    tiered = model.with_tier(tier)
    reports = []
    for li in dataset:
        fv = pipeline.run(li.inference, tiered)
        reports.append(engine.diagnose(li.inference, fv, tier))
    return reports


def evaluate(
    engine: DiagnosisEngine,
    dataset: list[LabeledInference],
    model: ModelHandle,
    tier: Tier,
    pipeline: Optional[SignalPipeline] = None,
) -> Metrics:
    reports = run_reports(engine, dataset, model, tier, pipeline)
    return compute_metrics(reports, dataset)


def train_and_evaluate(
    model: ModelHandle,
    dataset: list[LabeledInference],
    pipeline: Optional[SignalPipeline] = None,
    eval_tiers: tuple[Tier, ...] = (Tier.WHITE, Tier.GREY, Tier.BLACK),
    family_dropout: bool = True,
) -> dict:
    """Full offline benchmark. Returns metrics per eval tier + dataset sizes."""
    pipeline = pipeline or SignalPipeline()
    train, cal, test = split_dataset(dataset)

    # Family-dropout: train across mixed tiers so the heads are robust to missing
    # families (this is what keeps black-box calibration honest).
    train_tiers = None
    if family_dropout:
        cycle = [Tier.WHITE, Tier.GREY, Tier.BLACK]
        train_tiers = [cycle[i % 3] for i in range(len(train))]

    engine = train_engine(train, cal, model, pipeline, train_tiers=train_tiers)

    out = {"sizes": {"train": len(train), "cal": len(cal), "test": len(test)}, "tiers": {}}
    for tier in eval_tiers:
        out["tiers"][tier.label] = evaluate(engine, test, model, tier, pipeline).as_dict()
    out["_engine"] = engine
    return out
