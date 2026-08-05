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


def _ece_full(engine: DiagnosisEngine, test, model, pipeline, tier=Tier.WHITE,
              n_bins: int = 10, calibrated: bool = True) -> dict:
    """Expected calibration error over all (example, mode) probabilities.

    Uses EQUAL-MASS (quantile) bins rather than fixed-width ones, and additionally
    reports how much probability mass is actually *interior* (0.1 < p < 0.9).

    Why that matters: on a separable corpus the residual head saturates, so every
    probability sits at 0 or 1 and fixed-width ECE degenerates into a rescaled error
    rate — a number that looks like calibration evidence but carries none. Reporting
    ``interior_frac`` alongside makes that degeneracy visible instead of quotable.
    """
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
    if n == 0:
        return {"ece": 0.0, "interior_frac": 0.0, "n": 0}

    order = sorted(range(n), key=lambda i: confs[i])
    ece = 0.0
    edges = [round(b * n / n_bins) for b in range(n_bins + 1)]
    for b in range(n_bins):
        idx = order[edges[b]:edges[b + 1]]
        if not idx:
            continue
        acc = sum(labels[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    interior = sum(1 for c in confs if 0.1 < c < 0.9) / n
    return {"ece": round(ece, 4), "interior_frac": round(interior, 4), "n": n}


def _ece(engine, test, model, pipeline, tier=Tier.WHITE, n_bins: int = 10,
         calibrated: bool = True) -> float:
    return _ece_full(engine, test, model, pipeline, tier, n_bins, calibrated)["ece"]


def run_ablations(model: ModelHandle, dataset=None, pipeline: Optional[SignalPipeline] = None,
                  seeds: tuple[int, ...] = (0, 1, 2, 3)) -> dict:
    pipeline = pipeline or SignalPipeline()
    dataset = dataset if dataset is not None else build_dataset(model, pipeline, seeds=seeds)
    train, cal, test = split_dataset(dataset)

    # Use the SAME family-dropout tier schedule as train_and_evaluate: `ablate` and
    # `eval` were training differently while the docs presented them as
    # interchangeable, so their per-tier tables disagreed and a reader running the
    # other command saw different numbers.
    cycle = [Tier.WHITE, Tier.GREY, Tier.BLACK]
    train_tiers = [cycle[i % 3] for i in range(len(train))]
    full = train_engine(train, cal, model, pipeline, train_tiers=train_tiers)

    # 1. rules-only (cold-start: residual 0, no calibration/conformal fit) vs full
    rules_only = DiagnosisEngine(recommender=Recommender(model=model))
    ablation_learned = {
        "rules_only": evaluate(rules_only, test, model, Tier.WHITE, pipeline).as_dict(),
        "rules_plus_gbt": evaluate(full, test, model, Tier.WHITE, pipeline).as_dict(),
    }

    # 2. per-tier curves (full engine)
    per_tier = {t.label: evaluate(full, test, model, t, pipeline).as_dict() for t in _TIERS}

    # 3. per-family ablation. TWO distinct experiments — conflating them is what made
    #    the previous "which family is load-bearing" claim an artifact:
    #    (a) RETRAINED leave-one-family-out = the family's INFORMATION contribution.
    #    (b) missing-signal robustness = keep the full model, NaN out one family at
    #        inference = how the shipped engine copes when a family drops out live.
    per_family_retrained = {}
    per_family_robustness = {}
    for fam in (SignalFamily.PROMPT, SignalFamily.RETRIEVAL,
                SignalFamily.MECHANISTIC, SignalFamily.CONFIDENCE):
        kept = [ex for ex in default_extractors() if ex.family != fam]
        reduced = SignalPipeline(kept)
        per_family_robustness[fam.value] = evaluate(full, test, model, Tier.WHITE,
                                                    reduced).as_dict()
        refit = train_engine(train, cal, model, reduced)
        per_family_retrained[fam.value] = evaluate(refit, test, model, Tier.WHITE,
                                                   reduced).as_dict()

    # 4. calibration ECE (with the interior-mass caveat attached to the number)
    unc = _ece_full(full, test, model, pipeline, calibrated=False)
    cal_ = _ece_full(full, test, model, pipeline, calibrated=True)
    calibration = {
        "ece_uncalibrated": unc["ece"],
        "ece_calibrated": cal_["ece"],
        # Fraction of probabilities strictly inside (0.1, 0.9). Near 0 means the
        # scores are saturated and ECE is only a rescaled error rate — so the ECE
        # figure carries no calibration information and must not be quoted as such.
        "interior_mass_fraction": cal_["interior_frac"],
    }

    # 5. conformal coverage vs set size, per tier (already in per_tier metrics)
    conformal = {t: {"coverage": per_tier[t]["conformal_coverage"],
                     "mean_set_size": per_tier[t]["conformal_set_size"]} for t in per_tier}

    return {
        "sizes": {"train": len(train), "cal": len(cal), "test": len(test)},
        "ablation_learned_head": ablation_learned,
        "per_tier": per_tier,
        "per_family_retrained": per_family_retrained,
        "per_family_robustness": per_family_robustness,
        "calibration": calibration,
        "conformal": conformal,
    }


#: Continuous signal features that a real estimator (NLI, embeddings, a classifier,
#: attention readouts) would produce with observation noise, mapped to the range each
#: one is actually defined on. Structural counts (n_chunks, lengths, multihop flag)
#: are exact and left alone.
#:
#: The range matters: a blanket clamp to [0, 1.5] truncated the token entropies
#: (natural max ~1.9 nats here) and FLOORED gold_patch_effect, which is signed — a
#: negative causal effect is meaningful and was being silently destroyed.
_INF = float("inf")
_NOISY_FEATURES: dict[str, tuple[float, float]] = {
    "prompt_ambiguity": (0.0, 1.0),
    "answer_supported_by_context": (0.0, 1.0),
    "max_chunk_relevance": (0.0, 1.0),
    "mean_chunk_relevance": (0.0, 1.0),
    "gold_position_frac": (0.0, 1.0),
    "gold_recall_in_context": (0.0, 1.0),
    "semantic_entropy": (0.0, 1.0),
    "self_consistency": (0.0, 1.0),
    "mean_token_entropy": (0.0, _INF),
    "max_token_entropy": (0.0, _INF),
    "context_attention_ratio": (0.0, 1.0),
    "gold_attention_ratio": (0.0, 1.0),
    "logit_lens_answer_layer": (0.0, 1.0),
    "logit_lens_stability": (0.0, 1.0),
    "external_context_score": (0.0, 1.0),
    "parametric_knowledge_score": (0.0, 1.0),
    "gold_patch_effect": (-_INF, _INF),
}


class NoisyPipeline:
    """Wraps a pipeline and adds deterministic observation noise to the continuous
    signal features (modeling imperfect NLI/embedding/attention estimators), plus a
    small chance of flipping the correctness gate. Used to break the synthetic
    corpus's perfect separability so calibration/conformal/abstention are exercised."""

    def __init__(self, base: SignalPipeline, sigma: float):
        self.base = base
        self.sigma = sigma
        # Draw counter: keying the perturbation on (feature, prompt) ALONE made it a
        # fixed deterministic bias — the same row always got the same offset, so
        # re-observing an inference could never vary. Including the draw index makes
        # it genuine per-observation estimator noise while staying reproducible.
        self._draw = 0

    def run(self, inference, model):
        from tokentrace.models.mock import _seeded_unit

        fv = self.base.run(inference, model)
        if self.sigma <= 0:
            return fv
        self._draw += 1
        p = inference.prompt
        for name in list(fv.values):
            rng = _NOISY_FEATURES.get(name)
            if rng is None:
                continue
            lo, hi = rng
            u = _seeded_unit("fnoise", name, p, self._draw)
            val = fv.values[name] + self.sigma * (2 * u - 1) * 0.4
            fv.values[name] = min(hi, max(lo, val))
        if fv.has("is_correct") and _seeded_unit("flip", p, self._draw) < self.sigma * 0.2:
            fv.values["is_correct"] = 1.0 - fv.values["is_correct"]
        return fv


def run_robustness(
    noise_levels: tuple[float, ...] = (0.0, 0.2, 0.35),
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    pipeline: Optional[SignalPipeline] = None,
    model: Optional[ModelHandle] = None,
) -> dict:
    """Sweep signal-observation noise to show the metrics stop saturating and that
    calibration earns its keep. Labels come from the CLEAN pipeline (trustworthy);
    train + eval use the NOISY pipeline, mirroring real data where the estimators
    themselves are imperfect.

    ``model`` defaults to the deterministic mock handle, which is what every caller
    got before it was a parameter. It exists because the handle was hard-coded:
    ``tokentrace ablate --backend hf --model qwen3-4b`` ran :func:`run_ablations` on
    the real handle and this sweep on the mock, then printed both under one report
    with nothing distinguishing them.
    """
    from tokentrace.models.registry import load_model

    pipeline = pipeline or SignalPipeline()
    model = model if model is not None else load_model("mock-4b", backend="mock")
    dataset = build_dataset(model, pipeline, seeds=seeds)   # clean labels
    train, cal, test = split_dataset(dataset)

    # Same family-dropout training schedule as run_ablations / train_and_evaluate, so
    # the three entry points are genuinely comparable (they previously trained
    # differently while the docs presented their tables as interchangeable).
    cycle = [Tier.WHITE, Tier.GREY, Tier.BLACK]
    train_tiers = [cycle[i % 3] for i in range(len(train))]

    out = {}
    for nz in noise_levels:
        noisy = NoisyPipeline(pipeline, nz)
        engine = train_engine(train, cal, model, noisy, train_tiers=train_tiers)
        met = evaluate(engine, test, model, Tier.WHITE, noisy).as_dict()
        out[f"{nz:.2f}"] = {
            "diagnosis_accuracy": met["diagnosis_accuracy"],
            "top3_accuracy": met["top3_accuracy"],
            "abstention_rate": met["abstention_rate"],
            "conformal_set_size": met["conformal_set_size"],
            "debugging_time_reduction": met["debugging_time_reduction"],
            "mean_root_rank": met["mean_root_rank"],
            "ece_uncalibrated": _ece(engine, test, model, noisy, calibrated=False),
            "ece_calibrated": _ece(engine, test, model, noisy, calibrated=True),
        }
    return out
