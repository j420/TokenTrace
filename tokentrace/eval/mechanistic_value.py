"""Does mechanistic value EMERGE as mechanistically-separable cases grow?

Context: on the default corpus, removing the ENTIRE mechanistic tier costs about
0.01 diagnosis accuracy — inside the report's noise band. Two readings are
possible and this module is the experiment that separates them:

* corpus artifact — the synthetic corpus is so separable that cheap (prompt /
  retrieval / confidence) signals saturate, and mechanistic evidence would earn
  its keep on a corpus where mechanistically-separable cases are more prevalent;
* real result — the tier does not earn its keep even when the corpus favors it.

The knob is ``build_dataset(override_weight=...)``: the prevalence of the
``hallucination_override`` recipe, the one failure class that is behaviourally
identical to context dilution (gold present, answer wrong) and separable ONLY
mechanistically. For each prevalence level this sweep trains the standard
engine (same split, same family-dropout schedule as ``train_and_evaluate``) and
measures:

  (a) retrained leave-mechanistic-out accuracy delta — the mechanistic family's
      INFORMATION contribution (retrain without the family, not just mask it);
  (b) white / grey / black diagnosis accuracy — the tier curve;
  (c) F1 on the override discrimination slice — override vs dilution among the
      behaviourally-identical rows, positive = primary diagnosis HALLUCINATION.

A FLAT curve across levels says the mechanistic tier does not earn its keep even
when the corpus favors it; a RISING curve says the small default-corpus delta
was a prevalence artifact. Either answer is publishable; nothing here shades
toward one.

Scope: numbers come from the deterministic mock corpus — they are mechanism
evidence about THIS pipeline on THIS corpus, not real-model performance.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import FailureMode, SignalFamily, Tier
from tokentrace.data.synthetic import build_dataset, dataset_summary
from tokentrace.eval.benchmark import (
    evaluate,
    family_dropout_schedule,
    run_reports,
    split_dataset,
    train_engine,
)
from tokentrace.eval.metrics import predicted_primary
from tokentrace.models.base import ModelHandle
from tokentrace.signals.registry import SignalPipeline, default_extractors

#: The behaviourally-identical pair: gold present + answer wrong in both; only
#: mechanistic evidence separates them.
_AMBIGUOUS_RECIPES = ("hallucination_override", "context_dilution")


def _override_slice(engine, test, model, tier: Tier, pipeline) -> dict:
    """Override-vs-dilution discrimination on the behaviourally-identical slice.

    Positive prediction = primary diagnosis HALLUCINATION; positive class =
    ``hallucination_override`` rows. This is deliberately NOT the slice-internal
    multi-label F1: on an override-only slice every gold label set contains
    HALLUCINATION, so false positives are impossible by construction and
    precision cannot fail — a metric that cannot fail carries no evidence.
    Scoring against the dilution rows restores a real failure surface on both
    sides (calling override "dilution" costs recall; calling dilution
    "hallucination" costs precision).
    """
    rows = [li for li in test if li.injection_recipe in _AMBIGUOUS_RECIPES]
    reports = run_reports(engine, rows, model, tier, pipeline)
    tp = fp = fn = tn = 0
    for li, rep in zip(rows, reports):
        is_override = li.injection_recipe == "hallucination_override"
        pred_hal = predicted_primary(rep) is FailureMode.HALLUCINATION
        if is_override and pred_hal:
            tp += 1
        elif is_override:
            fn += 1
        elif pred_hal:
            fp += 1
        else:
            tn += 1
    n_override = tp + fn
    n_dilution = fp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_override if n_override else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_override": n_override, "n_dilution": n_dilution,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_mechanistic_value(
    levels: tuple[int, ...] = (1, 2, 4),
    seeds: tuple[int, ...] = (0, 1),
    model: Optional[ModelHandle] = None,
    pipeline: Optional[SignalPipeline] = None,
    noise: float = 0.0,
) -> dict:
    """Sweep override prevalence and measure what the mechanistic tier buys.

    ``levels`` are ``override_weight`` values (1 = the default corpus mix).
    ``seeds`` defaults to (0, 1) to keep the full sweep under a few minutes
    single-threaded; the levels/seeds actually used are echoed in the result so
    a reader never has to guess the configuration behind a number.

    Training matches the published path (``split_dataset`` +
    ``family_dropout_schedule`` + ``train_engine``) for BOTH the full and the
    leave-mechanistic-out engine, so (a) is a like-for-like retrained
    comparison. Recommendation validation is disabled — it does not feed any
    metric reported here and dominates runtime.

    ``noise``: optional signal-observation noise (the ablation harness's
    ``NoisyPipeline``, same sigma semantics as ``run_robustness``; labels always
    come from the CLEAN pipeline). The default 0.0 is the standard corpus. On
    the noise-free mock the override-slice F1 saturates at 1.0 at every level
    and tier, but the ACCURACY readouts do move (measured: level-4 white/grey
    0.9688, and the leave-one-out delta reaches -0.0312 — removing the
    mechanistic family IMPROVED white accuracy by two rows). An earlier version
    of this paragraph claimed every readout saturates; that underclaimed the
    instrument and was false in detail. The noisy setting remains the regime
    where the curve moves most; run both and read them together.
    """
    from tokentrace.models.registry import load_model

    model = model if model is not None else load_model("mock-4b", backend="mock")
    pipeline = pipeline or SignalPipeline()
    reduced = SignalPipeline(
        [ex for ex in default_extractors() if ex.family != SignalFamily.MECHANISTIC])

    def observed(base: SignalPipeline):
        if noise <= 0:
            return base
        from tokentrace.eval.ablations import NoisyPipeline

        return NoisyPipeline(base, noise)

    out: dict = {
        "config": {
            "levels": list(levels),
            "seeds": list(seeds),
            "noise": noise,
            "positive_class": "hallucination_override",
            "note": ("deterministic mock corpus; mechanism evidence only, "
                     "not real-model performance"),
        },
        "levels": {},
    }
    for level in levels:
        # Labels/gates always use the CLEAN pipeline (trustworthy supervision);
        # training and evaluation observe through the (possibly noisy) one.
        ds = build_dataset(model, pipeline, seeds=seeds, override_weight=level)
        train, cal, test = split_dataset(ds)
        schedule = family_dropout_schedule(len(train))
        obs_full = observed(pipeline)
        obs_reduced = observed(reduced)

        full = train_engine(train, cal, model, obs_full, train_tiers=schedule,
                            validate_recommendations=False)
        per_tier = {
            t.label: evaluate(full, test, model, t, obs_full).diagnosis_accuracy
            for t in (Tier.WHITE, Tier.GREY, Tier.BLACK)
        }

        loo = train_engine(train, cal, model, obs_reduced, train_tiers=schedule,
                           validate_recommendations=False)
        loo_white = evaluate(loo, test, model, Tier.WHITE, obs_reduced).diagnosis_accuracy

        out["levels"][str(level)] = {
            "override_weight": level,
            "n": len(ds),
            "n_train": len(train), "n_test": len(test),
            "by_recipe": dataset_summary(ds)["by_recipe"],
            "per_tier_accuracy": {k: round(v, 4) for k, v in per_tier.items()},
            "white_minus_grey": round(per_tier["white"] - per_tier["grey"], 4),
            "leave_mechanistic_out": {
                "white_full": round(per_tier["white"], 4),
                "white_without_mechanistic": round(loo_white, 4),
                "delta": round(per_tier["white"] - loo_white, 4),
            },
            "override_slice": {
                "white": _override_slice(full, test, model, Tier.WHITE, obs_full),
                "grey": _override_slice(full, test, model, Tier.GREY, obs_full),
                "black": _override_slice(full, test, model, Tier.BLACK, obs_full),
            },
        }
    return out
