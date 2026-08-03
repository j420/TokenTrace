"""Metrics for the diagnosis benchmark, aligned to the proposal's targets:

* diagnosis accuracy  (>= 80%)  — primary cause correct (clean => abstain)
* Top-3 accuracy      (>= 90%)  — gold primary inside the conformal set
* per-mode precision / recall / F1 (multi-label detection)
* recommendation precision (>= 75%) — fixes that actually correct the answer
* abstention rate, per-tier accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tokentrace.core.types import ALL_MODES, DiagnosisReport, FailureMode, LabeledInference
from tokentrace.engine.causal import _DEPTH


def gold_primary_set(li: LabeledInference) -> list[FailureMode]:
    """All equally-upstream root causes among an example's labels. Any of these is
    an acceptable primary diagnosis (a single example can have co-equal roots, e.g.
    ambiguity + retrieval both at DAG depth 0)."""
    if not li.labels:
        return []
    min_depth = min(_DEPTH[m] for m in li.labels)
    return [m for m in ALL_MODES if m in set(li.labels) and _DEPTH[m] == min_depth]


def gold_primary(li: LabeledInference) -> Optional[FailureMode]:
    """A single canonical root (first co-root) — used for conformal fitting."""
    s = gold_primary_set(li)
    return s[0] if s else None


def predicted_primary(report: DiagnosisReport) -> Optional[FailureMode]:
    if report.abstained:
        return None
    p = report.primary
    return p.mode if p else None


@dataclass
class Metrics:
    n: int = 0
    diagnosis_accuracy: float = 0.0
    top3_accuracy: float = 0.0                    # gold root in the 3 highest-prob modes
    conformal_coverage: float = 0.0              # gold root inside the adaptive set
    conformal_set_size: float = 0.0              # mean adaptive-set size (on failures)
    recommendation_precision: float = 0.0
    recommendation_n: int = 0
    abstention_rate: float = 0.0
    per_mode: dict = field(default_factory=dict)   # mode -> {precision, recall, f1, support}

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "diagnosis_accuracy": round(self.diagnosis_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "conformal_coverage": round(self.conformal_coverage, 4),
            "conformal_set_size": round(self.conformal_set_size, 4),
            "recommendation_precision": round(self.recommendation_precision, 4),
            "recommendation_n": self.recommendation_n,
            "abstention_rate": round(self.abstention_rate, 4),
            "per_mode": self.per_mode,
        }


def compute_metrics(
    reports: list[DiagnosisReport],
    labeled: list[LabeledInference],
    present_threshold: float = 0.5,
) -> Metrics:
    n = len(reports)
    diag_correct = top3_correct = abstained = 0
    conf_cov = conf_size_sum = conf_total = 0
    rec_hits = rec_total = 0
    # multi-label counters
    tp = {m: 0 for m in ALL_MODES}
    fp = {m: 0 for m in ALL_MODES}
    fn = {m: 0 for m in ALL_MODES}
    support = {m: 0 for m in ALL_MODES}

    for report, li in zip(reports, labeled):
        gp_set = gold_primary_set(li)
        pp = predicted_primary(report)
        if report.abstained:
            abstained += 1
        top3 = report.ranked_modes()[:3]

        # primary accuracy (any co-equal root counts); clean -> correct iff abstained
        if not gp_set:
            diag_correct += int(report.abstained)
            top3_correct += int(report.abstained)
        else:
            diag_correct += int(pp in gp_set)
            # TRUE Top-3: gold root among the 3 highest-probability modes. NOT gated
            # on abstention and NOT the conformal set (reported separately below).
            top3_correct += int(any(m in top3 for m in gp_set))
            conf_total += 1
            conf_cov += int(any(m in report.conformal_set for m in gp_set))
            conf_size_sum += len(report.conformal_set)

        # multi-label detection (predicted-present = prob >= threshold)
        gold = set(li.labels)
        pred = {d.mode for d in report.diagnoses if d.probability >= present_threshold}
        for m in ALL_MODES:
            if m in gold:
                support[m] += 1
            if m in pred and m in gold:
                tp[m] += 1
            elif m in pred and m not in gold:
                fp[m] += 1
            elif m not in pred and m in gold:
                fn[m] += 1

        # Recommendation precision: only score fixes that (a) come from a
        # non-abstained report and (b) target a mode that is actually a true label
        # — otherwise a fix for a wrongly-diagnosed mode could get spurious credit.
        if not report.abstained:
            for d in report.diagnoses:
                for r in d.recommendations:
                    if r.validated is None or r.targets_mode not in gold:
                        continue
                    rec_total += 1
                    rec_hits += int(r.validated)

    per_mode = {}
    for m in ALL_MODES:
        p = tp[m] / (tp[m] + fp[m]) if (tp[m] + fp[m]) else 0.0
        r = tp[m] / (tp[m] + fn[m]) if (tp[m] + fn[m]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_mode[m.value] = {"precision": round(p, 4), "recall": round(r, 4),
                             "f1": round(f1, 4), "support": support[m]}

    return Metrics(
        n=n,
        diagnosis_accuracy=diag_correct / n if n else 0.0,
        top3_accuracy=top3_correct / n if n else 0.0,
        conformal_coverage=conf_cov / conf_total if conf_total else 0.0,
        conformal_set_size=conf_size_sum / conf_total if conf_total else 0.0,
        recommendation_precision=rec_hits / rec_total if rec_total else 0.0,
        recommendation_n=rec_total,
        abstention_rate=abstained / n if n else 0.0,
        per_mode=per_mode,
    )
