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


def gold_primary(li: LabeledInference) -> Optional[FailureMode]:
    """The root cause among an example's labels (most upstream in the DAG)."""
    if not li.labels:
        return None
    min_depth = min(_DEPTH[m] for m in li.labels)
    roots = [m for m in li.labels if _DEPTH[m] == min_depth]
    # stable order by ALL_MODES
    return next(m for m in ALL_MODES if m in roots)


def predicted_primary(report: DiagnosisReport) -> Optional[FailureMode]:
    if report.abstained:
        return None
    p = report.primary
    return p.mode if p else None


@dataclass
class Metrics:
    n: int = 0
    diagnosis_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    recommendation_precision: float = 0.0
    recommendation_n: int = 0
    abstention_rate: float = 0.0
    per_mode: dict = field(default_factory=dict)   # mode -> {precision, recall, f1, support}

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "diagnosis_accuracy": round(self.diagnosis_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
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
    rec_hits = rec_total = 0
    # multi-label counters
    tp = {m: 0 for m in ALL_MODES}
    fp = {m: 0 for m in ALL_MODES}
    fn = {m: 0 for m in ALL_MODES}
    support = {m: 0 for m in ALL_MODES}

    for report, li in zip(reports, labeled):
        gp = gold_primary(li)
        pp = predicted_primary(report)
        if report.abstained:
            abstained += 1

        # primary accuracy
        if gp is None:
            diag_correct += int(report.abstained)
            top3_correct += int(report.abstained)
        else:
            diag_correct += int(pp == gp)
            top3_correct += int(gp in report.conformal_set and not report.abstained)

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

        # recommendation precision (only where a validation was run)
        for d in report.diagnoses:
            for r in d.recommendations:
                if r.validated is not None:
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
        recommendation_precision=rec_hits / rec_total if rec_total else 0.0,
        recommendation_n=rec_total,
        abstention_rate=abstained / n if n else 0.0,
        per_mode=per_mode,
    )
