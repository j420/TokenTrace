"""Metrics for the diagnosis benchmark, aligned to the proposal's targets:

* diagnosis accuracy  (>= 80%)  — primary cause correct (clean => abstain)
* Top-3 accuracy      (>= 90%)  — gold root among the 3 HIGHEST-PROBABILITY modes
  (independent of the conformal set and of the abstention gate)
* conformal coverage / mean set size — the adaptive set, reported separately
* per-mode precision / recall / F1 (multi-label detection)
* recommendation precision (>= 75%) — distinct simulated interventions that actually
  correct the answer, plus the count of negatives (a precision with no observed
  negative is vacuous, so the negative count is reported alongside)
* abstention, split into healthy_abstention_rate (correctly quiet on a clean row)
  and declined_rate (refused to diagnose a genuine failure) — the combined
  abstention_rate mixes a good behaviour with a bad one and is kept only for
  continuity
* debugging_time_reduction — mean rank of the true root vs a per-row unaided
  baseline. CEILING: a perfect ranker scores 1 - 1/((K+1)/2) = 0.667 at K=5, so this
  is a ranking proxy and is NOT comparable to a wall-clock target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tokentrace.core.types import ALL_MODES, DiagnosisReport, FailureMode, LabeledInference
from tokentrace.engine.causal import _DEPTH
from tokentrace.recommend.recommender import INTERVENTIONS


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
    recommendation_n: int = 0                    # distinct simulated interventions scored
    recommendation_negatives: int = 0            # of those, how many failed to fix the answer
    abstention_rate: float = 0.0                 # all rows (mixes the two below)
    healthy_abstention_rate: float = 0.0         # correct: abstained on a clean row
    declined_rate: float = 0.0                   # declined to diagnose a genuine failure
    debugging_time_reduction: float = 0.0        # vs. an unaided (random-order) baseline
    mean_root_rank: float = 0.0                  # mean rank of the true root in the differential
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
            "recommendation_negatives": self.recommendation_negatives,
            "abstention_rate": round(self.abstention_rate, 4),
            "healthy_abstention_rate": round(self.healthy_abstention_rate, 4),
            "declined_rate": round(self.declined_rate, 4),
            "debugging_time_reduction": round(self.debugging_time_reduction, 4),
            "mean_root_rank": round(self.mean_root_rank, 4),
            "per_mode": self.per_mode,
        }


def compute_metrics(
    reports: list[DiagnosisReport],
    labeled: list[LabeledInference],
    present_threshold: float = 0.5,
) -> Metrics:
    n = len(reports)
    diag_correct = top3_correct = 0
    clean_total = healthy_abstained = failure_total = declined = 0
    conf_cov = conf_size_sum = conf_total = 0
    rec_hits = rec_total = rec_neg = 0
    root_ranks: list[float] = []      # rank of the true root in the ranked differential
    baseline_ranks: list[float] = []  # per-row unaided expectation ((K_i + 1) / 2)
    # multi-label counters
    tp = {m: 0 for m in ALL_MODES}
    fp = {m: 0 for m in ALL_MODES}
    fn = {m: 0 for m in ALL_MODES}
    support = {m: 0 for m in ALL_MODES}

    for report, li in zip(reports, labeled):
        gp_set = gold_primary_set(li)
        pp = predicted_primary(report)
        top3 = report.ranked_modes()[:3]

        # primary accuracy (any co-equal root counts); clean -> correct iff abstained
        if not gp_set:
            clean_total += 1
            healthy_abstained += int(report.abstained)
            diag_correct += int(report.abstained)
            top3_correct += int(report.abstained)
        else:
            failure_total += 1
            declined += int(report.abstained)
            diag_correct += int(pp in gp_set)
            # TRUE Top-3: gold root among the 3 highest-probability modes. NOT gated
            # on abstention and NOT the conformal set (reported separately below).
            top3_correct += int(any(m in top3 for m in gp_set))
            conf_total += 1
            conf_cov += int(any(m in report.conformal_set for m in gp_set))
            conf_size_sum += len(report.conformal_set)
            # Debugging-time proxy: how far down the ranked differential a developer
            # must read to reach the true root (rank 1 = first hypothesis to check).
            # Abstained rows are charged the FULL unaided cost for their own row: the
            # tool declined to rank, so it saved the developer nothing. Excluding them
            # (the old behaviour) made the metric unable to fall when the engine got
            # less certain — black and white tiers scored bit-identically.
            ranked_all = report.ranked_modes()
            # Candidate modes for THIS row = the STRUCTURALLY POSSIBLE ones. A non-RAG
            # inference cannot have a retrieval-side cause, so an unaided developer
            # would never check those two. NB: do not derive this from "probability >
            # 0", because a confidently-excluded mode rounds to exactly 0.0 and would
            # shrink the baseline to ~1, making the reduction look like zero.
            n_candidates = len(ALL_MODES) if li.inference.is_rag else len(ALL_MODES) - 2
            baseline_ranks.append((n_candidates + 1) / 2)
            if report.abstained:
                root_ranks.append(float(n_candidates))
            else:
                root_ranks.append(
                    float(min(ranked_all.index(m) + 1 for m in gp_set if m in ranked_all)))

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
        # Counted per DISTINCT INTERVENTION per row: several templates can resolve to
        # the same underlying fix (ground_or_abstain IS add_gold_context), so keying
        # on (mode, template) double-credited one simulated intervention whenever it
        # attached to both a root and its sequela.
        if not report.abstained:
            seen_actions: set = set()
            for d in report.diagnoses:
                for r in d.recommendations:
                    if r.validated is None or r.targets_mode not in gold:
                        continue
                    fn_id = id(INTERVENTIONS.get(r.action)) if INTERVENTIONS.get(r.action) else r.action
                    if fn_id in seen_actions:
                        continue
                    seen_actions.add(fn_id)
                    rec_total += 1
                    rec_hits += int(r.validated)
                    rec_neg += int(not r.validated)

    per_mode = {}
    for m in ALL_MODES:
        p = tp[m] / (tp[m] + fp[m]) if (tp[m] + fp[m]) else 0.0
        r = tp[m] / (tp[m] + fn[m]) if (tp[m] + fn[m]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_mode[m.value] = {"precision": round(p, 4), "recall": round(r, 4),
                             "f1": round(f1, 4), "support": support[m]}

    # Debugging-time reduction vs an unaided baseline that inspects the candidate
    # failure modes in no particular order (expected rank (K_i + 1) / 2 PER ROW —
    # masked modes are not hypotheses anyone would check, so K is not a constant 5).
    # With TokenTrace the developer reads the ranked differential top-down, so the
    # mean rank of the true root is the expected number of hypotheses checked.
    # NOTE the ceiling: a perfect ranker scores 1 - 1/((K+1)/2), i.e. 0.667 at K=5.
    mean_root_rank = sum(root_ranks) / len(root_ranks) if root_ranks else 0.0
    mean_baseline = sum(baseline_ranks) / len(baseline_ranks) if baseline_ranks else 0.0
    dt_reduction = max(0.0, 1.0 - mean_root_rank / mean_baseline) if mean_baseline else 0.0

    return Metrics(
        n=n,
        diagnosis_accuracy=diag_correct / n if n else 0.0,
        top3_accuracy=top3_correct / n if n else 0.0,
        conformal_coverage=conf_cov / conf_total if conf_total else 0.0,
        conformal_set_size=conf_size_sum / conf_total if conf_total else 0.0,
        recommendation_precision=rec_hits / rec_total if rec_total else 0.0,
        recommendation_n=rec_total,
        recommendation_negatives=rec_neg,
        abstention_rate=(healthy_abstained + declined) / n if n else 0.0,
        healthy_abstention_rate=healthy_abstained / clean_total if clean_total else 0.0,
        declined_rate=declined / failure_total if failure_total else 0.0,
        debugging_time_reduction=dt_reduction,
        mean_root_rank=mean_root_rank,
        per_mode=per_mode,
    )
