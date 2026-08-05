"""Fleet triage: turn a pile of diagnosed traces into a ranked "fix this first" list.

TokenTrace diagnoses ONE inference. A developer with a week of production logs has
thousands, and the only question that matters is *which cluster of failures do I
attack first*. This module answers that, in three layers:

1. **What is failing?**  A distribution over the five failure modes, taken from the
   PRIMARY ROOT of each report. Healthy, abstained and errored traces are held in
   their own buckets and never enter that distribution.
2. **Which cluster first?**  Traces are grouped by ``(primary failure mode, headline
   evidence signal)``. The mode alone is too coarse to act on: "retrieval failure
   because ``ret.gold_absent`` fired" (the answer is not in the corpus at all → fix
   the index) and "retrieval failure because ``ret.irrelevant`` fired" (the retriever
   ranked nothing relevant → fix the query rewrite) are the same mode with different
   engineering work behind them. The headline signal is the engine's own strongest
   *substantive* evidence line (:attr:`Diagnosis.headline_evidence`), so the split is
   the engine's reasoning, not a heuristic layered on top.
3. **What do I do?**  Per cluster: the dominant recommended action, how many traces it
   covers, exemplar ids to spot-check, and — separately and carefully — whether that
   fix was ever *verified*.

Honesty contract — which numbers are MEASURED and which are DESCRIPTIVE
----------------------------------------------------------------------
This repo has twice shipped metrics that could not fail. Nothing here is allowed to
be one, so every field falls into exactly one of three classes:

* **MEASURED** (a real experiment ran and could have come out the other way) —
  ``fix_validated`` / ``fix_refuted`` and the derived :attr:`TriageCluster.measured_impact`.
  These come only from :class:`~tokentrace.recommend.recommender.Recommendation`
  objects where ``validated is not None``, i.e. the intervention was actually applied,
  the model was actually re-run, and the new answer was actually checked against a
  ground-truth reference. ``fix_refuted`` is the observed-negative count; a validation
  rate quoted without it is vacuous.
* **UNKNOWN** — ``fix_unknown`` and ``measured_impact is None``. A trace with no
  ground truth CANNOT have its fix verified, so it lands here. This is the default,
  and it is deliberately not zero and not an estimate: "we did not measure it" and
  "we measured it and it was zero" are different claims.
* **DESCRIPTIVE** (counts and means over what the engine *said*, carrying no claim
  that the engine was right) — every other number: ``n``, ``share``, the mode
  distribution, ``mean_diagnostic_confidence``, ``mean_headline_logodds``,
  ``priority_score``, the abstention/healthy/failure rates.

There is deliberately **no diagnosis-accuracy number in this module**. Triage runs on
unlabeled production traces; nothing here can tell you the engine was right. For that
you need labels — see :mod:`tokentrace.eval.metrics`.

Invariant worth checking in review: within a cluster,
``fix_validated + fix_refuted + fix_unknown == n``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from tokentrace.core.types import (
    ALL_MODES,
    DiagnosisReport,
    FailureMode,
    Inference,
    Recommendation,
    Tier,
)

#: Fallback when the engine emitted no substantive evidence line at all (possible on a
#: cold-start report whose ledger is just the base-rate prior). Kept as an explicit
#: sentinel rather than ``None`` so it survives JSON and sorts predictably.
NO_SIGNAL = "(no evidence)"

#: Engine default for :attr:`DiagnosisEngine.abstain_primary`; used only when the
#: supplied facade does not expose its own threshold.
DEFAULT_DETECTION_THRESHOLD = 0.40


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def _fin(x: Any) -> Any:
    """NaN/inf -> None, mirroring :func:`tokentrace.core.serialize._fin`.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens by default, which are not
    RFC 8259 and break any strict consumer. Means over odd inputs (an infinite
    log-odds contribution from a degenerate model) are exactly where a non-finite
    float would sneak into a triage report, so it is scrubbed at the boundary.
    """
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


# --------------------------------------------------------------------------- #
# Report pieces
# --------------------------------------------------------------------------- #
@dataclass
class TriageFailure:
    """One trace that raised during analysis.

    A 10k-trace run must not die because trace #4173 has a ``None`` where a string
    belongs. Failures are collected, counted and reported so they are visible rather
    than silently shrinking the denominator.
    """

    index: int                       # position in the input stream
    trace_id: Optional[str]
    error_type: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "trace_id": self.trace_id,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass
class TriageCluster:
    """One actionable group: same root cause, same driving evidence.

    ``dominant_mode`` and ``headline_signal`` together *are* the cluster key, so
    ``dominant_mode`` is constant within a cluster by construction — it is carried as
    a field because callers rendering a single cluster should not have to unpack a
    key tuple to learn what failed.
    """

    dominant_mode: FailureMode
    headline_signal: str
    headline_source: str = "rule"        # "rule" | "learned" | "prior" | "aggregate"
    headline_family: Optional[str] = None
    n: int = 0
    share: float = 0.0                   # of DIAGNOSED traces, not of all input
    mean_diagnostic_confidence: float = 0.0
    mean_primary_probability: float = 0.0
    mean_headline_logodds: float = 0.0   # signed: a headline line can OPPOSE the mode
    dominant_action: Optional[str] = None
    dominant_action_n: int = 0           # traces the dominant action covers
    action_counts: dict[str, int] = field(default_factory=dict)
    # --- verification split (see the module docstring's honesty contract) --- #
    fix_validated: int = 0               # MEASURED: intervention applied, answer corrected
    fix_refuted: int = 0                 # MEASURED: intervention applied, answer still wrong
    fix_unknown: int = 0                 # never scored (no ground truth / nothing to fix)
    n_with_ground_truth: int = 0         # why impact is (un)knowable for this cluster
    n_non_rag: int = 0
    exemplar_ids: list[str] = field(default_factory=list)
    priority_score: float = 0.0
    rank: int = 0

    # ------------------------------------------------------------------ #
    @property
    def key(self) -> tuple[str, str]:
        return (self.dominant_mode.value, self.headline_signal)

    @property
    def measured_impact(self) -> Optional[int]:
        """Traces whose recommended fix DEMONSTRABLY corrected the answer — or ``None``.

        ``None`` means "not measured", and it is the honest answer whenever no
        recommendation in this cluster was ever scored (which is the normal case in
        production, where there is no ground truth to score against). Returning 0
        there would assert that the fix was tried and did nothing; that assertion has
        not been earned. This is the ONLY number in a cluster that may be quoted as
        "fixing this corrects N traces".
        """
        if self.fix_validated + self.fix_refuted == 0:
            return None
        return self.fix_validated

    @property
    def impact_status(self) -> str:
        return "measured" if self.measured_impact is not None else "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "dominant_mode": self.dominant_mode.value,
            "headline_signal": self.headline_signal,
            "headline_source": self.headline_source,
            "headline_family": self.headline_family,
            "n": self.n,
            "share": _fin(round(self.share, 4)),
            "mean_diagnostic_confidence": _fin(round(self.mean_diagnostic_confidence, 4)),
            "mean_primary_probability": _fin(round(self.mean_primary_probability, 4)),
            "mean_headline_logodds": _fin(round(self.mean_headline_logodds, 4)),
            "priority_score": _fin(round(self.priority_score, 4)),
            "dominant_action": self.dominant_action,
            "dominant_action_n": self.dominant_action_n,
            "action_counts": dict(self.action_counts),
            "fix_validated": self.fix_validated,
            "fix_refuted": self.fix_refuted,
            "fix_unknown": self.fix_unknown,
            "measured_impact": self.measured_impact,     # None == not measured
            "impact_status": self.impact_status,
            "n_with_ground_truth": self.n_with_ground_truth,
            "n_non_rag": self.n_non_rag,
            "exemplar_ids": list(self.exemplar_ids),
        }


@dataclass
class TriageReport:
    """Fleet-level answer to "what is failing and which cluster do I fix first".

    Three top-level buckets partition the successfully analyzed traces and are kept
    strictly separate: ``n_diagnosed`` (a root cause was named), ``n_healthy``
    (nothing crossed the engine's detection threshold) and ``n_abstained`` (something
    did cross it, but the engine was not confident enough to rank a primary cause).
    Only ``n_diagnosed`` traces enter ``mode_counts`` and the clusters — declining to
    diagnose is not evidence of health, and treating it as such is how an
    observability tool ends up reporting a clean fleet that is on fire.
    """

    n_input: int = 0
    n_analyzed: int = 0
    n_failed: int = 0
    n_diagnosed: int = 0
    n_healthy: int = 0
    n_abstained: int = 0
    mode_counts: dict[FailureMode, int] = field(default_factory=dict)
    clusters: list[TriageCluster] = field(default_factory=list)
    failures: list[TriageFailure] = field(default_factory=list)
    tier: Optional[Tier] = None
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    ranking_criterion: str = "n * mean_diagnostic_confidence"
    n_with_ground_truth: int = 0
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def abstention_rate(self) -> float:
        """Share of analyzed traces the engine declined to diagnose, healthy ones included.

        This is the headline honesty number: it matches ``DiagnosisReport.abstained``
        one-for-one, so a run that abstained on 60% of its input says so loudly rather
        than quietly reporting a five-mode distribution over the other 40%.
        """
        return _rate(self.n_healthy + self.n_abstained, self.n_analyzed)

    @property
    def healthy_rate(self) -> float:
        return _rate(self.n_healthy, self.n_analyzed)

    @property
    def declined_rate(self) -> float:
        """Abstentions that were NOT "looks healthy" — genuine insufficient evidence."""
        return _rate(self.n_abstained, self.n_analyzed)

    @property
    def failure_rate(self) -> float:
        return _rate(self.n_failed, self.n_input)

    def mode_distribution(self) -> dict[FailureMode, float]:
        """Shares over the DIAGNOSED traces only.

        Returns ``{}`` when nothing was diagnosed: a distribution over an empty
        population is undefined, and emitting five 0.0 shares invites the reader to
        conclude "no retrieval failures" when the truth is "no diagnoses at all".
        """
        if self.n_diagnosed <= 0:
            return {}
        return {m: self.mode_counts.get(m, 0) / self.n_diagnosed for m in ALL_MODES}

    def top(self, k: int = 5) -> list[TriageCluster]:
        return self.clusters[:k]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view. Every float passes through :func:`_fin` first."""
        return {
            "n_input": self.n_input,
            "n_analyzed": self.n_analyzed,
            "n_failed": self.n_failed,
            "n_diagnosed": self.n_diagnosed,
            "n_healthy": self.n_healthy,
            "n_abstained": self.n_abstained,
            "n_with_ground_truth": self.n_with_ground_truth,
            "abstention_rate": _fin(round(self.abstention_rate, 4)),
            "healthy_rate": _fin(round(self.healthy_rate, 4)),
            "declined_rate": _fin(round(self.declined_rate, 4)),
            "failure_rate": _fin(round(self.failure_rate, 4)),
            "tier": self.tier.label if self.tier is not None else None,
            "detection_threshold": _fin(self.detection_threshold),
            "ranking_criterion": self.ranking_criterion,
            "mode_counts": {m.value: self.mode_counts.get(m, 0) for m in ALL_MODES},
            "mode_share": {m.value: _fin(round(v, 4))
                           for m, v in self.mode_distribution().items()},
            "clusters": [c.to_dict() for c in self.clusters],
            "failures": [f.to_dict() for f in self.failures],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Per-trace extraction
# --------------------------------------------------------------------------- #
def _top1(report: DiagnosisReport) -> float:
    """Highest marginal in the differential.

    NOT ``report.primary.probability``: the causal resolver may crown an upstream
    parent whose probability is lower than the strongest mode, while the engine's own
    abstention gate keys off the maximum. Using the wrong one here would misfile
    traces between the healthy and abstained buckets.
    """
    return max((d.probability for d in report.diagnoses), default=0.0)


def _headline(report: DiagnosisReport) -> tuple[str, str, Optional[str], float]:
    """(signal, source, family, contribution_logodds) of the primary's headline line.

    ``Diagnosis.headline_evidence`` skips the bookkeeping rows (base-rate prior and
    the aggregated remainder), so this is the strongest line a human would call "the
    reason" — which is exactly what should split one failure mode into separately
    fixable clusters.
    """
    primary = report.primary
    ev = primary.headline_evidence if primary is not None else None
    if ev is None:
        return NO_SIGNAL, "none", None, 0.0
    return ev.signal, ev.source, ev.family.value, float(ev.contribution_logodds)


def _chosen_recommendation(report: DiagnosisReport) -> Optional[Recommendation]:
    """The single fix a developer should apply first for this trace.

    Ordered by ``Recommendation.priority`` (which encodes the causal role: treat the
    root before the sequela), then by the rank of the diagnosis carrying it, then by
    template order. Exactly one recommendation per trace is scored, which sidesteps
    the double-counting trap ``eval.metrics`` had to fix: several action keys resolve
    to the same underlying callable in ``recommend.INTERVENTIONS``
    (``ground_or_abstain`` *is* ``add_gold_context``), so counting per (mode, template)
    would credit one simulated intervention twice.
    """
    best: Optional[Recommendation] = None
    best_key: Optional[tuple[int, int, int]] = None
    for rank, diag in enumerate(report.diagnoses):
        for i, rec in enumerate(diag.recommendations):
            k = (rec.priority, rank, i)
            if best_key is None or k < best_key:
                best_key, best = k, rec
    return best


# --------------------------------------------------------------------------- #
# Streaming accumulator (O(clusters) memory, not O(traces))
# --------------------------------------------------------------------------- #
@dataclass
class _Accum:
    mode: FailureMode
    signal: str
    source: str
    family: Optional[str]
    confidences: list[float] = field(default_factory=list)
    primaries: list[float] = field(default_factory=list)
    logodds: list[float] = field(default_factory=list)
    actions: Counter = field(default_factory=Counter)
    fix_validated: int = 0
    fix_refuted: int = 0
    fix_unknown: int = 0
    n_with_gt: int = 0
    n_non_rag: int = 0
    exemplars: list[str] = field(default_factory=list)

    def add(self, trace: Inference, report: DiagnosisReport, logodds: float,
            trace_key: str, max_exemplars: int) -> None:
        primary = report.primary
        self.confidences.append(float(report.diagnostic_confidence))
        self.primaries.append(float(primary.probability) if primary is not None else 0.0)
        self.logodds.append(logodds)
        self.n_with_gt += int(trace.has_ground_truth)
        self.n_non_rag += int(not trace.is_rag)
        if len(self.exemplars) < max_exemplars:
            self.exemplars.append(trace_key)

        rec = _chosen_recommendation(report)
        if rec is None:
            # No applicable fix at all -> nothing to verify, so it is unknown, not a
            # failed fix. (Happens when no diagnosis cleared the recommender's own
            # probability floor.)
            self.fix_unknown += 1
            return
        self.actions[rec.action] += 1
        if rec.validated is True:
            self.fix_validated += 1
        elif rec.validated is False:
            self.fix_refuted += 1
        else:
            self.fix_unknown += 1

    def finish(self, n_diagnosed: int, max_exemplars: int) -> TriageCluster:
        n = len(self.confidences)
        mean_conf = _mean(self.confidences)
        dominant_action, dominant_n = (None, 0)
        if self.actions:
            # most_common ties arbitrarily; sort explicitly so the report is stable
            # across runs and platforms.
            dominant_action, dominant_n = max(
                sorted(self.actions.items()), key=lambda kv: kv[1])
        return TriageCluster(
            dominant_mode=self.mode,
            headline_signal=self.signal,
            headline_source=self.source,
            headline_family=self.family,
            n=n,
            share=_rate(n, n_diagnosed),
            mean_diagnostic_confidence=mean_conf,
            mean_primary_probability=_mean(self.primaries),
            mean_headline_logodds=_mean(self.logodds),
            dominant_action=dominant_action,
            dominant_action_n=dominant_n,
            action_counts=dict(sorted(self.actions.items())),
            fix_validated=self.fix_validated,
            fix_refuted=self.fix_refuted,
            fix_unknown=self.fix_unknown,
            n_with_ground_truth=self.n_with_gt,
            n_non_rag=self.n_non_rag,
            exemplar_ids=self.exemplars[:max_exemplars],
            # RANKING CRITERION -- see triage()'s docstring for the justification.
            priority_score=n * mean_conf,
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def triage(
    traces: Iterable[Inference],
    tt,
    tier: Optional[Tier] = None,
    max_exemplars: int = 3,
    detection_threshold: Optional[float] = None,
    max_failures_recorded: int = 50,
) -> TriageReport:
    """Diagnose every trace and aggregate into a ranked, actionable fix list.

    Parameters
    ----------
    traces:
        Any iterable of :class:`Inference` — consumed lazily, so a 10k-row JSONL
        generator never has to be materialized. Note that
        :meth:`TokenTrace.analyze` fills in ``generated_answer`` in place when it is
        empty, so traces are mutated exactly as they would be by a single-trace run.
    tt:
        A :class:`~tokentrace.api.TokenTrace` facade (anything exposing
        ``analyze(inference, tier=...)``).
    tier:
        Optional tier override applied to every trace.
    detection_threshold:
        Probability below which an abstention is read as "healthy" rather than
        "declined". Defaults to the engine's own ``abstain_primary``, so the split
        reproduces the engine's own "appears healthy" note instead of inventing a
        second, disagreeing threshold.

    Ranking criterion
    -----------------
    Clusters are ranked by ``priority_score = n * mean_diagnostic_confidence``: the
    confidence-weighted size, i.e. the number of traces the engine would stand behind
    assigning to this cluster. Rationale, and the alternatives rejected:

    * It is monotone in both quantities a triage decision actually trades off — how
      many traces are affected, and how sure we are they belong together — and it is
      linear, so it does not invent a scale or a threshold nobody can defend.
    * It deliberately does **not** rank by validated impact. Validation needs ground
      truth; ranking by it would sink every production cluster (where there is none)
      to the bottom, which is exactly backwards for an observability tool. Verified
      impact is reported per cluster instead, where it can be read honestly.
    * It deliberately does **not** model severity, cost, or user harm. TokenTrace
      cannot observe any of those, and a made-up weighting would be unfalsifiable.

    ``priority_score`` is a DESCRIPTIVE ordering aid, not a measured quantity. It says
    "look here first", never "fixing this corrects N traces" — for that, and only that,
    see :attr:`TriageCluster.measured_impact`.

    Robustness
    ----------
    Empty input, a single trace, every trace abstaining, non-RAG traces, traces with
    no ground truth, and traces that raise are all handled. A raising trace is
    recorded in :attr:`TriageReport.failures` and the run continues;
    ``KeyboardInterrupt``/``SystemExit`` are deliberately *not* caught.
    """
    if detection_threshold is None:
        engine = getattr(tt, "engine", None)
        detection_threshold = float(
            getattr(engine, "abstain_primary", DEFAULT_DETECTION_THRESHOLD))

    accums: dict[tuple[str, str], _Accum] = {}
    mode_counts: dict[FailureMode, int] = {m: 0 for m in ALL_MODES}
    failures: list[TriageFailure] = []
    n_input = n_analyzed = n_failed = n_diagnosed = n_healthy = n_abstained = 0
    n_with_gt = 0
    run_tier: Optional[Tier] = tier

    for index, trace in enumerate(traces):
        n_input += 1
        try:
            # Only pass `tier` when the caller actually overrode it, so the minimal
            # contract a facade must satisfy stays `analyze(inference)`.
            report = tt.analyze(trace, tier=tier) if tier is not None else tt.analyze(trace)
        except Exception as exc:  # noqa: BLE001 - one bad trace must not kill the run
            n_failed += 1
            if len(failures) < max_failures_recorded:
                failures.append(TriageFailure(
                    index=index,
                    trace_id=getattr(trace, "id", None),
                    error_type=type(exc).__name__,
                    error=str(exc)[:300],
                ))
            continue

        n_analyzed += 1
        n_with_gt += int(trace.has_ground_truth)
        if run_tier is None:
            run_tier = report.tier

        if report.abstained:
            # Two very different reasons hide behind one flag; keep them apart. Below
            # the detection threshold the engine's own note reads "appears healthy";
            # above it, the engine saw something and refused to rank it.
            if _top1(report) < detection_threshold:
                n_healthy += 1
            else:
                n_abstained += 1
            continue

        primary = report.primary
        if primary is None:            # defensive: a report with no diagnoses at all
            n_abstained += 1
            continue

        n_diagnosed += 1
        mode_counts[primary.mode] = mode_counts.get(primary.mode, 0) + 1
        signal, source, family, logodds = _headline(report)
        key = (primary.mode.value, signal)
        acc = accums.get(key)
        if acc is None:
            acc = accums[key] = _Accum(primary.mode, signal, source, family)
        acc.add(trace, report, logodds,
                trace_key=trace.id if trace.id else f"#{index}",
                max_exemplars=max_exemplars)

    clusters = [a.finish(n_diagnosed, max_exemplars) for a in accums.values()]
    # Fully deterministic order: score, then size, then confidence, then the key.
    clusters.sort(key=lambda c: (-c.priority_score, -c.n, -c.mean_diagnostic_confidence,
                                 c.dominant_mode.value, c.headline_signal))
    for i, c in enumerate(clusters, start=1):
        c.rank = i

    out = TriageReport(
        n_input=n_input, n_analyzed=n_analyzed, n_failed=n_failed,
        n_diagnosed=n_diagnosed, n_healthy=n_healthy, n_abstained=n_abstained,
        mode_counts=mode_counts, clusters=clusters, failures=failures,
        tier=run_tier, detection_threshold=float(detection_threshold),
        n_with_ground_truth=n_with_gt,
    )
    out.notes = _notes(out)
    return out


def _notes(rep: TriageReport) -> list[str]:
    """Surface the things that would otherwise be read off a table wrongly."""
    notes: list[str] = []
    if rep.n_input == 0:
        notes.append("No traces supplied — nothing to triage.")
        return notes
    if rep.n_analyzed == 0:
        notes.append(f"Every one of the {rep.n_input} traces raised during analysis; "
                     "no triage is possible. See `failures`.")
        return notes
    if rep.n_failed:
        notes.append(f"{rep.n_failed}/{rep.n_input} traces raised during analysis and are "
                     "excluded from every rate below (see `failures`).")
    if rep.n_diagnosed == 0:
        notes.append("No trace was diagnosed: every analyzed trace was healthy or "
                     "abstained. The mode distribution is empty, NOT all-zero.")
    elif rep.abstention_rate >= 0.5:
        notes.append(f"TokenTrace abstained on {rep.abstention_rate:.0%} of analyzed traces "
                     f"({rep.healthy_rate:.0%} looked healthy, {rep.declined_rate:.0%} had "
                     "insufficient evidence). Clusters describe the "
                     f"{rep.n_diagnosed} diagnosed traces only.")
    if rep.n_with_ground_truth == 0 and rep.n_analyzed:
        notes.append("No trace carried a ground-truth reference: diagnoses are RISK "
                     "estimates, and NO recommended fix could be verified — every "
                     "cluster's impact is UNKNOWN (not zero).")
    elif rep.clusters and all(c.measured_impact is None for c in rep.clusters):
        notes.append("No recommended fix in this run was scored by a simulated "
                     "intervention; all cluster impact is UNKNOWN (not zero).")
    return notes


# --------------------------------------------------------------------------- #
# Text rendering
# --------------------------------------------------------------------------- #
def _impact_cell(c: TriageCluster) -> str:
    if c.measured_impact is None:
        return f"unknown ({c.fix_unknown} unscored)"
    return f"{c.fix_validated} fixed / {c.fix_refuted} not ({c.fix_unknown} unscored)"


def render_text(rep: TriageReport, top: int = 10) -> str:
    """Compact terminal table, in the style of the CLI's ``eval``/``ablate`` printers."""
    L: list[str] = []
    L.append(f"triage: {rep.n_input} traces  analyzed={rep.n_analyzed}  failed={rep.n_failed}"
             + (f"  tier={rep.tier.label}" if rep.tier is not None else ""))
    if rep.n_analyzed == 0:
        L.extend(f"  ! {n}" for n in rep.notes)
        return "\n".join(L)

    # ``n_abstained`` counts only the DECLINED traces, while ``abstention_rate``
    # is healthy + declined. Printing the bucket under the label "abstained" put
    # "abstained 0" next to "abstention_rate=0.136" on the same line, which reads
    # as a contradiction. The bucket is labelled by what it actually holds.
    L.append(f"  diagnosed {rep.n_diagnosed}  healthy {rep.n_healthy}  "
             f"declined {rep.n_abstained}   "
             f"abstention_rate={rep.abstention_rate:.3f} "
             f"(healthy {rep.healthy_rate:.3f} + declined {rep.declined_rate:.3f})")

    dist = rep.mode_distribution()
    L.append(f"\nfailure modes (primary root; {rep.n_diagnosed} diagnosed traces):")
    if not dist:
        L.append("  (none diagnosed — distribution undefined, not zero)")
    else:
        for m in ALL_MODES:
            L.append(f"  {m.value:20s} {rep.mode_counts.get(m, 0):6d} {dist.get(m, 0.0):7.1%}")

    L.append(f"\ntop clusters (ranked by {rep.ranking_criterion} — a triage ORDER, "
             "not measured impact):")
    if not rep.clusters:
        L.append("  (no clusters)")
    else:
        # Column widths fit the longest real values (feature names reach 27 chars,
        # action keys 18) so the discriminating column is never truncated into
        # ambiguity — telling two clusters apart is the whole point of the table.
        L.append(f"  {'#':>2s} {'n':>5s} {'share':>6s} {'conf':>5s} {'mode':18s} "
                 f"{'headline signal':27s} {'log-odds':>8s} {'action':18s} impact")
        for c in rep.top(top):
            L.append(f"  {c.rank:2d} {c.n:5d} {c.share:6.1%} "
                     f"{c.mean_diagnostic_confidence:5.2f} {c.dominant_mode.value:18.18s} "
                     f"{c.headline_signal:27.27s} {c.mean_headline_logodds:+8.2f} "
                     f"{(c.dominant_action or '-'):18.18s} {_impact_cell(c)}")
            L.append(f"     exemplars: {', '.join(c.exemplar_ids) or '-'}"
                     f"   (ground truth on {c.n_with_ground_truth}/{c.n})")

    if rep.failures:
        L.append(f"\nfailures ({rep.n_failed} total, first {len(rep.failures)} shown):")
        for f in rep.failures[:5]:
            L.append(f"  #{f.index} {f.trace_id or '-'}: {f.error_type}: {f.error[:80]}")
    for n in rep.notes:
        L.append(f"  ! {n}")
    return "\n".join(L)
