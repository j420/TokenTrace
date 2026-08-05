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
from tokentrace.recommend.recommender import INTERVENTIONS

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


def _mean(total: float, n: int) -> float:
    """Mean from a running sum (see :class:`_Accum` for why nothing is retained)."""
    return total / n if n else 0.0


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def _num(x: float, spec: str, width: int) -> str:
    """Format a float for the text table, rendering a non-finite value as ``n/a``.

    :func:`_fin` scrubs NaN/inf to ``null`` on the JSON side, but the table used a
    bare format spec and printed ``+nan`` / ``nan%``. The two surfaces then disagreed
    about the same guarantee — one said "this number does not exist", the other
    printed something that looks like a measurement — so a reader comparing them
    could not tell which to believe. Both now say "we do not have this number".
    """
    if isinstance(x, float) and not math.isfinite(x):
        return f"{'n/a':>{width}s}"
    return format(x, spec)


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
    share: float = 0.0                   # n / share_denominator
    #: The denominator ``share`` is taken over: :attr:`TriageReport.n_diagnosed`, NOT
    #: the number of traces supplied. Carried as a field (and emitted next to
    #: ``share`` by :meth:`to_dict`) because an unlabelled ratio is read as a share of
    #: the fleet: measured on 100 traces / 60 diagnosed, a 30-trace cluster renders as
    #: 50%, and a reader who assumes "/ n_input" concludes 30% of production is
    #: affected. The number is right; the missing denominator made it a lie.
    share_denominator: int = 0
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
    #: MEASURED: distinct interventions that were applied, re-run and scored. Counts
    #: INTERVENTIONS, not traces — one trace can carry several distinct fixes — so it
    #: is not part of the ``fix_*`` partition above and can exceed ``n``.
    n_scored_interventions: int = 0
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
            # The denominator is part of the number, so it ships with it: a bare
            # "share" is read as a share of the fleet (see `share_denominator`).
            "share": _fin(round(self.share, 4)),
            "share_of": "n_diagnosed",
            "share_denominator": self.share_denominator,
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
            "n_scored_interventions": self.n_scored_interventions,
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
    #: "engine" (read off the engine), "explicit" (caller supplied), or
    #: "assumed" (the facade exposed none, so the healthy/declined split rests
    #: on a guess). Surfaced because "assumed" makes those two buckets soft.
    threshold_origin: str = "engine"
    ranking_criterion: str = "n * mean_diagnostic_confidence"
    n_with_ground_truth: int = 0
    #: The iterable of traces itself raised, so iteration stopped early and every
    #: count here describes only what was read before that. Surfaced because
    #: ``failure_rate`` is otherwise silently computed against a truncated log.
    input_stream_failed: bool = False
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def abstention_rate(self) -> float:
        """Share of analyzed traces that got no named root cause, healthy ones included.

        This is the headline honesty number: a run that produced no diagnosis for 60%
        of its input says so loudly rather than quietly reporting a five-mode
        distribution over the other 40%.

        It is *almost* ``DiagnosisReport.abstained`` one-for-one, and the docstring
        used to claim exactly that — but one shape breaks the identity: a report that
        did NOT set ``abstained`` yet carries no rankable diagnosis at all is counted
        in ``n_abstained`` too, because from the reader's side nothing was diagnosed
        either way. :class:`~tokentrace.engine.diagnosis.DiagnosisEngine` always emits
        all five modes so it cannot produce that shape; a third-party facade can. The
        numerator is therefore ``n_healthy + n_abstained``, which is the honest
        quantity, rather than a literal count of the ``abstained`` flag.
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
            "input_stream_failed": self.input_stream_failed,
            "abstention_rate": _fin(round(self.abstention_rate, 4)),
            "healthy_rate": _fin(round(self.healthy_rate, 4)),
            "declined_rate": _fin(round(self.declined_rate, 4)),
            "failure_rate": _fin(round(self.failure_rate, 4)),
            "tier": self.tier.label if self.tier is not None else None,
            "detection_threshold": _fin(self.detection_threshold),
            "threshold_origin": self.threshold_origin,
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


def _scored_recommendations(report: DiagnosisReport) -> list[Recommendation]:
    """Every DISTINCT intervention on this trace, best-first.

    An earlier version kept only the single highest-priority recommendation, on the
    stated grounds that this "sidesteps the double-counting trap ``eval.metrics`` had
    to fix". That rationale was wrong, and the cost was severe: ``eval/metrics.py``
    dedupes by intervention *function identity*, which collapses the alias
    (``ground_or_abstain`` resolves to the same callable as ``add_gold_context``)
    while KEEPING genuinely distinct fixes. Taking one per trace instead discarded
    interventions that had actually been applied, re-run and scored against ground
    truth — so real observed negatives vanished into ``fix_unknown`` and the report
    then printed a note asserting nothing had been measured. Measured on an ordinary
    sweep: 5 of 12 diagnosed traces silently dropped a scored intervention.

    Dedupe here matches metrics.py exactly: by the identity of the callable.
    """
    ordered: list[tuple[tuple[int, int, int], Recommendation]] = []
    for rank, diag in enumerate(report.diagnoses):
        for i, rec in enumerate(diag.recommendations):
            ordered.append(((rec.priority, rank, i), rec))
    ordered.sort(key=lambda kv: kv[0])

    seen: set[int] = set()
    out: list[Recommendation] = []
    for _key, rec in ordered:
        fn = INTERVENTIONS.get(rec.action)
        marker = id(fn) if fn is not None else hash(("advisory", rec.action))
        if marker in seen:
            continue
        seen.add(marker)
        out.append(rec)
    return out


def _chosen_recommendation(report: DiagnosisReport) -> Optional[Recommendation]:
    """The single fix a developer should apply first (drives ``dominant_action``)."""
    scored = _scored_recommendations(report)
    return scored[0] if scored else None


# --------------------------------------------------------------------------- #
# Streaming accumulator (O(clusters) memory, not O(traces))
# --------------------------------------------------------------------------- #
@dataclass
class _Accum:
    """Per-cluster running state. Bounded: O(1) per cluster, O(0) per trace.

    The three means are kept as running SUMS, not as lists of per-trace values. That
    is not a micro-optimization — it is what makes the header above true. This module
    promises a fleet log is "consumed lazily, so a 10k-row JSONL generator never has
    to be materialized", and retaining three floats per trace quietly broke that
    promise for the only shape that matters (measured: 200k traces landing in one
    cluster held three 200k-element lists). The only structure that grows with traces
    is ``exemplars``, capped at ``max_exemplars`` on the way in.
    """

    mode: FailureMode
    signal: str
    source: str
    family: Optional[str]
    n: int = 0
    sum_confidence: float = 0.0
    sum_primary: float = 0.0
    sum_logodds: float = 0.0
    actions: Counter = field(default_factory=Counter)
    fix_validated: int = 0
    fix_refuted: int = 0
    fix_unknown: int = 0
    #: Interventions actually applied, re-run and scored (validated is not None).
    #: Distinct from the per-trace split above, which counts TRACES.
    n_scored_interventions: int = 0
    n_with_gt: int = 0
    n_non_rag: int = 0
    exemplars: list[str] = field(default_factory=list)

    def add(self, trace: Inference, report: DiagnosisReport, logodds: float,
            trace_key: str, max_exemplars: int) -> None:
        """Fold one trace into the cluster — all-or-nothing.

        Every value is read and coerced BEFORE any field is mutated. A malformed
        report raising halfway through (``diagnostic_confidence=None`` out of a
        hand-rolled ingest adapter is the realistic case) would otherwise leave the
        cluster having counted a trace it never finished adding, so its ``n`` and its
        ``fix_*`` split would disagree — an invariant the module docstring asks
        reviewers to check. :func:`triage` catches the exception; this keeps the
        state it is catching on top of consistent.
        """
        confidence = float(report.diagnostic_confidence)
        primary = report.primary
        primary_probability = float(primary.probability) if primary is not None else 0.0
        contribution = float(logodds)
        has_gt = int(bool(trace.has_ground_truth))
        non_rag = int(not trace.is_rag)
        key = str(trace_key)

        # `dominant_action` answers "what should I do about this cluster", so it
        # counts the first fix only. The verification split counts EVERY distinct
        # intervention that was scored: discarding a measured negative because a
        # higher-priority fix went unscored is how observed refutations disappeared.
        recs = _scored_recommendations(report)
        outcomes = [r.validated for r in recs]
        action = recs[0].action if recs else None
        scored = sum(1 for o in outcomes if o is not None)

        # ---- commit: nothing below this line can raise ---- #
        self.n += 1
        self.sum_confidence += confidence
        self.sum_primary += primary_probability
        self.sum_logodds += contribution
        self.n_with_gt += has_gt
        self.n_non_rag += non_rag
        if len(self.exemplars) < max_exemplars:
            self.exemplars.append(key)
        if action is None:
            # No applicable fix at all -> nothing to verify, so it is unknown, not a
            # failed fix. (Happens when no diagnosis cleared the recommender's own
            # probability floor.)
            self.fix_unknown += 1
            return
        self.actions[action] += 1
        if True in outcomes:
            self.fix_validated += 1
        elif False in outcomes:
            self.fix_refuted += 1
        else:
            self.fix_unknown += 1
        self.n_scored_interventions += scored

    def finish(self, n_diagnosed: int) -> TriageCluster:
        mean_conf = _mean(self.sum_confidence, self.n)
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
            n=self.n,
            share=_rate(self.n, n_diagnosed),
            share_denominator=n_diagnosed,
            mean_diagnostic_confidence=mean_conf,
            mean_primary_probability=_mean(self.sum_primary, self.n),
            mean_headline_logodds=_mean(self.sum_logodds, self.n),
            dominant_action=dominant_action,
            dominant_action_n=dominant_n,
            action_counts=dict(sorted(self.actions.items())),
            fix_validated=self.fix_validated,
            fix_refuted=self.fix_refuted,
            fix_unknown=self.fix_unknown,
            n_scored_interventions=self.n_scored_interventions,
            n_with_ground_truth=self.n_with_gt,
            n_non_rag=self.n_non_rag,
            # Capped once, on the way in (see the class docstring); re-slicing here
            # would be a second cap that hides the removal of the first.
            exemplar_ids=list(self.exemplars),
            # RANKING CRITERION -- see triage()'s docstring for the justification.
            priority_score=self.n * mean_conf,
        )


def _rank_key(c: TriageCluster) -> tuple[int, float, int, float, str, str]:
    """Total order over clusters that survives a non-finite score.

    NaN loses every comparison it takes part in, so one NaN
    ``mean_diagnostic_confidence`` (an infinite log-odds contribution from a
    degenerate model is enough) made the sort a no-op for that cluster and handed the
    ranking to the INPUT ORDER — while the call site's comment promised a fully
    deterministic one, and ``to_dict`` then printed ``"rank": 1`` next to
    ``"priority_score": null`` above a cluster scoring 0.9.

    Non-finite scores sort LAST: a cluster whose score could not be computed has not
    earned the top of a "fix this first" list. Within each group the same explicit
    tie-breakers apply (size, confidence, then the cluster key) so the order is a
    function of the clusters alone.
    """
    score, conf = c.priority_score, c.mean_diagnostic_confidence
    return (
        0 if math.isfinite(score) else 1,
        -score if math.isfinite(score) else 0.0,
        -c.n,
        -conf if math.isfinite(conf) else 0.0,
        c.dominant_mode.value,
        c.headline_signal,
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

    "Traces that raise" means anywhere in the per-trace path, not just inside
    ``tt.analyze``: reading the report, picking the headline evidence line and folding
    the trace into its cluster are equally able to raise on a malformed report (a
    ``None`` diagnostic confidence, an evidence line with no family), and an earlier
    version left all three outside the guard — so trace #2 of 3 took traces #1 and #3
    down with it and recorded nothing in ``failures``, which is precisely what
    :class:`TriageFailure` exists to prevent. **Producing** the traces counts too: an
    ingest adapter that dies on line 4173 of a log is the same event as a trace that
    fails to analyze, and everything read before it is kept.
    """
    threshold_origin = "explicit"
    if detection_threshold is None:
        engine = getattr(tt, "engine", None)
        engine_threshold = getattr(engine, "abstain_primary", None)
        if engine_threshold is None:
            # The facade did not expose its threshold, so we are about to invent the
            # "second, disagreeing threshold" this function's contract promises never
            # to invent. That guess splits healthy from declined, and guessing HIGH
            # reports genuinely-declined traces as a clean fleet — the wrong direction
            # for an observability tool. It is no longer silent.
            detection_threshold = DEFAULT_DETECTION_THRESHOLD
            threshold_origin = "assumed"
        else:
            detection_threshold = float(engine_threshold)
            threshold_origin = "engine"

    accums: dict[tuple[str, str], _Accum] = {}
    mode_counts: dict[FailureMode, int] = {m: 0 for m in ALL_MODES}
    failures: list[TriageFailure] = []
    n_input = n_analyzed = n_failed = n_diagnosed = n_healthy = n_abstained = 0
    n_with_gt = 0
    run_tier: Optional[Tier] = None      # discovered from the reports, never assumed

    def _record(index: int, trace_id: Optional[str], exc: BaseException) -> None:
        """Note one failed trace. The COUNT is never capped, only the detail list."""
        if len(failures) < max_failures_recorded:
            failures.append(TriageFailure(
                index=index,
                trace_id=trace_id,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            ))

    # Iterated by hand rather than with `for ... in enumerate(traces)` so that a
    # raising ITERATOR is a recordable failure like any other. The traces are usually
    # a generator reading a log file, and a decode error on one row used to escape
    # triage() and discard every trace already analyzed.
    stream = iter(traces)
    index = -1
    stream_broke = False
    while True:
        index += 1
        try:
            trace = next(stream)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - a broken log must not lose the rest
            # The input itself failed. Count it as a failed trace (it is one: a row we
            # were meant to triage and could not) and stop — a generator that raised is
            # closed, so pulling on it again would only raise StopIteration, and no
            # iterator protocol promises recovery. Everything read so far is kept.
            n_input += 1
            n_failed += 1
            _record(index, None, exc)
            stream_broke = True
            break

        n_input += 1
        try:
            # Only pass `tier` when the caller actually overrode it, so the minimal
            # contract a facade must satisfy stays `analyze(inference)`.
            report = tt.analyze(trace, tier=tier) if tier is not None else tt.analyze(trace)

            # Everything below reads a report we did not build and may raise on a
            # malformed one, so it lives INSIDE the guard. It also touches no shared
            # counter: the bucket decision is computed here and committed after the
            # guard, so a trace that dies mid-way cannot be half-counted (the one
            # mutation here, `acc.add`, is itself atomic).
            trace_tier = report.tier
            trace_has_gt = int(bool(trace.has_ground_truth))
            primary = None
            acc: Optional[_Accum] = None
            key: Optional[tuple[str, str]] = None
            if report.abstained:
                # Two very different reasons hide behind one flag; keep them apart.
                # Below the detection threshold the engine's own note reads "appears
                # healthy"; above it, the engine saw something and refused to rank it.
                bucket = "healthy" if _top1(report) < detection_threshold else "declined"
            elif report.primary is None:   # defensive: a report with no diagnoses at all
                bucket = "declined"
            else:
                bucket = "diagnosed"
                primary = report.primary
                signal, source, family, logodds = _headline(report)
                key = (primary.mode.value, signal)
                acc = accums.get(key)
                if acc is None:
                    acc = _Accum(primary.mode, signal, source, family)
                acc.add(trace, report, logodds,
                        trace_key=trace.id if trace.id else f"#{index}",
                        max_exemplars=max_exemplars)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - one bad trace must not kill the run
            n_failed += 1
            _record(index, getattr(trace, "id", None), exc)
            continue

        # ---- commit: nothing below this line can raise ---- #
        n_analyzed += 1
        n_with_gt += trace_has_gt
        # The tier that RAN, not the tier requested. ModelHandle.with_tier only ever
        # down-caps (min(tier, profile.max_tier)) -- the documented auto-degrade
        # ladder -- so seeding this from the parameter let a fleet report claim
        # white-box causal evidence for a run that never left grey-box.
        if trace_tier is not None and (run_tier is None or trace_tier < run_tier):
            run_tier = trace_tier
        if bucket == "healthy":
            n_healthy += 1
        elif bucket == "declined":
            n_abstained += 1
        else:
            n_diagnosed += 1
            mode_counts[primary.mode] = mode_counts.get(primary.mode, 0) + 1
            accums[key] = acc

    clusters = [a.finish(n_diagnosed) for a in accums.values()]
    clusters.sort(key=_rank_key)          # deterministic even with a NaN score
    for i, c in enumerate(clusters, start=1):
        c.rank = i

    out = TriageReport(
        n_input=n_input, n_analyzed=n_analyzed, n_failed=n_failed,
        n_diagnosed=n_diagnosed, n_healthy=n_healthy, n_abstained=n_abstained,
        mode_counts=mode_counts, clusters=clusters, failures=failures,
        tier=run_tier, detection_threshold=float(detection_threshold),
        threshold_origin=threshold_origin,
        n_with_ground_truth=n_with_gt,
        input_stream_failed=stream_broke,
    )
    out.notes = _notes(out)
    return out


def _notes(rep: TriageReport) -> list[str]:
    """Surface the things that would otherwise be read off a table wrongly."""
    notes: list[str] = []
    if rep.threshold_origin == "assumed" and (rep.n_healthy or rep.n_abstained):
        notes.append(
            f"The facade exposed no detection threshold, so healthy-vs-declined was "
            f"split at an assumed {rep.detection_threshold:.2f}. Those two buckets "
            f"({rep.n_healthy} healthy / {rep.n_abstained} declined) may not match the "
            f"engine's own classification.")
    if rep.input_stream_failed:
        # n_input is then a floor, not the size of the log, and every rate below is
        # computed against it. Say so before any of them are read.
        notes.append("The trace stream itself raised (an ingest adapter failing mid-log) "
                     f"and iteration stopped there: the {rep.n_input} traces counted below "
                     "are only those read before the break, not the whole input.")
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
    if rep.clusters and rep.n_diagnosed < rep.n_input:
        # Fires on ANY gap, not only past some abstention threshold: a cluster share
        # is misread as a share of the fleet at every ratio, and the reader has no way
        # to notice the denominator is missing.
        notes.append(f"Cluster `share` (%diag) is a fraction of the {rep.n_diagnosed} "
                     f"DIAGNOSED traces, not of the {rep.n_input} supplied "
                     f"({rep.n_healthy} healthy, {rep.n_abstained} declined, "
                     f"{rep.n_failed} failed).")
    non_finite = sum(1 for c in rep.clusters if not math.isfinite(c.priority_score))
    if non_finite:
        notes.append(f"{non_finite} cluster(s) had a non-finite priority score (a NaN or "
                     "infinite mean, e.g. from a degenerate log-odds contribution). They "
                     "are ranked LAST and serialize as null rather than being ordered by "
                     "an unusable number.")
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
        # The column is `%diag`, never a bare `share`: the header has to carry its own
        # denominator or the number is read as a share of the fleet, which overstates
        # every cluster by the abstention rate.
        L.append(f"  %diag = share of the {rep.n_diagnosed} DIAGNOSED traces "
                 f"(of {rep.n_input} supplied); non-finite means print as n/a")
        # Column widths fit the longest real values (feature names reach 27 chars,
        # action keys 18) so the discriminating column is never truncated into
        # ambiguity — telling two clusters apart is the whole point of the table.
        L.append(f"  {'#':>2s} {'n':>5s} {'%diag':>6s} {'conf':>5s} {'mode':18s} "
                 f"{'headline signal':27s} {'log-odds':>8s} {'action':18s} impact")
        for c in rep.top(top):
            L.append(f"  {c.rank:2d} {c.n:5d} {_num(c.share, '6.1%', 6)} "
                     f"{_num(c.mean_diagnostic_confidence, '5.2f', 5)} "
                     f"{c.dominant_mode.value:18.18s} "
                     f"{c.headline_signal:27.27s} "
                     f"{_num(c.mean_headline_logodds, '+8.2f', 8)} "
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
