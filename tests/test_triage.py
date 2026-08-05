"""Tests for fleet triage (:mod:`tokentrace.triage`).

Everything here runs on the deterministic ``mock`` backend — no downloads, CPU only.

Most tests drive the REAL pipeline so that the clustering is exercised against
evidence the engine actually produced. The few that use a stub facade do so where
the point is a property the real engine cannot be forced to exhibit on demand (two
clusters of identical size but different confidence; a fleet that abstains on
everything), and those stubs return genuine :class:`DiagnosisReport` objects.
"""

from __future__ import annotations

import json

import pytest

from tokentrace.core.types import (
    Chunk,
    Diagnosis,
    DiagnosisReport,
    DiagnosisRole,
    EvidenceItem,
    FailureMode as M,
    Inference,
    Recommendation,
    SignalFamily,
    Tier,
)
from tokentrace.triage import TriageCluster, render_text, triage

from _helpers import dilution, grounded, retrieval_failure


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tt():
    """Cold-start facade (interpretable rules only).

    Cold start is used for most tests because its probabilities are a pure function
    of the rule set, so a cluster assertion that fails points at the triage code
    rather than at LightGBM seed drift. One test below runs the fully trained engine
    to prove triage is not accidentally coupled to the cold-start path.
    """
    from tokentrace import TokenTrace

    return TokenTrace.default(train=False)


def _observe(tt, inf: Inference) -> Inference:
    inf.generated_answer = tt.model.bound_to(inf).generate(inf.prompt).text
    return inf


def gold_absent_trace(i: int) -> Inference:
    """Retrieval failure, cause #1: the answer is nowhere in the retrieved context.

    Diagnosed via the GT discriminator ``ret.gold_absent``. Engineering fix: the
    index/recall side of the retriever.
    """
    inf = retrieval_failure()
    inf.id = f"absent-{i}"
    return inf


def irrelevant_retrieval_trace(i: int) -> Inference:
    """Retrieval failure, cause #2: the answer is partly recoverable, but the
    retriever ranked nothing relevant to the query.

    Tuned so that ``gold_recall_in_context`` lands mid-ramp (~0.44) — enough to weaken
    ``ret.gold_absent`` below the reference-free ``ret.irrelevant`` rule, which then
    becomes the headline. Same failure mode as :func:`gold_absent_trace`, different
    driver, different engineering fix (query rewrite / ranker, not the index).
    """
    q = "Who designed the Zorbian aqueduct?"
    ctx = "Provincial records list the chief engineer for the northern province."
    return Inference(
        f"{q}\n{ctx}", "", [Chunk(ctx, 0.2, "p1")],
        ["Marek Voss chief engineer northern province appointed eighteen ninety"],
        question=q,
        meta={"_sim": {"answer": "Marek Voss", "gold_fact": "Marek Voss",
                       "distractor": "Tomas Lind"}},
        id=f"irrelevant-{i}",
    )


def production_trace(i: int) -> Inference:
    """A production trace: retrieval is unhelpful and there is NO ground truth.

    Nothing here can be verified — which is exactly the point of the impact tests.
    """
    q = "When was the Eiffel Tower completed?"
    return Inference(
        f"{q}\nParis is in France.", "", [Chunk("Paris is in France.", 0.05, "d2")],
        None, question=q,
        meta={"_sim": {"answer": "1889", "gold_fact": "1889", "distractor": "1920"}},
        id=f"prod-{i}",
    )


def nonrag_ambiguity_trace(i: int) -> Inference:
    """Non-RAG (``retrieved_context is None``) and still diagnosable."""
    return Inference(
        "When was it completed?", "", None, ["1889"], question="When was it completed?",
        meta={"_sim": {"answer": "1889", "ambiguous": True,
                       "readings": ["1889", "1887", "2000"]}},
        id=f"nonrag-{i}",
    )


# --- stub facade: full control over abstention / confidence / validation --- #
def stub_report(mode: M, signal: str, prob: float = 0.9, conf: float = 0.8,
                abstained: bool = False, action: str | None = "add_gold_context",
                validated: bool | None = None, logodds: float = 2.0,
                family: SignalFamily | None = SignalFamily.RETRIEVAL,
                prior_first: bool = False, tier: Tier = Tier.WHITE) -> DiagnosisReport:
    """A minimal but genuine :class:`DiagnosisReport`.

    The knobs exist to reach states the real engine cannot be made to produce on
    demand, and each one is used by a test below to make a triage number falsifiable:
    ``logodds`` (is the headline mean *signed*?), ``family=None`` and ``conf=None``
    (does a malformed report take the whole run down?), ``prior_first`` (does
    clustering key on the headline line or on the bookkeeping row above it?).
    """
    ledger = [EvidenceItem(signal=signal, family=family, value=0.0,
                           contribution_logodds=logodds, source="rule")]
    if prior_first:
        # The real ledger is magnitude-ordered and carries bookkeeping rows; the
        # base-rate prior routinely outweighs the line a human would call "the reason".
        ledger.insert(0, EvidenceItem(signal="base_rate", family=SignalFamily.PROMPT,
                                      value=0.0, contribution_logodds=-9.9,
                                      source="prior"))
    recs = ([Recommendation(action=action, description="", targets_mode=mode,
                            priority=0, validated=validated)] if action else [])
    diag = Diagnosis(mode=mode, probability=prob, rank=0, role=DiagnosisRole.PRIMARY_ROOT,
                     evidence=ledger, recommendations=recs)
    return DiagnosisReport(diagnoses=[diag], tier=tier, diagnostic_confidence=conf,
                           abstained=abstained)


class StubTT:
    """Facade double: replays a prepared report (or raises) per trace, in order."""

    def __init__(self, outcomes, abstain_primary=None):
        self._outcomes = list(outcomes)
        self._i = 0
        if abstain_primary is not None:
            # Mimic a real facade exposing its own detection threshold.
            self.engine = type("_Engine", (), {"abstain_primary": abstain_primary})()

    def analyze(self, inference, tier=None):
        out = self._outcomes[self._i]
        self._i += 1
        if isinstance(out, BaseException):
            raise out
        return out


def stub_traces(n: int) -> list[Inference]:
    return [Inference(f"q{i}", f"a{i}", [Chunk("c")], ["a"], id=f"s{i}") for i in range(n)]


# --------------------------------------------------------------------------- #
# 1. Clustering does real work: one mode, two causes, two clusters
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_cause_report(tt):
    traces = ([gold_absent_trace(i) for i in range(4)]
              + [irrelevant_retrieval_trace(i) for i in range(3)])
    return triage([_observe(tt, t) for t in traces], tt)


def test_one_failure_mode_splits_into_two_clusters_by_headline_signal(two_cause_report):
    rep = two_cause_report
    assert rep.n_diagnosed == 7, rep.notes
    # Both groups really are the SAME failure mode -- otherwise this test proves
    # nothing about clustering and everything about mode classification.
    assert rep.mode_counts[M.RETRIEVAL_FAILURE] == 7
    ret_clusters = [c for c in rep.clusters if c.dominant_mode is M.RETRIEVAL_FAILURE]
    assert len(ret_clusters) == 2, [c.key for c in rep.clusters]

    signals = {c.headline_signal for c in ret_clusters}
    assert signals == {"ret.gold_absent", "ret.irrelevant"}, signals

    by_signal = {c.headline_signal: c for c in ret_clusters}
    assert by_signal["ret.gold_absent"].n == 4
    assert by_signal["ret.irrelevant"].n == 3
    # And the traces genuinely landed apart, not just the counts.
    assert all(e.startswith("absent-") for e in by_signal["ret.gold_absent"].exemplar_ids)
    assert all(e.startswith("irrelevant-") for e in by_signal["ret.irrelevant"].exemplar_ids)


def test_cluster_key_is_finer_than_mode_or_recommended_action(two_cause_report):
    """The headline signal is load-bearing: mode AND action are identical here."""
    ret = [c for c in two_cause_report.clusters if c.dominant_mode is M.RETRIEVAL_FAILURE]
    assert len({c.dominant_mode for c in ret}) == 1
    assert len({c.dominant_action for c in ret}) == 1 != len(ret)


def test_headline_logodds_is_reported_per_cluster(two_cause_report):
    by_signal = {c.headline_signal: c for c in two_cause_report.clusters}
    # ret.gold_absent fires at its full 2.4 weight; ret.irrelevant at 1.6. If the mean
    # were not actually being computed from the evidence these would be equal (or 0).
    assert by_signal["ret.gold_absent"].mean_headline_logodds == pytest.approx(2.4, abs=1e-3)
    assert by_signal["ret.irrelevant"].mean_headline_logodds == pytest.approx(1.6, abs=1e-3)


def test_trained_engine_also_clusters(two_cause_report):
    """Triage must not be coupled to the cold-start path.

    With the learned head the headline line is usually a TreeSHAP feature rather than
    a rule id, so this also pins down that both evidence sources are accepted as
    cluster keys.
    """
    from tokentrace import TokenTrace

    trained = TokenTrace.default(train=True, seeds=(0, 1))
    traces = ([gold_absent_trace(i) for i in range(3)]
              + [_observe(trained, dilution()) for _ in range(2)])
    rep = triage([_observe(trained, t) for t in traces], trained)
    assert rep.n_analyzed == 5 and rep.n_failed == 0
    assert rep.n_diagnosed >= 1
    assert all(c.headline_signal for c in rep.clusters)
    assert sum(c.n for c in rep.clusters) == rep.n_diagnosed


# --------------------------------------------------------------------------- #
# 2. Ranking
# --------------------------------------------------------------------------- #
def test_ranking_puts_the_larger_cluster_first(tt):
    traces = ([gold_absent_trace(i) for i in range(6)]
              + [irrelevant_retrieval_trace(i) for i in range(2)])
    rep = triage([_observe(tt, t) for t in traces], tt)
    assert rep.clusters[0].headline_signal == "ret.gold_absent"
    assert rep.clusters[0].rank == 1 and rep.clusters[0].n == 6
    assert rep.clusters[1].n == 2


def test_ranking_prefers_the_more_confident_cluster_at_equal_size():
    """Size alone must not decide — the criterion is n * mean_diagnostic_confidence."""
    outcomes = ([stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", conf=0.3)] * 3
                + [stub_report(M.CONTEXT_DILUTION, "dil.buried", conf=0.9)] * 3)
    rep = triage(stub_traces(6), StubTT(outcomes))
    assert [c.n for c in rep.clusters] == [3, 3]
    assert rep.clusters[0].headline_signal == "dil.buried"
    assert rep.clusters[0].priority_score == pytest.approx(2.7)
    assert rep.clusters[1].priority_score == pytest.approx(0.9)


def test_clusters_are_sorted_by_priority_score_and_ranked_from_one(tt):
    traces = ([gold_absent_trace(i) for i in range(5)]
              + [irrelevant_retrieval_trace(i) for i in range(2)]
              + [_observe(tt, dilution()) for _ in range(3)])
    rep = triage([_observe(tt, t) if t.id else t for t in traces], tt)
    scores = [c.priority_score for c in rep.clusters]
    assert scores == sorted(scores, reverse=True)
    assert [c.rank for c in rep.clusters] == list(range(1, len(rep.clusters) + 1))
    assert len(rep.clusters) >= 3


def test_priority_score_is_size_times_mean_confidence(tt):
    rep = triage([_observe(tt, gold_absent_trace(i)) for i in range(3)], tt)
    c = rep.clusters[0]
    assert c.priority_score == pytest.approx(c.n * c.mean_diagnostic_confidence)


# --------------------------------------------------------------------------- #
# 3. Abstention is not a diagnosis
# --------------------------------------------------------------------------- #
def test_healthy_and_abstained_are_excluded_from_the_mode_distribution():
    """All three reports name a primary mode; only the non-abstained one may count."""
    outcomes = [
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", prob=0.9, abstained=False),
        # abstained but well above the detection threshold -> "declined", not healthy
        stub_report(M.CONTEXT_DILUTION, "dil.buried", prob=0.85, abstained=True),
        # abstained and below the detection threshold -> "looks healthy"
        stub_report(M.HALLUCINATION, "hal.redeep", prob=0.10, abstained=True),
    ]
    rep = triage(stub_traces(3), StubTT(outcomes))

    assert (rep.n_diagnosed, rep.n_abstained, rep.n_healthy) == (1, 1, 1)
    assert rep.mode_counts[M.CONTEXT_DILUTION] == 0, "an abstention was counted as a diagnosis"
    assert rep.mode_counts[M.HALLUCINATION] == 0, "a healthy trace was counted as a diagnosis"
    assert rep.mode_counts[M.RETRIEVAL_FAILURE] == 1
    assert sum(rep.mode_counts.values()) == rep.n_diagnosed
    # ...and neither leaks into the clusters either.
    assert [c.key for c in rep.clusters] == [("retrieval_failure", "ret.gold_absent")]
    assert rep.abstention_rate == pytest.approx(2 / 3)
    assert rep.healthy_rate == pytest.approx(1 / 3)
    assert rep.declined_rate == pytest.approx(1 / 3)


def test_abstention_rate_is_surfaced_in_the_notes_and_the_text_render():
    outcomes = [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", abstained=False)] + [
        stub_report(M.HALLUCINATION, "hal.redeep", prob=0.1, abstained=True)] * 3
    rep = triage(stub_traces(4), StubTT(outcomes))
    assert rep.abstention_rate == pytest.approx(0.75)
    assert any("abstained on 75%" in n for n in rep.notes), rep.notes
    assert "abstention_rate=0.750" in render_text(rep)


def test_buckets_partition_the_analyzed_traces(tt):
    traces = [_observe(tt, t) for t in (
        grounded(), retrieval_failure(), dilution(), grounded())]
    rep = triage(traces, tt)
    assert rep.n_diagnosed + rep.n_healthy + rep.n_abstained == rep.n_analyzed == 4
    assert rep.n_healthy >= 1, "the grounded traces should not be diagnosed as failures"


# --------------------------------------------------------------------------- #
# 4. Impact: measured, refuted, or honestly unknown
# --------------------------------------------------------------------------- #
def test_no_ground_truth_cluster_reports_impact_as_unknown_not_zero(tt):
    traces = [_observe(tt, production_trace(i)) for i in range(4)]
    rep = triage(traces, tt)
    assert rep.n_with_ground_truth == 0
    assert rep.n_diagnosed == 4, rep.notes
    c = rep.clusters[0]
    assert c.n_with_ground_truth == 0

    # The three counts, kept distinct.
    assert c.fix_validated == 0 and c.fix_refuted == 0
    assert c.fix_unknown == c.n == 4

    # The headline impact number must be UNKNOWN. Explicitly not 0: "we never
    # measured it" and "we measured it and it was zero" are different claims.
    assert c.measured_impact is None
    assert c.measured_impact != 0
    assert c.impact_status == "unknown"
    assert c.to_dict()["measured_impact"] is None
    assert "unknown" in render_text(rep)
    assert any("UNKNOWN (not zero)" in n for n in rep.notes), rep.notes


def test_measured_impact_is_reported_when_ground_truth_makes_it_measurable(tt):
    """Guards the previous test: `measured_impact` must be capable of being non-None,
    otherwise "it is None without ground truth" is an assertion that cannot fail."""
    rep = triage([_observe(tt, gold_absent_trace(i)) for i in range(3)], tt)
    c = rep.clusters[0]
    assert c.n_with_ground_truth == 3
    assert c.measured_impact == 3
    assert c.impact_status == "measured"
    assert c.fix_validated == 3 and c.fix_refuted == 0 and c.fix_unknown == 0
    assert c.dominant_action == "add_gold_context" and c.dominant_action_n == 3


def test_a_refuted_fix_is_counted_as_refuted_not_as_unknown():
    outcomes = [stub_report(M.CONTEXT_DILUTION, "dil.buried", validated=False)] * 2 + [
        stub_report(M.CONTEXT_DILUTION, "dil.buried", validated=True)]
    rep = triage(stub_traces(3), StubTT(outcomes))
    c = rep.clusters[0]
    assert (c.fix_validated, c.fix_refuted, c.fix_unknown) == (1, 2, 0)
    assert c.measured_impact == 1        # 1 of 3, not 3


def test_fix_counts_partition_every_cluster(tt):
    traces = ([_observe(tt, gold_absent_trace(i)) for i in range(3)]
              + [_observe(tt, production_trace(i)) for i in range(3)]
              + [_observe(tt, dilution()) for _ in range(2)])
    rep = triage(traces, tt)
    assert rep.clusters
    for c in rep.clusters:
        assert c.fix_validated + c.fix_refuted + c.fix_unknown == c.n, c.key


def test_a_trace_with_no_recommendation_counts_as_unknown_not_refuted():
    rep = triage(stub_traces(2),
                 StubTT([stub_report(M.HALLUCINATION, "hal.redeep", action=None)] * 2))
    c = rep.clusters[0]
    assert (c.fix_validated, c.fix_refuted, c.fix_unknown) == (0, 0, 2)
    assert c.dominant_action is None and c.measured_impact is None


# --------------------------------------------------------------------------- #
# 5. Edge cases
# --------------------------------------------------------------------------- #
def test_empty_input(tt):
    rep = triage([], tt)
    assert (rep.n_input, rep.n_analyzed, rep.n_diagnosed) == (0, 0, 0)
    assert rep.clusters == [] and rep.mode_distribution() == {}
    assert rep.abstention_rate == 0.0 and rep.failure_rate == 0.0
    assert any("No traces supplied" in n for n in rep.notes)
    assert render_text(rep)                     # must not raise
    json.dumps(rep.to_dict(), allow_nan=False)


def test_single_trace(tt):
    rep = triage([_observe(tt, gold_absent_trace(0))], tt)
    assert rep.n_input == rep.n_analyzed == rep.n_diagnosed == 1
    assert len(rep.clusters) == 1
    c = rep.clusters[0]
    assert c.n == 1 and c.share == pytest.approx(1.0) and c.exemplar_ids == ["absent-0"]


def test_all_traces_abstained():
    rep = triage(stub_traces(5), StubTT(
        [stub_report(M.HALLUCINATION, "hal.redeep", prob=0.9, abstained=True)] * 5))
    assert rep.n_analyzed == 5 and rep.n_diagnosed == 0
    assert rep.n_abstained == 5 and rep.n_healthy == 0
    assert rep.clusters == []
    # An empty distribution, NOT five zero shares (which reads as "no failures").
    assert rep.mode_distribution() == {}
    assert rep.mode_counts[M.HALLUCINATION] == 0
    assert rep.abstention_rate == 1.0
    assert any("mode distribution is empty, NOT all-zero" in n for n in rep.notes)
    json.dumps(rep.to_dict(), allow_nan=False)


def test_one_bad_trace_does_not_kill_the_run(tt):
    """A malformed trace really does raise inside the pipeline (TypeError from the
    scorers); the run must absorb it and keep going."""
    bad = Inference(prompt=None, generated_answer="something", retrieved_context=None,
                    ground_truth=None, id="bad-1")
    traces = [_observe(tt, gold_absent_trace(0)), bad, _observe(tt, gold_absent_trace(1))]
    rep = triage(traces, tt)
    assert rep.n_input == 3 and rep.n_analyzed == 2 and rep.n_failed == 1
    assert rep.n_diagnosed == 2
    (f,) = rep.failures
    assert f.index == 1 and f.trace_id == "bad-1" and f.error_type == "TypeError"
    assert rep.failure_rate == pytest.approx(1 / 3)
    assert any("raised during analysis" in n for n in rep.notes)
    assert "bad-1" in render_text(rep)


def test_every_trace_failing_is_reported_rather_than_crashing(tt):
    rep = triage(stub_traces(3), StubTT([ValueError("boom")] * 3))
    assert rep.n_analyzed == 0 and rep.n_failed == 3
    assert rep.clusters == [] and rep.mode_distribution() == {}
    assert any("no triage is possible" in n for n in rep.notes)
    assert render_text(rep)
    json.dumps(rep.to_dict(), allow_nan=False)


def test_failure_list_is_capped_but_the_count_is_not():
    rep = triage(stub_traces(12), StubTT([RuntimeError("x")] * 12),
                 max_failures_recorded=4)
    assert rep.n_failed == 12 and len(rep.failures) == 4


def test_keyboard_interrupt_is_not_swallowed():
    """Collecting per-trace errors must not turn a Ctrl-C into a silent data point."""
    with pytest.raises(KeyboardInterrupt):
        triage(stub_traces(2), StubTT([KeyboardInterrupt()] * 2))


def test_non_rag_traces_never_produce_retrieval_side_clusters(tt):
    traces = [_observe(tt, nonrag_ambiguity_trace(i)) for i in range(3)]
    rep = triage(traces, tt)
    assert rep.n_analyzed == 3 and rep.n_failed == 0
    assert rep.n_diagnosed == 3, rep.notes          # keeps the assertions below honest
    assert rep.mode_counts[M.RETRIEVAL_FAILURE] == 0
    assert rep.mode_counts[M.CONTEXT_DILUTION] == 0
    assert rep.mode_counts[M.PROMPT_AMBIGUITY] == 3
    assert rep.clusters[0].n_non_rag == 3


def test_mixed_fleet_of_every_edge_case_at_once(tt):
    """Non-RAG, no-ground-truth, healthy, failing and diagnosable, in one run."""
    traces = [
        _observe(tt, grounded()),
        _observe(tt, gold_absent_trace(0)),
        _observe(tt, production_trace(0)),
        _observe(tt, nonrag_ambiguity_trace(0)),
        Inference(prompt=None, generated_answer="x", id="bad"),
    ]
    rep = triage(traces, tt)
    assert rep.n_input == 5 and rep.n_failed == 1 and rep.n_analyzed == 4
    assert rep.n_diagnosed + rep.n_healthy + rep.n_abstained == 4
    assert sum(c.n for c in rep.clusters) == rep.n_diagnosed
    assert sum(rep.mode_counts.values()) == rep.n_diagnosed
    json.dumps(rep.to_dict(), allow_nan=False)


def test_traces_may_be_a_lazy_generator(tt):
    gen = (_observe(tt, gold_absent_trace(i)) for i in range(3))
    rep = triage(gen, tt)
    assert rep.n_input == 3 and rep.n_diagnosed == 3


def test_traces_without_ids_get_positional_exemplars(tt):
    traces = []
    for i in range(2):
        t = gold_absent_trace(i)
        t.id = None
        traces.append(_observe(tt, t))
    rep = triage(traces, tt)
    assert rep.clusters[0].exemplar_ids == ["#0", "#1"]


# --------------------------------------------------------------------------- #
# 6. JSON output
# --------------------------------------------------------------------------- #
def test_to_dict_is_strictly_valid_json(tt):
    traces = ([_observe(tt, gold_absent_trace(i)) for i in range(3)]
              + [_observe(tt, production_trace(i)) for i in range(2)]
              + [_observe(tt, grounded())])
    blob = json.dumps(triage(traces, tt).to_dict(), allow_nan=False)
    loaded = json.loads(blob)
    assert loaded["n_input"] == 6
    assert loaded["clusters"] and loaded["mode_counts"]
    # strict JSON rejects the bare NaN / Infinity tokens json.dumps would emit
    json.loads(blob, parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(f"non-RFC-8259 token in output: {c}")))


def test_to_dict_scrubs_non_finite_floats():
    """A degenerate log-odds contribution must not make the report unserializable."""
    c = TriageCluster(dominant_mode=M.HALLUCINATION, headline_signal="hal.redeep", n=1,
                      mean_headline_logodds=float("inf"),
                      mean_diagnostic_confidence=float("nan"),
                      priority_score=float("-inf"))
    with pytest.raises(ValueError):        # proves the raw values really are poison
        json.dumps({"raw": [c.mean_headline_logodds, c.mean_diagnostic_confidence]},
                   allow_nan=False)
    d = c.to_dict()
    assert d["mean_headline_logodds"] is None
    assert d["mean_diagnostic_confidence"] is None
    assert d["priority_score"] is None
    json.dumps(d, allow_nan=False)


def test_report_to_dict_keeps_counts_and_shares_separate():
    rep = triage(stub_traces(2), StubTT(
        [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent"),
         stub_report(M.HALLUCINATION, "hal.redeep", prob=0.1, abstained=True)]))
    d = rep.to_dict()
    assert d["mode_counts"] == {"prompt_ambiguity": 0, "retrieval_failure": 1,
                               "context_dilution": 0, "hallucination": 0,
                               "reasoning_failure": 0}
    assert d["mode_share"]["retrieval_failure"] == 1.0
    assert d["n_healthy"] == 1 and d["n_diagnosed"] == 1
    assert d["ranking_criterion"] == "n * mean_diagnostic_confidence"


# --------------------------------------------------------------------------- #
# 7. Text rendering
# --------------------------------------------------------------------------- #
def test_render_text_contains_the_actionable_columns(tt):
    traces = ([_observe(tt, gold_absent_trace(i)) for i in range(3)]
              + [_observe(tt, dilution()) for _ in range(2)])
    out = render_text(triage(traces, tt))
    assert "triage: 5 traces" in out
    assert "ret.gold_absent" in out and "add_gold_context" in out
    assert "absent-0" in out                       # exemplars are spot-checkable
    assert "not measured impact" in out            # the ranking caveat is on screen
    assert max(len(line) for line in out.splitlines()) < 140


def test_render_text_survives_an_empty_and_an_all_failed_report(tt):
    assert "triage: 0 traces" in render_text(triage([], tt))
    out = render_text(triage(stub_traces(2), StubTT([ValueError("nope")] * 2)))
    assert "analyzed=0" in out and "failed=2" in out


# --------------------------------------------------------------------------- #
# 8. Crash isolation covers the WHOLE per-trace path, not just tt.analyze
# --------------------------------------------------------------------------- #
def test_a_report_that_breaks_after_analysis_does_not_take_the_run_with_it():
    """`analyze` returning is not the end of the risky part.

    Reading the report — its confidence, its headline evidence line, its
    recommendations — happens on data triage did not build, and a hand-rolled ingest
    adapter or a third-party facade will eventually hand over a report with a ``None``
    where a float belongs. With only the analyze call guarded, trace #2 of 3 killed the
    run: traces #1 and #3 were lost and `failures` was empty, which is the exact
    outcome `TriageFailure` exists to prevent.
    """
    good = lambda: stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent")
    rep = triage(stub_traces(3), StubTT([
        good(),
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", conf=None),   # float(None)
        good(),
    ]))
    assert (rep.n_input, rep.n_analyzed, rep.n_failed) == (3, 2, 1)
    assert rep.n_diagnosed == 2
    (f,) = rep.failures
    assert (f.index, f.trace_id, f.error_type) == (1, "s1", "TypeError")
    # The survivors are intact, and the cluster counted each of them exactly once.
    assert [c.n for c in rep.clusters] == [2]
    assert rep.clusters[0].exemplar_ids == ["s0", "s2"]
    assert render_text(rep) and json.dumps(rep.to_dict(), allow_nan=False)


def test_a_headline_evidence_line_that_raises_is_isolated_too():
    """Between `analyze` and the accumulator sits `_headline`, which was also outside
    the guard: an evidence row with no family is enough to raise there."""
    rep = triage(stub_traces(3), StubTT([
        stub_report(M.CONTEXT_DILUTION, "dil.buried"),
        stub_report(M.CONTEXT_DILUTION, "dil.buried", family=None),   # None.value
        stub_report(M.CONTEXT_DILUTION, "dil.buried"),
    ]))
    assert (rep.n_analyzed, rep.n_failed, rep.n_diagnosed) == (2, 1, 2)
    assert rep.failures[0].error_type == "AttributeError"
    assert [c.n for c in rep.clusters] == [2]


def test_a_trace_stream_that_raises_mid_log_keeps_what_it_already_yielded(tt):
    """An ingest adapter dying on row 2 is the same event as a trace that fails to
    analyze — everything read before it must survive, and the break must be visible."""
    def broken_stream():
        yield _observe(tt, gold_absent_trace(0))
        raise ValueError("truncated JSONL at line 2")

    rep = triage(broken_stream(), tt)
    assert (rep.n_input, rep.n_analyzed, rep.n_failed) == (2, 1, 1)
    assert rep.n_diagnosed == 1 and rep.clusters[0].n == 1
    (f,) = rep.failures
    assert f.index == 1 and f.error_type == "ValueError" and "truncated" in f.error
    assert rep.input_stream_failed is True
    assert any("stream itself raised" in n for n in rep.notes), rep.notes
    assert rep.to_dict()["input_stream_failed"] is True


def test_a_mid_path_failure_never_half_counts_a_trace():
    """The buckets must still partition. A trace that dies after `analyze` returned
    but before it was filed used to be counted as analyzed AND as failed."""
    rep = triage(stub_traces(5), StubTT([
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent"),
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", conf=None),
        stub_report(M.HALLUCINATION, "hal.redeep", prob=0.1, abstained=True),
        stub_report(M.CONTEXT_DILUTION, "dil.buried", family=None),
        stub_report(M.CONTEXT_DILUTION, "dil.buried", prob=0.9, abstained=True),
    ]))
    assert rep.n_analyzed + rep.n_failed == rep.n_input == 5
    assert rep.n_diagnosed + rep.n_healthy + rep.n_abstained == rep.n_analyzed == 3
    assert sum(c.n for c in rep.clusters) == rep.n_diagnosed == 1
    for c in rep.clusters:
        assert c.fix_validated + c.fix_refuted + c.fix_unknown == c.n
    assert rep.n_with_ground_truth == 3, "the failed traces must not be counted either"


def test_keyboard_interrupt_is_not_swallowed_from_anywhere_in_the_path():
    """Ctrl-C must stay a Ctrl-C at every stage now guarded, not become a data point."""
    class Exploding:
        """A report that raises while triage is reading it, not while analyzing."""

        tier = Tier.WHITE
        abstained = False
        diagnoses = []

        @property
        def primary(self):
            raise KeyboardInterrupt

    def interrupting_stream():
        yield stub_traces(1)[0]
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        triage(stub_traces(1), StubTT([Exploding()]))
    with pytest.raises(KeyboardInterrupt):
        triage(interrupting_stream(), StubTT([stub_report(M.HALLUCINATION, "hal.redeep")]))


# --------------------------------------------------------------------------- #
# 9. `share` names its denominator
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mixed_bucket_report():
    """4 diagnosed of 6 analyzed of 7 supplied — every denominator a different number."""
    return triage(stub_traces(7), StubTT([
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent"),
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent"),
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent"),
        stub_report(M.CONTEXT_DILUTION, "dil.buried"),
        stub_report(M.HALLUCINATION, "hal.redeep", prob=0.1, abstained=True),
        stub_report(M.HALLUCINATION, "hal.redeep", prob=0.9, abstained=True),
        ValueError("bad row"),
    ]))


def test_share_is_a_fraction_of_the_diagnosed_traces(mixed_bucket_report):
    rep = mixed_bucket_report
    assert (rep.n_input, rep.n_analyzed, rep.n_diagnosed) == (7, 6, 4)
    shares = {c.headline_signal: c.share for c in rep.clusters}
    # 3/4 and 1/4 -- NOT 3/6 and 1/6 (analyzed) and NOT 3/7 and 1/7 (supplied).
    assert shares == {"ret.gold_absent": pytest.approx(0.75),
                      "dil.buried": pytest.approx(0.25)}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert all(c.share_denominator == rep.n_diagnosed for c in rep.clusters)


def test_the_share_denominator_ships_with_the_share_in_json(mixed_bucket_report):
    """A bare ratio is read as a share of the fleet: 4 diagnosed out of 7 makes a
    3-trace cluster render as 75%, and a reader who assumes "/ n_input" concludes far
    more of production is affected than is. The denominator is part of the number."""
    d = mixed_bucket_report.to_dict()["clusters"][0]
    assert d["share"] == pytest.approx(0.75)
    assert d["share_of"] == "n_diagnosed"
    assert d["share_denominator"] == 4
    assert d["share"] * d["share_denominator"] == pytest.approx(d["n"])


def test_the_share_denominator_is_on_screen_in_the_text_table(mixed_bucket_report):
    out = render_text(mixed_bucket_report)
    assert "%diag" in out, "the column header must carry its own denominator"
    assert "share of the 4 DIAGNOSED traces (of 7 supplied)" in out, out
    # The legend must not have cost the table its width budget (notes are prose and
    # wrap in the terminal; the table must not).
    table = out.split("\ntop clusters")[1].split("\nfailures")[0].splitlines()
    assert max(len(line) for line in table) < 140, table


def test_the_share_caveat_is_noted_below_the_old_abstention_threshold(mixed_bucket_report):
    """The clarifying note used to fire only once abstention passed 50%. This run
    abstains on 33%, which is where the misreading is most plausible."""
    rep = mixed_bucket_report
    assert rep.abstention_rate < 0.5
    assert any("DIAGNOSED" in n and "not of the 7 supplied" in n for n in rep.notes), rep.notes


# --------------------------------------------------------------------------- #
# 10. Non-finite numbers: ranking, and the two output surfaces agreeing
# --------------------------------------------------------------------------- #
def _nan_and_finite(order: str):
    """Two clusters, one with a NaN mean confidence, in the requested input order."""
    nan = [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", conf=float("nan"))] * 2
    ok = [stub_report(M.CONTEXT_DILUTION, "dil.buried", conf=0.9)] * 2
    outcomes = nan + ok if order == "nan_first" else ok + nan
    return triage(stub_traces(4), StubTT(outcomes))


@pytest.mark.parametrize("order", ["nan_first", "nan_last"])
def test_a_non_finite_score_does_not_hand_the_ranking_to_the_input_order(order):
    """NaN loses every comparison, so a NaN priority score made the sort a no-op for
    that cluster and let the arrival order decide rank 1 — while the code claimed a
    fully deterministic order and the JSON showed `"rank": 1` above a real score."""
    rep = _nan_and_finite(order)
    assert [c.headline_signal for c in rep.clusters] == ["dil.buried", "ret.gold_absent"]
    assert [c.rank for c in rep.clusters] == [1, 2]
    ranked = rep.to_dict()["clusters"]
    assert ranked[0]["priority_score"] == pytest.approx(1.8)
    assert ranked[1]["priority_score"] is None, "an unscoreable cluster cannot rank first"
    assert any("non-finite priority score" in n for n in rep.notes), rep.notes


def test_the_text_table_and_the_json_agree_that_a_non_finite_mean_is_not_a_number():
    """`to_dict` scrubs NaN/inf to null; the table printed a bare `+nan`, so the two
    surfaces disagreed about the same guarantee."""
    rep = triage(stub_traces(2), StubTT(
        [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", logodds=float("inf"))] * 2))
    c = rep.clusters[0]
    assert c.mean_headline_logodds == float("inf")          # the raw value is poison
    assert c.to_dict()["mean_headline_logodds"] is None
    out = render_text(rep)
    assert "n/a" in out
    assert "nan" not in out.lower() and "inf" not in out.lower(), out
    json.dumps(rep.to_dict(), allow_nan=False)


# --------------------------------------------------------------------------- #
# 11. Clustering keys on the headline line, and the mean it reports is SIGNED
# --------------------------------------------------------------------------- #
def test_clusters_key_on_the_headline_line_not_the_bookkeeping_row_above_it():
    """`Diagnosis.evidence` is magnitude-ordered and includes the base-rate prior, which
    outweighs the real reason on a cold-start report. Keying on `evidence[0]` would
    collapse every such cluster onto "base_rate" — one bucket, no engineering signal."""
    rep = triage(stub_traces(2), StubTT([
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", prior_first=True),
        stub_report(M.RETRIEVAL_FAILURE, "ret.irrelevant", prior_first=True),
    ]))
    assert {c.headline_signal for c in rep.clusters} == {"ret.gold_absent", "ret.irrelevant"}
    assert all(c.headline_source == "rule" for c in rep.clusters)
    assert all(c.headline_family == "retrieval" for c in rep.clusters)
    # The prior's -9.9 is the largest contribution in the ledger; it must not be the
    # number reported as the cluster's driving evidence.
    assert all(c.mean_headline_logodds == pytest.approx(2.0) for c in rep.clusters)


def test_mean_headline_logodds_keeps_the_sign_of_evidence_that_opposes_the_mode():
    """A headline line can OPPOSE its mode (the field says so). Storing the magnitude
    would silently turn "the strongest evidence argues against this" into its
    opposite, and every |x| test would still pass."""
    rep = triage(stub_traces(3), StubTT([
        stub_report(M.HALLUCINATION, "hal.redeep", logodds=-2.0),
        stub_report(M.HALLUCINATION, "hal.redeep", logodds=-2.0),
        stub_report(M.HALLUCINATION, "hal.redeep", logodds=1.0),
    ]))
    c = rep.clusters[0]
    assert c.mean_headline_logodds == pytest.approx(-1.0)     # not +1.667
    assert c.to_dict()["mean_headline_logodds"] == pytest.approx(-1.0)
    assert "-1.00" in render_text(rep)


# --------------------------------------------------------------------------- #
# 12. The exemplar cap, and the accumulator's memory bound
# --------------------------------------------------------------------------- #
def test_exemplars_are_capped_and_the_cap_is_the_caller_s():
    outcomes = [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent")] * 5
    capped = triage(stub_traces(5), StubTT(list(outcomes)), max_exemplars=2)
    assert capped.clusters[0].exemplar_ids == ["s0", "s1"]
    assert capped.clusters[0].n == 5, "the cap trims the sample, never the count"
    default = triage(stub_traces(5), StubTT(list(outcomes)))
    assert default.clusters[0].exemplar_ids == ["s0", "s1", "s2"]


def test_the_accumulator_holds_no_state_that_grows_with_the_number_of_traces():
    """The module header promises O(clusters) memory, not O(traces) — that is what
    lets a 10k-row generator stream through. Keeping per-trace lists of confidences /
    primaries / log-odds broke it silently (200k traces in one cluster held three
    200k-element lists), and every mean stayed correct while it did."""
    from tokentrace.triage.aggregate import _Accum

    def sizes(n: int) -> dict[str, int]:
        acc = _Accum(M.RETRIEVAL_FAILURE, "ret.gold_absent", "rule", "retrieval")
        report = stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent")
        for i, trace in enumerate(stub_traces(n)):
            acc.add(trace, report, 1.5, trace_key=f"t{i}", max_exemplars=3)
        assert acc.finish(n).n == n and acc.finish(n).mean_headline_logodds == 1.5
        return {k: len(v) for k, v in vars(acc).items() if hasattr(v, "__len__")}

    small, large = sizes(10), sizes(500)
    assert small == large, f"container sizes grew with the trace count: {small} -> {large}"


# --------------------------------------------------------------------------- #
# 13. The healthy/declined threshold is the engine's, or the caller's — never a constant
# --------------------------------------------------------------------------- #
def _one_abstention(prob: float, **kw):
    return triage(stub_traces(1),
                  StubTT([stub_report(M.HALLUCINATION, "hal.redeep",
                                      prob=prob, abstained=True)],
                         abstain_primary=kw.pop("abstain_primary", None)), **kw)


def test_an_explicit_detection_threshold_moves_the_healthy_declined_split():
    """Same report, two thresholds, opposite buckets: a hard-coded constant cannot
    satisfy both. Below the threshold the engine's own note reads "appears healthy";
    above it, it saw something and refused to rank it."""
    low = _one_abstention(0.45, detection_threshold=0.40)
    high = _one_abstention(0.45, detection_threshold=0.50)
    assert (low.n_healthy, low.n_abstained) == (0, 1)
    assert (high.n_healthy, high.n_abstained) == (1, 0)
    assert high.detection_threshold == 0.50 and high.threshold_origin == "explicit"


def test_the_threshold_defaults_to_the_engine_s_own_abstain_primary():
    rep = _one_abstention(0.60, abstain_primary=0.70)
    assert rep.detection_threshold == pytest.approx(0.70)
    assert rep.threshold_origin == "engine"
    assert (rep.n_healthy, rep.n_abstained) == (1, 0), "0.60 < 0.70 -> looks healthy"


def test_a_facade_with_no_threshold_says_so_instead_of_guessing_silently():
    from tokentrace.triage.aggregate import DEFAULT_DETECTION_THRESHOLD

    rep = _one_abstention(0.60)
    assert rep.threshold_origin == "assumed"
    assert rep.detection_threshold == pytest.approx(DEFAULT_DETECTION_THRESHOLD)
    assert any("assumed" in n for n in rep.notes), rep.notes


def test_the_healthy_declined_split_reads_the_top_marginal_not_the_primary():
    """The causal resolver can crown an upstream parent whose probability is LOWER
    than the strongest mode, while the engine's abstention gate keys off the maximum.
    Reading the wrong one misfiles traces between healthy and declined."""
    weak_primary = Diagnosis(mode=M.RETRIEVAL_FAILURE, probability=0.2, rank=1,
                             role=DiagnosisRole.PRIMARY_ROOT,
                             evidence=[EvidenceItem("ret.gold_absent", SignalFamily.RETRIEVAL,
                                                    0.0, 2.0)])
    strong_symptom = Diagnosis(mode=M.HALLUCINATION, probability=0.8, rank=0)
    report = DiagnosisReport(diagnoses=[strong_symptom, weak_primary], tier=Tier.WHITE,
                             diagnostic_confidence=0.5, abstained=True)
    rep = triage(stub_traces(1), StubTT([report]), detection_threshold=0.4)
    assert (rep.n_healthy, rep.n_abstained) == (0, 1), (
        "top marginal 0.8 >= 0.4 -> the engine saw something and declined; "
        "using primary.probability (0.2) would call this fleet healthy")


# --------------------------------------------------------------------------- #
# 14. The "nothing was measured" note, and what counts as a usable reference
# --------------------------------------------------------------------------- #
def test_the_nothing_was_measured_note_is_absent_when_something_was_measured():
    """`all(impact is None)` guards this note. Weakened to `any(...)`, a run with one
    scored cluster and one unscored cluster — the ordinary case — would assert that
    nothing in it was measured, right next to the cluster that was."""
    rep = triage(stub_traces(2), StubTT([
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", validated=True),
        stub_report(M.CONTEXT_DILUTION, "dil.buried", validated=None),
    ]))
    measured = [c.measured_impact for c in rep.clusters]
    assert None in measured and 1 in measured, measured        # genuinely mixed
    assert not any("was scored" in n for n in rep.notes), rep.notes


def test_the_nothing_was_measured_note_fires_when_truly_nothing_was_scored():
    """Guards the test above: the note must be capable of appearing."""
    rep = triage(stub_traces(2), StubTT(
        [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", validated=None)] * 2))
    assert rep.n_with_ground_truth == 2, "otherwise a different note fires"
    assert all(c.measured_impact is None for c in rep.clusters)
    assert any("was scored" in n and "UNKNOWN (not zero)" in n for n in rep.notes), rep.notes


@pytest.mark.parametrize("refs", [None, [], [""], ["   "]])
def test_an_empty_reference_is_not_a_ground_truth_reference(refs):
    """`[""]` arrives routinely from a filtered list or a JSON payload. Counting it as
    a usable reference claims those traces' fixes COULD have been verified, which
    turns "unknown impact" into an unexplained zero."""
    traces = [Inference(f"q{i}", f"a{i}", [Chunk("c")], refs, id=f"e{i}") for i in range(2)]
    rep = triage(traces, StubTT(
        [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent")] * 2))
    assert rep.n_with_ground_truth == 0
    assert rep.clusters[0].n_with_ground_truth == 0
    assert any("No trace carried a ground-truth reference" in n for n in rep.notes), rep.notes


def test_a_usable_reference_is_counted():
    """Guards the test above: `n_with_ground_truth` must be capable of being non-zero."""
    rep = triage(stub_traces(2), StubTT(
        [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent")] * 2))
    assert rep.n_with_ground_truth == 2 and rep.clusters[0].n_with_ground_truth == 2


# --------------------------------------------------------------------------- #
# 15. The tier reported is the tier that RAN
# --------------------------------------------------------------------------- #
def test_the_reported_tier_is_the_lowest_one_that_actually_ran():
    """`ModelHandle.with_tier` only ever down-caps, so a run requested at white-box can
    execute at grey. Seeding the report's tier from the request let a fleet report
    claim mechanistic evidence it never had."""
    rep = triage(stub_traces(3), StubTT([
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", tier=Tier.WHITE),
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", tier=Tier.BLACK),
        stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", tier=Tier.GREY),
    ]), tier=Tier.WHITE)
    assert rep.tier is Tier.BLACK, "the requested tier is not evidence of what ran"
    assert rep.to_dict()["tier"] == "black"

    # The request cannot seed it in the other direction either: `run_tier` is READ
    # OFF the reports, so a run asked for black-box whose reports all came back
    # white-box is reported as the white-box run it was.
    upgraded = triage(stub_traces(2), StubTT(
        [stub_report(M.RETRIEVAL_FAILURE, "ret.gold_absent", tier=Tier.WHITE)] * 2),
        tier=Tier.BLACK)
    assert upgraded.tier is Tier.WHITE


# --------------------------------------------------------------------------- #
# 16. Scored interventions are counted as interventions, not as traces
# --------------------------------------------------------------------------- #
def test_scored_interventions_are_counted_separately_from_scored_traces(tt):
    """One trace can carry several distinct fixes. `fix_validated` counts TRACES (so it
    can never exceed n); `n_scored_interventions` counts the experiments that actually
    ran, and is the MEASURED denominator behind them."""
    rep = triage([_observe(tt, gold_absent_trace(i)) for i in range(3)], tt)
    c = rep.clusters[0]
    assert c.fix_validated == 3 and c.fix_validated <= c.n
    assert c.n_scored_interventions >= c.fix_validated + c.fix_refuted
    assert c.to_dict()["n_scored_interventions"] == c.n_scored_interventions
    unscored = triage(stub_traces(2), StubTT(
        [stub_report(M.HALLUCINATION, "hal.redeep", validated=None)] * 2))
    assert unscored.clusters[0].n_scored_interventions == 0
