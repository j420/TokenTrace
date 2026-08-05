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
                validated: bool | None = None) -> DiagnosisReport:
    ev = EvidenceItem(signal=signal, family=SignalFamily.RETRIEVAL, value=0.0,
                      contribution_logodds=2.0, source="rule")
    recs = ([Recommendation(action=action, description="", targets_mode=mode,
                            priority=0, validated=validated)] if action else [])
    diag = Diagnosis(mode=mode, probability=prob, rank=0, role=DiagnosisRole.PRIMARY_ROOT,
                     evidence=[ev], recommendations=recs)
    return DiagnosisReport(diagnoses=[diag], tier=Tier.WHITE, diagnostic_confidence=conf,
                           abstained=abstained)


class StubTT:
    """Facade double: replays a prepared report (or raises) per trace, in order."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self._i = 0

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
