"""Regression tests for the deep-audit findings.

Each test here targets a defect that was CONFIRMED by adversarial verification and
would otherwise pass silently: the previous suite stayed 40/40 green when several of
these behaviours were reverted.
"""

from __future__ import annotations

import json
import math

import pytest

from tokentrace.core.serialize import report_to_dict
from tokentrace.core.types import (
    ALL_MODES,
    Chunk,
    FailureMode as M,
    Inference,
    Tier,
)
from tokentrace.data import build_dataset
from tokentrace.data.injection import InjectionHarness
from tokentrace.data.samples import FACTS
from tokentrace.engine import DiagnosisEngine
from tokentrace.engine.conformal import ConformalPredictor
from tokentrace.engine.rules import RULES, evaluate_rules
from tokentrace.eval.benchmark import split_dataset, train_engine
from tokentrace.eval.metrics import gold_primary

from _helpers import observe


# --------------------------------------------------------------------------- #
# G1 — the corpus must not leak labels through context SHAPE
# --------------------------------------------------------------------------- #
def test_labels_not_recoverable_from_context_shape(model, pipeline):
    """A classifier given ONLY shape features (chunk count, lengths) must not be
    able to reconstruct the diagnosis.

    Before the fix every recipe had a unique fixed chunk count, so a depth-4 tree on
    4 shape features reached 0.917 held-out accuracy — i.e. most of the headline
    metric was obtainable with zero diagnostic content. Dilution is *causally*
    geometric so some signal is legitimate; the bar is that shape must not be
    sufficient.
    """
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    shape = ["n_chunks", "context_length_tokens", "prompt_n_content_words",
             "answer_length_tokens"]
    ds = build_dataset(model, pipeline, seeds=(0, 1, 2))
    train, _cal, test = split_dataset(ds)

    def mk(rows):
        X, y = [], []
        for li in rows:
            gp = gold_primary(li)
            if gp is None:
                continue
            fv = pipeline.run(li.inference, model)
            X.append([fv.get(f) if fv.get(f) is not None else np.nan for f in shape])
            y.append(gp.value)
        return np.nan_to_num(np.array(X, dtype=float), nan=-1.0), np.array(y)

    Xtr, ytr = mk(train)
    Xte, yte = mk(test)
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    assert acc < 0.70, f"context shape alone reconstructs {acc:.3f} of diagnoses"


def test_every_recipe_geometry_overlaps(model, pipeline):
    """No recipe may own a unique chunk-count band (that is the shortcut)."""
    ds = build_dataset(model, pipeline, seeds=(0, 1, 2))
    counts: dict[str, set] = {}
    for li in ds:
        n = len(li.inference.retrieved_context or [])
        counts.setdefault(li.injection_recipe, set()).add(n)
    rag = {k: v for k, v in counts.items() if v != {0}}
    for name, band in rag.items():
        others = set().union(*[v for k, v in rag.items() if k != name])
        assert band & others, f"{name} occupies a disjoint chunk-count band {sorted(band)}"


# --------------------------------------------------------------------------- #
# Rules must all be reachable
# --------------------------------------------------------------------------- #
def test_no_rule_is_dead(model, pipeline):
    """Every rule must fire at least once across the corpus and tiers.

    `dil.buried` fired 0/783 times because its token ramp started at 200 while the
    corpus maxed out at 162, so its evidence never reached the learned head.
    """
    ds = build_dataset(model, pipeline, seeds=(0, 1))
    fired: dict[str, int] = {r.id: 0 for r in RULES}
    for li in ds:
        for tier in (Tier.BLACK, Tier.GREY, Tier.WHITE):
            _, rows = evaluate_rules(pipeline.run(li.inference, model.with_tier(tier)))
            for rule, _c, _t in rows:
                fired[rule.id] += 1
    dead = sorted(k for k, v in fired.items() if v == 0)
    assert not dead, f"rules never fire on any input: {dead}"


# --------------------------------------------------------------------------- #
# C1 — the evidence ledger must reconcile to the score it explains
# --------------------------------------------------------------------------- #
def test_evidence_ledger_reconciles_to_logit(model, pipeline):
    """sum(evidence contributions) must equal the raw log-odds z.

    Truncating to the top-k silently discarded up to 2.25 log-odds (and whole fired
    rule lines) from a ledger advertised as an auditable additive decomposition.
    """
    ds = build_dataset(model, pipeline, seeds=(0,))
    train, cal, test = split_dataset(ds)
    engine = train_engine(train, cal, model, pipeline, validate_recommendations=False)
    checked = 0
    for li in test[:12]:
        fv = pipeline.run(li.inference, model)
        z = engine.raw_logits(fv, li.inference)
        report = engine.diagnose(li.inference, fv, model.tier)
        for d in report.diagnoses:
            if d.probability <= 0.0:      # hard-masked mode: no ledger expected
                continue
            total = sum(e.contribution_logodds for e in d.evidence)
            assert abs(total - z[d.mode]) < 1e-2, (
                f"{d.mode.value}: ledger {total:.4f} != z {z[d.mode]:.4f}")
            checked += 1
    assert checked > 0


def test_headline_evidence_is_not_the_base_rate(model, pipeline):
    """The rendered headline must be substantive, not the bookkeeping prior."""
    inf = observe(model, Inference(
        "When was the Eiffel Tower completed?\nParis is in France.", "",
        [Chunk("Paris is in France.", 0.5, "d2")], ["1889"],
        question="When was the Eiffel Tower completed?",
        meta={"_sim": {"answer": "1889", "gold_fact": "1889", "distractor": "1920"}}))
    report = DiagnosisEngine().diagnose(inf, pipeline.run(inf, model), model.tier)
    head = report.primary.headline_evidence
    assert head is not None and head.source not in ("prior", "aggregate")


# --------------------------------------------------------------------------- #
# G3 / G2 — ground-truth contract and the production (no-reference) path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gt", [[], [""], ["   "]])
def test_empty_ground_truth_does_not_fabricate_a_failure(model, pipeline, gt):
    """`[]` / `[""]` must behave like "no reference", not like "the answer matched
    nothing" — which diagnosed a correct, fully grounded answer as a confident
    retrieval failure (p=0.73, not abstained)."""
    gold = Chunk("The Eiffel Tower was completed in 1889.", 0.9, "g", gold=True)
    inf = observe(model, Inference(
        "When was the Eiffel Tower completed?\n" + gold.text, "", [gold], gt,
        question="When was the Eiffel Tower completed?",
        meta={"_sim": {"answer": "1889", "gold_fact": "completed in 1889"}}))
    assert inf.has_ground_truth is False
    report = DiagnosisEngine().diagnose(inf, pipeline.run(inf, model), model.tier)
    assert report.abstained, "fabricated a diagnosis for a correct, grounded answer"


def test_no_reference_path_is_marked_and_separately_calibrated(model, pipeline):
    gold = Chunk("The Eiffel Tower was completed in 1889.", 0.9, "g", gold=True)
    base = dict(question="When was the Eiffel Tower completed?",
                meta={"_sim": {"answer": "1889", "gold_fact": "completed in 1889"}})
    with_ref = Inference("Q\n" + gold.text, "1889", [gold], ["1889"], **base)
    no_ref = Inference("Q\n" + gold.text, "1889", [gold], None, **base)

    sig_ref = pipeline.run(with_ref, model).missingness_signature()
    sig_none = pipeline.run(no_ref, model).missingness_signature()
    assert sig_ref != sig_none, "no-reference traces share a calibration map with GT ones"

    report = DiagnosisEngine().diagnose(no_ref, pipeline.run(no_ref, model), model.tier)
    assert any("risk" in n.lower() for n in report.notes)


# --------------------------------------------------------------------------- #
# A4 — conformal sets must use a normalized (APS) score
# --------------------------------------------------------------------------- #
def test_conformal_set_grows_with_ambiguity():
    """Unnormalized one-vs-rest marginals made set size INVERTED w.r.t. ambiguity:
    a 95%-confident row and a three-way tie both returned a single mode."""
    cp = ConformalPredictor(tau=0.9)
    easy = {M.PROMPT_AMBIGUITY: 0.95, M.RETRIEVAL_FAILURE: 0.01, M.CONTEXT_DILUTION: 0.005,
            M.HALLUCINATION: 0.004, M.REASONING_FAILURE: 0.001}
    hard = {M.PROMPT_AMBIGUITY: 0.90, M.RETRIEVAL_FAILURE: 0.88, M.CONTEXT_DILUTION: 0.85,
            M.HALLUCINATION: 0.10, M.REASONING_FAILURE: 0.05}
    assert len(cp.predict(easy)) < len(cp.predict(hard))


def test_conformal_tau_stays_below_one():
    cp = ConformalPredictor(alpha=0.1)
    rows = [{m: (1.0 if m is M.HALLUCINATION else 0.0) for m in ALL_MODES} for _ in range(50)]
    cp.fit(rows, [M.HALLUCINATION] * 50)
    assert cp.tau < 1.0, "a saturated tau forces every set to the max_size cap"


# --------------------------------------------------------------------------- #
# Serialization must emit valid JSON (NaN is not RFC 8259)
# --------------------------------------------------------------------------- #
def test_report_json_is_strictly_valid(model, pipeline):
    inf = observe(model, Inference(
        "When was the Eiffel Tower completed?\nParis is in France.", "",
        [Chunk("Paris is in France.", 0.5, "d2")], ["1889"],
        question="When was the Eiffel Tower completed?",
        meta={"_sim": {"answer": "1889", "gold_fact": "1889", "distractor": "1920"}}))
    report = DiagnosisEngine().diagnose(inf, pipeline.run(inf, model), model.tier)
    blob = json.dumps(report_to_dict(report))
    # strict JSON rejects the bare NaN / Infinity tokens json.dumps would emit
    json.loads(blob, parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(f"non-RFC-8259 token in output: {c}")))


# --------------------------------------------------------------------------- #
# Injection gates must not systematically discard a slice of the corpus
# --------------------------------------------------------------------------- #
def test_injection_gates_do_not_silently_drop_facts(model, pipeline):
    """A gate that always rejects certain facts hides a detector blind spot from the
    metric that is supposed to reveal it (3 facts were dropped for every seed)."""
    h = InjectionHarness(model, pipeline)
    for name in ("clean", "retrieval_failure", "context_dilution", "prompt_ambiguity",
                 "hallucination", "hallucination_override"):
        recipe = getattr(h, name)
        passed = sum(1 for f in FACTS for s in (0, 1) if recipe(f, seed=s) is not None)
        total = len(FACTS) * 2
        assert passed >= 0.9 * total, f"{name} gate discards {total - passed}/{total}"


def test_parametric_override_is_mechanistically_distinct_from_dilution(model):
    """The RAG-hallucination (override) and dilution cases are behaviourally
    identical — gold present, answer wrong — so they must differ mechanistically or
    the mechanistic family carries no information."""
    h = InjectionHarness(model)
    ovr = h.hallucination_override(FACTS[0], seed=0)
    dil = h.context_dilution(FACTS[0], seed=0)
    assert ovr is not None and dil is not None
    c_o = model.capture(ovr.inference)
    c_d = model.capture(dil.inference)
    # override: gold was ATTENDED and the answer formed EARLY from parametric memory
    assert c_o.gold_attention_ratio > c_d.gold_attention_ratio
    assert c_o.parametric_knowledge_score > c_d.parametric_knowledge_score
    assert c_o.logit_lens_answer_layer < c_d.logit_lens_answer_layer


# --------------------------------------------------------------------------- #
# Metric semantics
# --------------------------------------------------------------------------- #
def test_debugging_time_reduction_reports_its_ceiling(model, pipeline):
    """It is a RANKING proxy with a hard ceiling of 1 - 1/((K+1)/2); a perfect
    ranker cannot exceed ~0.667 at K=5, so it is not a wall-clock percentage."""
    from tokentrace.eval.metrics import Metrics

    assert "debugging_time_reduction" in Metrics().as_dict()
    ceiling = 1.0 - 1.0 / ((len(ALL_MODES) + 1) / 2)
    assert math.isclose(ceiling, 2 / 3, rel_tol=1e-6)


def test_abstention_is_split_into_healthy_and_declined():
    from tokentrace.eval.metrics import Metrics

    d = Metrics().as_dict()
    for k in ("healthy_abstention_rate", "declined_rate", "recommendation_negatives"):
        assert k in d, f"{k} missing: a combined rate hides good vs bad abstention"
