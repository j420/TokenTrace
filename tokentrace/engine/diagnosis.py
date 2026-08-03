"""The diagnosis engine: rules-prior (+) learned residual -> calibrated ranked
diagnoses with an additive evidence ledger, causal roles, a conformal Top-3 set,
a diagnostic-confidence score, and an abstention gate.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import (
    ALL_MODES,
    Diagnosis,
    DiagnosisReport,
    FailureMode,
    FeatureVector,
    Inference,
    SignalFamily,
    Tier,
)
from tokentrace.engine.calibration import Calibrator, sigmoid
from tokentrace.engine.causal import CausalResolver, EvidenceAttributor
from tokentrace.engine.classifier import ResidualClassifier
from tokentrace.engine.conformal import ConformalPredictor
from tokentrace.engine.rules import evaluate_rules

M = FailureMode
_MASK_LOGIT = -50.0
# Contrasts that genuinely need mechanistic evidence to separate.
_MECH_CONTRASTS = {frozenset({M.HALLUCINATION, M.CONTEXT_DILUTION}),
                   frozenset({M.HALLUCINATION, M.RETRIEVAL_FAILURE}),
                   frozenset({M.CONTEXT_DILUTION, M.REASONING_FAILURE})}


class DiagnosisEngine:
    def __init__(
        self,
        classifier: Optional[ResidualClassifier] = None,
        calibrator: Optional[Calibrator] = None,
        resolver: Optional[CausalResolver] = None,
        attributor: Optional[EvidenceAttributor] = None,
        conformal: Optional[ConformalPredictor] = None,
        recommender=None,
        abstain_confidence: float = 0.35,
        abstain_primary: float = 0.40,
    ):
        self.classifier = classifier or ResidualClassifier()
        self.calibrator = calibrator
        self.resolver = resolver or CausalResolver(primary_min=abstain_primary)
        self.attributor = attributor or EvidenceAttributor()
        self.conformal = conformal or ConformalPredictor()
        self.recommender = recommender
        self.abstain_confidence = abstain_confidence
        self.abstain_primary = abstain_primary

    # ------------------------------------------------------------------ #
    def raw_logits(self, fv: FeatureVector, inference: Inference,
                   rule_logits=None, residual=None) -> dict[FailureMode, float]:
        """z_mode = rule_prior + learned_residual, with hard masks applied.

        ``rule_logits``/``residual`` may be passed in when the caller has already
        computed them, so a single diagnosis does not evaluate the rules twice and
        the LightGBM heads twice.
        """
        if rule_logits is None:
            rule_logits, _ = evaluate_rules(fv)
        if residual is None:
            residual = self.classifier.residual(fv)
        z = {m: rule_logits[m] + residual[m] for m in ALL_MODES}
        if not inference.is_rag:
            z[M.RETRIEVAL_FAILURE] = _MASK_LOGIT
            z[M.CONTEXT_DILUTION] = _MASK_LOGIT
        return z

    def probabilities(self, fv: FeatureVector, inference: Inference,
                      rule_logits=None, residual=None) -> dict[FailureMode, float]:
        z = self.raw_logits(fv, inference, rule_logits, residual)
        sig = fv.missingness_signature()
        out = {}
        for m in ALL_MODES:
            if z[m] <= _MASK_LOGIT:
                out[m] = 0.0
            elif self.calibrator is not None:
                out[m] = round(self.calibrator.transform(m, sig, z[m]), 4)
            else:
                out[m] = round(sigmoid(z[m]), 4)
        return out

    # ------------------------------------------------------------------ #
    def diagnose(self, inference: Inference, fv: FeatureVector, tier: Tier) -> DiagnosisReport:
        rule_logits, fired = evaluate_rules(fv)
        residual, shap = self.classifier.contributions(fv)   # one LightGBM pass
        probs = self.probabilities(fv, inference, rule_logits, residual)
        roles = self.resolver.resolve(probs)

        ranked = sorted(ALL_MODES, key=lambda m: -probs[m])
        diagnoses: list[Diagnosis] = []
        for rank, m in enumerate(ranked):
            role, parents = roles.get(m, (None, []))
            evidence = self.attributor.attribute(m, fv, fired, shap.get(m, {}), inference)
            diagnoses.append(Diagnosis(
                mode=m, probability=probs[m], rank=rank, role=role,
                causal_parents=parents, evidence=evidence,
                confidence=self._per_mode_confidence(probs[m], evidence),
            ))

        conformal_set = self.conformal.predict(probs)
        diag_conf = self._diagnostic_confidence(probs, conformal_set, fv, inference.is_rag)
        top1 = probs[ranked[0]]
        abstained = (top1 < self.abstain_primary) or (diag_conf < self.abstain_confidence)

        notes = self._notes(inference, fv, probs, ranked, diag_conf, tier, abstained)

        report = DiagnosisReport(
            diagnoses=diagnoses, tier=tier, inference_id=inference.id,
            conformal_set=conformal_set, diagnostic_confidence=round(diag_conf, 4),
            abstained=abstained, notes=notes,
        )
        if self.recommender is not None:
            self.recommender.attach(report, inference, fv)
        return report

    # ------------------------------------------------------------------ #
    def _per_mode_confidence(self, prob: float, evidence) -> float:
        mass = sum(abs(e.contribution_logodds) for e in evidence)
        return round(min(1.0, prob * (0.6 + 0.4 * min(1.0, mass / 4.0))), 4)

    def _diagnostic_confidence(self, probs, conformal_set, fv: FeatureVector,
                              is_rag: bool = True) -> float:
        ranked = sorted((probs[m] for m in ALL_MODES), reverse=True)
        top1 = ranked[0]
        margin = top1 - (ranked[1] if len(ranked) > 1 else 0.0)
        # Normalize by APPLICABLE families: the retrieval family can never be present
        # for non-RAG inputs, so dividing by 4 there would unfairly depress confidence.
        applicable = 4 if is_rag else 3
        present = sum(1 for f in SignalFamily if fv.family_present.get(f))
        family_completeness = min(1.0, present / applicable)
        set_penalty = 1.0 / max(1, len(conformal_set))
        conf = top1 * (0.55 + 0.45 * margin) * (0.6 + 0.4 * family_completeness)
        conf *= (0.7 + 0.3 * set_penalty)
        return max(0.0, min(1.0, conf))

    def _notes(self, inference, fv, probs, ranked, diag_conf, tier, abstained) -> list[str]:
        notes: list[str] = []
        top1 = probs[ranked[0]]

        if top1 < self.abstain_primary:
            notes.append("No failure mode exceeds the detection threshold — the answer "
                         "appears healthy on the available evidence.")
        elif abstained:
            notes.append("Insufficient evidence to rank a primary cause confidently "
                         f"(diagnostic confidence {diag_conf:.2f}).")

        # Production path: no reference, so nothing can confirm the answer is wrong.
        if not inference.has_ground_truth:
            notes.append("No ground-truth reference: these are RISK estimates, not "
                         "confirmed errors (correctness-dependent evidence is unavailable).")

        # Right-answer-for-wrong-reasons / fragile.
        is_correct = fv.get("is_correct")
        supported = fv.get("answer_supported_by_context")
        if is_correct == 1.0 and supported is not None and supported < 0.3:
            notes.append("Correct-but-unsupported: the answer matches ground truth but is "
                         "not grounded in the retrieved context (parametric recall — fragile).")

        # Tier-escalation when a mechanistic-dependent contrast is unresolved.
        if tier < Tier.GREY and diag_conf < 0.5:
            top2 = frozenset(ranked[:2])
            if top2 in _MECH_CONTRASTS:
                notes.append("Escalate to grey/white-box: separating "
                             f"{ranked[0].pretty} from {ranked[1].pretty} needs mechanistic "
                             "evidence (attention-to-context / logit-lens / causal patching).")
        return notes
