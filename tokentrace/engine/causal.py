"""Causal resolver + evidence attributor.

Causal resolver: a fixed failure-mode DAG turns five independent marginals into a
root -> sequela story ("treat the root, not the symptom"). Edges are *validated at
runtime* by the engine's confirmatory interventions; here we assign roles from the
DAG given which modes are active.

Evidence attributor: because the learned heads use the rule log-odds as init_score,
the total score decomposes into rule contributions + TreeSHAP contributions, both
in log-odds. The attributor merges them into one ranked, auditable ledger.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import (
    ALL_MODES,
    DiagnosisRole,
    EvidenceItem,
    FailureMode,
    FeatureVector,
    Inference,
    SignalFamily,
)
from tokentrace.engine.rules import Rule
from tokentrace.signals.features import family_of

M = FailureMode

# Fixed causal DAG: parent -> children (upstream causes -> downstream effects).
CAUSAL_EDGES: dict[FailureMode, tuple[FailureMode, ...]] = {
    M.PROMPT_AMBIGUITY: (M.REASONING_FAILURE, M.HALLUCINATION),
    M.RETRIEVAL_FAILURE: (M.HALLUCINATION,),
    M.CONTEXT_DILUTION: (M.HALLUCINATION, M.REASONING_FAILURE),
}

# Depth = longest path from a root (used for topological ordering).
_DEPTH: dict[FailureMode, int] = {
    M.PROMPT_AMBIGUITY: 0,
    M.RETRIEVAL_FAILURE: 0,
    M.CONTEXT_DILUTION: 0,
    M.REASONING_FAILURE: 1,
    M.HALLUCINATION: 1,
}


def parents_of(mode: FailureMode) -> list[FailureMode]:
    return [p for p, kids in CAUSAL_EDGES.items() if mode in kids]


class CausalResolver:
    def __init__(self, active_threshold: float = 0.25, primary_min: float = 0.40,
                 parent_ratio: float = 0.7):
        self.active_threshold = active_threshold
        self.primary_min = primary_min
        self.parent_ratio = parent_ratio

    def resolve(
        self, probs: dict[FailureMode, float]
    ) -> dict[FailureMode, tuple[DiagnosisRole, list[FailureMode]]]:
        active = [m for m in ALL_MODES if probs[m] > self.active_threshold]
        roles: dict[FailureMode, tuple[DiagnosisRole, list[FailureMode]]] = {}
        if not active:
            return roles

        # Start from the strongest active mode and walk *upstream* only while a
        # parent is comparably strong (>= parent_ratio x child prob). This treats a
        # dominant downstream mode as the root when its supposed cause is a weak
        # co-fire, instead of blindly preferring upstream modes.
        primary = max(active, key=lambda m: probs[m])
        walked = True
        while walked:
            walked = False
            for p in parents_of(primary):
                if p in active and probs[p] >= self.parent_ratio * probs[primary]:
                    primary = p
                    walked = True
                    break

        for m in active:
            active_parents = [p for p in parents_of(m) if p in active]
            if m == primary:
                roles[m] = (DiagnosisRole.PRIMARY_ROOT, [])
            elif active_parents:
                roles[m] = (DiagnosisRole.SEQUELA, active_parents)
            else:
                roles[m] = (DiagnosisRole.CONTRIBUTING, [])
        return roles


class EvidenceAttributor:
    """Builds the per-mode additive log-odds evidence ledger."""

    def __init__(self, max_items: int = 6):
        self.max_items = max_items

    def attribute(
        self,
        mode: FailureMode,
        fv: FeatureVector,
        fired_rules: list[tuple[Rule, float, str]],
        shap: dict[str, float],
        inference: Optional[Inference] = None,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []

        # Rule contributions for this mode.
        for rule, contrib, text in fired_rules:
            if rule.mode != mode:
                continue
            items.append(EvidenceItem(
                signal=rule.id,
                family=rule.family,
                value=self._lead_value(fv, rule.required),
                contribution_logodds=round(contrib, 4),
                direction="supports" if contrib > 0 else "opposes",
                source="rule",
                provenance=self._rule_provenance(rule, fv, inference),
                rendered=text,
            ))

        # Learned (TreeSHAP) contributions for this mode.
        for feat, contrib in sorted(shap.items(), key=lambda kv: -abs(kv[1])):
            val = fv.get(feat)
            items.append(EvidenceItem(
                signal=feat,
                family=family_of(feat) if feat in _KNOWN_FEATS() else SignalFamily.PROMPT,
                value=val if val is not None else float("nan"),
                contribution_logodds=round(contrib, 4),
                direction="supports" if contrib > 0 else "opposes",
                source="learned",
                provenance={"feature": feat},
                rendered=f"{feat} = {val} contributes {contrib:+.2f} log-odds (learned)",
            ))

        items.sort(key=lambda e: -abs(e.contribution_logodds))
        return items[: self.max_items]

    @staticmethod
    def _lead_value(fv: FeatureVector, required: tuple[str, ...]) -> float:
        for r in required:
            v = fv.get(r)
            if v is not None:
                return v
        return float("nan")

    @staticmethod
    def _rule_provenance(rule: Rule, fv: FeatureVector, inference: Optional[Inference]) -> dict:
        prov: dict = {"rule": rule.id, "signals": {r: fv.get(r) for r in rule.required}}
        # Point retrieval evidence back to a concrete chunk when possible.
        if inference is not None and inference.retrieved_context and rule.family == SignalFamily.RETRIEVAL:
            gold = next((c for c in inference.retrieved_context if c.gold), None)
            if gold is not None:
                prov["chunk"] = {"source_id": gold.source_id, "text": gold.text[:120]}
        return prov


def _KNOWN_FEATS() -> set[str]:
    from tokentrace.signals.features import ALL_FEATURES

    return set(ALL_FEATURES)
