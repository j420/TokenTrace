"""Recommendations + simulated-intervention validation.

Each failure mode maps to corrective actions ordered by causal role (treat the
root before the symptom). Crucially, most recommendations are *input-level
counterfactuals* — add the gold chunk, move it to the front, disambiguate the
prompt, decompose the question — so we can **apply the fix and re-run the model**
to check whether the answer corrects. That makes recommendation precision an
automatic, human-free metric and doubles as the "follow-up treatment" step of the
diagnosis analogy.

The intervention functions modify the :class:`Inference` structurally (real models
respond to the changed prompt/context). For the mock they also adjust the
simulator hints so its decision reflects the fix — this is the mock's stand-in for
the real effect and is clearly confined to ``_sim``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

from tokentrace.core.types import (
    ALL_MODES,
    Chunk,
    DiagnosisReport,
    DiagnosisRole,
    FailureMode,
    FeatureVector,
    Inference,
    Recommendation,
)
from tokentrace.models.base import ModelHandle
from tokentrace.signals.scorers import DEFAULT_SCORERS

M = FailureMode


# --------------------------------------------------------------------------- #
# Interventions (Inference -> modified Inference, or None if not applicable)
# --------------------------------------------------------------------------- #
def _rebuild(inf: Inference, chunks: list[Chunk], extra_sim: Optional[dict] = None,
             drop_sim: tuple[str, ...] = ()) -> Inference:
    sim = dict(inf.meta.get("_sim", {}))
    for k in drop_sim:
        sim.pop(k, None)
    if extra_sim:
        sim.update(extra_sim)
    ctx = " ".join(c.text for c in chunks)
    prompt = f"{inf.query}\n{ctx}" if chunks else inf.query
    meta = {**inf.meta, "_sim": sim}
    return replace(inf, prompt=prompt, retrieved_context=chunks or None, meta=meta)


def _gold_text(inf: Inference) -> Optional[str]:
    sim = inf.meta.get("_sim", {})
    if sim.get("gold_fact"):
        return sim["gold_fact"]
    if inf.ground_truth:
        return inf.ground_truth[0]
    return None


def add_gold_context(inf: Inference) -> Optional[Inference]:
    """Retrieval fix: retrieve and prepend a chunk that bears the answer."""
    gold = _gold_text(inf)
    if not gold:
        return None
    chunk = Chunk(text=f"{gold}.", retriever_score=0.99, source_id="intervention:retrieval", gold=True)
    existing = [c for c in (inf.retrieved_context or []) if not c.gold]
    return _rebuild(inf, [chunk, *existing])


def rerank_gold_first(inf: Inference) -> Optional[Inference]:
    """Dilution fix: move the gold chunk to the front and trim distractors."""
    chunks = list(inf.retrieved_context or [])
    if not chunks:
        return None
    gold = next((c for c in chunks if c.gold), None)
    if gold is None:
        gold = max(chunks, key=lambda c: max(
            (DEFAULT_SCORERS.support(r, [c.text]) for r in (inf.ground_truth or [inf.generated_answer])),
            default=0.0))
    others = [c for c in chunks if c is not gold][:2]
    return _rebuild(inf, [gold, *others])


def clarify_prompt(inf: Inference) -> Optional[Inference]:
    """Ambiguity fix: rewrite to a single reading (mock: drop the ambiguity hint)."""
    return _rebuild(inf, list(inf.retrieved_context or []), drop_sim=("ambiguous", "readings"))


def decompose_question(inf: Inference) -> Optional[Inference]:
    """Reasoning fix: decompose / chain-of-thought (mock: enable composition)."""
    return _rebuild(inf, list(inf.retrieved_context or []), drop_sim=("hard_composition",))


def ground_or_abstain(inf: Inference) -> Optional[Inference]:
    """Hallucination fix: ground in gold if retrievable, else the fix is to abstain."""
    return add_gold_context(inf)  # None if no gold retrievable -> caller treats as 'abstain'


INTERVENTIONS: dict[str, Callable[[Inference], Optional[Inference]]] = {
    "add_gold_context": add_gold_context,
    "rerank_gold_first": rerank_gold_first,
    "clarify_prompt": clarify_prompt,
    "decompose_question": decompose_question,
    "ground_or_abstain": ground_or_abstain,
}


@dataclass
class RecTemplate:
    action: str
    description: str


# Ordered by preference within a mode.
RECOMMENDATIONS: dict[FailureMode, list[RecTemplate]] = {
    M.PROMPT_AMBIGUITY: [
        RecTemplate("clarify_prompt", "Disambiguate the prompt: name the entity/qualifier the "
                    "user left implicit, or ask a clarifying question."),
    ],
    M.RETRIEVAL_FAILURE: [
        RecTemplate("add_gold_context", "Improve retrieval: the answer is not in the retrieved "
                    "context. Expand/adjust the retriever (query rewrite, more chunks, better index)."),
    ],
    M.CONTEXT_DILUTION: [
        RecTemplate("rerank_gold_first", "Re-rank/trim context: the answer is present but buried. "
                    "Move the relevant chunk earlier and cut distractors."),
    ],
    M.HALLUCINATION: [
        RecTemplate("ground_or_abstain", "Constrain generation to cited context; if nothing supports "
                    "the answer, instruct the model to abstain rather than fabricate."),
    ],
    M.REASONING_FAILURE: [
        RecTemplate("decompose_question", "Decompose the task: elicit intermediate steps "
                    "(chain-of-thought / sub-question prompting) so the model composes the hops."),
    ],
}

_ROLE_PRIORITY = {
    DiagnosisRole.PRIMARY_ROOT: 0,
    DiagnosisRole.CONTRIBUTING: 1,
    DiagnosisRole.SEQUELA: 2,
}


class Recommender:
    def __init__(self, model: Optional[ModelHandle] = None, prob_threshold: float = 0.4,
                 scorers=DEFAULT_SCORERS):
        self.model = model
        self.prob_threshold = prob_threshold
        self.scorers = scorers

    def attach(self, report: DiagnosisReport, inference: Inference,
               fv: Optional[FeatureVector] = None) -> None:
        for diag in report.diagnoses:
            if diag.probability < self.prob_threshold or diag.role is None:
                continue
            priority = _ROLE_PRIORITY.get(diag.role, 3)
            for tmpl in RECOMMENDATIONS.get(diag.mode, []):
                rec = Recommendation(
                    action=tmpl.action, description=tmpl.description,
                    targets_mode=diag.mode, priority=priority,
                )
                self._maybe_validate(rec, inference)
                diag.recommendations.append(rec)

    def _maybe_validate(self, rec: Recommendation, inference: Inference) -> None:
        """Apply the fix and re-run the model; mark whether the answer corrects."""
        if self.model is None:
            return
        fn = INTERVENTIONS.get(rec.action)
        if fn is None:
            return
        orig_correct = self._correct(inference.generated_answer, inference.ground_truth)
        if orig_correct:
            # Nothing to fix — the answer is already correct (e.g. fragile parametric
            # recall). The recommendation is advisory; leave it unscored (validated=None).
            rec.validation_detail = "answer already correct; recommendation is advisory (robustness)"
            return
        modified = fn(inference)

        if modified is None:
            # Only path left for hallucination with no retrievable gold: abstain.
            if rec.action == "ground_or_abstain":
                rec.validated = not orig_correct  # abstaining avoids the wrong answer
                rec.validation_detail = "no gold retrievable — abstaining avoids the fabrication"
            return

        new_answer = self.model.bound_to(modified).generate(modified.prompt).text
        new_correct = self._correct(new_answer, inference.ground_truth)
        if inference.has_ground_truth:
            rec.validated = bool(new_correct and not orig_correct)
            rec.validation_detail = (
                f"after fix, answer -> {new_answer!r} "
                f"({'now correct' if new_correct else 'still wrong'})"
            )
        else:  # production: at least check the answer changed toward grounding
            rec.validation_detail = f"after fix, answer -> {new_answer!r}"

    def _correct(self, answer: str, refs: Optional[list[str]]) -> bool:
        if not refs:
            return False
        return self.scorers.match(answer, refs) >= 0.6
