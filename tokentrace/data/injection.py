"""Deterministic failure-injection harness with verification gates.

Each recipe transforms a clean example into a corrupted one carrying a known
failure-mode label — but only if a **verification gate** confirms the corruption
produced the intended effect (otherwise the example is discarded, so we never
mislabel). Injection naturally yields multi-label + causal-edge supervision that
public single-label datasets do not provide.

Provenance is always SYNTHETIC here; real RAGTruth labels and the human-audited
seed come in as REAL via :mod:`tokentrace.data.loaders` and are reserved for
headline evaluation.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import (
    Chunk,
    FailureMode,
    Inference,
    LabeledInference,
    Provenance,
)
from tokentrace.data.samples import Fact, MultiHopFact, distractor_pool
from tokentrace.models.base import ModelHandle
from tokentrace.signals.registry import SignalPipeline
from tokentrace.signals.scorers import DEFAULT_SCORERS, norm

M = FailureMode
_POOL = distractor_pool()


def _filler_chunks(n: int, avoid: str, start: int = 0) -> list[Chunk]:
    """n topical hard-negative chunks (deterministic), none bearing the answer."""
    avoid_toks = set(norm(avoid))
    picks: list[Chunk] = []
    i = start
    guard = 0
    while len(picks) < n and guard < len(_POOL) * 3:
        a = _POOL[i % len(_POOL)]
        b = _POOL[(i + 3) % len(_POOL)]
        text = f"{a} {b}"
        if not (avoid_toks & set(norm(text))):  # ensure it does not leak the answer
            picks.append(Chunk(text=text, retriever_score=0.4, source_id=f"neg{i}", gold=False))
        i += 1
        guard += 1
    return picks


def _v(fv, name: str, default: float) -> float:
    """None-safe feature read (distinguishes a real 0.0 from a missing value)."""
    x = fv.get(name)
    return default if x is None else x


class InjectionHarness:
    def __init__(self, model: ModelHandle, pipeline: Optional[SignalPipeline] = None,
                 scorers=DEFAULT_SCORERS):
        self.model = model
        self.pipeline = pipeline or SignalPipeline()
        self.scorers = scorers

    # ------------------------------------------------------------------ #
    def _finalize(self, inf: Inference, labels: list[FailureMode],
                  edges: list[tuple[FailureMode, FailureMode]], recipe: str,
                  gate) -> Optional[LabeledInference]:
        inf.generated_answer = self.model.bound_to(inf).generate(inf.prompt).text
        fv = self.pipeline.run(inf, self.model)
        passed, checks = gate(fv)
        if not passed:
            return None
        return LabeledInference(
            inference=inf, labels=labels, causal_edges=edges,
            provenance=Provenance.SYNTHETIC, injection_recipe=recipe,
            verification={"gate_passed": True, **checks},
        )

    # ------------------------------------------------------------------ #
    # Recipes
    # ------------------------------------------------------------------ #
    def clean(self, fact: Fact, seed: int = 0) -> Optional[LabeledInference]:
        gold = Chunk(fact.gold_fact, 0.95, "gold", gold=True)
        chunks = [gold, *_filler_chunks(2, fact.answer, seed)]
        inf = Inference(
            prompt=f"{fact.question}\n" + " ".join(c.text for c in chunks),
            generated_answer="", retrieved_context=chunks, ground_truth=[fact.answer],
            question=fact.question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "parametric": fact.parametric}},
        )

        def gate(fv):
            ic = _v(fv, "is_correct", 0.0)
            return ic == 1.0, {"is_correct": ic}

        return self._finalize(inf, [], [], "clean", gate)

    def retrieval_failure(self, fact: Fact, seed: int = 0,
                          parametric_recovery: bool = False) -> Optional[LabeledInference]:
        chunks = _filler_chunks(3, fact.answer, seed)  # gold removed, hard negatives only
        inf = Inference(
            prompt=f"{fact.question}\n" + " ".join(c.text for c in chunks),
            generated_answer="", retrieved_context=chunks, ground_truth=[fact.answer],
            question=fact.question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "distractor": fact.wrong_answer,
                           "parametric_known": parametric_recovery}},
        )
        if parametric_recovery:
            # right-for-wrong-reasons: retrieval failed but the model recovered.
            def gate(fv):
                gr, ic = _v(fv, "gold_recall_in_context", 1.0), _v(fv, "is_correct", 0.0)
                return (gr < 0.3 and ic == 1.0), {"gold_recall": gr, "is_correct": ic}

            return self._finalize(inf, [M.RETRIEVAL_FAILURE], [], "retrieval_failure+parametric", gate)

        def gate(fv):
            gr, ic = _v(fv, "gold_recall_in_context", 1.0), _v(fv, "is_correct", 1.0)
            return (gr < 0.3 and ic == 0.0), {"gold_recall": gr, "is_correct": ic}

        return self._finalize(inf, [M.RETRIEVAL_FAILURE, M.HALLUCINATION],
                              [(M.RETRIEVAL_FAILURE, M.HALLUCINATION)], "retrieval_failure", gate)

    def context_dilution(self, fact: Fact, seed: int = 0, n_distractors: int = 10) -> Optional[LabeledInference]:
        fillers = _filler_chunks(n_distractors, fact.answer, seed)
        gold = Chunk(fact.gold_fact, 0.6, "gold", gold=True)
        mid = len(fillers) // 2
        chunks = fillers[:mid] + [gold] + fillers[mid:]           # gold buried in the middle
        inf = Inference(
            prompt=f"{fact.question}\n" + " ".join(c.text for c in chunks),
            generated_answer="", retrieved_context=chunks, ground_truth=[fact.answer],
            question=fact.question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "distractor": fact.wrong_answer}},
        )

        def gate(fv):
            gr, ic = _v(fv, "gold_recall_in_context", 0.0), _v(fv, "is_correct", 1.0)
            return (gr >= 0.6 and ic == 0.0), {"gold_recall": gr, "is_correct": ic}

        return self._finalize(inf, [M.CONTEXT_DILUTION], [], "context_dilution", gate)

    def prompt_ambiguity(self, fact: Fact, seed: int = 0) -> Optional[LabeledInference]:
        gold = Chunk(fact.gold_fact, 0.9, "gold", gold=True)
        chunks = [gold, *_filler_chunks(2, fact.answer, seed)]
        inf = Inference(
            prompt=f"{fact.ambiguous_question}\n" + " ".join(c.text for c in chunks),
            generated_answer="", retrieved_context=chunks, ground_truth=[fact.answer],
            question=fact.ambiguous_question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "ambiguous": True,
                           "readings": [fact.answer, fact.wrong_answer, "another reading"]}},
        )

        def gate(fv):
            amb, ic = _v(fv, "prompt_ambiguity", 0.0), _v(fv, "is_correct", 1.0)
            return (amb >= 0.5 and ic == 0.0), {"prompt_ambiguity": amb, "is_correct": ic}

        return self._finalize(inf, [M.PROMPT_AMBIGUITY], [], "prompt_ambiguity", gate)

    def hallucination(self, fact: Fact, seed: int = 0) -> Optional[LabeledInference]:
        # Non-RAG parametric fabrication -> isolates hallucination (retrieval masked).
        inf = Inference(
            prompt=fact.question, generated_answer="", retrieved_context=None,
            ground_truth=[fact.answer], question=fact.question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "distractor": fact.wrong_answer, "parametric_known": False}},
        )

        def gate(fv):
            ic, se = _v(fv, "is_correct", 1.0), _v(fv, "semantic_entropy", 0.0)
            return (ic == 0.0 and se > 0.15), {"is_correct": ic, "semantic_entropy": se}

        return self._finalize(inf, [M.HALLUCINATION], [], "hallucination", gate)

    def reasoning_failure(self, mh: MultiHopFact, seed: int = 0) -> Optional[LabeledInference]:
        gold = Chunk(" ".join(mh.facts), 0.9, "gold", gold=True)
        chunks = [gold, *_filler_chunks(2, mh.answer, seed)]
        inf = Inference(
            prompt=f"{mh.question}\n" + " ".join(c.text for c in chunks),
            generated_answer="", retrieved_context=chunks, ground_truth=[mh.answer],
            question=mh.question,
            meta={"_sim": {"answer": mh.answer, "gold_fact": " ".join(mh.facts),
                           "requires_multihop": True, "hard_composition": True,
                           "distractor": mh.wrong_answer}},
        )

        def gate(fv):
            mhq = _v(fv, "multihop_query", 0.0)
            gr, ic = _v(fv, "gold_recall_in_context", 0.0), _v(fv, "is_correct", 1.0)
            return (mhq == 1.0 and gr >= 0.6 and ic == 0.0), \
                {"multihop": mhq, "gold_recall": gr, "is_correct": ic}

        return self._finalize(inf, [M.REASONING_FAILURE], [], "reasoning_failure", gate)
