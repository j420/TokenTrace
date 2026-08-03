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

import hashlib
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


def _u(*parts: object) -> float:
    """Deterministic float in [0,1) from arbitrary parts.

    Used instead of an RNG so a corpus is byte-reproducible across runs while still
    varying per (recipe, fact, seed).
    """
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def _pick(lo: int, hi: int, *key: object) -> int:
    """Deterministic integer in [lo, hi]."""
    return lo + int(_u(*key) * (hi - lo + 1))


def _filler_chunks(n: int, avoid: str, start: int = 0, sents: int = 2) -> list[Chunk]:
    """n topical hard-negative chunks (deterministic), none bearing the answer.

    ``sents`` controls how many pool sentences are concatenated per chunk, which is
    what lets callers vary CONTEXT LENGTH independently of CHUNK COUNT — see the
    anti-shortcut note on :class:`InjectionHarness`. Texts are de-duplicated so a
    chunk list never contains the same string under two source ids.
    """
    avoid_toks = set(norm(avoid))
    picks: list[Chunk] = []
    seen: set[str] = set()
    i = start
    guard = 0
    limit = len(_POOL) * 6
    while len(picks) < n and guard < limit:
        parts = [_POOL[(i + 5 * j) % len(_POOL)] for j in range(max(1, sents))]
        text = " ".join(parts)
        if text not in seen and not (avoid_toks & set(norm(text))):
            seen.add(text)
            picks.append(Chunk(text=text, retriever_score=0.4, source_id=f"neg{i}", gold=False))
        i += 1
        guard += 1
    return picks


def _place(gold: Chunk, fillers: list[Chunk], pos_frac: float) -> list[Chunk]:
    """Insert ``gold`` into ``fillers`` at the given fractional position."""
    idx = int(round(pos_frac * len(fillers)))
    idx = max(0, min(len(fillers), idx))
    return [*fillers[:idx], gold, *fillers[idx:]]


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
        # ~1/4 of clean rows are NON-RAG (correct parametric recall, no context) so
        # that n_chunks == 0 does not uniquely identify the hallucination class.
        non_rag = fact.parametric and _u("clean_nonrag", fact.question, seed) < 0.25
        if non_rag:
            inf = Inference(
                prompt=fact.question, generated_answer="", retrieved_context=None,
                ground_truth=[fact.answer], question=fact.question,
                meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                               "parametric_known": True}},
            )
        else:
            gold = Chunk(fact.gold_fact, 0.95, "gold", gold=True)
            n = _pick(2, 11, "clean_n", fact.question, seed)
            # short fillers keep the total under the dilution threshold, so a clean
            # row can carry MANY chunks and still be answered correctly.
            fillers = _filler_chunks(n, fact.answer, seed, sents=1)
            pos = _u("clean_pos", fact.question, seed)
            chunks = _place(gold, fillers, pos)
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
        # gold removed, hard negatives only; count and length vary per row.
        n = _pick(2, 11, "ret_n", fact.question, seed, parametric_recovery)
        sents = _pick(1, 3, "ret_len", fact.question, seed)
        chunks = _filler_chunks(n, fact.answer, seed, sents=sents)
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

    def context_dilution(self, fact: Fact, seed: int = 0,
                         n_distractors: Optional[int] = None) -> Optional[LabeledInference]:
        # Dilution IS causally geometric (lost-in-the-middle), so it legitimately
        # correlates with context shape — but the count still varies per row, and it
        # overlaps the other recipes' range, so shape alone cannot separate the rest.
        if n_distractors is None:
            n_distractors = _pick(6, 13, "dil_n", fact.question, seed)
        # Scale filler LENGTH to the chunk count so the context reliably clears the
        # lost-in-the-middle token threshold at every count. Previously a fixed
        # length left the low end of the range ~134 tokens — just under the 150-token
        # trigger — so 10/64 dilution rows were silently discarded by their own gate.
        # Scaling length (not raising the count floor) keeps n_chunks overlapping the
        # other recipes, which is what prevents a shape shortcut.
        sents = max(3, -(-240 // (8 * max(1, n_distractors))))
        fillers = _filler_chunks(n_distractors, fact.answer, seed, sents=sents)
        gold = Chunk(fact.gold_fact, 0.6, "gold", gold=True)
        pos = 0.3 + 0.4 * _u("dil_pos", fact.question, seed)   # buried in the middle third
        chunks = _place(gold, fillers, pos)
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
        sim = {"answer": fact.answer, "gold_fact": fact.gold_fact, "ambiguous": True,
               "readings": [fact.answer, fact.wrong_answer, "another reading"]}
        # ~30% non-RAG: an ambiguous question is ambiguous with or without context.
        if _u("amb_nonrag", fact.question, seed) < 0.30:
            inf = Inference(
                prompt=fact.ambiguous_question, generated_answer="", retrieved_context=None,
                ground_truth=[fact.answer], question=fact.ambiguous_question, meta={"_sim": sim},
            )
        else:
            gold = Chunk(fact.gold_fact, 0.9, "gold", gold=True)
            n = _pick(2, 11, "amb_n", fact.question, seed)
            fillers = _filler_chunks(n, fact.answer, seed, sents=1)
            chunks = _place(gold, fillers, _u("amb_pos", fact.question, seed))
            inf = Inference(
                prompt=f"{fact.ambiguous_question}\n" + " ".join(c.text for c in chunks),
                generated_answer="", retrieved_context=chunks, ground_truth=[fact.answer],
                question=fact.ambiguous_question, meta={"_sim": sim},
            )

        def gate(fv):
            amb, ic = _v(fv, "prompt_ambiguity", 0.0), _v(fv, "is_correct", 1.0)
            return (amb >= 0.5 and ic == 0.0), {"prompt_ambiguity": amb, "is_correct": ic}

        return self._finalize(inf, [M.PROMPT_AMBIGUITY], [], "prompt_ambiguity", gate)

    def hallucination(self, fact: Fact, seed: int = 0) -> Optional[LabeledInference]:
        # Non-RAG parametric fabrication -> isolates hallucination (retrieval masked).
        # The question is paraphrased per seed: previously this recipe ignored `seed`
        # entirely, so every seed produced a byte-identical row and the dedup pass
        # silently deleted 2/3 of the hallucination class.
        frames = ["{q}", "Answer concisely: {q}", "{q} Be specific.", "Question: {q}"]
        question = frames[_pick(0, len(frames) - 1, "hal_frame", fact.question, seed)].format(
            q=fact.question)
        inf = Inference(
            prompt=question, generated_answer="", retrieved_context=None,
            ground_truth=[fact.answer], question=question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "distractor": fact.wrong_answer, "parametric_known": False}},
        )

        def gate(fv):
            ic, se = _v(fv, "is_correct", 1.0), _v(fv, "semantic_entropy", 0.0)
            return (ic == 0.0 and se > 0.15), {"is_correct": ic, "semantic_entropy": se}

        return self._finalize(inf, [M.HALLUCINATION], [], "hallucination", gate)

    def hallucination_override(self, fact: Fact, seed: int = 0) -> Optional[LabeledInference]:
        """RAG hallucination: the gold IS retrieved and attended, but a wrong
        parametric belief overrides it.

        This is the hard case. Behaviourally it is indistinguishable from context
        dilution (gold present, answer wrong); only mechanistic evidence separates
        them, so this recipe is what makes the mechanistic family load-bearing. It
        also removes the "no context => hallucination" shortcut, since hallucination
        now occurs with and without retrieval.
        """
        gold = Chunk(fact.gold_fact, 0.95, "gold", gold=True)
        n = _pick(2, 10, "ovr_n", fact.question, seed)
        fillers = _filler_chunks(n, fact.answer, seed, sents=1)
        # gold at an EDGE position and a short context, so the mock's dilution
        # trigger cannot fire — the wrong answer must come from the override alone.
        chunks = _place(gold, fillers, 0.0 if _u("ovr_pos", fact.question, seed) < 0.5 else 1.0)
        inf = Inference(
            prompt=f"{fact.question}\n" + " ".join(c.text for c in chunks),
            generated_answer="", retrieved_context=chunks, ground_truth=[fact.answer],
            question=fact.question,
            meta={"_sim": {"answer": fact.answer, "gold_fact": fact.gold_fact,
                           "distractor": fact.wrong_answer, "parametric_override": True}},
        )

        def gate(fv):
            gr, ic = _v(fv, "gold_recall_in_context", 0.0), _v(fv, "is_correct", 1.0)
            par = _v(fv, "parametric_knowledge_score", 0.0)
            # gold really is recoverable, the answer is still wrong, and (when the
            # tier exposes it) the parametric score is high.
            return (gr >= 0.6 and ic == 0.0), {"gold_recall": gr, "is_correct": ic,
                                               "parametric": par}

        return self._finalize(inf, [M.HALLUCINATION], [], "hallucination_override", gate)

    def reasoning_failure(self, mh: MultiHopFact, seed: int = 0) -> Optional[LabeledInference]:
        gold = Chunk(" ".join(mh.facts), 0.9, "gold", gold=True)
        n = _pick(2, 11, "reason_n", mh.question, seed)
        fillers = _filler_chunks(n, mh.answer, seed, sents=1)
        chunks = _place(gold, fillers, _u("reason_pos", mh.question, seed))
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
