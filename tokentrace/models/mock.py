"""Deterministic mock model — a coherent stand-in for a real LLM.

The mock is NOT random noise. It runs a small, internally-consistent "world
model": it answers correctly iff the query is unambiguous AND the answer is
either derivable from the visible context or known parametrically AND (for
multi-hop) the composition succeeds. Its confidence and mechanistic outputs are
then derived from *that same decision and the text geometry*, exactly as a real
model's internals would reflect them. This makes the full pipeline
(inject -> extract signals -> diagnose -> validate recommendation) exercisable and
internally consistent with zero downloads.

Real backends (GGUF, HF hooks) implement the same :class:`ModelHandle` interface
and ignore the simulator hints in ``inference.meta['_sim']``.

Simulator hints (``inference.meta['_sim']``), all optional:
    answer: str              correct answer
    gold_fact: str           text that grounds the answer if visible
    parametric_known: bool   model knows the answer without context
    requires_multihop: bool  answer needs composing >1 fact
    hard_composition: bool   composition fails even with facts present (reasoning)
    ambiguous: bool          query underspecified
    readings: list[str]      valid answers under different readings (ambiguity)
    distractor: str          the wrong answer emitted on failure
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from tokentrace.core.types import Inference, ModelProfile, Tier
from tokentrace.models.base import CaptureResult, GenerationResult, ModelHandle

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return _norm(needle) in _norm(haystack)


def _seeded_unit(*parts: object) -> float:
    """Deterministic float in [0,1) from arbitrary parts (stand-in for sampling
    noise; we avoid Math.random-style nondeterminism so runs are reproducible)."""
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


@dataclass
class _Decision:
    answer: str
    correct: bool
    grounded: bool           # derived answer from visible context
    parametric: bool         # pulled answer from parametric memory
    ambiguous: bool
    reasoning_failed: bool
    gold_present: bool        # answer IS derivable from retrieved context
    gold_pos_frac: float      # position of gold chunk in [0,1] (0.5 = middle)
    n_chunks: int
    context_tokens: int


class MockModel(ModelHandle):
    """A deterministic, tier-aware simulated LLM."""

    def __init__(self, profile: Optional[ModelProfile] = None, tier: Tier = Tier.WHITE,
                 noise: float = 0.0):
        self.profile = profile or ModelProfile(
            name="mock-4b",
            n_layers=32,
            n_heads=32,
            d_model=2560,
            max_tier=Tier.WHITE,
            lost_in_middle_baseline=[0.9, 0.75, 0.6, 0.72, 0.88],  # U-shaped
        )
        self.tier = tier
        # Deterministic signal noise in [0,~0.5]: jitters the mechanistic/confidence
        # estimates so the failure modes are NOT perfectly separable — makes the
        # benchmark exercise calibration/conformal/abstention like real, noisy data.
        self.noise = noise

    # ------------------------------------------------------------------ #
    # Core decision
    # ------------------------------------------------------------------ #
    def _decide(self, inference: Inference) -> _Decision:
        sim = inference.meta.get("_sim", {})
        answer = sim.get("answer", inference.ground_truth[0] if inference.ground_truth else "")
        distractor = sim.get("distractor", "an unrelated entity")

        chunks = inference.retrieved_context or []
        context_text = " ".join(c.text for c in chunks)
        visible = inference.prompt + " " + context_text
        context_tokens = len(_norm(context_text).split())

        # Is the answer derivable from retrieved context?
        gold_flags = [c for c in chunks if c.gold]
        gold_fact = sim.get("gold_fact", answer)
        gold_present = bool(gold_flags) or (_contains(visible, gold_fact) if gold_fact else False)

        # Gold position (for dilution): fraction through the chunk list.
        gold_pos_frac = 0.0
        if gold_flags and len(chunks) > 1:
            idx = chunks.index(gold_flags[0])
            gold_pos_frac = idx / (len(chunks) - 1)

        ambiguous = bool(sim.get("ambiguous")) or bool(sim.get("readings"))
        requires_multihop = bool(sim.get("requires_multihop"))
        hard_composition = bool(sim.get("hard_composition"))
        parametric_known = bool(sim.get("parametric_known"))

        # --- the world model --- #
        if ambiguous:
            readings = sim.get("readings") or [answer, distractor]
            # Pick a valid reading that is *not* the assumed ground truth, to model
            # the "answered a different valid interpretation" mismatch.
            chosen = next((r for r in readings if not _contains(answer, r)), readings[0])
            correct = bool(inference.ground_truth and _contains(chosen, inference.ground_truth[0]))
            return _Decision(chosen, correct, False, False, True, False,
                             gold_present, gold_pos_frac, len(chunks), context_tokens)

        # Dilution: gold present but buried among many distractors ("lost in the
        # middle"). Driven by position among many chunks, not raw token count.
        diluted = (
            gold_present
            and len(chunks) >= 5
            and 0.2 <= gold_pos_frac <= 0.8
            and context_tokens > 150
        )

        if requires_multihop and hard_composition and gold_present:
            # Facts present, sub-hops answerable, composition fails.
            return _Decision(distractor, False, False, False, False, True,
                             gold_present, gold_pos_frac, len(chunks), context_tokens)

        if gold_present and not diluted:
            return _Decision(answer, True, True, False, False, False,
                             gold_present, gold_pos_frac, len(chunks), context_tokens)

        if parametric_known:
            # Right answer without (usable) context: parametric recall. In RAG with
            # retrieval failure this is "right for the wrong reasons".
            return _Decision(answer, True, False, True, False, False,
                             gold_present, gold_pos_frac, len(chunks), context_tokens)

        # No usable evidence, not known -> confident fabrication (hallucination),
        # or a miss under dilution.
        return _Decision(distractor, False, False, False, False, False,
                         gold_present, gold_pos_frac, len(chunks), context_tokens)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _entropy_profile(self, dec: _Decision) -> float:
        if dec.ambiguous:
            return 1.2
        if dec.reasoning_failed:
            return 0.85
        if dec.grounded:
            return 0.22
        if dec.parametric:
            return 0.30            # low = confident (the dangerous parametric override)
        return 1.45                # unsupported fabrication / miss: high uncertainty

    def generate(
        self, prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        # Decide on the bound full inference when available (keeps generate()
        # consistent with capture() for structure-dependent modes like dilution).
        # Simulated interventions rebind, so the bound inference always reflects
        # the current prompt/context.
        bound = getattr(self, "_ctx_inference", None)
        if bound is not None:
            inf = bound
        else:
            inf = Inference(prompt=prompt, generated_answer="", meta=getattr(self, "_ctx_meta", {}))
        dec = self._decide(inf)
        text = dec.answer
        # Temperature spreads uncertain answers across samples (semantic entropy).
        if temperature > 0 and not dec.grounded and not dec.parametric:
            readings = inf.meta.get("_sim", {}).get("readings")
            pool = readings if (dec.ambiguous and readings) else [dec.answer, "a different guess",
                                                                   "yet another guess"]
            text = pool[int(_seeded_unit(prompt, seed) * len(pool)) % len(pool)]

        base_H = self._entropy_profile(dec)
        if self.noise > 0:  # blur the confidence estimate
            base_H = max(0.02, base_H * (1 + self.noise * (2 * _seeded_unit("noiseH", prompt, seed) - 1)))
        toks = text.split() or [text]
        entropies, logprobs, ttexts = [], [], []
        for i, t in enumerate(toks):
            h = max(0.02, base_H * (0.7 + 0.6 * _seeded_unit(prompt, seed, i)))
            entropies.append(round(h, 4))
            logprobs.append(round(-h, 4))     # crude: chosen-token logprob ~ -entropy
            ttexts.append(t)
        return GenerationResult(text=text, token_texts=ttexts,
                                token_logprobs=logprobs, token_entropies=entropies)

    def sample(self, prompt: str, k: int = 10, temperature: float = 1.0) -> list[str]:
        return [self.generate(prompt, temperature=temperature, seed=i).text for i in range(k)]

    def confidence(self, prompt: str, answer: Optional[str] = None) -> GenerationResult:
        self.require(Tier.GREY)
        return self.generate(prompt, temperature=0.0)

    # ------------------------------------------------------------------ #
    # Mechanistic capture (grey/white-box)
    # ------------------------------------------------------------------ #
    def capture(self, inference: Inference) -> CaptureResult:
        self.require(Tier.GREY)
        dec = self._decide(inference)
        n_layers = self.profile.n_layers

        if dec.grounded:
            ctx_attn, gold_attn = 0.62, 0.55
            ext, param = 0.70, 0.35
            answer_layer, stability = 0.70, 0.9    # forms late, after reading context
        elif dec.parametric:
            ctx_attn, gold_attn = 0.18, 0.10
            ext, param = 0.20, 0.82                # low context read, high parametric push
            answer_layer, stability = 0.30, 0.85   # forms early from memory
        elif dec.ambiguous:
            # Confident commitment to an alternate reading: the model DOES attend to
            # the context and forms an answer early and stably (not buried/unattended
            # like dilution, not late/unstable like reasoning). The tell is high
            # *semantic* entropy across resamples (in generate), not a mechanistic
            # dilution signature.
            ctx_attn, gold_attn = 0.50, 0.40
            ext, param = 0.55, 0.40
            answer_layer, stability = 0.40, 0.85
        elif dec.reasoning_failed:
            ctx_attn, gold_attn = 0.5, 0.45
            ext, param = 0.55, 0.5
            answer_layer, stability = 0.9, 0.35    # forms very late, unstable (flip-flops)
        elif dec.gold_present:  # diluted: gold there but not attended
            ctx_attn, gold_attn = 0.35, 0.12       # low attention to the (buried) gold
            ext, param = 0.4, 0.55
            answer_layer, stability = 0.8, 0.5
        else:  # hallucination / miss
            ctx_attn, gold_attn = 0.15, 0.05
            ext, param = 0.15, 0.85
            answer_layer, stability = 0.35, 0.8

        # Dilution correction: long context with buried gold suppresses gold attention
        # relative to the model's measured lost-in-the-middle baseline.
        if dec.gold_present and dec.n_chunks >= 5:
            mid_penalty = 1.0 - 0.5 * (1.0 - abs(dec.gold_pos_frac - 0.5) * 2)  # min at middle
            gold_attn *= mid_penalty

        # White-box only: causal effect of the gold tokens on the answer.
        gold_patch = None
        if self.supports(Tier.WHITE):
            if dec.grounded:
                gold_patch = 0.8
            elif dec.ambiguous:
                gold_patch = 0.3          # reads context, but the ambiguity is in the query
            elif dec.gold_present:
                gold_patch = 0.05
            else:
                gold_patch = 0.02

        if self.noise > 0:
            p = inference.prompt

            def jit(x: Optional[float], key: str) -> Optional[float]:
                if x is None:
                    return None
                delta = self.noise * (2 * _seeded_unit("noise", key, p) - 1) * 0.4
                return min(1.0, max(0.0, x + delta))

            ctx_attn, gold_attn = jit(ctx_attn, "ctx"), jit(gold_attn, "gold")
            ext, param = jit(ext, "ext"), jit(param, "param")
            answer_layer, stability = jit(answer_layer, "al"), jit(stability, "st")
            gold_patch = jit(gold_patch, "gp")

        return CaptureResult(
            n_layers=n_layers,
            context_attention_ratio=round(ctx_attn, 3),
            gold_attention_ratio=round(gold_attn, 3),
            logit_lens_answer_layer=round(answer_layer, 3),
            logit_lens_stability=round(stability, 3),
            external_context_score=round(ext, 3),
            parametric_knowledge_score=round(param, 3),
            gold_patch_effect=gold_patch,
        )

    # ------------------------------------------------------------------ #
    # Convenience: bind an inference's meta so generate(prompt) can see hints.
    # ------------------------------------------------------------------ #
    def bound_to(self, inference: Inference) -> "MockModel":
        """Return a shallow copy whose generate()/sample() see this inference's
        full structure and sim hints, so simulated interventions (re-generate with
        modified context) behave consistently. Used by the recommender and
        edge-validator."""
        import copy

        m = copy.copy(self)
        m._ctx_inference = inference
        m._ctx_meta = inference.meta
        return m
