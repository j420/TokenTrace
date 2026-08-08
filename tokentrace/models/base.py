"""Model abstraction.

A :class:`ModelHandle` is the single interface every signal extractor talks to.
Concrete backends:

* :class:`~tokentrace.models.mock.MockModel` — deterministic, no downloads, used for
  tests and the offline demo. Implements every tier.
* GGUF (llama.cpp) — quantized CPU generation + resampling (grey-box logprobs).
* HF + hooks / NNsight — grey/white-box mechanistic capture.

The interface is deliberately capability-gated: a handle advertises a ``tier`` and
the methods above ``tier`` raise :class:`TierUnavailable`. This is what makes the
compute-adaptive degradation (and "this model's capture backend cannot reach
white-box, cap it at grey" — ``ModelProfile.max_tier``) a property of one object
rather than branches everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from tokentrace.core.types import Inference, ModelProfile, Tier


class TierUnavailable(RuntimeError):
    """Raised when a signal needs a higher tier than the handle supports."""


@dataclass
class GenerationResult:
    """Output of a (greedy or sampled) generation.

    Grey-box fields (``token_logprobs``/``token_entropies``) are populated when
    the backend exposes logprobs; they are empty at black-box tier.
    """

    text: str
    token_texts: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)   # logprob of chosen token
    token_entropies: list[float] = field(default_factory=list)  # entropy of next-token dist (nats)

    @property
    def has_logprobs(self) -> bool:
        return len(self.token_logprobs) > 0


@dataclass
class CaptureResult:
    """Mechanistic capture from a single forward pass (grey/white-box).

    All ratio/score fields are intended to be consumed *model-relative* (the
    extractor normalizes them against :class:`ModelProfile` baselines), so they
    are comparable across Gemma 3 / Qwen3 / Phi-4.

    ReDeEP anchors: ``external_context_score`` = how strongly attention heads read
    the retrieved context; ``parametric_knowledge_score`` = how strongly FFNs push
    the answer from parametric memory. Low external + high parametric on an
    unsupported answer is the hallucination signature.
    """

    n_layers: int = 0
    context_attention_ratio: Optional[float] = None   # attn mass to context / total
    gold_attention_ratio: Optional[float] = None      # attn mass to gold-chunk positions
    logit_lens_answer_layer: Optional[float] = None   # depth answer forms, as fraction [0,1]
    logit_lens_stability: Optional[float] = None      # 1 = stable once formed, 0 = flip-floppy
    external_context_score: Optional[float] = None     # ReDeEP external (context) score
    parametric_knowledge_score: Optional[float] = None  # ReDeEP parametric score
    # White-box only: causal effect of ablating/patching the gold-chunk tokens on
    # the answer logit. Large positive => answer causally depends on the context.
    gold_patch_effect: Optional[float] = None


class ModelHandle(ABC):
    """Backend-agnostic handle over one model at some observability tier."""

    profile: ModelProfile
    tier: Tier

    # --- generation (all tiers) --- #
    @abstractmethod
    def generate(
        self, prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        ...

    def sample(self, prompt: str, k: int = 10, temperature: float = 1.0) -> list[str]:
        """K stochastic completions for semantic-entropy estimation."""
        return [self.generate(prompt, temperature=temperature, seed=i).text for i in range(k)]

    # --- grey-box --- #
    def confidence(self, prompt: str, answer: Optional[str] = None) -> GenerationResult:
        """Token-level logprobs/entropies for the answer. Grey-box+.

        Default routes through :meth:`generate`; backends that can score a *given*
        answer should override.
        """
        self.require(Tier.GREY)
        return self.generate(prompt, temperature=0.0)

    # --- grey/white-box mechanistic --- #
    def capture(self, inference: Inference) -> CaptureResult:  # noqa: ARG002
        """Single-pass mechanistic capture. Grey-box gives attention + logit-lens;
        white-box adds ``gold_patch_effect`` (causal)."""
        self.require(Tier.GREY)
        raise TierUnavailable("capture() not implemented for this backend")

    # --- helpers --- #
    def supports(self, tier: Tier) -> bool:
        return self.tier >= tier

    def require(self, tier: Tier) -> None:
        if not self.supports(tier):
            raise TierUnavailable(
                f"{self.profile.name}: need {tier.label} tier, handle caps at {self.tier.label}"
            )

    def bound_to(self, inference: Inference) -> "ModelHandle":  # noqa: ARG002
        """Return a handle whose generation is aware of this inference's full
        structure. No-op for real backends (they read everything from the prompt
        string); the mock uses it to stay consistent between generate() and
        capture(). Simulated interventions rebind after modifying the inference."""
        return self

    def with_tier(self, tier: Tier) -> "ModelHandle":
        """Return a view of this handle down-capped to ``tier`` (for tier
        experiments / graceful-degradation tests). Never up-caps past the
        backend's real capability or the profile's ``max_tier``."""
        import copy

        capped = min(tier, self.profile.max_tier)
        view = copy.copy(self)
        view.tier = Tier(min(int(self.tier), int(capped)))
        return view
