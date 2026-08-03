"""Quantized CPU generation backend (llama.cpp / GGUF).

This is the *fast* generation path on CPU: a 4B model at Q4_K_M is ~2.5 GB RAM and
runs at a usable few-to-tens of tokens/sec with no GPU. It provides black-box
generation and grey-box token logprobs/entropies (enough for perplexity, token
entropy, and K-sample semantic entropy). It does NOT expose activations — use the
HF backend for mechanistic capture.

Lazy import: ``llama_cpp`` is only imported when this class is instantiated, so
the dependency-light core never needs it.
"""

from __future__ import annotations

import math
from typing import Optional

from tokentrace.core.types import ModelProfile, Tier
from tokentrace.models.base import GenerationResult, ModelHandle


class GGUFModel(ModelHandle):
    def __init__(
        self,
        profile: ModelProfile,
        tier: Tier = Tier.GREY,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
        logprobs_top_k: int = 20,
        **llama_kwargs,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GGUF backend needs llama-cpp-python. Install: pip install 'tokentrace[generate]'"
            ) from e

        self.profile = profile
        self.tier = min(tier, Tier.GREY)  # llama.cpp gives logprobs, not activations
        self.logprobs_top_k = logprobs_top_k

        if model_path is None:
            model_path = self._download(profile)
        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
            **llama_kwargs,
        )

    @staticmethod
    def _download(profile: ModelProfile) -> str:  # pragma: no cover - network
        if not profile.gguf_repo:
            raise ValueError(f"no gguf_repo registered for {profile.name}; pass model_path=")
        from huggingface_hub import hf_hub_download

        # Heuristic: pick a Q4_K_M file; callers can pass model_path to be explicit.
        from huggingface_hub import list_repo_files

        files = [f for f in list_repo_files(profile.gguf_repo) if f.lower().endswith(".gguf")]
        pick = next((f for f in files if "q4_k_m" in f.lower()), files[0])
        return hf_hub_download(profile.gguf_repo, pick)

    def generate(
        self, prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        out = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            logprobs=self.logprobs_top_k,
            echo=False,
        )
        choice = out["choices"][0]
        text = choice["text"]
        lp = choice.get("logprobs") or {}
        token_texts = lp.get("tokens", []) or []
        token_logprobs = [x for x in (lp.get("token_logprobs") or []) if x is not None]
        # Entropy of the next-token distribution from the returned top-k logprobs.
        entropies: list[float] = []
        for top in lp.get("top_logprobs", []) or []:
            if not top:
                continue
            ps = [math.exp(v) for v in top.values()]
            z = sum(ps) or 1.0
            ps = [p / z for p in ps]
            entropies.append(-sum(p * math.log(p + 1e-12) for p in ps))
        return GenerationResult(
            text=text,
            token_texts=token_texts,
            token_logprobs=token_logprobs,
            token_entropies=entropies,
        )
