"""Quantized CPU generation backend (llama.cpp / GGUF).

This is the *fast* generation path on CPU: a 4B model at Q4_K_M is ~2.5 GB RAM and
runs at a usable few-to-tens of tokens/sec with no GPU. It provides black-box
generation and grey-box token logprobs/entropies (enough for perplexity, token
entropy, and K-sample semantic entropy). It does NOT expose activations — use the
HF backend for mechanistic capture.

Memory note (``n_ctx`` / ``need_logprobs``)
-------------------------------------------
``logprobs=`` is rejected by llama-cpp-python unless the context was built with
``logits_all=True``, and that flag switches the score buffer from
``(n_batch=512, n_vocab)`` to ``(n_ctx, n_vocab)`` float32 — for a 152k-token
vocabulary that is 311 MB vs 4.98 GB at the old ``n_ctx=8192`` default, i.e. ~2x
the advertised footprint of the *whole* backend, before the transient copies
``logits_to_logprobs`` makes over the same span. So ``n_ctx`` now defaults to 4096
and ``logits_all`` follows ``need_logprobs``. ``need_logprobs`` stays **True** by
default because turning it off removes the entire confidence signal family; when
it is off the handle caps itself at BLACK tier so that absence is visible in the
missingness mask instead of silently yielding empty lists.

Entropy scale: token entropies are reported on the **full-vocabulary** scale (the
same one models/hf.py produces), by adding the truncated tail mass back in — see
:meth:`GGUFModel._token_entropy`. The scale is also recorded on the ModelProfile.

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
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        logprobs_top_k: int = 20,
        need_logprobs: bool = True,
        **llama_kwargs,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GGUF backend needs llama-cpp-python. Install: pip install 'tokentrace[generate]'"
            ) from e

        self.profile = profile
        self.need_logprobs = bool(need_logprobs)
        self.logprobs_top_k = logprobs_top_k
        self.n_ctx = n_ctx
        # llama.cpp gives logprobs, not activations -> never above GREY; and
        # without the logits_all buffer there are no logprobs at all, so the
        # handle must advertise BLACK rather than return empty confidence lists.
        self.tier = min(tier, Tier.GREY if self.need_logprobs else Tier.BLACK)

        if model_path is None:
            model_path = self._download(profile)
        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            # REQUIRED for logprobs=, but it reserves (n_ctx, n_vocab) float32.
            logits_all=self.need_logprobs,
            verbose=False,
            **llama_kwargs,
        )
        self._n_vocab = self._detect_vocab()
        profile.tokenizer_quirks["entropy_scale"] = {
            "estimator": "topk+uniform_tail" if self._n_vocab else "topk_renormalised",
            "top_k": logprobs_top_k,
            "max_nats": round(math.log(self._n_vocab), 4) if self._n_vocab
            else round(math.log(max(2, logprobs_top_k)), 4),
            "vocab_size": self._n_vocab or None,
        }

    @staticmethod
    def _download(profile: ModelProfile) -> str:  # pragma: no cover - network
        if not profile.gguf_repo:
            raise ValueError(f"no gguf_repo registered for {profile.name}; pass model_path=")
        from huggingface_hub import hf_hub_download

        # Heuristic: pick a Q4_K_M file; callers can pass model_path to be explicit.
        from huggingface_hub import list_repo_files

        import re as _re

        files = [f for f in list_repo_files(profile.gguf_repo) if f.lower().endswith(".gguf")]
        if not files:
            raise FileNotFoundError(f"no .gguf files in {profile.gguf_repo}; pass model_path=")
        # Prefer a single-file Q4_K_M; skip multi-part shards (they need merging first).
        singles = [f for f in files if not _re.search(r"-\d{5}-of-\d{5}", f)]
        pool = singles or files
        pick = next((f for f in pool if "q4_k_m" in f.lower()), pool[0])
        return hf_hub_download(profile.gguf_repo, pick)

    def _detect_vocab(self) -> int:
        """Vocabulary size, across llama-cpp-python layouts (0 if unknown)."""
        candidates = (
            lambda: self._llama.n_vocab(),
            lambda: self._llama._n_vocab,
            lambda: self._llama.model.n_vocab(),
            lambda: self._llama._model.n_vocab(),
        )
        for get in candidates:
            try:
                v = int(get())
            except Exception:
                continue
            if v > 1:
                return v
        return 0

    # ------------------------------------------------------------------ #
    def _token_entropy(self, top: dict) -> float:
        """Next-token entropy in nats from a top-k logprob dict.

        The previous implementation renormalised over the top-k alone, which hard-
        capped the value at ln(k)=2.996 nats (hf.py reports full-vocab entropy, max
        ln(151936)=11.93 for Qwen3) and was not even monotone: a maximally
        uncertain step measured 0.676 while a peaked one measured ~0, so the same
        canonical feature had two incompatible scales with nothing in
        signals/features.py or engine/calibration.py to distinguish them.

        Fix: keep the top-k probabilities unnormalised and add the residual tail
        mass back under a maximum-entropy (uniform-over-the-rest) assumption. That
        recovers ln|V| exactly for a uniform distribution, is monotone in the tail
        mass, and is an upper bound otherwise. If the vocabulary size cannot be
        read we fall back to the old renormalised value — and the profile's
        ``entropy_scale`` records which estimator was used.
        """
        ps = [math.exp(v) for v in top.values() if v is not None]
        ps = [p for p in ps if p > 0.0]
        if not ps:
            return 0.0
        mass = sum(ps)
        if mass > 1.0:  # numerical slack in the reported logprobs
            ps = [p / mass for p in ps]
            mass = 1.0
        h = -sum(p * math.log(p) for p in ps)
        n_tail = self._n_vocab - len(ps)
        tail = 1.0 - mass
        if tail > 1e-9 and n_tail > 0:
            h += -tail * math.log(tail / n_tail)
        elif not self._n_vocab:
            # Unknown vocab: renormalise (legacy, ln(k)-capped) rather than guess.
            z = mass or 1.0
            qs = [p / z for p in ps]
            h = -sum(q * math.log(q) for q in qs)
        return h

    def _token_stats(self, logprobs: dict, keep=None):
        """Build the three aligned per-token lists from a llama.cpp logprobs dict.

        Positions without a logprob (e.g. the very first token) are dropped from
        ALL lists so indices stay in sync — consumers zip them positionally.
        ``keep(i)`` optionally restricts to a token range (used to score only the
        answer part of an echoed prompt).
        """
        tokens = logprobs.get("tokens") or []
        tlp = logprobs.get("token_logprobs") or []
        ttop = logprobs.get("top_logprobs") or []
        token_texts: list[str] = []
        token_logprobs: list[float] = []
        entropies: list[float] = []
        for i, tok in enumerate(tokens):
            lpi = tlp[i] if i < len(tlp) else None
            if lpi is None:
                continue
            if keep is not None and not keep(i):
                continue
            token_texts.append(tok)
            token_logprobs.append(float(lpi))
            top = ttop[i] if i < len(ttop) else None
            entropies.append(self._token_entropy(top) if top else 0.0)
        return token_texts, token_logprobs, entropies

    def generate(
        self, prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        out = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            # Only ask for logprobs when the context was built to serve them;
            # llama.cpp raises otherwise (and logits_all=False is the whole point
            # of the small-buffer mode).
            logprobs=self.logprobs_top_k if self.need_logprobs else None,
            echo=False,
        )
        choice = out["choices"][0]
        text = choice["text"]
        lp = choice.get("logprobs") or {}
        token_texts, token_logprobs, entropies = self._token_stats(lp)
        return GenerationResult(
            text=text,
            token_texts=token_texts,
            token_logprobs=token_logprobs,
            token_entropies=entropies,
        )

    def confidence(self, prompt: str, answer: Optional[str] = None) -> GenerationResult:
        """Token logprobs/entropies **of the given answer** (teacher-forced).

        The base implementation ignores ``answer`` and re-generates, so the
        confidence features would describe a fresh 64-token continuation instead
        of the answer being diagnosed. llama.cpp can score an existing string with
        ``echo=True``: we run the prompt+answer through and keep only the tokens
        whose ``text_offset`` falls inside the answer. If the build does not report
        ``text_offset`` (or the scoring call fails, e.g. n_ctx overflow) we fall
        back to the documented default behaviour.
        """
        self.require(Tier.GREY)
        if not answer or not answer.strip() or not self.need_logprobs:
            return self.generate(prompt, temperature=0.0)
        lead = "" if (not prompt or prompt[-1].isspace() or answer[:1].isspace()) else " "
        scored = prompt + lead + answer
        boundary = len(prompt) + len(lead)
        try:
            # max_tokens=1 (not 0): a 0/None max_tokens is interpreted as
            # "fill the context" by llama-cpp-python. The single extra token is
            # discarded by the text_offset filter below.
            out = self._llama(
                scored,
                max_tokens=1,
                temperature=0.0,
                seed=0,
                logprobs=self.logprobs_top_k,
                echo=True,
            )
        except Exception:
            return self.generate(prompt, temperature=0.0)
        lp = (out["choices"][0].get("logprobs") or {})
        offsets = lp.get("text_offset") or []
        if not offsets:
            return self.generate(prompt, temperature=0.0)
        end = len(scored)

        def keep(i: int) -> bool:
            # Keep a token if its char span *ends* after the prompt/answer
            # boundary, not if it starts there: the answer's first token usually
            # carries the preceding space (' The' starts one char early), and a
            # start-only test silently drops it — the same off-by-one that lost
            # the first token of every chunk in hf._context_spans.
            if i >= len(offsets) or offsets[i] >= end:
                return False          # the throwaway generated token
            stop = offsets[i + 1] if i + 1 < len(offsets) else end
            return stop > boundary

        token_texts, token_logprobs, entropies = self._token_stats(lp, keep=keep)
        if not token_logprobs:
            return self.generate(prompt, temperature=0.0)
        return GenerationResult(
            text=answer,
            token_texts=token_texts,
            token_logprobs=token_logprobs,
            token_entropies=entropies,
        )
