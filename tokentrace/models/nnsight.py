"""NNsight mechanistic-capture backend.

NNsight traces arbitrary HuggingFace models and, unlike plain forward hooks, makes
*interventions* first-class — so this backend is the rung of the fallback ladder we
reach for when we want true **activation patching** for ``gold_patch_effect`` (the
causal white-box signal), and it covers models TransformerLens does not yet support.

Fallback ladder (see ModelProfile.max_tier):
    HookedTransformer -> TransformerBridge -> **NNsight** -> raw HF hooks (models/hf.py) -> grey-box

Grey-box capture (attention-to-context/gold, logit-lens, ReDeEP scores) mirrors the
HF backend; white-box adds an activation-patching pass that zeroes the gold-chunk
positions in the layer-0 residual stream and measures the drop in the answer-token
log-prob.

Implementation notes (verify against your pinned nnsight/transformers versions):
* The model is loaded with ``attn_implementation="eager"`` so attention weights are
  materialised (SDPA — the >=4.44 default — returns ``None`` for them).
* Saved proxies are read via ``.value`` after the ``trace`` block, and the lists
  that hold them are created **outside** the ``with`` block (names bound inside a
  trace body are not reliably visible after it).
* **One visit per layer, in forward order.** nnsight >= 0.4 installs its hooks
  lazily and each hook self-removes after firing, so requesting *all* layers'
  ``self_attn.output`` first and only then looping again for ``layer.output``
  makes the second pass wait for a provider that can never fire again ->
  ``Mediator.MissedProviderError``. That is not a ``TierUnavailable``, so it
  escapes ``SignalPipeline.run`` and aborts the entire diagnosis.
* Decoder-layer outputs are normalised through :meth:`NNsightModel._hidden`:
  transformers < 4.54 returns ``(hidden_states, ...)`` while >= 4.54 returns a bare
  tensor, where ``[0]`` would silently consume the *batch* dim (scalar logit-lens
  values, ``IndexError`` in the patching pass).
* Generation is scored: :meth:`generate` and :meth:`confidence` return real token
  logprobs/entropies (full-vocabulary nats, same scale as the HF backend) so the
  confidence signal family is not silently absent at WHITE tier.

Lazy imports; needs ``pip install 'tokentrace[mechanistic]'`` (nnsight + torch +
transformers). Runs on CPU.
"""

from __future__ import annotations

import math
from typing import Optional

from tokentrace.core.types import Inference, ModelProfile, Tier
from tokentrace.models.base import (
    CaptureResult,
    GenerationResult,
    ModelHandle,
    TierUnavailable,
)
from tokentrace.models.hf import HFModel


class NNsightModel(ModelHandle):
    #: Attribute chains tried when locating parts of the LM inside the traced
    #: model. Composite models (e.g. Gemma3ForConditionalGeneration) nest the
    #: decoder under ``model.language_model``.
    _LAYER_PATHS = ("model.layers", "model.language_model.layers",
                    "model.decoder.layers", "transformer.h")
    _NORM_PATHS = ("model.norm", "model.language_model.norm",
                   "model.decoder.final_layer_norm", "transformer.ln_f")
    _HEAD_PATHS = ("lm_head", "language_model.lm_head", "model.lm_head")

    def __init__(
        self,
        profile: ModelProfile,
        tier: Tier = Tier.WHITE,
        hf_repo: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        try:
            from nnsight import LanguageModel
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError(
                "NNsight backend needs the 'mechanistic' extra: pip install 'tokentrace[mechanistic]'"
            ) from e
        repo = hf_repo or profile.hf_repo or profile.name
        # eager attention so self_attn returns weights (SDPA returns None for them).
        self.lm = LanguageModel(repo, device_map=device, dispatch=True,
                                attn_implementation="eager", **kwargs)
        self.tokenizer = self.lm.tokenizer
        self.device = device
        cfg = self.lm.config
        profile.n_layers = getattr(cfg, "num_hidden_layers", profile.n_layers)
        profile.n_heads = getattr(cfg, "num_attention_heads", profile.n_heads)
        profile.d_model = getattr(cfg, "hidden_size", profile.d_model)
        self.profile = profile
        self.tier = Tier(min(int(tier), int(profile.max_tier)))
        # Confidence features are full-vocabulary entropies (see _score_tokens),
        # i.e. the same scale as models/hf.py.
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        if vocab > 1:
            profile.tokenizer_quirks["entropy_scale"] = {
                "estimator": "full_vocab",
                "max_nats": round(math.log(vocab), 4),
                "vocab_size": vocab,
            }

    # ------------------------------------------------------------------ #
    # Envoy/version plumbing
    # ------------------------------------------------------------------ #
    def _resolve(self, paths):
        """First resolvable attribute chain on the traced model, else None."""
        for path in paths:
            obj = self.lm
            for part in path.split("."):
                try:
                    obj = getattr(obj, part)
                except Exception:
                    obj = None
                    break
            if obj is not None:
                return obj
        return None

    @staticmethod
    def _hidden(layer_output):
        """Normalise a decoder-layer output to its hidden-state tensor.

        transformers < 4.54 returns ``(hidden_states, ...)``; >= 4.54 returns a
        bare tensor, and ``[0]`` on that consumes the BATCH dim — measured effect:
        the logit lens reads a scalar (constant garbage token, answer_layer=1.0,
        stability=0.0) and the patching pass raises "too many indices for tensor
        of dimension 2". nnsight >= 0.5 hands us the concrete value inside a
        trace, so isinstance/ndim are meaningful; older nnsight hands us a lazy
        proxy, for which we keep the historical tuple contract.
        """
        if isinstance(layer_output, (tuple, list)):
            return layer_output[0]
        try:
            ndim = layer_output.ndim
        except Exception:
            ndim = None
        if isinstance(ndim, int):
            return layer_output          # concrete tensor: already the hidden states
        return layer_output[0]           # lazy proxy (nnsight <= 0.4): tuple contract

    # ------------------------------------------------------------------ #
    # Generation / confidence
    # ------------------------------------------------------------------ #
    def _generate_text(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> str:
        import torch

        torch.manual_seed(seed)
        with self.lm.generate(prompt, max_new_tokens=max_tokens,
                              do_sample=temperature > 0, temperature=temperature or None):
            out = self.lm.generator.output.save()
        seq = getattr(out, "value", out)
        gen_ids = seq[0][len(self.tokenizer(prompt)["input_ids"]):]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _score_tokens(self, prompt: str, continuation: str):
        """Teacher-force ``continuation`` after ``prompt`` in ONE trace.

        Returns ``(token_texts, token_logprobs, token_entropies)`` with entropies
        in full-vocabulary nats, so the confidence family is populated on this
        backend (it used to be silently empty at WHITE tier — a strictly worse
        signal set than the grey-box GGUF path) and is comparable with the HF
        backend. Note the boundary is located by re-tokenising, so a token that
        merges across the prompt/answer seam can shift the first scored position
        by one; that is the same approximation the HF backend makes.
        """
        import torch

        if not continuation:
            return [], [], []
        lead = "" if (not prompt or prompt[-1].isspace() or continuation[:1].isspace()) else " "
        full = prompt + lead + continuation
        n_prompt = len(self.tokenizer(prompt)["input_ids"])
        ids = self.tokenizer(full)["input_ids"]
        start = max(1, n_prompt)
        head = self._resolve(self._HEAD_PATHS)
        if start >= len(ids) or head is None:
            return [], [], []
        with self.lm.trace(full):
            # Slice inside the trace so only the scored rows are retained.
            logits_save = head.output[0, start - 1:].save()
        logits = getattr(logits_save, "value", logits_save)
        texts, logprobs, entropies = [], [], []
        for k, tok in enumerate(ids[start:]):
            logp = torch.log_softmax(logits[k].float(), dim=-1)
            logprobs.append(float(logp[int(tok)]))
            p = logp.exp()
            entropies.append(float(-(p * logp).sum()))
            texts.append(self.tokenizer.decode([int(tok)]))
        return texts, logprobs, entropies

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0,
                 seed: int = 0) -> GenerationResult:
        text = self._generate_text(prompt, max_tokens, temperature, seed)
        if not self.supports(Tier.GREY) or not text.strip():
            return GenerationResult(text=text)
        texts, logprobs, entropies = self._score_tokens(prompt, text)
        return GenerationResult(text, texts, logprobs, entropies)

    def sample(self, prompt: str, k: int = 10, temperature: float = 1.0) -> list[str]:
        # Text only: semantic entropy needs the strings, and the base
        # implementation would pay _score_tokens' extra forward pass k times.
        return [self._generate_text(prompt, 64, temperature, i) for i in range(k)]

    def confidence(self, prompt: str, answer: Optional[str] = None) -> GenerationResult:
        """Token logprobs/entropies **of the given answer** (teacher-forced).

        The base implementation ignores ``answer`` and re-generates, which would
        make answer_perplexity / *_token_entropy describe a fresh continuation
        rather than the answer being diagnosed.
        """
        self.require(Tier.GREY)
        if not answer or not answer.strip():
            return self.generate(prompt, temperature=0.0)
        texts, logprobs, entropies = self._score_tokens(prompt, answer)
        if not logprobs:
            return self.generate(prompt, temperature=0.0)
        return GenerationResult(answer, texts, logprobs, entropies)

    # ------------------------------------------------------------------ #
    def capture(self, inference: Inference) -> CaptureResult:
        self.require(Tier.GREY)
        enc = self.tokenizer(inference.prompt, return_tensors="pt",
                             return_offsets_mapping=True)
        # _context_spans reads only inference.prompt + the offset mapping (not self).
        ctx_pos, gold_pos = HFModel._context_spans(
            self, inference, {**{k: v for k, v in enc.items()},
                              "offset_mapping": enc["offset_mapping"]})
        last = enc["input_ids"].shape[1] - 1
        layers = self._resolve(self._LAYER_PATHS)
        if layers is None:
            # TierUnavailable (not a bare RuntimeError): SignalPipeline.run only
            # catches this one, so anything else aborts the whole diagnosis.
            raise TierUnavailable(
                f"{self.profile.name}: cannot locate decoder layers for nnsight capture")
        norm = self._resolve(self._NORM_PATHS)
        head = self._resolve(self._HEAD_PATHS)
        n = len(layers)

        # Declared OUTSIDE the trace: names bound inside the `with` body are not
        # reliably readable afterwards on nnsight >= 0.4.
        attn_saves: list = []
        lens_tok_saves: list = []
        with self.lm.trace(inference.prompt):
            # SINGLE forward-order pass: grab everything a layer needs while we are
            # visiting it. Two passes over `layers` make the second one request a
            # one-shot hook that has already fired and self-removed ->
            # MissedProviderError, which aborts the whole pipeline.
            for layer in layers:
                attn_saves.append(layer.self_attn.output[1].save())  # (attn_out, weights)
                if norm is not None and head is not None:
                    # logit-lens INSIDE the trace: project each layer's (normed)
                    # residual. Envoy modules only execute in a trace context.
                    hpos = self._hidden(layer.output)[0, last]
                    lens_tok_saves.append(head(norm(hpos)).argmax(dim=-1).save())

        attentions = [getattr(a, "value", a) for a in attn_saves]
        lens_tokens = [int(getattr(t, "value", t)) for t in lens_tok_saves]

        attn_to = HFModel._attention_mass(attentions, last, ctx_pos,
                                          self.profile.retrieval_heads or None)
        gold_attn = (HFModel._attention_mass(attentions, last, gold_pos,
                                             self.profile.retrieval_heads or None)
                     if gold_pos else None)
        if lens_tokens:
            # The final layer's normed residual through lm_head IS the model's own
            # next-token distribution, so its argmax is the answer token. Reading it
            # from `lens_tokens[-1]` avoids also requesting `lm_head.output`, which
            # the manual per-layer head() calls above would otherwise race to
            # provide. Same reference the HF backend's lens uses (hidden[-1]).
            answer_layer, stability = self._lens_stats(lens_tokens, lens_tokens[-1], n)
        else:
            # No norm/head found for this architecture -> abstain instead of
            # reporting an unnormalised lens as a present feature.
            answer_layer, stability = None, None
        external = attn_to
        # NOTE (documented limitation, mirrors models/hf.py): `parametric` is a
        # deterministic function of `external`, so the two ReDeEP anchors are not
        # independent measurements on this backend.
        parametric = min(1.0, max(0.0, 1.0 - external) * (1.0 - (answer_layer or 0.5)) * 2)

        result = CaptureResult(
            n_layers=n,
            context_attention_ratio=round(attn_to, 3),
            gold_attention_ratio=round(gold_attn, 3) if gold_attn is not None else None,
            logit_lens_answer_layer=answer_layer,
            logit_lens_stability=stability,
            external_context_score=round(external, 3),
            parametric_knowledge_score=round(parametric, 3),
        )
        if self.supports(Tier.WHITE) and gold_pos:
            result.gold_patch_effect = self._patch_gold(inference, gold_pos)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _lens_stats(lens_tokens: list[int], answer_tok: int, n: int) -> tuple[float, float]:
        first, matches = None, 0
        for i, tok in enumerate(lens_tokens):
            if tok == answer_tok:
                matches += 1
                if first is None:
                    first = i + 1
        answer_layer = (first / n) if first else 1.0
        return round(answer_layer, 3), round(matches / n, 3)

    def _patch_gold(self, inference: Inference, gold_pos: list[int]) -> float:
        """Activation patching: zero the gold positions in the layer-0 residual
        stream and measure the drop in the answer-token LOG-PROB (same scale as the
        HF backend's causal test, so the field is comparable across backends)."""
        answer = inference.generated_answer.strip().split()
        if not answer:
            return 0.0
        tgt = self.tokenizer(" " + answer[0], add_special_tokens=False)["input_ids"]
        if not tgt:
            return 0.0
        tgt = tgt[0]
        layers = self._resolve(self._LAYER_PATHS)
        head = self._resolve(self._HEAD_PATHS)
        if layers is None or head is None:
            return 0.0
        with self.lm.trace(inference.prompt):
            clean = head.output[0, -1].log_softmax(dim=-1)[tgt].save()
        with self.lm.trace(inference.prompt):
            # _hidden(): indexing [0] first would zero a slice of the BATCH dim.
            self._hidden(layers[0].output)[:, gold_pos, :] = 0
            ablated = head.output[0, -1].log_softmax(dim=-1)[tgt].save()
        return round(float(getattr(clean, "value", clean)) - float(getattr(ablated, "value", ablated)), 3)
