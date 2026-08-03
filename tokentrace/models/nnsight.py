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
* Saved proxies are read via ``.value`` after the ``trace`` block.
* The logit-lens is computed **inside** the trace (Envoy modules only execute in a
  trace context). Each layer's raw output is pre-final-norm, so ``model.norm`` is
  applied once to every layer here.
* Decoder layers are assumed to return a tuple ``(hidden_states, ...)`` (true for the
  Llama/Qwen/Gemma families); adjust if a version returns a bare tensor.

Lazy imports; needs ``pip install 'tokentrace[mechanistic]'`` (nnsight + torch +
transformers). Runs on CPU.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import Inference, ModelProfile, Tier
from tokentrace.models.base import CaptureResult, GenerationResult, ModelHandle
from tokentrace.models.hf import HFModel


class NNsightModel(ModelHandle):
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

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0,
                 seed: int = 0) -> GenerationResult:
        import torch

        torch.manual_seed(seed)
        with self.lm.generate(prompt, max_new_tokens=max_tokens,
                              do_sample=temperature > 0, temperature=temperature or None):
            out = self.lm.generator.output.save()
        seq = getattr(out, "value", out)
        gen_ids = seq[0][len(self.tokenizer(prompt)["input_ids"]):]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(text=text)

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
        layers = self.lm.model.layers
        n = len(layers)

        with self.lm.trace(inference.prompt):
            attn_saves = [layer.self_attn.output[1].save() for layer in layers]
            # logit-lens INSIDE the trace: project each layer's (normed) residual.
            lens_tok_saves = []
            for layer in layers:
                hpos = layer.output[0][0, last]              # tuple output -> hidden_states
                lens_tok_saves.append(
                    self.lm.lm_head(self.lm.model.norm(hpos)).argmax(dim=-1).save())
            final_tok_save = self.lm.lm_head.output[0, last].argmax(dim=-1).save()

        attentions = [getattr(a, "value", a) for a in attn_saves]
        answer_tok = int(getattr(final_tok_save, "value", final_tok_save))
        lens_tokens = [int(getattr(t, "value", t)) for t in lens_tok_saves]

        attn_to = HFModel._attention_mass(attentions, last, ctx_pos,
                                          self.profile.retrieval_heads or None)
        gold_attn = (HFModel._attention_mass(attentions, last, gold_pos,
                                             self.profile.retrieval_heads or None)
                     if gold_pos else None)
        answer_layer, stability = self._lens_stats(lens_tokens, answer_tok, n)
        external = attn_to
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
        with self.lm.trace(inference.prompt):
            clean = self.lm.lm_head.output[0, -1].log_softmax(dim=-1)[tgt].save()
        with self.lm.trace(inference.prompt):
            self.lm.model.layers[0].output[0][:, gold_pos, :] = 0
            ablated = self.lm.lm_head.output[0, -1].log_softmax(dim=-1)[tgt].save()
        return round(float(getattr(clean, "value", clean)) - float(getattr(ablated, "value", ablated)), 3)
