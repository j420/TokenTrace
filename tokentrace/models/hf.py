"""HuggingFace mechanistic-capture backend (grey/white-box, CPU-friendly).

Grey-box (single forward pass, no GPU needed):
  * attention-to-context / attention-to-gold from ``output_attentions``
  * logit-lens answer-formation depth from ``output_hidden_states``
  * ReDeEP-lite external-context vs parametric-knowledge scores

White-box (a few extra forward passes):
  * ``gold_patch_effect`` — causal effect of the gold-chunk tokens on the answer
    logit, computed by input ablation (remove gold, measure answer-logit drop).
    This is the CPU-affordable stand-in for activation patching and gives a true
    causal read without a GPU.

All heavy imports are lazy. This backend is the fallback ladder's rung above
NNsight/TransformerLens: it uses only ``transformers`` forward hooks
(``output_attentions``/``output_hidden_states``), so it works on any HF model —
including ones TransformerLens does not yet support.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import Chunk, Inference, ModelProfile, Tier
from tokentrace.models.base import CaptureResult, GenerationResult, ModelHandle


class HFModel(ModelHandle):
    def __init__(
        self,
        profile: ModelProfile,
        tier: Tier = Tier.WHITE,
        hf_repo: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cpu",
        **model_kwargs,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError(
                "HF backend needs torch+transformers. Install: pip install 'tokentrace[mechanistic]'"
            ) from e
        import torch

        repo = hf_repo or profile.hf_repo or profile.name
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=getattr(torch, dtype, torch.float32),
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",  # needed to read attention weights
            **model_kwargs,
        ).to(device).eval()
        self.device = device

        # Fill exact architecture facts from the real config.
        cfg = self.model.config
        profile.n_layers = getattr(cfg, "num_hidden_layers", profile.n_layers)
        profile.n_heads = getattr(cfg, "num_attention_heads", profile.n_heads)
        profile.d_model = getattr(cfg, "hidden_size", profile.d_model)
        self.profile = profile
        self.tier = Tier(min(int(tier), int(profile.max_tier)))

    # ------------------------------------------------------------------ #
    def generate(
        self, prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        import torch

        torch.manual_seed(seed)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        out = self.model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            output_scores=True,
            return_dict_in_generate=True,
        )
        gen_ids = out.sequences[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        logprobs, entropies, ttexts = [], [], []
        for i, scores in enumerate(out.scores):
            logp = torch.log_softmax(scores[0], dim=-1)
            tok = gen_ids[i]
            logprobs.append(float(logp[tok]))
            p = logp.exp()
            entropies.append(float(-(p * logp).sum()))
            ttexts.append(self.tokenizer.decode([tok]))
        return GenerationResult(text, ttexts, logprobs, entropies)

    # ------------------------------------------------------------------ #
    def _context_spans(self, inference: Inference, enc) -> tuple[list[int], list[int]]:
        """Return (context_token_positions, gold_token_positions) via char offsets."""
        offsets = enc["offset_mapping"][0].tolist()
        prompt = inference.prompt
        ctx_positions: list[int] = []
        gold_positions: list[int] = []
        for chunk in inference.retrieved_context or []:
            start = prompt.find(chunk.text)
            if start < 0:
                continue
            end = start + len(chunk.text)
            for ti, (a, b) in enumerate(offsets):
                if a >= start and b <= end and b > a:
                    ctx_positions.append(ti)
                    if chunk.gold:
                        gold_positions.append(ti)
        return ctx_positions, gold_positions

    def capture(self, inference: Inference) -> CaptureResult:
        import torch

        self.require(Tier.GREY)
        enc = self.tokenizer(
            inference.prompt, return_tensors="pt", return_offsets_mapping=True
        )
        offset = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        enc_with_off = {**enc, "offset_mapping": offset}

        with torch.no_grad():
            out = self.model(**enc, use_cache=False)
        attentions = out.attentions          # tuple[L] of [1, H, S, S]
        hidden = out.hidden_states           # tuple[L+1] of [1, S, D]
        seq_len = attentions[0].shape[-1]
        last = seq_len - 1

        ctx_pos, gold_pos = self._context_spans(inference, enc_with_off)

        # --- attention to context / gold from the answer-forming position --- #
        # Average attention from the last position over the retrieval heads (or all
        # heads if the model's retrieval heads are not yet catalogued).
        heads = self.profile.retrieval_heads or None
        attn_to = self._attention_mass(attentions, last, ctx_pos, heads)
        gold_attn = self._attention_mass(attentions, last, gold_pos, heads) if gold_pos else None

        # --- logit lens: depth at which the final answer token stabilizes --- #
        answer_layer, stability = self._logit_lens(hidden, last)

        # ReDeEP-lite: external = context attention (heads reading context);
        # parametric = how early/strongly the answer forms without context support.
        external = attn_to
        parametric = max(0.0, 1.0 - external) * (1.0 - (answer_layer or 0.5)) * 2
        parametric = min(1.0, parametric)

        result = CaptureResult(
            n_layers=len(attentions),
            context_attention_ratio=round(attn_to, 3),
            gold_attention_ratio=round(gold_attn, 3) if gold_attn is not None else None,
            logit_lens_answer_layer=answer_layer,
            logit_lens_stability=stability,
            external_context_score=round(external, 3),
            parametric_knowledge_score=round(parametric, 3),
        )

        if self.supports(Tier.WHITE) and gold_pos:
            result.gold_patch_effect = self._gold_ablation_effect(inference)
        return result

    @staticmethod
    def _attention_mass(attentions, from_pos: int, to_positions: list[int], heads) -> float:

        if not to_positions:
            return 0.0
        vals = []
        for layer_idx, att in enumerate(attentions):
            row = att[0, :, from_pos, :]           # [H, S]
            if heads:
                sel = [h for (l, h) in heads if l == layer_idx]
                if not sel:
                    continue
                row = row[sel]
            mass = row[:, to_positions].sum(dim=-1) / row.sum(dim=-1).clamp_min(1e-9)
            vals.append(float(mass.mean()))
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _logit_lens(self, hidden, pos: int) -> tuple[Optional[float], Optional[float]]:

        try:
            lm_head = self.model.get_output_embeddings()
            norm = getattr(self.model.model, "norm", None)
        except Exception:
            return None, None
        final_logits = lm_head(hidden[-1][0, pos])
        answer_tok = int(final_logits.argmax())
        n = len(hidden) - 1
        first_match = None
        matches = 0
        for layer in range(1, len(hidden)):
            h = hidden[layer][0, pos]
            if norm is not None:
                h = norm(h)
            top = int(lm_head(h).argmax())
            if top == answer_tok:
                matches += 1
                if first_match is None:
                    first_match = layer
        answer_layer = (first_match / n) if first_match else 1.0
        stability = matches / n
        return round(answer_layer, 3), round(stability, 3)

    def _gold_ablation_effect(self, inference: Inference) -> float:
        """Causal test: answer-logit(full) - answer-logit(gold removed).

        Large positive => the answer causally depends on the gold context (grounded).
        ~0 => the answer is produced regardless of the gold (parametric/hallucinated).
        """
        import torch

        answer = inference.generated_answer.strip().split()
        if not answer:
            return 0.0
        target = self.tokenizer(" " + answer[0], add_special_tokens=False)["input_ids"]
        if not target:
            return 0.0
        target = target[0]

        def answer_logit(prompt: str) -> float:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc, use_cache=False).logits[0, -1]
            return float(torch.log_softmax(logits, dim=-1)[target])

        full = answer_logit(inference.prompt)
        ablated_chunks = [c for c in (inference.retrieved_context or []) if not c.gold]
        ablated = Inference(
            prompt=self._rebuild_prompt(inference, ablated_chunks),
            generated_answer=inference.generated_answer,
            retrieved_context=ablated_chunks,
        )
        return round(full - answer_logit(ablated.prompt), 3)

    @staticmethod
    def _rebuild_prompt(inference: Inference, chunks: list[Chunk]) -> str:
        """Best-effort: drop gold-chunk text from the prompt string."""
        prompt = inference.prompt
        for c in inference.retrieved_context or []:
            if c.gold:
                prompt = prompt.replace(c.text, "")
        return prompt
