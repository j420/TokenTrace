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

Capture cost guard (``max_capture_tokens``, default 1024)
--------------------------------------------------------
``output_attentions=True`` materialises ``n_layers x n_heads x S x S`` weights
for *every* layer simultaneously: for the registered qwen3-4b profile (36 layers
x 32 heads, bf16) that is 2.25 GiB at S=1024, 9 GiB at S=2048 and 36 GiB at
S=4096 — and the resulting ``MemoryError`` is not a ``TierUnavailable``, so it
aborts the whole diagnosis. An over-length prompt does **not** raise on RoPE
models either (measured: 640 tokens through a ``max_position_embeddings=512``
model returns normally); it degrades *silently*. An explicit token cap is
therefore the only defence, and capture keeps the **last** ``max_capture_tokens``
tokens because every capture feature is read from the final (answer-forming)
position.

Confidence semantics: :meth:`HFModel.confidence` teacher-forces the *given*
answer instead of re-generating (the base-class default would describe a
different string than the one being diagnosed).

All heavy imports are lazy. This backend is the fallback ladder's rung above
NNsight/TransformerLens: it uses only ``transformers`` forward hooks
(``output_attentions``/``output_hidden_states``), so it works on any HF model —
including ones TransformerLens does not yet support.
"""

from __future__ import annotations

import math
from typing import Optional

from tokentrace.core.types import Chunk, Inference, ModelProfile, Tier
from tokentrace.models.base import CaptureResult, GenerationResult, ModelHandle


class HFModel(ModelHandle):
    #: Where the final pre-``lm_head`` norm lives across architectures. Composite
    #: models (e.g. ``Gemma3ForConditionalGeneration``, a REGISTERED profile) keep
    #: it under ``model.language_model``; ``model.norm`` is ``None`` there.
    _NORM_PATHS = (
        "model.norm",
        "model.language_model.norm",
        "model.language_model.final_layernorm",
        "model.decoder.final_layer_norm",
        "model.final_layernorm",
        "model.transformer.ln_f",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
    )

    def __init__(
        self,
        profile: ModelProfile,
        tier: Tier = Tier.WHITE,
        hf_repo: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cpu",
        max_capture_tokens: int = 1024,
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
            attn_implementation="eager",  # needed to read attention weights
            **model_kwargs,
        ).to(device).eval()
        # NB: output_attentions/hidden_states are requested per-forward in capture()
        # (not globally in config) so generate() doesn't waste memory accumulating them.
        self.device = device
        # Hard cap on the sequence length any capture/scoring forward may see; see
        # the module docstring for the O(L*H*S^2) blow-up this prevents.
        self.max_capture_tokens = max(8, int(max_capture_tokens))

        # Fill exact architecture facts from the real config.
        cfg = self.model.config
        profile.n_layers = getattr(cfg, "num_hidden_layers", profile.n_layers)
        profile.n_heads = getattr(cfg, "num_attention_heads", profile.n_heads)
        profile.d_model = getattr(cfg, "hidden_size", profile.d_model)
        self.profile = profile
        self.tier = Tier(min(int(tier), int(profile.max_tier)))

        # Does this transformers version let us skip the [1, S, vocab] logits
        # tensor on capture/scoring forwards? (`logits_to_keep` landed in 4.49;
        # older versions called it `num_logits_to_keep`.) Detected by signature so
        # we never pass an unknown kwarg into a forward that swallows **kwargs.
        self._logits_to_keep_kw: Optional[str] = None
        try:
            import inspect

            params = inspect.signature(self.model.forward).parameters
            for kw in ("logits_to_keep", "num_logits_to_keep"):
                if kw in params:
                    self._logits_to_keep_kw = kw
                    break
        except (TypeError, ValueError):  # pragma: no cover - exotic forward
            self._logits_to_keep_kw = None

        # Record the entropy scale so calibration can never silently mix a
        # full-vocabulary entropy (here) with a truncated top-k one (gguf.py).
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        if vocab > 1:
            profile.tokenizer_quirks["entropy_scale"] = {
                "estimator": "full_vocab",
                "max_nats": round(math.log(vocab), 4),
                "vocab_size": vocab,
            }

    # ------------------------------------------------------------------ #
    def _encode(self, prompt: str, offsets: bool = False):
        """Tokenize with the capture guard applied (keep the TAIL of the prompt).

        Right-truncation would move the final position into the middle of the
        retrieved context, and every capture feature is read from that position;
        left-truncation keeps the answer-forming position and only drops the
        earliest context. Offsets stay char offsets into the *original* prompt,
        so :meth:`_context_spans` still aligns.
        """
        kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_capture_tokens,
        }
        if offsets:
            kwargs["return_offsets_mapping"] = True
        prev = getattr(self.tokenizer, "truncation_side", None)
        try:
            if prev is not None:
                self.tokenizer.truncation_side = "left"
            return self.tokenizer(prompt, **kwargs)
        finally:
            if prev is not None:
                self.tokenizer.truncation_side = prev

    def _forward(self, enc, *, keep_logits: Optional[int] = None, **fwd):
        """Forward pass, asking for only the last ``keep_logits`` logit rows when
        the installed transformers supports it ([1, S, vocab] is 0.6 GiB at
        S=2048 for a 152k vocab and capture never reads it)."""
        if keep_logits and self._logits_to_keep_kw:
            fwd[self._logits_to_keep_kw] = keep_logits
        return self.model(**enc, use_cache=False, **fwd)

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
            logp = torch.log_softmax(scores[0].float(), dim=-1)
            tok = gen_ids[i]
            logprobs.append(float(logp[tok]))
            p = logp.exp()
            entropies.append(float(-(p * logp).sum()))
            ttexts.append(self.tokenizer.decode([tok]))
        return GenerationResult(text, ttexts, logprobs, entropies)

    def confidence(self, prompt: str, answer: Optional[str] = None) -> GenerationResult:
        """Token logprobs/entropies **of the given answer** (teacher-forced).

        The base implementation ignores ``answer`` and re-generates, so
        ``answer_perplexity`` / ``mean_token_entropy`` / ``max_token_entropy``
        would describe the model's own fresh 64-token continuation rather than
        the answer being diagnosed — for a logged production trace, a different
        string entirely. Entropies are full-vocabulary nats, the same scale as
        :meth:`generate`.
        """
        import torch

        self.require(Tier.GREY)
        if not answer or not answer.strip():
            return self.generate(prompt, temperature=0.0)

        p_ids = list(self.tokenizer(prompt)["input_ids"])
        # A continuation is tokenized with its leading space attached; mirror how
        # the answer would actually have been produced after this prompt.
        lead = "" if (not prompt or prompt[-1].isspace() or answer[:1].isspace()) else " "
        a_ids = list(self.tokenizer(lead + answer, add_special_tokens=False)["input_ids"])
        if not p_ids or not a_ids:
            return self.generate(prompt, temperature=0.0)
        # Keep the scored window inside the capture guard (same reasoning as
        # capture(): drop the OLDEST prompt tokens, never the answer).
        budget = max(1, self.max_capture_tokens - len(a_ids))
        if len(p_ids) > budget:
            p_ids = p_ids[-budget:]

        ids = torch.tensor([p_ids + a_ids], device=self.device)
        with torch.no_grad():
            logits = self._forward({"input_ids": ids}, keep_logits=len(a_ids) + 1).logits[0]
        # logits[i] predicts ids[i+1]; with logits_to_keep=len(a)+1 the returned
        # rows are already the last len(a)+1 positions, so the answer starts at 0.
        base = 0 if (self._logits_to_keep_kw and logits.shape[0] == len(a_ids) + 1) \
            else len(p_ids) - 1
        texts, logprobs, entropies = [], [], []
        for k, tok in enumerate(a_ids):
            logp = torch.log_softmax(logits[base + k].float(), dim=-1)
            logprobs.append(float(logp[tok]))
            p = logp.exp()
            entropies.append(float(-(p * logp).sum()))
            texts.append(self.tokenizer.decode([tok]))
        return GenerationResult(answer, texts, logprobs, entropies)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunk_char_spans(prompt: str, chunks) -> list[tuple[Chunk, int, int]]:
        """Locate each chunk's char span in ``prompt``, walking a cursor.

        ``prompt.find(text)`` always returns the FIRST occurrence, so duplicate or
        nested passages (routine for a real retriever) all collapsed onto the same
        span — measured 3 identical chunks reporting a 0.7059 context-attention
        ratio against a true 0.2353, unbounded above 1.0. The cursor makes the
        n-th duplicate map to the n-th occurrence.
        """
        spans: list[tuple[Chunk, int, int]] = []
        cursor = 0
        for chunk in chunks or []:
            text = chunk.text
            if not text:
                continue
            start = prompt.find(text, cursor)
            if start < 0:  # chunk order != prompt order: fall back to any occurrence
                start = prompt.find(text)
            if start < 0:
                continue
            end = start + len(text)
            cursor = max(cursor, end)
            spans.append((chunk, start, end))
        return spans

    def _context_spans(self, inference: Inference, enc) -> tuple[list[int], list[int]]:
        """Return (context_token_positions, gold_token_positions) via char offsets.

        Sets, not lists: a repeated position would otherwise be summed once per
        occurrence by :meth:`_attention_mass`. (Also called unbound by the NNsight
        backend, so it must not touch ``self``.)
        """
        offsets = enc["offset_mapping"][0].tolist()
        prompt = inference.prompt
        ctx: set[int] = set()
        gold: set[int] = set()
        for chunk, start, end in HFModel._chunk_char_spans(prompt, inference.retrieved_context):
            for ti, (a, b) in enumerate(offsets):
                # Overlap, not containment: HF fast BPE tokenizers can fold the
                # preceding whitespace into a token's offsets (' The' -> [72,76]
                # for a chunk starting at 73), and strict containment then dropped
                # the first token of every space-preceded chunk.
                if b > a and max(a, start) < min(b, end):
                    ctx.add(ti)
                    if chunk.gold:
                        gold.add(ti)
        return sorted(ctx), sorted(gold)

    def capture(self, inference: Inference) -> CaptureResult:
        import torch

        self.require(Tier.GREY)
        enc = self._encode(inference.prompt, offsets=True)
        offset = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        enc_with_off = {**enc, "offset_mapping": offset}

        with torch.no_grad():
            # keep_logits=1: capture reads hidden states + attentions only.
            out = self._forward(enc, keep_logits=1,
                                output_attentions=True, output_hidden_states=True)
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
        # KNOWN LIMITATION (documented, not fixed here): `parametric` is a
        # deterministic function of `external`, so a rule that treats "low external
        # AND high parametric" as two corroborating signals is reading one
        # measurement twice. A true ReDeEP parametric score needs the FFN/MLP
        # contribution to the answer logit; changing the scale here would desync
        # the calibrator, which is fit on the mock's independent values.
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

    def _find_final_norm(self):
        """The final pre-``lm_head`` norm, or None if this architecture hides it
        somewhere we don't know about.

        ``getattr(self.model.model, 'norm', None)`` returns None for composite
        models (``Gemma3ForConditionalGeneration.model`` holds vision_tower /
        multi_modal_projector / language_model), and the lens then ran *without
        any normalisation* while still reporting its features as present.
        """
        for path in self._NORM_PATHS:
            obj = self.model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and callable(obj):
                return obj
        return None

    def _logit_lens(self, hidden, pos: int) -> tuple[Optional[float], Optional[float]]:

        try:
            lm_head = self.model.get_output_embeddings()
        except Exception:
            return None, None
        norm = self._find_final_norm()
        if lm_head is None or norm is None:
            # Abstain rather than report unnormalised garbage as a present feature:
            # pre-norm residual streams are on a different scale from lm_head's
            # input, so the per-layer argmax would be meaningless. Returning None
            # marks the mechanistic lens features MISSING, which the missingness
            # mask already knows how to handle.
            return None, None
        # hidden[-1] is the post-final-norm state (HF appends it after model.norm),
        # so it feeds lm_head directly; earlier states are pre-norm residual streams.
        final_logits = lm_head(hidden[-1][0, pos])
        answer_tok = int(final_logits.argmax())
        n = len(hidden) - 1
        last_idx = len(hidden) - 1
        first_match = None
        matches = 0
        for layer in range(1, len(hidden)):
            h = hidden[layer][0, pos]
            # Apply the final norm to pre-norm intermediates only; hidden[-1] is
            # already normed (applying norm twice — RMSNorm is not idempotent —
            # would make the final layer spuriously mismatch answer_tok).
            if layer != last_idx:
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
            enc = self._encode(prompt)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._forward(enc, keep_logits=1).logits[0, -1]
            return float(torch.log_softmax(logits.float(), dim=-1)[target])

        full = answer_logit(inference.prompt)
        kept_chunks = [c for c in (inference.retrieved_context or []) if not c.gold]
        ablated = Inference(
            prompt=self._rebuild_prompt(inference, kept_chunks),
            generated_answer=inference.generated_answer,
            retrieved_context=kept_chunks,
        )
        return round(full - answer_logit(ablated.prompt), 3)

    @staticmethod
    def _rebuild_prompt(inference: Inference, chunks: list[Chunk]) -> str:
        """Rebuild the prompt keeping exactly ``chunks`` (order/template preserved).

        Deletes each dropped chunk's *located span* plus one adjacent separator,
        instead of ``prompt.replace(gold_text, '')``. The old version had three
        defects: (a) ``replace`` is GLOBAL, so a non-gold chunk whose text equals
        the gold text was deleted from the prompt while still being listed as
        kept; (b) it left dangling separators ('...Paris.\\n', or a blank line for
        a middle chunk), shifting the continuation point the ablated logprob is
        read at; (c) the ``chunks`` argument was ignored entirely, so
        ``_rebuild_prompt(inf, [])`` and ``_rebuild_prompt(inf, all)`` returned the
        same string. Deleting located spans also keeps any instruction/header
        scaffolding that a template put around the chunks.
        """
        prompt = inference.prompt
        all_chunks = list(inference.retrieved_context or [])
        if not all_chunks:
            return prompt

        # Consume each keeper once (by identity, then by value) so duplicated
        # chunk texts drop exactly as many occurrences as were dropped.
        remaining = list(chunks or [])
        drop: list[tuple[int, int]] = []
        for chunk, start, end in HFModel._chunk_char_spans(prompt, all_chunks):
            hit = next((k for k in remaining if k is chunk), None)
            if hit is None:
                hit = next((k for k in remaining
                            if k.text == chunk.text and k.source_id == chunk.source_id), None)
            if hit is not None:
                remaining.remove(hit)
            else:
                drop.append((start, end))
        if not drop:
            return prompt

        out = prompt
        for start, end in sorted(drop, reverse=True):  # right-to-left keeps offsets valid
            s, e = start, end
            j = e
            while j < len(out) and out[j].isspace():
                j += 1
            if j < len(out):
                e = j                      # content follows: eat the trailing separator
            else:
                e = j                      # chunk was last: eat trailing whitespace ...
                while s > 0 and out[s - 1].isspace():
                    s -= 1                 # ... and the separator that preceded it
            out = out[:s] + out[e:]
        return out
