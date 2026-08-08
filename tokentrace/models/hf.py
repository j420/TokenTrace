"""HuggingFace mechanistic-capture backend (grey/white-box, CPU-friendly).

Grey-box (single forward pass, no GPU needed):
  * attention-to-context / attention-to-gold from ``output_attentions``
  * logit-lens answer-formation depth from ``output_hidden_states``
  * ReDeEP-lite external-context vs parametric-knowledge scores — two DISTINCT
    measurements (see :meth:`HFModel._external_copy_score` and
    :meth:`HFModel._parametric_share` for exactly what each one reads)

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

All heavy imports are lazy. Backends are selected explicitly via
``registry.load_model(backend=...)`` — there is no automatic backend-to-backend
fallback. This is the default capture backend: it uses only ``transformers``
built-ins (``output_attentions``/``output_hidden_states`` plus read-only forward
hooks), so it works on any HF causal LM; ``models/nnsight.py`` sits beside it for
intervention-style (activation-patching) capture. The only automatic degradation
is the TIER ladder (white -> grey -> black) driven by ``ModelProfile.max_tier``.
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
        # transformers renamed `torch_dtype` -> `dtype` in 4.56 and 5.x warns on
        # the old name; below 4.56 the NEW name is silently swallowed into the
        # config (fp32 weights, no error), so this must be a version check, not
        # try/except. Parsed leniently: "5.0.0.dev0"-style suffixes are ignored.
        dtype_kw = "torch_dtype"
        try:
            import transformers

            _v = tuple(int("".join(ch for ch in p if ch.isdigit()) or 0)
                       for p in transformers.__version__.split(".")[:2])
            if _v >= (4, 56):
                dtype_kw = "dtype"
        except Exception:  # pragma: no cover - unparseable version string
            pass
        model_kwargs.setdefault(dtype_kw, getattr(torch, dtype, torch.float32))
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
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

        # Read-only taps on each decoder layer + its FFN sublayer, so the SAME
        # forward pass also yields the per-layer FFN residual contributions the
        # parametric score needs (output_hidden_states alone cannot separate a
        # layer's attention-sublayer update from its FFN-sublayer update).
        taps = self._register_parametric_taps()
        try:
            with torch.no_grad():
                # keep_logits=1: capture reads hidden states + attentions only.
                out = self._forward(enc, keep_logits=1,
                                    output_attentions=True, output_hidden_states=True)
        finally:
            if taps is not None:
                for handle in taps["handles"]:
                    handle.remove()
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

        # --- ReDeEP-lite: two DECOUPLED anchors (Sun et al., ICLR 2025) --- #
        # external: attention-weighted copying from context toward the emitted
        # token (mass x content alignment) — NOT the raw attention mass, so it is
        # not an alias of context_attention_ratio.
        # parametric: share of the emitted token's logit-lens promotion done by
        # FFN sublayers rather than attention sublayers — measured from the
        # residual-stream decomposition, independent of context attention.
        external = self._external_copy_score(attentions, hidden[-1][0], last, ctx_pos, heads)
        parametric = self._parametric_share(hidden, taps, last)

        result = CaptureResult(
            n_layers=len(attentions),
            context_attention_ratio=round(attn_to, 3),
            gold_attention_ratio=round(gold_attn, 3) if gold_attn is not None else None,
            logit_lens_answer_layer=answer_layer,
            logit_lens_stability=stability,
            external_context_score=round(external, 3),
            parametric_knowledge_score=round(parametric, 3) if parametric is not None else None,
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

    # ------------------------------------------------------------------ #
    # ReDeEP-lite: decoupled external-context / parametric-knowledge scores
    # ------------------------------------------------------------------ #
    #: Where the decoder-layer list lives across architectures (same set the
    #: NNsight backend probes).
    _LAYER_PATHS = ("model.layers", "model.language_model.layers",
                    "model.decoder.layers", "transformer.h", "gpt_neox.layers")
    #: The module whose OUTPUT is the FFN contribution added to the residual
    #: stream. Gemma-2/3 style layers norm the MLP output before adding it
    #: (``post_feedforward_layernorm``), so that norm must be tapped there —
    #: tapping ``mlp`` would decompose the residual stream incorrectly.
    _FFN_TAP_ATTRS = ("post_feedforward_layernorm", "mlp", "feed_forward")

    def _find_layers(self):
        """The decoder-layer ModuleList, or None if this architecture hides it
        somewhere we don't know about (then the parametric score abstains)."""
        for path in self._LAYER_PATHS:
            obj = self.model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            try:
                if obj is not None and len(list(obj)) > 0:
                    return list(obj)
            except TypeError:        # resolved to a non-iterable module
                continue
        return None

    @staticmethod
    def _as_hidden(output):
        """A decoder layer returns ``(hidden, ...)`` on transformers < 4.54 and a
        bare tensor on >= 4.54; normalise to the hidden-state tensor."""
        return output[0] if isinstance(output, (tuple, list)) else output

    def _register_parametric_taps(self):
        """Register read-only forward hooks capturing, at every decoder layer,
        the layer's output residual and its FFN sublayer's residual contribution
        (both at the final position). Returns ``{"handles", "layer_out",
        "ffn_out", "calls"}`` or None when the architecture can't be tapped —
        the parametric score then abstains rather than guessing.
        """
        layers = self._find_layers()
        if layers is None:
            return None
        ffn_mods = []
        for layer in layers:
            tap = next((getattr(layer, a) for a in self._FFN_TAP_ATTRS
                        if getattr(layer, a, None) is not None), None)
            if tap is None or not callable(tap):
                return None          # e.g. OPT keeps fc1/fc2 loose on the layer
            ffn_mods.append(tap)

        n = len(layers)
        taps = {"handles": [], "layer_out": [None] * n, "ffn_out": [None] * n,
                "calls": [0] * n}

        def _mk(store, idx, unwrap):
            def hook(_module, _args, output):
                h = unwrap(output)
                store[idx] = h[0, -1, :].detach()
                if store is taps["ffn_out"]:
                    taps["calls"][idx] += 1
            return hook

        for i, (layer, ffn) in enumerate(zip(layers, ffn_mods)):
            taps["handles"].append(
                layer.register_forward_hook(_mk(taps["layer_out"], i, self._as_hidden)))
            taps["handles"].append(
                ffn.register_forward_hook(_mk(taps["ffn_out"], i, lambda o: o)))
        return taps

    @staticmethod
    def _external_copy_score(attentions, content, last, ctx_pos, heads) -> float:
        """External Context Score (ReDeEP-lite): attention-weighted copying from
        the retrieved context toward the emitted token.

        ``content`` is the [S, D] matrix of final-layer (post-final-norm) hidden
        states. Per layer: (attention mass on context positions from the
        answer-forming position) x (cosine alignment, mapped to [0,1], between
        the attention-weighted pooling of the context tokens' final-layer hidden
        states and the final-layer hidden state that produces the emitted
        token), averaged over layers; ``heads`` restricts to catalogued
        retrieval/copying heads when the profile has them. This follows ReDeEP's
        ECS (cosine between the pooled last-layer states of attended context
        tokens and the generating token's state), with attended-token pooling by
        attention weight instead of the paper's top-k selection. Unlike
        ``context_attention_ratio`` (pure mass), this is low when the model
        reads the context but emits something else — the two are distinct
        measurements, not aliases. (Also called by the NNsight backend.)
        """
        import torch

        if not ctx_pos:
            return 0.0
        content = content.float()            # final-layer states = token content
        emit = content[last]                 # the state that produces the answer
        vals = []
        for layer_idx, att in enumerate(attentions):
            row = att[0, :, last, :]         # [H, S]
            if heads:
                sel = [h for (l, h) in heads if l == layer_idx]
                if not sel:
                    continue
                row = row[sel]
            a = row.float().mean(dim=0)      # head-averaged attention [S]
            ctx_w = a[ctx_pos]
            ctx_sum = float(ctx_w.sum())
            if ctx_sum <= 0.0:
                vals.append(0.0)
                continue
            mass = ctx_sum / float(a.sum().clamp_min(1e-9))
            pooled = (ctx_w.unsqueeze(-1) * content[ctx_pos]).sum(dim=0) / ctx_sum
            cos = float(torch.nn.functional.cosine_similarity(pooled, emit, dim=0))
            vals.append(mass * (cos + 1.0) / 2.0)
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _parametric_share(self, hidden, taps, last) -> Optional[float]:
        """Parametric Knowledge Score (ReDeEP-lite): the share of the emitted
        token's logit-lens promotion contributed by FFN sublayers.

        Decomposes each layer's residual update at the answer-forming position
        into its attention part and its FFN part (``x_mid = x_out - ffn_out``,
        exact for pre-norm residual architectures incl. parallel-sublayer ones;
        Gemma-style post-FFN norms are tapped after the norm so the identity
        still holds), reads the emitted token's log-prob through the final-norm
        logit lens at ``x_in``/``x_mid``/``x_out``, and returns

            sum_l max(0, ffn_gain_l) / (sum_l max(0, attn_gain_l) + sum_l max(0, ffn_gain_l))

        in [0, 1]: 1 = the answer's probability was promoted entirely by FFNs
        (parametric memory, per ReDeEP's "Knowledge FFNs over-add parametric
        knowledge"), 0 = entirely by attention moving information in. This is the
        bounded, regression-free stand-in for ReDeEP's per-FFN logit-lens
        divergence, computed for the emitted token only; it shares NO term with
        the external score. Returns None (feature missing, handled by the
        missingness mask) when the architecture could not be tapped or the
        emitted token was never promoted.
        """
        import torch

        if taps is None or any(c != 1 for c in taps["calls"]) \
                or any(v is None for v in taps["layer_out"]) \
                or any(v is None for v in taps["ffn_out"]):
            return None
        try:
            lm_head = self.model.get_output_embeddings()
        except Exception:
            return None
        norm = self._find_final_norm()
        if lm_head is None or norm is None:
            return None

        n = len(taps["layer_out"])
        # hidden[l] is the (pre-norm) residual ENTERING layer l; the tapped
        # layer_out[l] is the (pre-norm) residual LEAVING it — used instead of
        # hidden[l+1] because hidden[-1] is already final-normed.
        x_in = [hidden[l][0, last] for l in range(n)]
        x_out = list(taps["layer_out"])
        x_mid = [o - f for o, f in zip(x_out, taps["ffn_out"])]
        stacked = torch.stack(x_in + x_mid + x_out)          # [3n, D] pre-norm
        with torch.no_grad():
            logits = lm_head(norm(stacked)).float()
            # Emitted token = argmax of the true final distribution (same
            # reference _logit_lens uses: hidden[-1] feeds lm_head directly).
            answer_tok = int(lm_head(hidden[-1][0, last]).float().argmax())
            lp = torch.log_softmax(logits, dim=-1)[:, answer_tok]
        lp_in, lp_mid, lp_out = lp[:n], lp[n:2 * n], lp[2 * n:]
        return self._ffn_share([float(v) for v in lp_in], [float(v) for v in lp_mid],
                               [float(v) for v in lp_out])

    @staticmethod
    def _ffn_share(lp_in, lp_mid, lp_out) -> Optional[float]:
        """The parametric-share formula on per-layer emitted-token log-probs.

        ``lp_in[l]`` / ``lp_mid[l]`` / ``lp_out[l]``: log-prob of the emitted
        token through the final-norm logit lens at layer ``l``'s input residual,
        post-attention residual, and post-FFN residual. An ``lp_in`` entry may be
        None (backend could not read that layer's input, e.g. NNsight without a
        resolvable embedding module for layer 0) — that layer's attention gain is
        then skipped. Returns None when the emitted token was never promoted.
        (Shared by the HF and NNsight backends so the score means the same thing
        on both.)
        """
        attn_gain = sum(max(0.0, m - i) for i, m in zip(lp_in, lp_mid) if i is not None)
        ffn_gain = sum(max(0.0, o - m) for m, o in zip(lp_mid, lp_out))
        total = attn_gain + ffn_gain
        if total <= 1e-9:
            return None
        return ffn_gain / total

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

    def _gold_ablation_effect(self, inference: Inference) -> Optional[float]:
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

        def answer_logit(prompt: str) -> Optional[float]:
            enc = self._encode(prompt)
            # A transformer cannot run a zero-length forward pass — it dies deep in
            # a tensor reshape. The ablated prompt IS empty whenever the gold chunk
            # was the entire prompt (a one-chunk RAG trace with no question text
            # survives ablation as ""), so the causal contrast is undefined there,
            # not zero: "we could not measure it" and "ablating gold changed
            # nothing" are opposite conclusions about groundedness.
            if enc["input_ids"].numel() == 0:
                return None
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
        removed = answer_logit(ablated.prompt)
        if full is None or removed is None:
            return None
        return round(full - removed, 3)

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
