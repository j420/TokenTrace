"""First real-backend integration tests: the REAL HF (and, when installed,
NNsight) capture path, driven end to end with zero network access.

A ~0.9M-param ``LlamaForCausalLM`` is built from a config and a BPE tokenizer is
trained on a scrap of local text, both saved into a session tmp dir — so the
exact code path used for registered production models (load -> encode -> capture
-> pipeline) runs against real transformers/torch with no downloads. Random
weights make the *diagnoses* meaningless; the tests therefore assert structural
properties (features present, distinct measurements not aliases, tier gating),
never diagnostic quality.

The whole module skips cleanly where torch/transformers are absent (the main CI
matrix); the backend-integration CI job runs it with real wheels.
"""

from __future__ import annotations

import os

import pytest

# Must be set before transformers is imported anywhere in this process: the
# tiny model lives on disk and nothing may ever touch the hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tokentrace.core.types import Chunk, Inference, Tier  # noqa: E402
from tokentrace.models.hf import HFModel  # noqa: E402
from tokentrace.models.registry import ModelProfile, register_profile  # noqa: E402

GOLD_TEXT = "The Eiffel Tower was completed in 1889."


# --------------------------------------------------------------------------- #
# Fixtures: build the tiny model once per session (seconds, no network).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-dev")
    cfg = LlamaConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)  # deterministic weights -> reproducible feature values
    model = LlamaForCausalLM(cfg)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    corpus = [
        (GOLD_TEXT + " Paris is in France. Marie Curie was born in 1867. "
         "Who signed the treaty? ") * 40
    ]
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(vocab_size=1000,
                            special_tokens=["<unk>", "<s>", "</s>", "<pad>"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>",
    )
    model.save_pretrained(out)
    fast.save_pretrained(out)
    return str(out)


@pytest.fixture(scope="session")
def hf_white(tiny_model_dir):
    profile = register_profile(
        ModelProfile(name="tiny-dev-test", n_layers=4, n_heads=4, d_model=128,
                     max_tier=Tier.WHITE)
    )
    return HFModel(profile, hf_repo=tiny_model_dir, tier=Tier.WHITE, dtype="float32")


def _fillers(n: int) -> list[Chunk]:
    return [
        Chunk(f"Unrelated filler sentence number {i} about other topics entirely.",
              0.3, f"d{i}")
        for i in range(n)
    ]


def _rag(question: str, chunks: list[Chunk], answer: str = "1889") -> Inference:
    return Inference(
        prompt=question + "\n" + "\n".join(c.text for c in chunks),
        generated_answer=answer,
        retrieved_context=list(chunks),
        ground_truth=[answer],
        question=question,
    )


def _subset_inference() -> Inference:
    """Gold is a STRICT subset of context: 1 gold chunk + 3 distractors."""
    return _rag("When was the Eiffel Tower completed?",
                [Chunk(GOLD_TEXT, 0.9, "gold", gold=True)] + _fillers(3))


def _battery() -> list[Inference]:
    """Varied prompts/contexts so per-feature variation is observable."""
    gold = Chunk(GOLD_TEXT, 0.9, "gold", gold=True)
    return [
        _rag("When was the Eiffel Tower completed?", [gold]),
        _subset_inference(),
        _rag("Who was Marie Curie?",
             [Chunk("Marie Curie was born in 1867.", 0.8, "g", gold=True),
              Chunk("Paris is in France.", 0.4, "d")], answer="1867"),
        _rag("Where is Paris?", [Chunk("Paris is in France.", 0.9, "g", gold=True)],
             answer="France"),
        _rag("Who signed the treaty?",
             [Chunk("The treaty was signed. " * 8, 0.6, "g", gold=True)] + _fillers(2),
             answer="treaty"),
    ]


MECH_GREY = ("context_attention_ratio", "gold_attention_ratio",
             "logit_lens_answer_layer", "logit_lens_stability",
             "external_context_score", "parametric_knowledge_score")


# --------------------------------------------------------------------------- #
# Capture: grey and white complete; the pipeline populates every feature.
# --------------------------------------------------------------------------- #
def test_grey_and_white_capture_complete(hf_white):
    inf = _subset_inference()
    grey = hf_white.with_tier(Tier.GREY).capture(inf)
    white = hf_white.capture(inf)
    for cap in (grey, white):
        for name in MECH_GREY:
            assert getattr(cap, name) is not None, f"{name} missing"
    assert grey.n_layers == 4


def test_pipeline_all_mechanistic_features_nonnull_at_white(hf_white):
    from tokentrace.signals.registry import SignalPipeline

    fv = SignalPipeline().run(_subset_inference(), hf_white)
    for name in MECH_GREY + ("gold_patch_effect",):
        assert fv.values.get(name) is not None, f"{name} missing from pipeline"
    # The confidence family must also extract against the real backend
    # (teacher-forced logprobs + resampling), not just the mock.
    for name in ("answer_perplexity", "mean_token_entropy", "max_token_entropy",
                 "semantic_entropy", "self_consistency"):
        assert fv.values.get(name) is not None, f"{name} missing from pipeline"


def test_capture_completes_without_rag_context(hf_white):
    cap = hf_white.capture(Inference(prompt="Where is Paris?",
                                     generated_answer="France"))
    assert cap.context_attention_ratio == 0.0
    assert cap.gold_attention_ratio is None
    assert cap.gold_patch_effect is None  # no gold chunk to ablate


# --------------------------------------------------------------------------- #
# The two aliasing regressions this session found on the first-ever real run.
# --------------------------------------------------------------------------- #
def test_gold_attention_is_not_context_attention(hf_white):
    cap = hf_white.capture(_subset_inference())
    # Gold positions are a strict subset of context positions, and softmax
    # attention puts nonzero mass on the distractors, so gold mass must be
    # strictly below context mass. Equality = the historical aliasing bug.
    assert cap.gold_attention_ratio is not None
    assert cap.gold_attention_ratio < cap.context_attention_ratio
    assert 0.0 <= cap.gold_attention_ratio <= 1.0
    assert 0.0 <= cap.context_attention_ratio <= 1.0


def test_external_score_is_not_context_attention_ratio(hf_white):
    """external_context_score must be its own measurement (mass x copy
    alignment), not a rename of context_attention_ratio: on the first real run
    the two were byte-identical across every configuration tried."""
    pairs = [(c.external_context_score, c.context_attention_ratio)
             for c in (hf_white.capture(i) for i in _battery())]
    assert all(0.0 <= e <= 1.0 for e, _ in pairs)
    # Definitional invariant: copying FROM context (mass x alignment) can never
    # exceed the total attention mass ON context — alignment only discounts.
    assert all(e <= c + 1e-3 for e, c in pairs), pairs
    differing = sum(1 for e, c in pairs if e != c)
    assert differing >= len(pairs) - 1, (
        f"external aliases context attention in {len(pairs) - differing}/{len(pairs)} "
        f"configurations: {pairs}"
    )


def test_parametric_score_not_structurally_pinned(hf_white):
    """The old formula multiplied by (1 - answer_layer), pinning the score to
    EXACTLY 0.0 whenever the lens said the answer forms at the final layer —
    which is the typical case. The FFN-share score must stay measurable there,
    and must vary across inputs rather than being a constant."""
    caps = [hf_white.capture(i) for i in _battery()]
    params = [c.parametric_knowledge_score for c in caps]
    assert all(p is not None for p in params), params
    assert all(0.0 <= p <= 1.0 for p in params), params
    assert len(set(params)) > 1, f"parametric constant across battery: {params}"
    late = [c.parametric_knowledge_score for c in caps
            if c.logit_lens_answer_layer == 1.0]
    assert late, "battery lost its answer_layer==1.0 cases; extend it"
    assert any(p != 0.0 for p in late), (
        f"parametric pinned to 0.0 whenever answer_layer==1.0: {late}"
    )


# --------------------------------------------------------------------------- #
# Generation paths (confidence-family inputs) against the real backend.
# --------------------------------------------------------------------------- #
def test_generate_and_sample_paths_run(hf_white):
    gen = hf_white.generate("Where is Paris?", max_tokens=6)
    assert isinstance(gen.text, str)
    assert gen.has_logprobs
    assert len(gen.token_logprobs) == len(gen.token_entropies) == len(gen.token_texts)
    assert all(e >= 0.0 for e in gen.token_entropies)

    samples = hf_white.sample("Where is Paris?", k=2, temperature=1.0)
    assert len(samples) == 2
    assert all(isinstance(s, str) for s in samples)


def test_confidence_teacher_forces_the_given_answer(hf_white):
    conf = hf_white.confidence("When was the Eiffel Tower completed?", answer="1889")
    assert conf.text == "1889"
    assert conf.has_logprobs
    # Scores the given answer's tokens, not a fresh 64-token continuation.
    n_ans = len(hf_white.tokenizer(" 1889", add_special_tokens=False)["input_ids"])
    assert len(conf.token_logprobs) == n_ans


# --------------------------------------------------------------------------- #
# Tier gating of the causal white-box feature.
# --------------------------------------------------------------------------- #
def test_gold_patch_effect_white_only(hf_white):
    inf = _subset_inference()
    assert hf_white.capture(inf).gold_patch_effect is not None
    assert hf_white.with_tier(Tier.GREY).capture(inf).gold_patch_effect is None


def test_white_capture_survives_gold_chunk_being_the_whole_prompt(hf_white):
    """Regression: ablating the gold chunk of a one-chunk trace leaves an EMPTY
    prompt, and a transformer cannot run a zero-length forward — capture died in
    a tensor reshape (RuntimeError, not TierUnavailable, so the whole diagnosis
    aborted). The causal contrast is UNDEFINED there, not zero: the feature must
    come back missing, because "we could not measure the ablation" and "ablating
    gold changed nothing" argue opposite things about groundedness.
    """
    gold_text = "The Eiffel Tower was completed in 1889."
    inf = Inference(
        prompt=gold_text,
        generated_answer="1889",
        retrieved_context=[Chunk(gold_text, 0.9, "g", gold=True)],
    )
    cap = hf_white.capture(inf)          # must not raise
    assert cap.gold_patch_effect is None, (
        "an unmeasurable ablation must be a MISSING feature, not a number")


# --------------------------------------------------------------------------- #
# NNsight backend (skips when nnsight is not installed).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def nn_white(tiny_model_dir):
    pytest.importorskip("nnsight")
    from tokentrace.models.nnsight import NNsightModel

    profile = ModelProfile(name="tiny-dev-nn", n_layers=4, n_heads=4, d_model=128,
                           max_tier=Tier.WHITE)
    return NNsightModel(profile, hf_repo=tiny_model_dir, tier=Tier.WHITE)


def test_nnsight_capture_complete_and_not_aliased(nn_white):
    cap = nn_white.capture(_subset_inference())
    for name in MECH_GREY + ("gold_patch_effect",):
        assert getattr(cap, name) is not None, f"{name} missing"
    assert cap.external_context_score != cap.context_attention_ratio
    assert cap.gold_attention_ratio < cap.context_attention_ratio


def test_nnsight_grey_features_match_hf(nn_white, hf_white):
    """Same fp32 weights, same definitions -> the two backends must read the
    same grey-box numbers (the causal white-box feature legitimately differs:
    input ablation vs activation patching)."""
    inf = _subset_inference()
    nn_cap = nn_white.capture(inf)
    hf_cap = hf_white.capture(inf)
    for name in MECH_GREY:
        a, b = getattr(nn_cap, name), getattr(hf_cap, name)
        assert a == pytest.approx(b, abs=5e-3), f"{name}: nnsight {a} != hf {b}"


def test_nnsight_confidence_scores_given_answer(nn_white):
    conf = nn_white.confidence("When was the Eiffel Tower completed?", answer="1889")
    assert conf.text == "1889"
    assert conf.has_logprobs
