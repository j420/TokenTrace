"""Known model profiles + a backend factory.

Profiles carry architecture facts so mechanistic features are consumed
model-relative. The numbers here are sensible defaults/hints; when a real HF
model is loaded the exact values (n_layers, n_heads, d_model) are read from its
config and overwrite these.

``max_tier`` records how far the mechanistic fallback ladder can go for a model.
If TransformerLens/hooks cannot reach a model, drop its ``max_tier`` to GREY and
the tier system handles the rest — no special-casing.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import ModelProfile, Tier
from tokentrace.models.base import ModelHandle
from tokentrace.models.mock import MockModel

# --------------------------------------------------------------------------- #
# Profiles for the proposal's three application models + small dev models.
# The small models are the ones you run white-box (activation patching) on CPU;
# the 4B models default to grey-box on CPU (single forward pass) and only need a
# GPU for optional full-4B patching.
# --------------------------------------------------------------------------- #
PROFILES: dict[str, ModelProfile] = {
    "mock-4b": ModelProfile(
        name="mock-4b", n_layers=32, n_heads=32, d_model=2560, max_tier=Tier.WHITE,
        lost_in_middle_baseline=[0.9, 0.75, 0.6, 0.72, 0.88],
    ),
    # --- application models (grey-box on CPU; white-box optional/GPU) --- #
    "qwen3-4b": ModelProfile(
        name="qwen3-4b", n_layers=36, n_heads=32, d_model=2560, max_tier=Tier.WHITE,
        gguf_repo="Qwen/Qwen3-4B-GGUF", hf_repo="Qwen/Qwen3-4B",
    ),
    "gemma3-4b": ModelProfile(
        name="gemma3-4b", n_layers=34, n_heads=16, d_model=2560,
        # Gemma 3 multimodal has known TransformerLens quirks -> cap capture at grey
        # until a clean bridge lands; grey-box (HF attentions/hidden states) is fine.
        max_tier=Tier.GREY,
        gguf_repo="google/gemma-3-4b-it-qat-q4_0-gguf", hf_repo="google/gemma-3-4b-it",
    ),
    "phi4-mini": ModelProfile(
        name="phi4-mini", n_layers=32, n_heads=24, d_model=3072, max_tier=Tier.WHITE,
        gguf_repo="microsoft/Phi-4-mini-instruct-gguf", hf_repo="microsoft/Phi-4-mini-instruct",
    ),
    # --- small dev models: fast white-box on CPU --- #
    "qwen3-0.6b": ModelProfile(
        name="qwen3-0.6b", n_layers=28, n_heads=16, d_model=1024, max_tier=Tier.WHITE,
        gguf_repo="Qwen/Qwen3-0.6B-GGUF", hf_repo="Qwen/Qwen3-0.6B",
    ),
    "qwen3-1.7b": ModelProfile(
        name="qwen3-1.7b", n_layers=28, n_heads=16, d_model=2048, max_tier=Tier.WHITE,
        gguf_repo="Qwen/Qwen3-1.7B-GGUF", hf_repo="Qwen/Qwen3-1.7B",
    ),
}


def get_profile(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown model '{name}'. known: {sorted(PROFILES)}")
    return PROFILES[name]


def load_model(
    name: str = "mock-4b",
    backend: str = "mock",
    tier: Tier = Tier.WHITE,
    **kwargs,
) -> ModelHandle:
    """Instantiate a :class:`ModelHandle`.

    backend:
      * ``"mock"``  — deterministic simulator, no downloads (default; the offline path).
      * ``"gguf"``  — quantized CPU generation + logprobs (grey-box). Needs ``llama-cpp-python``.
      * ``"hf"``    — HF forward-pass capture + generation (grey/white-box). Needs ``torch``+``transformers``.

    ``tier`` is clamped to the profile's ``max_tier``.
    """
    profile = get_profile(name) if name in PROFILES else ModelProfile(name=name, max_tier=tier)
    tier = Tier(min(int(tier), int(profile.max_tier)))

    if backend == "mock":
        return MockModel(profile=profile, tier=tier)
    if backend == "gguf":
        from tokentrace.models.gguf import GGUFModel

        return GGUFModel(profile=profile, tier=min(tier, Tier.GREY), **kwargs)
    if backend == "hf":
        from tokentrace.models.hf import HFModel

        return HFModel(profile=profile, tier=tier, **kwargs)
    if backend == "nnsight":
        from tokentrace.models.nnsight import NNsightModel

        return NNsightModel(profile=profile, tier=tier, **kwargs)
    raise ValueError(f"unknown backend '{backend}' (mock|gguf|hf|nnsight)")


def available_backends() -> dict[str, bool]:
    """Report which real backends are importable in this environment."""
    import importlib.util as u

    _hf = u.find_spec("torch") is not None and u.find_spec("transformers") is not None
    return {
        "mock": True,
        "gguf": u.find_spec("llama_cpp") is not None,
        "hf": _hf,
        "nnsight": _hf and u.find_spec("nnsight") is not None,
    }


def resolve_model_name(name: Optional[str]) -> str:
    return name or "mock-4b"
