"""Core data contracts for TokenTrace.

Everything in the framework hangs off these types. They are intentionally
dependency-light (standard-library dataclasses + enums only) so that importing
the contracts never pulls in torch/transformers/faiss. Numpy is imported lazily
inside the one method that needs it.

Design note — the *missingness mask* on :class:`FeatureVector` is the load-bearing
idea: tiers (black/grey/white) and most edge cases are expressed as "which signal
families are present", not as bespoke branches scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid importing numpy at module load
    import numpy as np


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Tier(IntEnum):
    """Operating tier = how much of the model we can observe.

    Ordered so that ``Tier.WHITE >= Tier.GREY >= Tier.BLACK``. This maps onto
    what is *CPU-affordable*, not onto whether a GPU is present:

    * BLACK  — input/output only (API-style). Prompt + retrieval + resampled confidence.
    * GREY   — logprobs + hidden states + attention from a single HF forward pass.
    * WHITE  — activation patching / causal interventions.
    """

    BLACK = 0
    GREY = 1
    WHITE = 2

    @property
    def label(self) -> str:
        return self.name.lower()


class FailureMode(str, Enum):
    """The five diagnosable root-cause failure modes."""

    PROMPT_AMBIGUITY = "prompt_ambiguity"
    RETRIEVAL_FAILURE = "retrieval_failure"
    CONTEXT_DILUTION = "context_dilution"
    HALLUCINATION = "hallucination"
    REASONING_FAILURE = "reasoning_failure"

    @property
    def pretty(self) -> str:
        return self.value.replace("_", " ").title()


#: Canonical ordering used everywhere a fixed 5-vector is needed.
ALL_MODES: tuple[FailureMode, ...] = (
    FailureMode.PROMPT_AMBIGUITY,
    FailureMode.RETRIEVAL_FAILURE,
    FailureMode.CONTEXT_DILUTION,
    FailureMode.HALLUCINATION,
    FailureMode.REASONING_FAILURE,
)


class SignalFamily(str, Enum):
    PROMPT = "prompt"
    RETRIEVAL = "retrieval"
    MECHANISTIC = "mechanistic"
    CONFIDENCE = "confidence"


class Provenance(str, Enum):
    """Where a label came from — governs how it may be used (train vs eval)."""

    REAL = "real"          # RAGTruth human labels + human-audited seed -> HEADLINE EVAL ONLY
    SEMI = "semi"          # real outputs, auto-derived weighted labels -> train
    SYNTHETIC = "synthetic"  # injection harness -> train


class DiagnosisRole(str, Enum):
    PRIMARY_ROOT = "primary_root"
    CONTRIBUTING = "contributing"
    SEQUELA = "sequela"


# --------------------------------------------------------------------------- #
# Inference (the unit of analysis)
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A retrieved context chunk."""

    text: str
    retriever_score: float = 0.0
    source_id: str = ""
    gold: Optional[bool] = None            # None = unknown whether it bears the answer
    char_span: Optional[tuple[int, int]] = None


@dataclass
class Inference:
    """One LLM inference to be diagnosed.

    ``retrieved_context is None`` marks a non-RAG inference (retrieval-related
    modes are then hard-masked). ``ground_truth is None`` marks a production
    setting with no reference — diagnoses become *risk* estimates rather than
    confirmed errors. ``ground_truth`` is a *set* of acceptable answers so that
    genuinely ambiguous questions are scored fairly.
    """

    prompt: str
    generated_answer: str
    retrieved_context: Optional[list[Chunk]] = None
    ground_truth: Optional[list[str]] = None
    question: Optional[str] = None          # raw user question if distinct from full prompt
    meta: dict[str, Any] = field(default_factory=dict)  # lang, modality, task_type, model_name...
    id: Optional[str] = None

    @property
    def is_rag(self) -> bool:
        return self.retrieved_context is not None

    @property
    def has_ground_truth(self) -> bool:
        """True only when a USABLE reference is present.

        ``[]`` and ``[""]`` must count as absent: they arrive routinely from a
        filtered list or a JSON payload, and treating them as present made the
        scorers return a real 0.0 (rather than "missing"), which diagnosed a
        perfectly grounded, correct answer as a confident retrieval failure.
        """
        if not self.ground_truth:
            return False
        return any(g and g.strip() for g in self.ground_truth)

    @property
    def query(self) -> str:
        """Best-available user intent string for prompt/retrieval analysis."""
        return self.question or self.prompt


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass
class FeatureVector:
    """Named signal features + an explicit record of what is *absent*.

    A feature absent because of the tier / non-RAG / OOM is recorded in
    ``missing`` and rendered as NaN in :meth:`to_array` so the gradient-boosted
    heads route around it natively.
    """

    values: dict[str, float] = field(default_factory=dict)
    missing: set[str] = field(default_factory=set)
    family_present: dict[SignalFamily, bool] = field(default_factory=dict)
    #: Whether a usable ground-truth reference was available when these features were
    #: extracted. This is part of the missingness signature because the GT-dependent
    #: features (is_correct, gold_recall_in_context) live INSIDE already-present
    #: families, so without it a production (no-reference) trace would silently be
    #: served a calibration map fitted only on reference-bearing data.
    reference_available: bool = True

    def get(self, name: str) -> Optional[float]:
        if name in self.missing:
            return None
        return self.values.get(name)

    def set(self, name: str, value: Optional[float], family: Optional[SignalFamily] = None) -> None:
        if value is None:
            self.missing.add(name)
            self.values.pop(name, None)
        else:
            self.values[name] = float(value)
            self.missing.discard(name)
        # A family counts as "present" only once it contributes a real value.
        # A missing feature must never flip its family to present, or the
        # missingness signature (and thus calibration-map selection) is wrong.
        if family is not None and value is not None:
            self.family_present[family] = True

    def has(self, name: str) -> bool:
        return name in self.values and name not in self.missing

    def missingness_signature(self) -> str:
        """Stable key identifying which families are present.

        Used to select the matching per-missingness calibration map so that
        probabilities stay meaningful under degradation.
        """
        present = [f.value for f in SignalFamily if self.family_present.get(f)]
        sig = "+".join(sorted(present)) or "none"
        return sig if self.reference_available else f"{sig}|noref"

    def to_array(self, feature_names: list[str]) -> "np.ndarray":
        import numpy as np

        row = np.full(len(feature_names), np.nan, dtype=np.float64)
        for i, name in enumerate(feature_names):
            if name in self.values and name not in self.missing:
                row[i] = self.values[name]
        return row


# --------------------------------------------------------------------------- #
# Evidence & diagnoses
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceItem:
    """One line in a diagnosis's additive log-odds ledger.

    ``contribution_logodds`` is directly comparable and summable across rule and
    learned (TreeSHAP) contributions because both live in log-odds space.
    ``provenance`` points back to the raw artifact (chunk id, (layer, head),
    resampled answers) so a developer can audit the finding.
    """

    signal: str
    family: SignalFamily
    value: float
    contribution_logodds: float
    direction: str = "supports"           # "supports" | "opposes"
    source: str = "rule"                  # "rule" | "learned"
    provenance: dict[str, Any] = field(default_factory=dict)
    rendered: str = ""


@dataclass
class Recommendation:
    """A corrective action, optionally validated by a simulated intervention."""

    action: str                           # machine key, e.g. "add_gold_context"
    description: str
    targets_mode: FailureMode
    priority: int = 0                     # lower = do first (treat the root)
    validated: Optional[bool] = None      # set after simulated intervention
    validation_detail: str = ""


@dataclass
class Diagnosis:
    mode: FailureMode
    probability: float                    # calibrated marginal P(mode present)
    rank: int = 0
    role: Optional[DiagnosisRole] = None
    causal_parents: list[FailureMode] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0               # per-diagnosis meta-confidence
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def headline_evidence(self) -> "Optional[EvidenceItem]":
        """Strongest SUBSTANTIVE evidence line.

        The ledger is magnitude-ordered and includes bookkeeping rows (the base-rate
        prior and the aggregated remainder) so that it reconciles to the total
        log-odds. Those rows are not explanations, so anything rendering "the reason"
        should use this rather than ``evidence[0]`` — otherwise a cold-start
        diagnosis headlines "base rate / prior".
        """
        for e in self.evidence:
            if e.source not in ("prior", "aggregate"):
                return e
        return self.evidence[0] if self.evidence else None


@dataclass
class DiagnosisReport:
    """The full ranked output for one inference."""

    diagnoses: list[Diagnosis] = field(default_factory=list)   # ranked, all 5 modes
    tier: Tier = Tier.BLACK
    inference_id: Optional[str] = None
    conformal_set: list[FailureMode] = field(default_factory=list)
    diagnostic_confidence: float = 0.0
    abstained: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def primary(self) -> Optional[Diagnosis]:
        for d in self.diagnoses:
            if d.role == DiagnosisRole.PRIMARY_ROOT:
                return d
        return self.diagnoses[0] if self.diagnoses else None

    def top_k(self, k: int = 3) -> list[Diagnosis]:
        return self.diagnoses[:k]

    def ranked_modes(self) -> list[FailureMode]:
        return [d.mode for d in self.diagnoses]


# --------------------------------------------------------------------------- #
# Supervision
# --------------------------------------------------------------------------- #
@dataclass
class LabeledInference:
    """An inference with ground-truth failure-mode labels for train/eval.

    Injection naturally yields multi-label + causal-edge supervision, which the
    public single-label datasets do not provide.
    """

    inference: Inference
    labels: list[FailureMode] = field(default_factory=list)
    causal_edges: list[tuple[FailureMode, FailureMode]] = field(default_factory=list)
    severity: dict[FailureMode, float] = field(default_factory=dict)
    provenance: Provenance = Provenance.SYNTHETIC
    injection_recipe: Optional[str] = None
    verification: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def label_vector(self) -> list[int]:
        """Binary indicator over :data:`ALL_MODES` (multi-label)."""
        s = set(self.labels)
        return [1 if m in s else 0 for m in ALL_MODES]


# --------------------------------------------------------------------------- #
# Model metadata
# --------------------------------------------------------------------------- #
@dataclass
class ModelProfile:
    """Static facts about a model so mechanistic features are consumed
    *model-relative* (depth as a fraction of layers, attention ratios normalized
    to a per-model baseline) and comparable across architectures."""

    name: str
    n_layers: int = 0
    n_heads: int = 0
    d_model: int = 0
    max_tier: Tier = Tier.WHITE           # cap if grey/white capture unsupported
    tokenizer_quirks: dict[str, Any] = field(default_factory=dict)
    lost_in_middle_baseline: Optional[list[float]] = None   # expected recall vs position
    retrieval_heads: list[tuple[int, int]] = field(default_factory=list)  # (layer, head)
    activation_baseline: dict[str, float] = field(default_factory=dict)
    gguf_repo: Optional[str] = None       # HF repo for the quantized generation weights
    hf_repo: Optional[str] = None         # HF repo for the full weights (capture)
