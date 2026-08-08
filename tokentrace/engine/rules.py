"""Likelihood-ratio rule layer — the interpretable log-odds *prior*.

Each rule is a soft predicate over named signals contributing ``weight * activation``
log-odds toward ONE failure mode. A rule declares ``required`` signals and
**abstains (contributes 0) when any are missing** — this is what makes tier
degradation automatic: a rule that needs mechanistic evidence simply drops out at
black-box, and the prior uses only the evidence actually present.

The learned residual (see ``classifier.py``) is added on top of this prior, so the
final score is one additive, decomposable log-odds ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from tokentrace.core.types import ALL_MODES, FailureMode, FeatureVector, SignalFamily

M = FailureMode

#: Prior logit per mode (multi-label; base rate ~0.2 => logit ~= -1.4).
PRIOR_LOGIT = -1.4


# --- smooth ramps so contributions are graded, not hard step functions --- #
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def up(x: Optional[float], lo: float, hi: float) -> float:
    """0 below lo, ramps to 1 at hi (activation for 'high x')."""
    if x is None or hi == lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo))


def down(x: Optional[float], lo: float, hi: float) -> float:
    """1 below lo, ramps to 0 at hi (activation for 'low x')."""
    if x is None:
        return 0.0
    return 1.0 - up(x, lo, hi)


@dataclass
class Rule:
    id: str
    mode: FailureMode
    required: tuple[str, ...]
    weight: float
    family: SignalFamily
    predicate: Callable[[FeatureVector], float]   # -> activation, usually [0,1]
    explain: Callable[[FeatureVector], str]

    def evaluate(self, f: FeatureVector) -> Optional[tuple[float, str]]:
        if any(not f.has(r) for r in self.required):
            return None
        act = self.predicate(f)
        contrib = self.weight * act
        if abs(contrib) < 1e-6:
            return None
        return contrib, self.explain(f)


def g(f: FeatureVector, name: str, default: float = 0.0) -> float:
    v = f.get(name)
    return default if v is None else v


# --------------------------------------------------------------------------- #
# The rule set
# --------------------------------------------------------------------------- #
RULES: list[Rule] = [
    # ---------------- Prompt Ambiguity ---------------- #
    Rule("amb.signal", M.PROMPT_AMBIGUITY, ("prompt_ambiguity",), 3.0, SignalFamily.PROMPT,
         lambda f: up(g(f, "prompt_ambiguity"), 0.4, 0.9) - 0.4 * down(g(f, "prompt_ambiguity"), 0.1, 0.4),
         lambda f: f"prompt ambiguity = {g(f,'prompt_ambiguity'):.2f}"),
    Rule("amb.entropy_valid", M.PROMPT_AMBIGUITY, ("semantic_entropy", "prompt_ambiguity"), 1.2,
         SignalFamily.CONFIDENCE,
         lambda f: up(g(f, "semantic_entropy"), 0.3, 0.8) * up(g(f, "prompt_ambiguity"), 0.4, 0.9),
         lambda f: f"high answer spread (semantic entropy {g(f,'semantic_entropy'):.2f}) on an ambiguous prompt"),

    # ---------------- Retrieval Failure ---------------- #
    # GT discriminator: the correct answer is not even present in retrieved context.
    Rule("ret.gold_absent", M.RETRIEVAL_FAILURE, ("gold_recall_in_context",), 2.4,
         SignalFamily.RETRIEVAL,
         lambda f: down(g(f, "gold_recall_in_context"), 0.3, 0.6),
         lambda f: f"answer not recoverable from context (gold recall {g(f,'gold_recall_in_context'):.2f} < 0.3)"),
    # Reference-free fallback (works black-box / production): nothing relevant retrieved
    # and the answer is unsupported and unstable.
    Rule("ret.irrelevant", M.RETRIEVAL_FAILURE, ("max_chunk_relevance", "answer_supported_by_context"), 1.6,
         SignalFamily.RETRIEVAL,
         lambda f: down(g(f, "max_chunk_relevance"), 0.1, 0.35) * down(g(f, "answer_supported_by_context"), 0.2, 0.5),
         lambda f: f"weak retrieval (max relevance {g(f,'max_chunk_relevance'):.2f}) and unsupported answer"),

    # ---------------- Context Dilution ---------------- #
    # Core signature: the correct answer IS present but the model got it wrong.
    Rule("dil.present_wrong", M.CONTEXT_DILUTION, ("gold_recall_in_context", "is_correct"), 2.0,
         SignalFamily.RETRIEVAL,
         lambda f: up(g(f, "gold_recall_in_context"), 0.5, 0.9) * down(g(f, "is_correct"), 0.5, 1.0),
         lambda f: "correct answer present in context but not used (answer wrong)"),
    # Geometry: gold buried in the middle of a long, many-chunk context.
    Rule("dil.buried", M.CONTEXT_DILUTION, ("gold_position_frac", "n_chunks", "context_length_tokens"), 1.4,
         SignalFamily.RETRIEVAL,
         # NB: the token ramp is scaled to the contexts the pipeline actually builds.
         # It previously started at 200 while the corpus maxed out at 162, so this
         # rule fired 0/783 times and its log-odds never reached the learned head.
         lambda f: (1 - abs(g(f, "gold_position_frac") - 0.5) * 2) * up(g(f, "n_chunks"), 4, 10)
                   * up(g(f, "context_length_tokens"), 60, 400),
         lambda f: f"gold chunk buried mid-context (pos {g(f,'gold_position_frac'):.2f}, {int(g(f,'n_chunks'))} chunks)"),
    # Mechanistic confirmation: model did not attend to the (present) gold chunk.
    Rule("dil.gold_unattended", M.CONTEXT_DILUTION, ("gold_attention_ratio", "gold_recall_in_context"), 1.6,
         SignalFamily.MECHANISTIC,
         lambda f: down(g(f, "gold_attention_ratio"), 0.1, 0.3) * up(g(f, "gold_recall_in_context"), 0.5, 0.9),
         lambda f: f"model under-attended to the present gold chunk (attention {g(f,'gold_attention_ratio'):.2f})"),

    # Model DID attend to the gold chunk -> the info was seen, so a wrong answer is
    # mis-use (reasoning) or override (hallucination), not dilution.
    Rule("dil.attended", M.CONTEXT_DILUTION, ("gold_attention_ratio",), -1.4, SignalFamily.MECHANISTIC,
         lambda f: up(g(f, "gold_attention_ratio"), 0.3, 0.55),
         lambda f: f"model attended to the gold chunk ({g(f,'gold_attention_ratio'):.2f}) — argues against dilution"),

    # ---------------- Hallucination ---------------- #
    # ReDeEP signature: answer unsupported by context, high parametric push, low external read.
    Rule("hal.redeep", M.HALLUCINATION, ("answer_supported_by_context", "parametric_knowledge_score", "external_context_score"),
         2.2, SignalFamily.MECHANISTIC,
         lambda f: down(g(f, "answer_supported_by_context"), 0.2, 0.5)
                   * up(g(f, "parametric_knowledge_score"), 0.5, 0.85)
                   * down(g(f, "external_context_score"), 0.2, 0.5),
         lambda f: f"unsupported answer with high parametric / low context read (ReDeEP: param {g(f,'parametric_knowledge_score'):.2f}, ext {g(f,'external_context_score'):.2f})"),
    # Reference-free: unsupported + confabulation (high semantic entropy).
    Rule("hal.confab", M.HALLUCINATION, ("answer_supported_by_context", "semantic_entropy"), 1.5,
         SignalFamily.CONFIDENCE,
         lambda f: down(g(f, "answer_supported_by_context"), 0.2, 0.5) * up(g(f, "semantic_entropy"), 0.3, 0.8),
         lambda f: f"unsupported answer with high answer spread (semantic entropy {g(f,'semantic_entropy'):.2f})"),
    # PARAMETRIC OVERRIDE: the gold was retrieved AND attended, yet a parametric
    # belief won. Mechanically the mirror image of dilution (which has LOW gold
    # attention and a LATE answer), so this is the discrimination that genuinely
    # requires mechanistic evidence.
    Rule("hal.override", M.HALLUCINATION,
         ("gold_recall_in_context", "gold_attention_ratio", "parametric_knowledge_score",
          "logit_lens_answer_layer"), 2.4, SignalFamily.MECHANISTIC,
         lambda f: up(g(f, "gold_recall_in_context"), 0.5, 0.9)
                   * up(g(f, "gold_attention_ratio"), 0.25, 0.45)
                   * up(g(f, "parametric_knowledge_score"), 0.6, 0.85)
                   * down(g(f, "logit_lens_answer_layer"), 0.3, 0.55),
         lambda f: (f"gold was retrieved and attended ({g(f,'gold_attention_ratio'):.2f}) but the "
                    f"answer formed early from parametric memory "
                    f"(param {g(f,'parametric_knowledge_score'):.2f}, layer "
                    f"{g(f,'logit_lens_answer_layer'):.2f}) — parametric override")),
    # WHITE-exclusive causal confirmation of the override. The gold chunk was
    # retrieved AND attended AND the answer is wrong, yet input-ablating the gold
    # tokens barely moves the answer (LOW gold_patch_effect): the model was not
    # actually USING the context, so the wrong answer came from parametric memory.
    # Direction fixed by measurement on the mock corpus, not by intuition: patch
    # effect is LOW for override (0.05) — and ALSO low for dilution (a diluted
    # model is not using the gold either), so the attention + parametric terms
    # carry the override-vs-dilution split while the patch term supplies the
    # causal, white-only confirmation (it separates override from grounded ~0.8
    # and ambiguous ~0.3 traces, where the context IS causally read).
    # ``required`` includes gold_patch_effect, so this rule ABSTAINS below WHITE —
    # that abstention is the tier-degradation mechanism that makes white vs grey
    # distinct experiments (they were previously bit-identical).
    Rule("hal.override_causal", M.HALLUCINATION,
         ("gold_recall_in_context", "gold_attention_ratio", "parametric_knowledge_score",
          "is_correct", "gold_patch_effect"), 1.6, SignalFamily.MECHANISTIC,
         lambda f: up(g(f, "gold_recall_in_context"), 0.5, 0.9)
                   * up(g(f, "gold_attention_ratio"), 0.25, 0.45)
                   * up(g(f, "parametric_knowledge_score"), 0.55, 0.8)
                   * down(g(f, "is_correct"), 0.5, 1.0)
                   * down(g(f, "gold_patch_effect"), 0.15, 0.45),
         lambda f: (f"ablating the attended gold chunk barely changes the answer "
                    f"(patch effect {g(f,'gold_patch_effect'):.2f} with attention "
                    f"{g(f,'gold_attention_ratio'):.2f}) — causal evidence the wrong "
                    f"answer is parametric, not context-driven")),
    # Non-RAG confabulation: hallucination must be diagnosable with NO retrieval
    # signals at all (all other hallucination rules require answer_supported_by_context,
    # which only exists for RAG inputs, so cold-start called every non-RAG
    # fabrication "healthy").
    Rule("hal.confab_nonrag", M.HALLUCINATION, ("semantic_entropy", "self_consistency"), 1.8,
         SignalFamily.CONFIDENCE,
         lambda f: up(g(f, "semantic_entropy"), 0.35, 0.8) * down(g(f, "self_consistency"), 0.3, 0.7),
         lambda f: (f"answers disagree across resamples (semantic entropy "
                    f"{g(f,'semantic_entropy'):.2f}, self-consistency "
                    f"{g(f,'self_consistency'):.2f}) with no retrieval to ground them")),

    # ---------------- Reasoning Failure ---------------- #
    # Info present, multi-hop question, wrong composite answer.
    Rule("reason.hops_present", M.REASONING_FAILURE, ("multihop_query", "gold_recall_in_context", "is_correct"), 2.0,
         SignalFamily.PROMPT,
         lambda f: g(f, "multihop_query") * up(g(f, "gold_recall_in_context"), 0.5, 0.9)
                   * down(g(f, "is_correct"), 0.5, 1.0),
         lambda f: "multi-hop question with facts present but wrong composite answer"),
    # Mechanistic: answer forms very late and unstably (aggregation failure).
    Rule("reason.late_unstable", M.REASONING_FAILURE, ("logit_lens_answer_layer", "logit_lens_stability"), 1.5,
         SignalFamily.MECHANISTIC,
         lambda f: up(g(f, "logit_lens_answer_layer"), 0.75, 0.95) * down(g(f, "logit_lens_stability"), 0.3, 0.6),
         lambda f: f"answer forms late and unstably across layers (layer {g(f,'logit_lens_answer_layer'):.2f}, stability {g(f,'logit_lens_stability'):.2f})"),

    # ---------------- Ambiguity suppressors ---------------- #
    # A strongly ambiguous prompt is an upstream alternative explanation for a
    # wrong answer, so it lowers the evidence for dilution / reasoning (which
    # otherwise also fire on "info present but answer wrong").
    Rule("amb.explains_dilution", M.CONTEXT_DILUTION, ("prompt_ambiguity",), -1.8, SignalFamily.PROMPT,
         lambda f: up(g(f, "prompt_ambiguity"), 0.55, 0.9),
         lambda f: f"wrong answer plausibly explained by prompt ambiguity ({g(f,'prompt_ambiguity'):.2f})"),
    Rule("amb.explains_reasoning", M.REASONING_FAILURE, ("prompt_ambiguity",), -1.2, SignalFamily.PROMPT,
         lambda f: up(g(f, "prompt_ambiguity"), 0.55, 0.9),
         lambda f: f"wrong answer plausibly explained by prompt ambiguity ({g(f,'prompt_ambiguity'):.2f})"),

    # ---------------- Correctness suppressors ---------------- #
    # A correct final answer argues against the modes defined by a wrong answer.
    Rule("ok.suppress_dilution", M.CONTEXT_DILUTION, ("is_correct",), -1.6, SignalFamily.PROMPT,
         lambda f: up(g(f, "is_correct"), 0.5, 1.0), lambda f: "answer is correct"),
    Rule("ok.suppress_reasoning", M.REASONING_FAILURE, ("is_correct",), -1.6, SignalFamily.PROMPT,
         lambda f: up(g(f, "is_correct"), 0.5, 1.0), lambda f: "answer is correct"),
    Rule("ok.suppress_halluc", M.HALLUCINATION, ("is_correct", "answer_supported_by_context"), -1.0,
         SignalFamily.PROMPT,
         lambda f: up(g(f, "is_correct"), 0.5, 1.0) * up(g(f, "answer_supported_by_context"), 0.5, 0.9),
         lambda f: "answer is correct and supported by context"),
]


def evaluate_rules(f: FeatureVector) -> tuple[dict[FailureMode, float], list[tuple[Rule, float, str]]]:
    """Return per-mode prior log-odds and the list of fired (rule, contribution, text)."""
    logits: dict[FailureMode, float] = {m: PRIOR_LOGIT for m in ALL_MODES}
    fired: list[tuple[Rule, float, str]] = []
    for rule in RULES:
        res = rule.evaluate(f)
        if res is None:
            continue
        contrib, text = res
        logits[rule.mode] += contrib
        fired.append((rule, contrib, text))
    return logits, fired
