# TokenTrace — Methodology & Results (synthetic benchmark)

> **Scope & honesty note.** Every number below is on a **disjoint synthetic test
> split** produced by the deterministic mock model + failure-injection harness, with
> no model or dataset downloads. These results validate the *method and machinery*
> end-to-end (signal extraction → calibrated ranked diagnosis → causal ordering →
> validated fix → graceful degradation) and let the ablations isolate each
> component's contribution. They are **not** real-data performance. Headline numbers
> on real data require the real backends + datasets (Qwen3-4B / RAGTruth / HotpotQA),
> which need HuggingFace access; the code paths are wired (`models/{gguf,hf,nnsight}`,
> `data/loaders`, `retrieval/rag`) and audited but not yet run.

Reproduce: `tokentrace ablate` (or `tokentrace eval`).

## 1. Method (recap)

TokenTrace diagnoses *why* an LLM inference is wrong across five failure modes
(Prompt Ambiguity, Retrieval Failure, Context Dilution, Hallucination, Reasoning
Failure) by correlating four signal families (prompt, retrieval, mechanistic,
confidence) into calibrated, ranked root-cause diagnoses with an auditable evidence
ledger and a corrective recommendation validated by a simulated intervention.

- **Diagnosis engine:** interpretable likelihood-ratio **rules** produce a log-odds
  prior; a **LightGBM residual** (rule log-odds as `init_score`) fits only the
  correction, so the total is one additive ledger `z = prior + rule_Σ + shap_base +
  shap_Σ`. Cold-start (no data) runs on pure rules.
- **Calibration:** isotonic per `(mode, missingness-signature)`, fit across all tiers.
- **Sets:** APS conformal set (adaptive Top-k), excluding hard-masked modes.
- **Causal:** fixed DAG resolves root vs sequela; a dominant downstream mode stays
  primary unless a *comparably strong* parent is present.
- **Supervision:** deterministic failure-injection with verification gates + provenance
  tiering (synthetic → train, real → eval).

## 2. Experimental setup

- Corpus: injection harness over a 15-fact + 6-multihop seed pool, seeds 0–3, all
  verification gates enforced. Sizes: **train 205 / cal 68 / test 69** (hash-shuffled,
  disjoint). Multi-label + causal-edge supervision.
- Model: deterministic `mock-4b` (coherent generation + confidence + mechanistic
  proxies). Evaluated at white / grey / black tiers.
- Targets (from the proposal): diagnosis ≥ 0.80, Top-3 ≥ 0.90, recommendation
  precision ≥ 0.75.

## 3. Headline results (per tier)

| tier | diagnosis acc | Top-3 acc | conformal coverage | mean set size | abstention |
|---|---|---|---|---|---|
| white-box | **1.000** | 1.000 | 1.000 | 1.00 | 0.130 |
| grey-box  | **1.000** | 1.000 | 1.000 | 1.00 | 0.130 |
| black-box | **0.942** | 1.000 | 1.000 | 1.02 | 0.188 |

Top-3 here is the **true** metric (gold root among the 3 highest-probability modes),
decoupled from the conformal set and the abstention gate. Black-box loses the
mechanistic family, so diagnosis dips (0.942) and abstention rises (0.130→0.188) —
the tool becomes *less certain*, by design, rather than wrong.

## 4. Ablations

### 4.1 Learned head — rules-only vs rules ⊕ GBT residual (white-box)

| configuration | diagnosis acc | macro-F1 |
|---|---|---|
| rules only (cold-start) | 0.826 | 0.843 |
| rules ⊕ GBT residual | **1.000** | **1.000** |

The interpretable rules alone already clear the 0.80 target (0.826); the learned
residual sharpens them by **+0.17 accuracy** to 1.000. This is the hybrid design's
payoff: usable from day one with zero labels, and it improves monotonically as
labeled data arrives — without discarding the interpretable prior.

### 4.2 Per-signal-family ablation (drop one family, white-box)

Diagnosis accuracy and per-mode F1 when each family's extractors are removed:

| dropped family | diagnosis acc | retrieval F1 | dilution F1 | ambiguity F1 | reasoning F1 |
|---|---|---|---|---|---|
| — (all present) | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 |
| **retrieval** | **0.522** | 0.00 | 0.00 | 1.00 | 1.00 |
| **prompt** | 0.754 | 1.00 | 1.00 | 0.00 | 1.00 |
| mechanistic | 0.942 | 1.00 | 1.00 | 1.00 | 1.00 |
| confidence | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 |

This is the most informative result and it matches the design intent exactly:
- The **retrieval family is the most load-bearing** (accuracy → 0.522): removing it
  collapses **both** Retrieval Failure and Context Dilution, because
  `gold_recall_in_context` is the single decisive discriminator between them.
- The **prompt family carries Prompt Ambiguity** (its F1 → 0.00 when dropped) and
  nothing else — a clean, localized dependency.
- **Mechanistic** contributes at the margin on this (separable) synthetic set
  (accuracy → 0.942); its value is expected to grow on real, noisier data where the
  behavioral signals are weaker discriminators.
- **Confidence** is redundant here (accuracy unchanged) — the other families already
  separate the modes on clean synthetic data.

### 4.3 Calibration & conformal

Expected Calibration Error over all (example, mode) probabilities:
`uncalibrated ≈ 0.000`, `calibrated ≈ 0.000`. On a *perfectly separable* synthetic
corpus the raw scores are already near-perfectly calibrated, so isotonic calibration
is (correctly) a near-no-op — its value surfaces on noisy/real distributions and
under tier degradation, where the raw score distribution shifts. Conformal coverage
is 1.00 at every tier with mean set size ≈ 1.0 (the sets are confident singletons on
separable data; on real data they will widen on genuinely ambiguous cases).

### 4.4 Recommendations

Recommendation precision is **1.00** (n = 62 scored fixes), now honestly gated: only
fixes that target a *true* failure mode on a *non-abstained* report are counted.
**Caveat (from the review):** on the mock this is partly tautological — the simulated
`add_gold_context` fix injects the reference answer, which the mock then reads. Offline
it is best read as a *plumbing check* that the intervention loop works; real fix
efficacy must be measured against a real model.

## 5. Findings

1. The interpretable rules are strong on their own (0.826) and the learned residual
   closes the gap to 1.000 — validating the rules ⊕ residual hybrid.
2. The retrieval family is the backbone (carries 2 of 5 modes via the gold-recall
   discriminator); prompt carries ambiguity; mechanistic/confidence are supporting on
   synthetic data.
3. Degradation is graceful and *self-aware*: black-box drops accuracy modestly and
   raises abstention rather than producing confident wrong answers.

## 6. Threats to validity

- **Synthetic self-consistency.** The mock's signals are internally coherent by
  construction, so the modes are more separable than real data. This is why several
  metrics saturate at 1.0; the *ablations* (which measure relative contribution) are
  more meaningful than the absolute headline numbers.
- **Recommendation tautology** (§4.4).
- **Real backends unvalidated at runtime** (HF unreachable in the build env). They
  were audited by a three-reviewer correctness pass and the confirmed defects fixed
  (see the review-fixes commit), but first-real-run verification is pending.

## 7. Next steps

Run the same pipeline on Qwen3-4B (GGUF) with RAGTruth's human hallucination labels +
a small human-audited multi-label seed as the REAL evaluation split; report per-mode
precision/recall on real data; and run the debugging-time user study for the 30–40%
reduction target.
