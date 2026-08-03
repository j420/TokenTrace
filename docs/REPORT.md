# TokenTrace — Methodology & Results (synthetic benchmark)

> **Every number in this document is produced by `python3 scripts/regen_report.py`.**
> It is regenerated from code, not transcribed, because an earlier revision of this
> report contained figures that no longer corresponded to any code path. If you
> change anything that could move a metric, re-run that script and paste its output.
>
> **Scope.** These results come from a deterministic mock model and a synthetic
> failure-injection corpus, with no model or dataset downloads. They validate the
> *machinery* end-to-end (signal extraction → calibrated ranked diagnosis → causal
> ordering → validated fix → graceful degradation) and let the ablations isolate each
> component. They are **not** real-data performance, and §6 lists the specific ways
> this corpus is easier than reality.

Reproduce: `tokentrace eval`, `tokentrace ablate`, or `scripts/regen_report.py`
(all three now share one training schedule; they previously did not).

## 1. Method

TokenTrace diagnoses *why* an LLM inference is wrong across five failure modes
(Prompt Ambiguity, Retrieval Failure, Context Dilution, Hallucination, Reasoning
Failure) by correlating four signal families (prompt, retrieval, mechanistic,
confidence) into calibrated, ranked root-cause diagnoses with an auditable evidence
ledger and a corrective recommendation validated by a simulated intervention.

- **Engine:** interpretable likelihood-ratio **rules** produce a log-odds prior; a
  **LightGBM residual** (rule log-odds as `init_score`) fits only the correction, so
  the total is one additive ledger `z = prior + Σrules + shap_base + Σshap`, which
  the ledger is tested to reconcile to.
- **Calibration:** per `(mode, missingness-signature)`, **self-validating** — a map
  is adopted only if it beats the raw sigmoid on held-out Brier score, because on
  sharp, accurate scores any smoothing map makes probabilities strictly worse.
  Isotonic on large buckets, Platt on thin ones.
- **Sets:** APS conformal over *normalized* marginals (the engine emits independent
  one-vs-rest probabilities, which do not sum to 1).
- **Supervision:** deterministic failure injection with verification gates that
  **discard rather than mislabel**; all seven recipes currently pass 100% of gates.

## 2. Setup

- Corpus: **426 rows** from a 16-fact + 6-multihop seed pool, seeds 0–3, gates
  enforced. Split **255 train / 85 cal / 86 test** (hash-shuffled, disjoint).
- Labels present: retrieval_failure 112, hallucination 174, context_dilution 64,
  prompt_ambiguity 56, reasoning_failure 24, and 60 clean (no-failure) rows.
  Multi-label with causal edges.
- Model: deterministic `mock-4b`. Targets from the proposal: diagnosis ≥ 0.80,
  Top-3 ≥ 0.90, recommendation precision ≥ 0.75.

### 2.1 Shortcut audit (run before believing any number below)

A classifier given **only context shape** (chunk count, context length, prompt and
answer length — zero diagnostic content) scores:

| probe | held-out accuracy |
|---|---|
| shape-only features | **0.644** |
| majority-class prior | 0.356 |

An earlier version of this corpus scored **0.917** here: every recipe had a fixed,
unique chunk count, so most of the headline metric was recoverable by counting
chunks. Context geometry is now varied per row, chunk-count bands overlap across
recipes, and hallucination occurs both with and without retrieval. The **residual
0.644 is expected and partly legitimate** — context dilution *is* defined by
geometry (lost-in-the-middle), so a shape probe should find it — but it is also the
single largest caveat on the numbers below, and it is enforced by a standing test
(`test_labels_not_recoverable_from_context_shape`).

## 3. Headline results (per tier)

| tier | diagnosis | Top-3 | conformal cov. | set size | rec. precision (n, negatives) | healthy abstention | declined | debug-time↓ |
|---|---|---|---|---|---|---|---|---|
| white-box | 0.977 | 0.977 | 1.000 | 1.27 | 0.787 (75, 16) | 0.846 | 0.000 | 0.635 |
| grey-box | 0.977 | 0.977 | 1.000 | 1.27 | 0.787 (75, 16) | 0.846 | 0.000 | 0.635 |
| black-box | 0.988 | 1.000 | 1.000 | 1.27 | 0.784 (74, 16) | 1.000 | 0.014 | 0.616 |

All three proposal targets are met. Read the following caveats with the table:

- **The tiers are not distinguishable on this corpus, and white ≡ grey exactly.**
  `gold_patch_effect` is the only WHITE-exclusive feature; no rule uses it and it
  does not change the missingness signature, so the two tiers are the same
  experiment. Black-box scores marginally *higher* here, which is noise on 86 test
  rows, not evidence that less information helps. **This benchmark does not
  demonstrate the tier-degradation story**; earlier revisions claimed it did.
- **Recommendation precision is now non-vacuous**: 16 of 75 scored interventions
  *failed* to fix the answer. It previously read 1.000 having never observed a
  single negative, partly because one intervention was counted twice under two
  names.
- **Abstention is split** into `healthy_abstention_rate` (correctly silent on a
  clean row) and `declined_rate` (refused to diagnose a real failure). A combined
  rate mixes a good behaviour with a bad one.
- **debugging-time is a ranking proxy with a hard ceiling of 0.667** (= 1 − 1/((K+1)/2)
  at K=5). 0.635 is close to that ceiling; it is *not* a wall-clock percentage and
  is not comparable to the proposal's 30–40% target without a user study.

## 4. Ablations

### 4.1 Learned head (white-box)

| configuration | diagnosis | macro-F1 |
|---|---|---|
| rules only (cold start) | 0.814 | 0.767 |
| rules ⊕ GBT residual | **0.977** | **0.950** |

The interpretable rules alone already clear the 0.80 target, and the learned residual
adds ~0.16. This is the hybrid design's payoff: usable with zero labels, improving
as labels arrive, without discarding the interpretable prior.

### 4.2 Per-family ablation — two different questions

These were previously conflated, which made the published "which family is
load-bearing" claim an artifact. They are now reported separately.

**(a) Retrained leave-one-family-out** — the family's *information contribution*
(drop the family, retrain everything):

| dropped family | diagnosis | hallucination F1 |
|---|---|---|
| prompt | 1.000 | 1.00 |
| retrieval | 1.000 | 1.00 |
| mechanistic | 0.988 | 0.95 |
| confidence | 1.000 | 1.00 |

**No single family is necessary on this corpus.** Each family independently carries
enough information to reconstruct the diagnosis — which is itself a finding about the
corpus (§2.1), not a strength of the method. Only the mechanistic family shows any
loss, and it is concentrated exactly where theory predicts: the *parametric-override*
hallucinations, where the gold was retrieved **and attended** yet a parametric belief
won. That case is behaviourally identical to context dilution and is separable only
by mechanistic evidence.

**(b) Missing-signal robustness** — how the *shipped* engine copes when a family
drops out at inference (no retraining):

| dropped family | diagnosis | dilution F1 | halluc. F1 | ambiguity F1 | reasoning F1 | retrieval F1 |
|---|---|---|---|---|---|---|
| retrieval | **0.523** | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| confidence | 0.767 | 1.00 | 0.00 | 1.00 | 0.75 | 1.00 |
| prompt | 0.861 | 1.00 | 1.00 | 0.00 | 0.75 | 1.00 |
| mechanistic | 0.988 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 |

Losing a family at inference degrades exactly the modes that family serves — retrieval
carries both retrieval-failure and dilution (they share the `gold_recall` discriminator),
confidence carries hallucination, prompt carries ambiguity. This is the operationally
useful table; (a) is the scientific one.

### 4.3 Calibration

`ECE 0.0083 → 0.0053`, but **interior mass = 0.012**: only ~1% of predicted
probabilities lie strictly inside (0.1, 0.9). On a corpus this separable the scores
are saturated, so ECE is essentially a rescaled error rate and **carries almost no
calibration information**. It is reported with its interior-mass fraction precisely so
it cannot be quoted as calibration evidence. The calibrator is self-validating, so
where a map would hurt it is simply not adopted.

### 4.4 Robustness to observation noise

Deterministic per-observation noise added to the continuous signal features (modelling
imperfect NLI / embedding / attention estimators); labels stay clean.

| σ | diagnosis | abstention | debug-time↓ | ECE uncal | ECE cal |
|---|---|---|---|---|---|
| 0.00 | 0.977 | 0.128 | 0.635 | 0.0083 | 0.0053 |
| 0.25 | 1.000 | 0.151 | 0.654 | 0.0000 | 0.0025 |
| 0.50 | 1.000 | 0.151 | 0.649 | 0.0035 | 0.0011 |
| 0.75 | 0.988 | 0.163 | 0.630 | 0.0067 | 0.0109 |

**Honest reading: this sweep does not degrade the system.** Accuracy stays within
noise of 1.00 across the range. The decisive discriminators (`gold_recall_in_context`,
`is_correct`) are categorical and sit far from their rule thresholds, so ±0.4σ of
feature noise cannot move them. Abstention does rise monotonically (0.128 → 0.163),
which is the intended "less certain under noisier evidence" behaviour, but the effect
is small. An earlier revision presented this sweep as demonstrating graceful
degradation; it does not.

## 5. Findings

1. The interpretable rules are strong alone (0.814) and the learned residual closes
   the gap to 0.977 — the hybrid design is validated.
2. **No signal family is individually necessary on this corpus**, and the mechanistic
   family's only measurable contribution is on parametric-override hallucination.
3. At inference time the families are strongly specialised: losing retrieval halves
   accuracy, and each family's loss maps onto the modes it serves.
4. Recommendations are validated by simulated intervention and genuinely fail 21% of
   the time, so the ≥0.75 target is met on a metric that *can* fail.
5. The observability tiers are **not** distinguished by this benchmark.

## 6. Threats to validity

- **Residual shape shortcut** (§2.1): shape alone still reaches 0.644 vs a 0.356
  prior. Partly causal (dilution), partly corpus artifact.
- **Saturation.** The modes are far more separable than real data; this is why most
  metrics sit near 1.0, why ECE is uninformative, and why the noise sweep is flat.
  The *ablations* and the *shortcut probe* are more meaningful than the absolute
  headline numbers.
- **White ≡ grey.** The three-tier table reports two distinct conditions.
- **Recommendation validation is partly circular on the mock**: `add_gold_context`
  injects text containing the reference answer, which the mock then reads. The
  negatives are informative; the absolute precision is optimistic.
- **Real backends are unvalidated at runtime.** `hf`, `gguf` and `nnsight` have never
  executed (HuggingFace is unreachable in this environment). They were audited and
  substantially corrected — the nnsight capture path could not complete at all, and
  gguf would have reserved ~5 GB — but first-real-run verification is still pending.
- **Small test split** (86 rows): differences under ~0.05 are not meaningful.

## 7. Next steps

Run this same pipeline on Qwen3-4B (GGUF) with RAGTruth's human hallucination labels
plus a human-audited multi-label seed as the REAL evaluation split; report per-mode
precision/recall on real data; give the white tier a signal no other tier has (a rule
keyed on `gold_patch_effect`) so the tier comparison becomes a real experiment; and
run the debugging-time user study the ranking proxy stands in for.
