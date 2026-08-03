# TokenTrace — Architecture

## Pipeline

```
Inference ─▶ SignalExtractors[4 families]        (order labs)
          ─▶ FeatureVector (+ missingness mask)
          ─▶ DiagnosisEngine: rules-prior ⊕ GBT-residual → per-mode calibrated P   (differential)
          ─▶ CausalResolver (DAG + runtime edge validation) → root + sequelae      (pathophysiology)
          ─▶ EvidenceAttributor (additive log-odds ledger)                         (chart findings)
          ─▶ Recommender + simulated intervention                                  (treat + verify)
          ─▶ DiagnosisReport
```

The **missingness mask** on `FeatureVector` is load-bearing: tiers and most edge
cases are expressed as "which signal families are present", not as bespoke
branches. `FeatureVector.missingness_signature()` selects the matching
calibration map.

## Core contracts (`core/types.py`)

`Inference` (prompt, `retrieved_context: list[Chunk] | None`, `generated_answer`,
`ground_truth: list[str] | None`) · `FeatureVector{values, missing, family_present}`
· `Diagnosis{mode, probability, role, causal_parents, evidence, recommendations}`
· `DiagnosisReport{diagnoses, conformal_set, diagnostic_confidence, abstained,
notes}` · `LabeledInference{labels, causal_edges, provenance, verification}` ·
`ModelProfile` · `Tier(BLACK<GREY<WHITE)`.

## Signal families (`signals/`)

| family | features | min tier | needs |
|---|---|---|---|
| prompt | `prompt_ambiguity`, `multihop_query`, `prompt_n_content_words` | black | — |
| retrieval | `answer_supported_by_context`, `max/mean_chunk_relevance`, `gold_position_frac`, `n_chunks`, `context_length_tokens`, **`gold_recall_in_context`** | black | RAG (GT for the last) |
| confidence | `semantic_entropy`, `self_consistency` (resample); `mean/max_token_entropy`, `answer_perplexity` (logprobs) | black / grey | generation / logprobs |
| mechanistic | `context_attention_ratio`, `gold_attention_ratio`, `logit_lens_answer_layer`, `logit_lens_stability`, `external_context_score`, `parametric_knowledge_score`; `gold_patch_effect` | grey / white | forward pass / causal |

Scorers are pluggable (`HeuristicScorers` = dependency-free CPU defaults;
`ModelScorers` = NLI + embeddings for fidelity). `gold_recall_in_context` (is the
*correct* answer present in context?) is the decisive **retrieval-failure vs
context-dilution** discriminator.

## Diagnosis engine (`engine/`)

- **Rules (`rules.py`)** — likelihood-ratio accumulators. Each rule contributes
  `weight × activation` log-odds toward one mode and **abstains (0) if any required
  signal is missing** → automatic tier degradation. Includes suppressor rules
  (correctness, ambiguity-explains-away, gold-attended-so-not-dilution).
- **Residual (`classifier.py`)** — LightGBM one-vs-rest with the rule log-odds as
  `init_score`; trees fit the residual. Total `z_mode = rule_logit + trees_margin`.
  Native NaN handling routes around missing families.
- **Calibration (`calibration.py`)** — isotonic per `(mode, missingness_signature)`
  with fallback to `(mode, global)` then sigmoid.
- **Conformal (`conformal.py`)** — APS sets targeting Top-3 coverage; set size is a
  difficulty readout.
- **Causal (`causal.py`)** — fixed DAG (`{ambiguity, retrieval, dilution} →
  {reasoning, hallucination}`); resolver starts from the strongest active mode and
  walks upstream only while a parent is comparably strong; `EvidenceAttributor`
  merges rule + TreeSHAP contributions (both log-odds) into one ranked ledger with
  provenance.
- **Engine (`diagnosis.py`)** — hard-masks retrieval modes for non-RAG, calibrates
  per signature, computes diagnostic confidence + abstention, emits notes
  (healthy / insufficient-evidence / fragile / tier-escalation).

## Supervision (`data/`)

No public dataset ships `(inference → failure-mode)` labels, so:

1. **Clean seed pool** (bundled + real loaders) — keep only examples the model
   answers correctly with gold context (the control group).
2. **Injection harness** — one deterministic transform per mode, each kept only if
   a **verification gate** confirms the intended effect (else discarded). Yields
   multi-label + causal-edge supervision.
3. **Provenance tiers** — `REAL` (RAGTruth + human seed) → headline eval only;
   `SEMI` + `SYNTHETIC` → train. Family-dropout at train time (per-row tier
   schedule) hardens the heads against missing families and keeps per-tier
   calibration honest.

## Compute-adaptive tiers

- **Black-box** — prompt + retrieval + resampled confidence. CPU-cheap.
- **Grey-box** — + logit-lens, attention-to-context/gold, ReDeEP, from one HF
  forward pass. Default for 4B models on CPU.
- **White-box** — + causal test (`gold_patch_effect` via input ablation on CPU, or
  activation patching on small models / optional GPU).

Missing families are handled by: GBT NaN routing · per-signature calibration ·
rules abstaining · wider conformal sets → lower confidence → abstention/escalation.

## Edge cases

| case | handling |
|---|---|
| multiple simultaneous modes | native multi-label + causal DAG → root→sequela chain |
| right-answer-for-wrong-reasons | correct + unsupported + high parametric → "fragile" note |
| no ground truth (production) | reference-free signals; GT features → missing; output = *risk* |
| non-RAG | retrieval + dilution hard-masked to P=0 |
| genuinely ambiguous (multiple correct) | set-valued GT; ambiguity suppresses dilution/reasoning |
| reasoning vs knowledge-gap | multihop + gold present + wrong + late/unstable formation |
| retrieval-failure vs dilution | `gold_recall_in_context` (absent vs present) |
| tokenizer / architecture differences | model-relative features (depth as fraction of layers; normalized attention) via `ModelProfile` |
| TransformerLens unsupported | fallback ladder → auto-degrade to grey-box (`ModelProfile.max_tier`) |
| confident-but-wrong vs unconfident-but-right | confidence family orthogonal to correctness → the 2×2 is a feature |

## Evaluation (`eval/`)

Diagnosis accuracy (primary correct; clean→abstain), Top-3 (gold primary in
conformal set), per-mode P/R/F1, **recommendation precision via simulated
intervention** (apply the fix, re-run, check correction), abstention rate,
per-tier curves. In the real pipeline the eval split is RAGTruth + human seed
(REAL); offline it is a disjoint synthetic split.
