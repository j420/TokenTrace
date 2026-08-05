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

One trace at a time is the *diagnostic* unit, not the *operational* one. A developer
has a log of thousands, so `triage/` runs the same pipeline over a fleet and
aggregates (see [Fleet triage](#fleet-triage-triage)).

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
- **Calibration (`calibration.py`)** — per `(mode, missingness_signature)`, where the
  signature also records whether a usable GROUND-TRUTH REFERENCE was available (the
  GT-dependent features live inside otherwise-present families, so without this a
  production trace would silently reuse a map fitted on reference-bearing data).
  Self-validating: a map is adopted only if it beats the raw sigmoid on held-out
  Brier score. Isotonic on large buckets, Platt on thin ones. Hierarchical
  fallback: exact signature -> ref/noref -> global -> raw sigmoid, where the
  `global` rung is **withheld from a reference-free trace** unless the pool was
  fitted from at least `min_samples` reference-free records
  (`Calibrator.global_admits_noref`) — otherwise `__global__` is the
  reference-bearing map under another name and serving it defeats the `|noref`
  guard one rung above. A trace that clears no rung gets the raw sigmoid, which is
  the honest outcome and is what production-shape traces actually get today (see
  REPORT §4.5).
- **Conformal (`conformal.py`)** — APS sets over NORMALIZED marginals (the engine
  emits independent one-vs-rest probabilities that do not sum to 1; accumulating
  them raw made tau fit to 1.0 and inverted the set-size signal). Masked modes are
  excluded so an impossible cause never pads a set.
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
   `SEMI` + `SYNTHETIC` → train.

**Family-dropout, and where it actually applies.** Training across a per-row tier
schedule (`[WHITE, GREY, BLACK]` cycled over the training rows) hardens the heads
against missing families and keeps per-tier calibration honest — but it is a
property of the *research harness*, not of training in general. It is passed by
`train_and_evaluate(family_dropout=True)`, `eval/ablations.py` and
`eval/noref.py`; `train_engine` itself defaults `train_tiers=None`, and
**`TokenTrace.default()` — the user-facing path, used by the CLI, the API examples
and the Streamlit app — does not pass it**. The engine a user gets is therefore
trained at the handle's own tier only. This document previously stated the
dropout as unconditional; it is not, and the benchmark tables (which do use it)
are not evidence about the default engine.

## Compute-adaptive tiers

- **Black-box** — prompt + retrieval + resampled confidence. CPU-cheap.
- **Grey-box** — + logit-lens, attention-to-context/gold, ReDeEP, from one HF
  forward pass. Default for 4B models on CPU.
- **White-box** — + causal test: `gold_patch_effect` via INPUT ABLATION (two extra
  forward passes) on CPU, or activation patching on small dev models. Choosing
  ablation over GPU-scale patching is what keeps the top tier CPU-affordable; no
  tier requires an accelerator.

Missing families are handled by: GBT NaN routing · per-signature calibration ·
rules abstaining · wider conformal sets → lower confidence → abstention/escalation.

### Backends: what selection actually does

`load_model(name, backend=...)` instantiates **one** of four `ModelHandle`
implementations — `mock`, `gguf` (llama.cpp), `hf` (`transformers` forward hooks),
`nnsight` — chosen by the caller's `backend` argument. There is **no runtime
fallback between backends**: an unavailable backend raises on import rather than
silently degrading to another, and `available_backends()` only *reports*
importability, nothing consumes it to pick one. Earlier revisions of this document
and of the README advertised a ladder
"HookedTransformer → TransformerBridge → NNsight → raw HF hooks"; TransformerLens
is not imported anywhere, is not declared in any extra, and neither of the first
two rungs exists in code.

What *is* automatic is degradation along the **tier** axis, and only downwards:
`load_model` clamps the requested tier to `ModelProfile.max_tier`,
`ModelHandle.with_tier` can only down-cap, methods above the handle's tier raise
`TierUnavailable`, and `MechanisticExtractor` catches that and emits nothing — so
the features land in the missingness mask and every mechanism above keys off them.
That is the whole auto-degrade story.

## Fleet triage (`triage/`)

`triage(traces, tt)` streams N inferences through the engine and aggregates them
into a ranked fix list — the observability question ("which cluster do I fix
first?") rather than the diagnostic one ("why did *this* fail?").

- **Cluster key = (primary failure mode, headline evidence signal)**, not mode
  alone. A retrieval failure driven by `gold_recall_in_context` needs a different
  fix from one driven by `max_chunk_relevance`; collapsing them hides that. Note
  the key vocabulary differs by engine state: cold-start clusters key on rule ids
  (`ret.gold_absent`), a trained engine on TreeSHAP feature names.
- **Ranked by `n × mean_diagnostic_confidence`** — labelled everywhere as a triage
  ORDER, not measured impact. Deliberately *not* ranked by validated impact, which
  would sink every production cluster (no ground truth) to the bottom.
- **Three buckets, never merged:** diagnosed / healthy / declined. An all-abstained
  run cannot be misread as a clean fleet, and `mode_distribution()` returns `{}`
  rather than five zero shares when nothing was diagnosed.
- **Honesty contract.** Every field is classified MEASURED, UNKNOWN or DESCRIPTIVE
  in the module docstring. Fix impact comes only from recommendations the simulated
  intervention actually scored, and is `None` — never `0` — without ground truth,
  because `0` would assert the fix was tried and did nothing. There is deliberately
  no diagnosis-accuracy number here: triage runs on unlabeled traces.
- Streaming (O(clusters) memory), tolerant of a trace that raises, and importing it
  pulls in no numpy/lightgbm/sklearn.

## CLI surface

`demo` · `analyze <trace.json>` · **`triage <traces.jsonl>`** · `inject` · `eval` ·
`ablate`. The first three are user-facing; the last three are the research harness.

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
| a backend cannot reach white-box capture | `ModelProfile.max_tier` caps the handle; `MechanisticExtractor` catches `TierUnavailable` and the features go MISSING, not zero |
| confident-but-wrong vs unconfident-but-right | confidence family orthogonal to correctness → the 2×2 is a feature |

## Evaluation (`eval/`)

Diagnosis accuracy (primary correct; clean→abstain), Top-3 (gold primary in
conformal set), per-mode P/R/F1, **recommendation precision via simulated
intervention** (apply the fix, re-run, check correction), abstention rate,
per-tier curves. In the real pipeline the eval split is RAGTruth + human seed
(REAL); offline it is a disjoint synthetic split.
