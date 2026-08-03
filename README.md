# TokenTrace

**Evidence-based root-cause analysis for LLM applications.** Existing observability
tools *visualize* model internals; TokenTrace answers the question a developer
actually has — **"why did the model get this wrong, and what do I change?"**

Given one LLM inference (prompt, optional retrieved context, generated answer,
optional ground truth), TokenTrace correlates **prompt, retrieval, mechanistic,
and confidence** signals into **ranked, calibrated root-cause diagnoses** across
five failure modes, each backed by an **auditable evidence ledger** and mapped to
a **corrective recommendation that is validated by actually re-running the fix**.

> Think of it as *differential diagnosis for LLMs*: collect evidence → rank a
> differential → run a confirmatory test → prescribe a treatment → verify it.

**No GPU required.** Everything — build, test, demo — runs on CPU. A GPU is an
optional accelerator for one tier only (see [CPU-first](#cpu-first-no-gpu-required)).

---

## The five failure modes

| Mode | Meaning |
|---|---|
| **Prompt Ambiguity** | the input is underspecified / has multiple valid readings |
| **Retrieval Failure** | (RAG) the retrieved context does not contain the answer |
| **Context Dilution** | the answer *is* present but buried / ignored ("lost in the middle") |
| **Hallucination** | the answer contains content unsupported by context or knowledge |
| **Reasoning Failure** | the info is available but the model composes/aggregates it wrong |

## Quickstart

```bash
pip install -e .            # dependency-light CPU core (numpy, scikit-learn, lightgbm)
tokentrace demo             # train on the offline synthetic corpus + diagnose 5 scenarios
tokentrace eval             # train + evaluate against the proposal's targets
```

```text
=== retrieval failure ===
  Q: When was the Eiffel Tower completed?
  A: '1920'   (ground truth: ['1889'])
  tier=white  diagnostic_confidence=0.58
  ── ranked diagnoses ──
     0.93  Retrieval Failure  [primary_root]
           evidence: answer not recoverable from context (gold recall 0.00 < 0.3)
           fix: add_gold_context ✓ validated
     0.77  Hallucination      [sequela] <- Retrieval Failure
           evidence: unsupported answer, high parametric / low context read (ReDeEP)
           fix: ground_or_abstain ✓ validated
  Top-k set: ['Retrieval Failure']
```

Python API:

```python
from tokentrace import TokenTrace
from tokentrace.core.types import Inference, Chunk

tt = TokenTrace.default()                      # mock model, engine trained on synthetic
report = tt.analyze(Inference(
    prompt="When was the Eiffel Tower completed?\nParis is in France.",
    generated_answer="", retrieved_context=[Chunk("Paris is in France.", gold=False)],
    ground_truth=["1889"], question="When was the Eiffel Tower completed?",
    meta={"_sim": {"answer": "1889", "gold_fact": "1889", "distractor": "1920"}},
))
print(report.primary.mode, report.primary.probability)   # RETRIEVAL_FAILURE 0.93
for r in report.primary.recommendations:
    print(r.action, r.validated)                          # add_gold_context True
```

Interactive tool: `pip install 'tokentrace[app]' && streamlit run tokentrace/app/streamlit_app.py`

## Results (offline benchmark, mock model)

Trained + calibrated on the synthetic injection corpus, evaluated on a disjoint split:

| tier | diagnosis acc | Top-3 acc | recommendation precision |
|---|---|---|---|
| white-box | 1.00 | 1.00 | 1.00 |
| grey-box  | 1.00 | 1.00 | 1.00 |
| black-box | 0.97 | 0.97 | 1.00 |

Targets: diagnosis ≥ 0.80, Top-3 ≥ 0.90, recommendation precision ≥ 0.75. Black-box
loses mechanistic evidence, so reasoning-vs-dilution separation degrades *by design*
(reasoning recall drops) — the tool reports lower confidence and recommends
escalation rather than guessing.

> **Honest caveat.** These numbers are on a *self-consistent synthetic* corpus with
> a deterministic mock model; they validate the full machinery (signal extraction →
> calibrated diagnosis → validated fix → graceful degradation) end-to-end, on CPU,
> with no downloads. Headline numbers on **real** data require the real backends and
> datasets (below); real data is harder and is exactly why the framework is
> calibrated, multi-tier, and reference-free where possible.

## How it works

```
Inference ─▶ 4 signal families ─▶ FeatureVector (+ missingness mask)
          ─▶ rules-prior ⊕ learned residual ─▶ per-mode calibrated probability
          ─▶ causal DAG resolver (root vs sequela)   ─▶ evidence ledger
          ─▶ conformal Top-3 · diagnostic confidence · abstention
          ─▶ recommendations, each validated by a simulated intervention
```

- **Rules ⊕ learned residual.** Interpretable likelihood-ratio rules produce a
  log-odds *prior*; a LightGBM head (using the rule log-odds as `init_score`) fits
  only the *residual*. Total score is one additive, decomposable ledger — and with
  no training data it runs on pure rules (cold-start).
- **Calibration + conformal.** Isotonic calibration *per missingness signature*
  (so black-box probabilities stay meaningful) and APS conformal sets to hit Top-3.
- **Causal ordering.** A fixed failure-mode DAG turns five marginals into a
  *root → sequela* story ("treat the root, not the symptom"), with a dominant
  downstream mode kept as primary when its parent is only a weak co-fire.
- **Validated recommendations.** Most fixes are input-level counterfactuals
  (add gold context, re-rank, disambiguate, decompose) — so we *apply the fix and
  re-run the model* to check whether the answer corrects. That is the
  recommendation-precision metric, computed with no humans in the loop.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design, including
the edge-case catalog (multi-mode, right-for-wrong-reasons, no-ground-truth,
non-RAG, abstention, reasoning-vs-knowledge-gap, tokenizer differences, …).

## CPU-first (no GPU required)

| what | how (CPU) |
|---|---|
| generation / resampling | quantized **GGUF** via llama.cpp (`tokentrace[generate]`) |
| grey-box mechanistic | single **HF** forward pass: attention-to-context, logit-lens, ReDeEP scores (`tokentrace[mechanistic]`) |
| white-box (causal) | **input-ablation** causal test on the 4B models, or activation patching on small dev models |
| retrieval / NLI | `faiss-cpu` + small sentence-transformer (`tokentrace[retrieval]`) |

Three **compute-adaptive tiers** map onto what CPU can afford — black-box (I/O only),
grey-box (logprobs + one forward pass), white-box (causal). A GPU only *speeds up*
full-4B activation patching; nothing depends on it.

## Real models & datasets

```bash
pip install 'tokentrace[mechanistic,generate,retrieval,data]'
```

- **Models** (`tokentrace/models/registry.py`): Qwen3-4B, Gemma-3-4B, Phi-4-mini
  profiles + small dev models. Backend-agnostic capture with a fallback ladder
  (HookedTransformer → TransformerBridge → NNsight → raw HF hooks → grey-box).
- **Datasets** (`tokentrace/data/loaders.py`): RAGTruth (real hallucination labels),
  HotpotQA, Natural Questions, TruthfulQA → clean seeds for the injection harness.
- **Retrieval** (`tokentrace/retrieval/rag.py`): FAISS index → `Chunk` lists.

Point the same pipeline at a real model:

```python
tt = TokenTrace.default(backend="gguf", model_name="qwen3-4b")   # or backend="hf"
```

## Project layout

```
tokentrace/
  core/        data contracts (+ JSON serde)
  models/      ModelHandle: mock · gguf · hf · nnsight · registry (fallback ladder)
  signals/     4 extractor families · heuristic + model (NLI/embedding) scorers · pipeline
  engine/      rules · residual classifier · calibration · conformal · causal · diagnosis
  recommend/   recommendations + simulated-intervention validation
  data/        seed pool · injection harness · synthetic builder · real loaders
  eval/        metrics · train/eval benchmark
  retrieval/   FAISS RAG
  cache/       content-addressed on-disk trace cache
  app/         Streamlit trace explorer
  cli.py  api.py
tests/         26 tests (mock coherence, tiers, discriminators, causal chain, benchmark)
```

## Status & roadmap

- ✅ **Phase 1 (MVP):** full pipeline end-to-end on CPU across all five modes, trained
  + calibrated, meeting targets on the offline benchmark; CLI, tests, Streamlit tool.
- ⏭ **Phase 2:** wire the GGUF/HF backends to real Qwen3/Gemma3/Phi-4; white-box
  patching on small models; scale to real datasets; cross-model consistency.
- ⏭ **Phase 3:** headline benchmark on RAGTruth + human-audited seed; ablations;
  debugging-time user study; report.

## License

Apache-2.0.
