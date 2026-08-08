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

**CPU-only, by design — not as a fallback.** Running without a GPU is a project
objective, and *every* tier honours it, including white-box causal analysis: the
causal test is input ablation and activation patching runs on small models, both on
CPU. No code path imports CUDA, and every backend defaults to `device="cpu"`
(enforced by a test). See [CPU-first](#cpu-first-no-gpu-required).

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
pip install -e .              # dependency-light CPU core (numpy, scikit-learn, lightgbm)
tokentrace demo               # train on the offline synthetic corpus + diagnose 5 scenarios
tokentrace triage log.jsonl   # diagnose a whole log; ranked clusters + the fix for each
tokentrace eval               # train + evaluate against the proposal's targets
tokentrace noref              # the no-reference (production-shape) benchmark
tokentrace ablate             # ablation studies (learned head, per-family, calibration)
```

**Bring your own traces.** `tokentrace.ingest` converts what you already log —
OpenAI/Anthropic chat logs, LangChain and LlamaIndex RAG runs, OpenTelemetry GenAI
spans, or arbitrary JSON via a field map — into the analysis contract, with no
reference answer and no gold annotations required:

```python
from tokentrace import TokenTrace
from tokentrace.ingest import load_traces
from tokentrace.triage import render_text, triage

report = triage(load_traces("prod_traces.jsonl"), TokenTrace.default())
print(render_text(report))
```

Real output, pasted verbatim from a run over the 221-trace offline corpus,
regenerated at this revision (two consecutive runs agree byte-for-byte; an earlier
sample here had been hand-edited and no longer matched `render_text`):

```text
triage: 221 traces  analyzed=221  failed=0  tier=white
  diagnosed 192  healthy 29  declined 0   abstention_rate=0.131 (healthy 0.131 + declined 0.000)

failure modes (primary root; 192 diagnosed traces):
  prompt_ambiguity         31   16.1%
  retrieval_failure        56   29.2%
  context_dilution         32   16.7%
  hallucination            60   31.2%
  reasoning_failure        13    6.8%

top clusters (ranked by n * mean_diagnostic_confidence — a triage ORDER, not measured impact):
  %diag = share of the 192 DIAGNOSED traces (of 221 supplied); non-finite means print as n/a
   #     n  %diag  conf mode               headline signal             log-odds action             impact
   1    60  31.2%  1.00 hallucination      parametric_knowledge_score     +6.96 ground_or_abstain  28 fixed / 32 not (0 unscored)
     exemplars: #4, #5, #11   (ground truth on 60/60)
   2    56  29.2%  0.69 retrieval_failure  gold_recall_in_context        +13.62 add_gold_context   32 fixed / 0 not (24 unscored)
     exemplars: #1, #6, #8   (ground truth on 56/56)
   3    32  16.7%  1.00 context_dilution   context_length_tokens         +16.22 rerank_gold_first  32 fixed / 0 not (0 unscored)
     exemplars: #2, #9, #16   (ground truth on 32/32)
   4    31  16.1%  0.97 prompt_ambiguity   prompt_ambiguity              +16.55 clarify_prompt     20 fixed / 11 not (0 unscored)
     exemplars: #3, #10, #17   (ground truth on 31/31)
   5    13   6.8%  0.95 reasoning_failure  answer_supported_by_context    +7.15 decompose_question 12 fixed / 0 not (1 unscored)
     exemplars: #42, #108, #109   (ground truth on 13/13)
  ! Cluster `share` (%diag) is a fraction of the 192 DIAGNOSED traces, not of the 221 supplied (29 healthy, 0 declined, 0 failed).
```

Note this run has ground truth on every trace, which is why `impact` is measured at
all. **Real production traces do not**, and every cluster's impact would then read
`unknown (N unscored)` — see the honesty contract in `tokentrace/triage/aggregate.py`.

Clusters key on *(failure mode, headline evidence signal)* — a retrieval failure driven
by `gold_recall_in_context` needs a different fix from one driven by
`max_chunk_relevance`. Impact counts only fixes the simulated intervention actually
scored; without ground truth it reports **unknown**, never `0`.

Full write-up of methodology, results, ablations, and honest caveats:
[`docs/REPORT.md`](docs/REPORT.md).

Actual output of `tokentrace demo` (trained engine, so the learned residual — not the
rule prior — supplies the strongest evidence line):

```text
=== retrieval failure ===
  Q: When was the Eiffel Tower completed?
  A: '1920'   (ground truth: ['1889'])
  tier=white  diagnostic_confidence=0.47
  ── ranked diagnoses ──
     1.00  Retrieval Failure  [primary_root]
           evidence: gold_recall_in_context = 0.0 contributes +13.63 log-odds (learned)
           fix: add_gold_context ✓ validated
     1.00  Hallucination      [sequela] <- Retrieval Failure
           evidence: parametric_knowledge_score = 0.85 contributes +7.04 log-odds (learned)
           fix: ground_or_abstain ✓ validated
  Top-k set: ['Retrieval Failure', 'Hallucination']
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
print(report.primary.mode, report.primary.probability)   # RETRIEVAL_FAILURE 1.0
print(report.primary.headline_evidence.rendered)         # strongest substantive line
for r in report.primary.recommendations:
    print(r.action, r.validated)                          # add_gold_context True
```

Interactive tool: `pip install 'tokentrace[app]' && streamlit run tokentrace/app/streamlit_app.py`

## Results (offline benchmark, mock model)

Trained + calibrated on the synthetic injection corpus, evaluated on a disjoint split:

| tier | diagnosis acc | Top-3 acc | recommendation precision (n, failures) |
|---|---|---|---|
| white-box | 0.977 | 0.977 | 0.787 (75, 16) |
| grey-box  | 0.977 | 0.977 | 0.787 (75, 16) |
| black-box | 0.988 | 1.000 | 0.784 (74, 16) |

Targets: diagnosis ≥ 0.80, Top-3 ≥ 0.90, recommendation precision ≥ 0.75 — all met,
on a recommendation metric that *can* fail (16 of 75 simulated fixes did not correct
the answer). Every figure is regenerated by `scripts/regen_report.py`, never
transcribed.

**On production-shape traces (no ground truth) the shipped engine scores 0.919, not
0.977** — and **0.895** when the tier is also dropped to black-box, which is the shape
traces from `tokentrace.ingest` actually arrive in. Both are measured
(`tokentrace/eval/noref.py`, [§4.5](docs/REPORT.md)) and both are now reachable with
`tokentrace noref`. The failure is quiet and small: Top-3 stays 0.977, declined rises
only 0.000 → 0.014, and primary-root precision drops 0.973 → **0.919** (68 of 74
named roots correct), concentrated on retrieval failure (F1 1.00 → 0.87). The
recovery is not free — the conformal sets that hold coverage at 1.000 widen from
1.27 to **2.49** modes. Retraining reference-free reaches 0.977.

> These numbers improved this revision by an engineering change, not a correction:
> `train_engine` now fits real `|noref` calibration maps by default (a
> reference-stripped pass over the calibration split), so a production-shape trace
> is served a map actually fitted on its own signature. The previous shipped
> configuration measured **0.593** raw transfer, **0.674** with the simulator
> artifact controlled for, and **0.663** at black-box — numbers that were true of
> the old engine and are kept as a measured counterfactual in §4.5.1. With `|noref`
> maps in place the artifact control no longer moves anything: the transfer and
> frozen-mock conditions agree on every metric (§4.5).

> **Read these with the caveats, which are load-bearing:**
> - The corpus is synthetic and highly separable, so most metrics sit near 1.0. A
>   shape-only probe (chunk counts and lengths, zero diagnostic content) still
>   reaches **0.644** against a 0.356 class prior — down from 0.917 before the
>   generator was fixed. The standing test now builds the same four-seed corpus as
>   the published figure, so 0.644 is both the published number and the tested one.
>   See REPORT §2.1.
> - **The tiers are still not distinguished by this benchmark.** White and grey are
>   now *distinct experiments* — `hal.override_causal` consumes the WHITE-only
>   `gold_patch_effect` — but their scores coincide on this saturated corpus, and a
>   prevalence sweep shows why: the mock's confidence profile already separates the
>   supposedly mechanistic-only cases at black-box tier (REPORT §4.2, §5). The
>   differences above are noise on 86 test rows.
> - The **`hf` and `nnsight` backends now execute**: 13 integration tests drive
>   grey+white capture, generation, and cross-backend feature parity against a
>   ~1M-param model built locally from a config (no downloads), in CI. They have
>   not yet run against real pretrained weights, and **`gguf` has still never
>   executed** (it needs weights).
>
> The [ablations](docs/REPORT.md) are more informative than these headline numbers.

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
- **Calibration + conformal.** Calibration *per missingness signature* (so a
  black-box or no-reference trace never reuses a map fitted on richer evidence), and
  it is **self-validating**: a map is adopted only if it beats the raw scores on
  held-out Brier, because on already-sharp scores any smoothing map makes them
  worse. APS conformal sets are built over *normalized* marginals.
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

| what | how (CPU) | cost |
|---|---|---|
| generation / resampling | quantized **GGUF** via llama.cpp (`tokentrace[generate]`) | ~3 GB RAM, Q4_K_M 4B |
| grey-box mechanistic | single **HF** forward pass: attention-to-context, logit-lens, ReDeEP scores (`tokentrace[mechanistic]`) | ~8 GB RAM |
| white-box (causal) | **input-ablation** causal test (2 extra forward passes), or activation patching on small dev models | ~2–4 GB RAM |
| retrieval / NLI | `faiss-cpu` + small sentence-transformer (`tokentrace[retrieval]`) | < 1 GB |

All three **compute-adaptive tiers** — black-box (I/O only), grey-box (logprobs + one
forward pass), white-box (causal) — run on CPU. The white tier is CPU-affordable
because the causal signal is obtained by **input ablation** rather than GPU-scale
activation patching; that substitution is the design decision that keeps the whole
framework GPU-free.

**Whole-project requirement: a 16 GB laptop.** No VM, no container, no accelerator,
no cloud spend. Everything above (build, tests, demo, the full benchmark, and real
4B models) fits there.

## Real models & datasets

```bash
pip install 'tokentrace[mechanistic,generate,retrieval,data]'
```

- **Models** (`tokentrace/models/registry.py`): Qwen3-4B, Gemma-3-4B, Phi-4-mini
  profiles + small dev models. `load_model(name, backend=...)` builds one of four
  `ModelHandle` backends — `mock`, `gguf` (llama.cpp), `hf` (`transformers`
  forward hooks), `nnsight` — chosen explicitly by the caller. **There is no
  automatic fallback between backends**, and no TransformerLens rung: earlier
  revisions of this README advertised a ladder "HookedTransformer →
  TransformerBridge → NNsight → raw HF hooks", but `transformer_lens` is not
  imported anywhere, is not declared in any extra, and neither of the first two
  rungs exists. What *is* automatic is tier degradation, downwards only:
  `ModelProfile.max_tier` caps the handle, methods above the handle's tier raise
  `TierUnavailable`, and the mechanistic extractor turns that into *missing*
  features rather than an error. Execution status: `hf` and `nnsight` run in CI
  against a tiny locally-built model (`tests/test_hf_backend.py`; compatibility
  verified against `transformers` 4.48.3, 4.55.4 and 5.14.1, `nnsight` 0.7.0);
  `gguf` has never executed — it needs real weights.
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
  models/      ModelHandle: mock · gguf · hf · nnsight · registry (profiles, tier caps)
  signals/     4 extractor families · heuristic + model (NLI/embedding) scorers · pipeline
  engine/      rules · residual classifier · calibration · conformal · causal · diagnosis
  recommend/   recommendations + simulated-intervention validation
  data/        seed pool · injection harness · synthetic builder · real loaders
  eval/        metrics · train/eval benchmark · noref (production split) ·
               mechanistic-value prevalence sweep
  retrieval/   FAISS RAG
  cache/       content-addressed on-disk trace cache
  ingest/      trace adapters: openai · langchain · llamaindex · otel · field-map
  triage/      fleet aggregation: cluster, rank, dominant fix per cluster
  app/         Streamlit trace explorer
  cli.py  api.py
scripts/       regen_report.py — regenerates every published number from code
tests/         coherence, tiers, discriminators, causal chain, benchmark, ingest
               adapters, triage clustering, no-reference split + its frozen-mock
               and black-box controls, training parity, mechanistic-value sweep,
               real-backend integration (hf + nnsight on a tiny local model),
               headless Streamlit app, shape-shortcut probe, ledger
               reconciliation, CLI seam
```

The exact test count is deliberately not quoted as a standing claim. This README
once said **229**; the suite collected **252** when that was checked, **431** a few
hours later as concurrent work landed, and **452** when regenerated at this revision
— measured in an environment *without* the `[app]` and `[mechanistic]` extras, so
`tests/test_app.py` (Streamlit) and `tests/test_hf_backend.py` (torch) both skip
and are not in that figure. A number nobody regenerates is exactly the kind of
claim this project is trying not to make, and this one drifts faster than a doc
edit can follow. Run `python3 -m pytest tests/ --collect-only -q` for the current
figure in your own environment.

## Status & roadmap

- ✅ **Phase 1 (MVP):** full pipeline end-to-end on CPU across all five modes, trained
  + calibrated, meeting targets on the offline benchmark; CLI, tests, Streamlit tool.
- ⏳ **Phase 2 (started):** the `hf` and `nnsight` capture paths execute in CI
  against a tiny locally-built model — no downloads, so not yet real weights.
  Still open: execute `gguf`; run real Qwen3/Gemma3/Phi-4 end-to-end (the
  tiny-model recipe proves the path; a laptop with internet can fetch
  Qwen2.5-0.5B); white-box patching on small models; scale to real datasets;
  cross-model consistency.
- ⏭ **Phase 3:** headline benchmark on RAGTruth + human-audited seed; ablations;
  debugging-time user study; report.

## License

Apache-2.0.
