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

Real output, pasted verbatim from a run over the 221-trace offline corpus (the
previous sample here had been hand-edited and no longer matched `render_text`: it
dropped the abstention split, the `(N unscored)` counts, and the `exemplars:` lines):

```text
triage: 221 traces  analyzed=221  failed=0  tier=white
  diagnosed 191  healthy 30  declined 0   abstention_rate=0.136 (healthy 0.136 + declined 0.000)

failure modes (primary root; 191 diagnosed traces):
  prompt_ambiguity         31   16.2%
  retrieval_failure        56   29.3%
  context_dilution         32   16.8%
  hallucination            60   31.4%
  reasoning_failure        12    6.3%

top clusters (ranked by n * mean_diagnostic_confidence — a triage ORDER, not measured impact):
  %diag = share of the 191 DIAGNOSED traces (of 221 supplied); non-finite means print as n/a
   #     n  %diag  conf mode               headline signal             log-odds action             impact
   1    60  31.4%  1.00 hallucination      parametric_knowledge_score    +12.54 ground_or_abstain  28 fixed / 32 not (0 unscored)
     exemplars: #4, #5, #11   (ground truth on 60/60)
   2    56  29.3%  0.57 retrieval_failure  gold_recall_in_context        +13.62 add_gold_context   32 fixed / 0 not (24 unscored)
     exemplars: #1, #6, #8   (ground truth on 56/56)
   3    32  16.8%  0.72 context_dilution   context_length_tokens         +15.40 rerank_gold_first  32 fixed / 0 not (0 unscored)
     exemplars: #2, #9, #16   (ground truth on 32/32)
   4    31  16.2%  0.72 prompt_ambiguity   prompt_ambiguity              +16.55 clarify_prompt     20 fixed / 11 not (0 unscored)
     exemplars: #3, #10, #17   (ground truth on 31/31)
   5    12   6.3%  0.73 reasoning_failure  logit_lens_answer_layer       +13.89 decompose_question 12 fixed / 0 not (0 unscored)
     exemplars: #108, #109, #110   (ground truth on 12/12)
  ! Cluster `share` (%diag) is a fraction of the 191 DIAGNOSED traces, not of the 221 supplied (30 healthy, 0 declined, 0 failed).
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
           evidence: parametric_knowledge_score = 0.85 contributes +12.61 log-odds (learned)
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

**On production-shape traces (no ground truth) the shipped engine scores 0.674, not
0.977** — and **0.663** when the tier is also dropped to black-box, which is the shape
traces from `tokentrace.ingest` actually arrive in. Both are measured
(`tokentrace/eval/noref.py`, [§4.5](docs/REPORT.md)). It mostly fails *quiet* — Top-3
barely moves (0.954), so the true cause stays in the differential, and what rises is
silence (declined 0.000 → 0.178). But it does not fail quiet entirely: precision on
the traces still given a primary root drops to **0.727**, mostly by naming
hallucination where retrieval failure was the root. Retraining reference-free
recovers 0.977.

> The raw transfer number is 0.593. 0.674 is that number with the simulator artifact
> controlled for — `MockModel._decide` reads gold annotations that no real model sees,
> so stripping them changes the *simulated model*, not just the evidence. Until this
> revision that control was asserted in prose; it is now a measured condition and
> lands where the prose said. §4.5 shows the decomposition and why the rows must not
> be read additively.

> **Read these with the caveats, which are load-bearing:**
> - The corpus is synthetic and highly separable, so most metrics sit near 1.0. A
>   shape-only probe (chunk counts and lengths, zero diagnostic content) still
>   reaches **0.644** against a 0.356 class prior — down from 0.917 before the
>   generator was fixed. The standing test bounds a *related* quantity, not this
>   one: it builds its corpus from `seeds=(0,1,2)` (0.661 vs a 0.304 prior) while
>   the published figure uses `seeds=(0,1,2,3)`. See REPORT §2.1.
> - **The tiers are not distinguished by this benchmark**, and white ≡ grey exactly
>   (no rule consumes a white-exclusive feature). The differences above are noise on
>   86 test rows.
> - The **real backends have never executed** (HuggingFace is unreachable here).
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
  features rather than an error.
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
  eval/        metrics · train/eval benchmark · noref (production split)
  retrieval/   FAISS RAG
  cache/       content-addressed on-disk trace cache
  ingest/      trace adapters: openai · langchain · llamaindex · otel · field-map
  triage/      fleet aggregation: cluster, rank, dominant fix per cluster
  app/         Streamlit trace explorer
  cli.py  api.py
scripts/       regen_report.py — regenerates every published number from code
tests/         coherence, tiers, discriminators, causal chain, benchmark, ingest
               adapters, triage clustering, no-reference split + its frozen-mock
               and black-box controls, headless Streamlit app, shape-shortcut
               probe, ledger reconciliation, CLI seam
```

The exact test count is deliberately not quoted here. This README claimed **229**; the
suite actually collected **252** when that was checked, and **431** a few hours later
as concurrent work landed. A number nobody regenerates is exactly the kind of claim
this project is trying not to make, and this one drifts faster than a doc edit can
follow. Run `python3 -m pytest tests/ --collect-only -q` for the current figure. Note
the Streamlit suite is skipped unless the `[app]` extra is installed, so the number
also depends on which extras you have.

## Status & roadmap

- ✅ **Phase 1 (MVP):** full pipeline end-to-end on CPU across all five modes, trained
  + calibrated, meeting targets on the offline benchmark; CLI, tests, Streamlit tool.
- ⏭ **Phase 2:** wire the GGUF/HF backends to real Qwen3/Gemma3/Phi-4; white-box
  patching on small models; scale to real datasets; cross-model consistency.
- ⏭ **Phase 3:** headline benchmark on RAGTruth + human-audited seed; ablations;
  debugging-time user study; report.

## License

Apache-2.0.
