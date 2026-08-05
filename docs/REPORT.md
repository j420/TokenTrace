# TokenTrace — Methodology & Results (synthetic benchmark)

> **Every number in this document is printed by `python3 scripts/regen_report.py`.**
> It is regenerated from code, not transcribed, because an earlier revision of this
> report contained figures that no longer corresponded to any code path. If you
> change anything that could move a metric, re-run that script and paste its output.
>
> **That sentence was false until this revision, and §4.5 is why.** `regen_report.py`
> never imported `run_noref`, so every figure in the no-reference section — the
> transfer table, the per-mode F1s, the confound controls, the calibration-path
> counts, the ECE chain, the guard-cost table — was hand-transcribed with no
> regeneration path and no test pinning it. That is exactly the failure mode the
> banner was written to prevent, and it had already caused one drift. `run_noref` is
> now wired into the script and §4.5 regenerates with everything else. Whole run:
> **57 s** single-threaded on this machine (`OMP_NUM_THREADS=1`), of which ~20 s is
> the no-reference split; it was ~50 s before §4.5 was wired in.
>
> **Scope.** These results come from a deterministic mock model and a synthetic
> failure-injection corpus, with no model or dataset downloads. They validate the
> *machinery* end-to-end (signal extraction → calibrated ranked diagnosis → causal
> ordering → validated fix → graceful degradation) and let the ablations isolate each
> component. They are **not** real-data performance, and §6 lists the specific ways
> this corpus is easier than reality.

Reproduce: `tokentrace eval`, `tokentrace ablate`, or `scripts/regen_report.py`
(all three now share one training schedule; they previously did not). **§4.5 is
reachable only through `scripts/regen_report.py`** — no CLI subcommand runs the
no-reference split, so the production-shape numbers are not something a user can
regenerate with the documented commands.

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
single largest caveat on the numbers below.

**The standing test does not bound this number.**
`test_labels_not_recoverable_from_context_shape` asserts `acc < 0.70`, but it
builds its corpus with `seeds=(0, 1, 2)` while the figures above come from
`seeds=(0, 1, 2, 3)`. Measured, both ways, on the current code:

| seed set | probe accuracy | class prior | test rows |
|---|---|---|---|
| `(0,1,2)` — what the test runs | 0.661 | 0.304 | 56 |
| `(0,1,2,3)` — what is published | 0.644 | 0.356 | 73 |

The test therefore guards a *related* quantity on a *different* corpus, not the
published one. It would still catch the regression it was written for (0.917 on
either seed set), so it is not useless — but "enforced by a standing test", which
this section previously claimed, overstates it. Aligning the two is a one-line
change to a file this revision does not own; until then, treat 0.644 as
regenerated by `scripts/regen_report.py` and unpinned by any test.

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

## 4.5 The no-reference (production) split — the most consequential result here

Everything above is measured on traces carrying **ground truth AND gold-chunk
flags**. Production traces carry neither.

An earlier revision opened this section by saying it measured "the operating mode the
product will actually run in". **It did not, and this revision narrows the claim.**
It varied *one* dimension — the reference — while holding `Tier.WHITE` with full
mechanistic capture. A trace arriving through `tokentrace.ingest` has no reference
*and* is typically black-box, and §4.2(b) shows that a single family lost at inference
*can* be expensive (dropping retrieval alone costs 0.523), so the second dimension was
being assumed free rather than shown to be. Both axes are now varied; **§4.5.2** is the
one that bounds the production claim.

`tokentrace/eval/noref.py` strips reference and gold flags (labels survive: they come
from the injection recipes, not from the reference) and reports **four** conditions.

| | baseline (reference-bearing) | (a) transfer | **(c) frozen-mock** | (b) retrained on stripped |
|---|---|---|---|---|
| diagnosis accuracy | 0.977 | 0.593 (−0.384) | **0.674** (−0.302) | 0.977 (±0.000) |
| Top-3 accuracy | 0.977 | 0.954 | 0.954 | 0.977 |
| conformal coverage / set size | 1.000 / 1.27 | 0.918 / 1.37 | 0.918 / 1.37 | 0.973 / 1.30 |
| declined rate | 0.000 | 0.274 | **0.178** | 0.027 |
| debugging-time↓ | 0.635 | 0.194 | 0.327 | 0.583 |
| mean root rank | 1.05 | 2.33 | 1.95 | 1.21 |
| primary-root precision (n named) | 0.973 (73/75) | 0.727 (40/55) | 0.758 (47/62) | 1.000 (71/71) |
| recommendation precision | 0.787 | `UNAVAILABLE(no_ground_truth)` | `UNAVAILABLE` | `UNAVAILABLE` |

**(a) transfer** — train and calibrate exactly as shipped, then evaluate on stripped
traces: 0.977 → 0.593. **(c) frozen-mock** — the same strip, but with the simulated
model's decision *pinned* to the one it made on the reference-bearing original, so the
diagnosis is scored against an unchanged simulated model. **This is the figure to
quote: 0.674.** **(b) retrained** — train on stripped data too: the ceiling if you
accept you will never have references. It fully recovers.

Per-mode F1 shows precisely what breaks:

| mode | baseline | (a) transfer | (c) frozen-mock | (b) retrained | support |
|---|---|---|---|---|---|
| prompt_ambiguity | 1.00 | 1.00 | 1.00 | 1.00 | 12 |
| **retrieval_failure** | 1.00 | **0.00** | **0.00** | 1.00 | 26 |
| context_dilution | 1.00 | 0.20 | **1.00** | 1.00 | 9 |
| hallucination | 1.00 | 0.97 | 0.97 | 0.97 | 34 |
| reasoning_failure | 0.75 | 0.75 | 0.75 | 0.80 | 6 |

Three things make this more than a bad number:

- **It mostly fails quiet, but not entirely.** What rises is silence — `declined`
  0.000 → 0.274 (0.178 with the mock frozen) — and Top-3 barely moves (0.977 → 0.954),
  so the true root stays in the differential; the engine loses the ability to *rank* it
  first, not to find it. But **precision does fall**: of the 55 traces still given a
  primary root, 40 are right — **0.727**, against 0.973 at baseline. 12 retrieval
  failures are named as *hallucination* (a true co-label, but the sequela rather than
  the root, so the fix offered is `ground_or_abstain` where `add_gold_context` was
  needed) and **1 as reasoning failure**. Two clean rows are also called reasoning
  failures — but those two are the baseline's *own* two misses (73/75), present with
  the reference intact, so they are not part of the reference-loss story and the
  earlier revision was wrong to list them alongside it. The incremental damage from
  stripping is exactly the 13 retrieval-family misses. An earlier revision also
  claimed "precision stays 1.000 wherever a prediction is made". That was false, and
  the per-mode table two rows up already contradicted it (`reasoning_failure` P=0.600
  in every condition).
- **The predicted retrieval-vs-dilution collapse did NOT happen.** Zero cross-confusions
  in either direction. They collapse *separately*: retrieval failure into abstention
  (13), into its own labelled sequela hallucination (12), and into reasoning failure
  (1) — 13 + 12 + 1 = 26, the full support; dilution into abstention (7), with 2 still
  correct. (An earlier revision wrote "abstention (13) and hallucination (12)" for a
  class of 26 and left the missing row unaccounted.) `ret.gold_absent` and
  `dil.present_wrong` both require `gold_recall_in_context`, so they abstain *together*
  and neither mode receives evidence to win on. The honest statement is sharper than
  the prediction: **without a reference the engine still sees that something is wrong,
  but not what caused it.**
- **The drop decomposes — but only because the artifact is now measured, not inferred.**
  `MockModel._decide` reads `Chunk.gold` as its own oracle (it derives `gold_pos_frac`
  from the flag's position, and the dilution trigger needs `0.2 ≤ gold_pos_frac ≤ 0.8`),
  so stripping the annotation changes what the *simulated model does* — impossible for
  a real model, which never sees eval metadata.

  | control | diagnosis | declined | dilution F1 |
  |---|---|---|---|
  | annotations removed, reference kept | 0.977 | — | — |
  | reference removed, annotations kept | 0.674 | — | — |
  | both removed (= (a) transfer) | 0.593 | 0.274 | 0.20 |
  | **both removed, mock decision frozen (= (c))** | **0.674** | **0.178** | **1.00** |

  Until this revision the bottom row did not exist: the report *asserted* that freezing
  the mock would return diagnosis to 0.674 and dilution F1 to 1.00, from prose, with no
  code path and no test. It is now a measured condition (`freeze_mock_decisions` in
  `noref.py`), and it lands where the prose said it would — 0.674 / 0.178 / 1.00. The
  artifact accounts for all 9 re-decided rows, every one of them `context_dilution`.

  **The control is not an optimization and does not always help.** On a two-seed
  corpus at black-box tier it *lowers* diagnosis accuracy (0.667 → 0.644); a test that
  required it to improve the number was written, went red, and was corrected rather
  than the code. Removing an artifact moves a score in whichever direction the
  artifact happened to push, and here that direction is corpus-dependent. What the
  control guarantees is a comparison against an unchanged simulated model — not a
  better one.

  Note that the *first* row cannot serve as the artifact's control: keeping the
  reference means `dil.present_wrong` still fires and re-diagnoses the re-decided rows,
  so it *masks* the artifact rather than isolating it. The artifact is present in that
  condition too — the same 9 rows re-decide — it just costs nothing there. The rows are
  therefore **not additive**.

  **`retrieval_failure`'s collapse is genuine and survives the control bit-for-bit**
  (F1 0.00 in both; 13 → abstain, 12 → hallucination, 1 → reasoning failure, identical
  under the freeze). That is the real finding. The dilution collapse is not.

### 4.5.1 A calibration bug this experiment found

Stripped traces do select `|noref` signatures, but a shipped calibrator has no `|noref`
map — `train_engine` builds calibration records only from reference-bearing data.
`_coarse()` blocks the `__ref__` rung, but `__global__` sits one below and *is* the
reference-bearing pool under another name, so **326 of 412** production-shape
probabilities were served by it (ECE 0.0053 → 0.0691). Fixed: the `__global__` rung is
admissible only if the pool contains *enough* reference-free records to have shaped it —
a presence test was not enough, since 400 reference-bearing records plus one
reference-free one produced a 99.75%-reference-bearing pool that a presence test happily
served, restoring the bug.

That 326 is now regenerated rather than remembered: `run_noref` re-scores the transfer
condition with the guard's admission predicate forced open, which reproduces the
pre-guard ladder exactly. With the guard live the same count is **0 of 412** — every
production-shape probability falls through to the raw sigmoid, which is why the transfer
condition's uncalibrated and calibrated ECE are the same number (0.0749).

**The guard's full cost, since a partial disclosure is its own kind of burying.** It
moves **seven** of the metrics reported here on the transfer condition — five of
`noref.py`'s nine "comparable" aggregate metrics (`top3_accuracy`,
`conformal_coverage`, `conformal_set_size`, `debugging_time_reduction`,
`mean_root_rank`) plus `context_dilution` F1 and calibrated ECE. Not the two first
reported, and not the "at least eight" a later revision claimed without counting;
the count is now printed by the script rather than asserted:

  | transfer metric | before guard | after guard |
  |---|---|---|
  | ECE (calibrated) | 0.0691 | 0.0749 |
  | Top-3 | 0.977 | 0.954 |
  | conformal coverage | 0.973 | **0.918** |
  | conformal set size | 1.534 | 1.370 |
  | context_dilution F1 | 0.364 | **0.200** |
  | debugging-time↓ | 0.208 | 0.194 |
  | mean root rank | 2.288 | 2.329 |

  `diagnosis_accuracy` (0.593) and `declined_rate` (0.274) are the two it does *not*
  move. Two of the seven appear in the tables above attributed to losing the reference:
  the 0.918 coverage is 0.027 reference-loss and 0.055 *guard*, and the dilution F1 of
  0.20 was 0.36 before it. The guard is still right — the map was adopted for beating
  raw sigmoid on *reference-bearing* held-out data and was never validated on
  reference-free traces — but it is not cheap, and the better fix remains fitting real
  `|noref` maps rather than refusing the pooled one.

### 4.5.2 Black-box × no-reference — the shape traces actually arrive in

The conditions above hold `Tier.WHITE`. `tokentrace.ingest` delivers neither a
reference nor activations, so the same three conditions are re-scored at `Tier.BLACK`
on the same shipped engine:

| condition (black-box) | diagnosis | Top-3 | declined | debug-time↓ | halluc. F1 | retrieval F1 | dilution F1 |
|---|---|---|---|---|---|---|---|
| baseline (reference-bearing) | 0.988 | 1.000 | 0.014 | 0.616 | 1.00 | 1.00 | 1.00 |
| (a) transfer | 0.581 | 0.954 | 0.288 | 0.175 | 0.97 | 0.00 | 0.36 |
| **(c) frozen-mock** | **0.663** | 0.954 | 0.192 | 0.280 | 0.88 | 0.00 | 1.00 |

**The result is unflattering to the premise, not to the system: the black-box axis is
nearly free here.** Against the white-box equivalents it costs 0.012 diagnosis on the
transfer condition and 0.011 on the frozen-mock one, and on the reference-bearing
baseline it *gains* 0.011 (0.977 → 0.988, the same direction §3 already flags as
noise). All three are inside the ±0.05 band this report declares for an 86-row split.
That is *not* evidence that mechanistic evidence
is worthless; it is the §3 finding again (this corpus does not distinguish the tiers,
and white ≡ grey exactly), now confirmed on the reference-free side as well. It also
does not contradict §4.2(b): that ablation removes the **retrieval** family, which is
present at every tier, whereas dropping to black-box removes only **mechanistic**,
which §4.2 already measured as the least load-bearing family (0.988).

So the bounded production estimate is **0.663**, not 0.674 and not 0.593. The honest
summary of the whole section is that on this corpus essentially all of the production
loss comes from the missing reference, and none of it measurably from the missing
activations.

One asymmetry is worth recording: at black tier the frozen-mock control *lowers*
hallucination F1 (0.97 → 0.88) while raising dilution F1 (0.36 → 1.00). Freezing
returns the 9 re-decided rows to being wrong-and-diluted, and without mechanistic
evidence some of them are no longer detected as co-labelled hallucinations. It is a
real trade in the confusion matrix, not a free win.

### 4.5.3 Caveats

(b)'s ceiling is optimistic for `context_dilution`: its surviving reference-free
separators include `context_length_tokens`, which is the §2.1 shape shortcut.

`retrieval_failure`'s reference-free recovery is *partly* clean, and the previous
sentence here overstated it. It read "`max_chunk_relevance` 0.14 vs 0.51,
`answer_supported_by_context` 0.08 vs 1.00 separate it with no reference at all"
without naming the comparison class — and **no single class produces both numbers**.
Measured per gold-primary class on the test split (means; identical before and after
stripping, since neither feature needs a reference):

(Both features come from `RetrievalExtractor`, which does not run on a non-RAG
inference, so the means are over the `n_rag` rows, not all `n`.)

| class | n | n_rag | `max_chunk_relevance` | `answer_supported_by_context` |
|---|---|---|---|---|
| retrieval_failure | 26 | 26 | 0.139 | 0.077 |
| context_dilution | 9 | 9 | 0.516 | 0.278 |
| hallucination | 20 | 14 | 0.577 | 0.071 |
| prompt_ambiguity | 12 | 10 | 0.307 | 0.150 |
| reasoning_failure | 6 | 6 | 0.249 | 1.000 |
| (clean) | 13 | 12 | 0.454 | 1.000 |

The "0.51" was `context_dilution`; the "1.00" was `(clean)`/`reasoning_failure` — two
different contrasts quoted as one. Against **`hallucination`**, which is where the 12
mis-ranked retrieval failures actually land, `answer_supported_by_context` is 0.077 vs
0.071: **no separation at all.** `max_chunk_relevance` does separate those two
(0.139 vs 0.577), so the honest claim is that retrieval failure keeps *one*
reference-free discriminator against its nearest confusion, not two.

And this is **not** "you can evaluate in production" — labels survive stripping only
because the injection recipes supply them; in production you have neither the
reference nor the label. Small split (86 rows, 9 dilution).

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
6. **Every other number in this document is measured on traces carrying a ground-truth
   reference, which production traces do not have.** On production-shape traces the
   shipped engine drops to **0.674** with the simulator artifact controlled for, and
   to **0.663** once the tier is also dropped to black-box — the shape traces from
   `tokentrace.ingest` actually arrive in (§4.5). The raw 0.593 is *not* the figure to
   quote: it includes a mock artifact with no production analogue. It fails mostly
   quiet — silence rises, Top-3 barely moves — but precision does fall (0.973 → 0.727),
   and retraining reference-free recovers it fully. The headline figures above should
   not be read as production performance.
7. **Losing mechanistic capture costs essentially nothing here**, on either the
   reference-bearing or the reference-free side (§4.5.2). That is a fact about this
   corpus, consistent with finding 5, not evidence that white-box evidence is
   dispensable in general.

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
- **Reference dependence.** The headline table assumes a ground-truth reference. §4.5
  measures what happens without one: 0.977 → 0.674 with the mock artifact controlled
  for (0.663 at black-box tier). This is the single largest gap between these results
  and deployed behaviour.
- **The mock reads eval metadata.** `MockModel._decide` consults `Chunk.gold`, so any
  experiment that removes annotations changes the simulated model as well as the
  evidence. §4.5 now controls for it explicitly; nothing else in this document does,
  so any *other* result involving removed annotations would inherit the same confound.
- **The `noref_served_by_pooled_map` flag cannot currently fire.** It detects a
  reference-free trace being served the pooled reference-bearing calibration map, and
  the calibrator now refuses that inside `transform` under the same predicate — so the
  flag is a regression detector for the guard, not a measurement. A `False` there
  means "the guard is in place", not "pooling was measured and found absent".
- **Small test split** (86 rows): differences under ~0.05 are not meaningful. Two
  numbers in §4.5.2 sit inside that band and are reported as such.

## 7. Next steps

Run this same pipeline on Qwen3-4B (GGUF) with RAGTruth's human hallucination labels
plus a human-audited multi-label seed as the REAL evaluation split; report per-mode
precision/recall on real data; give the white tier a signal no other tier has (a rule
keyed on `gold_patch_effect`) so the tier comparison becomes a real experiment; and
run the debugging-time user study the ranking proxy stands in for.

Known gaps this revision documented but did not close (each touches a file outside
its scope):

- `test_labels_not_recoverable_from_context_shape` builds its corpus from
  `seeds=(0,1,2)` while §2.1 publishes `seeds=(0,1,2,3)`. Align the two so the
  standing test bounds the published number (§2.1).
- No CLI subcommand runs the no-reference split; `scripts/regen_report.py` is the only
  entry point. `tokentrace eval --noref` would make the production figures
  user-reachable.
- `tokentrace/models/registry.py`, `models/nnsight.py`, `models/base.py` and
  `core/types.py` still describe the non-existent
  "HookedTransformer → TransformerBridge" fallback ladder in docstrings and comments.
  README and ARCHITECTURE.md have been corrected; those four have not.
- Fitting real `|noref` calibration maps would remove the need for the `__global__`
  guard and recover the seven metrics §4.5.1 shows it costs.
