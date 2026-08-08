# TokenTrace — Methodology & Results (synthetic benchmark)

> **Every number in this document is printed by `python3 scripts/regen_report.py`,
> except where a section names the other run that produced it** (§3's rule-firing
> counts, §4.2's prevalence sweep via `run_mechanistic_value`, §4.5's
> replay-provenance counts via `tokentrace noref --json`, §4.5.1's
> counterfactual table via `run_noref(fit_noref=False)`). Numbers are regenerated
> from code, not transcribed, because an earlier revision of this report contained
> figures that no longer corresponded to any code path. If you change anything that
> could move a metric, re-run and paste the output.
>
> **That sentence was false until a previous revision, and §4.5 is why.** `regen_report.py`
> never imported `run_noref`, so every figure in the no-reference section — the
> transfer table, the per-mode F1s, the confound controls, the calibration-path
> counts, the ECE chain, the guard-cost table — was hand-transcribed with no
> regeneration path and no test pinning it. That is exactly the failure mode the
> banner was written to prevent, and it had already caused one drift. `run_noref` is
> now wired into the script and §4.5 regenerates with everything else. Whole run:
> **~70 s** single-threaded on this machine at this revision (three consecutive
> runs agreed byte-for-byte); it was 57 s before `train_engine` grew its
> reference-stripped calibration pass and `run_noref` its extra conditions.
>
> **Scope.** These results come from a deterministic mock model and a synthetic
> failure-injection corpus, with no model or dataset downloads. They validate the
> *machinery* end-to-end (signal extraction → calibrated ranked diagnosis → causal
> ordering → validated fix → graceful degradation) and let the ablations isolate each
> component. They are **not** real-data performance, and §6 lists the specific ways
> this corpus is easier than reality.

Reproduce: `tokentrace eval`, `tokentrace ablate`, `tokentrace noref`, or
`scripts/regen_report.py` (all of which share one training schedule; they
previously did not). §4.5 is now user-reachable: `tokentrace noref` runs the
no-reference split directly (`--json` for machine-readable output) — an earlier
revision had to disclose that the production-shape numbers were reachable only
through the regen script.

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

**The standing test now bounds this number.** An earlier revision had to disclose
that `test_labels_not_recoverable_from_context_shape` built its corpus from
`seeds=(0, 1, 2)` while the published figure used `seeds=(0, 1, 2, 3)` — a
related quantity on a different corpus. The test now builds the same four-seed
corpus as the published probe, and `regen_report.py` prints a single probe line
labelled "published AND tested": 0.644 is both the published number and the one
the test asserts `< 0.70` against.

## 3. Headline results (per tier)

| tier | diagnosis | Top-3 | conformal cov. | set size | rec. precision (n, negatives) | healthy abstention | declined | debug-time↓ |
|---|---|---|---|---|---|---|---|---|
| white-box | 0.977 | 0.977 | 1.000 | 1.27 | 0.787 (75, 16) | 0.846 | 0.000 | 0.635 |
| grey-box | 0.977 | 0.977 | 1.000 | 1.27 | 0.787 (75, 16) | 0.846 | 0.000 | 0.635 |
| black-box | 0.988 | 1.000 | 1.000 | 1.27 | 0.784 (74, 16) | 1.000 | 0.014 | 0.616 |

All three proposal targets are met. Read the following caveats with the table:

- **The tiers are still not distinguished on this corpus.** White and grey are no
  longer the same experiment: `hal.override_causal` (`engine/rules.py`) consumes
  `gold_patch_effect`, the WHITE-only causal feature, and abstains below WHITE —
  measured by evaluating the rules over the four-seed corpus, it fires on 64 of 64
  `hallucination_override` rows, on nothing else, at +1.36 log-odds each (the
  fire/abstain behaviour is test-pinned in `tests/test_mechanistic_value.py`). But
  the white and grey *scores* still coincide, because the corpus is saturated: the
  §4.2 prevalence sweep shows the override slice is fully separable at black-box
  tier already.
  Black-box scoring marginally *higher* is noise on 86 test rows, not evidence
  that less information helps. **This benchmark still does not demonstrate the
  tier-degradation story**; earlier revisions claimed it did.
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

| dropped family | diagnosis | halluc. F1 | ambiguity F1 | retrieval F1 |
|---|---|---|---|---|
| prompt | 0.965 | 1.00 | 0.86 | 1.00 |
| retrieval | 1.000 | 1.00 | 1.00 | 1.00 |
| mechanistic | 0.988 | 0.95 | 1.00 | 0.98 |
| confidence | 1.000 | 1.00 | 1.00 | 1.00 |

**No single family is necessary on this corpus** — the worst drop (prompt, 0.965)
still clears the 0.80 target by a wide margin. That is itself a finding about the
corpus (§2.1), not a strength of the method. The losses land where the families
serve: dropping prompt costs ambiguity F1 (0.86), dropping mechanistic costs a
little hallucination (0.95) and retrieval (0.98). An earlier revision read the
mechanistic loss as vindication — "concentrated exactly where theory predicts, on
the parametric-override hallucinations, separable only by mechanistic evidence."
The prevalence sweep below shows the "separable only mechanistically" half of that
sentence is false on this corpus.

**The mechanistic-value sweep — the thesis question, answered against the thesis.**
`run_mechanistic_value` (`tokentrace/eval/mechanistic_value.py`) scales the
prevalence of `hallucination_override` — the failure class that is behaviourally
identical to context dilution (gold present, answer wrong) and *supposed* to be
separable only mechanistically — through 1×/2×/4× (seeds `(0, 1)`: 32, 64 and 128
override rows) and retrains the standard engine at each level. From the run behind
this paragraph: **the curve is flat.** Override-vs-dilution slice F1 is **1.0 at
every tier — including BLACK — at every prevalence level**; white−grey is 0.0
everywhere; and at 4× the retrained leave-mechanistic-out delta is **−0.031**
(removing the mechanistic family *improved* white accuracy by two rows).
Mechanistic value does not emerge even when the corpus is stacked with
mechanistically-separable cases. The sweep also pinned the leak:
`MockModel._entropy_profile` hands override rows ~0.28 nats (confidently wrong)
and unsupported-fabrication/miss rows ~1.45 nats, so plain black-box confidence
separates the "behaviourally identical" pair. This is a statement about the mock's
confidence-profile leak, not about mechanistic interpretability on real models —
and it means this corpus cannot honestly evaluate the mechanistic tier until
`mock.py::_entropy_profile` is fixed (§7).

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
| 0.50 | 1.000 | 0.151 | 0.649 | 0.0032 | 0.0012 |
| 0.75 | 0.988 | 0.163 | 0.630 | 0.0064 | 0.0113 |

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
**What "as shipped" means changed this revision**: `train_engine` now fits
calibration from the cal split's reference-bearing rows AND a reference-stripped
pass over them (`fit_noref=True`, the default everywhere, including
`TokenTrace.default()`), so the shipped calibrator carries genuine `|noref` maps.
The table below is the first measurement against that engine; the previous shipped
configuration's numbers (transfer 0.593, frozen-mock 0.674) were true of the old
engine and are kept, regenerated, in §4.5.1.

| | baseline (reference-bearing) | (a) transfer | **(c) frozen-mock** | (b) retrained on stripped |
|---|---|---|---|---|
| diagnosis accuracy | 0.977 | **0.919** (−0.058) | **0.919** (−0.058) | 0.977 (±0.000) |
| Top-3 accuracy | 0.977 | 0.977 | 0.977 | 0.977 |
| conformal coverage / set size | 1.000 / 1.27 | 1.000 / **2.49** | 1.000 / 2.49 | 0.973 / 1.30 |
| declined rate | 0.000 | 0.014 | 0.014 | 0.027 |
| debugging-time↓ | 0.635 | 0.564 | 0.564 | 0.583 |
| mean root rank | 1.05 | 1.26 | 1.26 | 1.21 |
| primary-root precision (n named) | 0.973 (73/75) | 0.919 (68/74) | 0.919 (68/74) | 1.000 (71/71) |
| recommendation precision | 0.787 | `UNAVAILABLE(no_ground_truth)` | `UNAVAILABLE` | `UNAVAILABLE` |

**(a) transfer** — train and calibrate exactly as shipped, then evaluate on stripped
traces: 0.977 → 0.919. **(c) frozen-mock** — the same strip, but with the simulated
model's decision *pinned* to the one it made on the reference-bearing original, so
the diagnosis is scored against an unchanged simulated model. The condition also
reports how many decisions it replayed — `decisions_replayed: 860` here, in the
`tokentrace noref --json` output — so a reader can check the frozen ladder was
genuinely scored rather than relabelled (a verifier once relabelled this block
undetected). **(a) and (c) now agree on every metric: the
artifact delta is 0.000**, so 0.919 is the figure to quote either way. **(b)
retrained** — train on stripped data too: the ceiling if you accept you will never
have references. It fully recovers.

Per-mode F1 shows precisely what still breaks:

| mode | baseline | (a) transfer | (c) frozen-mock | (b) retrained | support |
|---|---|---|---|---|---|
| prompt_ambiguity | 1.00 | 1.00 | 1.00 | 1.00 | 12 |
| **retrieval_failure** | 1.00 | **0.87** | **0.87** | 1.00 | 26 |
| context_dilution | 1.00 | 0.95 | 0.95 | 1.00 | 9 |
| hallucination | 1.00 | 0.97 | 0.97 | 0.97 | 34 |
| reasoning_failure | 0.75 | 0.75 | 0.75 | 0.80 | 6 |

Three things to read out of this:

- **The entire recovery is a calibration effect, and that is verified, not assumed.**
  The heads are identical between this revision's transfer condition and the last
  one's; the only change is that stripped rows are now served calibration maps
  fitted on stripped data. Re-running with `fit_noref=False` reproduces the old
  0.593 exactly (§4.5.1). The old configuration did not lack information — it had
  no maps fitted for production-shape scores at all, so after the §4.5.1 guard
  every stripped probability fell through to the raw sigmoid, and the engine went
  silent (declined 0.274) or mis-ranked. With `|noref` maps: declined 0.014, and
  21 of 26 retrieval failures stay ranked first. Calibration also stops being a
  no-op on production-shape probabilities: transfer ECE is 0.0749 uncalibrated →
  **0.0295** calibrated (frozen-mock 0.0595 → 0.0322), where the old
  configuration's calibrated ECE *was* the raw-sigmoid 0.0749.
- **The cost moved from silence into set width, and the reference is still not
  free.** The 5 remaining retrieval-failure misses go 1 → abstain, 1 →
  `context_dilution`, 2 → `hallucination` (a true co-label, but the sequela — the
  fix offered is `ground_or_abstain` where `add_gold_context` was needed), 1 →
  `reasoning_failure`. Primary-root precision falls 0.973 → 0.919 (the 4
  mis-namings above, plus the baseline's own two clean-row misses, over 74 named).
  And what visibly pays for coverage staying at 1.000 is the conformal set:
  **1.27 → 2.49 modes**. A wider differential is the honest shape of "less certain
  without a reference", but a set of ~2.5 on a five-mode problem is a much weaker
  answer, and it is the price of the recovered accuracy. The earlier finding —
  retrieval failure and dilution collapsing *separately* into abstention and
  sequela rather than into each other — survives in miniature: of the 5 misses,
  exactly 1 crosses to dilution (there were zero cross-confusions before;
  `ret.gold_absent` and `dil.present_wrong` still both require
  `gold_recall_in_context` and still abstain together).
- **The mock artifact is still present; it now costs nothing.** `MockModel._decide`
  reads `Chunk.gold` as its own oracle (it derives `gold_pos_frac` from the flag's
  position, and the dilution trigger needs `0.2 ≤ gold_pos_frac ≤ 0.8`), so
  stripping the annotation still changes what the *simulated model does* — the same
  9 rows re-decide, every one `context_dilution`:

  | control | diagnosis | declined | dilution F1 |
  |---|---|---|---|
  | annotations removed, reference kept | 0.977 | — | — |
  | reference removed, annotations kept | 0.919 | — | — |
  | both removed (= (a) transfer) | 0.919 | 0.014 | 0.95 |
  | **both removed, mock decision frozen (= (c))** | **0.919** | **0.014** | **0.95** |

  Under the old configuration the artifact was worth −0.081 diagnosis (0.593 vs
  0.674); the calibrated `|noref` maps absorb the re-decided rows, and all 9
  dilution rows are diagnosed correctly whether the mock is frozen or not (the
  0.95 F1 is a precision hit from the one retrieval→dilution confusion, not a
  recall loss). The control is kept and scored precisely because its inertness is
  a *result*, not an assumption — `freeze_mock_decisions` exists because an
  earlier revision asserted this block's numbers from prose, and **the control is
  not an optimization**: on a two-seed corpus at black-box tier it *lowers*
  diagnosis accuracy (0.667 → 0.644); a test that required it to improve the
  number was written, went red, and was corrected rather than the code. What the
  control guarantees is a comparison against an unchanged simulated model — not a
  better one. (At black tier it still moves a metric today: §4.5.2.)

### 4.5.1 The `__global__` admission guard — a bug, its fix, and the fix's retirement

The history, kept because the current zeros are meaningless without it. Stripped
traces select `|noref` signatures, but the calibrator as originally shipped had no
`|noref` map — `train_engine` built calibration records only from
reference-bearing data. `_coarse()` blocked the `__ref__` rung, but `__global__`
sat one below and *was* the reference-bearing pool under another name, so **326 of
412** production-shape probabilities were served by it (ECE 0.0053 → 0.0692). The
first fix was a guard: the `__global__` rung is admissible only if the pool
contains *enough* reference-free records to have shaped it (a presence test was
not enough — 400 reference-bearing records plus one reference-free one produced a
pool a presence test happily served). That guard was honest but expensive, and an
earlier revision of this section itemised its seven-metric cost.

**This revision retires the cost by closing the underlying gap.** `train_engine`
now fits real `|noref` maps by default (§4.5), so the global pool genuinely
contains reference-free records — 392 of them on this corpus — the admission
predicate passes (`pool_admits_noref=True`), and the 86 transfer probabilities the
pooled map still serves are served *legitimately* (`BUG=False`; the flag
distinguishes the two cases and is test-pinned). `run_noref` still re-scores the
transfer condition with the admission predicate forced open, and the guard is now
**inert**: the regenerated cost line reads `comparable metrics moved by the
guard: 0 []`, with every before/after pair identical and the pooled-serving count
86/412 in both ladders.

**An inert guard's zeros are only a measurement if the same machinery can still
produce the nonzero table.** Two safeguards keep it falsifiable:

- `tests/test_noref_controls.py::test_guard_cost_machinery_reproduces_the_nonzero_table_without_noref_maps`
  re-runs the identical guard-cost machinery on an engine trained with
  `fit_noref=False` and fails unless a nonzero cost reappears — neutering the
  forced-admission re-score now turns *that* table to zeros and goes red.
- The old cost table itself is republished below **from a fresh
  `run_noref(fit_noref=False)` run at the published seeds** — the pre-fix shipped
  configuration, regenerated rather than remembered. It reproduces the previous
  revision's §4.5 wholesale: transfer 0.593, frozen-mock 0.674, declined
  0.274 / 0.178, dilution F1 0.20 / 1.00, black-tier 0.581 / 0.663 — those numbers
  were true of the engine as it then shipped, and the delta to today's 0.919 is an
  engineering change, not a corrected error.

  | transfer metric (`fit_noref=False`) | before guard | after guard |
  |---|---|---|
  | ECE (calibrated) | 0.0692 | 0.0749 |
  | Top-3 | 0.977 | 0.954 |
  | conformal coverage | 0.973 | **0.918** |
  | conformal set size | 1.534 | 1.370 |
  | context_dilution F1 | 0.364 | **0.200** |
  | debugging-time↓ | 0.208 | 0.194 |
  | mean root rank | 2.288 | 2.329 |

  Seven metrics moved (five of `noref.py`'s nine "comparable" aggregates, plus
  `context_dilution` F1 and calibrated ECE); `diagnosis_accuracy` (0.593) and
  `declined_rate` (0.274) are the two it does not move; pooled-map servings go
  326/412 → 0/412. One last-digit correction against the earlier hand-carried
  table: the before-guard ECE regenerates as 0.0692, not the 0.0691 previously
  printed.

### 4.5.2 Black-box × no-reference — the shape traces actually arrive in

The conditions above hold `Tier.WHITE`. `tokentrace.ingest` delivers neither a
reference nor activations, so the same three conditions are re-scored at `Tier.BLACK`
on the same shipped engine (`frozen_decisions_replayed: 688` in the
`tokentrace noref --json` output):

| condition (black-box) | diagnosis | Top-3 | declined | debug-time↓ | halluc. F1 | retrieval F1 | dilution F1 |
|---|---|---|---|---|---|---|---|
| baseline (reference-bearing) | 0.988 | 1.000 | 0.014 | 0.616 | 1.00 | 1.00 | 1.00 |
| (a) transfer | 0.895 | 0.977 | 0.041 | 0.554 | 0.97 | 0.87 | 0.95 |
| **(c) frozen-mock** | **0.895** | 0.977 | 0.041 | 0.554 | 0.89 | 0.87 | 0.95 |

**The black-box axis is still nearly free here.** Against the white-tier transfer
and frozen-mock conditions it costs 0.024 diagnosis (0.919 → 0.895), and on the
reference-bearing baseline it *gains* 0.011 (0.977 → 0.988, the same direction §3
already flags as noise) — all inside the ±0.05 band this report declares for an
86-row split. That is *not* evidence that mechanistic evidence is worthless; it is
the §3/§4.2 finding again (this corpus does not distinguish the tiers, and the
mock's confidence profile leaks the mechanistic distinction to the black tier),
now confirmed on the reference-free side as well. It also does not contradict
§4.2(b): that ablation removes the **retrieval** family, present at every tier,
whereas dropping to black-box removes only **mechanistic**.

So the bounded production estimate is **0.895** — not the white-tier 0.919, and a
long way from the old configuration's 0.663 (§4.5.1). The honest summary of the
whole section: on this corpus essentially all of the production loss comes from
the missing reference (0.977 → 0.919), a further small cost from the missing
activations sits inside the noise band (→ 0.895), and the recovered accuracy is
paid for in conformal set width.

One asymmetry is worth recording: at black tier the frozen-mock control still
moves a metric — it *lowers* hallucination F1 (0.97 → 0.89) at unchanged total
accuracy. Freezing returns the 9 re-decided rows to being wrong-and-diluted, and
without mechanistic evidence some of them are no longer detected as co-labelled
hallucinations. It is a real trade in the confusion matrix, and the reason the
control keeps being scored rather than assumed inert.

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
different contrasts quoted as one. Against **`hallucination`** — where 12 mis-ranked
retrieval failures landed under the old configuration and 2 of the 5 remaining
misses still land — `answer_supported_by_context` is 0.077 vs 0.071: **no
separation at all.** `max_chunk_relevance` does separate those two (0.139 vs
0.577), so the honest claim is that retrieval failure keeps *one* reference-free
discriminator against its nearest confusion, not two.

And this is **not** "you can evaluate in production" — labels survive stripping only
because the injection recipes supply them; in production you have neither the
reference nor the label. Small split (86 rows, 9 dilution).

## 5. Findings

1. The interpretable rules are strong alone (0.814) and the learned residual closes
   the gap to 0.977 — the hybrid design is validated.
2. **No signal family is individually necessary on this corpus** (worst retrained
   drop: prompt, 0.965).
3. At inference time the families are strongly specialised: losing retrieval halves
   accuracy, and each family's loss maps onto the modes it serves.
4. Recommendations are validated by simulated intervention and genuinely fail 21% of
   the time, so the ≥0.75 target is met on a metric that *can* fail.
5. The observability tiers are **not** distinguished by this benchmark. White and
   grey are now distinct experiments — `hal.override_causal` consumes the
   WHITE-only `gold_patch_effect` (64/64 override rows, nothing else, +1.36
   log-odds) — but their scores still coincide on this saturated corpus.
6. **Mechanistic value does not emerge even when the corpus is stacked in its
   favor.** The prevalence sweep (§4.2) holds override-slice F1 at 1.0 at every
   tier *including black-box* at every override prevalence (1×/2×/4×), and at 4×
   removing the mechanistic family *improved* white accuracy by 0.031. The
   pinned cause is the mock's confidence-profile leak (`_entropy_profile`:
   ~0.28 nats on override rows vs ~1.45 on misses), which hands the
   "mechanistic-only" discrimination to black-box confidence. This is unflattering
   to the project's premise and is stated as measured: on this corpus, the
   mechanistic tier earns nothing measurable that cheaper signals do not already
   provide (the §4.2(a) retrained delta, 0.012, is inside the ±0.05 band). It
   is a fact about the mock, not about mechanistic interpretability on real
   models — but until the mock is fixed, no result here can support the premise
   either.
7. **Every other number in this document is measured on traces carrying a
   ground-truth reference, which production traces do not have.** On
   production-shape traces the shipped engine now scores **0.919** (identical with
   the simulator artifact frozen out), and **0.895** once the tier is also dropped
   to black-box — the shape traces from `tokentrace.ingest` actually arrive in
   (§4.5). This is an engineering recovery from the previous configuration's
   0.593 / 0.674 / 0.663 (§4.5.1): the engine now ships `|noref` calibration maps.
   The residual failure is quiet and specific — retrieval failure F1 0.87,
   primary-root precision 0.973 → 0.919 — and the recovery's price is conformal
   set width (1.27 → 2.49). Retraining reference-free reaches 0.977. The headline
   figures above should still not be read as production performance.
8. **Losing mechanistic capture costs essentially nothing here**, on either the
   reference-bearing or the reference-free side (§4.5.2). That is a fact about
   this corpus, consistent with findings 5 and 6, not evidence that white-box
   evidence is dispensable in general.

## 6. Threats to validity

- **Residual shape shortcut** (§2.1): shape alone still reaches 0.644 vs a 0.356
  prior. Partly causal (dilution), partly corpus artifact.
- **Saturation.** The modes are far more separable than real data; this is why most
  metrics sit near 1.0, why ECE is uninformative, and why the noise sweep is flat.
  The *ablations* and the *shortcut probe* are more meaningful than the absolute
  headline numbers.
- **The mock's confidence profile leaks the mechanistic distinction.**
  `_entropy_profile` makes override rows confidently wrong (~0.28 nats) and misses
  uncertain (~1.45), so black-box confidence separates the pair the mechanistic
  tier exists for (§4.2). Until that is fixed, the tier comparison — white and grey
  now being genuinely distinct experiments notwithstanding — is measured on a
  corpus that cannot reward mechanistic evidence, and every "tier is nearly free"
  result inherits that ceiling.
- **Recommendation validation is partly circular on the mock**: `add_gold_context`
  injects text containing the reference answer, which the mock then reads. The
  negatives are informative; the absolute precision is optimistic.
- **Real weights have never been run.** The `hf` and `nnsight` backends now
  execute — 13 integration tests drive grey+white capture, generation, the full
  signal pipeline and cross-backend feature parity against a ~1M-param model built
  locally from a config, in a dedicated CI job, with compatibility verified
  against `transformers` 4.48.3 / 4.55.4 / 5.14.1 and `nnsight` 0.7.0 — and the
  ReDeEP scores were corrected in the process (`external_context_score` was
  literally the attention ratio, `parametric_knowledge_score` was structurally
  pinned; both are now distinct measurements, see the `hf.py` docstrings). But a
  tiny random-weight model exercises the *capture path*, not model behaviour, and
  **`gguf` has still never executed** (it needs weights no CI can build). One
  measurement edge is worth knowing: white-tier capture returns
  `gold_patch_effect` as MISSING (`None`) when ablating the gold chunk would leave
  an empty prompt — the causal contrast is undefined there, not zero.
- **Reference dependence.** The headline table assumes a ground-truth reference.
  §4.5 measures what happens without one: 0.977 → 0.919 (0.895 at black-box tier),
  with the accuracy recovery paid for in conformal set width (1.27 → 2.49). This
  remains the single largest gap between these results and deployed behaviour.
- **The mock reads eval metadata.** `MockModel._decide` consults `Chunk.gold`, so
  any experiment that removes annotations changes the simulated model as well as
  the evidence. §4.5 controls for it explicitly and currently measures its cost at
  0.000 diagnosis (the same 9 rows still re-decide; the calibrated `|noref` maps
  absorb them, and at black tier the freeze still moves hallucination F1). Nothing
  else in this document controls for it, so any *other* result involving removed
  annotations would inherit the confound.
- **Pooled-map servings are now legitimate, which is itself worth auditing.** With
  `|noref` maps fitted, the `__global__` pool contains 392 reference-stripped
  records, the admission guard passes, and 86 transfer probabilities are served by
  the pooled map with `noref_served_by_pooled_map=False`. The flag now separates
  that legitimate case from the §4.5.1 bug (pool shaped only by reference-bearing
  records), and tests pin both the distinction and that a mixed pooled bucket is
  validated per population, not only on the mixture (`tests/test_noref_controls.py`).
- **Small test split** (86 rows): differences under ~0.05 are not meaningful. The
  black-vs-white deltas in §4.5.2 (0.024 and 0.011) sit inside that band and are
  reported as such.

## 7. Next steps

Closed since the previous revision (each was listed here as a known gap or a next
step): the shape-probe standing test now builds the published four-seed corpus
(§2.1); `tokentrace noref` makes the production figures user-reachable; the stale
"HookedTransformer → TransformerBridge" ladder prose is gone from
`models/registry.py`, `models/nnsight.py`, `models/base.py` and `core/types.py`;
real `|noref` calibration maps are fitted by default, which recovered the
production numbers and made the `__global__` guard inert (§4.5, §4.5.1); the white
tier got its exclusive rule (`hal.override_causal`, §3); and the `hf`/`nnsight`
backends executed for the first time, with the ReDeEP scores corrected in the
process (§6).

What is genuinely next:

- **Fix `mock.py::_entropy_profile`.** The §4.2 sweep showed the mock hands the
  override/dilution discrimination to black-box confidence (~0.28 vs ~1.45 nats),
  so the mechanistic tier cannot be honestly evaluated on this corpus until
  override rows stop being confidently-wrong by construction.
- **Execute `gguf`** — the one backend that has never run. It needs real weights;
  no CI environment here can build them.
- **Run a real model end-to-end.** The tiny-model recipe in
  `tests/test_hf_backend.py` proves the capture path; a laptop with internet can
  fetch Qwen2.5-0.5B and point the same pipeline at real weights.
- **RAGTruth headline eval**: human hallucination labels plus a human-audited
  multi-label seed as the REAL evaluation split; per-mode precision/recall on real
  data.
- **The debugging-time user study** the §3 ranking proxy stands in for and cannot
  replace.
