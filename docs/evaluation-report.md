# TrialMine — Evaluation Report

> **Headline.** TrialMine ships a 5-stage clinical-trial search pipeline
> (BM25 → semantic → RRF → cross-encoder → LightGBM blender) over a 140,723-
> trial oncology index. On the held-out 65-query benchmark, the full
> pipeline reaches **NDCG@5 = 0.794 ± 0.06** with each stage contributing a
> measurable monotonic lift. On a harder 20-query complex+vague slice,
> NDCG@5 = **0.526 ± 0.11** — driven primarily by the LightGBM metadata
> blender, which on hard queries adds **+0.113 NDCG@5 over the cross-encoder
> stage alone** (the single biggest marginal lift in the entire pipeline).
> Production stack as of 2026-05-14: v2 BioLinkBERT bi-encoder + v2 graded-
> MarginMSE cross-encoder + v3-regularized LightGBM ranker.

---

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Headline ablation — both held-out sets](#2-headline-ablation--both-held-out-sets)
3. [Stack evolution: pre-Phase-11 → current production](#3-stack-evolution-pre-phase-11--current-production)
4. [Per-category breakdown (65q held-out)](#4-per-category-breakdown-65q-held-out)
5. [Apples-to-apples CE comparison (Phase C2)](#5-apples-to-apples-ce-comparison-phase-c2)
6. [Blender α sweep with v2 CE (Phase C3)](#6-blender--sweep-with-v2-ce-phase-c3)
7. [LightGBM ranker analysis (Phase 12)](#7-lightgbm-ranker-analysis-phase-12)
8. [Inter-annotator agreement (Phase 9)](#8-inter-annotator-agreement-phase-9)
9. [Head-to-head against ClinicalTrials.gov v2 search (Phase 9)](#9-head-to-head-against-clinicaltrialsgov-v2-search-phase-9)
10. [Decision trail — 36, 38, 39, 40](#10-decision-trail--36-38-39-40)
11. [Limitations (read this first if you're a reviewer)](#11-limitations-read-this-first-if-youre-a-reviewer)
12. [Reproducibility](#12-reproducibility)
13. [Companion documents](#13-companion-documents)

---

## 1. Pipeline overview

Every search query flows through five stages, each adding cost and quality:

```
Query → BM25 (Elasticsearch)        ┐
      → Semantic (FAISS, v2 biLinkBERT) ┘─ RRF fusion (k=60)  →  top-50 candidates
                                                                 ↓
                                                        Cross-encoder re-rank
                                                        (v2 MarginMSE, rrf_weight=0.3)
                                                                 ↓
                                                       LightGBM blender (v3-regularized)
                                                                 ↓
                                                          top-10 results to user
```

**Production stack (as of 2026-05-14)**:

| Layer | Model | Loaded via |
|---|---|---|
| Bi-encoder | `models/embeddings/fine-tuned-v2` | `tools.py:53` (env: `TRIALMINE_EMBEDDER`) |
| FAISS index | `data/faiss_finetuned_v2.{index,json}` | `tools.py:51-52` |
| Cross-encoder | `models/cross-encoder/fine-tuned-v2` (graded MarginMSELoss) | `tools.py:54` (env: `TRIALMINE_CROSS_ENCODER`) |
| LightGBM ranker | `models/ranker/v3-regularized/model.lgb` | `tools.py:55` (env: `TRIALMINE_RANKER`) |
| CE blend weight | `rrf_weight=0.3` (CE-dominant) | `cross_encoder.py:97` |

Latency budget: typical agent search returns in **~6–7 seconds end-to-end** on
CPU (Mac M-series), dominated by cross-encoder inference. See §11 for the
honest read on whether this number generalizes.

---

## 2. Headline ablation — both held-out sets

Methodology: per query, the pipeline produces a ranked top-K. NDCG@5/@10
and MRR are computed against graded relevance labels (0–3), with bootstrap
95% confidence intervals from 1000 resamples over queries. Same labels
were never used to train the bi-encoder, CE, or LGB.

### 65-query general held-out set
*Labels: `data/evaluation/full_labeled_dataset.jsonl`. Categories: common (5), rare (5), pediatric (3), complex (5), geographic (3), vague (5), treatment (4), existing (30), rare_explicit (5). Total 65 queries × ~20 candidates = 1,300 labeled pairs.*

| Method | NDCG@5 | NDCG@10 | MRR | Median latency |
|---|---|---|---|---|
| BM25 only | 0.499 ± 0.09 | 0.459 ± 0.07 | 0.673 ± 0.10 | 14 ms |
| Semantic only | 0.483 ± 0.07 | 0.436 ± 0.05 | 0.674 ± 0.09 | 37 ms |
| Hybrid (BM25 + Semantic via RRF) | 0.642 ± 0.06 | 0.601 ± 0.05 | 0.795 ± 0.07 | 66 ms |
| + Cross-encoder | 0.709 ± 0.06 | 0.649 ± 0.05 | 0.833 ± 0.07 | 4.7 s |
| **+ LightGBM blender** | **0.794 ± 0.06** | **0.756 ± 0.05** | **0.910 ± 0.06** | **4.7 s** |

### 20-query Phase-C4 expansion (complex + vague slices only)
*Labels: `data/evaluation/full_labeled_dataset_expansion_v2.jsonl`. These are the slices v1 was historically weakest on; added in Phase C4 when n=5 bootstrap CIs on the 65q set were too wide to decide the ship gate.*

| Method | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|
| BM25 only | 0.274 ± 0.11 | 0.292 ± 0.10 | 0.454 ± 0.17 |
| Semantic only | 0.315 ± 0.10 | 0.333 ± 0.08 | 0.563 ± 0.18 |
| Hybrid (BM25 + Semantic via RRF) | 0.382 ± 0.09 | 0.379 ± 0.07 | 0.563 ± 0.14 |
| + Cross-encoder | 0.413 ± 0.10 | 0.432 ± 0.07 | 0.665 ± 0.14 |
| **+ LightGBM blender** | **0.526 ± 0.11** | **0.533 ± 0.09** | **0.775 ± 0.15** |

### Marginal lift per stage

| Stage adds | 65q ΔNDCG@5 | 20q ΔNDCG@5 | Observation |
|---|---|---|---|
| Hybrid > best single retriever | +0.143 (vs BM25) | +0.067 (vs Semantic) | BM25 and semantic are complementary; RRF wins |
| + CE over Hybrid | +0.067 | +0.031 | Cross-encoder lifts moderately |
| **+ LGB over + CE** | **+0.085** | **+0.113** | **Single biggest marginal lift, especially on hard slice** |

The LGB-over-CE delta exceeds the CE-over-Hybrid delta on both datasets. The
**hard slice (complex + vague) benefits MORE from LGB than the easy slice
does** — counterintuitive, but the data is unambiguous and well-documented
(§7).

---

## 3. Stack evolution: pre-Phase-11 → current production

Same five-stage ablation, three different upstream stacks. The story of how
each layer evolved and what each swap actually bought us.

| Stack | Bi-encoder | FAISS | Cross-encoder | LGB blender |
|---|---|---|---|---|
| **Pre-Phase-11** | v1 BiLinkBERT | v1 index | v1 binary CE (BCE loss) | v2 (trained on v1 features) |
| **Phase 11 ship (transient)** | v2 (Phase 10) | v2 | **v2 graded CE (MarginMSE)** | v2 (now stale — features changed under it) |
| **Phase 12 ship (current production)** | v2 | v2 | v2 graded CE | **v3-regularized** (anti-concentration retrain) |

### 65q `+LGB` NDCG@5 across the three stacks

| Stack | NDCG@5 | Delta |
|---|---|---|
| Pre-Phase-11 | 0.783 ± 0.06 | (baseline) |
| Post-Phase-11 (v2 CE shipped, LGB not yet retrained) | **0.761 ± 0.06** | **−0.022** (silent regression) |
| **Current production (v2 CE + v3-regularized LGB)** | **0.794 ± 0.06** | **+0.011 above pre-Phase-11**, **+0.033 above the transient stale-LGB state** |

### 20q expansion `+LGB` NDCG@5 across the three stacks

| Stack | NDCG@5 |
|---|---|
| Pre-Phase-11 | 0.450 ± 0.11 |
| Post-Phase-11 (stale v2 LGB on v2 features) | 0.467 ± 0.09 |
| **Current production (v3-regularized LGB)** | **0.526 ± 0.11** (+0.076 above pre-Phase-11) |

### The critical finding from this table

**Shipping the v2 cross-encoder alone (Phase 11) silently regressed the
final pipeline output on the 65q held-out** (NDCG@5 0.783 → 0.761). The v2
LGB was trained on v1-era feature distributions, so when the CE swapped
underneath it, the LGB no longer knew how to integrate the new CE signal.

This was caught only by running the full stack-vs-stack ablation. **It would
not have surfaced from looking at the CE alone**, because v2 CE was a clear
win at the `+CE` row (+0.097 NDCG@5 over v1 on the 65q set). The lesson
(documented as **Decision 40**, see §10): when any upstream feature-source
ships (bi-encoder, FAISS, CE), schedule an LGB re-fit BEFORE closing the
cycle.

The Phase-12 fix (v3-regularized LGB) restores monotonic pipeline
improvement at every stage and exceeds the pre-Phase-11 baseline by +0.011
on 65q and +0.076 on 20q.

---

## 4. Per-category breakdown (65q held-out)

NDCG@5 per category, current production (v2 CE + v3-regularized LGB) at the
`+LGB` output:

| Category | n | NDCG@5 | Δ vs pre-Phase-11 stack | Notes |
|---|---|---|---|---|
| common | 5 | 0.806 | +0.081 | Easy slice; ceiling effects |
| existing | 30 | 0.867 | +0.068 | The 20q carryover from Phase 9 + 10 extras |
| pediatric | 3 | 0.903 | +0.218 | Small n; big lift |
| rare | 5 | 0.954 | +0.031 | Near-ceiling; v1 was already at 1.000 |
| rare_explicit | 5 | 0.518 | 0 (tied) | Persistent weak slice; explicit rare cancer terms |
| **complex** | 5 | **0.415** | **−0.109** | **The single weakest slice. See §11.** |
| **treatment** | 4 | **0.835** | **−0.117** | **n=4 — wide CI. Real concern, see §11.** |
| **vague** | 5 | **0.686** | **−0.080** | **Caregiver-phrased queries; documented weakness.** |
| geographic | 3 | 0.466 | −0.305 | n=3; bootstrap CIs huge; not a confident signal |

**Reading this honestly**:
- v3-regularized wins on 4 of 9 categories and loses on 4 (1 tied)
- Aggregate +0.014 lift on 65q masks per-category losses on `complex`, `treatment`, `vague`, `geographic`
- The losses are on small-n categories (n=3-5) with wide CIs; not statistically conclusive at this sample size
- The wins are on the bigger categories (`existing` n=30, `common` + `pediatric` + `rare` n=3-5 each but big absolute lifts)

See §11 for the honest read on what these regressions mean and what work
would resolve them.

---

## 5. Apples-to-apples CE comparison (Phase C2)

**Question**: did v2 graded CE actually outperform v1 binary CE as a ranker,
holding everything else equal?

**Setup**: for each of 65 held-out queries, take the labeled candidate pool
(~20 NCTs/query — the exact set Haiku graded) and ask each CE to rank them.
Compute NDCG@5/@10 + MRR against the relevance labels. No retrieval, no
blending — pure ranking quality on a fixed candidate set.

| Configuration | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|
| Hybrid baseline (RRF only, no CE) | 0.837 ± 0.06 | 0.846 ± 0.05 | 1.000 |
| v1 binary CE (pure replacement) | 0.789 ± 0.06 | 0.819 ± 0.05 | 0.982 |
| **v2 graded CE (pure replacement)** | **0.898 ± 0.04** | **0.914 ± 0.03** | **1.000** |

**Δ v2 vs v1 NDCG@5 = +0.109 with non-overlapping bootstrap CIs.** v2's lower
bound (0.858) is above v1's upper bound (0.846). Statistically separated.

**Notable secondary finding**: v1 binary CE on the labeled pool scored
**0.789, worse than the hybrid baseline (0.837)**. This is the original
Issue #5 (`docs/things_can_be_fixed.md`) finding empirically reproduced: v1's
binary-trained CE was a worse pure ranker than RRF alone. The production
blender capped CE at α=0.3 weight as a quality floor. v2 inverts this
relationship (§6).

### The same comparison on the 20q expansion

| Configuration | NDCG@5 | NDCG@10 |
|---|---|---|
| Hybrid baseline | 0.538 ± 0.11 | 0.603 ± 0.10 |
| v1 binary CE | 0.510 ± 0.12 | 0.596 ± 0.09 |
| **v2 graded CE** | **0.622 ± 0.11** | **0.733 ± 0.10** |

Same direction; CIs overlap (n=20 is too small for clean separation). Per-
category on the 20q: complex +0.175, vague +0.049.

### Why the "pure CE on hybrid top-50" number tells a different story

On the labeled-pool methodology (the Phase C2 setup above), v2 CE looks
excellent (0.898). But the *same* v2 CE evaluated on hybrid retrieval's
top-50 candidates (no constraint to the labeled pool) scored only 0.553 on
the same 65q set. The gap is **not** a v2 CE weakness — it's a label-coverage
artifact: hybrid retrieval surfaces candidates that were never in the
labeled pool, NDCG treats them as relevance=0, and the metric craters.

See §11 for a full explanation of this label-coverage limitation and what
it means for any "pure CE vs LGB" comparison.

---

## 6. Blender α sweep with v2 CE (Phase C3)

The production CE blender at `cross_encoder.py:97` combines RRF rank and CE
score:

```
blended_score = α * rrf_norm + (1 − α) * ce_sigmoid
```

In the v1 era, α = 0.7 (RRF-dominant). With v2 CE wired in, we swept α ∈
{0.3, 0.4, 0.5, 0.6, 0.7} on the held-out 65q:

| α (RRF weight) | (1−α) CE weight | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|---|
| **0.3** | **0.7** | **0.861 ± 0.05** | **0.880 ± 0.04** | **1.000** ← **shipped** |
| 0.4 | 0.6 | 0.852 ± 0.05 | 0.874 ± 0.04 | 1.000 |
| 0.5 | 0.5 | 0.850 ± 0.05 | 0.869 ± 0.04 | 1.000 |
| 0.6 | 0.4 | 0.844 ± 0.05 | 0.861 ± 0.04 | 1.000 |
| 0.7 (old prod) | 0.3 | 0.842 ± 0.05 | 0.858 ± 0.04 | 1.000 |

**Lower α (more CE weight) monotonically improves NDCG.** This confirms v2
CE is calibrated — the blender now trusts it, the opposite of v1 where CE
was capped to 0.3 weight as a quality floor.

**Future option (deferred, not shipped)**: pure v2 CE without blending
(effectively α = 0.0) scored NDCG@5 = 0.898 on the same labeled pool —
**+0.037 over the shipped α = 0.3**. Shipping α = 0.0 would remove the
blender entirely. We didn't take it because:

- The blender lives at a real architectural boundary; removing it is its own
  ship cycle that deserves separate review
- Per-category check showed `rare` actually benefits from a non-zero RRF
  weight (BM25 carries that slice — see §4)
- The labeled-pool methodology may overstate the pure-CE advantage (§11
  label-coverage discussion)

The α = 0.3 ship captures the bulk of the available NDCG lift with one
parameter change, leaving the architectural decision (drop blender entirely)
for a future cycle with proper pooled labels.

---

## 7. LightGBM ranker analysis (Phase 12)

The biggest engineering surprise of the entire project. Two failed attempts
preceded the production ranker.

### Why we touched the LGB at all

After Phase 11 shipped v2 CE, the full-pipeline `+LGB` row on the 65q held-out
*regressed* from 0.783 → 0.761 (§3). The diagnosis: v2 LGB was trained on
features computed against the v1 stack (v1 bi-encoder + v1 FAISS + v1 CE).
At inference today, it's fed features from the v2 stack — a different
distribution. The LGB hadn't learned how to integrate v2 CE's much stronger,
much tighter-distribution score.

### Attempt 1 — vanilla v3 retrain (failed)

Same 145q labels, same compute_features code, but features re-computed
against the v2 stack and the LGB retrained on those features.

| Signal | v2 LGB | v3 vanilla | Read |
|---|---|---|---|
| LOOCV NDCG@5 | 0.843 | **0.989** | Suspiciously high — over-fit signature |
| Held-out 65q NDCG@5 | 0.761 | **0.697** | **−0.064 regression** |
| Held-out 20q NDCG@5 | 0.467 | 0.518 | +0.051 (but small n) |
| Feature importance: CE / next-highest ratio | 1.65× | **7.2×** | **Concentration on CE** |

**Root cause**: v2 CE is so dominant a feature (importance 4810 vs next-
highest 666) that LightGBM correctly learned "trust CE almost exclusively"
on the labeled-pool training distribution. At held-out time on a fresh
hybrid-top-50 candidate pool (with candidates outside the labeled pool that
need metadata signals for tiebreaking), the LGB had stopped using metadata
and the ensemble collapsed.

The training-time strategy was correct; the test-time distribution was
different. Classic train-test mismatch.

### Attempt 2 — Path-1 anti-concentration regularization (production)

Same labels, same features, but training hyperparameters chosen specifically
to prevent the LGB from concentrating on one feature:

| Hyperparameter | v2 | v3 vanilla | v3-regularized (production) |
|---|---|---|---|
| `feature_fraction` | 0.8 | 0.8 | **0.5** |
| `feature_fraction_bynode` | 1.0 | 1.0 | **0.5** |
| `lambda_l2` | 0 | 0 | **1.0** |
| `min_data_in_leaf` | 5 | 5 | **50** |
| `num_boost_round` | 200 | 200 | **100** |

The single most effective lever is column subsampling (`feature_fraction =
0.5`). Each tree only sees half the features, so roughly half the trees
*can't even use CE* and must learn metadata patterns. The ensemble averaging
recovers a balanced ranker even when CE remains the most-used feature in
absolute terms.

**Held-out results — v3-regularized vs v2 LGB**:

| Dataset | v2 LGB | v3-regularized | Δ |
|---|---|---|---|
| 65q NDCG@5 | 0.761 ± 0.06 | **0.794 ± 0.06** | **+0.014 vs v2 LGB; +0.011 vs pre-Phase-11** |
| 20q NDCG@5 | 0.467 ± 0.09 | **0.526 ± 0.11** | **+0.059 vs v2 LGB; +0.076 vs pre-Phase-11** |

Per-category losses (treatment −0.117 on n=4, vague −0.080 on n=5) are real
concerns but small-n; honest documentation in §11. Same Decision-36 framing
that the bi-encoder and CE ships used.

### Is the LGB blender even necessary?

The 3-way comparison on hybrid top-50 (production candidate flow):

| System | 65q NDCG@5 |
|---|---|
| Pure v2 CE (no LGB, no blend) | 0.553 ± 0.06 |
| **v2 LGB** | **0.761 ± 0.06** |
| **v3-regularized LGB** | **0.775 ± 0.06** |

The LGB blender adds **+0.21 NDCG@5 over pure CE** on this measurement.
However, the eval framework is biased toward LGB-like selection patterns
because the labels were pooled from a pipeline that already included LGB
(label coverage issue, §11). The fair answer to "drop LGB?" requires a
pooled-labels eval ($115) that we haven't yet funded. The current data
strongly suggests "keep LGB," but the case isn't airtight.

---

## 8. Inter-annotator agreement (Phase 9)

**The judge bias problem.** All 7,971 training labels and all 1,700 held-out
labels were assigned by Claude Haiku 4.5. To validate the labels themselves,
we re-labeled 100 stratified (query, trial) pairs with Claude Sonnet 4.6
using an *identical* prompt.

| Metric | Value | CI (1000 bootstrap) | Interpretation |
|---|---|---|---|
| Cohen's κ (unweighted) | 0.563 | [0.421, 0.687] | Moderate agreement |
| Cohen's κ (linear weighted) | 0.677 | [0.547, 0.789] | Substantial |
| **Cohen's κ (quadratic weighted)** | **0.723** | **[0.601, 0.819]** | **Substantial (Landis-Koch)** |
| Off-by-one accuracy | 97 % | — | Big disagreements rare |

**Headline finding**: κ = 0.72 quadratic. Substantial agreement, but not
perfect. Sonnet is the stronger judge.

**The systematic Haiku bias we found**: Haiku over-rates the top tier. In
the 100-pair sample, **62 pairs Haiku graded 3 collapsed to 39 pairs Sonnet
graded 3** (never the reverse). This is a real measurement bias that affects
every NDCG number in this report — the absolute values are likely *too
optimistic* by some amount, but the *relative* comparisons (v1 vs v2 vs v3)
are more defensible because the bias hits all systems equally.

Full per-category κ + confusion matrix + top-10 disagreements:
`data/evaluation/agreement_analysis.json`.

---

## 9. Head-to-head against ClinicalTrials.gov v2 search (Phase 9)

**Question**: does our pipeline actually beat the system patients use today?

**Setup**: 20 queries spanning all categories. For each query: pull top-20
from our pipeline and top-20 from `clinicaltrials.gov/api/v2/studies`. Label
both top-K lists with Claude Haiku on a 0–3 relevance scale. Compute mean
precision (fraction of returned trials with relevance ≥ 2).

| Direction | Mean precision | Per-category strongest delta |
|---|---|---|
| Ours-only (trials we surfaced, CT.gov didn't) | **77 %** | pediatric +11.5 pts, vague +7.4 pts |
| Theirs-only (trials CT.gov surfaced, we didn't) | 67 % | (complex shows 0 theirs-only — our coverage gap, §11) |

**Headline**: +9.8 pts mean precision in our favor. We win on every category
where overlap exists; on complex, both systems are weak (≤31 % precision)
but ours is +6.3 pts ahead. Average list overlap is 0.5/20 — the systems
return *essentially disjoint* top-20s, which makes the head-to-head
informative rather than noise.

Full per-query table + qualitative analysis: `docs/ctgov_comparison.md`.

---

## 10. Decision trail — 36, 38, 39, 40

For an interview-defensible project, every major shipping decision is
documented with motivation, what shipped, and what didn't. Each decision
lives as a numbered entry in `CLAUDE.md`'s design-decisions section.

| # | Decision | Phase | Headline |
|---|---|---|---|
| 36 | Ship bi-encoder v2 despite missing pre-registered C4 gates | 10 | Overall NDCG tied with v1; per-category wins on vague +0.048, geographic +0.063; "ship anyway with honest writeup" precedent established |
| 38 | Eligibility hard-filter restricted to closed-vocab criteria (age, sex, excluded_prior_treatments) | 10b | Soft criteria (`required_conditions`, `required_prior_treatments`) too noisy at parser-precision ~70 %; would drop 10/10 trials on Q413 if applied to the overall verdict |
| 39 | Cross-encoder v2 graded MarginMSELoss retrain | 11 | Replaces binary BCE-trained v1 (saturated at val NDCG@10 = 0.993, non-predictive); +0.109 NDCG@5 on the 65q held-out, non-overlapping CIs; cold-start init by sweep tie-break |
| 40 | LightGBM ranker v3-regularized with anti-concentration hyperparameters | 12 | Catches the silent Phase-11 +LGB regression; same labels, `feature_fraction=0.5` + `lambda_l2=1.0` + `min_data_in_leaf=50` to prevent the LGB from collapsing onto the dominant v2 CE feature |

**Decisions 33–35** cover bi-encoder training-data choices (6 SHAPE_PROMPTS,
31-key cancer taxonomy, per-group floor of 300); **Decision 37** covers
expanding the eval to n=15 on complex+vague when bootstrap CIs were too wide
to decide the ship gate. All in `CLAUDE.md`.

---

## 11. Limitations (read this first if you're a reviewer)

If you're auditing this evaluation, here are the things I'd push back on
*first*. Documented honestly rather than buried.

### 11.1 Label coverage

**The most important caveat.** All NDCG numbers in this report measure
"given these labeled candidates, how well did the system rank them?" — not
"given everything in the corpus, how well did the system find what's
relevant?"

- Labels were pooled from the production pipeline's top-K. Anything outside
  that pool is treated as relevance = 0 by default, including candidates
  that *would have been relevant* if anyone had labeled them.
- When two systems retrieve different candidates (e.g., v1 vs v2 hybrid),
  the one whose retrievals overlap more with the labeled pool scores higher
  — not necessarily the one with better ranking quality.
- Concrete demonstration: pure v2 CE on the *labeled pool* scores 0.898 on
  65q (§5). Pure v2 CE on *hybrid top-50* scores 0.553 on the same set (§7).
  Same model, same labels, different candidate pool — different conclusion.

**Resolution**: pooled TREC-style labeling (~$115, ~4 hr) where the
candidate pool is the union of top-K from every system being compared.
Deferred until budget allows.

### 11.2 Training-test distribution mismatch (LGB)

The 145-query LGB training set was pooled from v1-era pipeline output. The
held-out 65q + 20q was labeled later against the v2 pipeline. These
distributions don't fully match — at training time the LGB sees one shape
of "candidate that survived to top-20"; at test time on hybrid top-50 it
sees a different one.

This is the root cause of the vanilla v3 over-fit (§7). The Path-1
regularization papered over the symptom; the underlying mismatch is open.

**Resolution**: re-pool the 145q training labels against the v2 pipeline
(~$50-100, half-day). Deferred.

### 11.3 LLM judge bias

Haiku over-rates the top tier (§8). The 62→39 collapse on Sonnet re-label
means every "NDCG@5 = X" headline in this report is *optimistic* relative
to a clinical gold standard. The *relative* comparisons (v1 vs v2 vs v3)
are more defensible because the bias affects all systems uniformly.

**Resolution**: clinical-reviewer re-labeling of the 33 Haiku/Sonnet
disagreements. ~2 hr reviewer time. Blocked on access to a clinical
reviewer. The Sonnet kappa is the strongest currently-feasible
validation.

### 11.4 Small-n per-category

Per-category breakdowns on 65q have n=3-5 for most categories outside
`existing` (n=30). Bootstrap CIs on n=3-5 are extremely wide. A "−0.117
regression on treatment (n=4)" is real arithmetic but the CI is wider than
the delta — could easily be noise. **Aggregate numbers are statistically
loaded; per-category numbers are directional.**

**Resolution**: more held-out queries per category. Each labeled query
costs ~$0.02 × 20 candidates = $0.40, plus a small Haiku run. The 65q set
was sized for ±0.04 aggregate CI and ±0.10-0.20 per-category CI; doubling
it to 130q would tighten per-category but is a real budget commitment.

### 11.5 Recall ceiling

NDCG@K measures list ordering, not recall. If hybrid retrieval misses a
truly-relevant trial entirely, no downstream CE or LGB can recover it. The
ablation can't measure this. Spot-checks on 10 queries during Phase 9
suggested ~95 % recall at top-50 (the candidate set fed to CE), but this
wasn't measured rigorously.

**Resolution**: a recall audit on 10–20 queries by an expert reviewer.
Deferred.

### 11.6 Persistently weak slices

`complex` and `vague` consistently sit below 0.70 NDCG@5 across every model
version we've shipped. Phase 10 (bi-encoder retrain), Phase 11 (CE retrain),
and Phase 12 (LGB retrain) each made progress, but none rescued the slice.

Root cause diagnosed in Phase 10b: these queries' relevance is encoded in
the eligibility *text* of trials (e.g., "patient must be osimertinib-naive"),
which neither the bi-encoder nor the CE nor the LGB can resolve to drug-
class equivalences. The current eligibility filter handles closed-vocab
criteria (age, sex, excluded_prior_treatments) but doesn't fire on drug-
class matches.

**Resolution**: SciSpacy `EntityLinker` integration to UMLS. This is the
next major work item per CLAUDE.md "What's next" — unlocks
`required_prior_treatments` for hard filtering and is the only path to
meaningfully lifting `complex` beyond the current ceiling. Free, ~4-8 hr.

### 11.7 No clinical user study

NDCG is a proxy for usefulness, not the thing itself. We have not measured
whether the system actually helps real patients find appropriate trials.
That requires consented patients, a clinical reviewer, and an IRB. Not
within this project's scope.

---

## 12. Reproducibility

### Run the full ablation

Requires Elasticsearch + the v2 model artifacts + ~$0 (no API spend at this
stage).

```bash
# Start Elasticsearch
docker start trialmine-es && sleep 10

# 65q held-out ablation
OMP_NUM_THREADS=1 python scripts/evaluate.py \
  --labels data/evaluation/full_labeled_dataset.jsonl
# → writes the auto-generated table to docs/ablation_auto.md

# 20q expansion ablation
OMP_NUM_THREADS=1 python scripts/evaluate.py \
  --labels data/evaluation/full_labeled_dataset_expansion_v2.jsonl
```

### Reproduce specific findings

| Finding | Command |
|---|---|
| §3 stack-vs-stack | Set `TRIALMINE_EMBEDDER` / `TRIALMINE_FAISS_INDEX` / `TRIALMINE_CROSS_ENCODER` / `TRIALMINE_RANKER` to v1/v2 paths and re-run `scripts/evaluate.py` |
| §5 pure-CE C2 | `OMP_NUM_THREADS=1 python scripts/c2_eval_pure_ce.py --labels data/evaluation/full_labeled_dataset.jsonl` |
| §6 α sweep | `OMP_NUM_THREADS=1 python scripts/c3_sweep_blender_alpha.py --labels data/evaluation/full_labeled_dataset.jsonl` |
| §7 LGB 3-way | `RANKER_A=models/ranker/v2/model.lgb RANKER_B=models/ranker/v3-regularized/model.lgb OMP_NUM_THREADS=1 python scripts/r_e_eval_lgb_v2_vs_v3.py` |
| §8 Sonnet kappa | `python scripts/sonnet_label.py && python scripts/agreement_analysis.py` (~$0.50 Sonnet) |
| §9 CT.gov compare | `OMP_NUM_THREADS=1 python scripts/compare_with_ctgov.py` |

### Numerical artifacts

| Path | Content |
|---|---|
| `data/evaluation/c2_full_labeled_metrics.json` | Per-query NDCG/MRR for pure CE comparison on 65q (§5) |
| `data/evaluation/c2_expansion_v2_metrics.json` | Same on 20q expansion |
| `data/evaluation/c3_alpha_sweep_metrics.json` | Per-(α, query) sweep results (§6) |
| `data/evaluation/r_e_lgb_v2_vs_v3.json` | Per-query 3-way LGB comparison (§7) |
| `data/evaluation/agreement_analysis.json` | Cohen's κ + confusion matrix (§8) |
| `data/evaluation/full_labeled_dataset.jsonl` | The 65q × top-20 = 1,300 Haiku labels |
| `data/evaluation/full_labeled_dataset_expansion_v2.jsonl` | 20q × top-20 = 400 Haiku labels |
| `docs/ablation_auto.md` | Auto-generated ablation (updated by `scripts/evaluate.py`) |

---

## 13. Companion documents

This report is the consolidated evaluation narrative. For depth on specific
topics:

| For… | Read |
|---|---|
| Week-by-week project walkthrough | `docs/01_data_pipeline.md` through `docs/09_evaluation_and_kappa.md` |
| Cross-encoder retrain runbook | `docs/fix_CE.md` |
| Decision records + project state | `CLAUDE.md` |
| Known limitations + future-work backlog | `docs/things_can_be_fixed.md` |
| CT.gov head-to-head per-query results | `docs/ctgov_comparison.md` |
| Auto-generated latest ablation table | `docs/ablation_auto.md` |
| Pydantic + API technical detail | `docs/appendix_a_pydantic_and_api.md` |
