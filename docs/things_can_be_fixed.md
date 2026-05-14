# Things That Can Be Fixed

> A running list of small-to-medium issues in the codebase that are
> known, defensible to leave for now, but worth fixing when there's
> time. Each entry includes: where the issue is, what it costs us,
> the proposed fix, and how risky/expensive the fix is.
>
> Order is by *leverage* (impact ÷ effort), not by severity.

---

## 1. Character-based truncation in `prepare_trial_text`

**File:** `src/TrialMine/models/embeddings.py:141-144`

**The code:**

```python
# Rough truncation: ~4 chars per token, 512 tokens ≈ 2048 chars
max_chars = 2048
if len(text) > max_chars:
    text = text[:max_chars]
```

### Why it's there

BioLinkBERT-base has a hard 512-token input limit. *Some* truncation is
mandatory. The author chose character-based truncation (rather than
running the tokenizer) for two reasons:

1. **Speed** — `len(text)` is O(1); tokenizing is O(n) and needs the
   tokenizer loaded.
2. **Pre-filter before the encoder's tokenizer** — the
   `SentenceTransformer.encode` call already truncates at 512 tokens
   internally, so this is a "don't pass a 10 KB string to the
   tokenizer" safeguard.

### Why it's wrong

Character-truncation is a poor proxy for token-truncation:

- **The 4-chars-per-token rule is wrong for biomedical text.**
  In-vocab terms like `"pembrolizumab"` (13 chars) tokenize to 1 token;
  OOV terms like `"chondrosarcoma"` (14 chars) tokenize to 4 subwords.
  So 2048 chars could be anywhere from ~250 to ~700 tokens depending
  on the trial. We never hit exactly 512 tokens.
- **Slicing mid-token produces garbage.** `text[:2048]` cuts at byte
  2048 with no boundary awareness. If character 2048 lands inside
  `"EGFR-positive"`, the tokenizer sees `"EGFR-pos"` and produces
  different subwords than it would for the full term. The embedded
  vector reflects junk at the boundary.
- **Sometimes more aggressive than the encoder's own truncation.** A
  trial whose text spans 2200 chars but only 350 tokens gets cut to
  ~320 tokens by us — the encoder would have kept all 350.

### How often it hurts us — *measured on the actual corpus (140,723 trials)*

| Statistic | Value |
|---|---|
| Total trials | **140,723** |
| Trials with concatenated text > 2048 chars (truncation triggers) | **8,448 (6.0%)** |
| Trials > 3,000 chars | 1,784 (1.3%) |
| Trials > 4,000 chars | 538 (0.4%) |
| Trials > 5,000 chars | 131 (0.1%) |
| Trials > 8,000 chars | 4 (0.0%) |
| **Max length observed** | **12,224 chars** |

Distribution of concatenated text length (`title [SEP] conditions [SEP] brief_summary`):

| Percentile | Length (chars) |
|---|---|
| min | 74 |
| p25 | 458 |
| median | 656 |
| mean | 867 |
| p75 | 1,051 |
| p90 | 1,686 |
| p95 | 2,167 ← **truncation starts biting around here** |
| p99 | 3,184 |
| max | 12,224 |

**Of the 8,448 trials that do get truncated:**

| Statistic | Chars dropped |
|---|---|
| median | 437 (~110 tokens at 4 chars/tok) |
| mean | 648 |
| p90 | 1,588 |
| max | 10,176 |

### What this actually costs us

**Aggregate:** modest. 94% of trials never see the cut. The median
trial is 656 chars — well under the limit.

**Tail:** real. About 6% of trials lose roughly 110 tokens of content
on average (median of dropped chars / 4), and ~1.3% lose 250+ tokens.
The cost lands on **long-summary trials**, which tend to be:

- complex multi-arm protocols (description goes into detail about each arm)
- rare-disease trials (long etiology and inclusion rationale in the summary)
- biomarker-stratified trials (extensive description of the biomarker subgroups)

These are exactly the trial categories where the Week 9 eval flagged
the worst per-category NDCG: **complex** (NDCG@5 = 0.626) and **vague**
(0.688). The truncation isn't proven to be the dominant cause of those
failures, but it's a plausible contributor for the long-summary subset.

### The fix

One-line change. Delete the manual truncation and let the encoder's
tokenizer handle it cleanly at the token boundary:

```python
def prepare_trial_text(self, trial: Trial) -> str:
    parts = []
    if trial.title:
        parts.append(trial.title)
    if trial.conditions:
        parts.append(" ".join(trial.conditions))
    if trial.brief_summary:
        parts.append(trial.brief_summary)
    return " [SEP] ".join(parts) if parts else ""
```

`SentenceTransformer` loads BioLinkBERT-base with `max_seq_length=512`
by default, and its tokenizer truncates at that boundary on a token
edge — no mid-word cuts, no wasted budget.

### Cost / risk of fixing

- **Code change:** delete 4 lines.
- **Re-indexing:** rebuild `data/faiss_finetuned.index` (~hours on the
  existing pipeline). Required because the embeddings will change for
  the ~6% of trials currently affected.
- **Risk:** none. Embeddings for the other 94% are bit-identical
  (same input text → same output vector). The 6% that change move from
  "embedding of corrupted truncated text" to "embedding of cleanly
  truncated text" — strictly an improvement.
- **Observable benefit:** measurable on the 60-query labeled set,
  particularly the *complex* category. Re-eval after rebuild.

### Verdict

**Fix when there's a free re-index window.** Not urgent (94% of corpus
unaffected), but it's a one-line cleanup that removes a confusing piece
of code and marginally improves the long-tail. The character ceiling
also masks a deeper question — *should* we be truncating at 512 tokens,
or should we use multi-vector encoding to keep all the content? See
**Issue #4** below.

---

## 2. Synthetic patient queries are 0.6% of training data — most batches never see patient prose

**Files:** `configs/training_data.yaml`, `scripts/generate_training_data.py`,
`configs/training/embeddings.yaml`, `scripts/finetune_embeddings.py`

### Why it's there

The bi-encoder fine-tuning data has three sources (see CLAUDE.md
"Training data generation"):

1. **Source 1** — 242K metadata-derived pairs. Keyword-shaped queries
   like `"lung cancer"`, `"EGFR for non-small cell lung cancer"`,
   `"phase 2 melanoma trial"`.
2. **Source 2** — **1,500 synthetic patient-style queries** from Claude
   Haiku. Fluent sentences like *"my mother has stage IV NSCLC with an
   EGFR mutation, are there trials for her?"*
3. **Source 3** — hard-negative mining. Takes every `(query, positive)`
   from Sources 1+2 and emits 3 triplets per pair (3 different hard
   negatives) → ~730K final triplets.

Source 2 was sized 1,500 as "distribution-shift insurance" — meant to
seed the patient-prose manifold without ballooning Haiku API spend.
The author appears to have assumed 1,500 was enough to nudge the
encoder; there's no record of a coverage analysis.

### Why it's wrong / what it costs us

After Source 3 expansion, synthetic triplets are ~4,500 out of ~730K
total — **0.6% of training data.**

With MNRL at batch size 32, the probability a batch contains *zero*
synthetic anchors is `(1 − 0.006)^32 ≈ 83%`. So **roughly 5 out of
every 6 training steps the model never sees a patient-style sentence
as an anchor.** The natural-language → trial-text signal is essentially
a rounding error in the gradient.

This is consistent with the Week 9 eval failure shape:

| Category | NDCG@5 | What the anchor looks like |
|---|---|---|
| common, rare, treatment, existing | > 0.92 | keyword-shaped (close to Source 1) |
| **complex** | **0.626** | long patient sentence, multiple constraints |
| **vague** | **0.688** | natural-language paraphrase |

The two worst categories are exactly the queries that look like
Source 2. This is *correlational*, not proven causal — complex queries
are harder for other reasons too (Issue #3's eligibility-text exclusion
is a separate contributor, see Q413/Q416 there). But the coverage gap
matches the failure pattern.

### The fix — three independent levers, can stack

1. **Generate more synthetic queries.** Bump from 1,500 to ~10,000 via
   the existing `--resume`-able script. Cost: ~$17 in Haiku API
   (10K × ~$0.0017/query). Brings synthetic share to ~4%; at batch 32
   that's `(1 − 0.04)^32 ≈ 27%` of batches with zero synthetic — down
   from 83%.
2. **Upsample synthetic rows during training.** Repeat each synthetic
   triplet 5–10× in the training JSONL. Costs nothing, immediately
   biases the gradient toward patient prose. Risk: overfitting on
   1,500 unique anchors — *always do lever 1 before lever 2.*
3. **Bump `batch_size: 32 → 128`** in
   `configs/training/embeddings.yaml`. The current 32 looks like the
   sentence-transformers default rather than a measured choice (no
   sweep in MLflow, no comment justifying it). On A100 80GB with
   BioLinkBERT-base + fp16 + seq=512, 128 fits with headroom. Two
   gains at once: (a) 255 negatives per anchor instead of 63 — harder
   MNRL denominator at no extra data cost; (b) at synthetic share 4%,
   `(1 − 0.04)^128 ≈ 0.5%` of batches lack a synthetic anchor — i.e.
   ~99% coverage per step.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Synthetic generation | ~$17 Haiku spend, ~3 hr wall-clock (resumable). |
| Re-finetune | One A100 run. Current run was 288 min at batch 32; batch 128 should be *faster* per epoch (better GPU utilization) — guess ~3–4 hr for 3 epochs. |
| Re-index | Rebuild `data/faiss_finetuned.index` (~hours). Required — bi-encoder weights change. |
| Re-eval | `scripts/build_full_eval_dataset.py --resume` against the 60-query labeled set, ~$1. **Caveat:** labels were drawn from the *current* pipeline's top-20, so this measures list-ordering only. To honestly measure recall improvement, re-label the new model's top-20 — adds ~$1 and a few hours. |
| Risk to existing wins | Low. Source 1 still dominates training (96%); we're adding distribution coverage, not redirecting the optimization. |
| Risk of overfitting | Real on lever 2 alone (only 1,500 unique synthetic anchors). Mitigated by always doing lever 1 first. |

### Verdict

**Do soon — ~$20 of API spend addresses two of the worst NDCG
categories from the headline eval.** Cheap enough that the answer is
"why not." Sequencing:

1. Generate 10K synthetic queries (lever 1) — overnight Haiku job.
2. Re-finetune at batch 128 (lever 3 baked in) — one A100 run.
3. Rebuild FAISS, re-eval on the 60-query labeled set.
4. If `complex` / `vague` lift is below expectations, layer in lever 2
   (upsampling) and re-run.

Clean experimental design either way: a single intervention tested
against a labeled set with known per-category baselines. If the needle
moves on `complex` / `vague`, we've validated the coverage hypothesis.
If it doesn't, we've ruled it out and the next suspect is Issue #3
(eligibility text not embedded) — both outcomes are valuable.

---

## 3. Per-cancer-group sampling cap is a ceiling without a floor — rare cancers stay underrepresented

**Files:** `configs/training_data.yaml:2`,
`scripts/generate_training_data.py:140-159`

### Why it's there

Training data is stratified across 23 cancer groups via
`max_trials_per_cancer_group: 2000`. The implementation is
`sample_trials_stratified` (lines 140–159):

```python
for group, group_trials in trials_by_group.items():
    if len(group_trials) <= max_per_group:
        sampled[group] = list(group_trials)
    else:
        sampled[group] = rng.sample(group_trials, max_per_group)
```

The cap exists to prevent the most common cancer groups (breast, lung,
prostate) from drowning out the taxonomy — without it, breast cancer
alone would be ~30% of training pairs and the bi-encoder would
effectively become a "breast-cancer recognizer."

### Why it's wrong / what it costs us

The cap is a **ceiling, not a floor.** It triggers only for groups
*above* 2,000 trials — the common ~5–7 cancers. The long tail
(sarcoma, medulloblastoma, mesothelioma, etc.) is untouched because
there simply aren't enough trials in the corpus to cap.

Approximate corpus distribution after Source 1 (metadata) pair
generation:

| Group | Trials in corpus | After cap | Final triplet share |
|---|---|---|---|
| breast, lung, prostate, colon, leukemia, lymphoma, melanoma | each 3–15K | 2,000 each | ~3.3% each |
| sarcoma | ~600 | ~600 (cap inert) | **~1.0%** |
| medulloblastoma | ~150 | ~150 (cap inert) | **~0.25%** |

With MNRL at batch 32 and a rare class at ~1% share, the probability a
batch contains zero sarcoma anchors is `(1 − 0.01)^32 ≈ 72%`. So **for
roughly 3 out of every 4 training steps, the gradient sees no sarcoma
signal at all.**

This shows up directly in the bi-encoder eval: the single 20-query
benchmark loss was *"sarcoma clinical trials for young adults"* — the
lone rare-cancer query, and the fine-tuned model lost there. The
Week 9 60-query eval doesn't isolate sarcoma as its own category, but
the `complex` (0.626) and `vague` (0.688) failures include rare-cancer
constituents.

### The asymmetric-cap trap

The instinctive "raise the cap to 4,000" makes the failure mode
**worse**, not better:

| Cap policy | Common groups | Sarcoma share | P(batch-32 contains ≥1 sarcoma) |
|---|---|---|---|
| Current (cap 2,000) | 2,000 each | 1.0% | 28% |
| Raise to 4,000 | 4,000 each | **0.67%** | **20%** |
| Lower to 500 | 500 each | **~3%** | **62%** |
| Add a floor (upsample rare to 1,000) | 2,000 each | **~4%** | **73%** |

Raising the cap gives the model *more* breast/lung data (where it's
already saturated) while shrinking the gradient share of the cancers
it's losing on. Lowering the cap or adding a floor is the right
direction.

### The fix — two levers, both cheap

1. **Add a `min_trials_per_cancer_group` floor.** Upsample
   (with-replacement repetition) rare-cancer trials up to a target
   floor — e.g., 1,000. New config entry in `configs/training_data.yaml`;
   ~30 LOC change in `sample_trials_stratified`. Risk: overfitting on
   the same handful of trials per rare cancer (a sarcoma trial seen
   3× in one epoch contributes the same gradient direction each time).
   Mitigated by lever 2.
2. **Over-stratify synthetic generation toward the tail.** When
   generating Source 2 synthetic queries (especially when expanding
   from 1,500 → 10,000 per Issue #2), bias the per-group quota *away*
   from proportional and toward the tail — e.g., 200 sarcoma queries
   instead of the proportional ~65. Same total Haiku spend, but
   produces *diverse* rare-cancer anchors instead of repeated ones.

Optional third lever:

3. **Lower the cap to 500.** Treats all 23 groups equally, which is
   what stratified contrastive training actually wants. Trade-off:
   loses diversity within common cancers (5K unique breast trials →
   500). Probably acceptable since common-cancer NDCG is already
   saturated (> 0.92), but worth measuring.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Code changes | ~30 LOC for floor + upsampling in `sample_trials_stratified`; ~20 LOC to bias synthetic quotas. |
| Data regeneration | Re-run `scripts/generate_training_data.py` (~30 min if `--skip-synthetic`, hours otherwise). |
| Synthetic regeneration | Folded into Issue #2's expansion — no extra cost. |
| Re-finetune | One A100 run. **Pairs with Issue #2's re-finetune** — same compute, two issues fixed. |
| Re-index | Rebuild `data/faiss_finetuned.index`. |
| Re-eval | 60-query labeled set, ~$1. **Strongly recommend** adding 5–10 explicit rare-cancer queries to the eval set first, otherwise tail improvements will be invisible in the aggregate NDCG (sarcoma at 1% of training data is also ~1% of any unstratified eval set). |
| Risk to existing wins | Low. The fix rebalances; it doesn't redirect. Worst case: rare-cancer NDCG flat, common-cancer NDCG unchanged. Best case: rare-cancer lift, common holds. |
| Risk of overfitting on upsampled rare cancers | Real on lever 1 alone (sarcoma trial seen 3× gives correlated gradient). Lever 2 supplies the diversity that mitigates this. |

### Verdict

**Do this in the same re-finetune cycle as Issue #2.** Both are
training-data distribution fixes, both require the same A100 run, and
they stack cleanly — Issue #2 expands synthetic *volume*, this issue
fixes synthetic *stratification* plus the trial-side floor.

Sequencing inside that cycle:

1. Generate 10K synthetic queries with **over-stratified** quotas
   (lever 2 here + Issue #2 lever 1).
2. Run `generate_training_data.py` with the new floor (lever 1) and
   the cap unchanged at 2,000.
3. Re-finetune at batch 128 (Issue #2 lever 3).
4. Re-eval on 60-query set + ≥5 explicit rare-cancer queries.
5. If common-cancer NDCG drops, revert the floor. If rare-cancer lifts
   and common holds, ship.

### Principle worth keeping

In contrastive fine-tuning, what matters is the **share** of a class
in the batch, not its absolute count. Adding more of class A doesn't
help class B — it *competes* with class B for gradient. If the failure
mode is on a tail class, the only useful interventions are (a) make
the tail proportionally larger, or (b) make the head proportionally
smaller. Raising a ceiling does neither.

---

## 4. Single-vector encoding caps semantic search at 3 fields

**Files:** `src/TrialMine/models/embeddings.py`,
`scripts/build_index.py`, `src/TrialMine/retrieval/semantic.py`

### Why it's there

BioLinkBERT-base has a hard 512-token input limit (see Issue #1 for
*why* — pretrained positional embedding table has 512 rows; this is a
property of the weights, not a configurable knob).

To stay under that ceiling, `prepare_trial_text` concatenates only
**three** of the trial's text fields:

```python
parts = [trial.title, " ".join(trial.conditions), trial.brief_summary]
text = " [SEP] ".join(parts)
```

Three fields are excluded from the semantic index:

| Field | Excluded from semantic? | Indexed by BM25? |
|---|---|---|
| `interventions` | ❌ excluded | ✅ |
| `eligibility_criteria` | ❌ excluded | ✅ |
| `detailed_description` | ❌ excluded | ❌ (also out) |

### Why it's wrong / what it costs us

The hybrid story (BM25 + semantic + RRF) was supposed to compensate:
"BM25 catches lexical, semantic catches paraphrase." But for any query
whose intent lives in the *excluded* fields, **semantic contributes
zero signal** — RRF only sees a BM25-side match.

Concrete failure modes from the Week 9 eval:

| Query ID | Query | Failure | Likely root cause |
|---|---|---|---|
| Q413 | *"failed osimertinib"* | osimertinib-naive trials at rank 5 | Prior-treatment requirement lives in `eligibility_criteria` — invisible to semantic |
| Q416 | *"post-trastuzumab progression"* | STOP-HER2 trial at #1 (opposite clinical situation) | Same — eligibility text not embedded |
| (general) | *"treatments using anti-PD-1 antibodies"* | Trials whose interventions list `pembrolizumab` / `nivolumab` (no lexical overlap with "anti-PD-1") | `interventions` not embedded; BM25 misses paraphrase |

These are the *complex* and *vague* category failures driving the
NDCG@5 = 0.626 / 0.688 floors.

### The fix — multi-vector encoding

Split each trial into multiple sub-documents, embed each independently
(staying within 512 tokens per chunk), store all vectors in FAISS with
a back-pointer to the parent `nct_id`, dedupe + aggregate at query time.

**Concrete chunking strategy:**

| Chunk | Source fields | Typical token count |
|---|---|---|
| `core` | title + conditions + brief_summary | ~200–500 |
| `interventions` | interventions list (joined) | ~30–150 |
| `eligibility` | eligibility_criteria | ~300–800 (split if >512) |

Each chunk is embedded separately by the same fine-tuned BioLinkBERT.
FAISS index stores `(chunk_vector, nct_id, chunk_type)`.

**Query-time aggregation:**

```python
def semantic_search(query, top_k):
    q_vec = embed(query)
    chunk_hits = faiss.search(q_vec, top_k * 5)   # over-retrieve
    by_doc = {}
    for nct_id, chunk_type, score in chunk_hits:
        by_doc.setdefault(nct_id, []).append((chunk_type, score))
    # max-aggregation: doc score = highest-scoring chunk
    doc_scores = {nct_id: max(s for _, s in chunks)
                  for nct_id, chunks in by_doc.items()}
    return sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
```

`max` works for "any chunk matches strongly → doc is relevant".
Alternative: weighted sum (eligibility hits less than core hits) — tune
on the labeled eval.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Code changes | ~3 files, ~150 LOC: `embeddings.py` (chunking), `build_index.py` (write chunk_type + nct_id alongside vectors), `semantic.py` (dedup + aggregate). FAISS still works — `IndexFlatIP` doesn't care what doc the vectors map to. |
| Index size | ~2.5–3× current FAISS index (~412 MB → ~1.0–1.2 GB). Disk + RAM at query time both grow. |
| Re-indexing time | Same as current full rebuild + a small constant for the extra chunks. Hours on the existing pipeline. |
| Re-finetuning | **Not required.** Same BioLinkBERT, same MNRL training data — only the *inference-time* chunking changes. Could optionally re-finetune with chunk-aware negatives later. |
| Retrieval latency | One extra dedup-and-aggregate pass over ~1000 chunks. Negligible (<5 ms). |
| Risk to existing wins | None of the current single-vector behavior is destroyed — the `core` chunk is exactly the existing single vector minus the truncation bug. So worst case, multi-vector ≥ current. |

### Verdict

**Plan to do this.** Higher leverage than Issue #1 (which is itself
a freebie). Multi-vector encoding is the standard production fix for
the BERT 512-token cap and would directly address the highest-NDCG-loss
queries from the Week 9 eval.

**Sequencing:**
1. Fix Issue #1 first (one-line truncation cleanup) and re-eval — gives
   a clean baseline for the `core` chunk.
2. Implement multi-vector chunking. Start with `core` + `eligibility`
   (skip interventions; it's short and BM25 handles drug names well).
3. Re-eval against the 60-query labeled set. Expect lift on *complex*
   and queries with eligibility-related intent.
4. Add `interventions` chunk only if step 3's lift on drug-class-paraphrase
   queries is below expectations.

**Why not just switch to a long-context model (BGE-M3, Nomic Embed)?**
That's the alternative path. Tradeoff: lose BioLinkBERT's biomedical
pretraining (most long-context encoders are general-domain) and have
to re-finetune from scratch on our 586K triplets, possibly with worse
biomedical lexical coverage. Multi-vector keeps the model investment
intact. Worth revisiting if a *biomedical* long-context encoder ships.

---

## 5. Cross-encoder is trained on binary labels — converges to a disease matcher, not a graded ranker

> **✅ CLOSED 2026-05-13.** Fixed by Phase 11 (graded MarginMSELoss retrain). Headline
> lift on the same held-out 65-query benchmark: pure CE NDCG@5 **0.789 → 0.898**
> (Δ = +0.109, non-overlapping bootstrap CIs); val/prod gap closed from ~16 pt to 3–8 pt;
> blender α default flipped from 0.7 (RRF-dominant safety cap) → 0.3 (CE-dominant) per
> the C3 sweep. v1 binary CE preserved at `models/cross-encoder/fine-tuned-v1/` for
> revert. Full writeup in `docs/evaluation-report.md §10` + `CLAUDE.md` Decision 39.
> See `docs/fix_CE.md` for the end-to-end runbook this followed.

**Files:** `configs/training/cross_encoder.yaml`,
`scripts/finetune_cross_encoder.py`,
`models/cross-encoder/fine-tuned/metadata.json`

### Why it's there

The cross-encoder was trained on the bi-encoder's existing triplet
data. Each `(query, positive, negative)` triplet was split into two
binary-labeled pairs — `(query, positive, 1.0)` and
`(query, negative, 0.0)` — and the model was fine-tuned with
`BinaryCrossEntropyLoss` against those 0/1 targets. 100K triplets →
200K binary pairs, one epoch on a T4 in ~73 minutes.

This was the path of least resistance: the triplet pipeline already
existed for the bi-encoder, so reusing it for the cross-encoder
required no new data labelling. At the time, the 6K Haiku-graded
relevance pairs (`data/evaluation/labeled_queries.jsonl` and
siblings) were thought of as *evaluation* data, not *training* data.

### Why it's wrong / what it costs us

The model converges to *"same condition + same intervention as
positive → push toward 1; same condition + different intervention →
push toward 0"* — essentially a disease/intervention matcher, not a
graded relevance ranker. The Week 5 cross-encoder eval is brutal on
this:

> *"Fine-tuned BioLinkBERT CE (pure CE) HURTS: NDCG@5 = 0.657
> (−19.5%) — binary training labels don't teach graded relevance"*

The val/production NDCG disconnect is the clearest evidence:

| Metric | Value | What it means |
|---|---|---|
| Val `CERerankingEvaluator_ndcg@10` (in `metadata.json`) | **0.993** | Trivially easy — 1 positive + 1 negative per query, where the negative is a Source-3 hard negative the model can spot by vocabulary alone. |
| Production NDCG@5 (pure CE replacement on 60-query labeled set) | **0.657** | Real ranking task — 50 candidates per query with a long tail of "kind of relevant" (tier-2) trials. The binary rule produces overconfident, near-binary scores that destroy ordering. |

That 33-point gap is not a training-not-converged issue — early
stopping correctly fired at ~1 epoch when val NDCG plateaued. It's a
**training-converged-on-the-wrong-objective** issue. More epochs
make it *worse* (overfitting on triplet patterns), not better.

The blender currently masks this. LightGBM v2 takes CE as a feature
(importance 1673 — the #1 feature) and decides how much to trust it;
the production blended score is `0.7 · RRF + 0.3 · CE_sigmoid`, with
the 0.7 RRF weight doing the actual ordering work. So the
binary-trained CE provides *some* signal — but it's a feature, not a
ranker. Pure-CE replacement hurts because the CE was never trained
to rank.

### Knock-on symptoms

- **Val metric is not predictive of production.** Picking the best
  checkpoint by `CERerankingEvaluator_ndcg@10 = 0.993` picks the
  checkpoint that's best at the trivial 1-pos-1-neg task, not the
  one that's best at production ranking. The two could easily
  disagree.
- **Blender weights are constrained.** Until the CE is trustworthy
  on its own, the blender can't lean on it harder than 0.3. The
  Decision-22-era reasoning ("CE is a feature, not the final score")
  is correct *given* the current CE, but it's a workaround, not a
  design ideal.
- **LightGBM training data is partially poisoned.** Six of the eleven
  LightGBM features include the CE score directly or derivatives —
  if the CE is overconfident on disease matches, the blender
  inherits that bias.

### The fix — three coordinated changes, all data/loss/eval, no inference change

The CE inference path doesn't need to change. The fix lives entirely
in how the CE is trained.

**1. Switch the training data from binary triplets to graded pairs.**
The ~7K Haiku-graded `(query, trial, grade ∈ {0,1,2,3})` rows
already exist, scattered across `labeled_queries.jsonl`,
`test_labels.jsonl`, `train_labels_extra.jsonl`, and
`full_labeled_dataset.jsonl`. Split **by query** (not by pair, to
avoid lexical leakage), holding out the 60-query
`full_labeled_dataset.jsonl` as a never-touched test set. From the
~145 remaining queries, form per-query ranking pairs of the form
*"doc A is more relevant than doc B"* whenever
`grade(A) > grade(B)`. With ~30 graded docs per query and 4 grade
tiers, this yields a training set of roughly **40–50K graded pairs**
— 4–5× smaller than the current 200K binary set, but every example
carries graded supervision.

**2. Swap the loss from `BinaryCrossEntropyLoss` to `MarginMSELoss`.**
`MarginMSELoss` is shipped by `sentence-transformers` out of the
box and is the SOTA recipe for cross-encoder rerankers
(MS-MARCO leaderboard, Hofstätter et al. distillation papers). For
each training pair, the teacher margin is
`grade(A) − grade(B)`; the model's predicted margin is
`score(q, A) − score(q, B)`; the loss is MSE between the two. The
model only has to get the *relative* order right, which is what
ranking actually needs. Absolute calibration emerges as a side
effect.

Alternative considered: pure MSE on rescaled grades (treat 0/1/2/3
divided by 3 as a regression target). Simpler than MarginMSE but
doesn't optimize ranking directly — useful as a baseline only.
Listwise / LambdaRank-style losses are more principled but
`sentence-transformers` doesn't ship one for cross-encoders, so
deferred.

**3. Fix the val metric.** Replace the degenerate 1-positive +
1-negative-per-query val task with a production-shaped task: for
each held-out val query, pool ~30 graded candidates and let
`CERerankingEvaluator` measure NDCG@10 over the full pool. The
val NDCG will *drop* from 0.99 to ~0.80–0.85 — and that's the point.
A val metric that's hard enough to plateau slowly is the
diagnostic that's currently missing. The val/production gap should
close from 33 points to roughly 5.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Code changes | New `scripts/build_ce_graded_training.py` for graded-pair construction (~200 LOC); `finetune_cross_encoder.py` swaps the loss import and dataset shape (~30 LOC change); new `configs/training/cross_encoder_v2.yaml` (mostly a copy with `loss: MarginMSELoss` and a smaller `learning_rate`). |
| Data needed | None new — the ~7K Haiku-graded labels already exist. Optionally regrade with Sonnet 4.6 (~$10, would reduce Haiku-label noise — see κ=0.72 disagreement set in Week 9 analysis). |
| Re-training | ~73 min on T4 was the binary-CE run; graded data is ~4× smaller, so faster per epoch — but graded supervision is harder so more epochs will be useful. Budget similar (~1–2 hr on T4). |
| Re-eval | Already-built `scripts/evaluate_cross_encoder.py` and the 60-query `full_labeled_dataset.jsonl` test set both apply. No new harness. |
| Risk: small training set | ~45K pairs is small for BERT fine-tuning. Higher seed variance (Mosbach 2021). Mitigations: train 3 seeds and average; or warm-start from the existing binary-trained checkpoint (transfer learning preserves the disease/intervention discrimination, the graded fine-tune refines it). |
| Risk: noisy labels | Haiku-graded labels have κ=0.72 vs Sonnet (Week 9). Training on noisy targets caps the achievable model quality. Mitigation: regrade with Sonnet for a cleaner training set. |
| Risk: losing pure-CE's existing lexical strength | Real if you train from scratch on only 45K pairs. The warm-start option (initialize from `models/cross-encoder/fine-tuned/` and fine-tune with MarginMSE) preserves it. |
| Risk to existing wins | Low. The blender currently treats CE as a feature, so even if the new CE is no better on absolute terms, it's used through the same interface. Worst case: revert to the binary checkpoint. |

### Expected outcomes (honest predictions, not promises)

- **Val NDCG@10**: 0.993 → ~0.80–0.85 (good — the metric is now informative).
- **Held-out test NDCG@5 (pure CE replacement)**: 0.657 → ~0.80–0.85. Possibly above the hybrid baseline (0.816), in which case pure-CE replacement becomes viable.
- **Blended scoring NDCG@5**: 0.829 → ~0.85–0.87. Blender weights may need re-tuning — likely shift from 0.7/0.3 toward 0.5/0.5 or further toward CE as it becomes trustworthy.
- **LightGBM CE feature importance**: currently #1 at 1673. With cleaner CE signal, expect it to climb further and the gap to other features to widen.
- **Failure modes that won't be fixed by this alone**: queries whose intent lives in eligibility text (Q413 / Q416 from Week 9) — that's Issue #4 (multi-vector encoding). CE on full-trial text still can't see fields the bi-encoder didn't surface.

### Verdict

**Highest-leverage modeling fix in the codebase right now.** Unlike
Issues #2 and #3 (which need re-finetuning a 430 MB bi-encoder on
730K triplets and rebuilding a 412 MB FAISS index — hours of work
per cycle), this fix needs ~45K training pairs and ~1–2 hr on a T4,
operates on a smaller model checkpoint, and re-eval uses an existing
harness. The 33-point val/production gap is a strong signal that
returns to this work are real, not speculative.

**Sequencing:**

1. Build graded training set from existing `*labels*.jsonl` files
   (split by query, hold out the 60-query test set).
2. Swap loss to `MarginMSELoss`. Warm-start from the current
   binary-trained checkpoint to preserve its lexical signal.
3. Fix the val metric to pooled multi-doc per query.
4. Train + re-eval pure CE on the held-out 60-query set.
5. If pure-CE NDCG@5 > 0.80, re-tune blender weights (try 0.5/0.5
   and 0.3/0.7) and re-eval blended.
6. If pure-CE NDCG@5 < 0.75, the bottleneck is label quality —
   regrade ~2K hardest pairs with Sonnet and retry.

**Pairs naturally with Issues #2 and #3 *only* via re-eval timing,
not training compute** — those re-finetune the bi-encoder; this
re-finetunes the cross-encoder; they're independent training runs.
But doing all three before one round of held-out evaluation gives
the cleanest "what moved the needle?" signal.

### Status — deferred (2026-05-10)

Not executed today. Planned execution: local RTX 4070 (12 GB VRAM,
fits BERT-base CE + MarginMSE at batch 16, fp16 — no Colab/Lambda
needed). Estimated wall-clock ~30–50 min for one warm-started run
(vs ~73 min on T4 for the original binary CE). Existing
`scripts/finetune_cross_encoder.py` is the starting point — needs
loss swap (`BinaryCrossEntropyLoss` → `MarginMSELoss`), data loader
pointed at the new graded JSONL (built from
`data/evaluation/*labels*.jsonl`, query-level split, 60-query
`full_labeled_dataset.jsonl` held out), and warm-start from
`models/cross-encoder/fine-tuned/`. Issue #6's Sonnet-relabel of
the 60-query test set must land before re-eval is published.

### Principle worth keeping

A loss function defines what the model is allowed to learn. Binary
labels teach binary discrimination; graded labels teach graded
ordering. No amount of hyperparameter tuning, longer training, or
larger batch size will turn one into the other. When the val metric
saturates near 1.0 and production is far below, the diagnosis is
almost always *the training signal isn't rich enough* — not *the
optimizer needs more steps*.

---

## 6. Same LLM judge grades both training labels and evaluation labels — judge-bias loop

**Files:** `data/evaluation/labeled_queries.jsonl`,
`data/evaluation/test_labels.jsonl`,
`data/evaluation/train_labels_extra.jsonl`,
`data/evaluation/full_labeled_dataset.jsonl`,
`scripts/build_full_eval_dataset.py`, `scripts/agreement_analysis.py`,
`docs/evaluation-report.md`

### Why it's there

The evaluation pipeline uses **Claude Haiku 4.5** as the LLM judge
for all ~7,200 labeled `(query, trial)` pairs across ~205 queries.
Driven by cost — Haiku is ~3× cheaper than Sonnet — and by the
absence of clinical-reviewer time at the scale required (Week 9
close-out deferred human labelling to ~2 hr of future effort).

A 100-pair subset was re-labelled with **Claude Sonnet 4.6** using an
identical prompt to establish inter-LLM agreement (Decision 32 in
CLAUDE.md; κ = 0.72 quadratic). That subset lives in
`full_labeled_dataset_human.jsonl` and was used purely to measure the
gap between LLMs, never to replace or train against the Haiku labels.

When the cross-encoder retraining plan (Issue #5) lands, the proposed
flow is: train CE on Haiku-graded labels, hold out the 60-query
`full_labeled_dataset.jsonl` as test, then report NDCG@5 against
those *same Haiku-graded labels*. The same labeler grades both train
and test.

### Why it's wrong / what it costs us

This is a **judge-bias loop**, distinct from data leakage. Even with
a clean query-level holdout (the 60-query test set is never touched
during training), the model can still learn the labeler's
*systematic biases* and be rewarded for reproducing them at
evaluation time.

The Week 9 kappa analysis already established that Haiku has
measurable, one-sided bias:

| Finding | Value |
|---|---|
| Cohen's κ (quadratic, Haiku vs Sonnet) | 0.723 [0.601, 0.819] — substantial, not "near-perfect" |
| Off-by-one accuracy | 97% |
| **Confusion matrix top tier** | **62 Haiku-3s collapse to 39 Sonnet-3s; never the reverse** |

So Haiku grades carry two coupled signals:

1. **Real relevance** — the part that agrees with Sonnet (and
   presumably with humans), backed by κ = 0.72.
2. **Haiku-specific bias** — systematic over-rating of tier-2 trials
   as tier-3.

Training a model on Haiku grades + evaluating it against Haiku grades
means the model learns *both* signals and the eval rewards *both*. A
headline like "CE retraining lifted pure-CE NDCG@5 from 0.657 to 0.85"
decomposes into something like *0.7 × real lift + 0.3 × bias
reproduction*, with no way to separate the two from Haiku-only
metrics. The same concern applies — less acutely — to the existing
per-category NDCG breakdowns, the ablation table in
`docs/evaluation-report.md`, and any future "we trained X on this
signal" claim.

The CT.gov head-to-head (Decision 32) is partially insulated: both
systems are judged by the same Haiku, so single-judge bias mostly
cancels in the *relative* comparison. But absolute NDCG numbers and
Issue-#5-style training-evaluation comparisons are not insulated.

### What this does *not* break

For honest scope, three reasons it's bounded rather than catastrophic:

1. **Most of the signal is real.** κ = 0.72 is substantial agreement.
   A model trained on Haiku grades will mostly learn genuine relevance.
2. **The bias is one-sided.** Haiku over-rates the top tier; it
   doesn't flip pairs randomly. Failure mode is *false confidence at
   the top of the ranking*, not chaos.
3. **Relative comparisons are partially protected.** Ablation deltas,
   hybrid-vs-CE-blended margins, and head-to-head wins are less
   affected because the bias mostly cancels in differences.

What *is* compromised: absolute headline numbers, per-category NDCG
breakdowns, and the claim that any specific training intervention
"improved" the model when the eval judge is the training judge.

### The fix — in order of effort

**1. Re-label the 60-query test set with Sonnet 4.6.**
Cost: ~$3 in API spend, ~30 minutes wall-clock. The script already
exists (`scripts/sonnet_label.py`). This makes the test labeler
*different* from the training labeler. Any NDCG lift on the
Sonnet-judged version of the test set is a defensible measure of
real improvement, and the **gap between Haiku-judged and Sonnet-judged
NDCG on the same test set** is itself the bias diagnostic — it
quantifies how much of the reported lift is judge-bias reproduction.

This is the minimum-defensibility fix. Required before publishing
any Issue-#5 (CE retraining) numbers; probably worth backfilling on
the current Week 9 headline numbers too.

**2. Human-labelled gold subset.**
~2 hr of clinical-reviewer time, already flagged in CLAUDE.md "What's
next" as deferred Week 9 work. Even 30 queries × 20 trials = 600
human-graded pairs anchors everything else. The 33 Haiku/Sonnet
disagreement pairs from the κ analysis are the natural starting pool
("validate against humans on the disagreement set"). Once the gold
subset exists, LLM-judged metrics become *consistency checks* against
the gold, not the source of truth.

**3. Multi-judge ensemble for the test set.**
~$15 in API spend (Haiku labels already exist; just add Sonnet,
optionally a third). Take the average grade or majority vote with
ties broken by the stronger judge. Reduces any single-judge bias
without requiring human labels. Weaker than human gold but stronger
than single-judge.

**4. Train and test on *different* judges.**
Sonnet labels the training set, Haiku labels the test set (or vice
versa). Explicitly isolates labeler-specific bias from generalizable
relevance. Probably overkill given (1) is so cheap — but it's the
canonical methodological move if a reviewer pushes back hard.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Fix 1 (Sonnet-relabel test set) | ~$3 API, ~30 min. Script exists. |
| Fix 2 (human gold) | ~2 hr reviewer; cost depends on internal vs external sourcing. |
| Fix 3 (multi-judge ensemble) | ~$15 API. |
| Risk to existing reported numbers | Re-evaluation with Sonnet will probably *lower* the headline NDCG by a few points (Haiku's top-tier over-rating inflates the current numbers). Uncomfortable but correct — the new numbers are more defensible. |
| Risk of over-rotating on judge choice | Sonnet has its own biases too. Mitigation: ship an explicit limitations section (already convention per Decision 32) and pair LLM-judged metrics with the human gold subset wherever it exists. |

### Verdict

**Fix 1 is required before publishing any Issue-#5 results.** $3 and
30 minutes is not a meaningful cost; the alternative is reporting
NDCG numbers that partly measure "the CE successfully learned
Haiku's biases." That's not a defensible claim under interview
scrutiny.

Fix 2 (human gold) is the longer-arc upgrade — defer until a 2-hour
reviewer block is available. Once it exists, all headline metrics
re-anchor to human-judged numbers with LLM-judged numbers reported
alongside.

Fixes 3 and 4 are nice-to-have, off the critical path.

**This issue compounds with Issue #5.** Doing Issue #5 (CE
retraining) without doing this issue (judge separation) produces
results that are technically interesting but methodologically
suspect. The two should be sequenced as one combined cycle:
relabel the test set with Sonnet → retrain CE on Haiku grades →
evaluate on the Sonnet-labelled test set → report both Haiku-judged
and Sonnet-judged NDCG with the gap as the explicit bias diagnostic.

The same logic applies to Issues #2 and #3 (bi-encoder retraining) —
any future re-eval should go through the Sonnet-labelled test set,
not the Haiku-labelled one, to avoid the same loop.

### Principle worth keeping

Train/test split solves *data* leakage; it does not solve *labeler*
leakage. When the same source generates both the supervision and the
evaluation, the model can be rewarded for reproducing the source's
biases even on a clean held-out split. The structural fix is to
separate the source of training signal from the source of
evaluation signal. In LLM-as-judge work this translates to a simple
rule: **don't let the same model grade both sides of any comparison
where one of the models was trained on that judge's grades.**

---

## 7. LightGBM ranker training set is borderline — 145 queries limits HP tuning and per-category resolution

> **Update 2026-05-14 (Phase 12 ship):** v3-regularized ships as the new
> production default (`models/ranker/v3-regularized/model.lgb`), with
> anti-concentration regularization on the same 145q labels. Held-out 65q
> NDCG@5 = 0.794 (vs 0.761 v2 LGB; vs 0.783 pre-Phase-11 baseline) and 20q
> expansion = 0.526 (+0.076 above pre-Phase-11). The 145q-training-set
> ceiling concern is unresolved — the v3 over-fit confirmed it (LOOCV
> 0.989 vs held-out 0.697 for vanilla). Path-1 regularization papered over
> the immediate issue; the deeper "training-test distribution mismatch
> from a v1-era labeled pool" is open. **What to try next when budget
> allows:** pooled-labels eval (~$115) to re-label the 145q training pool
> against the v2 production pipeline, then retrain. See CLAUDE.md
> Decision 40 + the Phase-12 entry in "Current State" for context.

**Files:** `models/ranker/v3-regularized/model.lgb` (production),
`models/ranker/v2/model.lgb` (preserved, superseded),
`models/ranker/v3/model.lgb` (failed vanilla attempt, preserved as evidence),
`data/evaluation/ranking_features_v3.csv` (v2-stack features),
`data/evaluation/ranking_features_v2.csv` (v1-stack features, preserved),
`scripts/train_ranker.py` (now supports `--regularize --output-suffix`),
`scripts/evaluate.py`, `scripts/r_e_eval_lgb_v2_vs_v3.py` (the 3-way A/B harness)

### Why it's there

The LightGBM LambdaRank blender (v2) was trained on **145 queries /
6,018 labeled (query, trial) pairs**, pooled from three label files:
`labeled_queries.jsonl` (20 queries), `test_labels.jsonl` (50), and
`train_labels_extra.jsonl` (75). The dataset was scaled up
deliberately from v1's 20-query/990-pair pool when the v1 model
under-trusted the CE score (CE feature importance: 1012 in v1 →
1673 in v2 once more data was added).

The size cap is driven by labeling cost: every additional query
requires labeling top-50 candidates with Claude Haiku at ~$0.02 per
labeled pair (≈ $1 per query). The current 6K pairs cost ~$120 of
Haiku spend cumulatively.

### Why it's wrong / what it costs us

145 queries is comparable to **academic LTR benchmarks** (LETOR 4.0 =
84 queries, MQ2008 = 784) and well below **industry production
scale** (MS-MARCO LTR = 370K queries; production search teams retrain
continuously on millions of click signals). It's a defensible scale
for an interview project. It's a borderline scale for confident
modeling decisions.

The data-adequacy signals from the current model are mixed:

| Signal | Evidence | What it suggests |
|---|---|---|
| Pipeline monotonicity on held-out 50-query fair eval | NDCG@5: BM25 0.617 → Semantic 0.606 → Hybrid 0.636 → +CE 0.651 → +LGB 0.670 | LightGBM adds genuine value on unseen queries ✅ |
| Sample-to-feature ratio | 6K pairs / 11 features ≈ 545 per feature | Comfortably above tree-model stability rules ✅ |
| **MRR drop from CE → LGB** | **MRR: 0.825 (after CE) → 0.806 (after LGB)** | Possible list-level overfitting — blender helps list ranking at the cost of first-result quality. Could be real, could be noise. 145 queries is too few to tell. ⚠️ |
| **Per-category bootstrap CI half-widths (Week 9 eval)** | ±0.03 to ±0.32 across 7 categories | Some categories (pediatric, vague) have so few queries that single-query swings dominate the metric ⚠️ |
| Feature importance ranking | CE (1673) > RRF (1012) > phase (593) > title overlap (557) > semantic (520) > BM25 (516) > enrollment (487) > … | Coarse ranking is stable; fine-grained importance numbers between adjacent features are within noise ⚠️ |

What 145 queries IS enough for:

- Demonstrating that LightGBM blending adds value end-to-end.
- Establishing coarse feature-importance rankings (CE > metadata >
  retrieval scores).
- Beating the simpler hybrid baseline robustly on a held-out set.

What 145 queries is NOT enough for:

- **Confident hyperparameter tuning.** Sweeping
  `num_leaves × min_data_in_leaf × learning_rate × feature_fraction`
  via Optuna at this scale risks finding HP combinations that
  overfit on 145 queries and don't generalize. The 50-query
  held-out fair eval is the only check, and a lucky Optuna trial
  could look like a winner without being one.
- **Trusting fine-grained feature decisions.** Is `enrollment_log`
  better than `enrollment_raw`? Is `title_overlap_jaccard` better
  than `title_overlap_count`? Differences of 0.005–0.01 NDCG@5 are
  below the noise floor at 145 queries.
- **Per-category model claims.** Per-category CIs from the Week 9
  eval range up to ±0.32 (categories with n=3 queries). Any claim
  like "the model does better on pediatric than on rare cancers" is
  not statistically distinguishable from noise without more
  per-category queries.
- **Detecting small feature additions.** Wiring the
  `EligibilityParser` outputs (`min_age_fits`, `sex_matches`,
  `prior_treatments_match`) into the feature vector likely adds
  +0.01–0.03 NDCG@5 on the queries that fail today — but the lift
  would barely be detectable at this dataset size without
  multi-seed averaging.

### The deeper architectural inversion

Data scale across the pipeline is **inverted relative to what tree
models need**:

| Component | Training data | Inductive bias from pretraining | Data adequacy |
|---|---|---|---|
| Bi-encoder | 730K triplets | Strong (BioLinkBERT) | Plenty |
| Cross-encoder | 200K binary pairs | Strong (BioLinkBERT) | Adequate for binary, inadequate for graded (Issue #5) |
| **LightGBM blender** | **6K graded pairs / 145 queries** | **Weak (no pretraining; learns from scratch)** | **Borderline** |

Tree models have no pretrained prior — they have to learn every
relationship from the data alone. So they need *more* samples per
parameter than the bi-encoder, not fewer. The pipeline gives them
fewer because labeled-pair availability is the bottleneck, not
metadata-pair availability.

### The fix — expand the labeled query set

The lift mostly comes from more queries, not more candidates per
query (LambdaRank already saturates the per-query pairwise
comparison budget at ~41 candidates × 820 pairs/query). Target:
**300–500 training queries** (roughly 2–3× current). Two paths:

**1. Pair with Issue #2's synthetic query expansion.** When the
synthetic query pool grows from 1,500 → 10,000 for bi-encoder
retraining, the same pool seeds LightGBM training. Procedure:

- Generate 5K synthetic patient queries (cost: ~$8.50 in Haiku).
- For each query, run the current pipeline and take top-50
  candidates.
- Label each (query, candidate) pair with **Sonnet 4.6** (not Haiku,
  per Issue #6 — to break the judge-bias loop).
- Sample 200–400 queries randomly from the labeled pool for
  LightGBM training. Hold out the rest as test.
- Total marginal cost: ~$8.50 synthetic generation +
  ~$1,200 Sonnet labeling for 5K × 50 pairs. **Dominated by labeling.**

**2. Cheaper alternative — Haiku labels with Sonnet spot-check.**
Label with Haiku (~$425 for 5K × 50 pairs) and reserve Sonnet for
the held-out test set only. Saves ~$800 but inherits judge-bias
risk (Issue #6). Defensible if Issue #6's Sonnet-test-set fix is in
place, since the test labeler is then independent.

### What this fix does NOT do (and should be separated from)

- It does not address the binary-CE supervision problem (Issue #5).
  Those are independent fixes — Issue #5 makes LightGBM's #1
  feature cleaner *at the same dataset size*; Issue #7 makes
  LightGBM itself more confident *with cleaner-or-not features*.
  Doing #5 first is probably higher leverage because it costs almost
  nothing in labeling spend.
- It does not address feature engineering gaps. Adding the
  `EligibilityParser` features (which the existing parser at
  `src/TrialMine/features/eligibility.py` already produces) is
  orthogonal — that's a value-add at *any* dataset size and should
  be done independently.
- It does not fix per-category claims unless the new queries are
  *stratified* across the weak categories (`complex`, `vague`,
  rare-cancer per Issue #3). Adding 200 more "common cancer"
  queries doesn't shrink the `pediatric` CI.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Synthetic query generation | ~$8.50 (Haiku, 5K queries; resumable script exists) |
| Pair labeling (Sonnet for clean signal) | ~$1,200 (5K queries × 50 pairs × ~$0.005/Sonnet-call). The single largest cost. |
| Pair labeling (Haiku, cheap option) | ~$425. Requires Issue #6 fixed first for defensibility. |
| LightGBM retraining | Minutes. Tree models train fast. No GPU needed. |
| Re-eval | Existing 60-query held-out test set + the new Sonnet-labeled test queries. |
| Risk to existing wins | Low. New training data is additive; worst case the v3 model is no better than v2 and we keep v2. |
| Risk of new judge bias | Real if labeled with Haiku without Issue #6 fixed. Negligible if Sonnet-labeled. |

### Verdict

**Do this after Issues #5 and #6, not before.** Two reasons:

1. **Issue #5 is higher leverage for the same labeling spend.** A
   graded-CE-retrained model takes the existing 6K labels (no new
   labeling cost) and gives the LightGBM blender a much better #1
   feature. Try that first; the existing 145-query LightGBM might be
   noticeably better just from cleaner inputs.
2. **Issue #6 has to land before any new training data is collected,
   not after.** If new queries are Haiku-labeled and then Haiku is
   also the test judge, we've poured $400+ into reinforcing the
   bias loop. Issue #6's Sonnet-test-set fix needs to be in place
   before $1,200 of labeling spend gets committed.

Sequencing as a single coordinated cycle:

1. Issue #6 fix: re-label the 60-query test set with Sonnet (~$3).
2. Issue #5: retrain CE on graded pairs (~$0 in labeling — uses
   existing labels). Re-eval against Sonnet-labeled test set.
3. Issue #7: if LightGBM still has the MRR-drop signal or per-category
   CIs are still too wide for the claims you want to make, *then*
   generate 5K Sonnet-labeled queries and retrain LightGBM at 300+
   queries.

If Issues #5 and #6 collapse the MRR drop and tighten the CIs
sufficiently, Issue #7 may be deferrable — at which point the
labeling spend stays on the bench until a real product need
materializes.

### Principle worth keeping

Sample-size adequacy is not absolute — it's relative to the claims
you want to make from the model. 145 queries is plenty for
"LightGBM adds value end-to-end" and "CE is the most important
feature." It is not plenty for "feature A is better than feature B
by 0.005 NDCG" or "the model is well-tuned." When deciding whether
to expand the labeled set, the question to ask is not *"do we have
enough data?"* but *"what specific claim am I trying to make that
the current data can't support?"* — and only then collect more.

---

## 8. Eligibility parser emits noisy condition/treatment spans — downstream matcher must be tolerant

**Files:** `src/TrialMine/features/eligibility.py:561-605`,
`src/TrialMine/features/eligibility.py:600` (`_TREATMENT_RE` router),
`src/TrialMine/features/eligibility.py:1` (stop-list `_STOP_SPANS`)

### Why it's there

`EligibilityParser._extract_entities` runs SciSpacy `en_core_sci_lg`
on each inclusion/exclusion section, filters spans through a ~80-term
stop-list, then routes drug/therapy-shaped spans into
`required_/excluded_prior_treatments` via `_TREATMENT_RE`. Everything
else lands in `required_/excluded_conditions`.

Decision 17 in CLAUDE.md accepts this as a deliberate trade-off: a
SciSpacy + regex hybrid hits ~70% precision in days, while a custom
biomedical NER would need 80–120 hr of annotation work for marginal
gain on a non-load-bearing field.

### Why it's wrong / what it costs us

The ~70% precision number is correct but abstract. Concretely, the
parser failure modes on a real cancer trial (NCT05535569 — Phase Ib/II
nivolumab + paclitaxel for advanced gastric cancer; 37 required
conditions, 53 excluded conditions, 10 excluded prior treatments)
decompose into three categories, ordered by frequency:

**(a) Noise / over-extraction — dominant**

| In `excluded_conditions` | Why it's noise |
|---|---|
| `"severe"`, `"chronic"`, `"concurrent"` | Modifiers — describe the *next* word, not a condition |
| `"past history"`, `"diagnosed"`, `"imaging"`, `"CT"` | Methodology / temporal framing, not a disease |
| `"primary"`, `"positive"`, `"tumor"` | Generic clinical vocabulary, no constraint value |
| `"digned written informed"` | Source-text typo (CT.gov-side) + fragment of "signed informed consent" |

These don't hurt recall — they just add length. The downstream
matcher has to treat unknown spans as **Unknown**, not **Unmet**,
otherwise every trial fails on noise.

**(b) Lost context — subtle, harder to fix**

Spans arrive **flattened** — the relationship between adjacent spans
is gone:

- `"past history"` and `"interstitial lung disease"` are two separate
  spans. The matcher can't see the temporal qualifier — is this a
  patient who *currently* has ILD, or *ever had* ILD?
- `"surgery"` and `"radiotherapy"` land in `excluded_prior_treatments`
  but in the source they were inside a "no surgery within 28 days of
  treatment" sentence. The 28-day window is gone.
- `"Eastern Cooperative Oncology Group ("` — span boundary cut
  mid-parenthesis, losing the actual constraint (`ECOG 0-1`).

This is a **structural** limitation of NER-into-buckets — can't be
fixed by tightening the stop-list. Would need a full semantic parser
(SRL or a dependency-tree walker), which is the "custom NER"
path Decision 17 explicitly avoids.

**(c) Genuine missing — rare**

SciSpacy `en_core_sci_lg` is permissive — it errs toward
over-extraction. Real disease/biomarker spans like `"HER2-positive"`,
`"interstitial lung disease"`, `"anti-PD-L1 antibody"` get caught
reliably. The schema-level miss is real numeric thresholds
(`"creatinine clearance > 60 mL/min"`, `"life expectancy ≥ 3 months"`):
the *name* surfaces as a span, the *threshold* is dropped. This is a
schema gap (`EligibilityProfile` has no numeric-threshold field), not
a NER gap.

### The cost depends entirely on whether the matcher is wired in

The single highest-leverage open task in CLAUDE.md "What's next" is:

> **Wire `EligibilityProfile` into the BM25 filter (or as a post-RRF
> gate)** — three of the five worst Week 9 NDCG queries fail because
> the agent parses age / prior treatments / progression status into
> `PatientProfile`, but the orchestrator does not translate those into
> hard retrieval filters. Q416 (post-trastuzumab progression) ranks
> the STOP-HER2 trial at #1 — opposite clinical situation.

When that lands, the parser noise stops being a cosmetic issue and
becomes a **correctness** issue: the matcher must decide for each
parser span whether to treat it as a hard constraint or ignore it.
Today, the noise sits in `parsed_eligibility` rows nobody reads.

### The fix — incremental, scoped to the matcher contract

Don't try to make the parser cleaner. Make the matcher tolerant of
what the parser actually produces.

**1. Three-valued matcher output (Decision 19, restated).**
The eligibility matcher emits `Met | Unmet | Unknown` per criterion,
never binary. Trial-level rollup: any hard `Unmet` → trial `Unmet`;
all `Met` → trial `Met`; else `Unknown`. Critical: a noise span like
`"severe"` should produce `Unknown` (don't fail the trial), not
`Unmet` (false rejection cascade).

**2. Expand the stop-list with measured-noise terms.**
Run the parser over a 1,000-trial sample, count span frequencies, and
add the top-50 most-frequent non-clinical spans to `_STOP_SPANS`
(e.g. `"severe"`, `"chronic"`, `"past history"`, `"diagnosed"`,
`"imaging"`). Cheap; partial mitigation of category (a). Doesn't help
(b) or (c).

**3. Treat treatment-bucket spans as evidence, not constraints.**
The current router (`_TREATMENT_RE`) puts anything matching a
drug-name pattern in `excluded_prior_treatments`. But "surgery" and
"radiotherapy" are also routed there with no temporal context. The
matcher should treat the *treatment* lists as a *signal that this
trial has prior-therapy washout language*, not as a list of
hard exclusions. The hard exclusion still needs LLM verification per
candidate (the existing `check_trial_eligibility` tool does this).

**4. Optionally — add an `EligibilityProfile.numeric_thresholds`
field** for category (c). One regex pass over the raw text picking up
`<number> <unit>` patterns near known constraint phrases (`ECOG`,
`creatinine`, `platelet`, `hemoglobin`). ~50 LOC + a Pydantic field
addition. Separate from the NER pipeline, so the noise doesn't bleed
in.

### Cost / risk of fixing

| Dimension | Estimate |
|---|---|
| Code changes for (1) | Matcher module doesn't exist yet — needs to land alongside eligibility-filter wiring. CLAUDE.md "What's next" already flags `eligibility_matcher.py` extraction from `tools.py:check_trial_eligibility`. |
| Code changes for (2) | ~30 LOC + a 1,000-trial profiling script. Stop-list literal stays in `eligibility.py`. |
| Code changes for (3) | A contract change in the matcher, not the parser. ~50 LOC in the future matcher. |
| Code changes for (4) | ~50 LOC + Pydantic field. Independent of (1)–(3). |
| Re-parsing the corpus | `scripts/parse_eligibility.py` is `--resume`-able; ~100 min single-process over 140K trials. Required only for (2) or (4) (parser output changes). Not required for (1) or (3) (matcher-side changes only). |
| Re-indexing FAISS / BM25 | **Not needed.** The parser feeds `parsed_eligibility` SQLite, not the retrieval indices. Issues #1, #2, #3, #4 require re-indexing; this one doesn't. |
| Risk to existing wins | None. The parser output isn't consumed by anything in the hot path today. Even Q416-class failures are coming from the *agent profile* not being translated into filters, not from parser noise. |

### Verdict

**Don't fix the parser. Make the matcher tolerant.** The CLAUDE.md
"What's next" eligibility-filter wiring is the larger piece of work
this lives inside — the matcher contract is where Issue #8's
mitigations land naturally. Fold steps (1) and (3) into that work.
Step (2) is a 30-LOC freebie that can ship independently. Step (4)
is opportunistic — do it the next time someone touches
`eligibility.py` for any reason.

The interview-facing story is already correct per Decision 17 (we
chose this trade-off deliberately, ~70% precision was the budget). The
issue worth surfacing — and the reason this entry exists — is that
the **matcher must absorb the precision gap**, not the parser.
A binary matcher fed today's parser output would produce
confidently-wrong eligibility verdicts; a three-valued matcher
degrades gracefully to `Unknown`.

### Principle worth keeping

When a component has a known precision ceiling (NER on biomedical
text, OCR, automatic speech recognition), the right fix is rarely
"make the component cleaner." It's usually "design the downstream
contract to tolerate the noise." Binary outputs amplify upstream
noise into wrong answers; three-valued outputs (Met / Unmet /
Unknown, present / absent / uncertain) absorb it into degraded —
but honest — confidence.

---

## 9. Agent produces Unknown eligibility verdicts but has no mechanism to resolve them

**Files:** `src/TrialMine/agents/pipeline.py` (LangGraph state machine),
`src/TrialMine/agents/orchestrator.py:222-252` (eligibility check loop),
`src/TrialMine/agents/tools.py:649` (`check_trial_eligibility`),
`src/TrialMine/agents/query_parser.py` (`PatientProfile` extraction)

### Why it's there

Decision 19 specified a three-valued eligibility matcher (Met / Unmet /
Unknown) on purpose. Unknown is emitted when patient info is missing —
the matcher refuses to guess, which is correct. Decision 22 chose
template explanations over LLM-per-result to keep latency and cost
down: the agent makes exactly one LLM call (Haiku slot extraction)
and then runs a deterministic pipeline.

Combined, those two decisions produce a system that *honestly reports
uncertainty* and *never tries to reduce it*. On sparse queries
(`"lung cancer trials"`), the matcher correctly returns Unknown across
the board and the agent shrugs. That's by design — the design just
stops one step short of being useful.

### Why it's wrong / what it costs us

The agent's value is in **resolving uncertainty**, not just producing
it. Today the pipeline produces Unknown verdicts and ends. Two
specific symptoms:

1. **Sparse queries get no agent lift.** On
   `"lung cancer immunotherapy trials"`, the agent extracts only
   `condition`; the eligibility check returns 4 × Unknown per trial;
   the user gets the ranker output + boilerplate explanations.
   Strictly weaker than the hybrid retrieval baseline plus a UI for
   raw eligibility text.
2. **Rich queries with off-tier failures.** The Q416-class failures
   from Week 9 (post-trastuzumab progression mis-ranking the STOP-HER2
   trial at #1) happen because the user *did* mention prior
   trastuzumab, the parser caught it, the matcher noticed
   `excluded_prior_treatments` contains a trastuzumab-related span on
   STOP-HER2 — and then nothing happened with that information
   because eligibility check runs after retrieval (Issue #8). Even
   with #8 wired in, the trial-side eligibility text has many spans
   the parser can't unambiguously type ("post-progression" vs.
   "treatment-naive"), and resolving them needs LLM judgment, not
   stricter regex.

The agent framework (LangGraph, `agent_trace` reducer, conditional
routing) exists *for* this surface and is currently underutilized.
Decision 20 documented this as deliberate over-investment;
Issue #9 is the work that retroactively justifies it.

### The fix — two-stage clarify, with grounding constraint

**Stage 1 — Constrained, coverage-driven clarify.** New `clarify`
node in `pipeline.py` placed between `parse_query` and
`execute_search`. Logic:

1. Run a cheap BM25-only pre-pass (~10 ms) to get top-10 candidates.
2. Load `parsed_eligibility` for the 10; tally Unknown counts per
   criterion against the current `PatientProfile`.
3. If `unknown_rate ≥ threshold` (tune; start at "6 of 10 trials
   have ≥2 Unknowns"), route to clarify; else pass through.
4. Pick the field whose absence causes the most Unknowns. Render a
   single soft, skippable question. Show partial results below.
5. On user answer, augment `PatientProfile` and re-enter
   `execute_search`.

Questions in this stage come from a fixed checklist (age, sex,
condition stage, ECOG, primary biomarker). Selection is data-driven
from trial-side coverage; phrasing is static. No LLM call beyond what
already exists for slot extraction.

**Stage 2 — Open-ended LLM-generated clarify.** Replace the fixed
checklist with an LLM call that reads the top-10 trials' eligibility
text and generates the single highest-information question. **Hard
grounding constraint: every question must be derivable from a span
in the retrieved trials' eligibility text.** No questions about
fields not actually gating real trials in front of the user.

Treatment-history is the highest-leverage Stage-2 target:
*"You mentioned HER2-positive breast cancer — have you been on
trastuzumab? If yes, did it stop working?"* — exactly the question
that would have caught Q416. This is the version that needs the agent.

### Cost / risk of fixing

| Dimension | Stage 1 | Stage 2 |
|---|---|---|
| LangGraph changes | ~150 LOC — new `clarify` node, coverage-analysis helper, conditional routing | +~100 LOC — LLM-generated question prompt + grounding check |
| UI changes | Banner / chip surface above results; session-id for second pass | Same surface; richer answer parsing (free text, not multiple-choice) |
| LLM cost per sparse query | Zero (uses existing slot extraction) | ~$0.002 + ~1.5 s Haiku call to generate question |
| Eval scaffolding | Run NDCG twice — without clarify, and with `PatientProfile` labels as simulated answers. Delta is the lift. **No new labeling needed**, reuses existing labeled set. | Same eval shape; add a hallucination check (every generated question's relevant span must appear in retrieved trials' eligibility text) |
| Risk: irrelevant question | Mitigated by coverage-driven selection — only ask when ≥60% of top-10 actually gate on the field | Mitigated by span-grounding — refuse to ask anything not in retrieved trials |
| Risk: patient anxiety | Soft phrasing + always-skippable + partial results visible | Same, plus tone-check pass in the question-generation prompt |
| Risk: user can't answer | Stage 1's fixed checklist filters at design time (age/sex are lay-answerable; ECOG isn't) | Stage 2 needs explicit lay-vocabulary constraint in the prompt |
| Multi-turn state | Single round-trip (question → answer → re-search); session-id sufficient | Possibly multi-turn (answer → refined Unknowns → second question); needs proper conversation thread abstraction |
| Eval methodology break | Mild — same NDCG, just over re-augmented profile | Larger — need to evaluate *trajectory*, not single retrieval pass |

### Sequence

This issue compounds with Issue #8 (eligibility-filter wiring) and a
hypothetical "LLM-graded eligibility on top-3 Unknowns" surface.
Order them as:

1. **Issue #8 first** — wire `EligibilityProfile` into the BM25
   filter dict so the matcher's output actually affects retrieval.
   Without #8, clarify gathers information that doesn't change
   anything.
2. **Issue #9 Stage 1** — coverage-driven fixed-checklist clarify.
   Validates the UX, the session-id plumbing, and the eval-delta
   methodology with low risk and no new LLM calls.
3. **LLM-graded eligibility on close calls** — for the top-3 trials
   that still come back Unknown after clarify, send the eligibility
   text + augmented profile to an LLM. ~$0.005 × 3 = $0.015/query;
   resolves residual parser-noise Unknowns (the "post-history" /
   "concurrent" noise from Issue #8 category (a)).
4. **Issue #9 Stage 2** — open-ended LLM-generated clarify, grounded
   in retrieved-trial eligibility spans. Treatment-history first;
   demographic clarify already in Stage 1.

### Verdict

**Plan to ship. Stage 1 within the same cycle as Issue #8; Stage 2 as
the follow-on that earns the LangGraph investment.** Without this
work, the agent framework (LangGraph, `agent_trace`, conditional
routing) is overhead — Decision 22's templated explanations + Decision
21's one-shot slot extractor could be a 100-line async function. With
this work, the framework's structure (state machine, conditional
routing, reducer-based observability) is doing real work — the
clarify branch is exactly the conditional path the state machine was
shaped for.

### Principle worth keeping

A matcher that emits three values (Met / Unmet / Unknown) and a
system that only knows how to act on two of them is incomplete.
Unknown is not a degraded Met — it's a request for more information.
The agent's job is to translate that request into either (a) an LLM
read of the source text, or (b) a question to the user. A system
that emits Unknown and stops there is honest but not useful;
resolving Unknowns is the work the agent framework exists to do.

---

## 7. Drug-class UMLS bridge is over-broad for drug-specific exclusion semantics

**Files (Phase A code reverted in commit `a60950a`; recoverable via
`git show 82f467c:<path>` for each):**
- `src/TrialMine/features/concepts.py` — rolled back to pre-A3 (linker
  removed; `ConceptNormalizer` + lay-medical synonym map remain in `main`)
- `src/TrialMine/features/drug_classes.py` — deleted; recoverable via git
- `src/TrialMine/agents/tools.py` — rolled back (`_any_overlap_cui` removed)
- `src/TrialMine/config.py` — rolled back (`umls_drug_class_matching_enabled`
  removed)
- `docs/fix_parser_umls.md` — deleted; recoverable via git
- `scripts/eval_parser_umls.py` — removed after revert (imports broken once
  Phase A code was pulled); recoverable via git
- `data/evaluation/umls_eval_metrics.json` + per-query progress + 96-row
  Haiku-labels JSONL — **still in `main`** as frozen eval evidence backing
  the numbers below

### Why it's there

Phase 13A (2026-05-14) wired SciSpacy `EntityLinker` (UMLS) + a
hand-curated 28-entry `DRUG_TO_CLASS_CUIS` table into the eligibility
filter behind a default-OFF `DegradationConfig` toggle. The hypothesis,
per `docs/evaluation-report.md` §11.6's pre-registered expectation, was
that patient prior-treatment mentions ("osimertinib") and trial
eligibility exclusions ("no prior EGFR TKI") would bridge via the shared
class CUI, lifting `complex` NDCG@5 by +0.02 to +0.04. The runbook
`docs/fix_parser_umls.md` drives Phase A (local prep, scispacy + 28-entry
drug-class table + toggle + tests) → Phase B (focused re-eval) → Phase C
(decision gate) → Phase D (ship) / Phase E (hold/revert). Phase A
shipped clean as commit `82f467c` (60 tests pass; ~2.1 GB UMLS KB cached
at `~/.scispacy/datasets/`).

### What didn't work — Phase B numbers (`umls_eval_metrics.json`)

| Slice | OFF | ON | Δ | Expected |
|---|---|---|---|---|
| complex (n=15) | 0.636 [0.542, 0.728] | 0.624 [0.534, 0.714] | **−0.012** | +0.02 to +0.04 |
| vague (n=15) | 0.846 [0.771, 0.916] | 0.846 [0.771, 0.916] | **+0.000** | ~0.000 |

Only 3 of 30 queries triggered any verdict flip (Q413, Q417, Q608). The
Q413 spot-check surfaced the structural failure: trial `NCT03755102`
excludes prior `dacomitinib therapy`, but UMLS class-bridge matching
linked dacomitinib and osimertinib via their shared EGFR-TKI class CUI
(`C5574906` / `C1268567`). The trial is *actually* a **study of
dacomitinib+osimertinib FOR osimertinib-failure patients** — exactly the
target population. UMLS-on wrongly dropped it. 1 of 2 spot-checked true
triggers being a clinical false positive is a structural red flag, not a
coverage gap.

Q416 (post-trastuzumab) triggered 0 verdicts because the top-10 retrieved
trials are HER2-required (not HER2-excluded). Q417 (pembrolizumab ↔ ICI)
worked correctly as the canonical case. Q608 triggered 2 newly-Unmet on
`required_prior_treatments` (which doesn't gate hard-filter drops per
Decision 38) in the strictness-increase direction — information loss
relative to substring, not gain. 5 of 15 complex queries don't mention
any specific drug at all (Q601 BRCA neoadjuvant, Q606 AML post-allo,
Q607 PARP failure as class only, Q608 radiation / Stupp regimen, Q609
CAR-T modality) — UMLS bridges have nothing to act on for those.

### Why the structural issue matters

`DRUG_TO_CLASS_CUIS` is a one-way bridge: drug CUI → class CUI(s). The
matcher then checks for ANY class overlap between two text spans. This
is **correct** for trial exclusions that name a class ("no prior EGFR
TKI"): the patient's osimertinib should resolve to the EGFR-TKI class
and trigger Unmet. But the matcher is **wrong** for trial exclusions
that name a specific drug ("no prior dacomitinib"): the trial author's
intent was to exclude only dacomitinib-exposed patients, not all
EGFR-TKI-exposed patients, and the class-bridge collapses the
distinction. The parsed-eligibility text can't disambiguate — both look
like a string in `excluded_prior_treatments`. Adding more drugs to the
table would lift coverage *and* add false positives in equal measure;
net expected lift is unclear, possibly negative.

### The fix (deferred — not a single-PR change)

A real fix needs **per-criterion semantics** — knowing whether a trial
author meant the specific drug or the class. Three paths considered:

1. **UMLS REST API + hierarchy lookup** — query `umls_api_key`-authed
   endpoints (the field already exists on `Settings`, currently empty)
   to walk parent / child relations and infer specificity. Approx 1–2
   days; risk = rate limits, external dependency, hierarchy drift
   between scispacy KB and live UMLS.

2. **RxClass / NCIt fallback** — switch the linker from `umls` to
   `rxnorm` (or add a parallel path), then walk RxClass hierarchies for
   drug-class membership. Approx 3–5 days incl. re-index; risk = changing
   the linker substrate invalidates the existing test fixtures + curated
   drug-class table.

3. **LLM-at-matching-step** — a small per-(query, trial) Claude Haiku
   call at filter time to judge "does this exclusion apply to the
   patient?". Slow (~1.5 s/trial) and expensive (~$0.02 / query) but no
   parser limitation would constrain it. Out of scope for an
   eligibility-filter pass that's meant to be free.

### Cost / risk of fixing

- (1) UMLS REST: ~1–2 days; external API dependency at request time.
- (2) RxClass: ~3–5 days incl. re-parse / re-eval of 140K trials.
- (3) LLM-at-matching: ~1 day; recurring cost + latency budget pressure.

### Verdict

**Accept and document.** Phase A code, tests, and runbook were reverted
out of `main` in commit `a60950a` (full `git revert 82f467c`) — the
toggle was always default-OFF so production behavior was never affected,
and pulling ~600 lines of inert infrastructure out of `main` keeps the
working tree clean. Phase 13A is preserved in git history at `82f467c`
(ship) and `a60950a` (revert); the runbook is recoverable via
`git show 82f467c:docs/fix_parser_umls.md` for a future re-engagement.
Re-engage when one of the above paths becomes the next-most-promising
bottleneck OR when eligibility-text parsing gets richer at parse-time
(e.g., classifying exclusions as drug-specific vs class-level upstream so
the matcher inherits the distinction). Re-engagement means redoing Phase
A from the recovered runbook, not editing inert code already on `main`.

### Status — accepted (2026-05-14)

Phase 13A shipped the wiring in commit `82f467c` and was reverted in
commit `a60950a` on the Phase C decision gate. The runbook
`docs/fix_parser_umls.md` is no longer in `main`; recover it via
`git show 82f467c:docs/fix_parser_umls.md` if needed. See
`docs/evaluation-report.md` §11.6 for the eval-report writeup and
CLAUDE.md Decision 41 for the project-level record.

### Principle worth keeping

A class-level bridge is "right" or "wrong" depending on whether the
*trial author* meant the class or a specific drug. The parsed-eligibility
text doesn't carry that distinction. Future class-bridge designs must
encode specificity at parse-time or at matching-time — class collapse at
the matcher level alone is structurally too broad.

---

## Template for future entries

```markdown
## N. Short title

**File:** `path/to/file.py:line`

### Why it's there
(Original motivation — assume good faith.)

### Why it's wrong / what it costs us
(Concrete failure mode + measured impact if available.)

### The fix
(Code-level proposal.)

### Cost / risk of fixing
(Effort, reindex/retrain cost, blast radius.)

### Verdict
(Fix now / fix when convenient / accept and document.)
```
