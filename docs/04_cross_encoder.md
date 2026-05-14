# Week 4 — Cross-Encoder + LightGBM Blender + The Ablation Table (Student Edition)

A textbook walkthrough of week 4 — the most important week of the
project. By the end you should be able to (a) explain the difference
between a bi-encoder and a cross-encoder, (b) read a learning-to-rank
paper without panicking, (c) walk through our 5-stage retrieval
pipeline in your own words, and (d) defend the ablation table in an
interview.

This is the most concept-dense week of the project. We'll go slowly.

---

## Part 1 — Why this week matters

After week 3 we have a fine-tuned bi-encoder. Hybrid retrieval
(BM25 + semantic, RRF-fused) hits NDCG@10 = 0.796 on the 20 labeled
queries. **That's already pretty good.** Why aren't we done?

Two reasons.

**Reason 1 — The bi-encoder makes a fundamental compromise.** Recall
that a bi-encoder embeds the query and the trial *separately*, then
compares them with cosine similarity. But the most useful signal — *how
the query words relate to specific trial words* — is gone before the
comparison even happens. The query embedding is a single 768-dim
vector that has to capture everything about the question; the trial
embedding has to capture everything about the trial. Then we compute
one number from those two vectors. Most of the cross-information is
lost.

A **cross-encoder** does it differently. It takes the query AND the
trial as one combined input (`[CLS] query [SEP] trial text [SEP]`)
and runs a transformer over the full sequence. Now every query token
can attend to every trial token. The model can directly reason about
"this query word *immunotherapy* relates to that trial word
*pembrolizumab*." Much more accurate. Much slower.

**Reason 2 — Pure relevance isn't the only thing we should rank on.**
A patient looking for a clinical trial cares about more than "is the
trial about my disease?":
- Is it *recruiting*? A perfect-match trial that's COMPLETED is useless.
- What *phase* is it? Phase 3 trials are larger and more established.
- Is the *enrollment* large? Larger trials are more accessible.

These are **metadata features**. They're not relevance, but they're
real ranking signals. Combining metadata with retrieval scores requires
a *learned* model — a hand-tuned formula will leave value on the table.

So the week 4 plan:

1. **Train a cross-encoder** to re-rank the top 50 hybrid candidates.
2. **Train a LightGBM ranker** that combines all signals (BM25 score,
   semantic score, cross-encoder score, RRF score, plus 7 metadata
   features) into a final ranking.
3. **Wire it all together** as a 5-stage pipeline: BM25 → semantic →
   RRF merge → CE → LightGBM.
4. **Build THE ABLATION TABLE** — for every stage, measure NDCG@5,
   NDCG@10, MRR, and median latency, with bootstrap confidence
   intervals. *This table is the project's headline artifact.*
5. **Run a fair held-out evaluation** on 50 new queries the
   LightGBM never saw, to check we didn't overfit.

---

## Part 2 — Vocabulary you'll meet

### ML / engineering terms

- **Bi-encoder vs cross-encoder.** Bi-encoder: encode separately,
  compare with cosine (Part 3). Cross-encoder: encode jointly, output
  one score. Bi is fast, cross is accurate.
- **Re-ranking.** A two-stage pattern: a fast retriever surfaces ~200
  candidates; a slow, accurate re-ranker scores those candidates and
  reorders them. We use it because the cross-encoder is too slow to
  run on 140K trials directly.
- **BinaryCrossEntropyLoss (BCE).** The classic loss for binary
  classification. Predicts a probability that a (query, trial) pair
  is relevant; pushes that probability toward 1 for positives and 0
  for negatives.
- **CERerankingEvaluator.** The cross-encoder analog of week 3's
  `InformationRetrievalEvaluator`. During training, gives the model a
  positive + a negative per query and measures whether positive ranks
  above negative.
- **Learning-to-rank (LTR).** Train a model to *combine many features*
  into a single ranking score. Three styles:
  - **Pointwise.** Predict relevance per (query, doc), then sort.
  - **Pairwise.** Predict which of two docs is more relevant.
  - **Listwise.** Optimize the ranking *as a list* — NDCG-aware. ←
    what we use.
- **LambdaRank.** The listwise objective LightGBM implements.
  Approximates NDCG by weighting pairwise gradients by the change in
  NDCG that swapping the pair would produce.
- **LightGBM.** Microsoft's gradient-boosted decision tree library.
  Fast, interpretable, robust. Trains our LambdaRank model in seconds
  on our 6,000-row dataset.
- **Feature importance.** A LightGBM-internal score: "how much did
  each feature contribute to the model's decisions?" Used to explain
  and debug the ranker.
- **Leave-one-query-out (LOO) cross-validation.** With N queries,
  train N times — each time holding out one query — and average the
  held-out scores. Essential when you have few queries.
- **Ablation study.** Remove one component at a time, measure its
  contribution. The "ablation table" is the side-by-side comparison.
- **Bootstrap confidence interval.** Resample your data with
  replacement many times; compute the metric on each resample; take
  the 2.5th / 97.5th percentiles. Tells you how much your point
  estimate would wobble if you re-ran with fresh draws.
- **Fair held-out evaluation.** Evaluate on data the model truly
  never saw. For learning-to-rank, the *queries* are training data —
  so a fair eval needs fresh queries.

### Medical terms (only what's new)

Most medical vocabulary was glossed in weeks 1-3. New here:

- **Pembrolizumab** — a PD-1 immunotherapy drug. Often appears as a
  positive in our training pairs.
- **Trastuzumab** — a HER2 targeted-therapy drug. Often appears as a
  hard negative (same disease, different drug).
- **Neoadjuvant** — given before the main treatment (e.g., chemo
  before surgery, to shrink the tumor first).
- **Adjuvant** — given after the main treatment, to mop up any
  remaining disease.

---

## Part 3 — Bi-encoder vs cross-encoder, explained slowly

This is the most important interview concept of the entire project.

### The bi-encoder

```
                    query              trial
                      │                  │
                      ▼                  ▼
                ┌──────────┐       ┌──────────┐
                │  BERT    │       │  BERT    │
                └──────────┘       └──────────┘
                      │                  │
                      ▼                  ▼
                  768 dims           768 dims
                      \                  /
                       \                /
                        \              /
                         ▼            ▼
                    cosine_similarity(q, t) → score
```

Two passes through the model — one for the query, one for the trial.
Each side produces a single 768-d vector. The score is the cosine
similarity.

**Crucial property.** Trial embeddings can be **pre-computed once**
and stored in FAISS. At query time, only the *query* needs to be
embedded — that's a single forward pass. So bi-encoder retrieval over
140K trials is fast: one query forward pass + one FAISS lookup ≈ 30 ms.

### The cross-encoder

```
                          [CLS] query [SEP] trial text [SEP]
                                        │
                                        ▼
                                  ┌──────────┐
                                  │  BERT    │
                                  └──────────┘
                                        │
                                        ▼
                                   single output
                                   (logit, then sigmoid)
                                        │
                                        ▼
                                   relevance score
```

One pass through the model, with the query and trial *concatenated*.
Now every query token can attend to every trial token. The
self-attention mechanism is doing the work it was designed for —
modeling pairwise relationships across an entire sequence.

**Crucial limitation.** The model has to be re-run for *every
(query, trial) pair*. We can't pre-compute anything because the input
*is* the pair. So cross-encoder scoring of 140K trials would mean
140K forward passes — 140,000 × 50ms = 1.9 hours per query. Useless
in production.

### The two-stage pattern

Fix: don't run the cross-encoder on all 140K. Run the bi-encoder
first to retrieve a small candidate set, then run the cross-encoder
on just those.

```
   140K trials
       │
       ▼
   bi-encoder (FAISS) + BM25 + RRF        ← fast, broad
       │  top 50
       ▼
   cross-encoder                          ← slow, accurate
       │  top 20
       ▼
   ranked output
```

This pattern — **retrieve and re-rank** — is the standard for any
modern search system. The bi-encoder maximizes *recall* (catch the
right answer somewhere in the top 200). The cross-encoder maximizes
*precision* (put the right answer at the top of the top 50).

The numbers in our pipeline:
- bi-encoder + BM25 + RRF on 140K trials → top 200: ~70 ms
- cross-encoder on top 50: ~4 sec on CPU
- Total: ~4 sec per query

That's the budget. 4 seconds is OK for a search engine. 1.9 hours is not.

---

## Part 4 — A 30-second tour of LambdaRank

After the cross-encoder we still have just one score per (query, trial)
pair. But we have *many* useful signals:

- bm25_score (raw, not just rank)
- semantic_score (cosine similarity)
- cross_encoder_score
- rrf_score (the merged signal from week 2)
- phase (1, 2, 3, 4 — Phase 3 better than Phase 1 for most patients)
- status (recruiting? active? completed?)
- enrollment (larger trials accept more patients)
- condition_exact_match (does the query word appear in trial conditions?)
- title_query_overlap (fraction of query words in title)
- has_eligibility (do we even have eligibility text?)

**LightGBM** combines these features into a single ranking score by
training a gradient-boosted decision-tree ensemble. We use the
**LambdaRank** objective, which approximately optimizes NDCG.

Here's the core idea of LambdaRank in plain English:

1. For each pair of documents *(doc_i, doc_j)* in the same query,
   compute a *gradient* that tells the model "you should score doc_i
   higher than doc_j."
2. **Weight that gradient** by the change in NDCG that would result
   if you swapped doc_i and doc_j.
3. Pairs at the *top* of the ranking move NDCG more than pairs at the
   bottom, so they get more weight.

This is why LambdaRank is a "listwise" loss: it cares about the whole
ranking, not just individual scores. And why it's NDCG-aware: it's
literally weighted by NDCG deltas.

LightGBM's API:

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    ...
}
train_data = lgb.Dataset(X, label=y, group=groups)   # 'group' = num_docs per query
model = lgb.train(params, train_data, num_boost_round=200, ...)
```

The `group` parameter is what tells LightGBM "these 30 rows belong to
query 1, the next 28 to query 2, ..." — it learns to rank within each
group, not globally.

---

## Part 5 — What we built, step by step

### Step 1 — Fine-tune the cross-encoder

**File:** `scripts/finetune_cross_encoder.py` (~530 lines)
**Config:** `configs/training/cross_encoder.yaml`

Same skeleton as the bi-encoder fine-tuning from week 3, but with a
different model class, a different loss, and a different evaluator.

#### Convert triplets to binary pairs

The training data was triplets (`anchor`, `positive`, `negative`) for
the bi-encoder. The cross-encoder wants binary pairs: each pair is
either (query, positive_trial, label=1) or (query, negative_trial,
label=0).

```python
def load_and_convert(filepath: str) -> Dataset:
    rows = []
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            query = rec["query"]
            positive = rec["positive"]
            negative = rec.get("negative", "")

            if not query or not positive:
                continue

            rows.append({"sentence1": query, "sentence2": positive, "label": 1.0})
            if negative.strip():
                rows.append({"sentence1": query, "sentence2": negative, "label": 0.0})
    return Dataset.from_list(rows)
```

This converts ~100K triplets into ~200K binary pairs (≈100K positive
+ ≈100K negative; some triplets have no negative, so the count is
slightly under 200K).

#### Build the model + loss + evaluator

```python
model = CrossEncoder("michiyasunaga/BioLinkBERT-base",
                     num_labels=1, max_length=512, device=device)
loss  = BinaryCrossEntropyLoss(model=model)
evaluator = build_evaluator(val_file=...)    # CERerankingEvaluator
```

- **`num_labels=1`** means "regression-style" output: the model emits
  a single logit per (query, trial) pair. After sigmoid, that's the
  predicted probability of "relevant."
- **`BinaryCrossEntropyLoss`** is the standard binary classification
  loss — it pushes the logit toward 1 for positives, toward 0 for
  negatives.
- **`CERerankingEvaluator`** during training: for each val triplet,
  scores the positive and the negative; aggregates Recall@k, MRR,
  NDCG over all triplets. We track `CERerankingEvaluator_ndcg@10` as
  the "best model" criterion.

#### Training arguments

```python
training_args = CrossEncoderTrainingArguments(
    output_dir="models/cross-encoder/fine-tuned",
    num_train_epochs=3,                   # ← stopped at 1 in practice
    per_device_train_batch_size=16,       # ← half the bi-encoder's 32
    learning_rate=2e-5, warmup_ratio=0.1, fp16=True,
    eval_steps=2000, save_steps=2000, save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="CERerankingEvaluator_ndcg@10",
    report_to="mlflow",
)
```

Two non-obvious choices:

- **`batch_size=16`, not 32.** Cross-encoders process the (query +
  trial) pair as one sequence, so each example is roughly twice the
  size of a bi-encoder example. The smaller batch is forced by GPU
  memory.
- **3 epochs in config, ~1 epoch in practice.** The config reflects
  the *plan*. The actual run hit early stopping (best NDCG@10
  plateaued) after ~1 epoch on a Colab T4. The trained model's
  `metadata.json` records what actually ran.

Trained on a free-tier Colab T4 GPU (~16 GB). ~3-4 hours for ~200K
pairs, 1 epoch. Output: `models/cross-encoder/fine-tuned/` (~430 MB)
+ `metadata.json`.

### Step 2 — The CrossEncoderReranker class

**File:** `src/TrialMine/models/cross_encoder.py` (~125 lines)

A thin wrapper around the trained model, with two key methods.

#### `score()` — batch scoring

```python
def score(self, query: str, trial_texts: list[str]) -> list[float]:
    if not trial_texts:
        return []
    pairs = [(query, t) for t in trial_texts]
    with time_model_cm("biolinkbert-cross-encoder"):
        scores = self.model.predict(pairs, convert_to_numpy=True)
    return scores.tolist()
```

`self.model.predict(pairs)` runs the cross-encoder on each pair
*efficiently in a batch*. ~50 pairs per ~4 seconds on CPU. The
`time_model_cm` context manager records the latency to Prometheus
(week 7 plumbing).

#### `rerank()` — scoring + blended sort

The subtle method. **Pure cross-encoder replacement of the RRF score
actually hurts NDCG:**

| Configuration | NDCG@5 |
|---|---|
| Hybrid (RRF) only | **0.816** |
| Pure CE replacement | **0.657** ← *worse!* |

**Why does CE-only hurt?** The binary training labels (relevant=1,
irrelevant=0) don't teach *graded* relevance. The CE learns to say
"right disease" or "wrong disease" but loses the finer "highly
relevant vs somewhat relevant" ordering. RRF preserves that ordering
through the rank-based fusion. Replacing RRF with raw CE strips it
out.

**The fix: blended scoring.** Keep RRF as the primary signal, add CE
as a small boost. In code:

```python
def rerank(self, query, candidates, top_k=20):
    ce_scores = self.score(query, [c["trial_text"] for c in candidates])

    # Squash CE logits to [0, 1] with sigmoid; min-max normalize RRF too.
    for c, raw in zip(candidates, ce_scores):
        c["cross_encoder_score"] = 1 / (1 + math.exp(-raw))
    rrf_min, rrf_max = min(c["score"] for c in candidates), max(c["score"] for c in candidates)
    rng = (rrf_max - rrf_min) or 1.0

    for c in candidates:
        rrf_norm = (c["score"] - rrf_min) / rng
        c["blended_score"] = 0.7 * rrf_norm + 0.3 * c["cross_encoder_score"]

    return sorted(candidates, key=lambda x: x["blended_score"], reverse=True)[:top_k]
```

The 0.7/0.3 split was chosen by trial: 1.0 (pure CE) hurts; 0.0 (no
CE) doesn't help; 0.3 captures CE's signal without overruling RRF.

| Configuration | NDCG@5 |
|---|---|
| Hybrid only | 0.816 |
| **Hybrid + CE blended (0.7/0.3)** | **0.829** (+1.6%) |

Modest on average, but on the queries where CE helps it *really*
helps:
- *"sarcoma clinical trials for young adults"*: +31%
- *"glioblastoma that has come back"*: +14%

CE is most useful exactly where the bi-encoder/BM25 are weakest.

### Step 3 — Train the LightGBM ranker

**File:** `scripts/train_ranker.py` (~570 lines)

Now the headline week 4 step: a learning-to-rank model that fuses
*all* the signals we have.

#### Feature engineering

For each labeled (query, trial) pair, `compute_features()` returns a
dict of 11 numbers:

| Feature | Source | What it captures |
|---|---|---|
| `bm25_score` | Elasticsearch | raw BM25 relevance score |
| `semantic_score` | FAISS | cosine similarity, query ↔ trial embedding |
| `cross_encoder_score` | CE | CE logit after sigmoid → [0, 1] |
| `rrf_score` | hybrid retriever | the merged BM25+semantic rank |
| `phase_numeric` | trial.phase | Phase 1→1.0, Phase 2→2.0, Phase 1/2→1.5, Phase 3→3, Phase 4→4, unknown→0 |
| `is_recruiting` | trial.status | 1 if status == `RECRUITING` else 0 |
| `is_active` | trial.status | 1 if `ACTIVE_NOT_RECRUITING` or `ENROLLING_BY_INVITATION` else 0 |
| `enrollment_log` | trial.enrollment | `log1p(enrollment)` — log-transformed trial size |
| `condition_exact_match` | query ↔ conditions | 1 if any query word appears verbatim in trial conditions |
| `title_query_overlap` | query ↔ title | fraction of query words appearing in title |
| `has_eligibility` | trial.eligibility | 1 if eligibility text is present and >10 chars |

A few choices worth noting:

- **Mix raw scores and ranks.** `bm25_score` and `rrf_score` carry
  different information — raw scores are richer per-query, ranks are
  more comparable across queries.
- **Log-transform enrollment.** Trial sizes range from 5 to 50,000.
  A linear feature would be dominated by giants; `log1p` compresses
  the range so 10 vs 100 vs 1,000 gives roughly equal-spaced gradient
  signal.
- **Boolean features as 0.0/1.0.** Decision trees split on these
  natively — no one-hot encoding needed.

#### Three label files merged

The original 20-query labeled set wasn't enough to train LightGBM
without overfitting. So we expanded:

```python
LABELS_FILES = [
    Path("data/evaluation/labeled_queries.jsonl"),    # 20 queries (IDs 0-19)
    Path("data/evaluation/test_labels.jsonl"),         # 50 queries (IDs 100-149)
    Path("data/evaluation/train_labels_extra.jsonl"),  # 80 queries (IDs 200-279)
]
```

That's 145 queries × ~40 labeled trials each = ~6,000 labeled pairs
across the three files. Cost to label all 6K: ~$3.

#### Leave-one-query-out cross-validation

With only 145 queries, a standard 80/20 train/val split would leave
just 29 val queries — too few to give a stable estimate. So we use
**leave-one-query-out (LOO)**: 145 folds, each holding out one query,
training LightGBM on the other 144, and evaluating on the held-out
one. Average all 145 held-out scores.

The implementation is a `for` loop over `query_ids` building
`train_mask` / `val_mask`, calling `lgb.train` per fold, and reading
`model.best_score["val"]["ndcg@5"]` and `ndcg@10`. Nothing fancy.

**Result:** `cv_ndcg5_mean = 0.843 ± 0.18`, `cv_ndcg10_mean = 0.831 ± 0.15`.
This is the *honest* estimate of how the model performs on unseen
queries — and it's the number we report when someone asks "is the
LightGBM blender actually any good?"

#### Final model training + feature importance

After CV, we train one final model on *all* 145 queries and save it
to `models/ranker/v3-regularized/model.lgb` (the production default as
of Phase 12; v2 and v3-vanilla preserved as evidence/revert — see
CLAUDE.md Decision 40). We also save a bar chart of LightGBM's
"feature importance by gain" (a measure of how often each feature was
used to make a useful split) to `docs/feature_importance.png`:

| Feature | Gain |
|---|---:|
| `cross_encoder_score` | **1673** ← #1 |
| `rrf_score` | 1012 |
| `phase_numeric` | 593 |
| `title_query_overlap` | 557 |
| `semantic_score` | 520 |
| `bm25_score` | 516 |
| `enrollment_log` | 487 |
| `is_recruiting` | 62 |
| `condition_exact_match` | 58 |
| `has_eligibility` | 12 |
| `is_active` | 0 |

The cross-encoder is the **single most important feature** in v2 —
even though *as a standalone replacement* it hurt NDCG (Step 2). The
pattern: CE is great as a feature, lousy as a sole ranker. Why?
Because LightGBM uses CE *alongside* the RRF score and metadata,
weighting it where it helps and ignoring it where it doesn't. The
tree-based blender is doing exactly the work the hand-tuned 0.7/0.3
split couldn't.

### Step 4 — The RankingBlender class

**File:** `src/TrialMine/models/ranker.py` (~190 lines)

A thin wrapper around the saved LightGBM model. Two methods that
matter:

- `load(model_path)` — `lgb.Booster(model_file=model_path)`.
- `rerank(query, candidates, top_k=20)` — call `compute_features` on
  each candidate, stack into an `(N, 11)` array, run `model.predict`,
  attach `blender_score` to each candidate, sort, return top-k.

The interface intentionally mirrors `CrossEncoderReranker.rerank`:
same `(query, candidates, top_k)` signature. Plug-and-play with the
pipeline.

### Step 5 — The full pipeline

**File:** `src/TrialMine/retrieval/hybrid.py`'s `full_pipeline` method

The 5-stage pipeline lives as one method on `HybridRetriever`. It
takes a query (and optional filters), and returns ranked results +
per-stage timings:

```python
def full_pipeline(self, query, reranker, blender=None,
                  top_k=20, rerank_top_k=50, filters=None):
    # Stage 1: BM25 → top 200 (with filters applied)
    bm25_results = self.bm25.search(query, filters=filters, top_k=200)

    # Stage 2: Semantic → top 200 (no native filtering)
    query_emb = self.embedder.embed_text(query)
    semantic_results = self.semantic.search(query_emb, top_k=200)

    # Stage 3: RRF merge → drop semantic-only items that fail filters
    fused = reciprocal_rank_fusion(bm25_results, semantic_results)
    fused = self._post_filter(fused, filters)            # see "filter" note below
    candidates = self._enrich(fused[:rerank_top_k], bm25_results, semantic_results)

    # Stage 4: Cross-encoder → blended score per candidate
    texts = [c["trial_text"] for c in candidates]
    for c, raw in zip(candidates, reranker.score(query, texts)):
        c["cross_encoder_score"] = sigmoid(raw)

    # Stage 5: LightGBM blender → final top-k (with hand-tuned fallback)
    ranked = (blender.rerank(query, candidates, top_k=top_k)
              if blender else self._ce_blended_sort(candidates, top_k))
    return ranked, timings
```

(The real source records `t = perf_counter()` around each stage and
returns the `timings` dict; I've omitted those for clarity.)

Three non-obvious bits worth knowing:

**Why filter at the post-RRF level instead of pre-filtering FAISS?**
FAISS has no native filter. Even when the request asks for
`status=RECRUITING`, FAISS happily returns the top-200 by cosine
regardless. Without the post-RRF filter, COMPLETED trials with strong
embeddings leak in via the semantic side and the cross-encoder
scores them as if they were valid candidates. So after RRF merge we
drop any *semantic-only* item whose Elasticsearch metadata doesn't
match the filter. BM25 hits are already filtered, so they fast-path
through unchanged.

**Why fetch `brief_summary` separately for CE input?** BM25's search
response doesn't include the full `brief_summary` field (it's stored
but not returned in `_source` by default). We make one extra
`get_trial(nct_id)` call per candidate to pull it, so the
cross-encoder has the full trial text to reason over. Adds ~50 ms,
worth it.

**Why does the blender step have a CE-blended fallback?** During
early development we often ran the pipeline without a trained
LightGBM model on disk. The fallback (the same hand-tuned 0.7 RRF +
0.3 CE formula from the CrossEncoderReranker) keeps the pipeline
working end-to-end with one fewer dependency.

### Step 6 — THE ABLATION TABLE

**File:** `scripts/evaluate.py` (~700 lines)

For each of the 5 stages, we record NDCG@5, NDCG@10, MRR (each with a
bootstrap 95% confidence interval) and median latency. The script has
five `evaluate_*` functions (one per stage: `bm25_only`,
`semantic_only`, `hybrid`, `hybrid_plus_ce`, `full_pipeline`); each
returns per-query lists of metrics that we feed into bootstrap.

#### Bootstrap CI in one paragraph

`bootstrap_ci(values, n_bootstrap=1000)` resamples the per-query
metric values *with replacement* 1,000 times. Each resample gives one
mean. The 2.5th / 97.5th percentiles of those 1,000 means form the
95% CI. We report `mean ± half-width`. Intuition: "if I drew 20 fresh
queries from the same distribution and measured again, my answer
would land in this band."

#### The output

The script prints THE ABLATION TABLE to stdout AND writes it to
`docs/evaluation-report.md`:

```
| Method                            |    NDCG@5     |    NDCG@10    |      MRR      | Median Latency |
|-----------------------------------|---------------|---------------|---------------|----------------|
| BM25 only                         |  0.789±0.12   |  0.756±0.13   |  0.912±0.09   |      22ms      |
| Semantic only                     |  0.703±0.12   |  0.700±0.10   |  0.807±0.12   |      37ms      |
| Hybrid (BM25 + Semantic)          |  0.816±0.10   |  0.796±0.08   |  0.917±0.08   |      72ms      |
| + Cross-Encoder Re-ranking        |  0.829±0.09   |  0.817±0.07   |  0.950±0.06   |     6295ms     |
| + Metadata Blender                |  0.980±0.02   |  0.921±0.03   |  1.000±0.00   |     6472ms     |
```

NDCG monotonically increases at every stage. ✅ The pipeline is doing
what we hoped.

But notice the 0.980 NDCG@5 at the bottom. That's *suspicious*. It
implies near-perfect ranking. The honest explanation goes in the
report:

> **Overfitting warning on blender results.** The LightGBM model
> was trained on all 20 queries and evaluated on those same 20
> queries. The ablation numbers for "+ Metadata Blender" are
> optimistic. The honest number is the leave-one-query-out CV
> result: NDCG@5=0.844, NDCG@10=0.840.

This is the kind of caveat a senior interviewer values. *"I know the
table is biased; here's the honest number; the directional claim
still holds."*

#### MLflow logging

Each method is logged as a separate MLflow run in the
`trialmind-ablation` experiment, plus one combined `final-ablation`
summary run with the entire table as an artifact.

### Step 7 — Fair held-out evaluation

**File:** `scripts/build_fair_eval.py` (~700 lines)

The week-4 ablation table has a fairness problem: LightGBM was
trained on the same queries it was evaluated on. To get an honest
number, we built a **fair held-out test set**:

```python
TEST_QUERIES = [
    # 50 deliberately hard queries the LightGBM never saw:
    # - rare cancers (Merkel cell, Ewing sarcoma, GIST)
    # - patient phrasing ("my mom has stage 4 colon cancer")
    # - misspellings ("non hodgkins lympoma")
    # - specific biomarkers ("BRAF V600E melanoma dabrafenib trametinib")
    # - phase/status combinations
    # ... 50 total ...
]
```

Pool top-20 from BM25, semantic, and hybrid. Label the union with
Claude Haiku. Run the full ablation on these 50 *new* queries.

Result (`docs/fair-evaluation-report.md`):

```
| Method                            |    NDCG@5     |    NDCG@10    |      MRR      |  Latency   |
|-----------------------------------|---------------|---------------|---------------|------------|
| BM25 only                         | 0.617±0.09    | 0.614±0.08    | 0.768±0.10    |     21ms   |
| Semantic only                     | 0.606±0.07    | 0.603±0.06    | 0.815±0.08    |     36ms   |
| Hybrid (BM25 + Semantic)          | 0.636±0.08    | 0.644±0.06    | 0.807±0.09    |     71ms   |
| + Cross-Encoder Re-ranking        | 0.651±0.08    | 0.657±0.06    | 0.825±0.09    |   6166ms   |
| + Metadata Blender                | 0.670±0.08    | 0.657±0.07    | 0.806±0.09    |   6134ms   |
```

The numbers are lower (these queries were curated to be hard) but
**NDCG@5 still increases monotonically across stages.** The pipeline
generalizes.

Two interesting observations:
- The held-out gain from blender is modest: NDCG@5 0.651 → 0.670 (+3%).
  Nothing like the 0.829 → 0.980 from the same-query setup. That's
  the overfitting being honest about itself.
- MRR drops slightly from CE (0.825) to blender (0.806). The blender
  optimizes list-level NDCG, sometimes at the cost of first-result
  quality. We document this in the report.

---

## Part 6 — Why we made the choices we did

### Decision 13 — Binary classification (relevance ≥ 2) for cross-encoder labels

**Context.** Our 990 labels are graded 0-3. We could train CE on the
graded scores (regression) or on a binarized version (classification).

**We chose binary, threshold ≥ 2.**

**Why?** With only ~990 pairs, regression is noisy. The model can't
learn the difference between "1.7" and "2.3" — too few examples per
score level. Binary (relevant vs not) gives cleaner gradients.

**Why threshold ≥ 2?** Because score 2 means *"the patient could
potentially be eligible."* That's the practical bar — anything below
that wouldn't be shown to the patient anyway.

**Trade-off acknowledged.** Binary labels lose the "highly relevant
vs marginally relevant" distinction. The CE produces sigmoid scores
that *don't* capture graded relevance. That's why pure CE
replacement hurts NDCG (Step 2 above) — and why we use blended
scoring + LightGBM. They each compensate for the limitation.

### Decision 14 — LightGBM LambdaRank over a neural ranker

**Context.** Need a learning-to-rank model on 11 features.

**Options.**
- **LightGBM (gradient-boosted trees) with LambdaRank.** Industry
  standard at Bing, LinkedIn, Yandex.
- **XGBoost or CatBoost.** Other GBT libraries. Comparable.
- **Neural network.** TabNet, DeepFM, simple MLP.
- **Linear / logistic.** Trivial. Just regress the relevance.

**We chose LightGBM.** Three reasons:

- **Optimizes NDCG directly.** LambdaRank weights gradients by NDCG
  delta, so the loss function and the eval metric are aligned.
- **Trains in seconds on our scale.** 145 queries × 40 docs × 11
  features = 6K rows. LightGBM is built for this.
- **Interpretable feature importance.** A neural ranker would give
  us a black box. LightGBM gives us the bar chart. Crucial for
  debugging and interview talking points.

**Trade-off.** Less power than a deep neural ranker for very large
training sets. At our scale, neural would overfit catastrophically.
At >100K queries we'd revisit.

### Decision 15 — Top 50 candidates for cross-encoder

**Context.** The CE is the latency bottleneck (~80 ms/pair on CPU).
How many candidates to score?

**Math.** 50 × 80 ms = 4 sec. 100 × 80 ms = 8 sec. 200 × 80 ms = 16 sec.

**We chose 50.** Reasons:

- 4 seconds is acceptable for clinical search.
- We measured: among the 990 labeled trials, >95% of the relevant
  ones (score ≥ 2) appeared in the hybrid top 50. Doubling to 100
  doesn't catch many more truly-relevant trials — they're not in
  the candidate pool to begin with.
- Latency budget for the full agent (week 6) is 15 sec. CE at 4 sec
  leaves 11 sec for everything else. CE at 16 sec doesn't.

**Trade-off.** A few percent recall lost compared to top-100. The
ablation table confirms top-50 is a sweet spot — we couldn't
detect a NDCG@10 difference vs top-100.

### Decision 16 — Document if a stage *doesn't* improve NDCG

**Context.** The early ablation table showed pure CE *replacement*
(not blended) hurt NDCG. We documented this honestly, kept the
hybrid+blended formulation, and explained why.

**Interview answer.** *"Not every component improved metrics. Pure CE
replacement hurt NDCG by 12% because the binary training labels don't
teach graded relevance. We pivoted to blended scoring (0.7 RRF + 0.3
CE) which gave +1.6% NDCG. Then LightGBM exploits CE as a feature
*in combination with* RRF — which works because the model can learn
*when* to trust each signal. The CE became LightGBM's #1 feature by
gain even though it doesn't work standalone."*

This honesty is the most underrated interview signal. Anyone can
say *"my model is great."* What signals senior-level engineering is
*"my model fails this way; I measured the failure; I fixed it; here's
the lesson."*

---

## Part 7 — File map

| File | New in week 4 | Purpose |
|---|---|---|
| `configs/training/cross_encoder.yaml` | new | CE training hyperparameters |
| `scripts/finetune_cross_encoder.py` | new | CE training script (~530 lines) |
| `notebooks/finetune_cross_encoder.ipynb` | new | Colab T4 notebook |
| `src/TrialMine/models/cross_encoder.py` | new | `CrossEncoderReranker` (score + blended rerank) |
| `scripts/evaluate_cross_encoder.py` | new | Evaluate CE alone vs hybrid baseline |
| `scripts/demo_reranker.py` | new | Before/after CE re-ranking demo |
| `scripts/train_ranker.py` | new | LightGBM LambdaRank training (~570 lines) |
| `src/TrialMine/models/ranker.py` | new | Feature engineering + `RankingBlender` |
| `scripts/evaluate.py` | new | THE ABLATION TABLE script (~700 lines) |
| `src/TrialMine/retrieval/hybrid.py` | extended | added `full_pipeline()` method |
| `scripts/build_fair_eval.py` | new | 50-query held-out evaluation (~700 lines) |
| `scripts/expand_eval_data.py` | new | Generates the 75 + 50 extra training queries |
| `models/cross-encoder/fine-tuned/` | new artifact | ~430 MB CE weights + metadata.json |
| `models/ranker/v3-regularized/model.lgb` | new artifact | **Production LightGBM model** + metadata.json (Phase 12 ship; v2 + v3-vanilla preserved as evidence/revert) |
| `data/evaluation/ranking_features_v3.csv` | new artifact | 6,018 rows × 11 features computed against v2 stack (the input the production LGB sees) |
| `data/evaluation/ranking_features_v2.csv` | preserved | 6,018 rows × 11 features computed against v1 stack — historical |
| `data/evaluation/test_labels.jsonl` | new artifact | 50-query labels for training |
| `data/evaluation/train_labels_extra.jsonl` | new artifact | 75-query labels for training |
| `data/evaluation/test_labels_v2.jsonl` | new artifact | 50-query held-out labels |
| `docs/evaluation-report.md` | new | THE ABLATION TABLE markdown |
| `docs/fair-evaluation-report.md` | new | Fair held-out ablation markdown |
| `docs/feature_importance.png` | new | LightGBM feature-importance bar chart |

---

## Part 8 — Interview prep

### 8.1 The big question

> *"Walk me through your retrieval pipeline."*

Model answer:

> "Five stages. Stage 1 is BM25 keyword search via Elasticsearch with
> field boosting — title 3×, conditions 2× — and optional metadata
> filters like status=RECRUITING. Stage 2 is semantic search via a
> fine-tuned BioLinkBERT bi-encoder over a FAISS IndexFlatIP. Both
> retrievers return their top 200. Stage 3 merges them with Reciprocal
> Rank Fusion at k=60, then post-RRF filters out semantic-only hits
> that don't match the request filters. Stage 4 takes the top 50
> after fusion and scores them with a fine-tuned BioLinkBERT
> cross-encoder. Stage 5 takes those scores plus 7 metadata features
> — phase, status, enrollment, condition match, title overlap,
> eligibility presence — and feeds them to a LightGBM LambdaRank
> blender that's been trained on 145 labeled queries. The output is a
> ranked top-20.
>
> Latency budget: ~22 ms BM25, ~37 ms semantic, ~10 ms merge, ~4
> sec cross-encoder, ~10 ms LightGBM. Roughly 4-6 seconds end-to-end."

### 8.2 Specific questions

**"What's the difference between a bi-encoder and a cross-encoder?"**
Bi-encoder embeds query and document separately, compares with
cosine. Cross-encoder takes [CLS] query [SEP] doc [SEP] as one
sequence, every token can attend to every other token, output is one
score. Bi-encoder is fast (pre-compute doc embeddings). Cross-encoder
is accurate (sees cross-attention). We use bi-encoder for retrieval
(140K → 200), cross-encoder for re-ranking (200 → 20).

**"Why don't you just use the cross-encoder for everything?"**
Latency. 80 ms per pair × 140K trials = 1.9 hours per query. The
cross-encoder is only fast enough on a small candidate pool.

**"Pure cross-encoder replacement hurt NDCG. Why?"**
Two reasons. (1) Binary training labels don't teach graded relevance —
the CE outputs collapse to "right disease / wrong disease" rather
than "highly relevant / somewhat relevant / not relevant." (2) RRF
already encodes useful per-method signals. Replacing it with raw CE
strips those signals out. Blending at 0.7 RRF / 0.3 CE keeps the RRF
quality while adding CE as a boost.

**"What does LambdaRank actually optimize?"**
Approximately, NDCG. For each pair (i, j) within the same query,
compute a pairwise gradient that says "i should rank above j," then
weight that gradient by the change in NDCG that swapping i and j
would cause. Pairs at the top of the ranking move NDCG more, so they
get more weight. Net effect: gradients push the right docs toward
the top.

**"Why leave-one-query-out instead of train/val split?"**
With only 145 queries, a 80/20 split leaves 29 val queries — too few
to give a stable estimate. LOO gives 145 folds; aggregate is
near-deterministic.

**"What's the difference between the 0.980 NDCG and the 0.844 LOO
NDCG?"**
The 0.980 is in-sample: model trained on those queries, evaluated on
the same. The 0.844 is held-out: trained on 144, evaluated on 1,
averaged over all 145 folds. The 0.844 is the honest number. We
document this in the evaluation report so nobody gets fooled.

**"How does the post-RRF filter work?"**
After RRF merges BM25 + semantic top-200s, we look at every
candidate. If it came from BM25, it's already filtered (BM25 supports
the `filter` clause). If it came from semantic-only, FAISS didn't
filter — so we fetch the trial doc from ES and check whether the
fields match. Drop it if not. Adds ~250 ms but eliminates the
"COMPLETED trials dominating top-10 when status=RECRUITING was
requested" bug.

**"What's the bootstrap CI doing?"**
Resampling-with-replacement of the per-query NDCG values. Compute
the mean of each resample, take the 2.5th and 97.5th percentiles.
Tells me how much my point estimate would wobble if I drew 20 new
queries from the same distribution. With 20 queries, the wobble is
±0.10ish — wide. That's why we ran the fair eval on 50 new queries
and tightened it to ±0.07ish.

**"Why is feature importance #1 the cross-encoder when pure CE hurt
the ranking?"**
Because LightGBM uses CE *alongside* the other features. The CE
score is highly informative on its own — it perfectly distinguishes
"right disease / wrong disease." But it lacks the granularity to
order *within* the right-disease group. RRF gives us that intra-group
ordering. Together, they cover both signals: CE filters out the
wrong-disease results; RRF + metadata orders the right ones.

### 8.3 Code-review questions

1. **Why is the blender's `compute_features` called from both
   `train_ranker.py` and `RankingBlender.rerank`?**
   Single source of truth. The training-time and inference-time
   feature representations *must* match exactly, or the model will
   produce nonsense at inference. By calling the same function in
   both places, we guarantee they match.

2. **Why is `phase_numeric` a single number rather than separate
   columns for each phase?**
   LightGBM trees can split on numeric thresholds natively. A
   `phase_numeric ≤ 2.5` split cleanly partitions Phase 1+2 from
   Phase 3+4. Separate one-hot columns would force the tree to make
   3-4 separate splits to express the same thing. Less efficient.

3. **Why do you compute `enrollment_log` instead of raw enrollment?**
   Enrollment ranges from 5 to 50,000. Linear feature would be
   dominated by giant trials. `log1p(enrollment)` compresses the
   range so the relative difference between 5 and 50 (1× → 1.7×) is
   similar to between 50 and 5,000 (3× → 6×). Better gradient
   signal.

4. **Why does `full_pipeline` build `trial_text` again instead of
   using the embedder's output?**
   Because the cross-encoder is a different model with a different
   tokenizer and max sequence length. We rebuild the text with the
   format the CE expects (`title [SEP] conditions [SEP] summary`),
   truncate to 2048 chars (~512 tokens). The embedder's text
   representation is fine for FAISS but might be tokenized
   differently.

### 8.4 Tradeoff questions

**"Could you skip the cross-encoder if it's the latency bottleneck?"**
Yes — at the cost of ~3% NDCG@10 on the held-out eval (0.657 →
0.644 in the fair eval). For a free public-health tool serving
patients, 3% might be worth saving 4 seconds. For a clinical
decision-support tool serving doctors, 3% is meaningful. Different
products, different tradeoffs.

**"Could you replace LightGBM with a learned weighted average?"**
Yes — `final = w1 × bm25 + w2 × semantic + w3 × ce + w4 × phase + ...`
where the w's are learned by gradient descent. We tried it informally;
LightGBM was 3-5% better because it can model nonlinear interactions
(e.g., "trust phase_numeric only when enrollment is large").

**"How would you scale this to 100K queries/day in production?"**
Three changes. (1) Move CE to a GPU inference service with batching;
~10× throughput. (2) Cache CE scores per (query_template, trial)
since most queries fall into similar shapes. (3) For very common
queries, cache the entire ranked list (Redis) — reset on model
updates.

---

## Part 9 — How to run everything

```bash
# Pre-reqs from weeks 1-3:
# - Elasticsearch running
# - data/faiss_finetuned.index
# - data/evaluation/labeled_queries.jsonl

# Step 1: Train the cross-encoder (Colab T4 ~3-4 hrs)
# Open notebooks/finetune_cross_encoder.ipynb, run all cells.
# Download models/cross-encoder/fine-tuned/ to your laptop.

# Step 2: Verify the cross-encoder before-and-after
python scripts/demo_reranker.py
python scripts/evaluate_cross_encoder.py

# Step 3: Expand the labeled set (Claude Haiku, ~$2.50 in API)
python scripts/expand_eval_data.py            # generates 75 + 50 extra queries

# Step 4: Train LightGBM (~30 min for feature engineering + seconds for training)
OMP_NUM_THREADS=1 make train-ranker
# OMP_NUM_THREADS=1 is required on macOS — FAISS + LightGBM both ship libomp,
# loading both segfaults without OMP_NUM_THREADS=1.

# Step 5: Run the full ablation
OMP_NUM_THREADS=1 make evaluate
# Writes docs/evaluation-report.md, prints to stdout, logs to MLflow.

# Step 6: Run the fair held-out evaluation
OMP_NUM_THREADS=1 python scripts/build_fair_eval.py
# Writes docs/fair-evaluation-report.md.
```

In MLflow (`make mlflow`, then [http://localhost:5001](http://localhost:5001)):
- `trialmind-cross-encoder` — CE training run.
- `trialmind-ranker` — LightGBM training run with feature importance.
- `trialmind-ablation` — 5 runs (one per stage) plus a `final-ablation` summary.

---

## Part 10 — End of week 4 checklist

- [x] Cross-encoder fine-tuned on Colab T4 (BinaryCrossEntropyLoss,
      ~200K binary pairs, 1 epoch, NDCG@10 ≈ 0.42 on val)
- [x] `CrossEncoderReranker` with `score()` + blended `rerank()`
- [x] Pure CE *replacement* hurt NDCG (-12%); blended CE *boosts*
      NDCG (+1.6%) — both documented honestly
- [x] Expanded labeled set: 145 queries × ~40 trials = 6,018 pairs
      (3 source files merged)
- [x] LightGBM LambdaRank with 11 features, leave-one-query-out CV:
      NDCG@5=0.843, NDCG@10=0.831
- [x] Feature importance chart at `docs/feature_importance.png`
      (CE, RRF, phase_numeric top three)
- [x] Full 5-stage `full_pipeline` method on `HybridRetriever` with
      per-stage timings
- [x] THE ABLATION TABLE in `docs/evaluation-report.md` —
      monotonically increasing NDCG, with overfitting caveat
- [x] Fair held-out evaluation on 50 new queries
      (`docs/fair-evaluation-report.md`)
- [x] All experiments tracked in MLflow (`trialmind-cross-encoder`,
      `trialmind-ranker`, `trialmind-ablation`)
- [x] Design decisions 13-16 logged
- [x] CLAUDE.md updated with current state

---

## Part 11 — Where this leaves us

🎉 **Month 1 complete.** We have a working 5-stage retrieval pipeline
with quantitative evaluation:

```
   140K trials  →  BM25 (22ms)  →
                   Semantic (37ms)  →  RRF merge (10ms)
                                 →    Cross-encoder (4s, top 50)
                                 →    LightGBM blender (10ms, top 20)
                                 →    output
```

Fair held-out NDCG@5 = 0.670, NDCG@10 = 0.657, MRR = 0.806.
Latency ~6 seconds end-to-end on CPU.

What's missing for an actual demo a patient could use:
- **Eligibility parsing** (week 5): turn the eligibility text blob
  into structured (age, sex, prior_treatments, ...) fields.
- **An agent layer** (week 6): turn a free-text patient query into a
  structured profile, run the pipeline, generate explanations,
  return verdict + reasons.
- **Containerized deployment** (week 7): one `docker compose up` to
  bring up six services. UI in a browser. Prometheus + Grafana.
- **Tests, CI, production-thinking** (week 8): pytest suites, GitHub
  Actions, A/B test stub, data-quality checks.

The retrieval pipeline is the foundation. Everything else is product
and engineering on top of it.

---

If you can explain Parts 3, 4, and 5 in your own words — and answer
"why does pure CE replacement hurt NDCG, but CE as a feature ranks
#1?" — you've internalized the heart of week 4.
