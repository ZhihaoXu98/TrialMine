# Week 2 — Semantic Search + Hybrid Retrieval (Student Edition)

A textbook walkthrough of the second week of TrialMine. By the end you
should be able to (a) explain what an embedding *is*, (b) compare BM25
vs semantic search vs hybrid in plain English, (c) navigate every file
we touched, and (d) answer interview questions about why we made the
choices we did.

If you've never trained a neural network, that's fine. We don't train
anything this week — we use a pre-trained model off the shelf. Every
term gets defined the first time it shows up.

---

## Part 1 — Why we need this

Let's start with a story.

A patient types into the search box:

> *"my mom has stomach cancer that came back"*

A clinical trial in our database says:

> *"A Phase 3 Study of Pembrolizumab in Recurrent Gastric Adenocarcinoma"*

A human reading both sentences immediately knows they're about the same
thing. *Stomach* and *gastric* are synonyms. *Came back* and *recurrent*
are synonyms. *Adenocarcinoma* is a kind of cancer.

Last week's BM25 search — the one we built in week 1 — sees **zero
overlapping words** between the query and the trial title. Score: zero.
The patient never sees this trial.

This is called the **vocabulary mismatch problem**, and it's the single
biggest reason patients can't find trials with keyword search. Doctors
write in Latinate, hyphenated, capitalized medical English. Patients
write in plain spoken English. BM25 only matches literal words.

So we need a search method that understands **meaning** instead of just
**words**. That's what semantic search does.

The week 2 deliverable: any time a patient types in plain English, the
system finds the relevant trials *even if the trials are written in
medical jargon*. We'll do this by adding a second retriever (semantic)
alongside the existing BM25, and merging their results.

---

## Part 2 — Vocabulary you'll meet

### ML / engineering terms

- **Embedding** — a list of numbers (we use 768 of them) representing
  the *meaning* of a piece of text. Texts with similar meanings get
  similar embeddings.
- **Cosine similarity** — the cosine of the angle between two
  embedding vectors. Range: -1 (opposite) to +1 (identical). For
  sentence embeddings on related text, almost always in [0, 1].
- **BERT** — a transformer-based language model from Google (2018).
  Reads text and produces one vector per word.
- **BioLinkBERT** — a 2022 BERT variant from Stanford, pre-trained on
  PubMed (a database of biomedical research papers) plus the
  *citations* between papers. It already knows which medical concepts
  tend to appear together.
- **Sentence transformer** — a wrapper around BERT that combines all
  the per-word vectors into one vector representing the whole
  sentence (typically by averaging — "mean pooling").
- **FAISS** (Facebook AI Similarity Search) — a fast C++ library for
  nearest-neighbor search over vectors. Given a query vector, returns
  the closest stored vectors in milliseconds.
- **Reciprocal Rank Fusion (RRF)** — a way to merge two ranked lists
  using only the *ranks*, not the underlying scores. Useful when two
  rankers' scores live on different scales (BM25 vs cosine).
- **MLflow** — an experiment-tracking tool. Each run logs its
  hyperparameters, metrics, and output files to a database for
  side-by-side comparison.
- **Anisotropy / hub trial problem** — a known failure of pre-trained
  embeddings: instead of spreading texts across the vector space,
  almost everything maps near the same point. We'll see this hit us
  later in the week.

### Medical terms (only what you need)

Some of our 20 test queries use terminology you may not recognize.
Quick gloss:

- **CAR-T cell therapy** — a treatment that takes a patient's own
  immune cells, engineers them in a lab to recognize cancer, and puts
  them back. Used mainly for blood cancers.
- **Glioblastoma** (GBM) — an aggressive brain tumor.
- **Pediatric** — for children.
- **Triple-negative breast cancer** — a subtype lacking three common
  hormone/protein receptors; harder to treat with standard therapies.
- **Microsatellite instability (MSI)** — a DNA-repair defect; tumors
  with it often respond well to immunotherapy.
- **PARP inhibitor** — a class of drug used in ovarian/breast cancer.
- **Neuroblastoma** — a childhood cancer of nerve tissue.
- **Mesothelioma** — a rare cancer linked to asbestos exposure.
- **Hepatocellular carcinoma** — the main type of liver cancer.
- **BCG** (in *"bladder cancer BCG unresponsive"*) — a bacterial
  vaccine used to treat early-stage bladder cancer.
- **Ibrutinib** — a targeted-therapy drug for chronic lymphocytic
  leukemia.
- **EGFR mutation** — a common genetic alteration in lung cancer that
  responds to specific targeted drugs.
- **HPV** — human papillomavirus; causes some head/neck cancers and
  cervical cancer.
- **Multiple myeloma** — a blood cancer of plasma cells.
- **Recurrent / "came back"** — cancer that returned after treatment.
- **Metastatic / "spread"** — cancer that has moved from where it
  started to another organ.

---

## Part 3 — A 30-second tour of embeddings

Imagine each piece of text gets translated into a point in
768-dimensional space.

```
                "stomach cancer"
                       ●
                        \
                         ● "gastric neoplasm"
                          \
                           \
                            \
        "bicycle repair" ●---\----- ● "bone marrow transplant"
              (very far away)        (far from "bicycle"; close to medical text)
```

The translation is done by a neural network — in our case, BioLinkBERT.
The network was trained on millions of pairs of biomedical sentences
where the answer "are these related?" was already known (citations,
co-mentions, etc.). Through training, it learned to put related
sentences close together in vector space.

To search, we:
1. Embed every trial in the database. This is a one-time cost (~30 min
   on CPU, ~5 min on GPU).
2. At query time, embed the query.
3. Use FAISS to find the top-k stored embeddings closest (by cosine
   similarity) to the query embedding.

That's it. No literal word matching anywhere. *Stomach* in the query
matches *gastric* in the trial because those two words map to nearby
points.

---

## Part 4 — A 30-second tour of Reciprocal Rank Fusion (RRF)

After this week we have two retrievers:

- **BM25** returns its top 200 trials with scores like `8.4, 7.1, 6.9, ...`.
- **Semantic** returns its top 200 trials with cosine similarities like
  `0.62, 0.58, 0.55, ...`.

Each list ranks trials differently. Some trials appear in both lists.
Many appear in only one. The question: **how do we merge them into a
single ranked list?**

The wrong answer is to add the scores: `8.4 + 0.62 = 9.02`. BM25 scores
are unbounded and depend on corpus statistics; cosine similarities are
in [0, 1]. They live in incommensurable scales. Adding them is
nonsense.

The right answer is **rank-based fusion**: forget the scores, use the
*positions*.

Reciprocal Rank Fusion (RRF) gives every trial a score of:

```
RRF(d) = sum over each retriever of  1 / (k + rank(d))
```

with `k = 60` (the standard constant from Cormack et al., 2009).

Worked example. Trial NCT12345 is rank 3 in BM25 and rank 8 in semantic:

```
RRF = 1/(60+3) + 1/(60+8)
    = 1/63 + 1/68
    = 0.01587 + 0.01471
    = 0.03058
```

Trial NCT99999 is rank 1 in BM25 only (not in semantic top 200):

```
RRF = 1/(60+1) = 0.01639
```

NCT12345 wins because *it appeared in both lists*, even though
NCT99999 had the better rank in one. **Multi-method consensus matters
more than peak position in any single method.**

Three nice properties:
- **Score-scale independent.** RRF doesn't care if BM25 scores are 8 or
  800 — only their order.
- **Parameter-free.** k=60 works in practice. There's nothing to tune
  unless you want to.
- **Symmetric.** Treats both methods equally. If you trust one method
  more than the other, you'd switch to a learned weighting (we do this
  in week 4).

---

## Part 5 — What we built, step by step

### Step 1 — The embedder

**File:** `src/TrialMine/models/embeddings.py` (~150 lines)

A thin wrapper around `sentence-transformers` so the rest of the code
doesn't need to know HuggingFace internals. Three public methods:

```python
class TrialEmbedder:
    def embed_text(self, text: str) -> np.ndarray:
        """One string → one 768-d normalized embedding."""

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Many strings → (N, 768) array. Used for indexing the corpus."""

    def prepare_trial_text(self, trial: Trial) -> str:
        """Build the text we feed to the model from a Trial object."""
```

#### How the trial text is built

We concatenate three fields with a special separator:

```python
def prepare_trial_text(self, trial: Trial) -> str:
    parts = [trial.title, " ".join(trial.conditions), trial.brief_summary]
    parts = [p for p in parts if p]                  # drop empty fields
    text = " [SEP] ".join(parts)
    return text[:2048]                                # ~512 tokens
```

`[SEP]` is BERT's official "section separator" token. It tells the
model *"these are different sections; don't let words from one field
spill into the next."* Without it, a stopword in the title would
interfere with the analysis of the conditions section.

The 2048-character truncation is rough (~512 tokens in English).
BERT-family models cap at 512 tokens, so longer input would crash
the encoder. Losing the tail of a long summary is acceptable — most
of the signal is in the first paragraph.

#### One real-world wrinkle: explicit module loading

`sentence-transformers` was designed for models that ship with a
`modules.json` config file (e.g., `all-MiniLM-L6-v2`). BioLinkBERT
doesn't ship one — it's a plain HuggingFace BERT checkpoint. Loading
it with the default constructor falls back to a guessed configuration
that crashes (SIGSEGV) on macOS.

The fix is to construct the model *explicitly* — `Transformer +
Pooling(mean)` — instead of relying on the default. The check for
this lives in `_needs_explicit_modules` and adds ~10 lines to the
constructor; details are in the source.

**Lesson:** when a framework has a "magic" code path and an "explicit"
code path, prefer explicit. Magic breaks silently.

### Step 2 — The FAISS index wrapper

**File:** `src/TrialMine/retrieval/semantic.py` (~140 lines)

FAISS is a fast C++ library for nearest-neighbor search. We wrap it
in a small `FAISSIndex` class that adds two things FAISS doesn't
provide: (a) the mapping from FAISS row index → NCT ID, and (b) save
/ load helpers.

The class has four methods: `build(embeddings, trial_ids)`,
`search(query_embedding, top_k)`, `save(path)`, `load(path)`. All
straightforward; the only design choices worth flagging:

**Choice 1 — `IndexFlatIP`.** FAISS offers many index types. We pick
`IndexFlatIP` (Inner Product, exact). For unit-normalized vectors,
inner product *equals* cosine similarity, so we normalize every
embedding before adding it. Exact search over 140K vectors takes ~30
ms — fast enough that approximate indexes (IVF, HNSW) aren't worth
the complexity at this scale.

**Choice 2 — Save in two files.** FAISS only stores the vectors; it
has no concept of "this row corresponds to NCT12345." So we save:
- `data/trial_embeddings.faiss` — binary FAISS index (~412 MB)
- `data/trial_embeddings.json` — list of NCT IDs in row order (small)

Loading reads both back in sync.

### Step 3 — Build the index from SQLite

**File:** `scripts/build_index.py` (~260 lines, extended from week 1)

`build_semantic_index()` does the obvious thing — pull every trial
from SQLite, build the embedder text, encode in batches, normalize,
add to FAISS — but with two non-obvious choices.

**Choice 1 — Stream from SQLite in 2,000-trial chunks.** Loading all
140K trials at once would hold ~700 MB of text + ~412 MB of output
embeddings + the 430 MB BERT model in memory at the same time. Most
laptops can't fit this. We process 2,000 at a time and `del` each
chunk so Python releases the buffers. Peak memory: ~100 MB at a time.

**Choice 2 — Embed title + conditions + brief_summary only.** Not
eligibility criteria. Why? Eligibility text is mostly boilerplate
(*"informed consent required, no major surgery within 4 weeks"*) that
appears verbatim in thousands of trials. Including it would push
trials toward each other in vector space *because they share the same
legalese*, regardless of cancer type. We leave eligibility for BM25
(literal match) and the cross-encoder in week 4 (which sees the full
context).

**Result:**
- 140,723 trials embedded in ~32 minutes on CPU (8 minutes on a Colab
  T4 GPU).
- FAISS index: 412 MB on disk; mapping JSON: 3 MB.

### Step 4 — The hybrid retriever

**File:** `src/TrialMine/retrieval/hybrid.py` (~325 lines)

The fusion function is short enough to read end-to-end:

```python
def reciprocal_rank_fusion(bm25_results, semantic_results, k=60):
    scores, bm25_ranks, semantic_ranks = {}, {}, {}

    for rank, r in enumerate(bm25_results, start=1):
        scores[r["nct_id"]] = scores.get(r["nct_id"], 0.0) + 1.0 / (k + rank)
        bm25_ranks[r["nct_id"]] = rank

    for rank, (nct_id, _) in enumerate(semantic_results, start=1):
        scores[nct_id] = scores.get(nct_id, 0.0) + 1.0 / (k + rank)
        semantic_ranks[nct_id] = rank

    # Tag each fused result with its source for debugging / UI
    fused = []
    for nct_id, rrf in scores.items():
        in_bm25 = nct_id in bm25_ranks
        in_sem = nct_id in semantic_ranks
        source = "both" if (in_bm25 and in_sem) else ("bm25_only" if in_bm25 else "semantic_only")
        fused.append({"nct_id": nct_id, "rrf_score": rrf, "source": source,
                      "bm25_rank": bm25_ranks.get(nct_id),
                      "semantic_rank": semantic_ranks.get(nct_id)})
    return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)
```

The `source` tag (`bm25_only` / `semantic_only` / `both`) gets
surfaced in the UI so users can see *why* a trial appeared. It's also
a great debugging signal: if every result says `semantic_only`, your
BM25 is broken; if every result says `both`, your two retrievers are
producing the same thing.

The wrapping class `HybridRetriever.search()` does four things in
order: (1) ask BM25 for top-200 with the filter dict applied, (2) ask
FAISS for top-200 (no filter — FAISS doesn't support filters), (3)
RRF-merge the two, (4) attach metadata from the BM25 response to the
top-`top_k` fused results.

**Why 200 candidates from each retriever?** Three reasons:
- Enough overlap for RRF to find consensus. If we took top-20 from
  each and they were disjoint, RRF would be vacuous.
- 200 is the typical candidate-pool size in production search systems
  before re-ranking.
- The cost is bounded — both retrievers serve top-200 in ~30 ms.

**Why attach metadata from BM25 hits, not from SQLite?** BM25 already
returned the metadata in its response, so no extra network round-trip.
For *semantic-only* results we do have to fetch from Elasticsearch
separately, but those are typically <30 of the top-200 fused.

### Step 5 — Wire the API and UI

**Files:** `src/TrialMine/api/{schemas.py, routes.py, app.py}`,
`src/TrialMine/ui/app.py`

Two small changes from week 1:

- `SearchRequest` gains a `method: Literal["bm25", "semantic",
  "hybrid"]` field (default `"hybrid"`).
- `TrialResult` gains three optional fields: `source`, `bm25_rank`,
  `semantic_rank`. These are populated only on the hybrid path and
  shown in the UI as `bm25_only / semantic_only / both` tags.

The route handler dispatches on `method` to one of three paths
(BM25 only, FAISS only, or `HybridRetriever.search`). The Streamlit
sidebar gets a "Search Method" radio button.

### Step 6 — Compare methods on 20 test queries

**File:** `scripts/compare_methods.py` (~310 lines)

This script proves the hybrid is doing something useful. We run a
hand-picked set of 20 queries through all three methods and compare
side by side. The queries deliberately mix three styles:

- **Clinical phrasing** — *"colorectal cancer with microsatellite
  instability"*, *"chronic lymphocytic leukemia ibrutinib"*. BM25
  should shine here (literal medical jargon).
- **Patient phrasing** — *"clinical trial for glioblastoma that has
  come back"*, *"pancreatic cancer that has spread to the liver"*.
  Semantic search should shine (vocabulary mismatch).
- **Misspellings** — *"melanomt"* instead of *"melanoma"*. Both
  should partially recover; semantic should do better because
  embeddings are robust to typos.

For each query, the script runs all three methods, prints top-3 side
by side, computes the **BM25 ∩ Semantic** overlap, saves full results
to `data/evaluation/method_comparison.csv`, and logs each method as a
separate MLflow run.

#### What we found

The most striking number: **BM25 ∩ Semantic top-3 overlap was 0/3 on
all 20 queries** with the off-the-shelf BioLinkBERT.

Let me say that again. Before fine-tuning, the two methods produced
completely **disjoint** top-3 lists for every single query.

That's a red flag. Two reasonable methods should agree on at least
some clearly-relevant trials. If they don't, one of them is doing
something weird. We dug in and found two issues with the off-the-shelf
embeddings:

1. **Anisotropy.** The cosine similarity range across the entire top
   1000 was only 0.047. That means every trial in the database
   embeds to nearly the same point. A query gets a "winner" not because
   it's actually similar but because of microscopic numerical noise.
2. **Hub trial problem.** Three trials monopolized 33% of all top-10
   slots across all queries. The embedding space had a few "hubs"
   that everything mapped close to.

Both are well-known pathologies of pre-trained transformer embeddings
on out-of-domain text. The fix is **fine-tuning** — that's exactly
what we do in week 3.

The week 2 deliverable is *honest documentation of the problem*, not a
solution. We logged the 0% overlap, the cosine spread, and the hub
trial finding. Then in week 3 we'll fix it and watch the overlap rise
to ~7%.

### Step 7 — MLflow tracking

`compare_methods.py` writes one MLflow run per method (bm25,
semantic, hybrid) with three kinds of records:

- **Tags** — `method=hybrid`, `stage=baseline`. Used to filter runs.
- **Params** — `top_k`, `num_queries`, `rrf_k`. The *inputs* of the
  experiment.
- **Metrics** — `avg_latency_ms`, `unique_trials_found`,
  `avg_bm25_semantic_overlap`. The *outputs*.
- **Artifacts** — the comparison CSV file.

Tracking URI: `sqlite:///mlflow.db` (a single SQLite file, same
philosophy as our trial DB). Open the UI with `make mlflow` →
[http://localhost:5001](http://localhost:5001).

Why bother? Because in week 3 we'll re-run this comparison with
fine-tuned embeddings. We want to say *exactly* "NDCG@10 went from
0.534 to 0.796 between *this run* and *that run*." Eyeballing logs
doesn't scale.

### Step 8 — The metrics module (a placeholder for week 3)

**File:** `src/TrialMine/evaluation/metrics.py` (~150 lines)

We don't have relevance labels yet — those come in week 3 via Claude
Haiku. But we write the metric functions now so they're ready when
labels exist:

| Function | What it answers |
|---|---|
| `precision_at_k` | Of the top-k results, what fraction are relevant? |
| `recall_at_k` | Of all relevant docs, what fraction did we find in top-k? |
| `ndcg_at_k` | Are highly relevant docs at the top? (Graded, 0–1.) |
| `mrr` | Where does the *first* relevant doc appear in the ranking? |

We'll wire these up in week 3 with the labeled set, and explain the
NDCG formula there.

---

## Part 6 — Why we made the choices we did

### Decision 6 — BioLinkBERT over PubMedBERT or general models

**Context.** We need a biomedical sentence encoder.

**Options.**
- `all-MiniLM-L6-v2`: 384 dim, general English, very fast.
- `BioBERT`: BERT pre-trained on PubMed.
- `PubMedBERT`: BERT trained from scratch on PubMed.
- `BioLinkBERT`: like PubMedBERT but with citation links as training
  signal.

**We chose BioLinkBERT.** Three reasons:

1. **Domain match.** Trained on PubMed → understands biomedical
   vocabulary natively.
2. **Citation-link pre-training.** When PubMedBERT sees
   *"Trial NCT04 ... pembrolizumab ..."*, it has no signal about which
   other trials are similar. BioLinkBERT was trained to predict
   *citation relationships* between papers — exactly the same
   "are these two pieces of biomedical text related?" question we
   need to answer.
3. **768 dimensions.** Captures more nuance than MiniLM's 384.

**Trade-off.** Slower than MiniLM (3× the inference cost). But for
140K documents, the embedding step is a one-time cost — we run it
once and then never again.

**Interview answer.** *"BioLinkBERT because the citation-link
pre-training objective most closely matches our task: 'are these two
pieces of biomedical text about related concepts?'"*

### Decision 7 — Reciprocal Rank Fusion over linear score combination

**Context.** Two retrievers with different score scales. Need to merge.

**Options.**
- **Linear combo**: `final = α × bm25 + β × semantic` after
  normalization.
- **Learned merge**: train a model to predict relevance from both
  scores. (We do this in week 4 — the LightGBM blender.)
- **RRF**: rank-based fusion.

**We chose RRF as the simplest viable starting point.** Why?

- BM25 scores are unbounded and corpus-dependent (`5–25` is typical
  for our corpus). Cosine similarities are in `[0, 1]`. Normalizing
  them so they're "comparable" requires assumptions about distribution.
- RRF skips the whole problem: only ranks matter. A document at rank 1
  in BM25 contributes the same `1/61` regardless of whether its raw
  BM25 score was 8 or 80.
- No hyperparameters to tune (k=60 is the literature default and it
  works).

**Trade-off.** Treats both retrievers as equally trustworthy. If BM25
is consistently better on certain query types, RRF leaves that signal
on the table. The week 4 LightGBM blender uses both ranks *and* raw
scores as features and learns the weighting.

**Interview answer.** *"RRF is score-scale independent — no
normalization required. Cormack et al. 2009 showed it's competitive
with learned merges when training data is scarce. We'd switch to a
learned blender once we have labels, which is exactly what we do in
week 4."*

### Decision 8 — `IndexFlatIP` over approximate FAISS indexes

**Context.** FAISS offers many index types. Some are exact (slow but
correct), some are approximate (fast but lossy).

**Options for 140K vectors:**
- `IndexFlatIP`: exact. Searches all 140K vectors. ~30 ms.
- `IndexIVFFlat`: clusters vectors into Voronoi cells; searches a few
  cells. Approximate. ~3 ms but recall <100%.
- `IndexIVFPQ`: cells + product quantization. Even faster, much lower
  recall.

**We chose `IndexFlatIP`.** At 140K vectors, exact search takes ~30 ms.
That's well within our latency budget. Approximate indexes add
complexity (cluster training, recall tuning) without measurable benefit.

**At scale.** Switch to `IndexIVFFlat` at ~1M vectors, `IndexIVFPQ` at
~100M. The library has the same API — only the construction call
changes.

### Decision 9 — Equal BM25/semantic weight in RRF

**Context.** RRF treats both lists symmetrically. Should we?

**The honest answer: we don't know yet.**

Without labeled data, we can't tell whether BM25 or semantic is "more
right" on average. Equal weighting is the *zero-information prior*. If
we biased toward one method without evidence, we'd just be propagating
our hunch as a hyperparameter.

**Plan.** In week 4, train a LightGBM blender on labeled data. The
feature importance will tell us which retriever's signal carries more
weight. (Spoiler: cross-encoder ends up dominating; raw BM25 and
semantic each contribute about the same amount.)

---

## Part 7 — File map (where each thing lives)

| File | New in week 2 | Purpose | LOC |
|---|---|---|---|
| `src/TrialMine/models/embeddings.py` | new | BioLinkBERT wrapper | ~150 |
| `src/TrialMine/retrieval/semantic.py` | new | FAISS wrapper (build + search + persist) | ~140 |
| `src/TrialMine/retrieval/hybrid.py` | new | RRF fusion + HybridRetriever | ~325 (week 2 portion ~180) |
| `src/TrialMine/evaluation/metrics.py` | new | precision/recall/NDCG/MRR (used in week 3) | ~150 |
| `scripts/build_index.py` | extended | now builds FAISS too | ~260 |
| `scripts/compare_methods.py` | new | 20-query side-by-side + MLflow | ~310 |
| `src/TrialMine/api/{schemas, routes}.py` | extended | add `method` flag, `source` tag | +50 LOC |
| `src/TrialMine/ui/app.py` | extended | method radio, source tags | +30 LOC |

The data lives in `data/`:
- `data/trial_embeddings.faiss` — FAISS binary index (~412 MB).
- `data/trial_embeddings.json` — NCT ID list (small).
- `data/evaluation/method_comparison.csv` — comparison results.

`mlflow.db` and `mlruns/` track the experiments. Open `make mlflow`.

---

## Part 8 — Interview prep

### 8.1 The big design question

> *"Walk me through how you'd add semantic search to a keyword-based
> system."*

A good answer covers:

1. **Identify the failure mode.** Vocabulary mismatch — patient
   language vs medical jargon.
2. **Pick an embedding model.** Domain-matched (biomedical),
   multilingual if needed, 384–768 dim. Justify with the
   pre-training objective.
3. **Choose a vector store.** FAISS for offline/single-machine,
   pgvector for "I need transactions," Pinecone/Weaviate/Qdrant for
   managed production.
4. **Handle the index type tradeoff.** Exact (FlatIP) under ~1M;
   approximate (IVF, HNSW) above.
5. **Don't replace BM25 — fuse.** RRF is the simplest viable merge.
   Linear normalized combo and learned blenders are alternatives.
6. **Measure.** Top-K overlap with the existing system is a free
   sanity check. Real metrics require labeled data — that's a
   separate evaluation pipeline.

### 8.2 Specific questions

**"Why pool with mean instead of [CLS] token?"**
Mean pooling averages information from all tokens. The [CLS] token,
in BERT models *not* fine-tuned for classification, is often
under-trained — the gradient signal during pre-training mostly flowed
through the masked-LM objective on other tokens. For sentence-level
tasks, mean pooling is the safer default.

**"What does `normalize_L2` do and why is it required?"**
It scales each vector to unit length (sum of squares = 1). After
normalization, the inner product of two vectors equals the cosine of
the angle between them. Skipping normalization means the dot product
captures both *direction* (good signal) and *magnitude* (noise from
how long the input text was).

**"Your top-k is 200, then you fuse, then you take 50, then re-rank
50, then return 20. Why so many stages?"**
Each stage trades latency for quality. The retrievers are fast and
high-recall (200 candidates ensures we catch the right answer
*somewhere* in the pool). The cross-encoder (week 4) is slow and
high-precision — running it on 200 docs would be too slow, but
running it on 50 brings precision near-perfect. The blender then uses
its richer feature set to do the final tie-breaking.

**"How do you handle queries with no semantic match (e.g., NCT IDs)?"**
The NCT ID is in the BM25 keyword index. A user pasting `NCT04802759`
hits BM25's exact-match path immediately and doesn't need semantic.
RRF gives equal weight regardless, so the BM25 hit is rank 1 from one
side and the result is correct.

**"What about non-English queries?"**
Doesn't work today. BioLinkBERT is English-only. For multilingual
support we'd swap in `paraphrase-multilingual-mpnet` for the embedder
and add language detection at the query layer. The trial corpus is
already English-only (CT.gov is a U.S. registry).

**"How would you A/B test two embedding models?"**
Maintain two FAISS indexes. Bucket users deterministically (week 8's
`ABTestRouter`). Log clicks or user feedback. Compute lift. If the
new model wins, promote it. The interface that makes this easy is
`MODEL_ALIASES` in `build_index.py` — you build a separate index per
model and the rest of the system points at whichever one is current.

### 8.3 Code-review questions

1. **Why are the FAISS index and JSON mapping saved as separate files?**
   FAISS is a C++ library; it stores only the vector data. We need a
   parallel structure to map from FAISS row index to NCT ID. JSON is
   the simplest interchange format for that lookup table.

2. **Why is `prepare_trial_text` in the `TrialEmbedder` class instead
   of in `Trial`?**
   Because the text shape is *coupled to the embedder*: which fields
   to include, the [SEP] token, the truncation length — those are
   embedder-specific decisions. If we ever swap to a model with a
   different separator, the change should be co-located with the
   embedder.

3. **Why do `prepare_trial_text` (in the model) and `build_semantic_index`
   (in the script) duplicate the concatenation logic?**
   It's a real bug — they have to stay in sync, and they don't share a
   helper. The script's version exists because it streams chunks
   directly from SQLite (no full `Trial` objects), and we duplicated
   rather than refactor. A future cleanup would extract the join into
   a utility function.

4. **Why is `bm25_meta` populated only from BM25 results, not from
   ES queries on every fused doc?**
   Because BM25 results already carry the metadata. For
   `semantic_only` results we *do* fetch from ES separately. That
   way we make at most ~30 ES round trips per query (only for
   semantic-only docs in the top 50), instead of 50 (for every
   fused doc).

### 8.4 Tradeoff questions

**"Off-the-shelf embeddings are anisotropic. Why ship anyway?"**
We documented the failure honestly. The week 2 deliverable is a
*hybrid retrieval pipeline*, not "best-in-class results." Shipping the
broken-but-honest pipeline lets us layer the fix (fine-tuning,
week 3) on top with a clean A/B comparison: same code, swap the
model.

**"Why no approximate FAISS? Latency seems fine without it."**
Right — at 140K vectors and 30 ms exact search, approximation gives
us nothing. The decision rule is: when does latency become the
bottleneck? At ~1M vectors, exact search hits 200 ms and starts
dominating the budget. Then we'd switch to IVF.

**"Why not use a vector DB like Pinecone or Qdrant?"**
For a single-machine project, FAISS is simpler. Pinecone wins when
you want managed hosting, dynamic updates (we re-build the index
offline), or row-level filters integrated with vector search. Both
are valid; FAISS is the right tradeoff for week 2.

---

## Part 9 — How to run everything

```bash
# Pre-req from week 1:
docker compose up -d elasticsearch
make download                              # 75 min
python scripts/build_index.py --skip-semantic  # 80 sec for BM25

# New in week 2: build FAISS embeddings
python scripts/build_index.py --skip-bm25 --model off-the-shelf
# ~32 min on CPU, 8 min on a Colab T4

# Compare methods on 20 queries + log to MLflow
python scripts/compare_methods.py
# Open the comparison: data/evaluation/method_comparison.csv
# Open the MLflow UI: make mlflow, then http://localhost:5001

# Try the API + UI
make serve     # one terminal
make ui        # another terminal
# Open http://localhost:8501, change the method radio button
```

In the UI, run these "vocabulary mismatch" queries — pick `Hybrid`
in the sidebar:

- *"glioblastoma that came back"* → should find recurrent GBM trials
- *"lung cancer that has spread to bone"* → should find metastatic NSCLC trials
- *"my mom has stomach cancer"* → should find gastric cancer trials

If the off-the-shelf BioLinkBERT does poorly here, that's *expected*.
Week 3 fixes it with fine-tuning.

---

## Part 10 — What's next (week 3 preview)

Off-the-shelf BioLinkBERT is anisotropic — the cosine spread is too
narrow to discriminate. Week 3 fine-tunes it on **clinical-trial
specific data** to fix that. We'll:

1. Generate ~730K (query, positive trial, hard negative) triplets from
   three sources:
   - 242K metadata-derived pairs (free).
   - 1,500 synthetic patient queries via Claude Haiku (~$2).
   - 730K hard-negative triplets (same condition, different intervention).
2. Fine-tune BioLinkBERT with `MultipleNegativesRankingLoss` on a Colab
   A100 (~5 hours).
3. Build a *labeled evaluation dataset* using Claude Haiku as the
   judge (LLM-as-judge): ~990 (query, trial) pairs each rated 0-3
   for relevance.
4. Compute NDCG@5/10 and MRR for both the off-the-shelf and fine-tuned
   models, log to MLflow.

Spoiler: NDCG@10 jumps from 0.534 to 0.796 — a 49% improvement. The
BM25 ∩ Semantic overlap rises from 0% to 7%. The hub trial problem
disappears.

---

## Part 11 — End of week 2 checklist

- [x] BioLinkBERT embedder wrapper (with the explicit-modules fallback)
- [x] FAISS `IndexFlatIP` wrapper with save/load + JSON mapping
- [x] All 140,723 trials embedded → `data/trial_embeddings.faiss` (412 MB)
- [x] `HybridRetriever` with RRF (k=60) over 200+200 candidates
- [x] `compare_methods.py` runs all three methods on 20 queries
- [x] MLflow tracking (`trialmind-retrieval` experiment, 3 runs)
- [x] API supports `method = bm25 | semantic | hybrid`
- [x] Streamlit UI shows source tags (`bm25_only / semantic_only / both`)
- [x] Documented anisotropy + hub-trial failure (week 3 will fix it)
- [x] Design decisions 6-9 in `docs/design-decisions.md`

That's the full week. If you can explain *what an embedding is*, *why
RRF works without normalization*, and *what the hub-trial problem
means*, you've internalized the work.
