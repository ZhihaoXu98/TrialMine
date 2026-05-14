# Week 3 — Fine-Tuned Embeddings + LLM-Judged Evaluation (Student Edition)

A textbook walkthrough of week 3. By the end you should be able to (a)
explain the difference between *training* and *fine-tuning*, (b) describe
contrastive learning in plain English, (c) read our training scripts and
recognize each piece, and (d) defend our LLM-as-judge evaluation strategy
in an interview.

This is the first week with real ML training. If you've never trained a
neural network before, that's fine — we'll explain each part as it comes
up.

---

## Part 1 — Why are we doing this?

Recall from week 2 the disaster:

> **BM25 ∩ Semantic top-3 overlap was 0/3 on every single one of 20
> queries.** The cosine similarity range across the entire top 1000 was
> 0.047, meaning every trial in the database mapped to nearly the same
> point in vector space. Three "hub" trials monopolized 33% of all
> top-10 slots.

This is called **anisotropy** — pre-trained transformer embeddings are
notorious for this when used out-of-the-box on out-of-domain text. The
model knows biomedical English (it was trained on PubMed), but it has
never been told *"these two things should be near each other and these
two should be far apart."* The pretraining objective was masked-language
modeling (predict missing words), not similarity.

**Fine-tuning** fixes this. We take the same BioLinkBERT model and
continue training it on a *new* objective: contrastive learning. We
show it pairs of (query, matching trial) and pairs of (query,
not-matching trial), and update the model's weights so the matches
end up close in embedding space and the mismatches end up far.

After fine-tuning, the embedding space "spreads out" along the
dimensions that matter for our task. The 0.047 cosine spread becomes
0.10 in the top 5. The hub trials disappear. Real biomedical
distinctions emerge.

But fine-tuning needs **training data**. Lots of it. With realistic
patient queries on one side and matching trials on the other. That
data does not exist publicly. So before any training happens, we
have to **generate it ourselves**.

The week 3 plan:

1. **Generate** ~730K training triplets (query, positive, negative).
2. **Fine-tune** BioLinkBERT on those triplets.
3. **Build a labeled evaluation set** using Claude Haiku as the judge.
4. **Measure** NDCG@5/NDCG@10/MRR before and after fine-tuning.

---

## Part 2 — Vocabulary you'll meet

### ML / training terms

- **Pre-training** — the long initial training of a model on a giant
  corpus (PubMed, the Web, Wikipedia). Done by HuggingFace /
  Stanford, not us. Costs millions of dollars.
- **Fine-tuning** — taking a pre-trained model and continuing training
  on a smaller, task-specific dataset. Costs hours, not weeks.
- **Bi-encoder** — encodes the query and the trial *separately*; we
  compare with cosine similarity. Fast: trial embeddings are
  pre-computed once. We use this for retrieval.
- **Cross-encoder** — encodes the query and trial *jointly* and
  outputs one score. More accurate but must run per (query, trial)
  pair. Used in week 4 for re-ranking — not this week.
- **Contrastive learning** — training that pulls similar things close
  in vector space and pushes dissimilar things apart.
- **Triplet** — an `(anchor, positive, negative)` tuple. Anchor and
  positive should end up close; anchor and negative should end up far.
- **Hard negative** — a triplet's negative that *looks similar* to the
  positive but is actually wrong. ("breast cancer surgery" vs "breast
  cancer immunotherapy" — same disease, different treatment.) Hard
  negatives teach the model fine-grained discrimination; "easy"
  negatives ("breast cancer" vs "bicycle repair") don't.
- **MultipleNegativesRankingLoss (MNRL)** — our contrastive loss. In
  a batch of N examples, every other positive in the same batch is
  used as an "in-batch negative" — N examples give ~N² training
  signals for free.
- **InformationRetrievalEvaluator** — a sentence-transformers utility
  that, during training, evaluates Recall@k / MRR / NDCG@k on a
  held-out set. Lets you watch the *metric you care about*, not just
  the training loss.
- **NDCG@k** — *Normalized Discounted Cumulative Gain at rank k.* A
  ranking-quality score in [0, 1]: "are the most relevant items at
  the top?" Formula and worked example in Part 4.
- **MRR** — Mean Reciprocal Rank: 1 / (position of the first
  relevant result).
- **LLM-as-judge** — using a large language model to *label* data
  instead of paying humans. Validated by ACL/EMNLP papers since 2023.
- **Pooled labeling** — when comparing several search methods, label
  the *union* of their top-k results. Labeling only one method's
  results would unfairly penalize the others' unique discoveries.

### Medical terms

- **Pembrolizumab** — an immunotherapy drug (a PD-1 inhibitor) used
  in many cancers. Common positive example in our training data.
- **Trastuzumab** — a targeted-therapy drug for HER2-positive breast
  and gastric cancer. Used as a hard-negative example
  (different drug, same disease).
- **HER2-positive / hormone-receptor-positive** — breast-cancer
  subtypes defined by which proteins the tumor cells display.
- **Recurrent / "came back"** — cancer that returned after treatment.
- **Metastasized / "spread"** — cancer that moved to a different
  organ.
- **Neoadjuvant** — given *before* the main treatment (e.g.,
  chemotherapy before surgery, to shrink the tumor first).

---

## Part 3 — A 30-second tour of contrastive learning

The intuition. You have a bunch of (anchor, positive) pairs. You want
the model to embed each anchor close to its positive.

Naive attempt: minimize `||embed(anchor) − embed(positive)||²` for
every pair. **Doesn't work.** The model collapses all embeddings to a
single point — distance zero everywhere, loss = 0, model useless.

Fix: also *maximize* the distance to negatives. Now the model has
incentive to pull positives close *and* push negatives away. The
embedding space stays spread out.

The simplest contrastive loss is **triplet loss**:

```
loss = max(0, margin + sim(anchor, negative) − sim(anchor, positive))
```

In English: "the positive should be at least `margin` closer than the
negative; if not, pay a penalty."

We use a smarter version, **MultipleNegativesRankingLoss** (MNRL).
Instead of asking us to supply explicit negatives for every example,
it treats *every other positive in the same batch* as a negative for
free. Batch size 32 → each example gets 31 in-batch negatives → ~32×
the training signal per gradient step.

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss
loss = MultipleNegativesRankingLoss(model, scale=20.0)
```

The training loop just minimizes this via gradient descent.

---

## Part 4 — A 30-second tour of NDCG

We need a metric that says: *"my top-10 search results contain the
right answers, and the most relevant ones are at the top."*

**NDCG** (Normalized Discounted Cumulative Gain) does exactly this.
With graded relevance scores (0 = irrelevant ... 3 = highly relevant),
the formula is:

```
DCG@K  = Σ over i=1..K of  (2^rel_i − 1) / log2(i + 1)
NDCG@K = DCG@K / IDCG@K           # IDCG = DCG of the perfect ranking
```

In English:
- `(2^rel − 1)` rewards higher relevance scores exponentially
  (a "3" is worth 7 points, a "2" is worth 3, a "1" is worth 1).
- `log2(i + 1)` penalizes low positions (rank 1 → divide by 1, rank
  10 → divide by 3.46). Top-of-list relevance counts more.
- Dividing by IDCG (the ideal ranking's DCG) makes the score 0–1, so
  it's comparable across queries with different numbers of relevant
  results.

**Worked example.** Top 4 results have relevance `[3, 1, 2, 0]` and the
ideal ordering would be `[3, 2, 1, 0]`:

```
Our DCG  = 7/log2(2) + 1/log2(3) + 3/log2(4) + 0/log2(5)
         = 7/1.000   + 1/1.585   + 3/2.000   + 0
         = 7.000     + 0.631     + 1.500     + 0
         = 9.131

Ideal    = 7/log2(2) + 3/log2(3) + 1/log2(4) + 0
         = 7.000     + 1.893     + 0.500     + 0  =  9.393

NDCG     = 9.131 / 9.393 = 0.972     # close to perfect
```

We use NDCG@5 and NDCG@10 throughout the rest of the project.

---

## Part 5 — What we built, step by step

### Step 1 — Generate training data

**File:** `scripts/generate_training_data.py` (~700 lines)
**Config:** `configs/training_data.yaml`

We need ~500K+ (query, positive, negative) triplets to fine-tune
properly. They come from three sources.

#### Source 1 — Metadata-derived pairs (free)

For each trial, we already have structured metadata: title,
conditions list, intervention names, phase. We can manufacture
queries that *should* match that trial:

```python
trial = Trial(
    title="A Phase 3 Study of Pembrolizumab for Recurrent Gastric Cancer",
    conditions=["Gastric Adenocarcinoma", "Recurrent Cancer"],
    interventions=["Pembrolizumab"],
    phase="Phase 3",
)

# Pairs we generate from this one trial:
pairs = [
    ("Gastric Adenocarcinoma",                                   trial_text),
    ("Recurrent Cancer",                                          trial_text),
    ("Pembrolizumab",                                             trial_text),
    ("Pembrolizumab for Gastric Adenocarcinoma",                  trial_text),
    ("phase 3 gastric adenocarcinoma trial",                      trial_text),
]
```

Where `trial_text = "Title [SEP] Condition1 Condition2 [SEP] BriefSummary"`,
the same format the embedder uses. **The training data must be in the
same shape as the inference data**, otherwise the model learns to
optimize for one and fails at the other.

This pipeline is essentially free (no API calls). It produces ~242K
pairs from our 140K trials.

**Subtle choice: stratified sampling.** Not all cancer types are
equally represented in the corpus. Breast cancer has ~12,000 trials;
mesothelioma has ~400. If we trained on pairs proportional to corpus
size, the model would be 30× better at breast cancer than mesothelioma
— which is the opposite of what a search system should do for rare
cancers.

So we cap at 2,000 trials per cancer group:

```yaml
# configs/training_data.yaml
sampling:
  max_trials_per_cancer_group: 2000
  random_seed: 42

cancer_types:
  breast: ["breast"]
  lung: ["lung", "NSCLC", "SCLC", "non-small cell", "small cell lung"]
  prostate: ["prostate"]
  colorectal: ["colorectal", "colon", "rectal"]
  melanoma: ["melanoma"]
  leukemia: ["leukemia", "leukaemia"]
  # ... 17 more groups (23 total)
```

A trial's group is decided by simple keyword matching on its
conditions text. Imperfect — a trial about "gastric cancer with liver
metastasis" gets classified as `liver` (whichever keyword appears
first in the dict iteration order). For a coarser stratification this
is fine.

#### Source 2 — Synthetic patient queries (~$2 in API costs)

The metadata-derived queries above are clean and clinical. They look
nothing like what a real patient types. So we use Claude Haiku to
*write patient-language queries* for a sample of trials.

The prompt:

```
You generate realistic patient search queries for clinical trials.

Write a realistic 1-2 sentence search query that a PATIENT (not a doctor)
might type when looking for this trial. Use simple patient language.
DO NOT use medical jargon or acronyms.

Good examples:
- 'lung cancer treatment options after chemo stopped working'
- 'is there a trial for breast cancer near Chicago'
- 'my colon cancer came back, what trials can I try'

Trial: {title}
Conditions: {conditions}
Eligibility (first 300 chars): {eligibility}

Respond with ONLY the patient query. Nothing else.
```

We sample 1,500 trials (stratified across cancer groups) and call
Claude Haiku once per trial. Cost: ~$2.

Three engineering details to know about:

- **Rate limit.** 5 requests/second (`time.sleep(0.2)` between calls)
  to stay below Anthropic's tier limits.
- **Checkpoint every 100 calls.** Each generated query is appended to
  a JSONL checkpoint file and `flush()`-ed every 100 calls. If the
  script crashes at call 1,200, we've persisted ~1,100. Restart with
  `--resume` to skip already-done NCT IDs.
- **Strip quotes from Claude's response.** It sometimes returns
  `'lung cancer treatment...'` wrapped in quotes. We `.strip("'\"")`
  before saving so the training text is clean.

#### Source 3 — Hard negatives

For every (query, positive) pair from sources 1 and 2 we need a
*negative* — a trial that *looks* similar but is actually wrong.

The recipe in plain English:

1. Pre-compute a `keyword → list of NCT IDs` index over every trial's
   conditions text. (One pass, ~1 sec.)
2. For each positive pair, find every other trial sharing at least
   one condition keyword. Those are the candidates.
3. Split candidates into **hard** (different intervention drug) and
   **easy** (same intervention drug). Prefer hard.
4. Sample 3 negatives per positive — hard first, fall back to easy if
   we don't have enough.

**Why prefer different-intervention candidates?** Same-disease,
different-treatment is the textbook hard case in medical retrieval.
The model needs to know that *"pembrolizumab for breast cancer"* and
*"surgery for breast cancer"* are NOT interchangeable, even though
they share most of their words. Easy negatives ("breast cancer" vs
"liver cancer") teach a useful distinction the model already knows
from BioLinkBERT pre-training; hard negatives teach the new one.

#### Output: train + val splits

We then split the triplets 80/20 train/val *by trial NCT ID* (not by
triplet). All triplets that share a positive trial go into the same
split. Otherwise, the val set could contain triplets whose positive
trial appeared on the train side — leakage.

Final stats (after running the full pipeline):

| File | Triplets | Size |
|---|---|---|
| `data/training/train_pairs.jsonl` | 586K | 1.0 GB |
| `data/training/val_pairs.jsonl`   | 145K | 260 MB |
| `data/training/synthetic_queries.jsonl` | 1,500 | 1.5 MB |

### Step 2 — The fine-tuning script

**File:** `scripts/finetune_embeddings.py` (~600 lines)
**Config:** `configs/training/embeddings.yaml`

The script loads the model, the data, the loss, and an evaluator,
then hands them to the sentence-transformers `Trainer`.

#### Detect device + auto-reduce on CPU

```python
def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = detect_device()
if device == "cpu":
    config["training"]["epochs"] = 1     # 3 epochs would take ~12 hours on CPU
```

CPU is unusable for full training (12+ hours). We auto-reduce to 1
epoch as a fallback "smoke test" so the script at least runs end-to-end
on a developer laptop.

#### Load data, drop empty negatives, rename columns

```python
dataset = load_dataset("json", data_files={"train": ..., "val": ...})
train_ds = dataset["train"].filter(lambda x: x["negative"] and len(x["negative"].strip()) > 0)
val_ds = dataset["val"].filter(lambda x: x["negative"] and len(x["negative"].strip()) > 0)

# sentence-transformers expects "anchor", "positive", "negative" columns
train_ds = train_ds.rename_column("query", "anchor")
val_ds = val_ds.rename_column("query", "anchor")
train_ds = train_ds.select_columns(["anchor", "positive", "negative"])
val_ds = val_ds.select_columns(["anchor", "positive", "negative"])
```

`load_dataset("json", ...)` is HuggingFace's dataset loader — JSONL
files load as a `Dataset` object you can `filter`, `map`, `shuffle`,
etc.

#### Build the InformationRetrievalEvaluator

This is the *online* evaluator the trainer runs every 1000 steps. For
each held-out query, it asks the model: *"out of 5,000 candidate
trials, how high do you rank the correct one?"* Outputs Recall@10,
MRR, NDCG@10. We use NDCG@10 as the "best model" criterion.

`build_evaluator()` constructs three dicts that the evaluator needs:
- `queries: {qid: query_text}`
- `corpus: {doc_id: trial_text}`
- `relevant_docs: {qid: {doc_id, ...}}` — for each query, the set of
  trials we know to be correct.

We subsample the val set to `max_samples=5000`. Why? Because the
evaluator *re-encodes the entire corpus on every call*. With all 145K
val examples it would take ~10 min/call; with 5K it's ~30 sec. Eval
runs every 1000 training steps, so this is a real cost.

The point of having an evaluator at all is to watch the *retrieval
metric* (NDCG@10), not just the training loss. The loss can keep
dropping while the model overfits to training-set artifacts and
actually gets *worse* at retrieval. NDCG on held-out queries tells
the truth.

#### Wire it all together

```python
training_args = SentenceTransformerTrainingArguments(
    output_dir="models/embeddings/fine-tuned",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,                       # mixed precision — halves memory
    eval_steps=1000, save_steps=1000, save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="val-retrieval_cosine_ndcg@10",
    report_to="mlflow",
)

trainer = SentenceTransformerTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    loss=MultipleNegativesRankingLoss(model=model, scale=20.0),
    evaluator=build_evaluator(...),
)
trainer.train()
trainer.save_model("models/embeddings/fine-tuned")
```

The hyperparameters worth understanding:

- **`learning_rate=2e-5`** — standard for transformer fine-tuning.
  Too high → "catastrophic forgetting" (the model loses what it
  learned in pre-training). Too low → no learning.
- **`warmup_ratio=0.1`** — the learning rate ramps up from 0 over the
  first 10% of training steps. Without warmup, early gradients can be
  huge and destabilize the model.
- **`fp16=True`** — 16-bit "mixed precision" training. Halves GPU
  memory usage with minimal accuracy loss; required to fit
  `batch_size=32` on a single A100.
- **`scale=20.0`** in MNRL — a temperature parameter in the softmax
  used inside the loss. 20 is the literature default.
- **`metric_for_best_model="...ndcg@10"`** — save the checkpoint with
  the best NDCG@10 on the validation set, not the most-recent
  checkpoint. The loss can keep going down even after the model has
  started overfitting; NDCG on held-out data tells the truth.

#### Saving metadata

After training, we write `models/embeddings/fine-tuned/metadata.json`
with four sections:

- `training_date` — ISO timestamp of the run.
- `base_model` + `dataset_size` — what we started with.
- `hyperparameters` — every learning rate, batch size, loss, etc.
- `eval_metrics` — final NDCG@10 / MRR / Recall on the val set.

This is the **audit trail**. Every model on disk should be able to
answer *"which version was this, trained on what data, with what
config, and how did it score?"* By week 8, this schema becomes a
load-bearing contract enforced by the CI quality gate.

#### Where it actually ran

We don't run this on a laptop. The training script was uploaded to
Colab and executed on an A100 GPU:

```
notebooks/finetune_biolinkbert.ipynb
```

Total training time: ~5 hours on A100, processing 586K triplets over
3 epochs. The output (~430 MB of model weights + tokenizer files +
metadata.json) downloads back to `models/embeddings/fine-tuned/`.

### Step 3 — Rebuild FAISS with the fine-tuned model

**File:** `scripts/build_index.py` (extended again)

The fine-tuned model is just a different checkpoint path, so the
build script gets a `--model {off-the-shelf|fine-tuned}` flag and a
`MODEL_ALIASES` dict that picks the right paths:

```bash
python scripts/build_index.py --skip-bm25 --model fine-tuned
# → reads models/embeddings/fine-tuned/
# → writes data/faiss_finetuned.index
```

We keep **both** indexes side by side (`faiss_offshelf.index` and
`faiss_finetuned.index`) so the comparison in Step 5 can swap models
without re-indexing.

### Step 4 — Build the evaluation dataset (LLM-as-judge)

**File:** `scripts/build_eval_dataset.py` (~410 lines)

Now the question: *did fine-tuning actually help?* Without labeled
data we can only do "eyeball checks" on a few queries. We need real
metrics. So we generate labels with Claude Haiku.

#### The protocol: pooled labeling

For each of the 20 test queries from week 2, we run hybrid search
with *both* embedding models (off-the-shelf + fine-tuned) and pool
the top-30 unique trials from each. That gives us ~50 unique trials
per query (some appear in both lists, those get tagged
`retrieved_by="both"`).

**Why pool?** Suppose we labeled *only* the fine-tuned model's top-30.
Trials uniquely surfaced by the off-the-shelf model would never get a
label and would default to relevance 0 in the metric. The off-the-shelf
model would then look terrible *not* because its discoveries were bad,
but because they were unmeasured. Pooling guarantees both methods get
a fair shake.

#### The labeling prompt

```
Rate the relevance of this clinical trial to this patient's search query.

Patient query: {query}

Trial title: {trial_title}
Trial conditions: {conditions}
Trial phase: {phase}
Trial status: {status}
Trial eligibility (first 500 chars): {eligibility}

Rate on a scale of 0-3:
0 = Completely irrelevant — wrong cancer type entirely
1 = Marginally relevant — same general cancer area but wrong specifics
2 = Relevant — matches condition, patient could potentially be eligible
3 = Highly relevant — strong match for condition, treatment type, and eligibility

Respond with ONLY a JSON object: {"score": X, "reason": "brief 1-sentence explanation"}
```

Two design choices in this prompt:

- **Graded relevance (0-3) instead of binary.** Search ranking is a
  ranking problem, not a classification problem. NDCG@k uses graded
  scores and weighs higher relevance more. Binary labels (relevant /
  not) lose this signal.
- **JSON output with reason.** The reason field is *not used in
  metrics* but it's invaluable for spot-checking. Read 20 random
  reasons; if they mention real trial fields, the labels are sane.

We call the API:

```python
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=150,
    messages=[{"role": "user", "content": prompt}],
)
text = response.content[0].text.strip()
# Robust JSON parsing — Claude sometimes wraps in markdown ```json ... ```
if text.startswith("```"):
    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
result = json.loads(text)
score = int(result["score"])
reason = result.get("reason", "")
```

For each (query, trial) pair, save one line in
`data/evaluation/labeled_queries.jsonl`:

```json
{"query_id": 0, "query": "breast cancer hormone receptor positive phase 3",
 "nct_id": "NCT04802759", "trial_title": "...",
 "relevance": 3, "reason": "Strong match: phase 3 HR+ breast cancer trial",
 "labeler": "claude-haiku", "retrieved_by": "both"}
```

#### Resume + sanity check

Same checkpoint pattern as before. Each labeled pair is appended
immediately so a crash at pair 500 of 990 is recoverable. Before
running on all 990 pairs, the script supports `--limit 10` for a
preview — you read 10 results yourself and check they're reasonable
before spending real money.

We label 990 pairs. Cost: about $0.50.

Final score distribution:
- 0 (irrelevant): 22.7%
- 1 (marginal): 17.4%
- 2 (relevant): 13.9%
- 3 (highly relevant): 46.0%

The bimodal shape (22.7% are 0, 46% are 3, fewer in the middle) tells
us Claude Haiku is *willing* to commit — it doesn't hedge into the
middle. Good sign for label quality.

### Step 5 — Compare embeddings

**File:** `scripts/compare_embeddings.py` (~270 lines)

For each embedding model (off-the-shelf, fine-tuned), the script:
1. Loads the corresponding FAISS index.
2. Runs hybrid search on each of the 20 labeled queries (top-10).
3. Computes NDCG@5, NDCG@10, MRR per query using `metrics.py` from
   week 2 against the 990 labels.
4. Averages across queries; logs as one MLflow run per model.

#### Results

| Metric | Off-the-shelf | Fine-tuned | Difference |
|---|---|---|---|
| NDCG@5  | 0.577 | 0.816 | **+41.4%** |
| NDCG@10 | 0.534 | 0.796 | **+49.1%** |
| MRR     | 0.917 | 0.917 | 0% (BM25 dominates first-result quality in both) |

Fine-tuned wins on **19 of 20 queries**. The single loss
(*"sarcoma clinical trials for young adults"*) is interesting: sarcoma
trials are a very long tail in our corpus, and the fine-tuning
sample-size cap of 2,000 may not have given enough signal for that
group.

Two qualitative checks beyond the numbers:

- **Cosine spread in top 5.** Off-the-shelf: 0.047. Fine-tuned: 0.10.
  More than 2× the dynamic range — exactly the dispersion we wanted.
- **Hub trials.** Off-the-shelf: three trials cover 33% of all top-10
  slots across queries. Fine-tuned: every query returns distinct
  trials. The hub problem is gone.

The 41–49% NDCG improvement is the headline number — but for an
interviewer the more important point is *the cosine spread went from
0.047 to 0.10 and the hub trial problem disappeared*. Those numbers
explain *why* NDCG improved.

---

## Part 6 — Why we made the choices we did

### Decision 10 — Three-source training data (metadata + LLM-synthetic + hard negatives)

**Context.** Need ~500K+ triplets for fine-tuning. No real labeled
patient queries exist publicly.

**Three-source recipe.**
- **Metadata pairs (242K, free).** Volume.
- **Synthetic patient queries (1,500, ~$2).** Realism — what real
  patients actually type.
- **Hard negatives (~3 per pair).** Discrimination — teach the model
  the difference between "breast cancer surgery" and "breast cancer
  immunotherapy."

**Why all three?** Each addresses a different failure mode of a
single-source dataset.
- Metadata-only would be 100% clinical phrasing — model would never
  learn patient language.
- Synthetic-only would be too small (1,500 isn't enough to fine-tune
  a 110M-param model).
- Without hard negatives, MNRL's in-batch negatives are mostly
  unrelated — easy negatives that don't sharpen discrimination.

**Trade-off.** The synthetic queries are LLM-generated, so they may
not perfectly capture how *real* patients write. We mitigated this
with explicit examples in the prompt ("come back" not "recurrent",
"stomach" not "gastric"). True validation would come from logging
real user queries — week 8's plumbing supports that, but it'd take
weeks of users to accumulate.

**Interview answer.** *"Three sources to balance volume, realism, and
discrimination quality. Metadata gives 242K pairs for free. Synthetic
adds 1,500 realistic patient queries via Claude Haiku at $2 total.
Hard negatives drawn from same-condition different-intervention trials
teach the model fine-grained distinctions. Each source addresses a
different failure mode."*

### Decision 11 — Fine-tune vs use off-the-shelf (results-dependent)

This decision was kept open until we had data. Both outcomes are
valid:

- *"Fine-tuning improved NDCG@10 by 49%; the hub trial problem
  disappeared."* (Our actual result.)
- *"Fine-tuning produced marginal gains; pre-training already covers
  this domain well; ship off-the-shelf for simplicity."* (Plausible
  alternative outcome.)

Either way, the *honest measurement* is what matters in an interview.
"I fine-tuned because someone said I should" is a weak answer.
"I fine-tuned because the off-the-shelf cosine spread of 0.047 was
too tight to discriminate, and after fine-tuning we measured the
spread at 0.10 and NDCG@10 at 0.796 vs 0.534 — a 49% improvement on
20 labeled queries, with the hub trial problem fully eliminated"
is a strong one.

### Decision 12 — LLM-as-judge for evaluation labels

**Context.** Fine-tuning needs to be *measured*, not just hoped for.
Standard evaluation requires labeled (query, trial) relevance pairs.

**Options.**
- Hire human annotators (~$5/label × 990 labels = $5,000).
- LLM-as-judge — use Claude Haiku (~$0.50 for 990 labels).
- Click-data — log real user behavior and treat clicks as labels.
  No real users yet.

**We chose LLM-as-judge.** Reasons:

- **10,000× cheaper.** Lets us iterate quickly on the prompt.
- **Validated in literature.** ACL 2023, EMNLP 2024 papers show LLM
  judges correlate strongly with human annotators on graded
  relevance tasks (Pearson r ≈ 0.7-0.9 typical).
- **Reproducible.** Same model + same prompt + same temperature →
  same labels. Human labels drift with annotator fatigue.

**The risks and how we mitigated.**

- **Risk: LLM has the same blind spots as our retrieval system.** If
  Claude Haiku misses a trial type, our metric will too.
  *Mitigation:* spot-check 50 labels manually, and validate with
  human-annotated samples in a future week.
- **Risk: prompt sensitivity.** Wording the rubric differently could
  shift the score distribution.
  *Mitigation:* run with `--limit 10` first, eyeball the results,
  iterate on the prompt before labeling 990.
- **Risk: cost creep.** $0.50 for 990 labels is cheap; $50 for 99K
  labels is non-trivial.
  *Mitigation:* keep label sets small (20-50 queries), only expand
  when needed.

**Interview answer.** *"LLM-as-judge for cost and reproducibility,
validated against the literature (ACL/EMNLP 2023-2024). I'd
follow up with human-annotated calibration samples and Cohen's
kappa to verify before publishing the metric externally."*

---

## Part 7 — File map

| File | New in week 3 | Purpose |
|---|---|---|
| `configs/training_data.yaml` | new | Cancer-type taxonomy, sampling caps, API config |
| `configs/training/embeddings.yaml` | new | Hyperparameters for `finetune_embeddings.py` |
| `scripts/generate_training_data.py` | new | 3-source training-data pipeline |
| `scripts/finetune_embeddings.py` | new | sentence-transformers training script |
| `notebooks/finetune_biolinkbert.ipynb` | new | Colab notebook running the script on A100 |
| `scripts/build_eval_dataset.py` | new | Pooled labeling with Claude Haiku |
| `scripts/compare_embeddings.py` | new | NDCG@5/NDCG@10/MRR comparison + MLflow |
| `models/embeddings/fine-tuned/` | new artifact | ~430 MB of fine-tuned weights + metadata.json |
| `data/training/{train,val}_pairs.jsonl` | new artifact | 731K triplets total |
| `data/training/synthetic_queries.jsonl` | new artifact | 1,500 LLM-generated patient queries |
| `data/evaluation/labeled_queries.jsonl` | new artifact | 990 LLM-labeled (query, trial) pairs |
| `data/faiss_offshelf.index` + `.json` | new artifact | Off-the-shelf FAISS for comparison |
| `data/faiss_finetuned.index` + `.json` | new artifact | Fine-tuned FAISS (412 MB) |

The training data and FAISS indexes are gitignored. Model weights are
gitignored, but `models/embeddings/fine-tuned/metadata.json` *is*
checked in (week 8).

---

## Part 8 — Interview prep

### 7.1 The big question

> *"How would you fine-tune a sentence embedding model for a domain
> like clinical trial retrieval?"*

A good answer covers:

1. **Identify the failure mode.** Off-the-shelf anisotropy. Cosine
   spread test (`{breast, lung, bicycle}` distance ratios).
2. **Choose a base model.** Domain-matched pretraining (BioLinkBERT
   for biomedical), not generic English.
3. **Generate training data.** Three-source recipe — metadata
   (volume), LLM-synthetic (realism), hard negatives (discrimination).
   Stratify by category to avoid majority-class dominance.
4. **Choose a loss.** MNRL with in-batch negatives is the default for
   bi-encoder retrieval. Triplet loss with explicit margin if the
   in-batch negatives aren't hard enough.
5. **Train.** Batch size 32+, LR 2e-5, 1-3 epochs, fp16. Watch the
   *retrieval metric* (NDCG@10), not just the loss.
6. **Measure.** Pool top-k across all candidate models, label the
   union with LLM-as-judge, compute NDCG@5/10 and MRR, log to
   experiment tracking (MLflow / Weights & Biases / Vertex).

### 7.2 Specific questions

**"What's a hard negative and why does it matter more than an easy
negative?"**
A hard negative is one that's *similar* to the positive but actually
wrong. (Same disease, different drug.) Easy negatives ("breast cancer"
vs "bicycle repair") give the model gradient that pushes those two
apart — which it already knows. Hard negatives push it on the *fine*
distinction it doesn't know. Most of the learning signal comes from
hard negatives; easy ones are nearly free training but don't move the
needle.

**"Why MultipleNegativesRankingLoss instead of TripletLoss?"**
MNRL gets ~N negatives per anchor for free (in-batch), where N is
batch size. TripletLoss would need an explicit negative for each
anchor — but in-batch negatives from MNRL are *automatic*. The
trade-off: MNRL's in-batch negatives may be easy. We compensate by
also providing an explicit hard negative in our triplets, so each
anchor sees 1 hard negative + (N-1) easy in-batch negatives.

**"Why fp16 mixed precision?"**
Halves memory (lets us fit batch 32 on a 40 GB A100 even with the
overhead). Speeds up matmul on tensor cores. Minimal accuracy cost
because the loss is still computed in fp32.

**"How did you split train/val without leakage?"**
Split by *NCT ID*, not by triplet. All triplets sharing the same
positive trial go into the same split. Otherwise the val set would
contain triplets whose positive trial was already memorized by the
model from a train-side triplet — leakage.

**"Why pool from BOTH models when labeling?"**
Otherwise you'd systematically penalize whichever model your labels
*didn't* come from. If you only label fine-tuned's results,
off-the-shelf's unique discoveries (correct or not) never get
labeled, so they default to score 0 and tank its metrics. Pooling
removes this asymmetry.

**"How do you know the LLM judge isn't biased?"**
We don't, fully — that's a known limitation of LLM-as-judge. The
defenses are (1) label a small human-validated sample and check
agreement (Cohen's kappa); (2) use multiple LLMs and check
inter-model agreement; (3) sanity-check the label distribution
itself (it shouldn't be uniformly 3 or uniformly 0). For our 20
queries, ~46% of labels were 3 and ~22% were 0 — bimodal but not
collapsed, suggesting a calibrated rubric.

**"What's NDCG and why is it the right metric?"**
NDCG@k = Discounted Cumulative Gain over top-k, normalized by ideal.
DCG weights position-1 relevance fully and decays log2 with rank.
"Right metric" because (1) it uses *graded* relevance (a 3 is
better than a 2), and (2) it rewards getting relevant items at the
top, which matches user behavior.

### 7.3 Code-review questions

1. **Why does `print_training_examples` exist?**
   Defense in depth. Before training, eyeball the first 5 examples.
   If the anchor doesn't match the positive even to a human, the
   data pipeline has a bug — find it before burning A100 hours.

2. **Why is the metadata.json schema exactly these keys?**
   Because week 8's `ci_quality_gate.py` reads them. Standardized
   schema (`training_date`, `eval_metrics`, `hyperparameters`,
   `n_features`/`n_samples` where applicable) lets the audit gate
   work uniformly across model types.

3. **Why is `evaluator` subsampled to 5,000 instead of using all
   145K val examples?**
   Each eval pass re-encodes the corpus (~5,000 docs × 768 dim ≈ 30
   sec on A100). 145K would be 10+ minutes per eval, with eval every
   1000 steps. The training run would be ~50% eval time. 5,000 is
   small enough for fast eval and large enough that the metric
   doesn't bounce randomly.

4. **Why save metadata before saving the model weights?**
   Actually we save *after*. The trainer's `save_model()` writes
   weights; then we write metadata.json. If saving order matters,
   document it; in our case we'd rather have weights without metadata
   than metadata pointing at no weights.

### 7.4 Tradeoff questions

**"You spent $2 on synthetic queries. Could you have spent $20 for
15,000 instead?"**
Yes — diminishing returns. 1,500 was enough for the mid-rung of the
data pipeline; the marginal NDCG gain from 15K queries would be
small relative to the marginal cost. We chose to keep the budget tiny
so we can iterate on the prompt / sources without committing real
spend each iteration.

**"Why one labeled set of 990 instead of multiple folds?"**
Time. K-fold cross-validation would give us tighter confidence
intervals. We compromised: 1,000 bootstrap resamples on the 20-query
metric, computed in week 4. Bootstrap CI on a single sample is
weaker than k-fold CV but cheaper and fast enough to display in the
ablation table.

**"How did you decide max_per_group = 2,000?"**
Heuristic, not derived. The smallest cancer group is mesothelioma
(~400 trials). We wanted at least 5× over-sampling on common groups
so the model has more breast cancer signal than mesothelioma signal,
but not 30× (which would dominate the loss). 2,000 / 400 = 5×, which
is a defensible middle.

---

## Part 9 — How to run everything

```bash
# Pre-reqs: weeks 1 & 2 already done; you have data/trials.db,
# data/faiss_offshelf.index, ANTHROPIC_API_KEY in .env.

# Step 1: Generate training data (~30 min for metadata + ~10 min for synthetic)
python scripts/generate_training_data.py
# or to skip the synthetic API calls (free):
python scripts/generate_training_data.py --skip-synthetic
# or to resume after a crash:
python scripts/generate_training_data.py --resume

# Step 2: Fine-tune (5 hours on A100; do this in Colab via the notebook)
# Open notebooks/finetune_biolinkbert.ipynb and run all cells.
# When training finishes, download models/embeddings/fine-tuned/ to your laptop.

# Step 3: Build the fine-tuned FAISS index (~30 min on CPU)
python scripts/build_index.py --skip-bm25 --model fine-tuned

# Step 4: Build the labeled eval dataset (~10 min, ~$0.50)
python scripts/build_eval_dataset.py --limit 10        # preview first 10 labels
python scripts/build_eval_dataset.py                   # full 990 labels

# Step 5: Compare embeddings + log to MLflow
python scripts/compare_embeddings.py
make mlflow                                            # open the UI
```

In MLflow you'll see two runs (`off-the-shelf_hybrid_eval` and
`fine-tuned_hybrid_eval`). Compare side by side.

---

## Part 10 — What's next (week 4 preview)

After week 3 we have a high-quality bi-encoder for retrieval. But
retrieval ≠ ranking. The top 200 from hybrid retrieval contains the
right answers somewhere, but they're not in the right order yet.

Week 4 adds a *re-ranking* layer: a **cross-encoder** that scores each
(query, trial) pair jointly (instead of separately as the bi-encoder
does). Cross-encoders are slower but much more accurate. We'll fine-
tune one on the same 990 labeled pairs, plus convert the 730K
triplets to 200K binary pairs for additional training signal.

Then we'll add a **LightGBM LambdaRank** model on top of the
cross-encoder. LambdaRank uses our retrieval scores (BM25, semantic,
RRF, cross-encoder) plus metadata features (phase, status, enrollment,
condition match) and learns the optimal weighted combination. This
gives us the famous **ablation table** — *the* artifact of the entire
project.

---

## Part 11 — End of week 3 checklist

- [x] 731K (query, positive, negative) triplets generated from 3 sources
- [x] 1,500 synthetic patient queries via Claude Haiku ($2)
- [x] BioLinkBERT fine-tuned on Colab A100 (~5 hours, 3 epochs, fp16)
- [x] Output: `models/embeddings/fine-tuned/` (~430 MB) + `metadata.json`
- [x] Fine-tuned FAISS index built and saved to `data/faiss_finetuned.index`
- [x] 990 LLM-labeled (query, trial) pairs in `labeled_queries.jsonl`
- [x] Pooled labeling (off-the-shelf ∪ fine-tuned top-30) — no method bias
- [x] Comparison metrics computed: NDCG@5 +41%, NDCG@10 +49%, MRR same
- [x] Both embedding-comparison MLflow runs logged to
      `trialmind-retrieval`
- [x] Hub trial problem eliminated; cosine spread doubled
- [x] Design decisions 10-12 logged

That's the full week. If you can explain *why fine-tuning fixes
anisotropy*, *why hard negatives matter more than easy ones*, and
*why pooled labeling eliminates evaluation bias*, you've internalized
the work.
