# TrialMine Design Decisions Log

This document records every significant technical decision made during the project.
Each entry explains what we chose, why, and what we'd do differently at scale.

---

## Week 1

### Decision 1: ClinicalTrials.gov API v2 over XML Bulk Download

**Context:** Need to ingest clinical trial data. Two options: REST API v2 (JSON, filtered queries) or bulk XML download (full database dump, ~3GB compressed).

**Options:** API v2 (paginated JSON, server-side filtering) vs bulk XML (single download, local filtering).

**Choice:** API v2

**Why:** JSON is directly parseable into Pydantic models — no XML-to-dict conversion. Server-side `query.cond=cancer` filters to ~140K oncology trials at source, avoiding downloading all 500K+ trials. Pagination with `pageToken` supports resume on failure. Rate limit (0.5s delay) keeps us well within acceptable ingestion time.

**Trade-off:** Full download takes 30-60 min vs ~5 min for bulk XML. Cannot do complex cross-field filters server-side (e.g., "oncology AND recruiting AND Phase 3").

**At scale:** Bulk download for initial load + incremental API polling (filter by `lastUpdatePostDate`) for daily freshness. Would add a CDC (change data capture) pipeline.

> **Interview answer:** "API v2 because it filters at source — I only download the 140K oncology trials I need instead of 500K+, and JSON maps directly to my Pydantic models."

---

### Decision 2: SQLite over PostgreSQL

**Context:** Need persistent storage for 140K parsed trials between pipeline stages (download -> parse -> index).

**Options:** PostgreSQL (full RDBMS), SQLite (embedded), MongoDB (document store), flat files (Parquet/JSON).

**Choice:** SQLite

**Why:** Zero config, single portable file (912 MB), sufficient read throughput for batch indexing. SQL for ad-hoc debugging (`SELECT COUNT(*) FROM trials WHERE phase='Phase 3'`). No server process to manage. The data flows one direction (write once during ingestion, read during indexing) — no concurrent write pressure.

**Trade-off:** No concurrent writes, single-machine only, no built-in full-text search. Not suitable as a serving database for the API.

**At scale:** PostgreSQL with read replicas, or a managed database (Cloud SQL, RDS). Would also consider DuckDB for analytical queries over trial metadata.

> **Interview answer:** "SQLite because this is a batch pipeline — write once, read many. Zero operational overhead, and the 140K-row dataset fits in a single file."

---

### Decision 3: Elasticsearch for BM25 Search

**Context:** Need full-text search with relevance ranking over 140K clinical trial documents.

**Options:** Elasticsearch (BM25 + field boosting), PostgreSQL full-text search (ts_vector), Whoosh (pure Python), Tantivy/Meilisearch.

**Choice:** Elasticsearch 8.x

**Why:** Industry-standard BM25 with built-in field boosting (title 3x, conditions 2x), custom analysers (English stemming + stop words), and term-level filters (phase, status) in a single query. Handles 140K docs in a single shard with ~20-35ms query latency. Docker setup is one command. Scales to millions of documents without architecture changes.

**Trade-off:** Adds a Docker service (~512MB-1GB memory). Requires index rebuilding when mappings change. JVM-based — heavier than Tantivy for the same workload.

**At scale:** Managed Elasticsearch (Elastic Cloud) or OpenSearch. Multi-shard index with replicas for throughput. Would add synonym expansion (e.g., "chemo" -> "chemotherapy") at the analyser level.

> **Interview answer:** "Elasticsearch because BM25 with field boosting and custom analysers gives me relevance tuning out of the box — title matches rank higher than body matches, and English stemming handles morphological variants."

---

### Decision 4: FastAPI over Flask

**Context:** Need an HTTP API framework to serve search results to the Streamlit frontend.

**Options:** Flask (mature, synchronous), FastAPI (modern, async, typed), Django REST Framework (batteries-included).

**Choice:** FastAPI

**Why:** Pydantic models for request/response validation (shared with the data pipeline), auto-generated OpenAPI docs at `/docs`, native async support for concurrent search requests, type hints that match the project's coding standards. Lifespan context manager cleanly handles Elasticsearch + FAISS + embedder startup/shutdown.

**Trade-off:** Smaller ecosystem than Flask/Django. Async isn't strictly necessary at current scale (single user), but doesn't hurt.

**At scale:** Same choice — FastAPI with multiple uvicorn workers behind a load balancer. Would add rate limiting middleware, API key auth, and response caching for frequent queries.

> **Interview answer:** "FastAPI because Pydantic validation, auto-generated docs, and async support align with the project's type-first approach — and the same Pydantic models validate both API boundaries and internal data flow."

---

### Decision 5: Keep Trials with Missing Fields

**Context:** ~5-10% of trials from ClinicalTrials.gov have sparse metadata — missing summary, eligibility criteria, or intervention details.

**Options:** Drop incomplete trials, require a minimum field count, keep everything with only NCT ID + title required.

**Choice:** Require NCT ID + title only, allow all other fields to be None.

**Why:** Missing data does not mean irrelevant trial. A Phase 3 breast cancer trial with no eligibility text is still a valid search result — a patient should know it exists. BM25 naturally ranks sparse trials lower (fewer matching fields). Semantic search embeds whatever text is available. Dropping trials risks excluding rare-cancer or newly-posted studies.

**Trade-off:** Sparse trials may rank unexpectedly high if their short text happens to match well. Some UI cards look empty.

**At scale:** Add a data quality score per trial (0-1, based on field completeness) and surface it in the UI. Use the score as a feature in the LightGBM re-ranker to soft-penalise incomplete trials rather than hard-filtering them.

> **Interview answer:** "Keep everything — missing metadata is not the same as irrelevant. BM25 and the re-ranker naturally downweight sparse trials, and filtering would silently drop rare-cancer studies that patients need most."

---

## Week 2

### Decision 6: BioLinkBERT over PubMedBERT / General Models

**Context:** Need a biomedical embedding model for semantic search over clinical trial text.

**Options:** all-MiniLM-L6-v2 (general, 384d), PubMedBERT (biomedical, 768d), BioLinkBERT (biomedical + citation structure, 768d).

**Choice:** `michiyasunaga/BioLinkBERT-base` (768-dim) via sentence-transformers with mean pooling.

**Why:** Pre-trained on PubMed WITH citation link structure — understands which medical concepts are related, not just which words co-occur. 768 dimensions captures more biomedical nuance than 384. Citation-aware pre-training means "glioblastoma" and "temozolomide" are linked through the papers that study them, not just co-occurrence statistics.

**Trade-off:** Slower than MiniLM (~50ms vs ~15ms per query). Not trained for retrieval — the model encodes token-level biomedical knowledge, not query-document similarity. Our evaluation confirmed this: cosine score range across 1000 results is only 0.047 (severe anisotropy). The model understands biomedical relationships but can't rank them for retrieval without fine-tuning.

**At scale:** Fine-tune on patient-to-trial query pairs with contrastive loss. The biomedical knowledge is the right foundation — the ranking behaviour is what needs training.

> **Interview answer:** "BioLinkBERT because it understands biomedical concept relationships through citation-aware pre-training. General models miss domain-specific connections like drug-indication mappings."

---

### Decision 7: RRF over Linear Score Combination

**Context:** Need to merge BM25 and semantic ranked lists into one hybrid result set.

**Options:** Linear combination (alpha * bm25_norm + beta * semantic), RRF (rank-based), learned merge.

**Choice:** RRF with k=60 (standard constant from Cormack et al., 2009).

**Why:** BM25 scores (5-50) and cosine similarity (0.85-0.90) are on completely different scales. Linear combination requires normalising both to [0,1], which is fragile — BM25 score distributions shift with query length. RRF uses ranks, which are always comparable. Parameter-free (k=60 is the literature default) and robust.

**Trade-off:** Treats both retrievers equally. Cannot express "trust BM25 more" without a weighting parameter. The LightGBM blender adds learned weighting on top.

**At scale:** After fine-tuning embeddings, consider weighted RRF or learned score interpolation. The cross-encoder re-ranker compensates for noisy fusion regardless.

> **Interview answer:** "RRF because it's score-scale independent — no normalisation needed, and robust even when one retriever is weaker than the other."

---

### Decision 8: IndexFlatIP (Exact) over Approximate FAISS

**Context:** Need nearest-neighbour search over 140K 768-dim trial embeddings.

**Choice:** FAISS `IndexFlatIP` with L2-normalised vectors (inner product = cosine similarity).

**Why:** At 140K vectors, exact search takes 10-50ms — fast enough for interactive use. Approximate indexes (IVF, HNSW) add complexity, tuning parameters (nprobe, ef), and recall loss without meaningful benefit at this scale. Memory = 140K x 768 x 4 bytes = 412 MB, fits comfortably in RAM.

**Trade-off:** Brute-force linear scan. Latency scales linearly with corpus size.

**At scale:** Switch to IndexIVFFlat at 1M+ vectors, IndexIVFPQ at 100M+ (trades ~5% recall for 10-50x speedup). Would also add GPU acceleration via faiss-gpu.

> **Interview answer:** "IndexFlatIP because exact search is fast enough at 140K vectors and eliminates approximation error. I'd switch to IVF at the million-vector mark."

---

### Decision 9: Equal BM25/Semantic Weight in RRF

**Context:** Each retriever contributes 1/(60 + rank) per document in RRF. Need to decide whether to weight one higher.

**Choice:** Equal weighting — both methods contribute identically.

**Why:** Without labelled relevance data, we can't empirically determine which retriever is more trustworthy. Our 20-query evaluation shows 0% top-3 overlap (the methods find completely different trials), so both contribute unique signal. Equal is the safest uninformed default.

**Trade-off:** Our evaluation shows BM25 is currently the stronger retriever (60 unique trials vs 38 for semantic across 20 queries). Weighting BM25 higher would improve results today, but would hide semantic improvements after fine-tuning.

**At scale:** The LightGBM blender learns optimal weights from labelled data. Ablation table will reveal per-method contribution.

> **Interview answer:** "Equal weights because without relevance labels, asymmetric weighting is premature optimisation. The re-ranker in the next phase learns the right balance from data."

---

### Evaluation: Hybrid Retrieval Baseline (2026-03-25)

Ran 20 oncology test queries across BM25, semantic, and hybrid search. Queries range from clinical ("triple negative breast cancer neoadjuvant") to patient-language ("clinical trial for glioblastoma that has come back") to misspelled ("melanomt with checkpoint inhibitors"). Results logged as MLflow baseline runs in experiment `trialmind-retrieval`.

**Key metrics:**

| Metric | BM25 | Semantic | Hybrid |
|---|---|---|---|
| Avg latency | 25 ms | 50 ms | 77 ms |
| Unique trials (20 queries, top 3) | 60 | 38 | 57 |
| Top-3 BM25∩Semantic overlap | 0% | 0% | — |

**Finding 1: Zero top-3 overlap, but nonzero top-200 overlap.**

The methods return completely disjoint top-3 results. But at the top-200 candidate level, overlap ranges from 1% to 16%:

| Query type | Top-200 overlap | Example |
|---|---|---|
| Broad clinical terms | 16% (31/200) | "immunotherapy for non-small cell lung cancer" |
| Specific + clinical | 8-9% (17-18/200) | "breast cancer hormone receptor positive phase 3" |
| Patient language | 1-2% (2-4/200) | "glioblastoma that has come back" |

Relevant trials ARE in the semantic candidate pool — they're just buried at rank 30-100+ instead of surfacing at rank 1-3. RRF confirms this: 52% of hybrid top-3 results are tagged `source: "both"`.

**Finding 2: Semantic search has severe embedding space collapse.**

3 generic trials occupy 33% of all 60 semantic result slots ("Mebendazole" in 8/20 queries, "R2 Follicular Lymphoma" in 6/20, "Efficacy Prediction Model" in 6/20). Cosine score range across 1000 results is only 0.047 (0.8988 to 0.8517) — the model can barely distinguish relevant from irrelevant.

**Finding 3: The model understands paraphrase but can't retrieve.**

"come back" and "recurrent glioblastoma" map to the same embedding region (30% semantic overlap), but BM25 can't bridge them (0% overlap). The semantic model has the right knowledge — it just can't surface it through the anisotropic embedding space.

**Diagnosis:** Architecture sound, embeddings need work. Fixable via:

1. **Cross-encoder re-ranking** — highest leverage, works immediately
2. **Contrastive fine-tuning** — fixes the root cause by spreading the embedding space
3. **Embedding whitening** — quick experiment, zero retraining
4. **Model swap** — fallback to retrieval-trained model like S-PubMedBert-MS-MARCO

**Data:** `data/evaluation/method_comparison.csv` (180 rows). MLflow experiment: `trialmind-retrieval`.

---

## Week 3

### Decision 10: Three-Source Training Data Pipeline

**Context:** Need training data for contrastive fine-tuning of BioLinkBERT to fix embedding anisotropy (cosine range 0.047, 3 hub trials monopolising 33% of slots). No existing labelled query-trial relevance data.

**Options:** Manual annotation (gold standard), metadata-derived pairs only (free but robotic), LLM-generated queries only (natural but expensive), hybrid approach.

**Choice:** Three complementary sources in a single script (`scripts/generate_training_data.py`):

1. Metadata-derived pairs (242K) — conditions, interventions, phases extracted from trial fields
2. Synthetic patient queries (1,500) — Claude Haiku API generates natural patient language
3. Hard negatives (730K triplets) — same condition, different intervention/trial

**Why:** Each source addresses a different weakness. Metadata pairs teach the model basic concept-trial associations at scale. Synthetic queries bridge the clinical-to-patient language gap that BM25 can't handle (our evaluation showed 0% overlap on patient-language queries). Hard negatives force the model to distinguish confusingly similar trials — without them, contrastive loss only learns from random in-batch negatives which are too easy.

**Trade-off:** Metadata queries are repetitive ("breast cancer", "pembrolizumab for breast cancer") — the model sees many near-duplicate training signals. Synthetic queries are limited to 1,500 by API budget. Hard negative mining uses simple keyword overlap, not semantic similarity.

**At scale:** Replace keyword-based hard negative mining with embedding-based mining (encode all trials, find nearest neighbours that aren't relevant). Generate 10K+ synthetic queries using Claude. Add human relevance judgements for a gold evaluation set.

> **Interview answer:** "Three sources because each teaches a different skill: metadata pairs teach concept association, synthetic queries bridge the vocabulary gap between patients and clinicians, and hard negatives force fine-grained discrimination between similar trials."

---

### Decision 11: Stratified Sampling by Cancer Type

**Context:** 140K trials are heavily skewed — breast cancer has 15K trials, mesothelioma has 292. Need balanced training data.

**Options:** Random sampling (simple but biased), stratified sampling with caps, upsampling rare types.

**Choice:** Stratified sampling with 2000-trial cap per cancer group. 23 cancer type groups defined by keyword matching on `trial.conditions`, plus an "other" catch-all.

**Why:** Without caps, breast cancer (15K) and lung cancer (11K) would dominate training — the model learns "breast cancer" embeddings well but fails on neuroblastoma (361 trials). A 2000-trial cap means rare cancer groups keep all their trials while common groups are downsampled. This produces 40K sampled trials across 23 groups.

**Trade-off:** The "other" bucket (47K trials) includes basket trials, supportive care, and non-oncology studies that leaked through. Capping it at 2000 loses some diversity. The keyword-based taxonomy misclassifies trials with unusual condition strings.

**At scale:** Use MeSH terms or a medical NER model for cancer type classification instead of keyword matching. Consider curriculum learning: start training on easy (cross-cancer) negatives, then switch to hard (within-cancer) negatives.

> **Interview answer:** "Stratified sampling with caps because training data distribution directly controls what the model learns. A model trained on 90% breast cancer trials won't help a mesothelioma patient."

---

### Decision 12: Claude Haiku for Synthetic Patient Queries

**Context:** Need patient-language queries to bridge the vocabulary gap (our evaluation showed BM25 and semantic have 0% overlap on patient-language queries like "glioblastoma that has come back"). Template-based generation is free but robotic.

**Options:** Templates only (free, robotic), local LLM via Ollama (free, medium quality), Claude Haiku API (~$1-2 for 1,500 queries).

**Choice:** Claude Haiku API (`claude-haiku-4-5-20251001`) for 1,500 queries.

**Why:** Quality gap is large — Claude produces queries like "My daughter has a benign ovarian tumor and needs surgery — are there any new treatment options?" vs templates producing "I was diagnosed with ovarian cancer and looking for treatment options." The personal context, emotional tone, and natural phrasing from Claude better represent how real patients search. At $0.001 per query ($1.50 total), the cost is negligible compared to the quality improvement.

**Trade-off:** Only 1,500 queries (0.6% of training data) — limited impact on overall training. Requires API key. Non-reproducible (different outputs each run due to temperature).

**At scale:** Generate 10K-50K synthetic queries using Claude. Add few-shot examples from real patient forum posts (Reddit r/cancer, HealthUnlocked) to improve prompt quality. Consider fine-tuning a smaller model on the Claude outputs for cost-free generation.

> **Interview answer:** "Claude Haiku because the cost ($1.50) is trivial and the quality gap over templates is dramatic for patient-language queries — and that's exactly the vocabulary gap our embedding model needs to learn."

---

### Decision 13: Hard Negatives via Condition Keyword Overlap

**Context:** Contrastive learning needs hard negatives — trials that are similar enough to confuse the model but are not the correct match. In-batch negatives (random trials from other queries in the same batch) are too easy.

**Options:** Random negatives (easy), BM25-retrieved negatives (requires Elasticsearch), embedding-based negatives (requires encoding all trials), keyword overlap (simple, in-memory).

**Choice:** Keyword overlap on condition strings, preferring trials with different interventions. Build an in-memory condition-word -> NCT ID index, find trials sharing keywords, partition into hard (different intervention) and easy (same intervention) candidates, sample 3 negatives per positive.

**Why:** Simple and effective. A breast cancer trial about pembrolizumab vs a breast cancer trial about tamoxifen is the exact kind of distinction the model needs to learn. No external dependencies (no Elasticsearch, no GPU for encoding). The condition index builds in seconds over 40K sampled trials. 730K triplets generated in ~12 minutes.

**Trade-off:** Only uses condition text, not semantic similarity — misses some hard negatives that share interventions but different conditions. The keyword tokenisation is naive (split on spaces, skip <3 chars). 180 pairs (0.07%) found no candidates at all.

**At scale:** Mine negatives using the current model's embeddings — encode all trials, for each positive find the k nearest neighbours that aren't relevant. This produces the hardest possible negatives. Update the negative set every training epoch (dynamic hard negative mining).

> **Interview answer:** "Keyword overlap because it's simple, fast, and targets exactly what we need: same cancer, different drug. The 730K triplets generated in 12 minutes with no external dependencies — embedding-based mining would be better but requires encoding 140K trials first."

---

### Decision 14: SentenceTransformerTrainer over Legacy model.fit()

**Context:** Need to fine-tune BioLinkBERT as a bi-encoder for clinical trial retrieval. sentence-transformers 5.x offers two APIs: the new `SentenceTransformerTrainer` (HuggingFace Trainer-based) and the legacy `model.fit()`.

**Options:** Legacy `model.fit()` (simpler, well-documented) vs `SentenceTransformerTrainer` (new 5.x API, HF ecosystem integration).

**Choice:** `SentenceTransformerTrainer`

**Why:** Built on HuggingFace Trainer — automatic checkpointing, `load_best_model_at_end`, fp16 mixed precision, gradient accumulation, and MLflow integration via `report_to="mlflow"` for free. The `InformationRetrievalEvaluator` integrates natively for NDCG@10/MRR@10 evaluation during training. Dataset loading via HuggingFace `datasets` library handles the 1GB training file efficiently with memory mapping.

**Trade-off:** Newer API with fewer community examples. `warmup_ratio` parameter is deprecated in Transformers v5+ (works but warns). Required `accelerate` package as an extra dependency.

**At scale:** Same choice — the Trainer API supports distributed training across multiple GPUs with zero code changes via `accelerate launch`.

> **Interview answer:** "SentenceTransformerTrainer because it integrates with the HuggingFace ecosystem — automatic checkpointing, fp16, and MLflow logging with zero boilerplate."

---

### Decision 15: MultipleNegativesRankingLoss with Hard Negatives

**Context:** Need a loss function that teaches the model to rank relevant trials above irrelevant ones. Training data has triplets: (query, positive_trial, hard_negative_trial).

**Options:** CosineSimilarityLoss (pairwise), ContrastiveLoss (binary), TripletLoss (anchor/pos/neg), MultipleNegativesRankingLoss (in-batch + explicit negatives).

**Choice:** `MultipleNegativesRankingLoss` (MNRL) with scale=20.0 and 3-column input (anchor, positive, negative).

**Why:** MNRL uses BOTH in-batch negatives (all other positives in the batch become negatives) AND our explicit hard negatives from the third column. With batch_size=32, each query sees 1 positive + 31 in-batch negatives + 1 hard negative = 33 contrasts per step. This is far more efficient than TripletLoss (1 positive + 1 negative). The scale=20.0 parameter sharpens the softmax distribution, encouraging more decisive ranking.

**Trade-off:** Memory-intensive — 3 forward passes per step (anchor, positive, negative). Required batch_size reduction from 32 to 16 on T4 GPU (15GB VRAM), compensated with gradient accumulation. On A100 (40GB), batch_size=32 fits.

**At scale:** Consider GISTEmbedLoss (guided in-batch negatives using a teacher model) or CachedMultipleNegativesRankingLoss (supports larger effective batch sizes via gradient caching).

> **Interview answer:** "MNRL because it gets 33 contrasts per training step — 31 in-batch negatives plus our keyword-mined hard negative. Far more efficient than triplet loss which only sees 1 negative per step."

---

### Decision 16: Colab Pro (A100) over Local MPS Training

**Context:** BioLinkBERT fine-tuning with MNRL requires significant GPU memory. Apple M4 MPS has 18GB shared memory; T4 has 15GB dedicated VRAM.

**Options:** Local MPS (free, 18GB shared), Colab free T4 (free, 15GB), Colab Pro A100 ($10/mo, 40GB).

**Choice:** Colab Pro with A100 GPU.

**Why:** MNRL does 3 forward passes per step. With batch_size=32 and seq_length=512, peak memory exceeds 18GB — OOM on both MPS and T4. Reducing batch_size to 4 with gradient accumulation would fit but cripples training quality (fewer in-batch negatives) and increases wall time to 24+ hours. A100 (40GB) runs batch_size=32 with fp16 in ~5 hours. Cost: $10/mo, amortised across bi-encoder + cross-encoder training.

**Trade-off:** Requires data upload to Google Drive (~1.3GB). Colab session management (keep tab open). Model download back to local machine.

**At scale:** Dedicated GPU instances (Lambda Labs, RunPod) or cloud training (SageMaker, Vertex AI). Would use multi-GPU training with DeepSpeed for larger models.

> **Interview answer:** "A100 on Colab because MNRL's triple forward pass needs 40GB to maintain batch_size=32 — and in-batch negative count directly affects contrastive learning quality. $10 to avoid crippling the batch size."

---

### Evaluation: Post-Fine-Tuning Comparison (2026-03-26)

Re-ran the same 20 oncology queries after fine-tuning BioLinkBERT with MNRL on 586K triplets (3 epochs, A100, 288 min).

**Training metrics (val set, 5K queries against 5K corpus docs):**

| Metric | Step 1000 | Step 14000 (final) |
|---|---|---|
| Training Loss | 1.237 | 0.445 |
| NDCG@10 | 0.336 | 0.492 |
| MRR@10 | 0.284 | 0.426 |
| Recall@1 | 0.192 | 0.300 |
| Recall@10 | 0.504 | 0.700 |

**Retrieval comparison (20 queries, top 3):**

| Metric | Before | After |
|---|---|---|
| BM25∩Semantic top-3 overlap | 0% | 7% |
| Cosine range in top 5 | 0.047 (across 1000) | 0.10 |
| Hub trial monopolisation | 33% of slots | 0% |
| Semantic unique trials | 38 | 60 |

**Key improvements:**

1. **Anisotropy fixed at source.** Cosine scores now span 0.51-0.61 in top 5 results (was 0.85-0.90 for all results). The model genuinely differentiates relevant from irrelevant.
2. **Hub trials eliminated.** No single trial dominates semantic results — every query returns distinct, topically relevant trials.
3. **Qualitative relevance.** "bladder cancer BCG unresponsive" returns BCG-related trials. "EGFR mutated lung cancer" returns EGFR-targeted therapy trials. Before fine-tuning, these returned generic oncology trials.
4. **BM25 still complementary.** 93% of top-3 results remain disjoint — the methods find different relevant trials, which is ideal for hybrid fusion.

**Data:** `data/evaluation/method_comparison.csv` (180 rows). MLflow experiment: `trialmind-retrieval`.

---

### Decision 17: LLM-as-Judge over Human Annotation

**Context:** Need relevance labels for the 20 test queries to compute NDCG and MRR. Without labels, we can only measure overlap and latency — not ranking quality.

**Options:** Human annotation (gold standard), LLM-as-judge (Claude Haiku), heuristic labels (assume trials matching cancer type are relevant).

**Choice:** Claude Haiku API (`claude-haiku-4-5-20251001`) with a structured 0-3 relevance scale.

**Why:** Labeling 990 (query, trial) pairs manually would take 8+ hours. Claude Haiku labels at ~90 pairs/minute for ~$2 total. The 4-point scale (0=wrong cancer, 1=marginal, 2=relevant, 3=highly relevant) captures gradations that binary relevant/irrelevant misses — critical for NDCG which is graded. The prompt provides trial title, conditions, phase, status, and eligibility text for informed judgment.

**Trade-off:** No inter-annotator agreement measurement. No human calibration — we don't know if Haiku's "score 2" matches a clinician's. Score distribution may reflect Haiku's tendency toward generosity (46% score 3, 23% score 0). Results are non-reproducible across API versions.

**At scale:** Label a 50-100 pair gold set with domain experts, measure Haiku's agreement (Cohen's kappa), then use Haiku for the remaining labels with the calibrated confidence interval. Consider using multiple LLM judges and majority voting.

> **Interview answer:** "LLM-as-judge because labeling 990 pairs manually is infeasible at this stage, and graded relevance (0-3) is necessary for NDCG. I'd calibrate against human labels in production."

---

### Decision 18: Pooled Judgment over Single-Model Retrieval for Evaluation

**Context:** Need to evaluate off-the-shelf vs fine-tuned embeddings fairly. Initial approach retrieved top-30 results using only the fine-tuned model, labeled those, then compared both models against those labels.

**Options:** Label only fine-tuned results (fast, 600 pairs), label only off-the-shelf results (same bias, reversed), pool results from both models and label the union (fair, more pairs).

**Choice:** Pooled judgment — retrieve top-30 from each model, deduplicate, label all unique trials.

**Why:** The single-model approach had a critical bias: when the off-the-shelf model retrieved a trial not in the fine-tuned top-30, that trial defaulted to relevance 0 — even if it was highly relevant. This inflated the gap from +49% (real) to +123% (biased). Pooling produced 990 unique pairs with only 21% overlap between models (210 shared, 390 each model-only), confirming the bias was significant.

**Impact of the fix:**

| Metric | Biased eval | Pooled eval | What changed |
|---|---|---|---|
| Off-the-shelf NDCG@10 | 0.357 | 0.534 | +50% (was penalized) |
| Fine-tuned NDCG@10 | 0.797 | 0.796 | Unchanged (already labeled) |
| Reported gap | +123% | +49% | Honest now |
| MRR gap | +19% | 0% | BM25 drives first result |

**Trade-off:** 65% more API calls (990 vs 600). Requires loading both models for retrieval, doubling memory during dataset creation.

**At scale:** Use depth-k pooling from all candidate systems (standard in TREC evaluations). Include random documents from the corpus as negative controls to estimate labeling quality.

> **Interview answer:** "Pooled judgment because evaluating with one model's results biases against the other. Our initial +123% improvement was inflated to nearly +49% once we labeled the union — a textbook case of evaluation contamination."

---

### Decision 19: Explicit Transformer+Pooling for Non-ST Models

**Context:** `SentenceTransformer("michiyasunaga/BioLinkBERT-base")` auto-detects that the model has no `modules.json` and silently creates a wrapper with mean pooling. This appears to succeed — no exception is raised — but the model crashes with SIGSEGV during `.encode()`.

**Options:** Catch the crash (impossible — SIGSEGV kills the process), detect the warning and retry, proactively check for `modules.json` before loading.

**Choice:** Check for `modules.json` before loading. If absent (raw HuggingFace checkpoint), explicitly construct `Transformer` + `Pooling` modules instead of relying on auto-detection.

**Why:** The SIGSEGV is not catchable in Python — it terminates the process instantly. The auto-detection path in sentence-transformers creates an internally inconsistent model state for models without proper ST configuration. Explicit module construction bypasses the problematic code path entirely. The check works for both local paths (`Path(model_name) / "modules.json"`) and HuggingFace Hub models (`hf_hub_download`).

**Trade-off:** One extra HTTP request for Hub models to check `modules.json` existence. Fine-tuned models (which have `modules.json`) use the standard fast path.

**At scale:** Same approach — this is a robustness fix. Would file an upstream issue with sentence-transformers to fix the auto-detection path.

> **Interview answer:** "Proactive detection because SIGSEGV is uncatchable. I check for `modules.json` before loading — if absent, I wire up Transformer and Pooling modules explicitly, bypassing the buggy auto-detection path."

---

### Evaluation: Embedding Comparison — Pooled LLM-Labeled (2026-03-27)

Used 990 pooled relevance labels (20 queries x ~50 unique trials from both models) to compare off-the-shelf vs fine-tuned BioLinkBERT in hybrid search.

**Pooling statistics:**

| Source | Count | % |
|---|---|---|
| Both models | 210 | 21% |
| Fine-tuned only | 390 | 39% |
| Off-the-shelf only | 390 | 39% |

**Score distribution (990 labels):**

| Score | Count | % | Meaning |
|---|---|---|---|
| 0 | 225 | 22.7% | Wrong cancer type |
| 1 | 172 | 17.4% | Marginal — same area, wrong specifics |
| 2 | 138 | 13.9% | Relevant — patient could be eligible |
| 3 | 455 | 46.0% | Highly relevant — strong match |

**Comparison results:**

| Metric | Off-the-shelf | Fine-tuned | Improvement |
|---|---|---|---|
| NDCG@5 | 0.577 | 0.816 | +41.4% |
| NDCG@10 | 0.534 | 0.796 | +49.1% |
| MRR | 0.917 | 0.917 | +0.0% |

Fine-tuned wins on 19/20 queries (only loss: "sarcoma clinical trials for young adults").

**Key findings:**

1. **Fine-tuning improves ranking, not first-result quality.** MRR is identical (0.917) — BM25 places a relevant result at rank 1 regardless of the embedding model. The fine-tuned model's advantage is in the quality of results at positions 2-10.
2. **The real improvement is +49%, not +123%.** The initial biased evaluation inflated the gap because 390 off-the-shelf-only results (39% of the pool) defaulted to relevance 0 without labels.
3. **Off-the-shelf model is better than it looked.** NDCG@10 of 0.534 (was 0.357 in biased eval) means the base model does find relevant trials — they just weren't labeled in the first evaluation.
4. **Hybrid search matters more than the embedding model.** BM25 handles keyword matching, the embedding model handles semantic understanding. The combination delivers MRR 0.917 with either model.

**Remaining limitations:**

1. **20 queries is thin.** One query swings the average by 5%. No bootstrap confidence intervals or paired significance tests. The +49% is likely real given the consistency (19/20 queries), but the exact number has wide error bars.
2. **No human calibration.** Haiku's relevance scores are uncalibrated — 46% score-3 may reflect LLM generosity rather than true relevance. Without a human-labeled gold set, we can't measure agreement (Cohen's kappa) or systematic bias.
3. **No BM25-only baseline.** We compare hybrid(off-the-shelf) vs hybrid(fine-tuned), but don't isolate how much of each model's NDCG comes from BM25 alone. The identical MRR suggests BM25 does the heavy lifting for top results.
4. **Score distribution is top-heavy.** 60% of labels are score 2-3 (relevant). NDCG looks better when most results are relevant because even imperfect orderings score well. A harder evaluation set with more score-0 distractors would be more discriminating.

**Data:** `data/evaluation/labeled_queries.jsonl` (990 rows), `data/evaluation/per_query_*.json`. MLflow experiment: `trialmind-retrieval`.

---

## Week 4

### Decision 20: BinaryCrossEntropyLoss over Graded Loss for Cross-Encoder

**Context:** Need to train a cross-encoder re-ranker to refine hybrid retrieval rankings. Training data consists of 586K triplets (query, positive_trial, negative_trial) with binary relevance — positive means "correct disease match", negative means "wrong disease."

**Options:** BinaryCrossEntropyLoss (convert triplets to binary pairs), MarginMSELoss (distill from a teacher), graded loss on the 990 labeled pairs (0-3 scale).

**Choice:** BinaryCrossEntropyLoss — convert each triplet to two binary pairs: (query, positive, 1.0) + (query, negative, 0.0).

**Why:** The 990 labeled pairs with graded relevance are too few for a 110M-param model — 20 queries split 80/20 gives only ~800 training pairs, guaranteeing overfitting. The 586K triplets provide sufficient data, but only have binary labels. BinaryCrossEntropyLoss is the natural fit for binary relevance data. The loss is well-supported by sentence-transformers' CrossEncoderTrainer API.

**Trade-off:** Binary labels teach disease-matching ("is this the right cancer?") but NOT graded relevance ("is this Phase 3 EGFR trial better than that Phase 1 broad immunotherapy trial?"). This turned out to be the critical limitation — see evaluation below.

**At scale:** Collect 5K-10K graded relevance labels (0-3) via LLM-as-judge on diverse queries, then train with ordinal or regression loss. Alternatively, distill from a large cross-encoder teacher using MarginMSELoss.

> **Interview answer:** "BinaryCrossEntropyLoss because 990 graded labels would overfit a 110M-param model, and 586K binary triplets provide sufficient training signal. The trade-off is that binary labels only teach disease-matching, not graded relevance — which limits the model to a tiebreaker role."

---

### Decision 21: Blended Scoring (0.7 RRF + 0.3 CE) over Pure Cross-Encoder Re-Ranking

**Context:** Standard cross-encoder re-ranking replaces the retrieval score entirely — candidates are sorted by CE score alone. Our fine-tuned CE was trained on binary labels (right/wrong disease), while hybrid RRF captures multi-signal relevance from both BM25 keyword matching and semantic similarity.

**Options:** Pure CE ranking (standard approach), blended scoring (weighted combination of RRF + CE), CE score as a LightGBM feature (deferred to next phase).

**Choice:** Blended scoring: `0.7 * RRF_normalized + 0.3 * CE_sigmoid`

**Why:** Pure CE ranking DESTROYS results. Both off-the-shelf (ms-marco-MiniLM, NDCG@5: -11.9%) and fine-tuned (BioLinkBERT CE, NDCG@5: -19.5%) cross-encoders degrade quality when used as sole rankers. Root cause: the CE learned binary disease-matching from training data, which replaces useful multi-signal RRF quality with a blunt "right disease or not" score. A breast cancer Phase 3 EGFR trial and a breast cancer Phase 1 supportive care trial both get CE score ≈1.0 — the CE can't distinguish them.

Blending preserves RRF's multi-signal ranking while allowing CE to act as a tiebreaker when RRF scores are close. The 0.7/0.3 split was chosen conservatively — CE gets enough weight to help on hard queries but not enough to override correct RRF rankings.

**Trade-off:** The improvement is marginal (+1.6% NDCG@5, +2.7% NDCG@10) — the CE adds value primarily on the hardest queries (sarcoma +31%, glioblastoma +14%) while slightly hurting 4/20 queries. The 4-second CPU latency is significant for a marginal gain.

**At scale:** Feed CE score as one feature (among many) into LightGBM rather than using it for direct ranking. The tree model can learn when to trust CE vs RRF, and combine both with metadata features (phase, enrollment, status). This is the architecturally correct solution — the CE score is a signal, not a decision-maker.

> **Interview answer:** "Blended scoring because the CE was trained on binary labels and doesn't understand graded relevance — pure CE re-ranking replaces good RRF signals with blunt disease-matching. At 0.3 weight, CE helps on hard queries without overriding correct rankings. The right long-term fix is feeding CE as a feature into LightGBM."

---

### Decision 22: T4 with Subsampled Data over Full Dataset Training

**Context:** 586K triplets → 1.17M binary pairs at batch_size=16 = 219K steps per epoch. On Colab T4, this would take 39+ hours for 3 epochs — exceeding session limits.

**Options:** A100 (fast, $0.10/hr, limited availability), T4 with full data (39 hours, impractical), T4 with subsampled data (2 hours).

**Choice:** T4 with 100K subsampled triplets (200K pairs), 1 epoch, early stopping at patience=3.

**Why:** The cross-encoder converges fast — validation NDCG@10 plateaued at 0.992 within 12,500 steps (the model early-stopped). 100K triplets already provide sufficient disease-matching signal. Training loss decreased from 0.693 to 0.199 in one epoch. The remaining 486K triplets are mostly near-duplicates (same cancer type, slightly different trial pairs).

**Trade-off:** Less exposure to rare cancer types. The model may underperform on uncommon conditions that appeared in the dropped 83% of triplets.

**At scale:** Use A100 or multi-GPU training with the full dataset. Consider curriculum learning: start with easy (cross-cancer) negatives, then progressively harder (within-cancer) negatives.

> **Interview answer:** "Subsampled to 100K triplets because the model converged in 12,500 steps — NDCG@10 hit 0.992 and early-stopped. Throwing more data at a binary classification task with diminishing returns isn't worth 39 hours of GPU time."

---

### Decision 23: Off-the-Shelf CE Baseline Before Fine-Tuning

**Context:** Fine-tuning a cross-encoder requires GPU hours. Need to know if an off-the-shelf model already helps — that determines whether fine-tuning is about domain adaptation (incremental) or essential.

**Options:** Skip baseline and fine-tune directly, evaluate off-the-shelf ms-marco-MiniLM first.

**Choice:** Evaluate `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, trained on 500K MS MARCO pairs) before any fine-tuning.

**Why:** The baseline told us something critical: off-the-shelf CE HURTS (NDCG@5: -11.9%). This means a general-domain cross-encoder doesn't understand biomedical text well enough to improve over our domain-specific hybrid retrieval. Fine-tuning is essential, not incremental.

**Trade-off:** Added ~15 minutes of evaluation time before starting the fine-tuning pipeline. Worth it for the diagnostic value.

**At scale:** Always evaluate off-the-shelf baselines before training. The baseline comparison is free information that prevents wasted GPU hours.

> **Interview answer:** "Baseline first because it cost 15 minutes and told us fine-tuning is essential — a general-domain cross-encoder actively degrades biomedical search quality."

---

### Evaluation: Cross-Encoder Re-Ranking (2026-03-28)

Evaluated cross-encoder re-ranking on 20 labeled queries (990 graded relevance labels, hybrid search with fine-tuned BioLinkBERT bi-encoder).

**Progressive evaluation — each approach tested:**

| Approach | NDCG@5 | NDCG@10 | MRR | vs Baseline |
|---|---|---|---|---|
| Hybrid only (baseline) | 0.816 | 0.796 | 0.917 | — |
| + off-the-shelf ms-marco-MiniLM (pure CE) | 0.719 | 0.707 | — | -11.9% |
| + fine-tuned BioLinkBERT CE (pure CE, no summary) | 0.634 | 0.660 | — | -22.2% |
| + fine-tuned BioLinkBERT CE (pure CE, with summary) | 0.657 | 0.651 | — | -19.5% |
| + fine-tuned BioLinkBERT CE (blended 0.7 RRF + 0.3 CE) | 0.829 | 0.817 | 0.950 | +1.6% |

**Per-query analysis (blended scoring):**

- Wins: 7/20, Losses: 4/20, Ties: 9/20
- Biggest wins: sarcoma (+31%), melanoma (+25%), glioblastoma (+14%), pancreatic (+11%)
- Biggest losses: EGFR lung cancer (-4.3%), liver cancer (-4.1%)
- Re-ranking latency: ~4s for 50 candidates on CPU

**Key findings:**

1. **Cross-encoders are not magic.** Both off-the-shelf and fine-tuned CEs degraded results when used as pure re-rankers. The standard "retrieve then re-rank with CE" pipeline assumes the CE has better relevance understanding than the retriever — ours doesn't, because binary training labels only teach disease-matching.

2. **Train/inference text alignment matters.** Initial evaluation used only title+conditions as trial text, but training data included brief_summary. Adding brief_summary via Elasticsearch lookup improved pure CE from -22.2% to -19.5% — meaningful but not the root cause.

3. **The CE works as a boost, not a replacement.** At 0.3 weight, it helps on hard queries where RRF candidates are close in score and CE breaks ties correctly. At higher weight, it overrides correct RRF rankings with crude disease-matching.

4. **Binary training labels are the bottleneck.** The CE achieved 0.992 NDCG@10 on its validation set — it's excellent at binary classification. But search ranking is a graded problem (score 0-3), and the CE can't distinguish between "somewhat relevant" (2) and "highly relevant" (3) trials.

5. **The right role for CE is as a feature, not a ranker.** The CE score should feed into LightGBM alongside metadata features (phase, enrollment, status, intervention match). Let the tree model learn when CE is informative vs when to trust RRF.

**Remaining limitations:**

1. **4-second CPU latency.** Scoring 50 candidate pairs through a 110M-param model takes ~4s on CPU. Batch inference on GPU or model distillation (MiniLM-sized CE) would be needed for production.
2. **Marginal improvement doesn't justify complexity.** +1.6% NDCG@5 with 4 losses out of 20 queries is borderline. The CE's value will be tested more rigorously as a LightGBM feature.
3. **No graded training signal.** Need 5K+ graded labels to train a CE that understands relevance gradations, not just disease matching.

**Data:** `data/evaluation/per_query_reranked_fine-tuned.json`. MLflow experiment: `trialmind-cross-encoder`.

---

## Week 5

### Decision 24: LightGBM LambdaRank over Pointwise or Pairwise Ranking

**Context:** Need to combine retrieval scores (BM25, semantic, RRF, cross-encoder) with trial metadata (phase, enrollment, status) into a final ranking. The cross-encoder alone degrades results (+1.6% with blending), so a learned combination is needed.

**Options:** Pointwise (predict absolute relevance), pairwise (predict which of two is more relevant), listwise/LambdaRank (optimize NDCG directly).

**Choice:** LightGBM with `objective=lambdarank`, optimizing NDCG directly.

**Why:** LambdaRank defines gradients based on the effect of swapping two documents on NDCG — so it directly optimizes the metric we care about. With graded labels (0-3), pointwise regression doesn't account for position (a score-3 trial at rank 10 matters less than at rank 1). Pairwise doesn't account for the magnitude of the ranking position change. LambdaRank combines both: it cares about which swaps improve NDCG most.

**Trade-off:** Requires grouped data (all candidates for a query must be in the same group). With only 20 queries, the model sees limited query diversity. Leave-one-query-out CV is the only viable evaluation strategy.

**At scale:** Same approach with more queries. LambdaRank scales well — it's used in production at scale by search engines and recommendation systems.

> **Interview answer:** "LambdaRank because it directly optimizes NDCG by weighting gradient updates based on how much a rank swap would change the metric. With graded relevance labels, this outperforms pointwise regression which ignores position."

---

### Decision 25: Fixed Hyperparameters over Optuna Tuning

**Context:** 20 queries is a small dataset for hyperparameter optimization. Tuning with Optuna would search over num_leaves, learning_rate, etc. across cross-validation folds.

**Options:** Optuna with 30 trials, grid search, fixed reasonable defaults.

**Choice:** Fixed defaults: `num_leaves=31, learning_rate=0.05, min_data_in_leaf=5, feature_fraction=0.8, bagging_fraction=0.8, num_boost_round=200`.

**Why:** Optimizing hyperparameters on 20 queries would overfit the hyperparameters to these specific queries. The CV folds would be 19 train + 1 test — Optuna would find settings that happen to work for these 20 queries but may not generalize. Standard LightGBM defaults are well-studied and robust. The model is shallow enough (31 leaves) that overfitting risk is lower than with neural models.

**Trade-off:** May leave some performance on the table. If we had 200+ queries, Optuna tuning would be justified.

**At scale:** Optuna tuning with proper nested cross-validation (inner CV for tuning, outer CV for evaluation) once there are 100+ labeled queries.

> **Interview answer:** "Fixed hyperparameters because 20 queries isn't enough for hyperparameter search — Optuna would overfit the settings to these specific queries. Standard LightGBM defaults are robust enough for a 31-leaf model."

---

### Decision 26: Leave-One-Query-Out CV over Random Split

**Context:** Need to evaluate LightGBM ranking quality without biasing the results. With only 20 queries, standard k-fold cross-validation would split candidates randomly, potentially leaking query-level patterns.

**Options:** Random 80/20 split, k-fold CV (random), leave-one-query-out (LOQO).

**Choice:** Leave-one-query-out — train on 19 queries, evaluate on 1, repeat 20 times.

**Why:** Learning-to-rank models must generalize to unseen QUERIES, not unseen candidates. Random splitting could put candidates from the same query in both train and test, which leaks the query's ranking signal. LOQO ensures each fold tests on a completely unseen query, giving an honest estimate of how the model handles new queries.

**Results:** LOQO gave NDCG@5=0.844±0.175, NDCG@10=0.840±0.140 — significantly better than hybrid-only (0.816/0.796) even with honest evaluation. The high std (0.175) reflects the 20-query variance, not model instability.

**Trade-off:** Only 1 test query per fold makes per-fold metrics noisy. With 200+ queries, 5-fold grouped CV would be more stable.

**At scale:** 5-fold grouped CV (group by query_id) with stratified sampling to balance relevance distributions across folds.

> **Interview answer:** "Leave-one-query-out because the model must generalize to unseen queries, not unseen candidates. Random splitting would leak query-level patterns and give optimistic results."

---

### Decision 27: CE Score as LightGBM Feature over Standalone Re-Ranker

**Context:** The cross-encoder achieves only +1.6% NDCG@5 as a standalone re-ranker (blended scoring). Its binary training labels limit it to disease-matching. But CE scores might add value as one signal among many in a learned combination.

**Choice:** Feed CE score as a feature into LightGBM alongside retrieval scores and metadata.

**Why:** LightGBM feature importance confirms CE score is the #2 most important feature (gain=243, behind RRF at 393). The tree model learns WHEN to trust CE — for example, when two candidates have similar RRF scores but different CE scores, the CE breaks the tie correctly. When CE disagrees with strong RRF signal, LightGBM learns to trust RRF. This is exactly what the fixed 0.7/0.3 blending couldn't do — it was a static weight, not a learned conditional.

**Result:** CV NDCG@5 went from 0.829 (CE blended) to 0.844 (LightGBM with CE as feature) — the CE contributes more value as a feature (+1.5%) than as a standalone re-ranker (+1.6%) because the tree model uses it adaptively rather than blindly.

> **Interview answer:** "As a LightGBM feature because the tree model learns conditional trust — use CE when retrieval scores are close, trust RRF when they're decisive. Static blending can't adapt like this."

---

### Evaluation: Full Ablation Table (2026-03-28)

Evaluated all 5 pipeline stages on 20 labeled queries (990 graded relevance labels, bootstrap 95% CIs).

| Method | NDCG@5 | NDCG@10 | MRR | Median Latency |
|---|---|---|---|---|
| BM25 only | 0.789±0.12 | 0.756±0.13 | 0.912±0.09 | 22ms |
| Semantic only | 0.703±0.12 | 0.700±0.10 | 0.807±0.12 | 37ms |
| Hybrid (BM25 + Semantic) | 0.816±0.10 | 0.796±0.08 | 0.917±0.08 | 72ms |
| + Cross-Encoder Re-ranking | 0.829±0.09 | 0.817±0.07 | 0.950±0.06 | 6295ms |
| + Metadata Blender | 0.980±0.02 | 0.921±0.03 | 1.000±0.00 | 6472ms |

**Honest numbers (leave-one-query-out CV for blender):** NDCG@5=0.844, NDCG@10=0.840. The ablation table's 0.980/0.921 for the blender is optimistic because the model was trained and evaluated on the same 20 queries.

**Key findings:**

1. **Each pipeline stage adds value.** BM25 (0.756) → +Semantic via hybrid (0.796, +5.3%) → +CE (0.817, +2.6%) → +LightGBM (0.840 CV, +2.8%). Total improvement: +11.1% NDCG@10 from BM25-only to full pipeline.
2. **RRF score is the strongest signal.** Feature importance gain=393. The hybrid fusion of BM25+semantic is the foundation — everything else refines it.
3. **CE redeemed as a feature.** As a standalone re-ranker, CE barely helped (+1.6%). As a LightGBM feature, it's the #2 most important signal (gain=243). The tree model learned to use CE adaptively.
4. **Metadata matters.** enrollment_log (gain=97), phase_numeric (gain=57), and is_recruiting (gain=39) all contribute. These are signals no retrieval model can learn from text alone.
5. **Latency dominated by CE.** The full pipeline takes ~6.5s on CPU, of which 95% is cross-encoder inference. GPU or model distillation needed for production.

**Data:** `docs/evaluation-report.md`, `docs/feature_importance.png`, `data/evaluation/ranking_features.csv`. MLflow experiments: `trialmind-ranker`, `trialmind-ablation`.

---

## Week 6

(Add decisions 24-26 after completing Week 6)

---

## Week 7-12

(Continue adding as you build)
