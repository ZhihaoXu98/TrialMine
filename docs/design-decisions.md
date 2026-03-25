# TrialMind Design Decisions Log

This document records every significant technical decision made during the project.
Each entry explains what we chose, why, and what we'd do differently at scale.

---

## Week 1

### Decision 1: ClinicalTrials.gov API v2 over XML Bulk Download
**Context:** Need to ingest clinical trial data. Two options: REST API v2 (JSON, filtered queries) or bulk XML download (full database dump, ~3GB compressed).
**Options:** API v2 (paginated JSON, server-side filtering) vs bulk XML (single download, local filtering).
**Choice:** API v2
**Why:** JSON is directly parseable into Pydantic models — no XML→dict conversion. Server-side `query.cond=cancer` filters to ~140K oncology trials at source, avoiding downloading all 500K+ trials. Pagination with `pageToken` supports resume on failure. Rate limit (0.5s delay) keeps us well within acceptable ingestion time.
**Trade-off:** Full download takes 30-60 min vs ~5 min for bulk XML. Cannot do complex cross-field filters server-side (e.g., "oncology AND recruiting AND Phase 3").
**At scale:** Bulk download for initial load + incremental API polling (filter by `lastUpdatePostDate`) for daily freshness. Would add a CDC (change data capture) pipeline.
**Interview answer:** "API v2 because it filters at source — I only download the 140K oncology trials I need instead of 500K+, and JSON maps directly to my Pydantic models."

### Decision 2: SQLite over PostgreSQL
**Context:** Need persistent storage for 140K parsed trials between pipeline stages (download → parse → index).
**Options:** PostgreSQL (full RDBMS), SQLite (embedded), MongoDB (document store), flat files (Parquet/JSON).
**Choice:** SQLite
**Why:** Zero config, single portable file (912 MB), sufficient read throughput for batch indexing. SQL for ad-hoc debugging (`SELECT COUNT(*) FROM trials WHERE phase='Phase 3'`). No server process to manage. The data flows one direction (write once during ingestion, read during indexing) — no concurrent write pressure.
**Trade-off:** No concurrent writes, single-machine only, no built-in full-text search. Not suitable as a serving database for the API.
**At scale:** PostgreSQL with read replicas, or a managed database (Cloud SQL, RDS). Would also consider DuckDB for analytical queries over trial metadata.
**Interview answer:** "SQLite because this is a batch pipeline — write once, read many. Zero operational overhead, and the 140K-row dataset fits in a single file."

### Decision 3: Elasticsearch for BM25 Search
**Context:** Need full-text search with relevance ranking over 140K clinical trial documents.
**Options:** Elasticsearch (BM25 + field boosting), PostgreSQL full-text search (ts_vector), Whoosh (pure Python), Tantivy/Meilisearch.
**Choice:** Elasticsearch 8.x
**Why:** Industry-standard BM25 with built-in field boosting (title 3x, conditions 2x), custom analysers (English stemming + stop words), and term-level filters (phase, status) in a single query. Handles 140K docs in a single shard with ~20-35ms query latency. Docker setup is one command. Scales to millions of documents without architecture changes.
**Trade-off:** Adds a Docker service (~512MB-1GB memory). Requires index rebuilding when mappings change. JVM-based — heavier than Tantivy for the same workload.
**At scale:** Managed Elasticsearch (Elastic Cloud) or OpenSearch. Multi-shard index with replicas for throughput. Would add synonym expansion (e.g., "chemo" → "chemotherapy") at the analyser level.
**Interview answer:** "Elasticsearch because BM25 with field boosting and custom analysers gives me relevance tuning out of the box — title matches rank higher than body matches, and English stemming handles morphological variants."

### Decision 4: FastAPI over Flask
**Context:** Need an HTTP API framework to serve search results to the Streamlit frontend.
**Options:** Flask (mature, synchronous), FastAPI (modern, async, typed), Django REST Framework (batteries-included).
**Choice:** FastAPI
**Why:** Pydantic models for request/response validation (shared with the data pipeline), auto-generated OpenAPI docs at `/docs`, native async support for concurrent search requests, type hints that match the project's coding standards. Lifespan context manager cleanly handles Elasticsearch + FAISS + embedder startup/shutdown.
**Trade-off:** Smaller ecosystem than Flask/Django. Async isn't strictly necessary at current scale (single user), but doesn't hurt.
**At scale:** Same choice — FastAPI with multiple uvicorn workers behind a load balancer. Would add rate limiting middleware, API key auth, and response caching for frequent queries.
**Interview answer:** "FastAPI because Pydantic validation, auto-generated docs, and async support align with the project's type-first approach — and the same Pydantic models validate both API boundaries and internal data flow."

### Decision 5: Keep Trials with Missing Fields
**Context:** ~5-10% of trials from ClinicalTrials.gov have sparse metadata — missing summary, eligibility criteria, or intervention details.
**Options:** Drop incomplete trials, require a minimum field count, keep everything with only NCT ID + title required.
**Choice:** Require NCT ID + title only, allow all other fields to be None.
**Why:** Missing data does not mean irrelevant trial. A Phase 3 breast cancer trial with no eligibility text is still a valid search result — a patient should know it exists. BM25 naturally ranks sparse trials lower (fewer matching fields). Semantic search embeds whatever text is available. Dropping trials risks excluding rare-cancer or newly-posted studies.
**Trade-off:** Sparse trials may rank unexpectedly high if their short text happens to match well. Some UI cards look empty.
**At scale:** Add a data quality score per trial (0-1, based on field completeness) and surface it in the UI. Use the score as a feature in the LightGBM re-ranker to soft-penalise incomplete trials rather than hard-filtering them.
**Interview answer:** "Keep everything — missing metadata is not the same as irrelevant. BM25 and the re-ranker naturally downweight sparse trials, and filtering would silently drop rare-cancer studies that patients need most."

---

## Week 2

### Decision 6: BioLinkBERT over PubMedBERT / General Models
**Context:** Need a biomedical embedding model for semantic search over clinical trial text.
**Options:** all-MiniLM-L6-v2 (general, 384d), PubMedBERT (biomedical, 768d), BioLinkBERT (biomedical + citation structure, 768d).
**Choice:** `michiyasunaga/BioLinkBERT-base` (768-dim) via sentence-transformers with mean pooling.
**Why:** Pre-trained on PubMed WITH citation link structure — understands which medical concepts are related, not just which words co-occur. 768 dimensions captures more biomedical nuance than 384. Citation-aware pre-training means "glioblastoma" and "temozolomide" are linked through the papers that study them, not just co-occurrence statistics.
**Trade-off:** Slower than MiniLM (~50ms vs ~15ms per query). Not trained for retrieval — the model encodes token-level biomedical knowledge, not query-document similarity. Our evaluation confirmed this: cosine score range across 1000 results is only 0.047 (severe anisotropy). The model understands biomedical relationships but can't rank them for retrieval without fine-tuning.
**At scale:** Fine-tune on patient-to-trial query pairs with contrastive loss (Phase 5). The biomedical knowledge is the right foundation — the ranking behaviour is what needs training.
**Interview answer:** "BioLinkBERT because it understands biomedical concept relationships through citation-aware pre-training. General models miss domain-specific connections like drug-indication mappings."

### Decision 7: RRF over Linear Score Combination
**Context:** Need to merge BM25 and semantic ranked lists into one hybrid result set.
**Options:** Linear combination (alpha * bm25_norm + beta * semantic), RRF (rank-based), learned merge.
**Choice:** RRF with k=60 (standard constant from Cormack et al., 2009).
**Why:** BM25 scores (5–50) and cosine similarity (0.85–0.90) are on completely different scales. Linear combination requires normalising both to [0,1], which is fragile — BM25 score distributions shift with query length. RRF uses ranks, which are always comparable. Parameter-free (k=60 is the literature default) and robust.
**Trade-off:** Treats both retrievers equally. Cannot express "trust BM25 more" without a weighting parameter. The Week 4 LightGBM blender adds learned weighting on top.
**At scale:** After fine-tuning embeddings, consider weighted RRF or learned score interpolation. The cross-encoder re-ranker (Phase 3) compensates for noisy fusion regardless.
**Interview answer:** "RRF because it's score-scale independent — no normalisation needed, and robust even when one retriever is weaker than the other."

### Decision 8: IndexFlatIP (Exact) over Approximate FAISS
**Context:** Need nearest-neighbour search over 140K 768-dim trial embeddings.
**Choice:** FAISS `IndexFlatIP` with L2-normalised vectors (inner product = cosine similarity).
**Why:** At 140K vectors, exact search takes 10–50ms — fast enough for interactive use. Approximate indexes (IVF, HNSW) add complexity, tuning parameters (nprobe, ef), and recall loss without meaningful benefit at this scale. Memory = 140K x 768 x 4 bytes = 412 MB, fits comfortably in RAM.
**Trade-off:** Brute-force linear scan. Latency scales linearly with corpus size.
**At scale:** Switch to IndexIVFFlat at 1M+ vectors, IndexIVFPQ at 100M+ (trades ~5% recall for 10-50x speedup). Would also add GPU acceleration via faiss-gpu.
**Interview answer:** "IndexFlatIP because exact search is fast enough at 140K vectors and eliminates approximation error. I'd switch to IVF at the million-vector mark."

### Decision 9: Equal BM25/Semantic Weight in RRF
**Context:** Each retriever contributes 1/(60 + rank) per document in RRF. Need to decide whether to weight one higher.
**Choice:** Equal weighting — both methods contribute identically.
**Why:** Without labelled relevance data, we can't empirically determine which retriever is more trustworthy. Our 20-query evaluation shows 0% top-3 overlap (the methods find completely different trials), so both contribute unique signal. Equal is the safest uninformed default.
**Trade-off:** Our evaluation shows BM25 is currently the stronger retriever (60 unique trials vs 38 for semantic across 20 queries). Weighting BM25 higher would improve results today, but would hide semantic improvements after fine-tuning.
**At scale:** The Week 4 LightGBM blender learns optimal weights from labelled data. Ablation table will reveal per-method contribution.
**Interview answer:** "Equal weights because without relevance labels, asymmetric weighting is premature optimisation. The re-ranker in the next phase learns the right balance from data."

### Evaluation: Hybrid Retrieval Comparison (2026-03-25)

Ran 20 oncology test queries across BM25, semantic, and hybrid search. Queries range from clinical ("triple negative breast cancer neoadjuvant") to patient-language ("clinical trial for glioblastoma that has come back") to misspelled ("melanomt with checkpoint inhibitors"). Results logged as MLflow baseline runs in experiment `trialmind-retrieval`.

**Key metrics:**

| Metric | BM25 | Semantic | Hybrid |
|---|---|---|---|
| Avg latency | 25 ms | 50 ms | 77 ms |
| Unique trials (across 20 queries, top 3) | 60 | 38 | 57 |
| Top-3 BM25∩Semantic overlap | 0% | 0% | — |

**Finding 1: Zero top-3 overlap, but nonzero top-200 overlap.**
The methods return completely disjoint top-3 results. But at the top-200 candidate level, overlap ranges from 1% to 16%:

| Query type | Top-200 overlap | Example |
|---|---|---|
| Broad clinical terms | 16% (31/200) | "immunotherapy for non-small cell lung cancer" |
| Specific + clinical | 8–9% (17-18/200) | "breast cancer hormone receptor positive phase 3" |
| Patient language | 1–2% (2-4/200) | "glioblastoma that has come back" |

Relevant trials ARE in the semantic candidate pool — they're just buried at rank 30-100+ instead of surfacing at rank 1-3. RRF confirms this: 52% of hybrid top-3 results are tagged `source: "both"`.

**Finding 2: Semantic search has severe embedding space collapse.**
3 generic trials occupy 33% of all 60 semantic result slots ("Mebendazole" in 8/20 queries, "R2 Follicular Lymphoma" in 6/20, "Efficacy Prediction Model" in 6/20). Cosine score range across 1000 results is only 0.047 (0.8988 to 0.8517) — the model can barely distinguish relevant from irrelevant.

**Finding 3: The model understands paraphrase but can't retrieve.**
"come back" and "recurrent glioblastoma" map to the same embedding region (30% semantic overlap), but BM25 can't bridge them (0% overlap). The semantic model has the right knowledge — it just can't surface it through the anisotropic embedding space.

**Diagnosis: architecture sound, embeddings need work. Fixable via:**
1. **Cross-encoder re-ranking (Phase 3)** — highest leverage, works immediately, no training needed
2. **Contrastive fine-tuning (Phase 5)** — fixes the root cause by spreading the embedding space
3. **Embedding whitening** — quick experiment, zero retraining, 5-15% expected improvement
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
**Interview answer:** "Three sources because each teaches a different skill: metadata pairs teach concept association, synthetic queries bridge the vocabulary gap between patients and clinicians, and hard negatives force fine-grained discrimination between similar trials."

### Decision 11: Stratified Sampling by Cancer Type
**Context:** 140K trials are heavily skewed — breast cancer has 15K trials, mesothelioma has 292. Need balanced training data.
**Options:** Random sampling (simple but biased), stratified sampling with caps, upsampling rare types.
**Choice:** Stratified sampling with 2000-trial cap per cancer group. 23 cancer type groups defined by keyword matching on `trial.conditions`, plus an "other" catch-all.
**Why:** Without caps, breast cancer (15K) and lung cancer (11K) would dominate training — the model learns "breast cancer" embeddings well but fails on neuroblastoma (361 trials). A 2000-trial cap means rare cancer groups keep all their trials while common groups are downsampled. This produces 40K sampled trials across 23 groups.
**Trade-off:** The "other" bucket (47K trials) includes basket trials, supportive care, and non-oncology studies that leaked through. Capping it at 2000 loses some diversity. The keyword-based taxonomy misclassifies trials with unusual condition strings.
**At scale:** Use MeSH terms or a medical NER model for cancer type classification instead of keyword matching. Consider curriculum learning: start training on easy (cross-cancer) negatives, then switch to hard (within-cancer) negatives.
**Interview answer:** "Stratified sampling with caps because training data distribution directly controls what the model learns. A model trained on 90% breast cancer trials won't help a mesothelioma patient."

### Decision 12: Claude Haiku for Synthetic Patient Queries
**Context:** Need patient-language queries to bridge the vocabulary gap (our evaluation showed BM25 and semantic have 0% overlap on patient-language queries like "glioblastoma that has come back"). Template-based generation is free but robotic.
**Options:** Templates only (free, robotic), local LLM via Ollama (free, medium quality), Claude Haiku API (~$1-2 for 1,500 queries).
**Choice:** Claude Haiku API (`claude-haiku-4-5-20251001`) for 1,500 queries.
**Why:** Quality gap is large — Claude produces queries like "My daughter has a benign ovarian tumor and needs surgery — are there any new treatment options?" vs templates producing "I was diagnosed with ovarian cancer and looking for treatment options." The personal context, emotional tone, and natural phrasing from Claude better represent how real patients search. At $0.001 per query ($1.50 total), the cost is negligible compared to the quality improvement. The script supports `--skip-synthetic` for free-only generation and `--resume` for interrupted API calls.
**Trade-off:** Only 1,500 queries (0.6% of training data) — limited impact on overall training. Requires API key. Non-reproducible (different outputs each run due to temperature).
**At scale:** Generate 10K-50K synthetic queries using Claude. Add few-shot examples from real patient forum posts (Reddit r/cancer, HealthUnlocked) to improve prompt quality. Consider fine-tuning a smaller model on the Claude outputs for cost-free generation.
**Interview answer:** "Claude Haiku because the cost ($1.50) is trivial and the quality gap over templates is dramatic for patient-language queries — and that's exactly the vocabulary gap our embedding model needs to learn."

### Decision 13: Hard Negatives via Condition Keyword Overlap
**Context:** Contrastive learning needs hard negatives — trials that are similar enough to confuse the model but are not the correct match. In-batch negatives (random trials from other queries in the same batch) are too easy.
**Options:** Random negatives (easy), BM25-retrieved negatives (requires Elasticsearch), embedding-based negatives (requires encoding all trials), keyword overlap (simple, in-memory).
**Choice:** Keyword overlap on condition strings, preferring trials with different interventions. Build an in-memory condition-word → NCT ID index, find trials sharing keywords, partition into hard (different intervention) and easy (same intervention) candidates, sample 3 negatives per positive.
**Why:** Simple and effective. A breast cancer trial about pembrolizumab vs a breast cancer trial about tamoxifen is the exact kind of distinction the model needs to learn. No external dependencies (no Elasticsearch, no GPU for encoding). The condition index builds in seconds over 40K sampled trials. 730K triplets generated in ~12 minutes.
**Trade-off:** Only uses condition text, not semantic similarity — misses some hard negatives that share interventions but different conditions. The keyword tokenisation is naive (split on spaces, skip <3 chars). 180 pairs (0.07%) found no candidates at all.
**At scale:** Mine negatives using the current model's embeddings — encode all trials, for each positive find the k nearest neighbours that aren't relevant. This produces the hardest possible negatives. Update the negative set every training epoch (dynamic hard negative mining).
**Interview answer:** "Keyword overlap because it's simple, fast, and targets exactly what we need: same cancer, different drug. The 730K triplets generated in 12 minutes with no external dependencies — embedding-based mining would be better but requires encoding 140K trials first."

---

## Week 4

(Add decisions 13-16 after completing Week 4)

---

## Week 5

(Add decisions 17-19 after completing Week 5)

---

## Week 6

(Add decisions 20-23 after completing Week 6)

---

## Week 7-12

(Continue adding as you build)
