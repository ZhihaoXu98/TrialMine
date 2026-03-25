# TrialMind Design Decisions Log

This document records every significant technical decision made during the project.
Each entry explains what we chose, why, and what we'd do differently at scale.

---

## Week 1

### Decision 1: ClinicalTrials.gov API v2 over XML Bulk Download
**Context:** Needed to get trial data. Two options: REST API (JSON) or bulk XML download.
**Choice:** API v2
**Why:** JSON easier to parse, can filter to oncology at source (80K vs 500K), pagination with resume support.
**Trade-off:** Slower download (30-60 min). Worth it for cleaner data pipeline.
**At scale:** Would use bulk download + incremental API updates for freshness.

### Decision 2: SQLite over PostgreSQL
**Context:** Need storage for 80K parsed trials.
**Choice:** SQLite
**Why:** Zero config, single portable file, sufficient for 80K rows, SQL for debugging.
**Trade-off:** No concurrent writes, single-machine only.
**At scale:** PostgreSQL or a managed database (Cloud SQL, RDS).

### Decision 3: Elasticsearch for BM25 Search
**Context:** Need text search with relevance ranking.
**Choice:** Elasticsearch 8.x
**Why:** Industry standard, built-in BM25 with field boosting, Docker setup trivial, scales to millions.
**Trade-off:** Adds a Docker service (~512MB-1GB memory).
**At scale:** Managed Elasticsearch (Elastic Cloud) or OpenSearch with index sharding.

### Decision 4: FastAPI over Flask
**Context:** Need HTTP API framework.
**Choice:** FastAPI
**Why:** Pydantic validation, auto-generated OpenAPI docs, async support, type hints.
**At scale:** Same choice — FastAPI scales well with uvicorn workers.

### Decision 5: Keep Trials with Missing Fields
**Context:** ~5-10% of trials have sparse metadata.
**Choice:** Require title + NCT ID only, allow all else to be None.
**Why:** Missing data ≠ irrelevant trial. Search ranks sparse trials lower naturally.
**At scale:** Add data quality scores per trial, surface them in UI.

---

## Week 2

### Decision 6: BioLinkBERT-base for Semantic Embeddings
**Context:** Need a dense encoder for semantic search over clinical trial text.
**Choice:** `michiyasunaga/BioLinkBERT-base` (768-dim) via sentence-transformers with mean pooling.
**Why:** Pre-trained on PubMed with citation-informed objectives — captures biomedical entity relationships better than general BERT. 768-dim keeps FAISS index manageable (412 MB for 140K trials).
**Trade-off:** Not trained for retrieval or sentence similarity — the model encodes token-level biomedical knowledge, not query-document relevance.
**At scale:** Fine-tune on patient-to-trial query pairs (Phase 5), or switch to a retrieval-tuned model like BiomedBERT-mnli or a contrastive-trained variant.

### Decision 7: FAISS IndexFlatIP for Semantic Search
**Context:** Need nearest-neighbor search over 140K 768-dim trial embeddings.
**Choice:** FAISS `IndexFlatIP` with L2-normalised vectors (inner product = cosine similarity).
**Why:** Exact search, no approximation error, fast enough at 140K scale (~50 ms), simple to build and debug.
**Trade-off:** Brute-force — linear scan, no quantization. Memory = 140K × 768 × 4 bytes ≈ 412 MB.
**At scale:** Switch to `IndexIVFPQ` or `IndexHNSWFlat` for sub-linear search at millions of vectors. Would also add GPU acceleration.

### Decision 8: Reciprocal Rank Fusion (RRF) over Score Interpolation
**Context:** Need to merge BM25 and semantic ranked lists into one hybrid result set.
**Choice:** RRF with k=60 (standard constant from Cormack et al., 2009).
**Why:** Rank-based fusion is robust to incompatible score scales (BM25 scores are unbounded TF-IDF; semantic scores are 0–1 cosine). No tuning required for weight parameters.
**Trade-off:** Treats all rank positions equally across methods — a strong BM25 #1 is weighted the same as a weak semantic #1. Cannot express "trust BM25 more."
**At scale:** After fine-tuning the semantic model, consider weighted RRF or learned score interpolation (α × BM25_norm + (1-α) × semantic). The cross-encoder re-ranker (Phase 3) will compensate for noisy fusion regardless.

### Decision 9: Findings from the Hybrid Retrieval Comparison (2026-03-25)
**Context:** Ran 20 oncology test queries across BM25, semantic, and hybrid search to evaluate retrieval before re-ranking or fine-tuning. Queries range from clinical ("triple negative breast cancer neoadjuvant") to patient-language ("clinical trial for glioblastoma that has come back") to misspelled ("melanomt with checkpoint inhibitors").

**Key findings:**

| Metric | Value |
|---|---|
| BM25 ∩ Semantic overlap (top 3) | **0/3 for all 20 queries (0%)** |
| BM25 avg latency | 20–35 ms |
| Semantic avg latency | 48–54 ms |
| Hybrid avg latency | 80–130 ms |

**Finding 1: Zero top-3 overlap, but nonzero top-200 overlap — the signal is buried, not absent.**
BM25 and semantic return completely disjoint top-3 results across all 20 queries (0% overlap). But at the top-200 candidate level, overlap ranges from 1% to 16% depending on query type:

| Query type | Top-200 overlap | Example |
|---|---|---|
| Broad clinical terms | 16% (31/200) | "immunotherapy for non-small cell lung cancer" |
| Specific + clinical | 8–9% (17-18/200) | "breast cancer hormone receptor positive phase 3", "CAR-T pediatric leukemia" |
| Patient language | 1–2% (2-4/200) | "glioblastoma that has come back", "neuroblastoma high risk children" |

This means the semantic model captures *some* signal — relevant trials appear in its top 200 — but they're buried at rank 30-100+ instead of surfacing at rank 1-3. The RRF fusion confirms this: 52% of hybrid top-3 results are tagged `source: "both"`, meaning RRF successfully promotes trials found in both candidate pools.

**Finding 2: BM25 is the stronger retriever at this stage.**
BM25 consistently returns on-topic trials for specific clinical terms (drug names, disease subtypes, biomarkers). Examples: "ibrutinib" → ibrutinib CLL trials, "neuroblastoma high risk children" → neuroblastoma pediatric protocols, "BCG unresponsive" → BCG-refractory bladder cancer trials. BM25's field boosting (title 3×, conditions 2×) and English stemming handle clinical vocabulary well.

**Finding 3: Semantic search has severe embedding space collapse (anisotropy).**
The same handful of generic trials dominate semantic results across many unrelated queries (NCT03925662 "Mebendazole Adjuvant Therapy" appears in 8/20, NCT03715309 "R2 in Follicular Lymphoma" in 6/20, NCT05515796 "Efficacy Prediction Model" in 6/20 — these 3 trials occupy 33% of all 60 semantic result slots).

Root cause confirmed by cosine score distribution for "clinical trial for glioblastoma that has come back":

| Rank | Cosine score |
|---|---|
| #1 | 0.8988 |
| #100 | 0.8689 |
| #1000 | 0.8517 |
| **Range (#1 – #1000)** | **0.047** |

A spread of only 0.047 across 1000 results means the model cannot meaningfully distinguish relevant from irrelevant. All trial embeddings cluster in a narrow cone (classic BERT anisotropy), and "hub" trials near the centroid win every query.

**Finding 4: The model understands paraphrase but can't retrieve.**
Critical sanity check comparing "clinical trial for glioblastoma that has come back" (patient) vs "recurrent glioblastoma" (clinical):

| Test | Result |
|---|---|
| Semantic top-200 overlap between both phrasings | **59/200 (30%)** — the model knows they mean the same thing |
| BM25 top-200 overlap between both phrasings | **0/200** — BM25 can't bridge "come back" → "recurrent" |
| Semantic top 5 for "recurrent glioblastoma" | #1 Basal Cell Carcinoma, #2 Sarcoma Database, #3 Mebendazole, #4 Cisplatin Nephrotoxicity, **#5 Tofacitinib in Recurrent GBM** |
| BM25 top 5 for "recurrent glioblastoma" | **5/5 are actual recurrent GBM trials** |

The semantic model maps both patient and clinical phrasings to the same region of embedding space (30% overlap) — it understands they're semantically similar. But that entire region is dominated by hub trials, so neither phrasing produces useful top results. The first actually relevant result ("Tofacitinib in Recurrent GBM") appears at semantic rank #5 for the clinical phrasing, with a score of 0.8682 vs #1's 0.8825 — a gap of only 0.014.

**Diagnosis: fixable. The architecture is sound, the embeddings need work.**

The evidence shows: (a) relevant trials ARE in the top-200 candidate pools, (b) the model does understand semantic similarity between phrasings, (c) but the compressed score distribution buries good results under hub noise. This is a ranking quality problem, not a fundamental architecture problem.

**Fix paths, ordered by impact and urgency:**

1. **Cross-encoder re-ranking (Phase 3) — immediate, highest leverage.** A cross-encoder sees the full (query, trial_text) pair and scores from scratch — it doesn't suffer from embedding collapse. Since relevant trials ARE in the top-200 candidates, re-ranking should surface them. Off-the-shelf `cross-encoder/ms-marco-MiniLM-L-6-v2` can work without any training data. Expected impact: transforms hybrid from "BM25 with noise" into "BM25 + semantic diversity, properly ranked."

2. **Contrastive fine-tuning (Phase 5) — fixes the root cause.** Train BioLinkBERT with contrastive loss (e.g., InfoNCE) to spread the embedding space. Can bootstrap training data using BM25 results as pseudo-positives (GPL approach) or generate synthetic patient queries from trial text via LLM. Target: expand the 0.047 score range to 0.3+, push top-200 overlap from 1-16% to 30-60%.

3. **Embedding whitening — quick experiment.** Center embeddings (subtract corpus mean) and apply PCA whitening to the existing FAISS vectors. Literature shows 5-15% retrieval improvement on anisotropic embeddings. Zero retraining, just post-processing — worth trying as a baseline before fine-tuning.

4. **Model swap — moderate effort fallback.** If fine-tuning BioLinkBERT is slow to converge, consider `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical + retrieval-trained) or `sentence-transformers/all-MiniLM-L6-v2` (general retrieval). Requires rebuilding the FAISS index (~30 min).

**Data:** Full results in `data/evaluation/method_comparison.csv` (180 rows, 20 queries × 3 methods × top 3).

---

## Week 3

(Add decisions 10-12 after completing Week 3)

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
