# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
TrialMine: ML-powered clinical trial search engine for oncology.
Patients describe their situation → AI finds relevant trials with explanations.

## Architecture
1. Data: ClinicalTrials.gov API v2 → parse → SQLite + Elasticsearch + FAISS
2. Retrieval: BM25 (Elasticsearch) + Semantic (FAISS with fine-tuned BioLinkBERT)
3. Re-ranking: Cross-encoder (fine-tuned BioLinkBERT) → LightGBM metadata blender
4. Agents: LangGraph — QueryParser → SearchOrchestrator (with tools)
5. Serving: FastAPI backend + Streamlit frontend
6. Monitoring: Prometheus + Grafana
7. Tracking: MLflow experiments

## Tech Stack
Python 3.11+, PyTorch, HuggingFace Transformers, sentence-transformers,
FastAPI, Streamlit, Elasticsearch 8.x, FAISS, LangGraph, langchain-anthropic,
LightGBM, MLflow, Optuna, SciSpacy, Docker, Prometheus + GitHub Actions

## ClinicalTrials.gov API v2
- Base: https://clinicaltrials.gov/api/v2/studies
- Pagination: pageToken (NOT page numbers), pageSize max 1000
- Key paths:
  protocolSection.identificationModule.nctId
  protocolSection.identificationModule.officialTitle
  protocolSection.descriptionModule.briefSummary
  protocolSection.conditionsModule.conditions
  protocolSection.armsInterventionsModule.interventions
  protocolSection.eligibilityModule.eligibilityCriteria
  protocolSection.eligibilityModule.minimumAge / maximumAge / sex
  protocolSection.designModule.phases
  protocolSection.statusModule.overallStatus
  protocolSection.designModule.enrollmentInfo.count
  protocolSection.contactsLocationsModule.locations
  protocolSection.sponsorCollaboratorsModule.leadSponsor.name
- Rate limit: add 0.5s delay between requests

## Coding Standards — IMPORTANT
- Type hints on ALL public functions
- Pydantic models for ALL data structures
- YAML configs for hyperparameters — NEVER hardcode magic numbers
- Structured JSON logging via Python logging module — NEVER use print()
- Error handling: NEVER let the API return 500. Always return structured error responses.
- Every function that calls an external service (API, DB, file) MUST have try/except
- Tests for all data parsing and feature engineering functions
- Docstrings on all public classes and functions

## File Structure
See README.md for overview. Key directories:
- src/TrialMine/ — all source code
- scripts/ — training, indexing, evaluation scripts
- configs/ — YAML configs
- data/ — raw + processed data (gitignored except evaluation/)
- models/ — trained models (gitignored)
- docs/ — architecture, design decisions, model cards

## Current State
Last updated: 2026-03-28

Phase: 6b (expanded training + fair eval v2) — full pipeline working: BM25 + semantic + hybrid + CE + LightGBM v2 (145 queries)

### What's working
- **Data pipeline**: downloads oncology trials from ClinicalTrials.gov API v2, parses, stores in SQLite
  - `scripts/download_data.py` → `data/trials.db` (140,723 trials)
- **BM25 search**: Elasticsearch index with 140,723 trials (596 MB), searchable via API
  - `scripts/build_index.py` → Elasticsearch `trials` index (requires Docker)
  - `src/TrialMine/retrieval/bm25.py` (ElasticsearchIndex — create, bulk index, search with field boosting, get_trial)
- **Semantic search**: Fine-tuned BioLinkBERT embeddings + FAISS index
  - `scripts/build_index.py --skip-bm25 --model fine-tuned` → `data/faiss_finetuned.index` (412 MB) + `data/faiss_finetuned.json`
  - `scripts/build_index.py --skip-bm25 --model off-the-shelf` → `data/faiss_offshelf.index` (412 MB) + `data/faiss_offshelf.json`
  - Model aliases: `--model off-the-shelf` → `michiyasunaga/BioLinkBERT-base`, `--model fine-tuned` → `models/embeddings/fine-tuned`
  - `src/TrialMine/models/embeddings.py` (TrialEmbedder — mean-pooled BioLinkBERT, explicit Transformer+Pooling for non-ST models to avoid SIGSEGV)
  - `src/TrialMine/retrieval/semantic.py` (FAISSIndex — cosine similarity via IndexFlatIP)
  - Fine-tuned: cosine spread 0.51–0.61 in top 5 (was 0.047 range pre-tuning), hub trial problem eliminated
- **Hybrid search**: Reciprocal Rank Fusion (RRF, k=60) combining BM25 + semantic
  - `src/TrialMine/retrieval/hybrid.py` (HybridRetriever — 200 candidates per method, RRF fusion, metadata enrichment)
  - Each result tagged with source: "bm25_only", "semantic_only", or "both"
- **FastAPI backend** (port 8000): POST /api/v1/search (method: bm25|semantic|hybrid), GET /api/v1/trial/{nct_id}, GET /health
  - `src/TrialMine/api/app.py` — FastAPI with CORS, ES + FAISS + embedder lifespan
  - `src/TrialMine/api/routes.py` — endpoint handlers with multi-method routing
  - `src/TrialMine/api/schemas.py` — Pydantic models (SearchRequest with method field, SearchResponse with search_method, TrialResult with source/ranks)
- **Streamlit UI** (port 8501): search bar, 3 example query buttons, result cards with status/phase badges, sidebar (method selector, status, phase, top_k), source tags per result
  - `src/TrialMine/ui/app.py` — communicates with FastAPI via httpx
- **Method comparison**: `scripts/compare_methods.py` — runs 20 oncology queries across all 3 methods, logs to MLflow, prints side-by-side top 3, overlap stats, saves CSV
- **Evaluation pipeline**: LLM-as-judge + automated comparison
  - `scripts/build_eval_dataset.py` — Claude Haiku labels 600 (query, trial) pairs on 0-3 relevance scale, supports `--limit N` preview and `--resume`
  - `scripts/compare_embeddings.py` — runs hybrid search with both embedding models, computes NDCG@5/10 + MRR, logs to MLflow
  - `data/evaluation/labeled_queries.jsonl` — 990 pooled labels (20 queries x ~50 unique trials from both models)
- **MLflow tracking**: experiment `trialmind-retrieval` with baseline + eval runs
  - Tracking URI: `sqlite:///mlflow.db`
  - UI: `make mlflow` → http://localhost:5001
  - `src/TrialMine/evaluation/metrics.py` — precision@k, recall@k, NDCG@k, MRR
- **Training data generation**: `scripts/generate_training_data.py` — 3-source pipeline for BioLinkBERT fine-tuning
  - Source 1: 242K metadata-derived (query, trial) pairs from conditions, interventions, phases
  - Source 2: 1,500 synthetic patient queries via Claude Haiku API (resumable, checkpointed)
  - Source 3: 730K hard negative triplets (same condition, different intervention preferred)
  - Stratified sampling across 23 cancer types, capped at 2000 trials/group
  - Config: `configs/training_data.yaml` (cancer type taxonomy, sampling caps, API settings)
  - Output: `data/training/train_pairs.jsonl` (586K triplets), `data/training/val_pairs.jsonl` (145K triplets)
  - Run: `make training-data` or `python scripts/generate_training_data.py [--skip-synthetic] [--dry-run] [--resume]`
- **Fine-tuned BioLinkBERT bi-encoder**: contrastive fine-tuning with MultipleNegativesRankingLoss
  - `scripts/finetune_embeddings.py` — training script (SentenceTransformerTrainer API)
  - `notebooks/finetune_biolinkbert.ipynb` — Colab notebook for GPU training
  - Config: `configs/training/embeddings.yaml` (lr=2e-5, batch=32, 3 epochs, fp16, MNRL scale=20)
  - Trained on A100 (288 min), 586K triplets, best model selected by NDCG@10
  - Output: `models/embeddings/fine-tuned/` (model + metadata.json with eval metrics)
  - Final eval: NDCG@10=0.492, MRR@10=0.426, Recall@10=0.700, Recall@1=0.300
- **Cross-encoder re-ranker**: fine-tuned BioLinkBERT cross-encoder with blended scoring
  - `src/TrialMine/models/cross_encoder.py` (CrossEncoderReranker — score, rerank with blended 0.7 RRF + 0.3 CE, rerank_with_timing)
  - `scripts/finetune_cross_encoder.py` — training script (CrossEncoderTrainer + BinaryCrossEntropyLoss)
  - `notebooks/finetune_cross_encoder.ipynb` — Colab notebook for T4 GPU training
  - `scripts/evaluate_cross_encoder.py` — evaluates cross-encoder on 20 labeled queries, compares to hybrid-only baseline
  - `scripts/demo_reranker.py` — before/after re-ranking comparison on 3 demo queries
  - Config: `configs/training/cross_encoder.yaml` (BioLinkBERT-base, lr=2e-5, batch=16, 1 epoch, BinaryCrossEntropyLoss)
  - Trained on T4 (Colab), 100K triplets → 200K binary pairs, early stopping, best model by NDCG@10
  - Output: `models/cross-encoder/fine-tuned/` (model + metadata.json)
  - Blended scoring: `0.7 * RRF_normalized + 0.3 * CE_sigmoid` (pure CE replacement hurts; blending preserves RRF quality)
  - MLflow experiment: `trialmind-cross-encoder`

### Key evaluation findings
- **Pre-fine-tuning (base BioLinkBERT):**
  - BM25∩Semantic top-3 overlap: 0% across all 20 queries (completely disjoint results)
  - Semantic search had severe anisotropy: cosine range of only 0.047 across 1000 results
  - 3 hub trials monopolized 33% of semantic result slots (embedding space collapse)
- **Post-fine-tuning:**
  - BM25∩Semantic top-3 overlap: 7% (4/20 queries share results) — up from 0%
  - Cosine spread in top 5: 0.10 (was 0.047 across 1000 results) — model now differentiates
  - Hub trial problem eliminated — every query returns distinct, relevant results
  - Semantic results are qualitatively relevant (BCG trials for BCG queries, EGFR trials for EGFR queries)
- **Embedding comparison (LLM-labeled, 990 pooled pairs, hybrid search):**
  - Pooled evaluation: top-30 from BOTH models labeled (eliminates bias toward fine-tuned)
  - 990 pairs: 210 overlap, 390 fine-tuned only, 390 off-the-shelf only
  - NDCG@5:  Off-the-shelf 0.577 → Fine-tuned 0.816 (+41.4%)
  - NDCG@10: Off-the-shelf 0.534 → Fine-tuned 0.796 (+49.1%)
  - MRR:     Off-the-shelf 0.917 = Fine-tuned 0.917 (BM25 drives first-result quality)
  - Fine-tuned wins on 19/20 queries (only loss: "sarcoma clinical trials for young adults")
  - Score dist: 0→22.7%, 1→17.4%, 2→13.9%, 3→46.0%
- **Cross-encoder re-ranking (blended scoring, 990 labeled pairs, hybrid search):**
  - Off-the-shelf ms-marco-MiniLM (pure CE) HURTS: NDCG@5=0.719 (-11.9%) — doesn't understand biomedical text
  - Fine-tuned BioLinkBERT CE (pure CE) also HURTS: NDCG@5=0.657 (-19.5%) — binary training labels don't teach graded relevance
  - Root cause: CE trained on binary (right/wrong disease) replaces useful RRF quality signals with blunt disease-matching
  - Fix: blended scoring (0.7 RRF + 0.3 CE) instead of pure CE replacement
  - NDCG@5:  Hybrid 0.816 → + CE blended 0.829 (+1.6%)
  - NDCG@10: Hybrid 0.796 → + CE blended 0.817 (+2.7%)
  - MRR:     Hybrid 0.917 → + CE blended 0.950 (+3.6%)
  - Wins 7, Losses 4, Ties 9 — biggest gains on hardest queries (sarcoma +31%, glioblastoma +14%)
  - Re-ranking latency: ~4s for 50 candidates on CPU
- **LightGBM metadata blender v2 (LambdaRank, 6,018 labeled pairs, 145 queries, 11 features):**
  - `src/TrialMine/models/ranker.py` (RankingBlender — compute_features, LightGBM predict, rerank)
  - `scripts/train_ranker.py` — loads 3 label files, feature engineering + LambdaRank LOOCV
  - `scripts/evaluate.py` — full ablation evaluation across all 5 pipeline stages
  - `scripts/expand_eval_data.py` — generates 75 training + 50 test queries, labels with Claude Haiku
  - Features: bm25_score, semantic_score, cross_encoder_score, rrf_score, phase_numeric, is_recruiting, is_active, enrollment_log, condition_exact_match, title_query_overlap, has_eligibility
  - Training data: 3 label files merged (20 + 50 + 75 queries = 145 queries, 6,018 pairs)
  - LOOCV on 145 queries: NDCG@5=0.843, NDCG@10=0.831
  - Feature importance: cross_encoder_score (1673) > rrf_score (1012) > phase_numeric (593) > title_query_overlap (557) > semantic_score (520) > bm25_score (516) > enrollment_log (487)
  - CE is now #1 feature (was #2 with 20 queries) — more data lets LightGBM trust CE scores
  - MLflow experiment: `trialmind-ranker`, ablation: `trialmind-ablation`
  - Output: `models/ranker/v2/model.lgb` + `metadata.json` (v1 preserved at `models/ranker/v1/`)
  - Note: FAISS + LightGBM OpenMP conflict requires `OMP_NUM_THREADS=1` on macOS
- **Full pipeline** (hybrid.py `full_pipeline` method):
  - BM25 (22ms) → Semantic (37ms) → RRF merge → CE scoring (4s) → LightGBM blending → top-k results
  - Returns timings dict: bm25_ms, semantic_ms, merge_ms, cross_encoder_ms, blender_ms, total_ms
  - API response includes optional `timings` field
- **Ablation evaluation**: `scripts/evaluate.py` + `docs/evaluation-report.md`
- **Fair held-out evaluation v2** (50 new test queries, 1,953 labels, LightGBM v2):
  - `scripts/build_fair_eval.py` — pools BM25+semantic+hybrid, labels with Claude Haiku, runs ablation
  - Test queries: 50 deliberately hard queries (rare cancers, comorbidities, health equity, complex patients)
  - NDCG@5: BM25 0.617 → Semantic 0.606 → Hybrid 0.636 → +CE 0.651 → +LGB 0.670
  - NDCG@5 increases monotonically — each stage adds value on unseen queries
  - MRR drops from CE (0.825) to LGB (0.806) — blender optimizes list-level, sometimes at cost of first result
  - Report: `docs/fair-evaluation-report.md`

### Key files/data (not in git)
- `data/trials.db` — SQLite with 140K parsed trials (912 MB)
- `data/faiss_finetuned.index` + `.json` — fine-tuned FAISS index (412 MB, rebuild with `scripts/build_index.py --skip-bm25 --model fine-tuned`)
- `data/faiss_offshelf.index` + `.json` — off-the-shelf FAISS index (412 MB, rebuild with `scripts/build_index.py --skip-bm25 --model off-the-shelf`)
- `models/embeddings/fine-tuned/` — fine-tuned BioLinkBERT bi-encoder (~430 MB)
- `models/cross-encoder/fine-tuned/` — fine-tuned BioLinkBERT cross-encoder (~430 MB)
- `models/ranker/v2/model.lgb` — LightGBM LambdaRank v2 model + metadata.json (trained on 145 queries)
- `models/ranker/v1/model.lgb` — LightGBM v1 model (trained on 20 queries, preserved)
- `data/evaluation/ranking_features_v2.csv` — 6,018 rows x 11 features for LightGBM v2 training
- `data/evaluation/ranking_features.csv` — 990 rows x 11 features (v1, preserved)
- `data/evaluation/labeled_queries.jsonl` — 990 LLM-labeled pairs (20 queries, IDs 0-19)
- `data/evaluation/test_labels.jsonl` — 2,044 LLM-labeled pairs (50 queries, IDs 100-149, now used for training)
- `data/evaluation/train_labels_extra.jsonl` — 2,984 LLM-labeled pairs (75 queries, IDs 200-274)
- `data/evaluation/test_labels_v2.jsonl` — 1,953 LLM-labeled pairs (50 NEW test queries, IDs 300-349)
- `data/evaluation/method_comparison.csv` — comparison results from scripts/compare_methods.py
- `data/evaluation/per_query_*.json` — per-query metrics from compare_embeddings.py
- `data/training/train_pairs.jsonl` — 586K training triplets (1.0 GB)
- `data/training/val_pairs.jsonl` — 145K validation triplets (260 MB)
- `data/training/synthetic_queries.jsonl` — 1,500 Claude-generated patient queries (1.5 MB)
- `mlflow.db` — MLflow tracking database
- Elasticsearch `trials` index — requires `docker start es`
- `.env` — API keys (ANTHROPIC_API_KEY) — NEVER commit

### What's next
- LangGraph agents (query parsing, search orchestration, result explanation)
- Update FastAPI/Streamlit to use fine-tuned models + full pipeline
- Optional: retrain CE on graded labels (6,018 pairs) instead of binary — could improve CE feature quality
- Optional: Optuna hyperparameter tuning for LightGBM (viable now with 145 queries)
