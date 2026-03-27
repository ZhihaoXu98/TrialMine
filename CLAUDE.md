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
Last updated: 2026-03-26

Phase: 4 (Bi-encoder fine-tuned + evaluated) — BM25 + fine-tuned semantic + hybrid search working, FAISS indexes for both models, LLM-labeled evaluation complete

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

### Key files/data (not in git)
- `data/trials.db` — SQLite with 140K parsed trials (912 MB)
- `data/faiss_finetuned.index` + `.json` — fine-tuned FAISS index (412 MB, rebuild with `scripts/build_index.py --skip-bm25 --model fine-tuned`)
- `data/faiss_offshelf.index` + `.json` — off-the-shelf FAISS index (412 MB, rebuild with `scripts/build_index.py --skip-bm25 --model off-the-shelf`)
- `models/embeddings/fine-tuned/` — fine-tuned BioLinkBERT model (~430 MB)
- `data/evaluation/labeled_queries.jsonl` — 990 LLM-labeled (query, trial) pairs with 0-3 relevance scores (pooled from both models)
- `data/evaluation/method_comparison.csv` — comparison results from scripts/compare_methods.py
- `data/evaluation/per_query_*.json` — per-query metrics from compare_embeddings.py
- `data/training/train_pairs.jsonl` — 586K training triplets (1.0 GB)
- `data/training/val_pairs.jsonl` — 145K validation triplets (260 MB)
- `data/training/synthetic_queries.jsonl` — 1,500 Claude-generated patient queries (1.5 MB)
- `mlflow.db` — MLflow tracking database
- Elasticsearch `trials` index — requires `docker start es`
- `.env` — API keys (ANTHROPIC_API_KEY) — NEVER commit

### What's next
- Cross-encoder re-ranking + LightGBM metadata blending (highest leverage — re-ranks hybrid candidates)
- LangGraph agents (query parsing, search orchestration)
- Update FastAPI/Streamlit to use fine-tuned model path
