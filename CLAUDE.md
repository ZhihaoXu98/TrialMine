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
Last updated: 2026-03-25

Phase: 2 (Retrieval) — BM25 + semantic + hybrid search working, full stack running end-to-end

### What's working
- **Data pipeline**: downloads oncology trials from ClinicalTrials.gov API v2, parses, stores in SQLite
  - `scripts/download_data.py` → `data/trials.db` (140,723 trials)
- **BM25 search**: Elasticsearch index with 140,723 trials (596 MB), searchable via API
  - `scripts/build_index.py` → Elasticsearch `trials` index (requires Docker)
  - `src/TrialMine/retrieval/bm25.py` (ElasticsearchIndex — create, bulk index, search with field boosting, get_trial)
- **Semantic search**: BioLinkBERT-base embeddings + FAISS index
  - `scripts/build_index.py --skip-bm25` or `scripts/build_faiss.py` → `data/trial_embeddings.faiss` (412 MB) + `data/trial_embeddings.json`
  - `src/TrialMine/models/embeddings.py` (TrialEmbedder — mean-pooled BioLinkBERT)
  - `src/TrialMine/retrieval/semantic.py` (FAISSIndex — cosine similarity via IndexFlatIP)
  - Tested: clinical-language queries score ~0.90 cosine; patient-language queries ~0.87-0.89
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
- **MLflow tracking**: experiment `trialmind-retrieval` with baseline runs (bm25, semantic, hybrid)
  - Tracking URI: `sqlite:///mlflow.db`
  - UI: `make mlflow` → http://localhost:5001
  - `src/TrialMine/evaluation/metrics.py` — precision@k, recall@k, NDCG@k, MRR (ready for labelled data)

### Key evaluation findings
- BM25∩Semantic top-3 overlap: 0% across all 20 queries (completely disjoint results)
- Top-200 overlap: 1-16% depending on query type — signal is buried, not absent
- Semantic search has severe anisotropy: cosine range of only 0.047 across 1000 results
- 3 hub trials monopolize 33% of semantic result slots (embedding space collapse)
- The model understands paraphrase (30% overlap between patient/clinical phrasings) but can't rank
- Diagnosis: architecture sound, fixable via cross-encoder re-ranking (Phase 3) and contrastive fine-tuning (Phase 5)

### Key files/data (not in git)
- `data/trials.db` — SQLite with 140K parsed trials (912 MB)
- `data/trial_embeddings.faiss` — FAISS index (412 MB, rebuild with `scripts/build_index.py --skip-bm25`)
- `data/evaluation/method_comparison.csv` — comparison results from scripts/compare_methods.py
- `mlflow.db` — MLflow tracking database
- Elasticsearch `trials` index — requires `docker start es`

### What's next
- Phase 3: Cross-encoder re-ranking + LightGBM metadata blending (highest leverage — rescues noisy semantic results)
- Phase 4: LangGraph agents (query parsing, search orchestration)
- Phase 5: Fine-tune BioLinkBERT on patient-to-trial query pairs (fixes embedding collapse at source)
