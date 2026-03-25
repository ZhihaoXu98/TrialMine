# TrialMine — Design Decisions

This document captures the key architectural and technical decisions made during the design of TrialMine, along with the rationale behind each.

---

## 1. Hybrid Retrieval: BM25 + Semantic Search

**Decision:** Use a two-stage retrieval system combining Elasticsearch BM25 (lexical) with FAISS-based semantic search (dense vectors), fused via Reciprocal Rank Fusion (RRF).

**Rationale:**
- BM25 excels at exact keyword matching (drug names, NCT IDs, specific biomarkers) — critical in the clinical domain where precise terminology matters.
- Semantic search captures meaning-level similarity (e.g., "lung cancer immunotherapy" matching a trial titled "PD-L1 Checkpoint Inhibitor for NSCLC") where lexical overlap is low.
- Neither approach alone is sufficient. RRF fusion (with smoothing constant k=60) provides a principled way to merge two ranked lists without requiring score calibration between the systems.

**Configuration:** `configs/development.yaml` defines `bm25_top_k: 100` and `semantic_top_k: 100`, meaning each retriever independently returns its top 100 candidates before fusion.

---

## 2. Retrieve-Then-Rerank Funnel

**Decision:** Structure the pipeline as a multi-stage funnel: BM25/Semantic (100 each) → Hybrid Fusion → Cross-Encoder Rerank (top 20) → LightGBM Metadata Blender (final top 10).

**Rationale:**
- Cross-encoders (which jointly encode query + document) are far more accurate than bi-encoders but orders of magnitude slower — they cannot score thousands of candidates.
- The funnel lets cheap retrievers cast a wide net, then progressively more expensive and accurate models narrow the set.
- This is a standard pattern in production search/recommendation systems (e.g., Google, Spotify, LinkedIn).

**Trade-off:** More pipeline stages add latency and complexity. The chosen stage sizes (100 → 20 → 10) balance recall against response time for an interactive application.
√
---

## 3. LightGBM Metadata Blender as Final Ranker

**Decision:** The final ranking stage uses a LightGBM model that combines cross-encoder relevance scores with structured metadata features (phase, enrollment, status, eligibility match scores).

**Rationale:**
- Neural models (cross-encoder) capture text relevance well, but cannot easily incorporate structured signals like "this trial is recruiting" or "patient age falls within the trial's age range."
- LightGBM is fast at inference, handles heterogeneous features natively (continuous scores + categorical metadata), and is easy to train with Optuna hyperparameter tuning.
- This separation of concerns (neural model for text relevance, gradient-boosted tree for feature combination) is easier to debug and iterate on than a single end-to-end model.

**Feature groups fed to LightGBM:**
- Retrieval signals: BM25 rank/score, semantic rank/score, RRF score
- Cross-encoder: relevance score
- Trial metadata: phase, status, enrollment size, sponsor type
- Eligibility match: age match, sex match, concept overlap

---

## 4. BioNLP-Specific Models (SciBERT → BioLinkBERT)

**Decision:** Use domain-specific transformer models rather than general-purpose ones. Starting with `allenai/scibert_scivocab_uncased` for embeddings, with a plan to fine-tune and switch to BioLinkBERT. Cross-encoder starts with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Rationale:**
- Clinical trial text is highly specialized (biomarkers, drug mechanisms, staging criteria). General-purpose models (e.g., all-MiniLM-L6-v2) underperform on biomedical vocabulary.
- SciBERT provides a strong baseline with biomedical pre-training. BioLinkBERT adds link-based pre-training over PubMed citation graphs, which captures relational knowledge between medical concepts.
- The cross-encoder starts general-purpose (MS MARCO trained) because it will be fine-tuned on domain-specific relevance judgments.

---

## 5. Agentic Query Understanding with LangGraph

**Decision:** Use a LangGraph-based agent pipeline (QueryParser → SearchOrchestrator) with Claude (via `langchain-anthropic`) as the LLM backbone.

**Rationale:**
- Patient descriptions are unstructured and ambiguous ("I'm a 55-year-old woman with stage IIIA NSCLC, EGFR+, tried carboplatin"). A simple keyword search cannot extract structured clinical intent from this.
- The QueryParser agent uses an LLM to decompose patient descriptions into structured fields: cancer type, stage, biomarkers, age, sex, location, and a reformulated search query.
- The SearchOrchestrator runs a ReAct-style tool-use loop: it can call `search_trials`, `check_eligibility`, `get_trial_details`, and `explain_trial` — allowing multi-step reasoning (e.g., search → inspect top results → refine query → re-search).
- LangGraph (vs. raw LangChain chains) provides explicit state management and graph-based control flow, making the agent logic easier to test and debug.

---

## 6. Medical Concept Extraction via SciSpacy + UMLS

**Decision:** Use SciSpacy NER models with UMLS concept normalization for enriching queries and computing eligibility features.

**Rationale:**
- Matching "NSCLC" to "Non-Small Cell Lung Cancer" or "Keytruda" to "pembrolizumab" requires medical knowledge. UMLS provides canonical Concept Unique Identifiers (CUIs) that unify synonyms.
- SciSpacy models are pre-trained on biomedical text and can extract entities like cancer types, biomarkers, drugs, and anatomical sites.
- Extracted concepts feed into both the eligibility feature computation (for the LightGBM ranker) and the agent's structured query.

---

## 7. Pydantic Models as the Canonical Data Contract

**Decision:** Define a single `Trial` Pydantic model that flows through every layer: parse → store → index → API response.

**Rationale:**
- A single canonical type prevents data drift between pipeline stages. If the parser produces a `Trial`, and the store expects a `Trial`, schema mismatches are caught at validation time rather than at runtime deep in the pipeline.
- Pydantic provides runtime validation, serialization, and clear documentation of field types and constraints.
- API boundary types (`SearchRequest`, `SearchResponse`, etc.) are separate Pydantic models that project or reshape the canonical `Trial` for external consumers.

---

## 8. YAML-Driven Configuration, No Magic Numbers

**Decision:** All hyperparameters, model paths, and infrastructure settings live in `configs/development.yaml` (and future `training/*.yaml` files). No hardcoded constants in source code.

**Rationale:**
- Reproducibility: every experiment configuration is version-controlled and diffable.
- Environment flexibility: switch between dev/staging/prod by swapping config files, not editing code.
- Auditability: MLflow logs the full config YAML for each training run, creating a complete experiment record.

---

## 9. Elasticsearch with Custom English Analyzer

**Decision:** Use a custom Elasticsearch analyzer with English stemming, stop word removal, and field-specific boosting (title 3×, conditions 2×).

**Rationale:**
- Default Elasticsearch analysis misses domain-specific needs. English stemming ensures "recruiting" matches "recruited", and stop words reduce noise.
- An `all_text` catch-all field concatenates title + conditions + summary + eligibility + interventions, ensuring no query falls through the cracks.
- Field boosting prioritizes title and condition matches — a trial titled "Breast Cancer Phase 3" should rank above one that merely mentions breast cancer in the eligibility criteria.
- Single shard, zero replicas — appropriate for a development/single-node setup. Production would increase both.

---

## 10. Resumable Data Ingestion Pipeline

**Decision:** The data downloader persists state (page token, page count, trial count) to a `.download_state.json` file after every page, enabling resume after interruption.

**Rationale:**
- The ClinicalTrials.gov oncology dataset spans thousands of trials across 100+ API pages. A full download takes significant time.
- Network failures, rate limits, or process interruptions should not force a restart from scratch.
- The state file is minimal (3 fields) and atomic — page N is fully written to disk before the state file advances, preventing partial-page corruption.

---

## 11. SQLite as the Primary Data Store

**Decision:** Use SQLite (via SQLAlchemy ORM) as the canonical trial store, separate from Elasticsearch.

**Rationale:**
- SQLite is zero-config, file-based, and sufficient for the data volumes involved (tens of thousands of trials, each a few KB).
- Separating the data store from the search index means the source of truth is independent of Elasticsearch. If the index is corrupted or needs re-indexing, a full rebuild from SQLite is trivial.
- SQLAlchemy ORM provides migrations support and a clean separation between the data model and persistence logic.

**Indexes:** `nct_id` (unique), `status`, `phase` — supporting the most common lookup and filter patterns.

---

## 12. FastAPI + Streamlit Serving Architecture

**Decision:** FastAPI backend (port 8000) with a separate Streamlit frontend (port 8501), communicating over HTTP.

**Rationale:**
- Decoupled frontend/backend allows independent development, testing, and deployment. The API can serve multiple clients (Streamlit UI, CLI, other integrations).
- FastAPI provides automatic OpenAPI documentation, async support, and Pydantic integration.
- Streamlit enables rapid prototyping of the patient-facing UI without frontend engineering overhead.
- CORS is configured to allow all origins in development to support the Streamlit ↔ FastAPI connection.

---

## 13. MLflow + Optuna for Experiment Tracking and HPO

**Decision:** Use MLflow for experiment tracking and Optuna for hyperparameter optimization, with all parameters sourced from YAML configs.

**Rationale:**
- MLflow provides experiment comparison, metric visualization, and model artifact storage — essential when iterating on embeddings, cross-encoder fine-tuning, and ranker training.
- Optuna integrates natively with LightGBM and supports Bayesian optimization, which is more sample-efficient than grid/random search.
- Storing the full config YAML as an MLflow artifact alongside metrics creates a complete, reproducible experiment record.

---

## 14. Prometheus + Grafana for Production Monitoring

**Decision:** Instrument the application with Prometheus metrics (request latency, retrieval stage timing, cache hit rates) and visualize with Grafana.

**Rationale:**
- A search system needs observability into latency at each pipeline stage — is BM25 slow? Is the cross-encoder the bottleneck?
- Prometheus is the de facto standard for application metrics in containerized deployments.
- Planned metrics: per-endpoint latency histograms, per-retrieval-stage latencies, cache hit/miss counters, active request gauges.

---

## 15. Project Structure: src Layout with Entry Points

**Decision:** Use the `src/` layout (`src/TrialMine/`) with `pyproject.toml`-defined entry points (`trialmine-serve`, `trialmine-ui`, `trialmine-ingest`).

**Rationale:**
- The `src/` layout prevents accidental imports of the un-installed package (a common source of "works on my machine" bugs).
- Entry points provide clean CLI commands without requiring users to know file paths.
- `pyproject.toml` (PEP 621) is the modern standard for Python packaging, replacing `setup.py`.
