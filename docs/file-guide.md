# TrialMine — File Guide

This document describes the implemented files, what they do, and how they relate to each other.

---

## Data Flow

```
ClinicalTrials.gov API v2
        │
        ▼
  data/download.py           ← fetches raw JSON pages
        │
        ▼
  data/raw/page_*.json       ← raw API responses on disk
        │
        ▼
  data/parse.py              ← extracts fields into Trial objects
        │
        ▼
  data/models.py             ← Trial & Location Pydantic models
        │
        ├────────────────────────────┐
        ▼                            ▼
  data/store.py               retrieval/bm25.py
  (SQLite: data/trials.db)    (Elasticsearch index)
                                     │
                                     ▼
                              api/routes.py → api/app.py
                              (FastAPI endpoints)
                                     │
                                     ▼
                              ui/app.py
                              (Streamlit frontend)
```

---

## Configuration & Infrastructure

### `pyproject.toml`
Project metadata, dependencies, and tooling config. Defines three CLI entry points:
- `trialmine-serve` → `TrialMine.api.app:main`
- `trialmine-ui` → `TrialMine.ui.app:main`
- `trialmine-ingest` → `TrialMine.data.download:main`

Also configures ruff (linter) and pytest.

### `configs/development.yaml`
Central configuration for paths and parameters:
- Database path, Elasticsearch URL, FAISS index path
- Model names (embedder, cross-encoder, ranker)
- Retrieval top-k values at each stage
- MLflow tracking URI and experiment name

### `docker-compose.yml`
Defines the Elasticsearch 8.12 service (single-node, security disabled) with a persistent volume.

### `Makefile`
Convenience targets:
| Target | Command | Purpose |
|--------|---------|---------|
| `setup` | `pip install -e ".[dev]"` | Install package in editable mode |
| `download` | `python scripts/download_data.py` | Run full data pipeline |
| `index` | `python scripts/build_index.py` | Build Elasticsearch index |
| `serve` | `uvicorn ...` | Start FastAPI backend |
| `ui` | `streamlit run ...` | Start Streamlit frontend |
| `test` | `pytest tests/` | Run tests |
| `lint` | `ruff check src/` | Lint source code |

### `.env.example`
Template for environment variables: `ANTHROPIC_API_KEY`, `UMLS_API_KEY`, `ELASTICSEARCH_URL`.

### `src/TrialMine/config.py`
Pydantic Settings class that loads secrets from environment variables and `.env` files. Provides `get_settings()` for dependency injection. Fields: `anthropic_api_key`, `umls_api_key`, `elasticsearch_url`, `db_path`, `faiss_index_path`.

---

## Data Layer — `src/TrialMine/data/`

### `models.py`
Defines the canonical data types used throughout the pipeline:
- **`Location`** — A trial site (facility, city, state, country, zip).
- **`Trial`** — The core data model: nct_id, title, summary, conditions, interventions, eligibility, demographics, phase, status, enrollment, dates, sponsor, locations, URL.

Every other module imports from here. This is the single source of truth for what a "trial" looks like.

### `download.py`
Downloads oncology trials from ClinicalTrials.gov API v2.
- Paginates using opaque `pageToken` (not page numbers), saving each page as `data/raw/page_NNNN.json`.
- **Resumable**: persists download state (`.download_state.json`) after every page.
- Retries failed requests with exponential backoff (3 attempts).
- Respects API rate limits with a 0.5s delay between requests.

**Depends on:** httpx
**Produces:** `data/raw/page_*.json` files

### `parse.py`
Transforms raw API JSON into `Trial` objects.
- `parse_study(raw)` — Extracts fields from the deeply nested API response into a flat `Trial`. Returns `None` if `nct_id` is missing.
- `parse_raw_files(raw_dir)` — Batch-processes all `page_*.json` files. Logs warnings for trials missing eligibility criteria or conditions.
- `parse_locations()` and `parse_phase()` handle sub-structures.

**Depends on:** `models.py`
**Consumes:** `data/raw/page_*.json`
**Produces:** `list[Trial]` in memory

### `store.py`
SQLite persistence layer using SQLAlchemy ORM.
- `TrialRow` — ORM model mapping `Trial` to the `trials` table. Lists (conditions, interventions, locations) are stored as JSON text. Indexed on `nct_id` (unique), `status`, `phase`.
- `store_trials()` — Batch-inserts trials (500 at a time), skipping duplicates by `nct_id`.
- `load_trials()` — Loads all trials back into `Trial` objects.

**Depends on:** `models.py`, SQLAlchemy
**Consumes:** `list[Trial]`
**Produces:** `data/trials.db`

---

## Retrieval — `src/TrialMine/retrieval/`

### `bm25.py`
Elasticsearch BM25 retrieval via the `ElasticsearchIndex` class.
- `create_index()` — Creates the index with a custom English analyzer (stemming + stop words) and field mappings.
- `index_trials()` — Bulk-indexes `Trial` objects in batches of 5000. Each trial is flattened into text fields plus an `all_text` catch-all.
- `search()` — Multi-match BM25 query with field boosting (title 3×, conditions 2×). Supports keyword filters on `phase` and `status`.
- `get_trial()` — Looks up a single trial by `nct_id`.

**Depends on:** `models.py`, elasticsearch-py
**Consumes:** `list[Trial]` (indexing), search queries (retrieval)
**Produces:** Ranked list of `{nct_id, title, conditions, phase, status, enrollment, score}`

---

## API — `src/TrialMine/api/`

### `schemas.py`
Pydantic request/response models for the API boundary:
- `SearchRequest` — query (str, min_length=1), top_k (1–100, default 20), filters (optional dict).
- `TrialResult` — Single result: nct_id, title, conditions, phase, status, score, URL.
- `SearchResponse` — List of results + total count + query + search_time_ms.
- `TrialDetailResponse` — Full trial detail for single-trial lookups.
- `ErrorResponse` — Structured error (error_code, message, detail).

### `routes.py`
FastAPI route definitions:
- `POST /api/v1/search` — BM25 search. Reads `ElasticsearchIndex` from `app.state`, times the query, returns `SearchResponse`.
- `GET /api/v1/trial/{nct_id}` — Single trial lookup. Returns 404 with `ErrorResponse` if not found.
- `GET /health` — Liveness probe returning `{"status": "ok"}`.

**Depends on:** `schemas.py`, `retrieval/bm25.py` (via `app.state`)

### `app.py`
FastAPI application factory:
- `create_app()` — Builds the app with CORS middleware (all origins) and includes routes.
- `lifespan()` — Async context manager that connects to Elasticsearch on startup and stores the `ElasticsearchIndex` on `app.state`.
- `main()` — Entry point for `trialmine-serve`; runs uvicorn on `0.0.0.0:8000` with reload.

**Depends on:** `routes.py`, `retrieval/bm25.py`

---

## Frontend — `src/TrialMine/ui/`

### `app.py`
Streamlit patient-facing UI:
- Search bar with example query buttons ("breast cancer phase 3", "lung cancer immunotherapy recruiting", "pediatric leukemia trials").
- Sidebar filters: status selector, phase selector, top_k slider (5–100).
- Calls `POST localhost:8000/api/v1/search` via httpx, displays result cards with colored status/phase badges, relevance scores, and ClinicalTrials.gov links.
- Handles connection errors and HTTP errors with user-friendly messages.

**Depends on:** `api/` (via HTTP), streamlit, httpx

---

## Scripts — `scripts/`

### `download_data.py`
End-to-end data pipeline:
1. Downloads trials via `data/download.py` (skippable with `--skip-download`).
2. Parses raw JSON via `data/parse.py`.
3. Stores in SQLite via `data/store.py`.
4. Prints summary statistics (trials by status, phase, top conditions).

**Used by:** `make download`

### `build_index.py`
Builds the Elasticsearch index from SQLite:
1. Loads trials from SQLite via `data/store.py`.
2. Creates index and bulk-indexes via `retrieval/bm25.py`.
3. Runs a test query ("breast cancer") and displays top 5 results with timing.

**Used by:** `make index`
