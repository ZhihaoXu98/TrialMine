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
Last updated: 2026-05-07

Phase: 7 (LangGraph agent system) — Week 7 close-out. Demo-quality Streamlit UI (`src/TrialMine/ui/app.py`) renders the new agent response shape end-to-end: extracted patient profile, status/phase pills, rank-based match%, expandable explanation / eligibility / full-details cards, conditional sidebar filters, and an Agent Reasoning expander tied to `agent_trace`. Full stack containerised — single `docker compose up --build` brings up six services (api, ui, elasticsearch, redis, prometheus, grafana) with healthcheck-driven dependency ordering. `/metrics` instrumentation (request counter + latency histogram + per-stage histogram fed from `agent_trace`) wired into FastAPI; Grafana dashboard auto-provisioned with 4 panels (latency p50/p95, RPS by endpoint, search-stage p95, 5xx error rate).

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
- **Eligibility parser** (Phase 6c — Week 5):
  - `src/TrialMine/features/eligibility.py` (`EligibilityParser`, `EligibilityProfile` Pydantic model, `parse_age_string`)
  - SciSpacy `en_core_sci_lg` + regex hybrid: regex owns closed-vocab slots (age, ECOG/Karnofsky, sex), SciSpacy owns open-vocab biomedical entities
  - Output schema: `min_age_years` / `max_age_years` (float, months survive as fractions), `sex`, `required_conditions` / `excluded_conditions` / `required_prior_treatments` / `excluded_prior_treatments` (typed via keyword heuristic ~70% precision), `parse_confidence` (heuristic mean of section / age / sex sub-confidences, NOT calibrated), `section_source`
  - Section split: 4-tier fallback (`headers` → `single_header` → `variant` → `fallback`)
  - Age extraction: column-first, regex fallback (12 patterns including unicode ≥/≤ and `\<` escape leak)
  - Stop-list (~80 boilerplate terms) filters SciSpacy noise; treatment regex routes drug/therapy spans to treatments bucket
  - Tests: 37 (32 fast regex + 5 slow SciSpacy integration), all passing — `tests/unit/test_eligibility.py`
  - Demo: `scripts/demo_eligibility_parser.py --limit 20 [--show-buckets]`
  - Batch parse: `scripts/parse_eligibility.py [--limit N] [--resume]` writes to `parsed_eligibility` SQLite table; 23 trials/sec single-process; 1000-trial demo: 94.4% have min_age, 99.7% have conditions, avg conf 0.977, 91% canonical headers
- **Concept normalizer** (Phase 6c):
  - `src/TrialMine/features/concepts.py` (`ConceptNormalizer.normalize`, `expand_query`)
  - 45-entry hand-built lay→medical synonym dict (top 20 cancer sites + metastasis phrasings + stage Roman numerals + treatment vocab)
  - All mappings broaden to `... neoplasm` forms, never narrow to specific subtypes (e.g. liver cancer → hepatic neoplasm, not HCC)
  - No chained mappings; replacement is one hop only
  - `normalise_concept` and `extract_concepts` skeletons remain for future UMLS EntityLinker integration
- **Agent system** (Phase 7 — Week 6):
  - **LangChain `@tool` wrappers** — `src/TrialMine/agents/tools.py` exports `ALL_TOOLS = [search_trials, lookup_medical_concept, check_trial_eligibility, get_trial_details]`. Pydantic args schemas, JSON-string returns, `{"error": "..."}` envelopes on failure. Heavy resources (ES, FAISS, embedder, CE, blender) lazy-loaded into module-level singletons; failures cached so misconfiguration doesn't retry.
  - **QueryParserAgent** — `src/TrialMine/agents/query_parser.py`. Claude Haiku 4.5 via raw `anthropic` SDK with `messages.parse(output_format=_ExtractedFields)`. Extracts a `PatientProfile` (condition, condition_stage, age, sex, biomarkers, prior_treatments, preferences, location). Always returns a profile; on any failure (API error, validation, refusal, timeout) falls back to a `raw_query`-only profile so downstream never breaks. Measured: ~1.3–1.5 s warm, ~$0.0017/query at ~1370 input + 50 output tokens. Tested 10/10 on representative queries (incl. typo correction `ostate` → `prostate`, sex inference `daughter` → Female, biomarker capture `MSI-high` / `triple negative`).
  - **SearchOrchestrator** — `src/TrialMine/agents/orchestrator.py`. Rule-based async pipeline: `normalize → build_query → build_filters → retrieve (full_pipeline via tools.py) → check eligibility (parallel) → template explanation`. **No LLM in hot path** — only the upstream QueryParser is LLM-driven. Eligibility checks parallelized via `asyncio.gather` over `asyncio.to_thread`; benchmarked **24.6× speedup** on 10 cached checks (143 ms → 5.8 ms). Heavy retrieve step also wrapped in `to_thread` so it doesn't block the event loop. Default filter `status=RECRUITING`; phase from `preferences`; "completed/closed/finished trial" in query overrides the status filter.
  - **LangGraph pipeline** — `src/TrialMine/agents/pipeline.py` (`build_pipeline`, `search`). `StateGraph(SearchState)` with `parse_query → execute_search → fallback_search` + conditional routing on `state["error"]`. State carries `raw_query`, `patient_profile` (dict), `search_results`, `agent_trace` (Annotated with `operator.add` reducer — every node appends), `error`, `used_fallback`. 15 s wall-clock cap via `asyncio.wait_for`; on timeout returns degraded structured response. Smoke test: 5 queries incl. `""` and `"asdfghjkl"` — 0 uncaught exceptions across primary + fallback paths.
  - **API integration** — `POST /api/v1/search` defaults to `use_agent=true` (routes through pipeline). Set `use_agent=false` to bypass and run legacy bm25/semantic/hybrid. `SearchResponse` adds optional `agent_trace`, `used_fallback`, `patient_profile`, `error`; `TrialResult` adds optional `explanation`, `eligibility`, `warnings`. Lifespan in `app.py` is now ES-tolerant: API starts in degraded mode if ES is down, agent path returns structured fallback response, legacy paths return clear 503s.
  - **Smoke test**: `scripts/test_pipeline.py` — 5-query end-to-end test with full `agent_trace` printout per query. Pass criterion: zero uncaught exceptions.
  - Note: the @tool wrappers are unused by the orchestrator's hot path (it calls retrieval primitives directly). They remain as the surface for any future LangChain-bound chat agent (e.g., follow-up Q&A on a specific NCT).

### Design decisions (Week 7)
- **Decision 24** — Streamlit over React for the patient-facing UI. Single Python file collapses no-JS-toolchain + no-API-client + no-state-management. The agent_trace expander, conditional sidebar (filters hide when AI agent toggle is on), session-state-driven examples, and rank-based match% progress bars all shipped in one pass. Trade-offs: `st.columns(5)` doesn't reflow on narrow viewports (mobile is cramped), no granular component control, and Streamlit-specific patterns (form `on_click` handlers, reruns, `cache_resource`) can be inscrutable to non-Streamlit reviewers. Trigger to migrate to React / Next.js: real concurrent users, mobile-first requirement, or token-by-token streaming surface (Streamlit's rerun model fights live-stream UIs).
- **Decision 25** — Docker Compose, not Kubernetes. Single host, one `up --build` command, no operator burden. Six services with healthcheck-driven `depends_on` (api waits on ES + redis healthy; ui waits on api started; grafana waits on prometheus). Compose YAML does NOT translate to K8s manifests — migration path is Helm or Kustomize, not copy-paste. Trigger to migrate: >1 host, autoscaling, rolling deploys, multi-tenant. **Caveat (TODO):** no `restart:` policy declared on any service yet — add `restart: unless-stopped` if the stack is meant to live longer than a few hours unattended.
- **Decision 26** — ES heap = 1 GB (`ES_JAVA_OPTS=-Xms1g -Xmx1g`). 140,723 docs indexed, ~596 MB on disk. 1 GB heap gives comfortable headroom for query cache + filter cache + segment metadata. 512 MB sits below ES 8.x's recommended floor for non-trivial datasets and risks OOM under bulk-index or aggregation load — the choice is precaution, NOT measurement (did not benchmark 512 MB OOM directly). If host RAM becomes constraint-bound, the measurable next step is 768 MB.
- **Decision 27** — `OMP_NUM_THREADS=8` in the Linux container, overriding the project's macOS-host default of `=1`. The original `=1` was set to avoid the macOS FAISS + LightGBM OpenMP runtime collision (Apple Accelerate's libomp + faiss-cpu's libomp double-loading → SIGSEGV). Inside the Linux container both libraries link against glibc's libgomp1 and coexist without crash. Empirical impact: with `=1`, the BioLinkBERT cross-encoder was single-threaded and took ~30 s/call for 50 candidates; with `=8`, ~4–8 s, fitting the 15 s agent budget. **Note for host-Python runs**: keep `OMP_NUM_THREADS=1` for `python scripts/...` invocations on macOS — the container value doesn't help, and the host conflict still applies.
- **Decision 28** — Filter at the post-RRF candidate level, NOT at the semantic-search call. FAISS has no native filter primitive; semantic returns top-K by cosine regardless of status. The original `HybridRetriever.full_pipeline` only passed `filters=` to BM25, and well-described COMPLETED trials leaked in via the semantic side — dominating top-10 even when `filters={"status": "RECRUITING"}` (8/10 COMPLETED in the lung-cancer demo run on 2026-05-07). Fix: after RRF merge, drop semantic-only candidates whose ES metadata doesn't match the filter; BM25 hits fast-path through (already filtered). Cost: 1 extra `get_trial` ES call per semantic-only candidate that survived RRF — typically 50–100 calls × ~5 ms ≈ 250–500 ms added latency. Alternative considered: pre-filter semantic side with one ES mget — faster but couples the retriever to ES's mget API; rejected as premature optimisation. After the patch: 10/10 RECRUITING for the same query, "completed melanoma trials phase 3" still correctly returns COMPLETED via the existing keyword-override path.

### Design decisions (Week 6)
- **Decision 20** — LangGraph `StateGraph` over CrewAI / raw function calling. The flow is *deterministic* (parse → search → explain → fallback), not open-ended reasoning. State machine gives explicit conditional routing (`execute_search → fallback_search` on error), structured agent_trace via the `operator.add` reducer, and per-node testability. CrewAI's role-based abstraction is overhead for a fixed flow; raw function calling lacks the conditional-routing primitive and would re-implement what `StateGraph` provides.
- **Decision 21** — Claude Haiku 4.5 for slot extraction. Extraction task, not complex reasoning — Sonnet/Opus are overkill. Measured ~1.3–1.5 s warm latency and ~$0.0017/query (slightly above the original aspirational <1 s / $0.001; baseline Haiku 4.5 runs hotter than the napkin claim, but quality is 10/10 on test queries so the budget is fine). Cost dominated by ~1370-token input prompt; trimming the prompt would risk extraction quality regressions. Prompt is below Haiku 4.5's 4096-token cache threshold so prompt caching is a no-op for now.
- **Decision 22** — Two LLM-using components, not three (template explanations). LLM-per-result would add 10–20 s and ~$0.05 per query for marginal quality gain on this surface. Templates render in microseconds and surface the eligibility verdict counts directly (`Eligibility: likely a Match (5 met, 0 unmet, 1 unknown)`). Future option: one batched summarization pass over the top 3 explanations if user feedback shows the templates are too terse.
- **Decision 23** — 15 s wall-clock budget + fallback path. The pipeline always returns a structured `SearchResponse` — never a 500, never an unhandled exception. Quality degrades gracefully: agent path → `fallback_search` (plain hybrid, no eligibility) → empty results with `error` reason set. *The contract guarantees response shape, not non-empty results*: empty `results: []` is a valid outcome when ES is fully down or the input matches nothing. Conditional fallback fires only on system errors (`execute_search` exception), not on empty primary results — empty is treated as a legitimate answer, not a failure to mask.

### Design decisions (Week 5)
- **Decision 17** — SciSpacy + regex hybrid for eligibility parsing, NOT custom NER. Custom NER ≈ 80–120 hr; hybrid ≈ days at ~70% precision. Best accuracy/effort.
- **Decision 18** — Hand-built ~45-entry synonym dict, NOT UMLS yet. Covers 23-cancer taxonomy. UMLS via SciSpacy `EntityLinker` is target after dict outgrows ~50 entries. Each entry must be lay→broader medical (never narrower); no chained mappings; no replacements that collide with verbatim trial vocabulary.
- **Decision 19** — Eligibility checker (future, not yet built) will emit Met / Unmet / **Unknown** verdicts, not binary. Binary on partial info produces confidently-wrong outputs. Trial-level rollup rule: any hard Unmet → trial Unmet; all Met → Met; else Unknown. Distinguish parser-unknown (low parse_confidence) from patient-unknown (missing patient field) — same label, different UX.

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
- `parsed_eligibility` table inside `data/trials.db` — structured eligibility per trial (parser_version 0.1.0); written by `scripts/parse_eligibility.py`
- `data/evaluation/per_query_*.json` — per-query metrics from compare_embeddings.py
- `data/training/train_pairs.jsonl` — 586K training triplets (1.0 GB)
- `data/training/val_pairs.jsonl` — 145K validation triplets (260 MB)
- `data/training/synthetic_queries.jsonl` — 1,500 Claude-generated patient queries (1.5 MB)
- `mlflow.db` — MLflow tracking database
- Elasticsearch `trials` index — requires `docker start es`
- `.env` — API keys (ANTHROPIC_API_KEY) — NEVER commit

### What's next
- **Streamlit UI** for the new agent response shape — render `explanation`, `eligibility` verdict counts, `warnings`, and (optionally) the collapsed `agent_trace` per result. Currently still pointing at the legacy single-shot search.
- **Extract eligibility matcher** from `tools.py:check_trial_eligibility` into `src/TrialMine/features/eligibility_matcher.py` — Decision 19's checker module. Removes the awkward `.invoke({...})` + `json.loads()` indirection from `orchestrator.py` and lets the matcher be unit-tested standalone.
- ~~**Pre-warm cross-encoder + LightGBM at API startup.**~~ DONE 2026-05-07. `app.py` lifespan now calls `tools._get_hybrid()` + `_get_reranker_or_none()` + `_get_blender_or_none()` and runs a single throwaway `full_pipeline("warmup", top_k=1, rerank_top_k=5)` after the agent pipeline compile. Adds ~3 s to startup, removes the cold-load tax from the first request. **Subtle bug worth recording** for future startup-warming work: the first version warmed `app.state.hybrid_retriever`, but the orchestrator uses the *separate* singleton from `tools._get_hybrid()` — they share CE + blender singletons but not the embedder cache. Warm the same retriever the orchestrator will call, otherwise the first request still cold-starts the embedder.
- **Run the happy-path agent test** with ES up: `docker start es && OMP_NUM_THREADS=1 python scripts/test_pipeline.py` — confirms primary path returns ranked + eligibility-checked results across the 5 representative queries.
- Wire `ConceptNormalizer.expand_query` into the *BM25 side* of retrieval (orchestrator already uses it on the query side; BM25 still indexes verbatim trial text).
- Optional: retrain CE on graded labels (6,018 pairs) instead of binary — could improve CE feature quality.
- Optional: Optuna hyperparameter tuning for LightGBM (viable now with 145 queries).
- Optional: UMLS via SciSpacy `EntityLinker` to replace the synonym dict + provide typed entities (Decision 18 upgrade path).
- Optional: track per-query agent latency in MLflow for ongoing budget monitoring.
