# Week 1 — Data Pipeline + BM25 Search (Student Edition)

A textbook walkthrough of everything we built in the first week of TrialMine,
written as if I'm sitting next to you at a whiteboard. By the end, you should be
able to (a) explain the system to a friend who has never built a search engine,
(b) navigate every file we touched, and (c) answer interview questions about
the design choices.

If you've never built a search engine before — that's fine. Every term gets
defined the first time it appears.

---

## Part 1 — Why are we doing this?

Cancer patients with hard-to-treat diseases sometimes find their best
remaining option in a **clinical trial** — a research study where doctors
test a new drug, device, or procedure on real patients. The U.S.
government runs a public registry called
[ClinicalTrials.gov](https://clinicaltrials.gov) that lists every trial in
the world (~500K total, ~140K of them about cancer).

The problem: that registry is built for clinicians, not patients. A
patient typing *"my mom has stomach cancer that came back"* gets back
results matched by **literal text**. If the trial actually says
*"recurrent gastric adenocarcinoma"* (the medical name for the same
thing), no shared word ever matches. The patient never sees it.

TrialMine is a research project that builds a better front door. The
week 1 goal is the simplest possible version:

> Pull every cancer trial from the API → store them locally → make them
> searchable by typing words → show the answers in a web page.

No neural networks, no LLMs, no "agents" yet. But this is the foundation
every later week builds on, and *most production ML failures come from
data, not models*. Get this right and the rest goes smoothly.

---

## Part 2 — Vocabulary you'll meet

Skim this and come back when you hit a term you don't know.

### Engineering terms

- **API (Application Programming Interface)** — a website endpoint for
  programs (not humans). Returns JSON instead of an HTML page.
- **Pagination** — splitting a large response across many pages. The
  API gives you a *page token* and you ask for the next page until
  you've seen everything.
- **JSON** — the standard format APIs return. Looks like nested dicts
  and lists.
- **Pydantic** — a Python library for typed data classes. Validates
  inputs at runtime; gives your IDE autocomplete.
- **SQLite** — a database that lives in a single file. No server, no
  install. Ideal for a single-machine project.
- **SQLAlchemy** — the Python library for talking to SQL databases
  without writing raw SQL strings.
- **Inverted index** — a data structure mapping each *word* → *list of
  documents containing it*. Core idea behind every search engine since
  the 1960s.
- **BM25** — Best Match 25, a relevance-ranking formula from the
  1990s. Explained in Part 3.
- **Elasticsearch** — a search server (written in Java) that stores an
  inverted index and serves BM25 queries over HTTP. Runs in Docker.
- **FastAPI** — Python web framework. You write functions; FastAPI
  exposes them as HTTP endpoints with automatic JSON validation.
- **Streamlit** — Python library that turns a script into an
  interactive web app. Every widget click re-runs the whole script.
- **NCT ID** — every trial on ClinicalTrials.gov has a unique
  identifier like `NCT12345678`. We use this as the primary key.

### Medical terms (only what you need)

- **Oncology** — the branch of medicine concerned with cancer.
- **Trial phase** — clinical trials run in stages from earliest to
  latest: **Phase 1** (small, tests safety), **Phase 2** (efficacy),
  **Phase 3** (large, tests against current standard treatment),
  **Phase 4** (post-approval monitoring). Patients usually want
  Phase 2 or 3 — those are where most enrolled patients see benefit.
- **Trial status** — `RECRUITING` (accepting patients now),
  `ACTIVE_NOT_RECRUITING` (running but full), `COMPLETED` (finished),
  `TERMINATED` (stopped early). A "perfect match" trial that's
  COMPLETED is useless to a patient.
- **Eligibility criteria** — the rules a patient must satisfy to
  enroll. Two parts: **inclusion** (must be true — e.g., *"age 18 or
  older"*) and **exclusion** (must NOT be true — e.g., *"no prior
  chemotherapy in the past 4 weeks"*).
- **Cancer types** mentioned in queries:
  *carcinoma* = cancer of skin/organ-lining cells (most common kind);
  *lymphoma* = cancer of lymph nodes;
  *leukemia* = cancer of blood-forming cells;
  *melanoma* = cancer of pigment-producing skin cells;
  *sarcoma* = cancer of bone or soft tissue.
- **"Recurrent"** — cancer that came back after treatment.
- **"Metastatic" / "spread"** — cancer that has moved from its
  original site to another organ (e.g., breast → bone).

---

## Part 3 — A 30-second tour of BM25

The whole point of week 1 is to make text searchable. Searches need to be
*ranked* — the most relevant trials at the top, the least relevant at the
bottom. So the first question is: what does "relevant" mean?

A short history of the answer:

| Era | Method | What it does | Problem |
|---|---|---|---|
| 1970s | Boolean | Match or not. Ranking is arbitrary. | Useless ranking. |
| 1980s | TF-IDF | Score = (how often the word appears) × (how rare the word is). | 100 mentions ≠ 100× more relevant. Long docs cheat. |
| 1990s | **BM25** | Two fixes on TF-IDF: saturating term frequency and length normalization. | Still in use today. |

**Saturating term frequency.** If "breast cancer" appears once in a
document, that's a strong signal. If it appears 50 times, that's not 50×
stronger — it just means the document is talking about breast cancer
*a lot*, which we already knew. So BM25 caps the boost from repetition.

**Length normalization.** A long document has more chances to contain
your query words just by accident. BM25 penalizes long documents
proportionally so that a 100-word abstract isn't unfairly outranked by a
20,000-word PDF.

**Field boosting on top.** BM25 is per-field. We can also tell the engine
*"a hit in the title is worth more than a hit in eligibility."* In our
case:

- **title 3×** — if "breast cancer" is in the title, the trial *is* about
  breast cancer.
- **conditions 2×** — the conditions field is a curated tag list.
- **eligibility 1×** — these are inclusion/exclusion rules. "Breast
  cancer" appearing here might be in an exclusion (*"prior breast cancer
  patients are NOT eligible"*) — useful but ambiguous.

That's all the search-theory background you need for week 1.

---

## Part 4 — What we built, step by step

We built five things. Here they are in the order we did them.

### Step 1 — Project skeleton

**Files:** `pyproject.toml`, `Makefile`, `src/TrialMine/` directory tree,
`docker-compose.yml`, `.env.example`, `configs/development.yaml`.

Before any logic, we created an empty house with all the rooms in the right
places:

```
src/TrialMine/
├── data/        ← download, parse, store trials
├── retrieval/   ← BM25 / FAISS / hybrid search (most empty for week 1)
├── models/      ← embeddings, cross-encoder, ranker (all empty for week 1)
├── agents/      ← LangGraph agent (week 6)
├── features/    ← eligibility parsing, concept normalization (week 5)
├── api/         ← FastAPI app + routes + schemas
└── ui/          ← Streamlit frontend
scripts/         ← one-shot pipelines (download, build_index, ...)
configs/         ← YAML hyperparameters
```

The skeleton is a one-time ceremony, but it pays off forever after. The
moment you need to add a cross-encoder in week 4, you know exactly which
folder it goes in. No bikeshedding.

We also wrote a `Makefile` so common operations have short names:

```
make download   # download trials from ClinicalTrials.gov
make index      # build BM25 + FAISS indexes
make serve      # start the FastAPI server
make ui         # start the Streamlit frontend
make test       # run pytest
```

Why a `Makefile` and not a shell script? Because `make` is a *protocol*
every developer recognizes. New collaborators don't have to read your
script — `make download` Just Works.

**Student lesson:** *Set up the structure first; write the code second.* If
you start coding before you've decided where things live, you'll be
refactoring within a week.

### Step 2 — Download trials from ClinicalTrials.gov API v2

**File:** `src/TrialMine/data/download.py` (~160 lines)
**Driver:** `scripts/download_data.py`

The API lives at `https://clinicaltrials.gov/api/v2/studies`. Three details
matter:

1. **Pagination uses `pageToken`, not page numbers.** Each response has a
   `nextPageToken` field. You hand it back in your next request to get
   the next 1,000 trials. This continues until the API returns no token.
2. **`pageSize` maxes at 1,000.** We use the maximum to minimize round trips.
3. **Rate limits are unstated but real.** We sleep 0.5 seconds between
   requests to be polite.

Here's the core loop, simplified:

```python
def download_oncology_trials(output_dir: Path, query: str) -> int:
    state = _load_state(output_dir)              # for resume
    page_num = state["pages_saved"]
    next_token = state["next_page_token"]

    with httpx.Client() as client:
        while True:
            params = {"query.term": query, "pageSize": 1000, "format": "json"}
            if next_token:
                params["pageToken"] = next_token

            data = fetch_page(client, params)    # retries on transient errors
            studies = data.get("studies", [])
            if not studies:
                break

            (output_dir / f"page_{page_num:04d}.json").write_text(json.dumps(data))
            page_num += 1
            next_token = data.get("nextPageToken")
            _save_state(output_dir, {...})       # checkpoint EVERY page

            if not next_token:
                break
            time.sleep(0.5)
```

The non-obvious choices:

**Resume on interruption.** A 60-minute download will crash. So we save a
state file (`data/raw/.download_state.json`) after every page. Restart
the script and it picks up from the last saved page.

**Save raw JSON to disk before parsing.** We could parse and store as we
go — faster but riskier. If our parser has a bug, we'd have to re-hit
the API to fix it. By saving raw JSON first, *the parsing step is cheap to
re-run* over and over. This is the **separate I/O from compute** rule.

**Retry with exponential backoff.** `fetch_page` retries 3 times with
2s, 4s, 8s waits. Transient HTTP errors are common; we shouldn't fail the
whole download because of one blip.

The query string is a deliberately wide oncology net:

```python
ONCOLOGY_QUERY = (
    "cancer OR oncology OR tumor OR carcinoma OR lymphoma OR "
    "leukemia OR melanoma OR sarcoma"
)
```

**Result:** 140,723 oncology trials downloaded as ~145 JSON files in
`data/raw/`. Took ~75 minutes including resume from a couple of timeouts.

### Step 3 — Parse the nested JSON into typed objects

**File:** `src/TrialMine/data/parse.py` (~190 lines)
**Schema:** `src/TrialMine/data/models.py` (~40 lines)

The API returns deeply nested JSON. Here's the structure for one trial:

```
study
└── protocolSection
    ├── identificationModule
    │   ├── nctId            ← we want this
    │   └── officialTitle    ← and this
    ├── descriptionModule.briefSummary
    ├── conditionsModule.conditions       (list of strings)
    ├── armsInterventionsModule.interventions
    ├── eligibilityModule
    │   ├── eligibilityCriteria  (one big text blob)
    │   ├── minimumAge          ("18 Years")
    │   ├── maximumAge
    │   └── sex                 ("ALL", "MALE", "FEMALE")
    ├── designModule
    │   ├── phases              (["PHASE3"])
    │   └── enrollmentInfo.count
    ├── statusModule.overallStatus       ("RECRUITING")
    ├── sponsorCollaboratorsModule.leadSponsor.name
    └── contactsLocationsModule.locations  (list of dicts)
```

We map this onto a `Trial` Pydantic model:

```python
class Trial(BaseModel):
    nct_id: str
    title: str = ""
    brief_summary: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    eligibility_criteria: str | None = None
    min_age: str | None = None      # raw string like "18 Years"
    max_age: str | None = None
    sex: str | None = None
    phase: str | None = None        # "Phase 1", "Phase 1/Phase 2"
    status: str | None = None
    enrollment: int | None = None
    locations: list[Location] = Field(default_factory=list)
    url: str | None = None
```

**Why Pydantic and not a plain dict?** Three reasons:

1. **Validation at the seam.** If the API ever changes a field type
   (which has happened), Pydantic raises immediately. A dict would
   silently propagate `None` through the pipeline and you'd debug it
   three weeks later.
2. **Autocomplete.** Your IDE knows that `trial.conditions` is a list of
   strings. No `dict["conditons"]` typos.
3. **Documentation.** The class definition *is* the schema. New
   collaborators read one file to understand the shape of a trial.

#### Three subtle parsing choices

**Choice 1: nested-dict access without crashes.** Real API responses
sometimes drop entire modules. `study["protocolSection"]["designModule"]`
fails with `KeyError` if `designModule` is absent. So we wrote a tiny
helper:

```python
def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested dict access: _get(d, 'a', 'b', 'c')."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})
    return d if d != {} else default
```

Now `_get(ps, "designModule", "enrollmentInfo", "count")` returns either
the value or `None`. No crashes on missing fields.

**Choice 2: keep the trial even if many fields are missing.** Some
trials have no eligibility criteria. Some have no conditions. We still
keep them — only the `nctId` is required. The rest can be `None`. The
full reasoning is in Decision 5 below.

**Choice 3: pre-compute a phase mapping.** The API returns phases as
codes like `["PHASE3"]` or `["PHASE1", "PHASE2"]`. We translate to
human-readable strings (`"Phase 3"`, `"Phase 1/Phase 2"`) at parse time
so downstream code never needs to know about the codes:

```python
_PHASE_MAP = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "N/A",
}
```

#### What parsing actually does

```python
def parse_raw_files(raw_dir: Path) -> list[Trial]:
    page_files = sorted(raw_dir.glob("page_*.json"))
    trials, missing_eligibility, missing_conditions, errors = [], 0, 0, 0

    for page_file in page_files:
        data = json.loads(page_file.read_text())
        for raw_study in data.get("studies", []):
            trial = parse_study(raw_study)
            if trial is None:
                errors += 1
                continue
            trials.append(trial)
            if not trial.eligibility_criteria:
                missing_eligibility += 1
            if not trial.conditions:
                missing_conditions += 1

    logger.info("Parsed %d trials | missing elig: %d | missing cond: %d",
                len(trials), missing_eligibility, missing_conditions)
    return trials
```

We log coverage stats so we know how clean the data is *before* we trust
it. (Of 140K trials, ~5% are missing eligibility text and ~0.5% are
missing conditions.)

### Step 4 — Persist to SQLite

**File:** `src/TrialMine/data/store.py` (~170 lines)

We need a place to keep the parsed trials so we don't re-parse them every
time. SQLite is the answer for a single-machine project:

- **Zero install.** Python ships with `sqlite3`.
- **One file on disk.** `data/trials.db` — easy to back up, copy, share.
- **SQL syntax.** You can debug with `sqlite3 data/trials.db` from the
  shell.
- **Indexes are free.** A `CREATE INDEX ON trials(status)` makes
  filtering by status near-instantaneous.

The schema mirrors the Pydantic model. List fields (`conditions`,
`interventions`, `locations`) are stored as JSON text — SQLite has no
native list type, and JSON-in-text is good enough for our access pattern
(read all, never query inside the list).

```python
class TrialRow(Base):
    __tablename__ = "trials"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nct_id: Mapped[str] = mapped_column(String(20), unique=True)
    title: Mapped[str] = mapped_column(Text, default="")
    brief_summary: Mapped[str | None] = mapped_column(Text)
    conditions: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    interventions: Mapped[str] = mapped_column(Text, default="[]")
    # ... and so on
```

We add explicit indexes on the columns we filter by often:

```python
_idx_nct_id = Index("ix_trials_nct_id", TrialRow.nct_id)
_idx_status = Index("ix_trials_status", TrialRow.status)
_idx_phase = Index("ix_trials_phase", TrialRow.phase)
```

The store interface is two functions:

```python
def store_trials(trials: list[Trial], db_path: Path) -> int:
    """Insert trials, skipping duplicates by nct_id. Returns count inserted."""

def load_trials(db_path: Path) -> list[Trial]:
    """Read everything back. ~6 seconds for 140K rows."""
```

**Subtle choice:** we deduplicate on `nct_id` *before* insert (by
fetching the existing IDs once). Why not let the database raise on
duplicate-key? Because we re-run the pipeline often, and we don't want
to wrap every insert in a try/except. Asking up-front is faster and
cleaner.

**Result:** A 912 MB SQLite file at `data/trials.db` with 140,723 rows
and three indexes. Loads back in 6 seconds.

### Step 5 — Build the BM25 index in Elasticsearch

**File:** `src/TrialMine/retrieval/bm25.py` (~240 lines)
**Driver:** `scripts/build_index.py`

SQLite is great for storage but bad at relevance ranking. Asking SQLite
*"give me the trials most similar to 'breast cancer immunotherapy'"* is
either slow (full scan + LIKE) or impossible. We need a real search
engine.

Enter **Elasticsearch**. It's a Java service that stores an inverted
index and computes BM25 in a few milliseconds. We run it in Docker:

```yaml
# docker-compose.yml — week 1 version
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
```

`docker compose up -d elasticsearch` starts it; `curl localhost:9200`
verifies it's alive.

The wrapper class is `ElasticsearchIndex`. Three methods matter.

#### 5a. Creating the index with field mappings

The full settings dict is in `bm25.py`. Two design choices matter:

**`text` vs `keyword`.** Each field is one or the other:
- `text` (e.g., `title`, `brief_summary`, `conditions`,
  `eligibility_criteria`) is *analyzed* — lowercased, stemmed, and has
  stopwords removed before being stored in the inverted index. This is
  what BM25 ranks against.
- `keyword` (e.g., `nct_id`, `phase`, `status`) is stored as-is, used
  only for exact filtering. We filter `status="RECRUITING"`, we don't
  *search* it.

**The `english_custom` analyzer** is just `standard tokenizer +
lowercase + english_stemmer + english_stop`. Stemming means *"recurring"*
matches *"recur"*; stopword removal drops *"the"*, *"of"*, *"and"*. The
analyzer is what makes BM25 robust to phrasing differences.

We also add a synthetic `all_text` field at index time (title +
conditions + summary + eligibility + interventions, concatenated). It
acts as a catch-all for queries that don't match any single field
strongly but match across several.

#### 5b. Bulk indexing 140K trials

The naive approach — one `index` call per trial — would take hours.
Elasticsearch has a `bulk` API that batches many actions into one
request. We use it via `elasticsearch-helpers`:

```python
def index_trials(self, trials: list[Trial]) -> int:
    indexed = 0
    batch_size = 5000
    for start in range(0, len(trials), batch_size):
        batch = trials[start : start + batch_size]
        actions = [self._trial_to_action(t) for t in batch]
        try:
            success, _ = bulk(self.es, actions, raise_on_error=False)
            indexed += success
        except BulkIndexError as exc:
            indexed += len(batch) - len(exc.errors)
    self.es.indices.refresh(index=self.index_name)
    return indexed
```

`raise_on_error=False` means one bad document doesn't kill the whole
batch — same principle as parsing.

Result: ~80 seconds to index all 140,723 trials. The on-disk index is
about 596 MB.

#### 5c. The search query

This is the heart of the system. Given a query string and optional
filters, return the top-k trials by BM25 score:

```python
def search(self, query: str, filters: dict | None = None,
           top_k: int = 50) -> list[dict]:
    must = [{
        "multi_match": {
            "query": query,
            "fields": [
                "title^3",            # ← 3x boost
                "conditions^2",       # ← 2x boost
                "interventions",
                "brief_summary",
                "eligibility_criteria",
                "all_text",
            ],
            "type": "best_fields",
            "tie_breaker": 0.3,       # secondary fields can break ties
        }
    }]

    filter_clauses = []
    if filters:
        if "status" in filters:
            filter_clauses.append({"term": {"status": filters["status"]}})
        if "phase" in filters:
            filter_clauses.append({"term": {"phase": filters["phase"]}})

    body = {"size": top_k, "query": {"bool": {"must": must,
                                              "filter": filter_clauses}}}
    resp = self.es.search(index=self.index_name, body=body)
    return [
        {"nct_id": h["_source"]["nct_id"],
         "title": h["_source"].get("title", ""),
         "conditions": h["_source"].get("conditions", ""),
         "phase": h["_source"].get("phase"),
         "status": h["_source"].get("status"),
         "enrollment": h["_source"].get("enrollment"),
         "score": h["_score"]}
        for h in resp["hits"]["hits"]
    ]
```

Things worth noticing:

- **`type: "best_fields"`** picks the score from the *single* field
  with the strongest match (title-boosted), rather than summing across
  fields. The `tie_breaker: 0.3` lets a strong secondary field break a
  tie between two title matches.
- **`must` vs `filter`**. `must` clauses contribute to the score.
  `filter` clauses are pure yes/no — they don't change rankings, just
  exclude documents. Status and phase are filters.

### Step 6 — Wrap it in FastAPI

**Files:** `src/TrialMine/api/{app.py, routes.py, schemas.py}` (~250 lines total)

A class is great inside Python, but a UI needs a *web service*. FastAPI
is the obvious choice in modern Python:

- Pydantic validation comes built-in (your input/output schemas are
  just classes).
- Auto-generated OpenAPI docs at `/docs` (swagger UI for testing).
- Async-native (matters in week 6).

The schemas live in `schemas.py`:

```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(20, ge=1, le=100)
    filters: dict | None = None
    method: Literal["bm25", "semantic", "hybrid"] = "hybrid"

class TrialResult(BaseModel):
    nct_id: str
    title: str
    conditions: list[str]
    phase: str | None
    status: str | None
    score: float
    url: str | None

class SearchResponse(BaseModel):
    results: list[TrialResult]
    total: int
    query: str
    search_time_ms: float
    search_method: str
```

The route handler in `routes.py` for week 1:

```python
@router.post("/api/v1/search", response_model=SearchResponse)
async def search_trials(request: SearchRequest, req: Request) -> SearchResponse:
    es_index = req.app.state.es_index    # loaded at startup
    t0 = time.perf_counter()
    raw = es_index.search(query=request.query, filters=request.filters,
                          top_k=request.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = [TrialResult(
        nct_id=r["nct_id"], title=r["title"],
        conditions=[c.strip() for c in r.get("conditions", "").split(";") if c.strip()],
        phase=r.get("phase"), status=r.get("status"),
        score=r["score"],
        url=f"https://clinicaltrials.gov/study/{r['nct_id']}",
    ) for r in raw]

    return SearchResponse(results=results, total=len(results),
                          query=request.query, search_time_ms=elapsed_ms,
                          search_method="bm25")
```

There are also `GET /api/v1/trial/{nct_id}` (full details for one trial)
and `GET /health` (a liveness probe — return `{"status": "ok"}` if the
process is alive).

The startup wiring lives in `app.py` via FastAPI's `lifespan`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.es_index = ElasticsearchIndex(es_url=ES_URL, index_name="trials")
    yield
    app.state.es_index.es.close()
```

Run it: `uvicorn TrialMine.api.app:app --reload --port 8000`. Now the
search engine is reachable over HTTP.

### Step 7 — Streamlit UI on top

**File:** `src/TrialMine/ui/app.py`

A FastAPI endpoint isn't a *demo*. Real humans need a web page. The
fastest way to get one in Python is **Streamlit**: every `st.*` call
draws a widget; the script reruns top-to-bottom on every interaction.
No JavaScript, no React, no API client.

The week 1 UI is a single screen with:

- A title and a description.
- A big search box.
- A row of three example-query buttons that pre-fill the search box.
- A sidebar with status and phase dropdown filters.
- Results displayed as cards: bold title (linked to ClinicalTrials.gov),
  conditions as tags, status and phase badges.

Communication with the API is via `httpx`:

```python
import httpx
response = httpx.post("http://localhost:8000/api/v1/search",
                      json={"query": query, "top_k": 20, "filters": filters})
data = response.json()
for trial in data["results"]:
    st.markdown(f"### [{trial['title']}]({trial['url']})")
    st.caption(f"NCT: {trial['nct_id']} • Phase: {trial['phase']} • Status: {trial['status']}")
    st.write(", ".join(trial["conditions"]))
```

That's the whole demo for week 1. Type a query, see ranked trials, click
to ClinicalTrials.gov. About as primitive as a demo gets, but every later
week is iterating on this same skeleton.

---

## Part 5 — End-of-week file map (where each thing lives)

| File | Purpose | LOC |
|---|---|---|
| `pyproject.toml` | Project deps, entry points | ~70 |
| `Makefile` | Short names for common commands | ~50 |
| `docker-compose.yml` | Elasticsearch container (week 1) | ~15 |
| `configs/development.yaml` | DB path, ES host, model paths | ~25 |
| `src/TrialMine/data/models.py` | `Trial` and `Location` Pydantic | ~40 |
| `src/TrialMine/data/download.py` | API + pagination + resume | ~160 |
| `src/TrialMine/data/parse.py` | JSON → Trial objects | ~190 |
| `src/TrialMine/data/store.py` | SQLAlchemy ORM + insert/load | ~170 |
| `src/TrialMine/retrieval/bm25.py` | Elasticsearch wrapper | ~240 |
| `src/TrialMine/api/schemas.py` | Pydantic request/response | ~90 |
| `src/TrialMine/api/routes.py` | `POST /search`, `GET /trial/{id}` | ~270 |
| `src/TrialMine/api/app.py` | FastAPI app + lifespan | ~220 |
| `src/TrialMine/ui/app.py` | Streamlit search UI | ~100 (week 1) |
| `scripts/download_data.py` | End-to-end ingestion driver | ~100 |
| `scripts/build_index.py` | Build Elasticsearch (and FAISS later) | ~260 |

The data lives in `data/` (gitignored): `data/raw/page_*.json` and
`data/trials.db`.

---

## Part 6 — Why we made the choices we did

These are the design decisions logged in `docs/design-decisions.md`.

### Decision 1 — ClinicalTrials.gov API v2 over XML bulk download

**Two options.** The CT.gov API v2 returns paginated JSON. A separate
endpoint serves a single ~3 GB XML file with everything.

**We chose API v2.** Three reasons:
- JSON is easier to parse than XML.
- We can filter to oncology trials *at the source* (`query.term=cancer
  OR ...`) instead of downloading 500K trials and discarding 70%.
- Resume on interruption is built in (pageToken).

**Trade-off.** Slower than one big download. ~75 minutes vs ~10. Worth
it because the pipeline is cleaner and we don't have to deal with XML.

**At scale.** A production system would probably run the bulk XML once
and use the API for nightly incremental updates. That's overkill for
us.

### Decision 2 — SQLite over PostgreSQL

**Choice.** SQLite. One file. Zero install. SQL for debugging.

**Why not PostgreSQL?** We're a single-machine, single-writer system.
PostgreSQL gives us concurrent writes, replication, role-based access
control — none of which we need. Adding a service to docker-compose
just to run PostgreSQL would be cognitive overhead with no payoff.

**At scale.** When we get to multi-writer or want to share data across
machines, switch to managed PostgreSQL. The SQLAlchemy ORM means our
application code doesn't change.

### Decision 3 — Elasticsearch for BM25

**Choice.** Elasticsearch 8.x.

**Why Elasticsearch and not Whoosh, Lucene, Tantivy, Meilisearch?**
- Industry standard — every senior engineer recognizes it.
- Field boosting and analyzers built in.
- One Docker line to run.
- Scales to millions of documents per shard.

**Trade-off.** ~1 GB of RAM overhead just to keep the JVM alive. For a
laptop, that's a real cost.

**At scale.** Replace with managed Elasticsearch (AWS OpenSearch,
Elastic Cloud) and use index sharding to split across nodes. Our app
code uses `Elasticsearch(es_url=...)` so the URL is the only thing that
changes.

### Decision 4 — FastAPI over Flask / Django

**Choice.** FastAPI.

**Why?**
- Pydantic validation is built in — no `request.json` parsing dance.
- Auto-generated OpenAPI docs at `/docs`.
- Async-native (we'll need this in week 6 for the agent pipeline).
- Modern Python idioms — type hints, dependency injection.

**Trade-off.** Less mindshare than Flask/Django for older codebases.
For new projects in 2024+, FastAPI is the default.

### Decision 5 — Keep trials with missing fields (don't drop them)

**Choice.** Require `nct_id` and `title`. Allow every other field to
be `None`.

**Why?** Missing data is not the same as irrelevant data. A trial whose
eligibility criteria is empty might still be the most relevant trial
for the patient — perhaps because the trial is brand new and the
criteria haven't been written yet, or because the data extraction at
CT.gov dropped it. Filtering it out punishes the patient for an
upstream data problem.

**How we make this safe.** BM25 naturally penalizes sparse documents
(the `all_text` field is short, so the score is low). Our UI shows
fewer "match details" for sparse trials. Downstream, the eligibility
parser (week 5) explicitly emits *Unknown* verdicts when fields are
missing.

---

## Part 7 — Interview prep

What an MLE interviewer might actually ask about this work.

### 7.1 The big design question

> *"Walk me through how you'd build a search engine for clinical trials."*

A good answer covers:

1. **Clarify scope.** How many documents? Latency target? Update
   cadence? User vocabulary?
2. **Sketch the pipeline.** Source → ingest → store → index → query →
   present. Each step has a service.
3. **Choose tools per layer.** API for ingest, SQLite or Postgres for
   primary store, Elasticsearch (or Lucene-based equivalent) for
   inverted index, FastAPI for HTTP, Streamlit/React for UI.
4. **Handle quality.** Rate-limited downloads with resume. Schema
   validation at the seam (Pydantic). Field boosting for relevance
   tuning. Filters for exact-match constraints.
5. **Plan for scale.** Bulk download for first hydration, API for
   incremental updates. Index sharding when corpus exceeds one node.
   Read replicas for query throughput.

### 7.2 Specific questions

**"Why BM25 and not just regex matching?"**
BM25 ranks by relevance. Regex is binary (match or not). For 140K
documents, you need ranking — most users only look at the top 10.

**"What does the field boost actually do?"**
A multi-match query computes per-field BM25 scores and combines them.
With `title^3`, a title hit is worth 3× the BM25 contribution of a
non-boosted field. Critical for clinical trials because the title is
heavily curated.

**"How would you make this scale to millions of trials?"**
Three changes. (1) Move the bulk download offline (one-time XML pull
+ nightly API delta). (2) Shard the Elasticsearch index — say, 8
shards across 4 nodes. (3) Move the storage layer from SQLite to
PostgreSQL or BigQuery so multiple workers can write concurrently.

**"What if the API changes the field paths?"**
The `parse_study` function would silently start producing `None`
values. We'd see the change in the parsing log
(`missing_eligibility: 50%` jump) and our test queries would degrade.
Fix: a small daily smoke test that parses 10 known trials and checks
their fields against pinned values.

**"Why save raw JSON files instead of parsing on the fly?"**
*Separate I/O from compute.* If the parser has a bug, we re-run it
locally on the saved files. If we parsed on-the-fly, we'd have to
re-hit the API to fix bugs — slow, wasteful, possibly rate-limited.

### 7.3 Code-review questions

1. **Why does `_get` return `default` when the final value is `{}`?**
   Look at the implementation: at every step we do `d = d.get(key, {})`.
   That `{}` is a placeholder so the *next* `.get()` call doesn't crash
   on `None`. But if the very last key was missing, the final value is
   `{}` — which is not what the caller asked for. So we explicitly
   convert `{}` → `default` (usually `None`) at the end. Without this
   the caller would receive `{}` and treat it as a valid empty value.
2. **Why is `nct_id` a `keyword` field in Elasticsearch and not `text`?**
   Because we look it up by exact match (`{"term": {"nct_id": "NCT04..."}}`).
   `text` would tokenize it, lowercase it, possibly stem it — all bad for
   exact-match.
3. **Why dedupe before insert instead of catching `IntegrityError`?**
   Asking is cheaper than failing. We do one `SELECT nct_id FROM trials`
   to get the existing set, then filter the new batch in Python. Catching
   exceptions inside a loop adds latency and clutters the code.
4. **Why is the search response `Pydantic` if FastAPI already validates JSON?**
   Two reasons. (1) FastAPI uses Pydantic *under the hood* — the schema
   is the validator. (2) The schema is also our API contract, exported
   to the OpenAPI doc. If we changed `score: float` to `score: int`,
   every consumer would see the schema change in their generated client.

### 7.4 Tradeoff questions

**"BM25 scores are unbounded and depend on corpus size. Doesn't that make
them hard to compare across systems?"**
Yes. BM25 gives a relative ordering, not an absolute relevance
probability. We never compare scores across queries or corpora
directly. For systems that need absolute scores, calibrate against
labeled data — that's exactly what week 4's LightGBM does.

**"What if patients type queries in lay language ('stomach' instead of
'gastric')?"**
BM25 won't bridge that gap — it matches words literally. Week 2 adds
semantic search (embedding-based) which understands meaning. Week 5
adds a concept normalizer (lay → medical synonym dictionary).

---

## Part 8 — How to run everything

Assuming Python 3.11+, Docker, and ~1 GB of free disk:

```bash
# One-time setup
git clone <repo> && cd TrialMine
make setup                                 # pip install -e ".[dev]"
docker compose up -d elasticsearch         # week 1 only needs ES
cp .env.example .env                       # ANTHROPIC_API_KEY can wait until week 3

# Download + parse + store (~75 min)
make download

# Build the BM25 index (~80 sec)
python scripts/build_index.py --skip-semantic

# Start the API (one terminal)
make serve

# Start the UI (another terminal)
make ui
# Open http://localhost:8501
```

Try the search box with:
- `breast cancer phase 3` — should return PHASE3 trials about breast cancer.
- `lung cancer immunotherapy recruiting` — try the status filter.
- `asdfghjkl` — gibberish should return zero or near-zero results.

---

## Part 9 — What's next (week 2 preview)

BM25 is purely lexical — it can only match literal words. The
*vocabulary mismatch* problem from Part 1 (stomach cancer ↛ gastric
adenocarcinoma) will not go away with better field boosting; we need
a method that understands **meaning**, not just words.

Week 2 adds **semantic search** — turn each trial into a 768-number
"embedding" using a biomedical language model (BioLinkBERT), store the
embeddings in **FAISS** for fast nearest-neighbor lookup, then merge
the BM25 results and the semantic results into a single ranked list
via **Reciprocal Rank Fusion** (RRF).

---

## Part 10 — End of week 1 checklist

- [x] Project skeleton with all top-level dirs
- [x] 140,723 oncology trials downloaded from ClinicalTrials.gov v2
- [x] Trials parsed into typed `Trial` Pydantic objects
- [x] Trials persisted in `data/trials.db` (SQLite, ~912 MB)
- [x] Elasticsearch BM25 index with field boosting (title 3×, conditions 2×)
- [x] FastAPI with `POST /api/v1/search`, `GET /api/v1/trial/{nct_id}`,
      `GET /health`
- [x] Streamlit UI with search box, example buttons, status/phase filters
- [x] 5 design decisions written in `docs/design-decisions.md`
- [x] CLAUDE.md updated with current state
- [x] First commits pushed to GitHub

That's the full week. If you can explain Parts 4 and 6 in your own
words, you've internalized the foundation.
