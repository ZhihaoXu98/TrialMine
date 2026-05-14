# Appendix A — Pydantic and the API Layer

> A short, hands-on tour of how Pydantic and FastAPI work together in
> TrialMine. By the end, you should be able to read any file in
> `src/TrialMine/api/` and explain what it's doing — and why.
>
> This is a reference appendix, not part of the numbered chapter
> sequence (01–09). Read it when you need to work in `api/` or when
> you're preparing for an interview question on production serving.

---

## What you'll learn

1. What Pydantic actually *does*, and why we use it everywhere.
2. The two layers of models in this project (internal vs. API boundary)
   and why they're kept separate.
3. How FastAPI uses Pydantic to validate requests and shape responses
   automatically.
4. The roles of the three files in `src/TrialMine/api/`.
5. The full request → response lifecycle, end to end.

---

## Chapter 1 — Pydantic in 5 minutes

### The problem Pydantic solves

Imagine the API receives this request body from a client:

```json
{"query": "pembrolizumab", "top_k": "twenty"}
```

`top_k` should be an integer, but the client sent a string. Without
Pydantic, you'd write defensive code in every handler:

```python
# Hand-rolled validation — verbose, easy to forget
def search(body):
    if "query" not in body:
        return {"error": "query required"}, 422
    if not isinstance(body["query"], str):
        return {"error": "query must be a string"}, 422
    if "top_k" in body and not isinstance(body["top_k"], int):
        return {"error": "top_k must be an integer"}, 422
    if body.get("top_k", 0) < 1 or body.get("top_k", 0) > 100:
        return {"error": "top_k must be 1-100"}, 422
    # ... and now finally the business logic
```

Pydantic replaces all of that with a class declaration:

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(20, ge=1, le=100)
```

Now if a client sends bad data, Pydantic raises `ValidationError`
*before* your handler runs, with a clear message naming the failing
field.

### What a Pydantic model gives you

When you write `class Foo(BaseModel)`, you get four things for free:

1. **Type-checked construction.** `Foo(x=1, y="hi")` validates types and
   raises if they don't match (or coerces compatible scalars — see §1.3).
2. **JSON serialization.** `foo.model_dump_json()` returns a valid JSON
   string. `foo.model_dump()` returns a `dict`.
3. **JSON parsing.** `Foo.model_validate_json(raw)` builds a `Foo` from
   a JSON string, validating along the way.
4. **A schema.** `Foo.model_json_schema()` returns a JSON Schema document
   describing the shape — used by FastAPI to auto-generate API docs.

### What "validation" actually does (and doesn't do)

Pydantic v2 runs in **lax mode** by default — it coerces compatible
types instead of strictly rejecting them.

| Input | Field type | Behavior |
|---|---|---|
| `"500"` | `int` | Coerced to `500` ✓ |
| `500` | `str` | Coerced to `"500"` ✓ |
| `"twenty"` | `int` | **Raises** `ValidationError` |
| `["a"]` | `str` | **Raises** |
| missing required field | (any) | **Raises** |
| extra unknown field | (any) | **Silently ignored** by default |

So Pydantic catches *structural* problems (missing fields, wrong
container types, unparseable values) but is forgiving about minor type
differences. That's usually what you want at an API boundary — clients
shouldn't fail because they sent `"20"` instead of `20`.

### Useful `Field()` constraints

```python
query: str = Field(..., min_length=1)        # required, non-empty
top_k: int = Field(20, ge=1, le=100)         # default 20, in [1, 100]
method: Literal["bm25", "semantic", "hybrid"] = "hybrid"   # enum
warnings: list[str] = Field(default_factory=list)  # default []
```

`...` means "required, no default". `default_factory=list` is the right
way to default to a mutable value (never write `= []` as a default — it
shares the list across instances).

---

## Chapter 2 — Two layers of models

In TrialMine we have **two** separate sets of Pydantic models. This
separation is deliberate.

### Layer 1: internal models — `src/TrialMine/data/models.py`

```python
class Trial(BaseModel):
    nct_id: str
    title: str
    brief_summary: str | None
    detailed_description: str | None
    conditions: list[str]
    interventions: list[str]
    eligibility_criteria: str | None
    min_age: str | None
    max_age: str | None
    sex: str | None
    phase: str | None
    status: str | None
    enrollment: int | None
    start_date: str | None
    completion_date: str | None
    sponsor: str | None
    locations: list[Location]
    url: str
```

This is the **canonical, full-fat** representation of a trial used by
the parser, the database layer, the indexer, and the retriever. It has
every field we extract from ClinicalTrials.gov.

### Layer 2: API boundary models — `src/TrialMine/api/schemas.py`

```python
class TrialResult(BaseModel):
    nct_id: str
    title: str
    conditions: list[str]
    phase: str | None
    status: str | None
    score: float
    url: str | None
    source: str | None
    bm25_rank: int | None
    semantic_rank: int | None
    explanation: str | None
    eligibility: dict | None
    warnings: list[str]
```

This is the **projection** of a trial that the API exposes. Notice it
has a `score` field (only meaningful at search time, not in the DB) and
omits fields like `sponsor`, `start_date`, `locations` (the search UI
doesn't need them).

### Why keep them separate?

- **Internal models can change without breaking clients.** If we add an
  `arms` field to `Trial` for future use, the API contract is unchanged.
- **API models can carry fields that don't exist internally.** `score`,
  `bm25_rank`, `explanation`, `agent_trace` are computed at request time
  — they don't belong in `Trial`.
- **Security / privacy.** Internal models might hold fields you don't
  want to leak to clients. Separating them means accidental exposure
  requires deliberate work, not just `return trial`.

### Mental model

```
ClinicalTrials.gov JSON
        │
        ▼
   parse.py → Trial (internal, full-fat)
        │
        ▼
   stored in SQLite + indexed in ES + FAISS
        │
        ▼
   retrieved as Trial
        │
        ▼
   routes.py builds TrialResult (boundary, slimmed)
        │
        ▼
   FastAPI serializes TrialResult → JSON → client
```

---

## Chapter 3 — How FastAPI uses Pydantic

FastAPI is the glue. When you declare a route, you tell it the request
type and the response type:

```python
@router.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    ...
```

That single line buys you four things:

### 1. Automatic request validation

When a request arrives, FastAPI:
1. Reads the JSON body
2. Calls `SearchRequest.model_validate(body)`
3. If it succeeds → passes the typed object as `request` to your handler
4. If it fails → returns `422 Unprocessable Entity` with a JSON body
   listing every field that failed and why

You never write `if not body.get("query"): return error` again.

### 2. Automatic response serialization

When your handler returns a `SearchResponse`, FastAPI:
1. Calls `response.model_dump()` to convert to a `dict`
2. JSON-encodes the dict
3. Sets `Content-Type: application/json`
4. Sends it

You never write `return jsonify(...)`.

### 3. Auto-generated OpenAPI docs

Hit `http://localhost:8000/docs` while the API is running. You'll see a
Swagger UI built **entirely from your Pydantic classes** — every field,
every constraint, every docstring. No separate doc file to maintain.

### 4. Type safety inside your handler

Inside `search`, your IDE knows `request.top_k` is an `int` and
`request.method` is a `Literal["bm25", "semantic", "hybrid"]`. Typos and
wrong types are caught by mypy / your editor before runtime.

---

## Chapter 4 — The three files in `api/`

### `schemas.py` — the contract

What goes over the wire. Every request and response type lives here.

- `SearchRequest` — POST /api/v1/search input
- `SearchResponse` — POST /api/v1/search output
- `TrialResult` — one trial inside `SearchResponse.results`
- `TrialDetailResponse` — GET /api/v1/trial/{nct_id} output
- `ErrorResponse` — used everywhere errors go out (so we never return a
  raw 500)

Treat this file as **public**. Renaming a field here breaks every client.

### `routes.py` — the handlers

Thin adapters between HTTP and Python. Each function:
1. Receives a typed request object
2. Calls into retrieval / agent code
3. Returns a typed response object

Business logic does **not** live here. If a handler is more than ~30
lines, something has leaked in that should be in the orchestrator or
retriever.

### `app.py` — the wiring

The FastAPI application factory. It:
- Creates the `FastAPI()` instance
- Mounts the router from `routes.py`
- Adds CORS middleware
- Adds `/metrics` for Prometheus
- Runs the **lifespan**: at startup, loads the heavy resources
  (Elasticsearch client, FAISS index, embedder, cross-encoder, LightGBM,
  agent pipeline) once, runs a warmup query so the first real request
  isn't slow, and tolerates Elasticsearch being down (degraded mode
  instead of crash).

**Mental shorthand:** `schemas.py` = *what*, `routes.py` = *do*,
`app.py` = *boot*.

---

## Chapter 5 — One full request, end to end

Let's trace exactly what happens when the Streamlit UI sends a search.

### Step 1 — UI sends JSON

```python
# In src/TrialMine/ui/app.py
response = httpx.post(
    "http://api:8000/api/v1/search",
    json={"query": "pembrolizumab NSCLC", "top_k": 10, "use_agent": True},
)
```

The wire payload:
```http
POST /api/v1/search HTTP/1.1
Content-Type: application/json

{"query": "pembrolizumab NSCLC", "top_k": 10, "use_agent": true}
```

### Step 2 — FastAPI parses + validates

FastAPI sees the route declaration `async def search(request: SearchRequest)`
and runs:

```python
request = SearchRequest.model_validate({
    "query": "pembrolizumab NSCLC",
    "top_k": 10,
    "use_agent": True,
})
```

If `top_k` had been `0`, this raises and the client gets `422`
immediately. Otherwise the handler runs.

### Step 3 — Handler dispatches

```python
async def search(request: SearchRequest) -> SearchResponse:
    if request.use_agent:
        result = await agent_pipeline.search(request.query, top_k=request.top_k)
    else:
        result = hybrid_retriever.full_pipeline(request.query, top_k=request.top_k)

    return SearchResponse(
        results=[TrialResult(**r) for r in result["results"]],
        total=len(result["results"]),
        query=request.query,
        search_time_ms=result["timings"]["total_ms"],
        search_method="agent" if request.use_agent else request.method,
        timings=result["timings"],
        agent_trace=result.get("agent_trace"),
        used_fallback=result.get("used_fallback", False),
        patient_profile=result.get("patient_profile"),
    )
```

### Step 4 — Pydantic validates the response too

Building `SearchResponse(...)` runs validation again — this time on the
data **we** produced. If we accidentally pass `total="5"` (a string),
Pydantic catches it before the bug reaches the client.

### Step 5 — FastAPI serializes + returns

```python
response.model_dump()
# → {"results": [...], "total": 10, ...}

json.dumps(...)
# → '{"results": [...], "total": 10, ...}'

# Sent over the wire with Content-Type: application/json
```

### Step 6 — UI parses the JSON back into Python

```python
data = response.json()  # dict
for trial in data["results"]:
    st.write(trial["title"])
```

(The UI doesn't use Pydantic to parse — it just treats the response as
a dict. That's fine for a single consumer; if we had many clients we'd
publish the schema and let them generate typed clients from the OpenAPI
spec.)

---

## Chapter 6 — Why this matters for production

The Pydantic + FastAPI pattern is doing more than saving keystrokes. It
gives us four guarantees:

1. **No 500s for bad input.** Bad requests get `422` with a clear
   message, never an unhandled exception.
2. **No bad responses.** We can't accidentally return a malformed
   payload — Pydantic validates outgoing data too.
3. **Docs that can't go stale.** OpenAPI is generated from the same
   classes the code uses. Update the code, the docs update with it.
4. **One canonical contract.** Clients, server, tests, and docs all
   agree on the same Pydantic models. There is no second source of
   truth to drift.

These four properties are why CLAUDE.md mandates *"Pydantic models for
ALL data structures"* and *"NEVER let the API return 500"*. The rules
sound like style guidance; they're actually load-bearing infrastructure
for the production guarantees above.

---

## Check yourself

1. What's the difference between `data/models.py` and `api/schemas.py`?
   Why are they separate?
2. If a client sends `{"query": "", "top_k": 5}`, what happens, and
   where exactly does the rejection occur?
3. If a client sends `{"query": "hi", "top_k": "5"}` (string instead of
   int), what happens?
4. Which of the three files in `api/` would you edit to add a new
   `min_age` filter? (Hint: more than one.)
5. Why do we return `ErrorResponse` instead of just letting exceptions
   propagate?

---

## Further reading

- `src/TrialMine/api/schemas.py` — the actual contract
- `src/TrialMine/api/routes.py` — the handlers
- `src/TrialMine/api/app.py` — startup + lifespan
- `docs/07_deployment.md` — how this whole thing runs in Docker
- Pydantic v2 docs: https://docs.pydantic.dev/latest/
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
