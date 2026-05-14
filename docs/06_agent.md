# Week 6: Building the Agent System — a Student-Friendly Walkthrough

> **How to read this**: imagine I'm sitting next to you at a whiteboard. I'll explain everything from first principles, build up from the easy stuff to the harder stuff, and stop along the way to point out the things an interviewer is likely to ask. By the end you should be able to walk through this system in your own words without notes.
>
> **Total reading time**: 60-90 minutes if you read carefully. 20 minutes for a refresher.

---

## Part 1: What are we actually trying to do?

Let's ground this before we touch any code.

A patient comes to TrialMine and types something like:

> *"I'm 62 with stage 3 non-small cell lung cancer, tried carboplatin, looking for immunotherapy near Boston."*

What do we want to give them back?

1. **Trials that match** their condition, ranked best-first.
2. **Eligibility verdicts** — for each trial, do they likely qualify? Met / Unmet / Unknown for age, sex, prior treatments, etc.
3. **A short explanation** of why each trial is or isn't a fit.

Before this week, we had all the *ingredients*:

- Elasticsearch for keyword search (BM25)
- FAISS for semantic search (with a fine-tuned BioLinkBERT model)
- A cross-encoder for re-ranking
- A LightGBM ranker that combines all the signals
- A SciSpacy + regex eligibility parser
- A concept normalizer for translating "stomach cancer" to "gastric cancer"

But none of that is hooked up end-to-end. A patient typing the query above just got a list of trial titles from BM25 — no eligibility check, no explanation, no understanding of *what they actually meant*. Week 6's job: turn the ingredients into a meal.

The deliverable is one async function:

```python
result = await search(patient_description, pipeline)
# result["search_results"]["results"]              → ranked trials
# result["search_results"]["results"][i]["explanation"]  → a sentence about each trial
# result["agent_trace"]                            → step-by-step log
# result["used_fallback"]                          → did the degraded path run?
```

That's an "agent." But what does that word actually mean? Let's slow down.

---

## Part 2: What's an "agent," really?

Forget the marketing for a second. **An agent is just three things put together**:

1. **An LLM** — a model that reads and writes natural language.
2. **Tools** — functions the system can call (search, lookup, check, etc.).
3. **A control flow** — code that decides what runs when, in what order.

Think of a chef in a kitchen:

- The chef is the **LLM** (the brain).
- The knives, pots, oven are the **tools** (the hands).
- The recipe is the **control flow** (the plan).

Some kitchens have a chef who improvises — they look at what's in the fridge and decide what to cook. Others have a strict recipe — step 1, step 2, step 3. Both are "kitchens." Both make food. The difference is *who* makes the decisions: the chef's intuition, or the written-down plan.

In agent-land, those two styles have names:

| Style | Who decides what to run | When you'd want it |
|---|---|---|
| **ReAct agent** (improvising chef) | The LLM picks the next tool every step | Open-ended tasks. Research. "Find me information about X" where you don't know what X requires upfront. |
| **Rule-based agent** (strict recipe) | Python code; the LLM is used only at specific narrow steps | Tasks with a known fixed flow. Search engines. Data pipelines. Anything you could draw on a whiteboard before writing it. |

**TrialMine is the second kind**. Our flow is always:

1. Understand the patient's query (parse it).
2. Search for trials.
3. Check eligibility on the top results.
4. Explain.

There's no "the agent might want to take a left turn" decision. We can write the steps down before we run them. So the right shape is rule-based.

> **Why this distinction matters for interviews**: a really common pitfall is to use a ReAct agent for everything because "agents are cool." A good interview answer is to *recognize* that your task has a fixed flow and *choose* the simpler architecture. The follow-up question will be: "okay, but couldn't ReAct still work?" — and you should have a real answer (yes, but at 5× the latency and 10× the cost; we'll get to the numbers).

---

## Part 3: The pivot — why we threw out our first plan

This is the kind of story interviewers love because it shows engineering judgment.

**Our first plan** was a ReAct agent using `langgraph.prebuilt.create_react_agent`. The LLM would:
- Receive the patient's query.
- Pick a tool (maybe `lookup_medical_concept` first, then `search_trials`, then `check_trial_eligibility`).
- Read each tool's output.
- Decide what to do next.
- Eventually return a final answer.

This is the "improvising chef" style. Sounds smart. Has problems.

| Property | ReAct agent | Rule-based |
|---|---|---|
| **Latency per query** | 5–15 seconds | ~3 seconds |
| **Cost per query** | $0.02–$0.03 | ~$0.002 |
| **Predictability** | LLM might do something weird | Same path every time |
| **Debuggability** | Read a chain of LLM messages | Read structured trace |
| **Quality on this task** | Marginally better | Sufficient |

Run the math. For a free public-health tool serving 1,000 queries a day:
- ReAct: $25/day, ~10s wait per query.
- Rule-based: $2/day, ~3s wait per query.

The "marginally better quality" of ReAct doesn't justify either of those gaps. So we pivoted.

> **Interview tip**: when you're asked "why did you pick X?", the strongest answer is *"we considered Y, weighed the tradeoff, and picked X because [concrete reason]."* The weakest is *"it's the standard tool."* Always have the alternative in your back pocket.

---

## Part 4: The big picture (one diagram, one paragraph)

Here's the whole system in a diagram:

```
                    POST /api/v1/search?use_agent=true
                                  │
                                  ▼
           ┌────────────────────────────────────────┐
           │  pipeline.search()  (15s timeout cap)  │
           └────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────┐
                    │ LangGraph StateGraph │
                    └──────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────┐
        │   parse_query  (Claude Haiku 4.5 LLM)    │
        │   raw_query → PatientProfile             │
        └──────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────┐
        │       execute_search                     │
        │  (calls SearchOrchestrator, no LLM)      │
        │  ┌────────────────────────────────────┐  │
        │  │ 1. normalize via ConceptNormalizer │  │
        │  │ 2. build query string              │  │
        │  │ 3. compute filters                 │  │
        │  │ 4. retrieve (BM25+sem+CE+LGB)      │  │
        │  │ 5. eligibility check (parallel)    │  │
        │  │ 6. template explanation per result │  │
        │  └────────────────────────────────────┘  │
        └──────────────────────────────────────────┘
                                  │
              ┌───────success─────┼─────error──────┐
              ▼                                    ▼
            END                          fallback_search
                                                   │
                                                   ▼
                                              END (degraded)
```

**Read it like this**: a request enters at the top. We start a 15-second timer. The LangGraph state machine takes over. `parse_query` calls Claude Haiku to extract structured fields from the patient's text. `execute_search` runs a deterministic 6-step pipeline (no LLM!) and produces results. If it fails, the conditional edge sends the request to `fallback_search`, which runs a degraded-but-still-useful plain hybrid search. Either way, we always return *something* structured.

**The single most important fact about this system**: there is exactly **one LLM call** per request, and it's Claude Haiku at the very top. Everything else is plain Python and ML models. Cheap, fast, predictable.

---

## Part 5: Walking through the code, layer by layer

We'll go from the bottom up: the lowest-level building blocks (tools) first, then up to the orchestration. Each layer answers three questions:

1. **What does it do?**
2. **Why is it designed this way?**
3. **What's the trickiest pattern in here?**

---

### Layer 1: The tools (`src/TrialMine/agents/tools.py`)

**What it does**: wraps our four most-useful operations as `@tool`-decorated functions. Tools are the building blocks the orchestrator (and any future LLM-driven agent) calls when it needs to *do* something.

**The four tools**:

| Tool | What it does |
|---|---|
| `search_trials` | Run the full BM25+semantic+CE+LightGBM pipeline; return top-k trials. |
| `lookup_medical_concept` | Lay→medical synonym expansion (`stomach cancer` → `gastric neoplasm`). |
| `check_trial_eligibility` | Compare a patient profile against a trial's eligibility criteria; return Met/Unmet/Unknown verdicts. |
| `get_trial_details` | Fetch a full trial record by NCT ID (e.g., `NCT04802759`). |

Now the patterns inside this file. There are five worth knowing.

#### Pattern 1: The `@tool` decorator + Pydantic args schema

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchTrialsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Free-text clinical trial search query."
    )
    phase: str | None = Field(
        None,
        description="Optional phase filter. Allowed: 'Phase 1', 'Phase 2', ..."
    )
    top_k: int = Field(
        20, ge=1, le=50,
        description="Number of top results to return."
    )

@tool("search_trials", args_schema=SearchTrialsArgs)
def search_trials(query: str, phase: str | None = None, top_k: int = 20) -> str:
    """Search the ClinicalTrials.gov corpus for trials relevant to a query."""
    ...
```

There's a lot going on. Let's slow down.

The `@tool` decorator turns a regular Python function into a LangChain "tool" — an object with metadata (name, description, schema) that an LLM can be told about. When an LLM-driven agent decides to call this tool, LangChain does the JSON parsing, validation, and invocation for you.

The `args_schema=SearchTrialsArgs` tells LangChain what arguments the tool takes and what each one means. This becomes a JSON Schema sent to the LLM along with the tool's description. The LLM reads this schema to decide *how* to call the tool.

> **Critical insight**: the `description=` strings are not comments for human readers. They're **prompt engineering for the model**. If you write `description="phase"` the LLM has no clue what values are valid. If you write `description="Phase filter. Allowed values: 'Phase 1', 'Phase 2', ..."` the LLM gets it right almost always.

So even though our orchestrator doesn't currently use tools through an LLM, the schemas we wrote are still good documentation for the future and for testing.

#### Pattern 2: Tools always return JSON strings, never Python dicts

```python
@tool("lookup_medical_concept", args_schema=ConceptArgs)
def lookup_medical_concept(term: str) -> str:
    return json.dumps({
        "original": term,
        "normalized": normalizer.normalize(term),
        "expanded": normalizer.expand_query(term),
    })
```

**Why a JSON string?** Because LangChain's tool message protocol requires strings. When an LLM-driven agent receives a tool result, it gets it as text in a message. So we serialize to JSON for structure, then the caller `json.loads()` it.

This feels awkward but it's a hard requirement of the framework, not a choice. Our orchestrator does this dance:

```python
result_json = check_trial_eligibility.invoke({"nct_id": nct, ...})
result = json.loads(result_json)  # parse it back into a dict
```

**Common interview question**: "Why not just return a dict?" → "LangChain's tool message protocol requires string content because tool results are passed as text messages in the conversation. We picked JSON because it's the most structured choice that survives the round-trip."

#### Pattern 3: Error envelopes — never let an exception escape a tool

```python
def _err(message: str) -> str:
    return json.dumps({"error": message})

@tool("search_trials", args_schema=SearchTrialsArgs)
def search_trials(query: str, ...) -> str:
    try:
        # ... actual work ...
        return json.dumps({"results": ranked, ...})
    except Exception as exc:
        logger.exception("search_trials failed")
        return _err(f"search_trials failed: {exc}")
```

Every tool wraps its body in `try/except` and returns `{"error": "..."}` on failure. **Why?**

- If a tool raises, the agent loop crashes. Whether it's our deterministic orchestrator or a future ReAct loop, an unhandled exception means a 500 to the user.
- If a tool returns `{"error": "..."}`, the caller can read it and decide what to do (retry with different args, try another tool, give up gracefully).

This is the same idea as `Result<T, E>` in Rust or Go's `(value, err)` return. **Make failure a value, not a control-flow disruption.**

#### Pattern 4: Lazy module-level singletons for heavy resources

```python
_singletons: dict[str, Any] = {}
_load_failures: set[str] = set()

def _get_hybrid():
    if "hybrid" in _singletons:
        return _singletons["hybrid"]
    from TrialMine.retrieval.hybrid import HybridRetriever
    _singletons["hybrid"] = HybridRetriever(
        bm25=_get_es(), semantic=_get_faiss(), embedder=_get_embedder()
    )
    return _singletons["hybrid"]

def _get_reranker_or_none():
    if "reranker" in _singletons:
        return _singletons["reranker"]
    if "reranker" in _load_failures:
        return None  # cached failure; don't retry
    try:
        from TrialMine.models.cross_encoder import CrossEncoderReranker
        _singletons["reranker"] = CrossEncoderReranker(...)
        return _singletons["reranker"]
    except Exception as exc:
        _load_failures.add("reranker")
        return None
```

**Why?** The cross-encoder model is ~430 MB on disk and takes ~5 seconds to load into memory. The FAISS index is ~412 MB. If we loaded these on every API request, we'd waste minutes per request and run out of RAM in seconds.

So we use the **lazy singleton pattern**: load once, on first use, cache forever. Subsequent calls return the same instance.

There's also a `_load_failures` set. If loading the cross-encoder fails (file not found, corrupt model), we cache the *failure* and stop trying. Otherwise every call would attempt to reload, hammering the filesystem.

> **Common interview question**: "How would you handle expensive initialization in a stateless web service?" → "Module-level singletons with lazy init. First request pays the cost; subsequent ones are free. Cache failures too — otherwise misconfigured services hammer themselves with retry storms."

#### Pattern 5: Permissive filter normalization

```python
def _normalize_filters(phase: str | None, status: str | None) -> dict[str, str]:
    filters: dict[str, str] = {}
    if status:
        filters["status"] = status.strip().upper().replace(" ", "_")
    if phase:
        p = phase.strip()
        if p.lower().startswith("phase ") and "/" not in p:
            p = "Phase " + p.split()[-1]
        filters["phase"] = p
    return filters
```

ClinicalTrials.gov stores statuses as `"RECRUITING"` (uppercase, underscored) and phases as `"Phase 3"` (title case, space). But an LLM might pass `"recruiting"`, `"PHASE 3"`, `"phase 3"` — variations. Rather than reject these, we normalize them. **Be permissive with what you accept; be strict with what you produce.** That's the Postel principle from networking, and it applies to LLM-facing APIs too.

---

### Layer 2: The query parser (`src/TrialMine/agents/query_parser.py`)

**What it does**: takes the patient's free-text query and returns a structured `PatientProfile` with typed fields (condition, age, sex, biomarkers, etc.). This is the **only** place in the entire pipeline where an LLM is called.

**Why a separate component?** Because pulling structured data out of free text is a classic LLM job — it's what they're great at. Trying to do it with regex would be brittle (`"my daughter is 8"` should set `sex="Female"` and `age=8` — that's three pieces of inferred information). Trying to do it with a bigger model (Sonnet, Opus) is overkill. Haiku is fast, cheap, and accurate enough.

#### The schema

```python
class PatientProfile(BaseModel):
    condition: str | None = Field(None, description="...")
    condition_stage: str | None = Field(None, description="...")
    age: int | None = Field(None, ge=0, le=130)
    sex: Literal["Male", "Female"] | None = None
    prior_treatments: list[str] = Field(default_factory=list)
    biomarkers: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    location: str | None = None
    raw_query: str  # always populated by the agent, never the LLM
```

A few things to notice:

- **`age: int | None` with `ge=0, le=130`**: Pydantic enforces this. If the LLM hallucinates `age=200`, we get a validation error and fall back gracefully.
- **`sex: Literal["Male", "Female"] | None`**: this becomes an enum constraint in the JSON Schema. The LLM is *told* it can only output those exact strings.
- **`raw_query`**: always populated, even when the LLM fails. If everything else is empty, we still have the original text to fall back on.

#### Pattern 6: `messages.parse()` — structured output the right way

There are two ways to get structured output from Claude:

**The hard way** (manual JSON parsing):
```python
response = client.messages.create(
    model="claude-haiku-4-5",
    messages=[...],
)
text = response.content[0].text
data = json.loads(text)            # might fail (extra prose around the JSON)
profile = PatientProfile.model_validate(data)  # might fail (wrong types)
```

**The easy way** (`messages.parse`):
```python
response = client.messages.parse(
    model="claude-haiku-4-5",
    output_format=_ExtractedFields,  # Pydantic class
    messages=[...],
)
profile = response.parsed_output  # already a typed Pydantic instance
```

`messages.parse()` constrains Claude to output JSON matching the schema (using tool-use under the hood) and validates it for you. **Always use `messages.parse()` when you want structured output.** It's both more reliable and less code.

#### Pattern 7: The two-schema trick

Look closely at what we did:

```python
class _ExtractedFields(BaseModel):
    """What the LLM produces — no raw_query."""
    condition: str | None = None
    condition_stage: str | None = None
    # ... no raw_query here ...

class PatientProfile(BaseModel):
    """What the agent passes around — has raw_query."""
    # ... same fields plus ...
    raw_query: str = Field(...)
```

The LLM only sees `_ExtractedFields`. The agent (your code) attaches `raw_query` after parsing.

**Why?** Token efficiency. If `raw_query` were in the schema sent to the LLM, the model would feel obligated to echo it back in its output. The query is ~50-200 tokens; echoing it would double our output cost. Splitting the schema saves real money at scale.

This is a small example of a bigger pattern: **don't make the LLM generate data that the application already has.** Generate only what's new.

#### Pattern 8: Always return a profile, even on failure

```python
def parse(self, raw_query: str) -> PatientProfile:
    if not raw_query or not raw_query.strip():
        return PatientProfile(raw_query=raw_query or "")
    try:
        response = self._client.messages.parse(...)
        return PatientProfile(**response.parsed_output.model_dump(), raw_query=raw_query)
    except Exception as exc:
        logger.warning("QueryParser failed (%s); returning raw_query-only profile", exc)
        return PatientProfile(raw_query=raw_query)
```

Notice there's no `raise` anywhere. Every code path returns a `PatientProfile`. If parsing fails, we return one with only `raw_query` filled in.

**Why?** Because the next step in the pipeline shouldn't have to know *whether* parsing succeeded. It just gets a profile and works with it. If `condition` is `None`, the orchestrator falls back to using `raw_query`. If `age` is `None`, the eligibility check returns `Unknown` for the age criterion. Quality degrades smoothly.

This is the **null object pattern**. Instead of returning `None` and forcing every caller to check, you return a special "empty but valid" object that the caller can use as if it were normal.

#### Measured behavior

| Query | Latency (warm) | Cost | Slots populated |
|---|---|---|---|
| `"breast cancer trials"` | 1.4 s | $0.0017 | 1 (condition only) |
| `"My daughter is 8 and was diagnosed with ALL leukemia"` | 1.3 s | $0.0017 | 3 (condition, age, sex inferred from "daughter") |
| `"ostate cancer, PSA rising after radiation"` | 1.6 s | $0.0017 | 3 (condition typo-corrected, prior treatment, etc.) |
| `"asdfghjkl"` | 1.3 s | $0.0017 | 0 (raw_query only) |

10/10 queries extracted correctly in our test. The cost is dominated by the ~1370-token system prompt; the output is only ~50 tokens. If we shaved the prompt by 30%, we'd get to ~$0.001/query but risk regressing on edge cases like the `ostate → prostate` typo correction. We left it.

---

### Layer 3: The orchestrator (`src/TrialMine/agents/orchestrator.py`)

**What it does**: takes a `PatientProfile` and produces ranked + eligibility-checked trial results with template explanations. **Zero LLM calls inside.** This is pure Python orchestrating the existing ML stack.

This is where the deterministic flow lives. Six steps:

```python
async def search(self, profile: PatientProfile) -> tuple[dict, list[dict]]:
    trace: list[dict] = []

    # Step 1: normalize the condition (microseconds)
    normalized = self._normalize_condition(profile)
    trace.append({"step": "normalize", ...})

    # Step 2: build a search query string (microseconds)
    query = self._build_query(profile, normalized)
    trace.append({"step": "build_query", ...})

    # Step 3: compute filters (microseconds)
    filters = self._build_filters(profile)
    trace.append({"step": "build_filters", ...})

    # Step 4: retrieve trials (heavy — ~5s with cross-encoder)
    ranked, retrieve_timings, pipeline_kind = await asyncio.to_thread(
        self._do_retrieve, query, filters
    )
    trace.append({"step": "retrieve", ...})

    # Step 5: parallel eligibility checks (~5ms across 5-10 trials)
    n_to_check = min(len(ranked), self.eligibility_top_k)
    if n_to_check > 0:
        tasks = [
            asyncio.to_thread(self._check_eligibility, ranked[i]["nct_id"], profile)
            for i in range(n_to_check)
        ]
        partial = await asyncio.gather(*tasks)
        for i, result in enumerate(partial):
            eligibility_results[i] = result
    trace.append({"step": "check_eligibility", ...})

    # Step 6: template explanations per result (microseconds total)
    results = []
    for i, r in enumerate(ranked):
        explanation, warnings = self._build_explanation(r, eligibility_results[i])
        results.append({**r, "explanation": explanation, ...})
    trace.append({"step": "explain", ...})

    return result_dict, trace
```

Let's unpack three things: why it's `async`, the parallelization win, and the template explanations.

#### Pattern 9: Why we made the orchestrator async

The orchestrator looks like a normal sequential function. It walks through six steps, one after the other. So why is it `async`?

**Two reasons.**

**Reason 1: don't block the event loop on the heavy retrieve step.**

`full_pipeline()` does ~5 seconds of cross-encoder inference. If our orchestrator were synchronous, all of FastAPI would freeze for those 5 seconds. No other request could be served. Wrapping the retrieve in `asyncio.to_thread` moves it to the default thread pool — the event loop stays responsive and can handle other concurrent requests.

**Reason 2: parallelize the eligibility checks.**

This is the bigger win. Without parallelization:

```python
# Sequential — each call waits for the previous
for i in range(n_to_check):
    eligibility_results[i] = self._check_eligibility(ranked[i]["nct_id"], profile)
```

With parallelization:

```python
# Parallel — all calls run concurrently
tasks = [asyncio.to_thread(self._check_eligibility, nct, profile) for nct in ncts]
partial = await asyncio.gather(*tasks)
```

We measured the difference on 10 cached eligibility checks:

```
Sequential (10 checks):   143.1 ms total (14.3 ms/check avg)
Parallel    (10 checks):    5.8 ms total ( 0.6 ms/check effective)
Speedup: 24.6×
Saved:   137.3 ms per agent run
Verdicts match (sequential == parallel): True
```

**24× speedup.** The work is I/O-bound (SQLite reads), so threads work fine — Python's GIL releases during I/O. If the work were CPU-bound (like cryptography), we'd need a process pool instead.

> **Common interview question**: *"When does `asyncio.gather` actually parallelize work, and when doesn't it?"* → "`gather` parallelizes whatever you give it. If those are awaitables doing real async I/O (async HTTP, async DB), parallelism is real. If they're sync work, you need to wrap them in `asyncio.to_thread` to move them off the event loop. For CPU-bound sync work, threads don't help because of the GIL — you need `ProcessPoolExecutor`."

#### Pattern 10: Template explanations (no LLM)

```python
def _build_explanation(self, trial, eligibility) -> tuple[str, list[str]]:
    phase = trial.get("phase") or "unspecified-phase"
    status = trial.get("status") or "unknown-status"
    cond_text = "; ".join(...)

    if eligibility is None:
        match_info = "Eligibility: not checked (outside top candidates)"
    else:
        verdict = eligibility.get("verdict", "Unknown")
        n_met = sum(1 for c in criteria.values() if c.get("verdict") == "Met")
        n_unmet = sum(1 for c in criteria.values() if c.get("verdict") == "Unmet")
        n_unknown = sum(1 for c in criteria.values() if c.get("verdict") == "Unknown")
        counts = f"({n_met} met, {n_unmet} unmet, {n_unknown} unknown)"
        if verdict == "Met":
            match_info = f"Eligibility: likely a Match {counts}"
        elif verdict == "Unmet":
            match_info = f"Eligibility: likely NOT eligible {counts}"
        else:
            match_info = f"Eligibility: needs review {counts}"

    explanation = f"This {phase} trial ({status}) for {cond_text}. {match_info}."
    return explanation, warnings
```

Each explanation: ~5 microseconds. 10 results = 50 microseconds total.

If we used an LLM to generate explanations: ~1 second per call × 10 results = **10 seconds extra**. Plus ~$0.05 in LLM costs.

**Tradeoff acknowledged**: template explanations sound robotic. They're not as fluent as LLM-generated prose. But:
- They render in microseconds instead of seconds.
- They cost zero dollars.
- They're 100% deterministic (great for testing).
- The actual signal — "5 criteria met, 0 unmet, 1 unknown" — is what users care about.

> **Interview tip**: when you describe a tradeoff, say what you *gave up* alongside what you got. Saying "we used templates because they're fast" is weak. Saying "we used templates — we gave up prose nuance, in exchange for 10 seconds and 5 cents per query, which was the right tradeoff for a free public-health tool serving thousands of queries a day" is strong.

---

### Layer 4: The pipeline (`src/TrialMine/agents/pipeline.py`)

**What it does**: ties `parse_query` and `execute_search` together with explicit error handling, a fallback path, and a 15-second timeout. This is where LangGraph earns its keep.

#### What's a `StateGraph`?

A `StateGraph` is a finite-state machine where:

- **Nodes** are functions: `state → state_diff` (each node reads the state and returns what changed).
- **Edges** are transitions between nodes (and they can be conditional).
- **State** is a typed dict that flows through every node.
- The graph compiles into a runnable object you can `.ainvoke()`.

Think of it like a flowchart you can actually execute. AWS Step Functions, Airflow DAGs, and Temporal workflows are all variants of this idea.

**Why use one?**

For our case, three concrete reasons:

1. **Conditional routing** is a primitive: "if there's an error, go here; otherwise end."
2. **Accumulating state** (the agent_trace) has a clean reducer pattern.
3. **Each node is independently testable** — you can unit-test a node by passing a fake state in.

#### Pattern 11: The state shape with the `Annotated` reducer trick

```python
import operator
from typing import Annotated, TypedDict

class SearchState(TypedDict):
    raw_query: str
    patient_profile: dict | None
    search_results: dict | None
    agent_trace: Annotated[list[dict], operator.add]  # ← this is the trick
    error: str | None
    used_fallback: bool
```

Most TypedDict fields use **override** semantics. When a node returns `{"foo": 1}`, the new state's `foo` is `1` — replacing whatever was there before.

But `agent_trace` shouldn't override. Every node should *append* to it without erasing what came before.

The `Annotated[list[dict], operator.add]` syntax tells LangGraph "use `operator.add` as the reducer for this field." When a node returns `{"agent_trace": [new_entry]}`, LangGraph computes:

```python
new_state["agent_trace"] = old_state["agent_trace"] + [new_entry]
```

Because `operator.add` on lists concatenates.

This same pattern is how chat-style agents accumulate messages — `MessagesState` uses `Annotated[list, add_messages]` with a custom message reducer.

> **Common interview question**: *"How do you handle state updates from concurrent or sequential nodes in LangGraph?"* → "Annotated reducers. Default semantics is override: a node's diff replaces fields in the state. For accumulating fields like logs or message histories, you wrap the type with `Annotated[T, reducer]`. The most common reducer is `operator.add` for lists; LangGraph also ships `add_messages` for chat-style appends."

#### The three nodes

**Node 1: `parse_query`** — always succeeds.

```python
async def parse_query(state: SearchState) -> dict:
    parser = _get_parser()
    profile = await asyncio.to_thread(parser.parse, state["raw_query"])
    return {
        "patient_profile": profile.model_dump(),
        "agent_trace": [{"step": "parse_query", "duration_ms": ..., "decisions": {...}}],
    }
```

The QueryParser is sync (raw Anthropic SDK). To call it from an async node without blocking, we wrap with `asyncio.to_thread`. The parser itself swallows all errors and always returns a profile — so this node never raises.

**Node 2: `execute_search`** — can fail; on failure, sets `state["error"]`.

```python
async def execute_search(state: SearchState) -> dict:
    orchestrator = _get_orchestrator()
    profile = PatientProfile.model_validate(state["patient_profile"])
    try:
        result_dict, trace_entries = await orchestrator.search(profile)
        return {"search_results": result_dict, "agent_trace": trace_entries}
    except Exception as exc:
        return {
            "error": f"execute_search failed: {type(exc).__name__}: {exc}",
            "agent_trace": [{"step": "execute_search_error", ...}],
        }
```

**Pay attention to this pattern**: instead of letting the exception propagate, we *catch it* and return a state diff that includes `error`. The exception becomes data. Now the next routing decision can be made on that data.

**Node 3: `fallback_search`** — runs only when execute_search failed.

```python
async def fallback_search(state: SearchState) -> dict:
    raw_query = state["raw_query"]
    reason = state.get("error") or "execute_search failed"
    try:
        retriever = _get_hybrid()
        ranked = await asyncio.to_thread(retriever.search, raw_query, 20, None)
        # ... build results without eligibility, without templates ...
        return {"search_results": {...}, "used_fallback": True, "agent_trace": [...]}
    except Exception as exc:
        # Fallback also failed. Return empty + structured error.
        return {
            "search_results": {"results": [], ...},
            "used_fallback": True,
            "error": f"both paths failed — primary: {reason}; fallback: {exc}",
            "agent_trace": [{"step": "fallback_search_error", ...}],
        }
```

Notice: even the fallback has its own try/except. If the fallback fails too (e.g., ES is completely down), we still return an empty `results: []` with a structured error. The user *never* gets a 500.

#### The conditional edge — the heart of the routing logic

```python
def _route_after_search(state: SearchState) -> Literal["fallback_search", "__end__"]:
    return "fallback_search" if state.get("error") else "__end__"

def build_pipeline():
    g = StateGraph(SearchState)
    g.add_node("parse_query", parse_query)
    g.add_node("execute_search", execute_search)
    g.add_node("fallback_search", fallback_search)
    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "execute_search")
    g.add_conditional_edges(
        "execute_search",
        _route_after_search,  # routing function
        {
            "fallback_search": "fallback_search",
            "__end__": END,
        },
    )
    g.add_edge("fallback_search", END)
    return g.compile()
```

Read it like English:

- Start at `parse_query`.
- After `parse_query`, always go to `execute_search`.
- After `execute_search`, look at the state — if there's an `error`, go to `fallback_search`. Otherwise, end.
- After `fallback_search`, always end.

The routing function `_route_after_search` is just a regular Python function that takes the state and returns the name of the next node. LangGraph handles everything else.

#### Pattern 12: The 15-second timeout

```python
async def search(patient_description: str, pipeline, *, timeout: float = 15.0) -> dict:
    initial: SearchState = {
        "raw_query": patient_description,
        "patient_profile": None,
        "search_results": None,
        "agent_trace": [],
        "error": None,
        "used_fallback": False,
    }
    try:
        final = await asyncio.wait_for(pipeline.ainvoke(initial), timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "search_results": {"results": [], ...},
            "used_fallback": True,
            "error": f"pipeline exceeded {timeout}s budget",
            "elapsed_ms": ...,
        }
    return {...}
```

`asyncio.wait_for` is the canonical way to put a wall-clock cap on an async operation in Python. If the inner coroutine doesn't complete within `timeout` seconds, `wait_for` cancels it and raises `TimeoutError`.

We catch the `TimeoutError` and return a degraded response. Even on timeout, the user gets a structured 200 with `used_fallback=True` and an `error` field — never an exception, never a 500.

---

### Layer 5: API integration (`routes.py`, `app.py`)

**What it does**: makes the pipeline accessible via HTTP. This is the layer where the agent meets the rest of the world.

#### The `use_agent` flag

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    filters: dict | None = None
    method: Literal["bm25", "semantic", "hybrid"] = "hybrid"
    use_agent: bool = True  # NEW — default to the agent path
```

Every API request now has a `use_agent` flag. When `True` (the default), the request goes through the agent pipeline. When `False`, it uses the legacy direct-retrieval path. **We kept the legacy path** because:

1. It's useful for A/B testing (compare agent quality vs. plain hybrid).
2. It's a safety valve if the agent path has a bug.
3. The direct path is faster (~50ms) for simple queries that don't need eligibility checking.

#### The route handler

```python
@router.post("/api/v1/search")
async def search_trials(request: SearchRequest, req: Request) -> SearchResponse:
    if request.use_agent:
        return await _search_agent(request, req)

    # ... legacy path: bm25 / semantic / hybrid ...
```

If `use_agent=True`, delegate to `_search_agent`, which calls `pipeline.search(...)` and shapes the response. Otherwise, run the legacy code.

#### Pattern 13: ES-tolerant lifespan

Before this week, the API had a bug: if Elasticsearch wasn't running at startup, the lifespan crashed and uvicorn quit. The whole server wouldn't start. We fixed it:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Each resource loads independently and tolerantly.
    try:
        app.state.es_index = ElasticsearchIndex(es_url=ES_URL, ...)
    except Exception as exc:
        logger.warning("ES unavailable at startup: %s — degraded mode", exc)
        app.state.es_index = None
    # ... same try/except for FAISS, embedder ...
    app.state.pipeline = build_pipeline()
    yield
    if app.state.es_index is not None:
        app.state.es_index.es.close()
```

Now if ES is down, the API still starts. The agent path runs, hits an ES error inside `execute_search`, falls through to `fallback_search` (which also fails because also no ES), and returns a structured 200 with `used_fallback=True, error="..."`. The legacy path returns clear 503s.

> **Engineering principle**: never let a startup failure kill the whole service if you can degrade gracefully. *Fail loudly when you must, fail soft when you can.*

---

## Part 6: The four design decisions, explained simply

These are the four interview-relevant decisions. Each has a short version (one sentence) and a long version (the rationale you'd give if pressed).

### Decision 20: LangGraph over CrewAI / raw function calling

**Short**: deterministic pipeline, not open-ended reasoning.

**Long**: We needed three primitives — typed state flowing through nodes, conditional routing on that state, and accumulating list semantics for our agent_trace log. LangGraph's `StateGraph` gives us all three without bringing along the heavier abstractions of `create_react_agent`. CrewAI is built around multi-agent role-playing (one agent talks to another) — overkill for our single deterministic flow. Raw function calling would make us reinvent the conditional-routing primitive ourselves and hand-roll our own trace accumulator. We picked the smallest abstraction that gave us what we needed.

### Decision 21: Claude Haiku for parsing

**Short**: extraction task, not complex reasoning. Speed + cost.

**Long**: Slot extraction (turning free text into typed fields) is well within Haiku's competence. We tested 10 representative queries — all 10 extracted correctly. Sonnet or Opus would cost 3-10× more for no measurable quality gain. The original aspirational target was "<1s, $0.001/query." We measured 1.3-1.5s warm and $0.0017. Slightly hotter than the napkin claim, but within our 15-second overall budget and cheap enough for the use case. We chose to live with the slight gap rather than aggressively trim the system prompt — the prompt's strict-extraction rules are what give us the typo correction (`ostate` → `prostate`) and the gendered-noun inference (`"my daughter"` → `sex=Female`).

### Decision 22: Two LLM components, not three (template explanations)

**Short**: LLM-per-result is too expensive on a search engine.

**Long**: Generating explanations with an LLM would add 1 second × 10 results = 10 seconds latency per query, and ~$0.05 in LLM costs. For a free public-health tool serving thousands of queries a day, that's $50/day on explanations alone. Templates render in microseconds and surface the actually-important signal: the eligibility verdict counts (`5 met, 0 unmet, 1 unknown`). Patients care more about *whether they qualify* than about flowing prose. If we ever build a clinician-facing version where explanation quality matters more than latency, we'd revisit this — that's a different product with different tradeoffs.

### Decision 23: 15-second timeout + fallback path

**Short**: users always get a structured response. Quality degrades gracefully.

**Long**: We have three layers of robustness. (1) The Anthropic SDK has a 30-second per-call timeout. (2) The orchestrator catches every internal exception and turns it into data (`{"verdict": "Unknown"}`, `state["error"]`). (3) `asyncio.wait_for(timeout=15)` caps the whole pipeline. If everything succeeds, the user gets the full agent path. If `execute_search` fails (e.g., cross-encoder OOM), the conditional edge routes to `fallback_search`, which runs plain hybrid retrieval. If even *that* fails (ES is fully down), we return empty results with a structured error. The contract is: **you always get a 200 with a structured `SearchResponse` body**. Empty `results: []` is a legitimate outcome. The pipeline will not return a 500 or raise an exception under any failure mode we've identified.

---

## Part 7: Engineering principles you'll carry to other projects

These are the patterns from this work that apply far beyond TrialMine.

### Principle 1: Graceful degradation at every layer

Look for `try/except` wrappers throughout the agent system:

- **Tools**: catch all exceptions, return `{"error": "..."}` JSON envelope.
- **`_check_eligibility`**: returns `{"verdict": "Unknown", "error": "..."}` on failure.
- **`QueryParserAgent.parse`**: returns a skeleton `PatientProfile(raw_query=...)` on failure.
- **`execute_search` node**: catches exceptions, sets `state["error"]`.
- **`fallback_search` node**: even *its* try/except returns empty results + structured error.
- **`pipeline.search()`**: wraps everything in `asyncio.wait_for`, catches `TimeoutError`, returns degraded response.

The principle: **failure is data, not a control-flow disruption.** Each layer absorbs its own failures and returns a well-typed sentinel. The system gets dumber instead of crashing.

### Principle 2: Observability via structured trace

Every node in the pipeline appends a structured entry to `agent_trace`:

```python
{
    "step": "retrieve",
    "duration_ms": 4823.5,
    "decisions": {
        "pipeline": "full_pipeline",
        "n_results": 12,
        "retrieve_timings": {"bm25_ms": 22, "semantic_ms": 37, "cross_encoder_ms": 4760, ...},
        "top_nct_ids": ["NCT04802759", ...],
    },
}
```

When something goes wrong, you read the trace, not stdout logs.

**Why structured?** Because text logs don't aggregate. With a structured trace you can:

- Sort runs by `agent_trace[-1].duration_ms` to find the slowest queries.
- Filter on `step="retrieve" AND decisions.pipeline="hybrid_only"` to find queries that fell back.
- Chart `decisions.verdicts` distributions over time to spot drift.

This is the foundation for Prometheus metrics, MLflow tracking, and any alerting you build on top.

### Principle 3: Type safety at boundaries, dicts internally

- **Public boundaries**: Pydantic — `PatientProfile`, `SearchRequest`, `SearchResponse`, `TrialResult`. The validation cost (~10 µs) is worth it because malformed input fails *at the seam*, not three function calls deep.
- **State and intermediate values**: TypedDict and plain dicts. Zero runtime cost; documents the schema for typecheckers.

Pydantic everywhere would be too slow. Plain dicts everywhere would be too loose. The mix is the right answer.

### Principle 4: Lazy initialization with cached failures

Don't load expensive resources at import time. Don't load them on every request. Load them on *first use*, cache the result, and **also cache the failure** — otherwise misconfigured services beat themselves up retrying.

```python
def _get_reranker_or_none():
    if "reranker" in _singletons:
        return _singletons["reranker"]
    if "reranker" in _load_failures:
        return None  # don't try again
    try:
        # ... expensive load ...
        _singletons["reranker"] = ...
    except Exception:
        _load_failures.add("reranker")
        return None
```

### Principle 5: Async for I/O parallelism, not for fashion

We don't use `async` because it's modern. We use it because:

1. The retrieve step is 5 seconds of mostly I/O — wrapping it in `to_thread` keeps the event loop responsive for other concurrent requests.
2. Eligibility checks are I/O-bound (SQLite reads) — `asyncio.gather` parallelizes them for a 24× speedup.

For pure CPU-bound work, async wouldn't help (the GIL would serialize everything). For I/O-bound work, it's a clear win.

> **Common interview trap**: don't say "async makes things faster." Async makes I/O *parallelizable*, which sometimes makes things faster. The value comes from concurrency on I/O-bound work, not from `async` alone.

### Principle 6: Postel's principle — be liberal in what you accept

The orchestrator's `_normalize_filters` function takes whatever the LLM (or caller) passes — `"recruiting"`, `"RECRUITING"`, `"phase 3"`, `"Phase 3"` — and normalizes it to the exact format Elasticsearch expects. Don't reject sloppy input if you can fix it.

The opposite — being strict in what you produce — also applies. Our `SearchResponse` uses `Literal["bm25", "semantic", "hybrid", "agent"]` for `search_method` so callers know exactly what to expect.

---

## Part 8: How to actually run this

### Smoke test (no Docker, no ES required)

```bash
OMP_NUM_THREADS=1 python scripts/test_pipeline.py
```

5 queries including `""` and `"asdfghjkl"`. Expected with ES down: every query falls through to `fallback_search` → empty results, structured error, but **0 uncaught exceptions**.

### Microbenchmark the parallelization win (no ES required)

```bash
OMP_NUM_THREADS=1 python -c "
import asyncio, sqlite3, time
from dotenv import load_dotenv; load_dotenv('.env')
from TrialMine.agents.query_parser import PatientProfile
from TrialMine.agents.orchestrator import SearchOrchestrator

con = sqlite3.connect('data/trials.db')
nct_ids = [r[0] for r in con.execute(
    'SELECT nct_id FROM parsed_eligibility WHERE parse_confidence > 0.8 LIMIT 10')]
con.close()

profile = PatientProfile(raw_query='breast cancer', condition='breast cancer', age=55, sex='Female')
orch = SearchOrchestrator()

t0 = time.perf_counter()
[orch._check_eligibility(n, profile) for n in nct_ids]
print(f'sequential: {(time.perf_counter()-t0)*1000:.1f} ms')

async def par():
    t0 = time.perf_counter()
    await asyncio.gather(*[asyncio.to_thread(orch._check_eligibility, n, profile) for n in nct_ids])
    print(f'parallel:   {(time.perf_counter()-t0)*1000:.1f} ms')
asyncio.run(par())
"
```

Expected: ~143ms sequential, ~6ms parallel. ~24× speedup.

### Full happy-path test (needs Docker + ES)

```bash
docker start es                                # bring up ES
OMP_NUM_THREADS=1 python scripts/test_pipeline.py
```

Expected: queries 1-4 return ranked + eligibility-checked results (`used_fallback=False`); query 5 (`asdfghjkl`) returns empty results without crashing.

### API smoke test

```bash
# Start the API (degraded mode if ES is down)
uvicorn TrialMine.api.app:app --port 8000 &

# Agent path (default)
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "breast cancer trials"}'

# Legacy path
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "breast cancer trials", "use_agent": false, "method": "bm25"}'
```

---

## Part 9: Interview survival kit (15 likely questions, model answers)

Drilling these is the highest-leverage thing you can do before the interview. Read the answer, then close the doc and try to give it in your own words.

### Q1: Walk me through your agent architecture.

> "It's a small LangGraph state machine with three nodes: `parse_query`, `execute_search`, and `fallback_search`. `parse_query` is the only LLM call — Claude Haiku 4.5 extracts a structured `PatientProfile` from the patient's free-text query. `execute_search` runs a deterministic Python pipeline: normalize the condition, build a query string, compute filters, retrieve via the existing BM25+semantic+CE+LightGBM stack, check eligibility on the top results in parallel, and generate template explanations. The conditional edge routes to `fallback_search` if `execute_search` throws — the fallback runs plain hybrid retrieval as a degraded but non-empty path. The whole thing is bounded by a 15-second wall-clock timeout via `asyncio.wait_for`."

### Q2: Why LangGraph and not CrewAI or just function calling?

> "We needed three primitives: typed state that flows through nodes, conditional routing based on that state, and accumulating-list semantics for our `agent_trace` observability log. LangGraph's `StateGraph` gives all three with minimal abstraction overhead. CrewAI is built around multi-agent coordination, which we don't need. Raw function calling would make us reinvent the conditional-routing primitive ourselves. We use only `StateGraph` from LangGraph — not `create_react_agent` — because our flow is deterministic."

### Q3: Why didn't you use a ReAct agent?

> "We considered it. The tradeoff: ReAct is 5-15 seconds per query and $0.02-$0.03 in LLM costs because the LLM picks tools step by step. Our task isn't open-ended — it's a fixed pipeline of normalize → search → check → explain. We don't need an LLM to decide what to do; we already know what to do. Picking the deterministic path saved us 5× latency and 10× cost for negligible quality loss."

### Q4: What happens if Elasticsearch goes down?

> "Three things, in order. First, the API still starts in degraded mode — we made the lifespan ES-tolerant. Second, an incoming agent request runs `parse_query` successfully (no ES dependency), then `execute_search` throws a `ConnectionError`, which the node catches and sets `state['error']`. Third, the conditional edge routes to `fallback_search`, which also tries ES, also fails, and returns empty results with a structured error. The user gets a 200 with `used_fallback=True, error='both paths failed: ...'`, never a 500. We tested this with our 5-query smoke test against a stopped ES — 0 uncaught exceptions across all queries."

### Q5: How did you measure latency? What's the breakdown?

> "Every node appends a structured entry to `state['agent_trace']` with a `duration_ms` field. We can sort or aggregate over those. Approximate breakdown of a warm query: `parse_query` 1.3s, `normalize`/`build_query`/`build_filters` <1ms each, `retrieve` 4-5s (cross-encoder dominates), `check_eligibility` 5ms parallel (was 143ms sequential), template explanations <1ms total. Roughly 6 seconds end-to-end."

### Q6: Walk me through how you parallelized the eligibility checks.

> "Original implementation was a sequential `for` loop — for each of the top 5-10 trials, call `check_trial_eligibility`, get back a JSON verdict. Each call is around 14ms because of SQLite read + matcher logic. Five sequential calls is 70ms; ten is 140ms. Linear scaling. I made the orchestrator's `search()` method async and changed the loop to `asyncio.gather` over `asyncio.to_thread` — every check runs in a thread-pool worker, but they run concurrently. Wall-clock time becomes max(per-call), not sum. Measured 24.6× speedup on 10 cached checks. The work is I/O-bound (SQLite), so threads work even with the GIL."

### Q7: What's the difference between `asyncio.gather` and `asyncio.to_thread`?

> "`asyncio.gather` runs multiple awaitables concurrently and waits for all to complete. `asyncio.to_thread` takes a synchronous function and runs it in the default thread pool, returning an awaitable. Combined, they let you parallelize sync I/O-bound work from async code. For CPU-bound work, you'd want a process pool instead because of the GIL — threads can't actually run Python bytecode concurrently."

### Q8: Why do tools return JSON strings instead of dicts?

> "It's a constraint of LangChain's tool message protocol — tool results have to serialize to a string because they're carried in a conversation as text. We chose JSON because it's the most structured choice that survives the round-trip. The orchestrator does `json.loads()` to parse the result back to a dict. Slightly awkward, but a hard requirement of the framework."

### Q9: How do you handle a flaky LLM call?

> "The QueryParser wraps the entire `messages.parse` call in try/except. On any failure — API timeout, validation error, refusal — it returns a `PatientProfile` with only `raw_query` populated. The downstream pipeline treats this skeleton profile the same as a fully-populated one, just with fewer signals: the retrieve step still runs against the raw query, the eligibility check still runs but returns mostly Unknown verdicts. Quality degrades, but the pipeline never fails. This is the null object pattern."

### Q10: How would you scale this to 10,000 QPS?

> "Several levers. First, the async architecture means a single uvicorn worker can handle many concurrent requests without blocking. Second, lazy module-level singletons mean models load once per process — horizontal scaling just spins up more processes. Third, the cross-encoder dominates per-query latency at ~5 seconds — at scale I'd put it behind a separate inference service with batched GPU inference and a queue. Fourth, eligibility checks are SQLite — at high concurrency, replace with a proper RDBMS or a Redis cache for popular trial-condition combinations. Fifth, the Haiku LLM call can be batched server-side via the Batches API for non-latency-sensitive workloads. Sixth, an edge cache (Cloudflare-style) for canonical queries would absorb a lot of duplicate traffic."

### Q11: What did you learn that you'd do differently?

> "Two things. First, I'd extract the eligibility-matching logic out of the `@tool` wrapper into a standalone module from the start. Currently it lives inside `tools.py:check_trial_eligibility`, and the orchestrator calls it via `.invoke({...}) + json.loads(...)`. The cleaner architecture is one source-of-truth function called by both the `@tool` wrapper and the orchestrator directly. Second, I'd front-load latency calibration. We set a `<1s` parsing target that turned out to be aspirational — measured 1.3-1.5s. Measuring early would have prevented carrying that claim forward."

### Q12: What does `Annotated[list, operator.add]` do in your TypedDict?

> "It's telling LangGraph to use a custom reducer for that field. By default, when a node returns a state diff, the new value overrides the old. For accumulating fields like `agent_trace` — every node should append, not replace — we wrap the type in `Annotated` with a reducer function. `operator.add` concatenates lists. The same idea is used in LangGraph's built-in `MessagesState` for chat-style message histories, with the `add_messages` reducer instead."

### Q13: Why did you decide to *not* trigger fallback on empty results?

> "Empty results are a legitimate answer. If a patient asks for `'rare sarcoma in my leg'` and there genuinely aren't any matching trials, returning empty is the right answer — masking it behind a degraded fallback would just give them lower-quality results that also don't match. Fallback is reserved for *system errors* (exceptions, timeouts), not for *no matches*. This is a subtle distinction but important: the contract is 'always returns a structured response,' not 'always returns non-empty results.'"

### Q14: How does your prompt caching strategy work?

> "Honest answer: it doesn't, for now. The QueryParser's system prompt is about 1370 tokens, and Claude Haiku 4.5 has a minimum cacheable prefix of 4096 tokens. Anything below that doesn't cache — Anthropic returns `cache_creation_input_tokens: 0` silently. We added the `cache_control` marker anyway as a no-op so it activates automatically if we expand the prompt past the threshold. If we wanted real caching, we'd add few-shot examples to bulk the prompt to ~5000 tokens — would also probably improve extraction quality."

### Q15: How do you ensure the agent doesn't hallucinate trials that don't exist?

> "The agent doesn't generate trials. It only retrieves them. The LLM (Haiku) is used solely for slot extraction from the patient's query — it never produces NCT IDs or trial titles. All trial data comes from Elasticsearch and SQLite, both populated from the actual ClinicalTrials.gov API. The template explanations only use fields from the retrieved trial documents. This is a deliberate architectural property: the LLM's output surface area is constrained to typed slots that we validate with Pydantic, so it can't hallucinate trial information even if it wanted to."

---

## Part 10: What I'd ship next (the roadmap question)

If asked "what would you build next?", here's a list ordered by impact:

1. **Streamlit UI for the new agent response shape** — render the `explanation`, `eligibility` verdict counts, and `agent_trace` per result. Currently the UI still calls the legacy single-shot search.
2. **Extract the eligibility matcher** out of `tools.py` into a clean `features/eligibility_matcher.py` module — removes the awkward `.invoke({...}) + json.loads()` indirection from the orchestrator and makes the matcher independently unit-testable.
3. **Pre-warm the cross-encoder at API startup** — first agent query currently pays a ~5s CE model load cost. Adding a warm-up call to the lifespan moves that cost to startup.
4. **MLflow tracking of per-query agent latency** — log every `agent_trace` summary so we can chart latency distributions over time and catch regressions.
5. **UMLS via SciSpacy `EntityLinker`** for richer concept normalization — Decision 18's upgrade path. Replaces the hand-built ~45-entry synonym dict.
6. **Optuna hyperparameter tuning** for the LightGBM blender — viable now that we have 145 labeled queries.

---

## Part 11: Files and code volumes (the bibliography)

| File | Lines | What's in it |
|---|---|---|
| `src/TrialMine/agents/tools.py` | ~900 | 4 LangChain `@tool` wrappers, lazy singletons, Pydantic args schemas, error envelopes |
| `src/TrialMine/agents/query_parser.py` | ~250 | `QueryParserAgent`, Haiku 4.5, `messages.parse`, two-schema trick |
| `src/TrialMine/agents/orchestrator.py` | ~370 | Async rule-based `SearchOrchestrator`, parallel eligibility via `asyncio.gather` |
| `src/TrialMine/agents/pipeline.py` | ~300 | LangGraph `StateGraph`, conditional edges, 15s timeout |
| `src/TrialMine/api/schemas.py` | +20 | `use_agent` flag, agent-specific response fields |
| `src/TrialMine/api/routes.py` | +60 | `_search_agent` route handler, ES-tolerant 503 guards |
| `src/TrialMine/api/app.py` | +30 | ES-tolerant lifespan, pipeline build at startup |
| `scripts/test_pipeline.py` | ~130 | 5-query smoke test with full agent_trace printout |
| `docs/06_agent.md` | this | Educational walkthrough |

Total agent-system code: ~2000 lines including tests. Built in one focused week.

---

## Part 12: Glossary (one-line definitions)

- **Agent**: LLM + tools + control flow. Three shapes: ReAct (LLM-driven), rule-based (Python-driven), hybrid.
- **Tool**: a typed function the LLM (or rule-based code) can invoke. Decorated with `@tool` in LangChain.
- **State machine**: a graph where nodes are computations and edges are transitions. LangGraph's `StateGraph` is one.
- **Reducer**: a function that merges old state with a state diff. Default is "override"; for lists you usually want `operator.add`.
- **`Annotated[T, reducer]`**: Python typing syntax that LangGraph reads to know how to merge state updates for a field.
- **ReAct**: Reasoning + Acting — an agent design pattern where the LLM alternates between thinking out loud and calling tools.
- **Conditional edge**: a routing decision in a state graph based on the state's contents.
- **Fallback path**: a degraded but functional path that runs when the primary path fails.
- **Graceful degradation**: when a system gets dumber instead of crashing.
- **Null object pattern**: returning a special "empty but valid" object instead of `None` so callers don't need to check.
- **Lazy singleton**: a module-level instance created on first use and cached forever.
- **`asyncio.to_thread(fn, *args)`**: runs a sync function in the default thread pool, returning an awaitable.
- **`asyncio.gather(*awaitables)`**: runs multiple awaitables concurrently, awaits all.
- **`asyncio.wait_for(coroutine, timeout=N)`**: caps an async operation at `N` seconds; raises `TimeoutError` if exceeded.
- **GIL**: Global Interpreter Lock — Python's thread-safety mechanism that prevents true parallel execution of Python bytecode in threads. Releases during I/O, so threads work for I/O-bound work but not CPU-bound work.
- **Pydantic args schema**: a Pydantic class that defines a tool's input shape. Becomes a JSON Schema sent to the LLM.
- **`messages.parse()`**: Anthropic SDK method that constrains LLM output to a Pydantic schema and validates it.
- **Postel's principle**: "Be liberal in what you accept, conservative in what you send." Applies to LLM-facing APIs as much as networking.
- **Prompt caching**: a Claude API feature that caches large prompt prefixes for ~90% cost reduction. Has a minimum prefix size — Haiku 4.5's is 4096 tokens.
- **NCT ID**: ClinicalTrials.gov's unique identifier for a trial, e.g., `NCT04802759`.

---

## A final word for the interview

The interviewer is going to probe at three layers:

1. **What did you build?** Show them you can describe the architecture clearly. Use the diagram in Part 4.
2. **Why did you build it that way?** Show them you can articulate tradeoffs. Use Decisions 20-23 from Part 6.
3. **What would you change?** Show them you can self-critique. Use Q11 and Part 10.

The single best thing you can do is **be honest about tradeoffs**. Don't pretend everything was perfect. The "<1s parsing target was aspirational; we measured 1.3-1.5s; we accepted it because it's within the 15s budget" answer is much stronger than "we hit our latency targets." Calibrated honesty signals engineering maturity.

Good luck.
