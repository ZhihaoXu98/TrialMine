# Week 8 — Tests, CI, and Production Signal-Boosters (Student Edition)

A walk-through of everything we did this week. I'm assuming you've never
written a test, never seen a CI pipeline, and never thought about A/B
testing. Every term gets defined the first time it shows up. By the end
you'll know what we built, why we built it, and how to talk about it
when someone asks.

If you haven't read them yet, the previous weeks build the system this
doc tests:

- [`05_eligibility.md`](05_eligibility.md) — the eligibility parser
- [`06_agent.md`](06_agent.md) — the LangGraph agent
- [`07_deployment.md`](07_deployment.md) — Docker stack, Prometheus, Grafana

You don't *have* to re-read them, but I'll point back to specific
sections when relevant.

---

## Part 1 — Why this week matters

Here's the situation at the end of Week 7:

The system **worked**. You could open the Streamlit UI, type a query,
and get back ranked clinical trials with explanations and eligibility
verdicts. Prometheus showed metrics; Grafana showed dashboards.

But "it works on my laptop" is not the same as "I'm comfortable showing
this to a senior engineer at a hiring loop." There were four big gaps.

### Gap 1: No tests

A **test** is code that runs your code and checks the answer is right.
For example: *"if I ask the parser to parse `'lung cancer'`, the
condition field should equal `'lung cancer'`."*

We had zero. None. If I broke the parser, the only way I'd find out is
by clicking around the UI and noticing a wrong result. That's fine for
a hackathon. It's not fine for any real team — your teammates need to
trust that when you push code, you didn't break the parts they wrote.

### Gap 2: No CI

**CI** stands for *Continuous Integration*. The simplest mental model:
**a robot that runs all your tests every time you push code, and yells
at you if anything breaks.**

The robot is GitHub Actions in our case. We had no CI workflow at all.
Every push was an act of faith.

### Gap 3: No "production thinking"

If the cross-encoder model gets stuck in a slow loop in production,
what happens? Today: the user sees a 15-second hang. With "graceful
degradation" wired in: we notice the slowness, give up on the cross
encoder, and return faster (slightly worse) results from the simpler
hybrid retriever. Same idea for: experimentation infrastructure,
quality-gating new model versions, knowing your data is fresh.

We had none of these patterns. Every senior MLE looks for them.

### Gap 4: No reproducibility for models

We had three trained models on disk. Each had a `metadata.json`. But
the schemas were inconsistent — some had `cv_metrics`, some had
`eval_metrics`, some had no `training_date` at all. Without a
**standard schema** you can't answer the basic audit question: *"Which
version was this, trained on what data, with what config, and how well
did it score?"*

### What we actually built

```
This week:
├── Tests (67 of them across 3 suites)
├── CI (3 jobs: lint → test → quality gate)
├── Pre-commit hooks (auto-format on git commit)
├── DVC (initialized — version control for big files)
├── Prometheus metrics (finished what Week 7 started — 7 canonical metrics)
└── Four "production signal-boosters":
    ├── A/B test router (stub interface)
    ├── Data quality script
    ├── Graceful degradation policy
    └── Standardized model metadata
```

Each piece is small in lines of code. Each one says, in interview
shorthand: *"I have worked on a system where this mattered."*

---

## Part 2 — Vocabulary you'll meet

If any of these are unfamiliar, this is your reference. Skim, come back
when you hit one.

### Testing words

**Test** — code that runs your code and checks the answer.

**Pytest** — the Python test runner we use. You write functions like
`def test_thing(): assert ...`, then run `pytest` in the terminal and
it finds and runs all of them.

**Unit test** — tests one small piece of code in isolation. Fast.
No database, no network. Run in milliseconds.

**Integration test** — tests several pieces working together, usually
including a real external service (database, API). Slower. We use
real Elasticsearch.

**Fixture** — reusable test setup. Example: a function that loads our
50-trial sample dataset once and shares it across many tests.

**Mock** — a fake stand-in for a real thing. Example: instead of
calling Anthropic's real API in tests, we use a fake `messages.parse`
method that returns a pre-canned answer.

**Skipif** — a pytest decorator that says *"skip this test if X is
true."* We use it to skip ML-model tests when the model files aren't
on disk.

### CI / DevOps words

**CI / continuous integration** — the robot. Runs on every push.

**Workflow** — the recipe the robot follows. Lives in
`.github/workflows/ci.yml`.

**Job** — one step in the workflow. We have three: lint, test,
quality gate.

**Service container** — an extra container that runs alongside the
test job, just for the duration of that job. We use it to spin up
Elasticsearch so tests can hit a real ES instance.

**Pre-commit hook** — a script git runs *before* the commit lands.
We use it to auto-format code (so we never push code that fails
CI's format check).

**DVC (Data Version Control)** — git, but for files too big for git
(like our 912 MB SQLite database). The actual file lives in cloud
storage; you commit a tiny pointer file. We initialized it but haven't
set up cloud storage yet — that's a Week 9 thing.

### Production-thinking words

**A/B test** — show variant A to half your users, variant B to the
other half, measure which group has better outcomes (clicks,
conversions, NDCG, whatever). The hard part is *deterministic
routing*: the same user must always see the same variant, otherwise
the UI flickers between options and your statistics break.

**Graceful degradation** — when something heavy is slow or broken,
fall back to a simpler path that still produces an answer. *"The cross
encoder is hung — return plain hybrid results instead of timing out."*

**Soft skip** — a CI job exits successfully but says "I couldn't run,
but that's fine." The opposite of failing the build.

**Field coverage** — the percentage of rows in your dataset where a
field is populated (not null, not empty). Useful for spotting silent
breakage of upstream data pipelines.

---

## Part 3 — What we built, step by step

I'll walk through each piece. We'll start with the parts that build on
Week 7 (Prometheus metrics), move to tests and CI, then the four
production signal-boosters.

### Step 1 — Finish wiring up Prometheus metrics

In Week 7 we added the *bare bones* of metrics: a request counter, a
latency histogram, and a per-stage histogram fed from the agent's
trace. This week we finished the job: **seven canonical metrics**, each
instrumented at the right call site.

**Where they live**: `src/TrialMine/monitoring/metrics.py`.

**What they each tell you**:

| Metric | Question it answers |
|---|---|
| `REQUEST_COUNT` | Are we returning errors? (split by status code) |
| `REQUEST_LATENCY` | Are requests slow? |
| `SEARCH_STAGE_LATENCY` | *Which* stage is slow? (BM25 vs semantic vs cross encoder vs ...) |
| `SEARCH_RESULTS_COUNT` | How many trials are we returning per query? |
| `ZERO_RESULTS` | How often are we returning nothing? |
| `AGENT_FAILURES` | When the agent fails, where in the pipeline? |
| `MODEL_INFERENCE` | Is the *model itself* slow, or just the surrounding code? |

You can't answer these questions with one big "is the system slow"
metric. Each one isolates a different layer.

**The two helper decorators**

The annoying way to time something is to sprinkle `time.perf_counter()`
calls everywhere. Instead we have:

```python
# Decorator form — wraps a whole function
@time_stage("agent_search")
async def execute_search(state):
    ...

# Context manager form — wraps just one block inside a function
with time_model_cm("claude-haiku-4-5"):
    response = self._client.messages.parse(...)
```

Both work on **sync and async** functions, and both record observations
on **success and exception** (so a slow failure still shows up in the
histogram — that matters for catching incidents).

**The neat trick — `record_agent_trace`**

Our agent already produces a structured trace (a list of dicts like
`{"step": "parse_query", "duration_ms": 1342}`) for debuggability —
see `06_agent.md` Part 5. Rather than instrument every single node in
the LangGraph manually, we have a single function that walks the trace
after the request and emits one Prometheus observation per entry:

```python
def record_agent_trace(trace):
    for entry in trace:
        SEARCH_STAGE_LATENCY.labels(stage=entry["step"]).observe(entry["duration_ms"])
```

That's it. The trace was already structured for humans (debuggability);
we got rich metrics "for free" by piggybacking on it.

### Step 2 — Write the tests (67 of them)

Tests live under `tests/`. Three suites:

```
tests/
├── unit/            ← fast (no external services)
├── integration/     ← needs Elasticsearch
└── ml/              ← needs trained model files on disk
```

Why three? Because they have different *speeds* and *requirements*:

- **Unit tests** run in CI on every push (every PR, every push to
  main). They must be fast (~10 seconds total) and have no external
  dependencies, otherwise CI would be slow and flaky.
- **Integration tests** also run in CI but they need ES (we provide it
  via a service container — see Step 3).
- **ML tests** check that our trained models load and produce
  reasonable outputs. They need ~1 GB of model files. Those aren't in
  git. So these tests **skip** when the files aren't there.

#### The 50-trial sample fixture

For integration tests we need realistic trial data. The full database
is 912 MB and lives in `.gitignore`. So we wrote a tiny script that
pulled a *stratified* 50-trial sample from the live DB — biased to
cover diverse cancer types, phases, and statuses.

The result: `tests/fixtures/sample_trials.json` (~93 KB, fits in git).
A pytest fixture in `tests/integration/test_search.py` bulk-indexes
those 50 trials into a `trials_test` ES index, runs the tests, then
deletes the index when done.

The diversity is on purpose. If our sample only had RECRUITING trials,
the test `test_status_filter_excludes_completed_trials` would be
vacuous (everything would already match the filter). So we
intentionally pulled 4 COMPLETED, 2 ACTIVE_NOT_RECRUITING, and 1
TERMINATED trial alongside the 43 RECRUITING ones.

#### Testing the LLM-using parser without calling the LLM

This is the part that confuses people the first time. Our query parser
calls Anthropic's API. In CI we don't want to:

- Burn money on every push
- Be at the mercy of Anthropic's API uptime
- Have tests fail because Haiku phrased its output slightly differently

So we **mock** the SDK call. The mock pattern lives in
`tests/unit/test_parse.py`:

```python
def _make_parser_with_mocked_client(parsed):
    parser = QueryParserAgent(api_key="sk-test-not-used")
    fake_response = MagicMock()
    fake_response.parsed_output = parsed   # <- our pre-canned answer
    parser._client.messages.parse = MagicMock(return_value=fake_response)
    return parser
```

We replace just one method (`messages.parse`) with a mock that returns
our pre-built fake response. Tests then assert *our* code's behaviour
given that known input.

**Why mock at this exact level?** Three options:

1. **Don't mock — call the real LLM.** Slow, expensive, flaky.
2. **Mock the HTTP transport.** Brittle — when Anthropic ships a new
   SDK version that changes its response envelope, every test breaks.
3. **Mock at the SDK boundary.** Replace just `messages.parse`. Tests
   assert our code's behaviour, the SDK absorbs its own changes.

We picked option 3. This is *Decision 29* — explained in Part 4.

#### Tests that lie because of `Path.exists()`

This is a fun gotcha that took us one CI failure to find.

When we made `models/*/metadata.json` trackable in git, the **directory**
`models/embeddings/fine-tuned/` started existing in CI even though the
weights didn't. Our naive skipif:

```python
@pytest.mark.skipif(not EMBEDDER_PATH.exists(), reason="...")
```

started lying. The directory existed (because metadata.json was inside
it). The test ran. Tried to load the model. Crashed.

Fix: check for a *load-bearing file*, not just the parent dir:

```python
HAS_EMBEDDER = (EMBEDDER_PATH / "config.json").exists()
HAS_RANKER   = RANKER_PATH.exists()  # already a file
```

**Lesson**: a skipif gate has to match what the test actually requires,
not just what's vaguely associated with it.

### Step 3 — Build the CI workflow

This is the file `.github/workflows/ci.yml`. Three jobs run in
sequence:

```
push or PR to main
       │
       ▼
   ┌─────────┐
   │  lint   │  ruff check + format check + mypy
   └────┬────┘
        │ if green
        ▼
   ┌─────────┐
   │  test   │  install deps, spin up ES sidecar, pytest
   └────┬────┘
        │ if green
        ▼
   ┌──────────────┐
   │ quality-gate │  run NDCG@10 evaluation; fail if below 0.7
   └──────────────┘
```

#### How ES gets into CI

GitHub Actions has a built-in feature called **service containers**.
You declare a container in the workflow YAML and Actions starts it
alongside your job. We use it like this:

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    env:
      discovery.type: single-node
      xpack.security.enabled: "false"
      ES_JAVA_OPTS: "-Xms1g -Xmx1g"
    ports:
      - 9200:9200
```

It's the same image, env vars, and heap size as our `docker-compose.yml`.
Integration tests in CI hit `http://localhost:9200` exactly as they do
locally.

#### Why the quality gate is "soft"

Models live in DVC. DVC's remote storage isn't set up yet (that's a
Week 9 task). So in CI, we don't *have* the trained models or the
evaluation labels.

If quality gate said "fail unless you can produce an NDCG@10 number,"
every PR would be blocked. So instead, `scripts/ci_quality_gate.py`
checks for the artefacts, and if any are missing, prints a clear
"skipped — missing X" message and exits with code 0 (success).

Once DVC remote storage exists, we'll remove the early-return and the
gate becomes a real gate.

#### A bug we hit and fixed

When mypy ran for the first time in CI:

```
pyproject.toml:1: error: Error importing plugin "pydantic.mypy":
No module named 'pydantic'
```

Why? Our `pyproject.toml` has:

```toml
[tool.mypy]
plugins = ["pydantic.mypy"]
```

That plugin needs `pydantic` to be importable. The lint job (to keep it
fast) doesn't install the full project — that would pull torch, scispacy,
and everything else, taking 5+ minutes per run. Fix: install just
`pydantic` in the lint step.

**Lesson**: a tool's plugins need their underlying library, even if the
job doesn't otherwise use it.

### Step 4 — Pre-commit hooks + DVC

#### Pre-commit hooks

You write some code, run `git commit`, and git fires whatever's
configured in `.pre-commit-config.yaml` *before* the commit completes.
If a hook fixes something (like reformats your code), the commit fails;
you stage the fixes and try again.

Our config runs:

- **ruff** with `--fix` (auto-fixes lint issues)
- **ruff format** (formats code; same tool CI runs)
- Standard hygiene hooks (trailing whitespace, EOF newline,
  block-large-files, no-merge-conflicts)

#### A trap to remember: pin tool versions

Pre-commit pinned `ruff v0.6.9`. CI's `pip install ruff` pulled
`v0.15.12`. The two versions formatted code slightly differently.
What pre-commit committed locally, CI rejected.

Fix: pin both to v0.15.12. (Or to whatever exact version you choose —
just make them match.)

**Lesson**: *any* tool that runs in two places must be pinned to the
exact same version in both, or you'll get this exact bug.

#### DVC

We ran `dvc init`. This creates `.dvc/config` (empty) and `.dvcignore`.
Nothing is tracked yet.

Next steps (Week 9): set up an S3 / GCS / Azure remote and run
`dvc add data/trials.db` so the SQLite database is properly versioned
without bloating git.

For now, the init alone is the signal: *"this project is ready for
DVC when we're ready for cloud storage."*

### Step 5 — The four production signal-boosters

These are the new code contributions for this week. Each one is small.
Each one signals "I've worked on a real production system."

#### 5a — A/B test router (a stub, on purpose)

**File**: `src/TrialMine/experiments/ab_test.py`

What we built:

```python
class ExperimentVariant(BaseModel):
    name: str
    weight: float

class Experiment(BaseModel):
    name: str
    variants: list[ExperimentVariant]
    enabled: bool = True

class ABTestRouter:
    def route(self, subject_id, experiment_name) -> str: ...
    def log_exposure(self, subject_id, experiment_name, variant) -> None: ...
```

Today the router is in-memory and log-only. Tomorrow it'll plug into
a real telemetry sink (BigQuery / Kafka / Datadog). The interface is
the same; only the body of `log_exposure` changes.

**The non-obvious bit — the bucketing function**

```python
def _bucket(experiment_name, subject_id) -> float:
    raw = hashlib.sha256(f"{experiment_name}:{subject_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / (1 << 64)
```

Two properties matter:

1. **Deterministic.** The same `(experiment, subject)` always produces
   the same bucket. So `user-42` is in the same variant on every
   request. They don't see the UI flicker A → B → A.
2. **Independent across experiments.** Including `experiment_name` in
   the hash means `user-42` who landed in variant A of experiment X
   doesn't automatically land in A of experiment Y. Without this,
   experiments accidentally couple — and your statistical analysis
   becomes invalid.

See *Decision 30* in Part 4 for *why ship a stub at all*.

#### 5b — Data quality script

**File**: `scripts/data_quality_check.py`

What it does, plain English:

1. Open `data/trials.db`.
2. For each of 15 fields (title, conditions, eligibility_criteria,
   etc.), count what % of rows are populated. Print a bar chart.
3. Print the most recent `start_date` and `completion_date` in the
   corpus, plus the file's last-modified time. (Is the data fresh, or
   has the downloader not run for a month?)
4. Run six "suspicious record" queries and print counts + 3 example
   NCT IDs each:
   - Empty title
   - Empty conditions
   - RECRUITING but no eligibility text
   - Both `min_age` and `max_age` are null
   - Enrollment outside [1, 100000]
   - Status code we don't recognize

Output: pretty stdout report **and** a JSON file at
`data/data_quality_report.json` for machine consumers (CI alerts,
Grafana annotations, etc.).

When we ran it on the live 140K-trial DB, it surfaced two real issues
that nobody had noticed:

```
⚠ missing_age_bounds              6,814  (Both min_age and max_age are null)
⚠ implausible_enrollment          4,703  (Enrollment outside [1, 100000])
```

That's the *whole* point of a data quality script: most of the time
it's green. Once in a while it lights up and tells you the upstream
parser broke.

#### 5c — Graceful degradation

**File**: `src/TrialMine/config.py` (the `DegradationConfig` class)

The shape:

```python
class DegradationConfig(BaseModel):
    cross_encoder_enabled: bool = True
    eligibility_check_enabled: bool = True
    skip_cross_encoder_if_slow_s: float = 5.0
    skip_agent_if_slow_s: float = 10.0
```

**Two distinct controls per heavy component**:

1. **Hard toggle** (`*_enabled`). Flip `cross_encoder_enabled = False`
   to skip the cross encoder entirely. Useful for incident response —
   *"the CE is OOM-looping in prod, turn it off until we ship a fix."*
2. **Soft per-call budget** (`skip_*_if_slow_s`). A wall-clock cap
   enforced via `asyncio.wait_for`. If retrieval+CE takes longer than
   5 seconds on a single request, give up and fall back to plain
   hybrid for *that one request*.

Notice: the hard toggle is global. The soft budget is per-request.
You need both because they handle different failure modes — one model
is broken (use toggle), one query is unusually slow (use budget).

**Where it's wired in** (`src/TrialMine/agents/orchestrator.py`):

A new method `_retrieve_with_budget` has three exits:

```
cross_encoder_enabled = False?
   ├─yes─► run hybrid only          → "hybrid_only_disabled"
   └─no──► full_pipeline within budget?
           ├─yes─► CE-blended results → "full_pipeline" / "ce_blended"
           └─no──► hybrid fallback    → "hybrid_only_after_timeout"
```

Each path writes a different `pipeline_kind` into the agent trace, so
when you grep the structured logs you can immediately see *which
fallback path* served a request.

The other piece: `pipeline.py`'s `DEFAULT_TIMEOUT` (the outer wall-clock
budget for the whole agent) now reads from
`get_default_degradation().skip_agent_if_slow_s`. That's 10 seconds
now — tighter than the old hardcoded 15.

**Tests** at `tests/unit/test_degradation.py`. The interesting one:

```python
def test_cross_encoder_disabled_skips_full_pipeline():
    cfg = DegradationConfig(cross_encoder_enabled=False)
    orch, retriever = _orchestrator_with(cfg)
    asyncio.run(orch.search(profile))
    assert retriever.full_pipeline_calls == []     # CE never called
    assert retriever.search_calls != []            # hybrid called instead
```

Stub retriever — no ES, no FAISS. The test asserts: *"flipping the
toggle actually short-circuits the heavy path."* That's the contract.

#### 5d — Standardized model metadata

Each trained model now has a `metadata.json` with the same canonical
schema:

```json
{
  "model_type": "lightgbm_lambdarank",
  "model_version": "v2",
  "training_date": "2026-03-28T22:25:48.419934+00:00",
  "features": ["bm25_score", "semantic_score", ...],
  "n_features": 11,
  "n_queries": 145,
  "n_samples": 6018,
  "hyperparameters": { "objective": "lambdarank", "learning_rate": 0.05, ... },
  "eval_metrics": { "cv_ndcg5_mean": 0.843, ... }
}
```

Three keys are required: `training_date`, `eval_metrics`,
`hyperparameters`. Without them you can't answer *"which version was
trained on what data, with what config, and how well did it score?"*
That's the basic audit trail every ML system needs.

**Backwards-compat note**: the LightGBM ranker v1 / v2 used to have
`cv_metrics` and `params`. We renamed to the canonical names *and*
updated `scripts/train_ranker.py` so future training runs write the
new schema.

**The .gitignore puzzle**

We needed: keep gigabytes of model weights out of git, but check the
small JSON metadata files **into** git.

The naive attempt:

```
/models/                       ← ignore everything in models/
!models/**/metadata.json       ← oops, doesn't work
```

Doesn't work because git's pattern grammar says: *"if a parent
directory is excluded, you can't re-include a file inside it."*

The correct three-rule pattern:

```
/models/**                     ← ignore everything in models/
!models/**/                    ← but keep the directories themselves
!models/**/metadata.json       ← and re-include metadata.json inside them
```

Took us one CI failure to figure out. (See the next section.)

### Step 6 — Make CI pass (the iteration)

CI didn't pass on the first push. It took four. Each failure was
useful — they all revealed real issues, not flakes.

**Push 1** (`446c0d3`): `ruff format --check` failed.
Cause: pre-commit pinned ruff v0.6.9; CI's `pip install ruff` pulled
v0.15.12. The two versions formatted code slightly differently.
Fix: pin both to v0.15.12.

**Push 2** (`6ffdb36`): mypy failed with
`No module named 'pydantic'`. Cause: lint job didn't install pydantic;
the pydantic.mypy plugin couldn't load.
Fix: add `pip install pydantic` to the lint step.

**Push 3** (`75c5147`): all four ML tests failed with
`Unrecognized model in models/embeddings/fine-tuned`. Cause: the new
gitignore was tracking `metadata.json`, so the model directories
existed in CI but contained only metadata. The skipif's `Path.exists()`
returned True. Tests ran, tried to load weights, crashed.
Fix: skipif now checks for a load-bearing file (config.json /
model.lgb), not just the directory.

**Push 4** (`674d4ba`): all green. Lint ✓, test ✓, quality-gate ✓
(soft-skipped — artefacts not in CI).

**The lesson here**: each "failure" was a real bug. If we'd fixed them
by disabling the failing tests, we'd have shipped four real production
issues to main. CI failures aren't an obstacle. They're a **gift** —
they catch problems before users do.

---

## Part 4 — Why we made the choices we did

### Decision 29 — Test our code, not third-party internals

When writing `tests/unit/test_parse.py` we had to decide: where do we
mock the LLM call?

**Option A: don't mock — call the real LLM in tests.**

Cost: every CI run uses an Anthropic API key and burns money. Speed:
1.5 s per parse × 5 tests = 7.5 seconds extra. Reliability: depends on
Anthropic's uptime AND on Haiku phrasing its answer the same way every
time. **Verdict: bad.**

**Option B: mock at the network layer (the HTTP transport).**

You'd intercept the HTTP request, return a canned JSON response, and
let the SDK parse it. Brittle — when Anthropic ships a new SDK version
that adds a header or restructures the response envelope, every test
breaks. You'd be testing the *SDK's parsing of your mock*, not your
code's behaviour. **Verdict: bad.**

**Option C: mock at the SDK boundary (`messages.parse`).**

Replace just that one method. Tests assert *our* code's behaviour given
a known-good input. SDK changes are absorbed by the SDK; we don't
notice. **Verdict: good.**

We picked C.

But that raises a question: how do we test the LLM's *quality*? If
the parser sometimes extracts "metastatic" as a stage and sometimes
doesn't, we'd want to know. The answer: **not in unit tests.**
Quality lives in the **eval suite** at
`scripts/evaluate.py` + `data/evaluation/labeled_queries.jsonl` (60+
labeled queries with graded relevance, evaluated with NDCG and MRR
and bootstrap confidence intervals).

The principle:

> **Non-deterministic outputs need offline eval, not unit tests.**

A unit test that sometimes passes and sometimes fails because the LLM
phrased its answer differently is **worse than no test** — it trains
people to ignore CI failures.

The general rule: if you find yourself mocking more than three layers
of a third-party library (network → transport → schema → parsing),
your assertion is in the wrong place. Move it to an offline eval over
labeled examples.

### Decision 30 — A/B router shipped as a stub, not a real backend

We built `ABTestRouter` with deterministic hashing, Pydantic-typed
schemas, exposure logging — but **no real telemetry sink**, **no
bandit allocation**, **no analysis pipeline**.

Why ship a stub at all?

**Reason 1: the interface is the hard part.**

Once call sites use `router.route(...)` and `router.log_exposure(...)`,
swapping the in-memory list for BigQuery later is a one-file change.
If we waited until we had real users to design the interface, we'd
design it under time pressure with the wrong abstractions. Better to
sketch it carefully now.

**Reason 2: it's a hiring signal.**

Showing that you *think about* experimentation — deterministic hashing,
independent buckets, exposure-as-a-seam — is more valuable to a
reviewer than a half-built real backend. A senior engineer reading the
code immediately sees *"this person has run experiments before."*

**What we explicitly did NOT do**:

- No bandit allocation (every variant has a static `weight`).
- No traffic quotas (e.g. "only 5% of users in any one experiment").
- No mutual exclusion between experiments.
- No analysis pipeline.

Those need real outcome data to design correctly. We'd be guessing.

The general rule: when you have an interface that will outlive its
first implementation, **ship the contract**, document the constraints,
and **note explicitly what's a stub**. Don't pretend it's
production-ready when it isn't.

---

## Part 5 — File map (where each thing lives)

If you forget anything from this doc, just open the files in this order:

| File | What's in it |
|---|---|
| `src/TrialMine/monitoring/metrics.py` | All 7 Prometheus metrics, middleware, `time_stage` / `time_model` decorators, `record_agent_trace`. |
| `src/TrialMine/monitoring/__init__.py` | Re-exports — keeps old imports working. |
| `src/TrialMine/config.py` | `Settings` (env vars) + `DegradationConfig`. |
| `src/TrialMine/experiments/ab_test.py` | `ABTestRouter`, `Experiment`, `ExperimentVariant`. The stub. |
| `src/TrialMine/agents/orchestrator.py` | New `_retrieve_with_budget` honours `DegradationConfig`. |
| `src/TrialMine/agents/pipeline.py` | `DEFAULT_TIMEOUT` reads from `DegradationConfig.skip_agent_if_slow_s`. |
| `scripts/data_quality_check.py` | Coverage / freshness / suspicious-records report. |
| `scripts/ci_quality_gate.py` | NDCG@10 gate for CI. Soft-skips when artefacts missing. |
| `tests/conftest.py` | Shared pytest fixtures (50-trial sample, ES URL). |
| `tests/fixtures/sample_trials.json` | The 50-trial stratified sample. |
| `tests/unit/test_parse.py` | Mocked Anthropic SDK. The pattern to copy when LLMs are involved. |
| `tests/unit/test_concepts.py` | ConceptNormalizer (pure functions). |
| `tests/unit/test_api.py` | FastAPI TestClient with mocked `app.state`. |
| `tests/unit/test_eligibility.py` | The 37 SciSpacy + regex tests from Week 5 (moved here). |
| `tests/unit/test_ab_test.py` | Determinism + weight distribution. |
| `tests/unit/test_degradation.py` | Toggles + budget wiring. |
| `tests/integration/test_search.py` | Real ES, indexes the 50-trial fixture, asserts BM25 + filters. |
| `tests/ml/test_models.py` | Loads bi-encoder / CE / ranker. Skipif on load-bearing files. |
| `.github/workflows/ci.yml` | The CI pipeline. |
| `.pre-commit-config.yaml` | ruff lint + format + hygiene hooks. |
| `pyproject.toml` | `[tool.mypy]` with pydantic plugin; `markers = ["slow"]`. |
| `.gitignore` | Three-rule pattern that allows `models/**/metadata.json`. |
| `models/*/metadata.json` | The standardized model contract. |
| `data/data_quality_report.json` | Output of the data quality script. |

---

## Part 6 — How to actually run things

```bash
# Run the full test suite locally
OMP_NUM_THREADS=1 ANTHROPIC_API_KEY=sk-test-placeholder pytest tests/ -v

# Just the fast unit tests (no ES, no models needed)
pytest tests/unit/ -v

# Generate the data quality report
python scripts/data_quality_check.py
# → prints the report, writes data/data_quality_report.json

# Run the quality gate locally (will soft-skip without all artefacts)
python scripts/ci_quality_gate.py --threshold 0.7

# Lint + format + types — same commands CI runs
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/TrialMine/ --ignore-missing-imports

# See live metrics from the running stack
curl -L http://localhost:8000/metrics | grep ^trialmine_

# Query Prometheus
curl -s 'http://localhost:9090/api/v1/query?query=trialmine_search_stage_latency_ms_count' | jq

# Watch CI for a commit (web UI)
# https://github.com/<user>/TrialMine/actions
```

The macOS-only `OMP_NUM_THREADS=1` is the same workaround
[Decision 27 in CLAUDE.md](../CLAUDE.md) explained for Week 7 —
Apple's Accelerate library and faiss-cpu both ship their own OpenMP,
and loading both crashes Python on macOS unless we serialize to one
OpenMP thread on the host. The Linux Docker container doesn't have
this problem.

---

## Part 7 — Interview prep

These are the questions a senior MLE will probe with. Answer them out
loud (or in writing) before you peek at the suggested answers.

### 7.1 The big design question

> *"Walk me through how you decided what to test in unit tests vs.
> what to leave for offline evaluation."*

**What to lead with**: the principle. *Test things you own. Mock at
third-party boundaries. Hold non-deterministic behaviour to account
via offline eval.*

**Then walk through `tests/unit/test_parse.py`**: we mock at
`messages.parse` (the SDK boundary) so we're testing the parser logic
given a known LLM output. The LLM's *quality* — does it correctly
extract "metastatic" as a stage? — is held to account by
`scripts/evaluate.py` running over labeled queries with graded
relevance and bootstrap confidence intervals.

**The punchline**: a unit test that sometimes fails because the LLM
phrased its answer slightly differently trains people to ignore CI
failures. So we don't write those. We write **eval suites**.

### 7.2 Specific systems questions

**Q: Why ship an A/B router stub instead of the real thing?**

The interface is the hard part. Once the call sites use `route` and
`log_exposure`, swapping the in-memory list for BigQuery is one file.
Without real outcome data, designing bandit allocation / traffic
quotas would be guessing. Ship the contract; defer the implementation.

**Q: Your CI quality-gate skips when artefacts are missing — isn't
that defeating the point?**

It's a deliberate softness. Models live in DVC; DVC remote isn't set
up yet. A hard requirement would block PRs whenever DVC sync was
behind. Better: have the gate **available but not enforced**, then
turn it on selectively (manual workflow_dispatch, or nightly cron with
provisioned artefacts). When DVC remote storage is wired up, the
soft-skip becomes a hard-fail by removing one early return in
`ci_quality_gate.py`.

**Q: You instrument both `SEARCH_STAGE_LATENCY{stage='cross_encoder'}`
*and* `MODEL_INFERENCE{model='biolinkbert-cross-encoder'}`. Aren't
those the same thing?**

Almost, but not quite. `SEARCH_STAGE_LATENCY` measures the entire
cross-encoder *stage* — including building input texts, running the
predict call, and post-processing. `MODEL_INFERENCE` measures just
`model.predict(pairs)`. When the stage is slow but the inference is
fast, the bottleneck is in our pre-processing. When both are slow,
the model itself is the problem. Without the second metric you can't
tell those cases apart.

### 7.3 Code-review questions

Try answering these without peeking at the code:

- In `record_agent_trace`, why do we map `step="parse_query"` to
  `stage="agent_parse"` but leave other steps with their raw names?
- The `time_stage` decorator works on both sync and async functions.
  How does it tell them apart?
- Why does `DegradationConfig.skip_cross_encoder_if_slow_s` use
  `asyncio.wait_for` instead of a `signal.alarm` or thread-pool
  timeout?
- In `ABTestRouter._bucket`, why hash `f"{experiment_name}:{subject_id}"`
  and not just `subject_id`?
- The data-quality script checks `recruiting_no_eligibility` but not
  `completed_no_eligibility`. Why?

### 7.4 Tradeoff questions

**Q: Why not run mypy in `--strict` mode?**

`check_untyped_defs = false` because most ML libraries (scispacy,
sentence-transformers, lightgbm) ship without type stubs. Strict mypy
would either generate hundreds of `# type: ignore` markers (noise that
hides real type errors) or block adoption of useful libraries. Loose
mypy on annotated code is the pragmatic stance.

**Q: The `metadata.json` for ranker v2 is 8.6 KB because it contains
per-query NDCG. Why not strip those?**

They're useful for debugging — when v3 lands, we want to compare per-
query improvements. 8.6 KB is below the 1 MB pre-commit large-file
cap, so the cost is essentially zero. Strip when it actually starts
mattering (multi-MB metadata files would be a different conversation).

**Q: Why not set up `dvc remote` (cloud storage) this week?**

Because that needs real cloud credentials and a budget. The init alone
is the signal: *"this project is ready for DVC when the deployment
story is ready for cloud storage."* Honest stub > pretend production.

---

## Part 8 — What's next

Three things we explicitly punted to Week 9:

1. **Wire the A/B router into the ranker.** Pick one small experiment
   (e.g. "blend weight 0.3 vs 0.5"), route 50% of traffic each way,
   measure NDCG@10 differential.
2. **Set up `dvc remote`.** Probably S3 for ~$5/month. Track
   `data/trials.db` and `data/faiss_*.index`. Let CI's quality-gate
   actually run on real artefacts.
3. **Per-query latency in MLflow.** We log per-query NDCG (eval) and
   per-query latency (agent trace). Logging both into MLflow lets us
   spot "this query type is consistently slow AND consistently low
   NDCG" — those tend to be the same queries (rare cancers,
   comorbidities), and that's a productive thing to show in interviews.

Five files to re-read before next week:

1. `src/TrialMine/monitoring/metrics.py` — the metric definitions are
   the contract.
2. `src/TrialMine/agents/orchestrator.py` — `_retrieve_with_budget` is
   the cleanest example of graceful degradation in the codebase.
3. `tests/unit/test_parse.py` — the SDK-boundary mock pattern; you'll
   reuse this every time you have an LLM call to test.
4. `.github/workflows/ci.yml` — read top to bottom once. CI workflows
   look intimidating; they're 90% boilerplate once you've seen one.
5. `docs/06_agent.md` for the orchestrator context, and
   `docs/07_deployment.md` for what Prometheus and Grafana do with
   our metrics.

---

## TL;DR

- **67 tests** across `unit/`, `integration/`, `ml/`. CI runs them all
  on every push.
- **Three CI jobs**: lint, test (with ES service container), quality
  gate (soft-skips when artefacts missing).
- **Seven Prometheus metrics**, two timing decorators, and a structured
  trace fanout — production observability done right.
- **Four MLE signal-boosters**: A/B router stub, data quality script,
  graceful degradation policy, standardized model metadata.
- **Two new design decisions**: (29) test our code, not LLM internals;
  (30) ship experimentation interfaces as stubs.
- **Pre-commit hooks** + **DVC initialized** + **`.gitignore` reworked**
  so model metadata is in git while weights stay out.
- **Four CI iterations** before everything went green — each failure
  caught a real production issue, not a flake.
