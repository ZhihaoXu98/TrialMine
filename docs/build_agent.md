# Build Agentic Search Path — Vibe-Coding Runbook

> Step-by-step plan to add a Haiku-4.5 tool-using agent path to the
> TrialMine search pipeline, behind an A/B router, with persistent
> trace observability so the experiment can be measured and tuned in
> production.
>
> **Why we're building this.** The current `SearchOrchestrator` is
> rule-based by design (Decision 22) and works well on easy queries.
> But `complex_failure_attribution.json` shows 62.5 % of complex-slice
> misses are **rerank-bound** — the right trial is in the candidate
> pool but at rank 11–80. Model retraining (CE v2, LGB v3-reg, v4, v6)
> has not moved the complex slice (NDCG@5 stuck at ~0.34). A different
> *flow* — iterative query refinement + selective tool use — is the
> next intervention candidate, and the LangChain `@tool` wrappers in
> `src/TrialMine/agents/tools.py` already exist but have no driver.
>
> **How to use this doc.** Each step has a copy-paste prompt to send
> to Claude Code. After each prompt, run the **Verify** check and
> confirm the **Acceptance criteria** before moving on. If a check
> fails, run the **Rollback** and re-do that step — never proceed on
> a failed check.
>
> Phases are sequenced so:
> - Phase A is the agent build (~4–6 hr local, no GPU, no cloud).
> - Phase B is observability scaffolding (~2–3 hr local).
> - Phase C is the eval + decision gate (~3–4 hr, ~$2.50 Haiku total — see Appendix Cost recap).
> - Phase D is ship or revert (~1 hr).

---

## Overview

| | |
|---|---|
| **Goal** | Lift complex-slice NDCG@5 from 0.34 → ~0.45–0.55 on agent-routed queries, without regressing the overall held-out NDCG@5 (currently 0.78 on 65q). Ship behind an A/B router so the experiment is reversible by one flag flip, with per-stage Grafana panels + Prometheus counters so cost / latency / routing-rate are observable in production. |
| **Approach** | Add a second pipeline arm: a Haiku 4.5 ReAct loop bound to the existing `ALL_TOOLS` surface, with a hard `max_iters=5` cap and `submit_final_results` terminator tool. Route to it only when QueryParser confidence is low OR the query matches a complex-pattern heuristic. Both arms go through the same `ABTestRouter` (Decision 30 stub already exists). Trace every run to a SQLite store + 4 Grafana panels. |
| **Total cost** | Dev: ~$20–45 in Claude Code (Phase A scaffolding + Phase B observability). Eval: ~$2.50 Haiku (A8 smoke + C2 force-all agent + C2 new-NCT labeling at the project's measured $0.0008/pair rate from CLAUDE.md). Production: +$3.08 per 1K queries at 25 % routing rate (Haiku 4.5 at ~$0.014/agent query vs ~$0.0017/rule query). See `Appendix — Cost recap` for the line-item breakdown. |
| **Total time** | ~4–6 hr Phase A + ~2–3 hr Phase B + ~3–4 hr Phase C + ~1 hr Phase D = **~10–14 hr engineer wall-clock**, almost entirely local. |
| **Decision gate** | Phase C, Step C4 — **holistic review** of routed-NDCG lift, overall NDCG floor, cost/query, p95 latency, iteration distribution, tool-call success. No rigid pre-registered thresholds; ship/tune/revert based on the picture. Same Decision-36 framing the bi-encoder v2 and CE v2 ships used. |

**Files modified across all phases:**

| File | Phase | Note |
|---|---|---|
| `src/TrialMine/config.py` | A2 | Add 5 new `DegradationConfig` fields |
| `src/TrialMine/agents/query_router.py` | A3 (new) | Pure routing function |
| `src/TrialMine/agents/react_agent.py` | A4 (new) | Haiku 4.5 ReAct loop |
| `src/TrialMine/agents/tools.py` | A4 | Add `submit_final_results` tool |
| `src/TrialMine/agents/pipeline.py` | A5 | Conditional routing edge |
| `src/TrialMine/monitoring/metrics.py` | A6 | 4 new Prometheus counters |
| `src/TrialMine/experiments/ab_test.py` | A7 | Register `agent_path_v1` experiment |
| `src/TrialMine/monitoring/trace_store.py` | B2 (new) | SQLite trace writer |
| `scripts/init_trace_db.py` | B1 (new) | Schema migration |
| `docker-compose.yml` | B4 | Mount `agent_runs.db`, install SQLite plugin |
| `infrastructure/grafana/provisioning/datasources/sqlite.yml` | B4 (new) | Datasource config (path matches existing project layout — `infrastructure/grafana/`, not `monitoring/grafana/`) |
| `infrastructure/grafana/dashboards/agent_observability.json` | B5 (new) | 4-panel dashboard (auto-loaded by the existing `dashboards.yml` provider) |
| `scripts/eval_agent_path.py` | C2 (new) | Force-route eval harness |
| `tests/unit/test_query_router.py` | A3 (new) | Routing rule pins |
| `tests/integration/test_react_agent.py` | A4 (new) | Mocked-tools loop test |
| `data/evaluation/agent_vs_rule_v1.json` | C3 (new) | Per-query A/B output |
| `data/agent_runs.db` | B1 (new, gitignored) | Trace store |
| `docs/evaluation-report.md` | D1 | New §12 |
| `CLAUDE.md` | D1 | Decision 43 + Phase 14 entry |

---

## Best practices for these prompts

1. **One concern per prompt.** Each step is scoped to a single logical change. Don't combine.
2. **Verification is non-optional.** Every step has an explicit probe. If you skip it, you're flying blind.
3. **Quote acceptance criteria back.** When you run a probe, paste the output and explicitly check it against the **Acceptance criteria** before moving on.
4. **Context resets are fine.** Each prompt references this doc; Claude can re-orient even if your session was cleared.
5. **Never edit a file in two places at once.** Always finish + verify a step before starting the next.
6. **No "done" without a diff or a probe output.** If a step has no observable output, you didn't verify it.
7. **The agent must default-OFF until Phase D.** Phase A wires the path; Phase D flips the production default. Mid-build, anyone running the API gets the rule path unless they explicitly flip `agentic_path_enabled=True`.

---

# Phase A — Build the agentic search path (~4–6 hr, no GPU)

## Step A0 — Session bootstrap (paste this at the start of every session)

If you're returning after a context reset or a break, send this first:

```
We are working through docs/build_agent.md to add a Haiku 4.5
agentic search path to TrialMine behind an A/B router. Read
build_agent.md and report: (1) which phase steps appear complete
based on file state (DegradationConfig fields, query_router.py,
react_agent.py, pipeline.py wiring, Prometheus counters, trace_store.py,
Grafana dashboard, eval JSONLs), and (2) the next step to run. Do
not modify any files yet.
```

This gives Claude a chance to read the runbook and tell you where you are.

---

## Step A1 — Backups (10 min)

**Goal:** Snapshot the files we're about to modify so a revert is one `cp` away.

### Prompt to send

```
Run Phase A1 from docs/build_agent.md: create backups of the files
we'll modify in Phase A. After completing, run `ls -la` on each
backup target and report file sizes. Do not delete originals —
just copy.
```

### Files to back up

| Source | Backup |
|---|---|
| `src/TrialMine/agents/pipeline.py` | `src/TrialMine/agents/pipeline_v1.py.bak` |
| `src/TrialMine/agents/tools.py` | `src/TrialMine/agents/tools_v1.py.bak` |
| `src/TrialMine/config.py` | `src/TrialMine/config_v1.py.bak` |
| `src/TrialMine/monitoring/metrics.py` | `src/TrialMine/monitoring/metrics_v1.py.bak` |
| `src/TrialMine/experiments/ab_test.py` | `src/TrialMine/experiments/ab_test_v1.py.bak` |

`orchestrator.py` is **not** modified in this build — the rule path stays byte-identical. Don't back it up; it's a no-touch surface.

### Verify

```
List backup files with sizes and confirm 5 .bak files exist.
```

### Acceptance criteria

- 5 `.bak` files exist with non-zero sizes matching their originals.
- The originals are unmodified (compare with `diff <original> <backup>` — should be empty).

### Rollback

N/A (nondestructive).

---

## Step A2 — Add agent-path fields to `DegradationConfig` (~20 min)

**Goal:** Wire 5 new runtime flags so the agent build is gated behind defaults that keep production behaviour identical until Phase D.

### Prompt to send

```
Update src/TrialMine/config.py per Phase A2 of docs/build_agent.md.
Add 5 new fields to DegradationConfig. Use Pydantic Field with
docstrings matching the surrounding style. Preserve all existing
fields and their docstrings.

New fields:

1. agentic_path_enabled: bool = False
   - Hard toggle. When False (the default through Phase A–C), the
     pipeline always routes to the rule-based SearchOrchestrator;
     the agent path is built but inert. Flipped to True in Phase D
     once the eval gate passes.

2. agentic_routing_threshold_slots: int = 2
   - Route to the agent when the parsed PatientProfile has FEWER
     than this many populated slots (condition, stage, age, sex,
     biomarkers, prior_treatments, preferences, location). Sparse
     profiles are typically vague queries that benefit from
     iterative refinement. Range: 1 to 8.

3. agentic_complex_pattern_enabled: bool = True
   - When True, also route to the agent if the raw query matches
     the complex-pattern heuristic (multi-constraint phrases like
     "failed X" + "after Y", or 3+ medical entities in one query)
     even when slot count is high. Separate toggle so the two
     signals can be A/B'd independently.

4. agentic_max_iters: int = 5
   - Hard cap on the ReAct loop. On reaching the cap, the agent
     must call submit_final_results with whatever it has — or the
     pipeline falls back to the rule arm. Range: 1 to 10.

5. agentic_per_iter_timeout_s: float = 6.0
   - Per-iteration wall-clock budget for ONE LLM call + tool
     execution. Sized for the worst-case tool — search_trials runs
     the full retrieval pipeline (BM25 + semantic + ~4s CE rerank),
     so a per-iter under 5s would cancel almost every iteration
     that calls search_trials. 6s leaves 2s slack on top of CE's
     warm p95.
   - WARNING: the multiplied cap (5 * 6 = 30s) is HIGHER than the
     outer pipeline budget skip_agent_if_slow_s=10.0, which means
     the outer wait_for in pipeline.search() will cancel the agent
     long before its inner per-iter timeout fires. This is
     intentional for the Phase A–C build (agent is default-OFF, so
     the outer cap never matters in practice). At Phase D ship,
     raise skip_agent_if_slow_s to ~25s if you want the inner
     timeout to be the binding constraint; otherwise accept that
     the outer budget will be the active cap and document it in
     D1's CLAUDE.md decision entry.

Show me the diff.
```

### Verify

```
Run:
  python -c "from TrialMine.config import DegradationConfig; \
             c = DegradationConfig(); \
             print(c.agentic_path_enabled, c.agentic_routing_threshold_slots, \
                   c.agentic_complex_pattern_enabled, c.agentic_max_iters, \
                   c.agentic_per_iter_timeout_s)"

Expect: False 2 True 5 6.0
```

### Acceptance criteria

- All 5 new fields exist with the correct defaults.
- `agentic_path_enabled` defaults to `False` (CRITICAL — production behaviour must not change yet).
- Existing 5 fields (`cross_encoder_enabled`, etc.) are byte-identical to backup.
- `python -m pytest tests/unit/test_degradation.py` passes (existing degradation tests still green).

### Rollback

```
cp src/TrialMine/config_v1.py.bak src/TrialMine/config.py
```

---

## Step A3 — Build `query_router.py` (~45 min)

**Goal:** A pure, side-effect-free function that decides which arm a query goes to. No LLM, no IO. Testable in isolation. The routing decision is the single hardest thing to get right — cheap to misroute, expensive to do nothing.

### Prompt to send

```
Create src/TrialMine/agents/query_router.py per Phase A3 of
docs/build_agent.md. Plus tests in tests/unit/test_query_router.py.

Module surface:

  from dataclasses import dataclass
  from TrialMine.agents.query_parser import PatientProfile
  from TrialMine.config import DegradationConfig

  @dataclass(frozen=True)
  class RouteDecision:
      arm: Literal["agent", "rule"]
      reason: str          # one-line human-readable
      signals: dict        # raw signal values for trace
      # e.g. {"populated_slots": 1, "is_multi_constraint": True,
      #       "matched_pattern": "failed_X_after_Y"}

  def route_decision(
      profile: PatientProfile,
      config: DegradationConfig,
  ) -> RouteDecision:
      """Pure routing function. No IO. No LLM."""

Decision rules (apply in order, first match wins):

  1. If config.agentic_path_enabled is False:
     → arm="rule", reason="agentic_path_disabled"

  2. Count populated_slots from the profile. A slot is populated when:
     - condition / condition_stage / age / sex / location is not None
     - prior_treatments / biomarkers / preferences is non-empty list
     Total of 8 slots. Compute populated_slots ∈ [0, 8].

  3. Check complex-pattern heuristic (only if
     config.agentic_complex_pattern_enabled is True). Match if ANY:
     a. Regex /\b(failed|progressed on|refractory to|after|post[- ])\b/i
        matches raw_query AND profile.prior_treatments is non-empty
        (i.e., explicit prior-treatment-failure phrasing)
     b. Three or more of {condition, condition_stage, biomarkers
        (non-empty), prior_treatments (non-empty)} populated
        (multi-constraint query)
     Record which sub-rule fired in signals["matched_pattern"].

  4. Route to agent if EITHER:
     - populated_slots < config.agentic_routing_threshold_slots
       → arm="agent", reason="sparse_profile"
     - complex-pattern fired (rule 3 above)
       → arm="agent", reason="complex_pattern"
     (If both fire, prefer "sparse_profile" — it's the cheaper
     signal to compute and the more common trigger.)

  5. Otherwise: arm="rule", reason="rule_path_sufficient"

  These exact reason strings are pinned by the unit tests below
  and by the smoke probe in the Verify section — don't substitute
  synonyms like "vague_query" / "multi_constraint" without also
  updating the tests.

All decisions include in signals:
  - populated_slots (int)
  - is_multi_constraint (bool, from rule 3b)
  - matched_pattern (str | None, name of rule that fired)
  - threshold_slots (int, echoed back from config)

Unit tests (tests/unit/test_query_router.py) — write these BEFORE
the implementation, TDD-style. At minimum cover:
  - agentic_path_enabled=False forces rule arm (preserves Phase A invariant)
  - 0 populated slots with default threshold → agent arm
  - All 8 slots populated, no complex pattern → rule arm
  - "failed osimertinib" + prior_treatments=["osimertinib"] → agent
    arm with matched_pattern="failed_X_after_Y"
  - 3+ slots populated (condition + biomarker + prior_treatment),
    no failure phrasing → agent arm with matched_pattern="multi_constraint"
  - complex_pattern_enabled=False AND populated_slots=8 → rule arm
    (don't false-trigger when only one heuristic disabled)
  - signals dict always contains the 4 keys

Use type hints, no print(). Keep the core route_decision() pure
in this step — no IO, no logging, no module-level state.

(Note: Phase A7 will later wrap this with an AB-router integration
that DOES add minimal IO — an in-memory exposure log via
`router.log_exposure()`. That wrapping happens in the
`route_decision` LangGraph NODE (in pipeline.py), NOT inside the
pure function added here. Keep the pure function pure; let the
node compose: call route_decision → if agentic_path_enabled,
consult AB router → log_exposure → return final RouteDecision.)

Show me both files' diffs.
```

### Verify

```
Run:
  python -m pytest tests/unit/test_query_router.py -v

Expect: all tests pass. Report the count.

Then run a manual smoke:
  python -c "
  from TrialMine.agents.query_router import route_decision
  from TrialMine.agents.query_parser import PatientProfile
  from TrialMine.config import DegradationConfig
  c = DegradationConfig(agentic_path_enabled=True)
  # Vague query
  p = PatientProfile(raw_query='trials for my dad')
  print(route_decision(p, c))
  # Complex query
  p = PatientProfile(raw_query='failed osimertinib EGFR T790M NSCLC',
                      condition='non-small cell lung cancer',
                      biomarkers=['EGFR T790M'],
                      prior_treatments=['osimertinib'])
  print(route_decision(p, c))
  # Easy query
  p = PatientProfile(raw_query='breast cancer trials in Boston',
                      condition='breast cancer', location='Boston')
  print(route_decision(p, c))
  "

Expect (3 lines):
  RouteDecision(arm='agent', reason='sparse_profile', ...)
  RouteDecision(arm='agent', reason='complex_pattern', ...)
  RouteDecision(arm='rule', reason='rule_path_sufficient', ...)
```

### Acceptance criteria

- All unit tests pass (≥ 7 tests covering each branch).
- Manual smoke produces the expected 3-line output.
- `route_decision` is annotated as a pure function (no mutation of inputs).
- No LangChain / Anthropic imports — this module is dependency-light.
- `agentic_path_enabled=False` ALWAYS returns `arm="rule"` (the Phase A invariant).

### Rollback

```
rm src/TrialMine/agents/query_router.py
rm tests/unit/test_query_router.py
```

---

## Step A4 — Build `react_agent.py` + add `submit_final_results` tool (~120 min)

**Goal:** The agent itself. A LangGraph `create_react_agent` bound to Haiku 4.5 and the existing `ALL_TOOLS` surface, with a new terminator tool (`submit_final_results`) that the agent MUST call to end the loop. The terminator pattern is the key to making Haiku reliable in an agentic loop — without it, free-text JSON output is brittle.

### Prompt to send (Part 1: terminator tool)

```
Step A4.1 — add the submit_final_results tool to
src/TrialMine/agents/tools.py per docs/build_agent.md.

CRITICAL: pass `return_direct=True` to the @tool decorator. This is
how langgraph.prebuilt.create_react_agent actually terminates the
loop — when ALL tool_calls in a response are marked return_direct,
the agent exits without going back to the LLM. Without this kwarg
the loop continues after submit_final_results and the agent just
keeps reasoning until recursion_limit fires. Verified against
langgraph 1.1.10 source (see Appendix — Why return_direct).

Add a new args schema + @tool decorator at the bottom of tools.py
(above ALL_TOOLS). IMPORTANT: define the Args class FIRST, then
the @tool function — matches the existing pattern in tools.py
(SearchTrialsArgs / ConceptArgs / EligibilityArgs / TrialDetailsArgs
are all declared before their decorated functions).

class SubmitFinalResultsArgs(BaseModel):
    results: list[dict] = Field(
        ...,
        description=(
            "Ordered list of trial dicts (highest relevance first). "
            "Each must include 'nct_id'; optional 'explanation' string."
        ),
        min_length=1, max_length=20,
    )
    reasoning: str = Field(
        ...,
        description="One-paragraph rationale for the final ordering.",
        min_length=10,
    )

@tool("submit_final_results", args_schema=SubmitFinalResultsArgs, return_direct=True)
def submit_final_results(
    results: list[dict],
    reasoning: str,
) -> str:
    """Terminate the agent loop and return the final ranked trial list.

    The agent MUST call this exactly once to end its turn — and it
    must be the ONLY tool called in that turn (mixing it with another
    tool defeats the return_direct termination check, which only
    fires when ALL tool_calls in a response are return_direct=True).
    The pipeline parses this tool's args as the agent's final output.

    Args:
        results: Ordered list of trial dicts (highest relevance first).
            Each dict must include nct_id (str) and may include a
            per-trial explanation (str). Other fields are looked up
            from the SQLite trials table by nct_id during synthesis.
        reasoning: One-paragraph rationale for the final ordering.
            Used in agent_trace; not shown to the user directly.

    Returns:
        JSON envelope confirming the submission was received. The
        return value is informational — the orchestrator extracts
        the args from the tool call, not the return.
    """
    # The tool is a no-op marker — its purpose is to signal terminate.
    # The orchestrator inspects the tool-call args, not this return.
    return json.dumps({"submitted": True, "n_results": len(results)})

Then update ALL_TOOLS:

ALL_TOOLS = [
    search_trials,
    lookup_medical_concept,
    check_trial_eligibility,
    get_trial_details,
    submit_final_results,   # NEW — must always be last
]

Add SubmitFinalResultsArgs to __all__. Show me the diff.
```

### Verify (Part 1)

```
Run:
  python -c "from TrialMine.agents.tools import ALL_TOOLS, submit_final_results; \
             print(len(ALL_TOOLS), submit_final_results.name); \
             import json; print(submit_final_results.invoke({'results': [{'nct_id': 'NCT123'}], 'reasoning': 'test reasoning'}))"

Expect:
  5 submit_final_results
  {"submitted": true, "n_results": 1}
```

### Acceptance criteria (Part 1)

- `ALL_TOOLS` has 5 tools, `submit_final_results` is last.
- The tool invokes cleanly.
- Existing 4 tools unchanged.

### Prompt to send (Part 2: the agent module)

```
Step A4.2 — create src/TrialMine/agents/react_agent.py per
docs/build_agent.md.

Module surface:

  class AgenticSearchAgent:
      def __init__(
          self,
          tools: list[BaseTool] | None = None,    # langchain_core.tools.BaseTool
          model: str = "claude-haiku-4-5",
          max_iters: int = 5,
          per_iter_timeout_s: float = 6.0,         # matches DegradationConfig default
          api_key: str | None = None,
      ): ...

      async def search(
          self,
          profile: PatientProfile,
      ) -> tuple[dict, list[dict]]:
          """Run the agent loop. Returns (result_dict, trace_entries).
          Shape matches SearchOrchestrator.search() so the pipeline
          can route to either without diverging downstream."""

Implementation requirements:

1. Use `langgraph.prebuilt.create_react_agent` to build the agent
   graph, bound to ChatAnthropic (langchain_anthropic) with
   model_name="claude-haiku-4-5" (the canonical Pydantic field name;
   `model=` also works via alias but `model_name=` is the explicit
   form). Pass tools=ALL_TOOLS (the 5-tool list from tools.py —
   including submit_final_results, which carries return_direct=True
   so the loop terminates after it). Pass the system prompt via the
   keyword arg `prompt=AGENT_SYSTEM_PROMPT` (NOT `state_modifier=` —
   that was the langgraph 0.1 parameter name and is deprecated in 1.x).

   **CRITICAL — `ChatAnthropic(timeout=…)` is the per-HTTP-request cap,
   NOT the loop budget.** A natural-looking choice is
   `timeout=per_iter_timeout_s` (i.e., 6s) — DO NOT do that. The
   Anthropic API call for a tool-using turn with a ~2000-token system
   prompt + 5 tool schemas needs ~5–8s warm (no caching) just to return
   the first tool-use response; setting the HTTP timeout to 6s
   cancels the very first call and the loop never makes any tool
   calls. The A8 smoke surfaces this with `httpx.ReadTimeout` →
   `anthropic.APITimeoutError`. The fix is
   `timeout=max(15.0, max_iters * per_iter_timeout_s)` — gives each
   HTTP request the whole loop budget (the outer
   `asyncio.wait_for(timeout=max_iters * per_iter_timeout_s)` in
   step 3 still caps the loop's wall-clock at the same value, so
   nothing runs longer than intended).

2. System prompt should be ~1500-2000 tokens. Structure:
   - Role: clinical-trial search expert helping a patient
   - Description of the 5 tools (one short paragraph each — link
     back to the tool's docstring, don't duplicate it)
   - Step-by-step recipe:
     (a) parse what the patient is asking for from the profile dict
     (b) call search_trials with the patient's primary terms
     (c) if results look off-topic OR if patient mentions a prior
         failure / progression, call lookup_medical_concept to
         expand vocabulary, then re-search
     (d) for the top 3 candidates, call get_trial_details to verify
         clinical context (read brief_summary + eligibility_criteria)
     (e) for the top 3 verified candidates, call check_trial_eligibility
         with the patient's age, sex, condition, prior_treatments
     (f) call submit_final_results with the ranked top 5–10 trials
         and a one-paragraph reasoning
   - Hard constraints:
     * MUST call submit_final_results exactly once to end the turn
     * submit_final_results MUST be called ALONE in its turn — do
       not call it together with another tool in the same response,
       or the return_direct termination check fails and the loop
       continues
     * MAX 5 tool-use cycles total — count before each call
     * If a tool returns an error, do NOT retry the same call with
       the same args; either adjust the args or move on
   - 2 few-shot examples showing the full tool-call sequence for:
     (i) a complex query: "failed osimertinib EGFR T790M NSCLC"
         → search_trials → lookup_medical_concept("osimertinib resistance")
         → search_trials("4th generation EGFR TKI") → get_trial_details
         → check_trial_eligibility → submit_final_results
     (ii) a vague query: "trials for my mom with breast cancer"
         → search_trials("breast cancer recruiting trials")
         → get_trial_details → submit_final_results

3. The async search() method:
   - Builds the initial message list from the profile: format the
     profile as a HumanMessage whose content is the raw_query plus
     a compact JSON blob of the extracted fields. Pass to the agent
     as `{"messages": [HumanMessage(content=...)]}` — that's the
     state shape `create_react_agent` expects.
   - Invokes the agent via `await agent.ainvoke(initial_state,
     config={"recursion_limit": max_iters * 2 + 2})`. The +2 covers
     the entry/exit transitions; each tool-use cycle is 2 node
     transitions (agent → tools → agent), so `max_iters * 2` is the
     minimum cap for max_iters cycles.
   - Wraps the invocation in asyncio.wait_for with total budget =
     max_iters * per_iter_timeout_s
   - Streams intermediate tool calls into trace_entries (one entry
     per tool call: {"step": "tool_call", "tool": <name>,
     "args_summary": <truncated dict>, "duration_ms": <float>})
   - On termination, extracts the submit_final_results tool-call
     args (the LAST AI message with that tool_use block)
   - Synthesizes the final result_dict by looking up each nct_id
     in SQLite (via _db_conn from tools.py) and merging with the
     agent's per-trial explanation
   - On timeout / cap-hit / parse failure: return a degraded result
     with error="agent_did_not_terminate" so the pipeline can fall
     back to the rule arm

4. Trace shape matches SearchOrchestrator's trace (so the existing
   agent_trace reducer in pipeline.py works unchanged):
     [
       {"step": "agent_start", "duration_ms": 0,
        "decisions": {"model": <str>, "max_iters": <int>}},
       {"step": "tool_call", "duration_ms": <float>,
        "decisions": {"tool": <str>, "args_summary": <dict>,
                      "iter": <int>}},
       ...,
       {"step": "agent_submit", "duration_ms": <float>,
        "decisions": {"n_results": <int>, "iters_used": <int>,
                      "reasoning_excerpt": <str truncated>}},
     ]

5. result_dict shape (MUST match orchestrator's contract):
     {
       "results": [<enriched trial dict>],
       "query_used": <agent's last search_trials query>,
       "filters": {},   # agent uses tool args, not pipeline filters
       "normalized_condition": <agent's last lookup_medical_concept input or None>,
     }

   Per-result enrichment uses the same shape orchestrator returns
   in its Step 6 — fields: nct_id, title, phase, status, enrollment,
   conditions, score (None for agent — no blender), source ("agent"),
   explanation (the agent's per-trial reasoning), warnings ([]),
   eligibility (the last check_trial_eligibility result for this
   trial, or None), url.

6. The constructor must raise ValueError if ANTHROPIC_API_KEY is
   missing (mirrors QueryParserAgent's behaviour).

Type hints, structured logging via logging.getLogger, no print().
Show me the diff and any subtleties you ran into.
```

### Verify (Part 2)

```
Smoke test against a mocked LLM (don't hit the real API yet):

  python -c "
  import asyncio
  from unittest.mock import patch, MagicMock
  from TrialMine.agents.react_agent import AgenticSearchAgent
  from TrialMine.agents.query_parser import PatientProfile

  # Just verify the class instantiates with the API key present
  import os
  if not os.getenv('ANTHROPIC_API_KEY'):
      print('SKIP — set ANTHROPIC_API_KEY first')
  else:
      agent = AgenticSearchAgent()
      print('Instantiated. model=', agent.model, 'max_iters=', agent.max_iters)
  "

Expect: "Instantiated. model= claude-haiku-4-5 max_iters= 5"
```

### Acceptance criteria (Part 2)

- File exists, imports cleanly, no syntax errors.
- Instantiation succeeds with `ANTHROPIC_API_KEY` set; raises `ValueError` without.
- System prompt is ≥ 1500 tokens (rough check: ≥ 6 KB of text in the prompt constant).
- `result_dict` and `trace_entries` shapes match the orchestrator contract (so pipeline.py can route to either without divergence).
- No actual LLM call yet — that comes in A8.

### Rollback (Parts 1 + 2)

```
cp src/TrialMine/agents/tools_v1.py.bak src/TrialMine/agents/tools.py
rm src/TrialMine/agents/react_agent.py
```

---

## Step A5 — Wire the agent into `pipeline.py` with conditional routing (~60 min)

**Goal:** Add a `route_decision` node and a conditional edge that splits to either the existing `execute_search` (rule arm) or the new `execute_agent_search` (agent arm). Both lead to END, with the existing `fallback_search` still hanging off the rule arm's error path. The agent arm's error path falls back to the rule arm — NOT to `fallback_search` — so we don't double-degrade.

### Prompt to send

```
Modify src/TrialMine/agents/pipeline.py per Phase A5 of
docs/build_agent.md.

Target graph shape:

         parse_query
              │
       route_decision
         /        \
    "rule"      "agent"
       │            │
   execute_search   execute_agent_search
       │  (error)   │  (error)
       │   ↓        │
       │ fallback_search ← also reached if execute_agent_search errors
       │   │        │
       └───┴────────┴──→ END

Changes:

1. Add a new module-level singleton _agent_singleton plus a
   _get_agent() helper, mirroring _get_parser / _get_orchestrator.
   Lazy-instantiate AgenticSearchAgent on first call. Reads from
   get_default_degradation() to pull max_iters + per_iter_timeout_s.

2. Add a new SearchState field:
     route: dict | None  # written by the route_decision node
   The dict is {"arm": "rule"|"agent", "reason": str, "signals": {...}}.
   Use override semantics (not the operator.add reducer).

3. Add a new node `route_decision`:

   async def route_decision(state: SearchState) -> dict:
       t0 = time.perf_counter()
       profile = PatientProfile.model_validate(
           state.get("patient_profile") or {"raw_query": state["raw_query"]}
       )
       config = get_default_degradation()
       from TrialMine.agents.query_router import route_decision as _route
       decision = _route(profile, config)
       elapsed_ms = (time.perf_counter() - t0) * 1000
       return {
           "route": {"arm": decision.arm, "reason": decision.reason,
                     "signals": decision.signals},
           "agent_trace": [{
               "step": "route_decision",
               "duration_ms": round(elapsed_ms, 2),
               "decisions": {"arm": decision.arm, "reason": decision.reason,
                             **decision.signals},
           }],
       }

4. Add a new node `execute_agent_search`. Same try/except pattern
   as execute_search:

   async def execute_agent_search(state: SearchState) -> dict:
       t0 = time.perf_counter()
       agent = _get_agent()
       profile = PatientProfile.model_validate(
           state.get("patient_profile") or {"raw_query": state["raw_query"]}
       )
       try:
           with time_stage_cm("agent_react_search"):
               result_dict, trace_entries = await agent.search(profile)
       except Exception as exc:
           elapsed_ms = (time.perf_counter() - t0) * 1000
           logger.exception("execute_agent_search failed; will retry as rule")
           record_agent_failure("execute_agent_search")
           return {
               "error": f"agent_arm_failed: {type(exc).__name__}: {exc}",
               "agent_trace": [{
                   "step": "execute_agent_search_error",
                   "duration_ms": round(elapsed_ms, 2),
                   "decisions": {"error_type": type(exc).__name__,
                                 "error": str(exc)},
               }],
           }
       return {"search_results": result_dict, "agent_trace": trace_entries}

5. Add a conditional edge function:

   def _route_after_parse(state: SearchState) -> Literal["execute_search", "execute_agent_search"]:
       route = state.get("route") or {}
       return "execute_agent_search" if route.get("arm") == "agent" else "execute_search"

6. Update _route_after_search to also handle the agent arm's
   error path. When state["error"] starts with "agent_arm_failed:",
   route to execute_search (RETRY as rule), not fallback_search.
   Use a new label for this case in the trace.

   def _route_after_search(state: SearchState) -> Literal["fallback_search", "execute_search", "__end__"]:
       error = state.get("error") or ""
       if error.startswith("agent_arm_failed:"):
           return "execute_search"  # retry as rule
       if error:
           return "fallback_search"
       return "__end__"

7. Update build_pipeline() to add the new nodes + edges. The full
   topology is:

   g.add_node("parse_query", parse_query)
   g.add_node("route_decision", route_decision)
   g.add_node("execute_search", execute_search)
   g.add_node("execute_agent_search", execute_agent_search)
   g.add_node("fallback_search", fallback_search)

   g.set_entry_point("parse_query")
   g.add_edge("parse_query", "route_decision")
   g.add_conditional_edges("route_decision", _route_after_parse, {
       "execute_search": "execute_search",
       "execute_agent_search": "execute_agent_search",
   })
   # Rule-arm conditional: rule errors → fallback; success → END.
   # Rule arm CANNOT produce an "agent_arm_failed:" error, so we don't
   # register the agent-retry edge here (would be dead code).
   g.add_conditional_edges("execute_search", _route_after_search, {
       "fallback_search": "fallback_search",
       "__end__": END,
   })
   # Agent-arm conditional: agent error → retry as rule via execute_search;
   # other errors → fallback; success → END.
   g.add_conditional_edges("execute_agent_search", _route_after_search, {
       "execute_search": "execute_search",   # agent_arm_failed retry
       "fallback_search": "fallback_search",
       "__end__": END,
   })
   g.add_edge("fallback_search", END)

8. Update the search() function's initial SearchState to include
   "route": None.

9. Update the final return at the bottom of search() to include
   route in the response dict, so callers can see which arm fired:

   return {
       ...,
       "route": final.get("route"),
   }

Preserve everything else: docstrings, type hints, existing trace
shape, timeout behaviour. Show me the diff.
```

### Verify

```
Run the build (no LLM calls — should just exercise routing logic):

  python -c "
  import asyncio
  from TrialMine.agents.pipeline import build_pipeline, search
  from TrialMine.config import DegradationConfig, get_default_degradation

  # Verify the graph compiles
  p = build_pipeline()
  print('Pipeline compiled.')
  print('Nodes:', list(p.get_graph().nodes.keys()))
  "

Expect (note: LangGraph's get_graph().nodes always includes both
`__start__` and `__end__` sentinels, plus your 5 real nodes):
  Pipeline compiled.
  Nodes: ['__start__', 'parse_query', 'route_decision', 'execute_search',
          'execute_agent_search', 'fallback_search', '__end__']
```

### Acceptance criteria

- Pipeline compiles without LangGraph validation errors.
- 5 real nodes (`parse_query`, `route_decision`, `execute_search`, `execute_agent_search`, `fallback_search`) plus LangGraph's `__start__` / `__end__` sentinels → 7 entries in `get_graph().nodes`.
- `agentic_path_enabled=False` (the default) means `route_decision` always returns `arm="rule"` → only `execute_search` runs → behaviour matches pre-Phase-A baseline.
- Existing 5 smoke queries in `scripts/test_pipeline.py` still work (run the script if ES is up).
- The agent-arm error path retries as rule, doesn't drop to fallback.

### Rollback

```
cp src/TrialMine/agents/pipeline_v1.py.bak src/TrialMine/agents/pipeline.py
```

---

## Step A6 — Add Prometheus counters for agent observability (~30 min)

**Goal:** 4 new metrics in `monitoring/metrics.py`, wired into the agent path. These are the minimum viable observability that justifies skipping a full Grafana dashboard for the first ship — but we add the dashboard anyway in Phase B because the eval story benefits from it.

### Prompt to send

```
Update src/TrialMine/monitoring/metrics.py per Phase A6 of
docs/build_agent.md. Add 4 new metrics + their accessors, matching
the existing style (module-level Histogram/Counter declarations +
helper functions that the agent code calls).

New metrics:

1. AGENT_ROUTING = Counter(
     "trialmine_agent_routing_total",
     "Routing decisions by arm",
     labelnames=["arm", "reason"],
   )
   Helper: record_agent_routing(arm: str, reason: str) -> None

2. AGENT_ITERATIONS = Histogram(
     "trialmine_agent_iterations",
     "Number of ReAct tool-call cycles per agent invocation",
     labelnames=["outcome"],   # "submitted" | "capped" | "errored"
     buckets=(1, 2, 3, 4, 5, 6, 7, 8),
   )
   Helper: record_agent_iterations(n: int, outcome: str) -> None

3. AGENT_TOOL_CALLS = Counter(
     "trialmine_agent_tool_calls_total",
     "Tool calls by name and status",
     labelnames=["tool_name", "status"],   # status: "ok" | "error"
   )
   Helper: record_agent_tool_call(tool_name: str, ok: bool) -> None

4. AGENT_COST_USD = Counter(
     "trialmine_agent_estimated_cost_usd_total",
     "Estimated agent USD spend (Anthropic API only, computed from "
     "token counts + Haiku 4.5 published rates)",
     labelnames=["model"],
   )
   Helper: record_agent_cost(model: str, input_tokens: int, output_tokens: int) -> None
   # Pricing constants — Haiku 4.5 as of 2026-05: $1/M input, $5/M output.
   # Sonnet 4.6: $3/M input, $15/M output. Hardcode a small dict;
   # missing model → log warning and skip.

Add all 4 to the __all__ list and to the re-exports in
src/TrialMine/monitoring/__init__.py. Show me both diffs.

Then wire the counters into src/TrialMine/agents/pipeline.py and
src/TrialMine/agents/react_agent.py:

- pipeline.py route_decision node: call record_agent_routing(arm, reason)
  after the routing decision is made.
- react_agent.py: call record_agent_tool_call(tool_name, ok=True/False)
  on every tool-use response. Call record_agent_iterations(n, outcome)
  exactly once at agent termination. Call record_agent_cost on
  termination using the cumulative input_tokens / output_tokens from
  the Anthropic API response usage fields (sum across all LLM calls
  in the loop).

Show me the wiring diffs.
```

### Verify

```
Run the existing /metrics test to confirm no regressions:

  python -m pytest tests/unit/test_api.py -k metrics -v

Then run a smoke that touches the new metrics without needing an
agent invocation:

  python -c "
  from TrialMine.monitoring import (record_agent_routing,
                                     record_agent_iterations,
                                     record_agent_tool_call,
                                     record_agent_cost,
                                     metrics_app)
  record_agent_routing('agent', 'sparse_profile')
  record_agent_iterations(3, 'submitted')
  record_agent_tool_call('search_trials', ok=True)
  record_agent_cost('claude-haiku-4-5', 2500, 800)

  # Pull the /metrics text
  from prometheus_client import generate_latest
  text = generate_latest().decode()
  for line in text.splitlines():
      if 'trialmine_agent' in line and not line.startswith('#'):
          print(line)
  "

Expect 4 metric series printed with the values just recorded.
```

### Acceptance criteria

- 4 new metrics exposed at `/metrics` (Prometheus pull endpoint).
- Existing metrics (`REQUEST_COUNT`, `REQUEST_LATENCY`, `SEARCH_STAGE_LATENCY`, etc.) unchanged in name and label set.
- `record_agent_cost` uses the published Haiku/Sonnet rates; pricing dict lives in one place for easy update.
- `test_api.py -k metrics` still passes.

### Rollback

```
cp src/TrialMine/monitoring/metrics_v1.py.bak src/TrialMine/monitoring/metrics.py
# Revert the wiring in pipeline.py + react_agent.py by hand (small change)
```

---

## Step A7 — Register the `agent_path_v1` experiment in ABTestRouter (~20 min)

**Goal:** Plug the agent routing decision through the existing A/B router (Decision 30 stub) so the experiment is bookkept the same way any future ranking experiment would be. Subject ID is a stable hash of the raw query — so the same query always lands in the same arm.

### Prompt to send

```
Update src/TrialMine/experiments/ab_test.py per Phase A7 of
docs/build_agent.md.

Changes:

1. Add a module-level convenience: a process-wide singleton router
   pre-built with the experiments TrialMine knows about today. Lazy
   init pattern.

   _default_router: ABTestRouter | None = None

   def get_default_router() -> ABTestRouter:
       global _default_router
       if _default_router is None:
           _default_router = ABTestRouter([
               Experiment(
                   name="agent_path_v1",
                   variants=[
                       ExperimentVariant(name="control", weight=0.5),
                       ExperimentVariant(name="treatment", weight=0.5),
                   ],
                   enabled=False,  # CRITICAL — off until Phase D
               ),
           ])
       return _default_router

   The enabled=False means every subject lands in the first variant
   ("control" = rule arm). Phase D flips this to True.

2. Add to __all__ in ab_test.py: get_default_router. AND add it to
   the re-export list in src/TrialMine/experiments/__init__.py
   (currently re-exports ABTestRouter, Experiment, ExperimentVariant)
   so callers can `from TrialMine.experiments import get_default_router`.

Then update the `route_decision` NODE in
src/TrialMine/agents/pipeline.py (the LangGraph node added in A5,
NOT the pure function in query_router.py) to consult the A/B router
BEFORE applying the heuristics. The pure function stays pure; the
node composes the two:

   async def route_decision(state):
       t0 = time.perf_counter()
       profile = PatientProfile.model_validate(
           state.get("patient_profile") or {"raw_query": state["raw_query"]}
       )
       config = get_default_degradation()

       # AB-router gate (only when agent path is enabled).
       if config.agentic_path_enabled:
           import hashlib
           from TrialMine.experiments.ab_test import get_default_router
           subject_id = hashlib.sha256(
               (profile.raw_query or "").encode()
           ).hexdigest()[:16]
           router = get_default_router()
           variant = router.route(subject_id=subject_id,
                                   experiment_name="agent_path_v1")
           router.log_exposure(subject_id, "agent_path_v1", variant)
           if variant == "control":
               return {
                   "route": {"arm": "rule",
                             "reason": "ab_router_control",
                             "signals": {"subject_id": subject_id,
                                         "variant": variant}},
                   "agent_trace": [{
                       "step": "route_decision",
                       "duration_ms": round((time.perf_counter()-t0)*1000, 2),
                       "decisions": {"arm": "rule",
                                     "reason": "ab_router_control",
                                     "variant": variant},
                   }],
               }
           # variant == "treatment": fall through to the heuristic.

       # Heuristic from Phase A3 (pure function, no IO).
       from TrialMine.agents.query_router import route_decision as _route
       decision = _route(profile, config)
       return {
           "route": {"arm": decision.arm, "reason": decision.reason,
                     "signals": decision.signals},
           "agent_trace": [{
               "step": "route_decision",
               "duration_ms": round((time.perf_counter()-t0)*1000, 2),
               "decisions": {"arm": decision.arm, "reason": decision.reason,
                             **decision.signals},
           }],
       }

This means the A/B router gates 50% of traffic into the agent eligibility
pool; within that pool, the heuristics further select which queries
actually go to the agent. So actual agent coverage =
~50% × (sparse OR complex) ≈ 12-25% of all traffic, depending on
query mix.

(NOTE: this is a REPLACEMENT for the simpler route_decision node
sketched in Phase A5. If you implemented A5 already, this step
overwrites that node body. The pure function in query_router.py
is unchanged.)

Add an integration-style test in tests/integration/test_route_node.py
(NOT in test_query_router.py — the pure-function tests there stay
pure, no AB router):
   - With agentic_path_enabled=True and the same query, invoking the
     route_decision node twice yields the SAME arm (router is
     deterministic via hash bucketing).
   - With the experiment disabled (the Phase A default), every call
     returns arm="rule" with reason="ab_router_control" (router's
     control behaviour — Experiment.enabled=False short-circuits to
     the first variant per existing ABTestRouter semantics).

Show me both diffs.
```

### Verify

```
  python -m pytest tests/unit/test_query_router.py tests/unit/test_ab_test.py -v

Expect: all tests pass. The new determinism test should be green.

Then a manual smoke testing the AB router determinism directly
(NOT the pure function — the pure function doesn't see the AB
router; that's the node's job per the architecture):

  python -c "
  from TrialMine.experiments.ab_test import get_default_router
  router = get_default_router()
  # Experiment is enabled=False at construction (Phase A invariant),
  # so every subject must land in the first variant ('control').
  variants = {router.route(subject_id=f'q{i}',
                           experiment_name='agent_path_v1')
              for i in range(20)}
  print('All control?', variants == {'control'})

  # Determinism: same subject always returns the same variant.
  v1 = router.route('subj-A', 'agent_path_v1')
  v2 = router.route('subj-A', 'agent_path_v1')
  print('Deterministic?', v1 == v2)
  "

Expect:
  All control? True       (experiment.enabled=False short-circuits)
  Deterministic? True
```

### Acceptance criteria

- A/B router has the `agent_path_v1` experiment with `enabled=False`.
- The PHASE A invariant holds: with experiment disabled, every query routes to rule arm.
- Deterministic routing: same query → same arm across 10 calls.
- All existing AB test tests still pass.

### Rollback

```
cp src/TrialMine/experiments/ab_test_v1.py.bak src/TrialMine/experiments/ab_test.py
# Revert query_router.py changes by removing the AB integration block
```

---

## Step A8 — End-to-end smoke test with live API (~30 min, ~$0.50)

**Goal:** First time the agent actually runs. Hit 5 representative queries with the agent forcibly enabled, confirm the loop terminates, results come back in the expected shape, and the trace is sensible. This catches the most common Day-1 failures: prompt-shape bugs, tool-call parsing issues, infinite loops, JSON parse failures on `submit_final_results`.

### Prompt to send

```
Run Phase A8 of docs/build_agent.md: temporarily enable the agent
path via env var, run scripts/test_pipeline.py with 5 hard-coded
agent-targeted queries, then print the full agent_trace per query.

1. Make sure docker es is up:
     docker start es && sleep 5

2. Create scripts/test_agent_path.py (new) that:
   - Mutates the module-level _DEFAULT_DEGRADATION singleton
     directly — DO NOT just monkey-patch get_default_degradation,
     because the agent caches degradation at construction time per
     A4.2. The reliable pattern (must run BEFORE importing pipeline
     / react_agent so the lazy singleton picks up the new config
     on first construction):
         import TrialMine.config as cfg
         cfg._DEFAULT_DEGRADATION = cfg.DegradationConfig(
             agentic_path_enabled=True,
             agentic_routing_threshold_slots=2,
             agentic_complex_pattern_enabled=True,
             agentic_max_iters=5,
             # Bumped from the production 6.0s default. The smoke runs
             # on a CPU box without Anthropic prompt caching enabled,
             # so each Haiku call costs the full ~5-8s and a single
             # search_trials tool call adds ~4s of CE rerank. With
             # max_iters=5 the total budget at per_iter=6 (=30s) is
             # too tight for a full 5-cycle loop and queries hit
             # max-iters before submitting. 12s per-iter (60s budget)
             # fits a real loop with room to spare.
             agentic_per_iter_timeout_s=12.0,
             # Widen the outer pipeline cap to 90s so a single 60s
             # agent run has ~30s of headroom on top. Defaults would
             # cancel the agent at 10s.
             skip_agent_if_slow_s=90.0,
         )
   - Forcibly sets the AB router experiment to enabled=True
     (so route_decision actually exercises the heuristics). Same
     pattern — mutate the router's _catalog before any pipeline
     call.
   - Defines 5 test queries:
     a. "failed osimertinib EGFR T790M NSCLC, looking for trials"
     b. "trials for my dad with prostate cancer"
     c. "BRCA1 mutation, ovarian cancer, prior PARP inhibitor"
     d. "metastatic HER2-positive breast cancer after trastuzumab"
     e. "pediatric acute lymphoblastic leukemia trials"
   - For each query: call pipeline.search(query, build_pipeline(),
     timeout=90.0), print:
       - which arm was used (from result["route"])
       - n_results
       - iters used (extract from agent_submit trace entry)
       - tool-call sequence (list of tool names in order)
       - total cost (sum of agent_cost recorded — fetch from /metrics
         delta if convenient, or compute from trace usage fields)

3. Run the script: OMP_NUM_THREADS=1 python scripts/test_agent_path.py

4. Report:
   - Per-query: arm, n_results, iters, tool sequence, latency
   - Aggregate: agent-arm count, rule-arm count, mean iters,
     cumulative cost
   - Any anomalies (timeouts, parse failures, empty results)
```

### Verify

After the script runs, eyeball each query's trace:

```
For each of the 5 queries, confirm:

(a) If routed to agent arm:
    - 2–5 tool calls (NOT 1, not 7+)
    - submit_final_results is the LAST tool call
    - n_results in [3, 10]
    - No "agent_did_not_terminate" error
(b) If routed to rule arm:
    - Trace matches the existing 6-step rule-pipeline shape
    - No agent-related steps

(c) Aggregate cost: < $1.00 total for 5 queries
(d) Latency: each agent-routed query under ~60s on the smoke setup.
    The original "12s soft budget" target assumes Anthropic prompt
    caching is enabled on the system prompt (the auto-cache only
    fires past 4096 input tokens — our prompt is ~2200 tokens, so
    no auto-cache). With explicit `cache_control: {"type":
    "ephemeral"}` on AGENT_SYSTEM_PROMPT, expect a ~3-5× drop on
    repeat calls; without it the warm latency is 25-60s per agent
    query, dominated by (a) ~4s cross-encoder rerank inside every
    search_trials call and (b) ~5-8s per Haiku call across 4-5
    iterations.
```

### Acceptance criteria

- All 5 queries complete without raising.
- ≥ 3 of the 5 route to the agent arm (queries a, c, d are designed to trigger heuristics).
- For each agent-routed query: 2–5 tool calls, `submit_final_results` is last, `n_results > 0`.
- Cumulative API cost ≤ $1.00 (sanity check — if it's $5+, the prompt is leaking tokens).
- No "agent_did_not_terminate" errors.

### Rollback / Troubleshooting

Most likely Day-1 failure modes for this step, in order of frequency observed:

1. **Every agent query times out at ~30s with 0 tool calls and an
   `anthropic.APITimeoutError` traceback.** The
   `ChatAnthropic(timeout=…)` constructor argument is too small for
   a tool-using Haiku call with a 2000-token prompt; the HTTP request
   is cancelled before the first response. Fix per A4.2: set
   `timeout=max(15.0, max_iters * per_iter_timeout_s)`, not
   `timeout=per_iter_timeout_s`.

2. **A query loops past the cap (max_iters hit without
   submit_final_results).** Haiku is parallel-calling tools within
   one cycle (e.g., iter 4 fires 3× `get_trial_details`
   simultaneously), so "5 cycles" can mean 8–9 tool calls — the agent
   burns the iter budget on info gathering and never reaches the
   terminator. Two mitigations, often needed together: (i) strengthen
   the prompt with an explicit step-counter sentence — *"You MUST
   call submit_final_results within 5 tool-use cycles. On cycle ≥ 4
   you MUST be calling submit_final_results, even if eligibility
   checks are incomplete."* (ii) bump production
   `agentic_max_iters` to 6 to give buffer for parallel-call bursts.

3. **Per-query cost is 2–4× the projection (~$0.04 vs the projected
   ~$0.014).** Anthropic prompt caching isn't engaging on
   AGENT_SYSTEM_PROMPT because it's just below the 4096-token
   auto-cache threshold. Add `cache_control: {"type": "ephemeral"}`
   on the system prompt (mirror the pattern in
   `query_parser.py:248`). This is a one-line change and would
   drop the per-call input cost by ~10× on warm cache hits.

Don't move on to Phase B until the smoke shows the agent loop
producing ranked trials end-to-end on at least 3 of the 5 queries
with `submit_final_results` as the last tool call.

---

# Phase B — Trace observability (~2–3 hr, $0)

## Step B0 — Session bootstrap

Same as A0, but say:

```
We are in Phase B of docs/build_agent.md (trace observability).
Phase A is complete. Read build_agent.md and report which Phase B
steps are done based on file state (data/agent_runs.db,
src/TrialMine/monitoring/trace_store.py, scripts/init_trace_db.py,
docker-compose.yml SQLite plugin,
infrastructure/grafana/dashboards/agent_observability.json,
infrastructure/grafana/provisioning/datasources/sqlite.yml).
Do not modify yet.
```

---

## Step B1 — Create SQLite schema + init script (~30 min)

**Goal:** A tiny SQLite store for one row per pipeline invocation, plus child rows for per-stage durations and per-tool-call detail. Lives at `data/agent_runs.db`, separate from `trials.db` so it can be rotated / dropped independently. Gitignored.

### Prompt to send

```
Create scripts/init_trace_db.py per Phase B1 of docs/build_agent.md.

The script:
1. Opens (or creates) data/agent_runs.db
2. Runs the schema below in a single transaction with IF NOT EXISTS
3. Reports table counts after creation
4. Is idempotent — running twice is a no-op

Schema:

  CREATE TABLE IF NOT EXISTS agent_runs (
      query_id          TEXT PRIMARY KEY,    -- uuid4 generated per call
      raw_query         TEXT NOT NULL,
      ts                INTEGER NOT NULL,    -- unix ms
      arm               TEXT NOT NULL,       -- 'rule' | 'agent' | 'fallback'
      route_reason      TEXT,                -- from route_decision
      pipeline_kind     TEXT,                -- from rule arm's retrieve step
      used_fallback     INTEGER NOT NULL DEFAULT 0,
      total_ms          REAL,
      n_results         INTEGER,
      error             TEXT,                -- nullable
      route_signals     TEXT                 -- JSON blob of routing signals
  );
  CREATE INDEX IF NOT EXISTS idx_runs_ts ON agent_runs(ts);
  CREATE INDEX IF NOT EXISTS idx_runs_arm ON agent_runs(arm);
  CREATE INDEX IF NOT EXISTS idx_runs_kind ON agent_runs(pipeline_kind);

  CREATE TABLE IF NOT EXISTS agent_stages (
      query_id     TEXT NOT NULL,
      stage        TEXT NOT NULL,            -- e.g. 'parse_query', 'tool_call'
      duration_ms  REAL NOT NULL,
      seq          INTEGER NOT NULL,         -- order within the trace
      FOREIGN KEY (query_id) REFERENCES agent_runs(query_id)
  );
  CREATE INDEX IF NOT EXISTS idx_stages_qid ON agent_stages(query_id);
  CREATE INDEX IF NOT EXISTS idx_stages_stage ON agent_stages(stage);

  CREATE TABLE IF NOT EXISTS agent_tool_calls (
      query_id     TEXT NOT NULL,
      iter_n       INTEGER NOT NULL,         -- 1-indexed cycle in the loop
      tool_name    TEXT NOT NULL,
      ok           INTEGER NOT NULL,         -- 0/1
      duration_ms  REAL,
      args_summary TEXT,                     -- truncated JSON
      FOREIGN KEY (query_id) REFERENCES agent_runs(query_id)
  );
  CREATE INDEX IF NOT EXISTS idx_tools_qid ON agent_tool_calls(query_id);
  CREATE INDEX IF NOT EXISTS idx_tools_name ON agent_tool_calls(tool_name);

Also append to .gitignore (the main `.db` is already covered by the
existing `data/*.db` pattern; add only the SQLite sidecar files which
are not matched by `*.db`):
  # SQLite write-ahead log / shared-memory / hot-journal sidecars.
  # data/*.db (main DB file) is already ignored higher up.
  data/*.db-journal
  data/*.db-wal
  data/*.db-shm

Use type hints + structured logging. Show me the diff.
```

### Verify

```
Run:
  python scripts/init_trace_db.py
  sqlite3 data/agent_runs.db ".tables"
  sqlite3 data/agent_runs.db ".schema agent_runs"

Expect: 3 tables (agent_runs, agent_stages, agent_tool_calls).
Schema matches the spec.

Re-run the script — should report "already initialized" without errors.
```

### Acceptance criteria

- 3 tables created with correct schema.
- Re-running the init is idempotent.
- `.gitignore` updated.

### Rollback

```
rm data/agent_runs.db*
rm scripts/init_trace_db.py
```

---

## Step B2 — Build `trace_store.py` writer (~45 min)

**Goal:** A small synchronous writer that consumes the `agent_trace` from `pipeline.search()` and inserts one `agent_runs` row + N `agent_stages` + M `agent_tool_calls` rows per invocation. Synchronous is fine — SQLite writes a few rows in well under 1 ms; an async queue is over-engineered for this scale.

### Prompt to send

```
Create src/TrialMine/monitoring/trace_store.py per Phase B2 of
docs/build_agent.md.

Module surface:

  from pathlib import Path
  DEFAULT_DB_PATH = Path(os.getenv("TRIALMINE_TRACE_DB",
                                    "data/agent_runs.db"))

  def write_trace(
      query_id: str,
      raw_query: str,
      arm: str,
      route_reason: str | None,
      route_signals: dict | None,
      pipeline_kind: str | None,
      used_fallback: bool,
      total_ms: float,
      n_results: int,
      error: str | None,
      stages: list[dict],         # raw agent_trace from pipeline.search()
      db_path: Path | str = DEFAULT_DB_PATH,
  ) -> None:
      """Insert one run + N stages + M tool_calls.

      Synchronous. Single transaction. Logs + swallows any
      sqlite error — we never break the search response on a
      trace-write failure.
      """

Behavior:
  - Open SQLite with check_same_thread=False and a 1.0s busy_timeout
    (the pipeline runs from FastAPI thread pool; multiple workers
    might write concurrently)
  - Insert the agent_runs row (route_signals as JSON via json.dumps)
  - For each stage in `stages`, insert one agent_stages row with
    sequential seq starting at 1
  - For each stage where stage["step"] == "tool_call", also insert
    one agent_tool_calls row pulling tool_name + ok + args_summary
    from stage["decisions"]
  - Wrap the whole thing in try/except sqlite3.Error — on failure,
    log a warning and return (don't raise)

Also add a small helper:

  def init_db_if_missing(db_path: Path | str = DEFAULT_DB_PATH) -> None:
      """Run scripts/init_trace_db.py logic in-process if the DB file
      doesn't exist. Idempotent. Called at startup."""

Unit test (tests/unit/test_trace_store.py):
  - write_trace inserts the expected row counts
  - Two concurrent writes (use threading) don't deadlock
  - write_trace called with a malformed stages list logs + returns,
    doesn't raise
  - The init_db_if_missing helper is safe to call multiple times

Use a tmp_path fixture for the test DB. Show me both diffs.
```

### Verify

```
  python -m pytest tests/unit/test_trace_store.py -v

Expect: all pass.

Then a manual smoke:
  python -c "
  from TrialMine.monitoring.trace_store import write_trace, init_db_if_missing
  import uuid, time
  init_db_if_missing()
  qid = str(uuid.uuid4())
  write_trace(
      query_id=qid, raw_query='smoke test', arm='rule',
      route_reason='ab_router_control', route_signals={'variant':'control'},
      pipeline_kind='full_pipeline', used_fallback=False,
      total_ms=1234.5, n_results=10, error=None,
      stages=[
          {'step':'parse_query','duration_ms':50.0,'decisions':{}},
          {'step':'retrieve','duration_ms':800.0,'decisions':{}},
      ],
  )
  import sqlite3
  c = sqlite3.connect('data/agent_runs.db')
  print('runs:', c.execute('SELECT COUNT(*) FROM agent_runs WHERE query_id=?', (qid,)).fetchone()[0])
  print('stages:', c.execute('SELECT COUNT(*) FROM agent_stages WHERE query_id=?', (qid,)).fetchone()[0])
  "

Expect: runs: 1, stages: 2.
```

### Acceptance criteria

- All unit tests pass.
- Manual smoke writes 1 run + 2 stages.
- Concurrent-write test doesn't hang.

### Rollback

```
rm src/TrialMine/monitoring/trace_store.py
rm tests/unit/test_trace_store.py
```

---

## Step B3 — Wire `write_trace` into `pipeline.search()` (~20 min)

**Goal:** One call at the end of `pipeline.search()` to persist the trace. Failure is non-fatal.

### Prompt to send

```
Modify src/TrialMine/agents/pipeline.py per Phase B3 of
docs/build_agent.md.

Changes:

1. Add `import uuid` at the top and a TRACE_ENABLED env-toggle:
     TRACE_PERSIST_ENABLED = os.getenv("TRIALMINE_TRACE_PERSIST", "1") == "1"

2. At the end of search() — AFTER the final response dict is built
   but BEFORE returning — call write_trace. Wrap in try/except so a
   trace failure can never break the response.

   (This is ADDITIVE to the existing `record_agent_trace(...)` call
   in src/TrialMine/api/routes.py:168, which fans the trace into
   Prometheus SEARCH_STAGE_LATENCY histograms — different surface,
   different purpose. Don't remove or move that call; just add the
   SQLite persist here.)

   if TRACE_PERSIST_ENABLED:
       try:
           from TrialMine.monitoring.trace_store import write_trace
           write_trace(
               query_id=str(uuid.uuid4()),
               raw_query=patient_description,
               arm=(final.get("route") or {}).get("arm", "unknown"),
               route_reason=(final.get("route") or {}).get("reason"),
               route_signals=(final.get("route") or {}).get("signals"),
               pipeline_kind=_extract_pipeline_kind(final.get("agent_trace") or []),
               used_fallback=final.get("used_fallback", False),
               total_ms=round(elapsed_ms, 2),
               n_results=len((final.get("search_results") or {}).get("results") or []),
               error=final.get("error"),
               stages=final.get("agent_trace") or [],
           )
       except Exception:
           logger.exception("trace persist failed; not blocking response")

3. Add a small helper:
     def _extract_pipeline_kind(trace: list[dict]) -> str | None:
         """Pull pipeline_kind from the rule arm's retrieve step; None for agent."""
         for entry in trace:
             if entry.get("step") == "retrieve":
                 return (entry.get("decisions") or {}).get("pipeline")
         return None

4. The timeout and outer-except branches of search() don't have a
   `final` state object — `pipeline.ainvoke()` never returned. So
   the trace_store call there must build the args from raw values
   the branch already has, NOT from final.get(...). Concretely:

     # in the except TimeoutError branch:
     write_trace(
         query_id=str(uuid.uuid4()),
         raw_query=patient_description,
         arm="unknown",                     # never got to route_decision
         route_reason=None,
         route_signals=None,
         pipeline_kind=None,
         used_fallback=True,
         total_ms=round(elapsed_ms, 2),
         n_results=0,
         error=f"pipeline exceeded {timeout}s budget",
         stages=[{"step": "timeout", "duration_ms": round(elapsed_ms, 2),
                  "decisions": {"timeout_s": timeout}}],
     )

   The except Exception branch is the same shape with the actual
   exception type/message in error/stages.

5. Do NOT call init_db_if_missing() at module import — that fires on
   every Python process that imports pipeline.py (tests, scripts,
   the API), polluting random CWDs with empty DB files. Instead,
   call it from inside write_trace() guarded by a module-level
   `_initialized: bool = False` flag, so the schema check happens
   exactly once on the first real write. If TRIALMINE_TRACE_PERSIST=0
   the flag never flips and no DB is ever created — the desired
   behaviour for opt-out.

Show me the diff.
```

### Verify

```
Run scripts/test_pipeline.py with ES up, then check the DB:

  docker start es && sleep 5
  OMP_NUM_THREADS=1 python scripts/test_pipeline.py
  sqlite3 data/agent_runs.db "SELECT arm, COUNT(*) FROM agent_runs \
                              WHERE ts > (CAST(strftime('%s','now') AS INTEGER) - 600) * 1000 \
                              GROUP BY arm;"

Expect: at least one row per query in scripts/test_pipeline.py
(probably 5 rule-arm rows since agentic_path_enabled is False).
```

### Acceptance criteria

- Every pipeline invocation produces an `agent_runs` row.
- Stages are persisted; `agent_stages` count > `agent_runs` count.
- A simulated trace-write error doesn't break the response.

### Rollback

Set `TRIALMINE_TRACE_PERSIST=0` to disable persistence without code changes. Or revert the pipeline.py edit.

---

## Step B4 — Provision the SQLite datasource in Grafana (~20 min)

**Goal:** Make the trace DB visible to the Grafana container.

### Prompt to send

```
Update docker-compose.yml + add
infrastructure/grafana/provisioning/datasources/sqlite.yml per Phase B4
of docs/build_agent.md.

(Note: the project's Grafana provisioning lives at infrastructure/grafana/,
not monitoring/grafana/. The existing datasource.yml there registers
Prometheus; we're adding a sibling sqlite.yml for the trace store.)

1. In docker-compose.yml under the grafana service:
   - Add env GF_INSTALL_PLUGINS=frser-sqlite-datasource
   - Mount the trace DB read-only into the container, appending to
     the existing volumes block:
       - ./data/agent_runs.db:/var/lib/grafana/agent_runs.db:ro
   - Confirm the existing mount of
     ./infrastructure/grafana/provisioning:/etc/grafana/provisioning:ro
     is preserved (it already exists from the Phase 7 setup).

2. Create infrastructure/grafana/provisioning/datasources/sqlite.yml:

   apiVersion: 1
   datasources:
     - name: AgentTraces
       type: frser-sqlite-datasource
       uid: agent_traces_sqlite
       jsonData:
         path: /var/lib/grafana/agent_runs.db

   The existing datasource.yml in the same directory keeps Prometheus
   registered; Grafana provisioning loads all yml files in this dir.

Show me both diffs.
```

### Verify

```
  docker compose down && docker compose up -d grafana
  sleep 10
  # NOTE: project docker-compose.yml sets GF_SECURITY_ADMIN_PASSWORD=trialmind
  # (not "admin"). Use admin:trialmind here, not admin:admin.
  curl -s -u admin:trialmind http://localhost:3000/api/datasources | \
    python -c "import sys, json; \
               ds = json.load(sys.stdin); \
               print([d['name'] for d in ds])"

Expect: ['Prometheus', 'AgentTraces']  (Prometheus from Phase 7)
```

### Acceptance criteria

- Grafana lists the `AgentTraces` datasource.
- A test query in the Grafana Explore UI against AgentTraces returns rows.
- Existing Prometheus datasource is preserved and still default.

### Rollback

```
# Edit docker-compose.yml to drop the mount + env
rm infrastructure/grafana/provisioning/datasources/sqlite.yml
docker compose down && docker compose up -d grafana
```

---

## Step B5 — Build the 4-panel dashboard (~45 min)

**Goal:** The dashboard JSON checked into the repo so anyone can spin it up. Provisioned automatically from `infrastructure/grafana/dashboards/` via the existing `dashboards.yml` file provider.

### Prompt to send

```
Create infrastructure/grafana/dashboards/agent_observability.json per
Phase B5 of docs/build_agent.md.

(Note: the project's dashboard provider config already exists at
infrastructure/grafana/provisioning/dashboards/dashboards.yml — it
auto-loads every JSON file dropped into
infrastructure/grafana/dashboards/. So we just create the new
dashboard JSON; no provisioning file changes needed.)

The dashboard is named "TrialMine — Agent Observability" with 4
panels:

PANEL 1 — Routing distribution (pie chart)
  Title: "Routing distribution (last 24h)"
  Query (SQLite):
    SELECT arm AS metric, COUNT(*) AS value
    FROM agent_runs
    WHERE ts > (CAST(strftime('%s', 'now') AS INTEGER) - 86400) * 1000
    GROUP BY arm

PANEL 2 — Fallback rate over time (timeseries line)
  Title: "Fallback rate (1h bucket)"
  Query:
    SELECT
      datetime(ts/1000, 'unixepoch') AS time,
      AVG(used_fallback) AS fallback_rate
    FROM agent_runs
    WHERE ts > (CAST(strftime('%s','now') AS INTEGER) - 86400) * 1000
    GROUP BY strftime('%Y-%m-%d %H', datetime(ts/1000,'unixepoch'))
    ORDER BY time

PANEL 3 — Per-stage p50/p95 latency (timeseries, faceted)
  Title: "Per-stage latency (last 24h)"
  Two metrics from agent_stages:
    - p50 (median) per stage
    - p95 per stage
  Query (SQLite percentiles need a workaround — use a window function
  approximation or compute in two passes; if too gnarly, ship p50
  + max for v1 and document the limitation in the panel description):
    SELECT
      stage,
      AVG(duration_ms) AS mean_ms,
      MAX(duration_ms) AS max_ms,
      COUNT(*) AS n
    FROM agent_stages
    WHERE query_id IN (
        SELECT query_id FROM agent_runs
        WHERE ts > (CAST(strftime('%s','now') AS INTEGER) - 86400) * 1000
    )
    GROUP BY stage

PANEL 4 — Agent tool-call distribution (bar chart)
  Title: "Agent tool calls by name (last 24h)"
  Query:
    SELECT
      tool_name,
      SUM(ok) AS ok_count,
      SUM(1 - ok) AS error_count,
      COUNT(*) AS total
    FROM agent_tool_calls
    WHERE query_id IN (
        SELECT query_id FROM agent_runs
        WHERE ts > (CAST(strftime('%s','now') AS INTEGER) - 86400) * 1000
    )
    GROUP BY tool_name
    ORDER BY total DESC

Use uid = "trialmine-agent-obs". Refresh: 1m. Time range: now-24h.
Every panel's datasource UID should be "agent_traces_sqlite" (matches
what we set in B4's sqlite.yml).

(DO NOT create a new dashboards.yml — one already exists at
infrastructure/grafana/provisioning/dashboards/dashboards.yml from
the Phase 7 setup, with provider config that auto-loads every JSON
in infrastructure/grafana/dashboards/. Verify it's there with
`cat infrastructure/grafana/provisioning/dashboards/dashboards.yml`
before assuming you need to create one.)

Show me the new dashboard JSON.
```

### Verify

```
  docker compose restart grafana
  sleep 8
  # Open http://localhost:3000 → Dashboards → "TrialMine — Agent
  # Observability" — confirm 4 panels render. Some may be empty if
  # data/agent_runs.db has no rows yet.

  # Generate some traffic to populate panels:
  OMP_NUM_THREADS=1 python scripts/test_pipeline.py

  # Refresh the dashboard — Panel 1 should show a single "rule" wedge.
```

### Acceptance criteria

- Dashboard appears in Grafana automatically (provisioning).
- All 4 panels render without query errors (rows may be empty).
- After running `scripts/test_pipeline.py`, Panel 1 shows the rule arm count.

### Rollback

```
rm infrastructure/grafana/dashboards/agent_observability.json
# Do NOT delete the pre-existing dashboards.yml — it serves other
# dashboards too. Leave it alone.
```

---

## Step B6 — End-to-end observability smoke (~15 min)

**Goal:** Push 20 queries through (mixed arms), verify all 4 panels populate, sanity-check the counts.

### Prompt to send

```
Run Phase B6 of docs/build_agent.md: run 20 mixed queries (10
agent-targeted + 10 rule-targeted) through the pipeline with
agentic_path_enabled=True and the AB router experiment forcibly
enabled, then take screenshots of the 4 Grafana panels.

1. Create a one-off scripts/observability_smoke.py that:
   - Patches DegradationConfig and the AB router as in A8
   - Defines 20 queries: 10 "vague/complex" (likely agent arm) +
     10 "easy explicit" (likely rule arm)
   - Loops through them, prints arm + n_results for each

2. Run: OMP_NUM_THREADS=1 python scripts/observability_smoke.py

3. Open http://localhost:3000/d/trialmine-agent-obs
   Confirm all 4 panels show data with values matching the
   expected ~50/50 arm split (within ±2 due to AB hash variance).

4. Query the DB directly to cross-check Panel 1:
   sqlite3 data/agent_runs.db "SELECT arm, COUNT(*) FROM agent_runs \
                               WHERE ts > (CAST(strftime('%s','now') AS INTEGER) - 600) * 1000 \
                               GROUP BY arm;"
```

### Acceptance criteria

- 20 rows in `agent_runs` with ts within the last 10 minutes.
- Panel 1 totals match the DB count.
- Panel 4 shows ≥ 3 distinct tool names with non-zero ok_count.
- Panel 2 shows fallback_rate near 0 (we shouldn't be falling back).

### Rollback

N/A (read-only inspection). Drop rows if you want a clean slate:
```
sqlite3 data/agent_runs.db "DELETE FROM agent_runs WHERE ts > (CAST(strftime('%s','now') AS INTEGER) - 600) * 1000;"
```

---

# Phase C — Eval + decision gate (~3–4 hr, ~$2.50 Haiku total)

## Step C0 — Session bootstrap

```
We are in Phase C of docs/build_agent.md (eval + decision gate).
Phases A and B are complete. Read build_agent.md and report which
eval artifacts already exist:
  - data/evaluation/full_labeled_dataset.jsonl (the 65q baseline)
  - data/evaluation/full_labeled_dataset_expansion_v2.jsonl (the 20q)
  - data/evaluation/full_labeled_dataset_agent_all.jsonl (NEW Phase C2)
  - data/evaluation/agent_vs_rule_v1.json (NEW Phase C3)
Report what's missing. Do not modify yet.
```

---

## Step C1 — Baseline re-eval with agent OFF (~30 min, $0 — re-uses existing labels)

**Goal:** Confirm the rule arm hasn't regressed under the Phase A wiring changes. If this fails, something in the routing or pipeline edits is broken and we never get to compare against the agent.

### Prompt to send

```
Run Phase C1 of docs/build_agent.md: re-eval the rule arm on the
held-out 65q + 20q expansion to confirm Phase A's wiring didn't
regress baseline behavior.

NOTE: this is a *measurement* step against existing labels, not a
labeling step. We use scripts/evaluate.py (which reads --labels and
prints NDCG ablation tables), NOT scripts/build_full_eval_dataset.py
(which would call Haiku to generate new labels). The eval script
does not have an --apply-eligibility-filter flag; the filter is
controlled by DegradationConfig.eligibility_hard_filter_enabled
(default True) and runs inside the orchestrator path the evaluator
exercises.

1. Confirm degradation config (default — agentic_path_enabled=False):
     python -c "from TrialMine.config import get_default_degradation; \
                print(get_default_degradation().agentic_path_enabled)"
   Expect: False.

2. Run on the 65q held-out:
     docker start es && sleep 5
     OMP_NUM_THREADS=1 python scripts/evaluate.py \
         --labels data/evaluation/full_labeled_dataset.jsonl

3. Run on the 20q expansion:
     OMP_NUM_THREADS=1 python scripts/evaluate.py \
         --labels data/evaluation/full_labeled_dataset_expansion_v2.jsonl

4. Compute NDCG@5 / NDCG@10 / MRR for both, and compare to the
   pre-Phase-A baselines stored in CLAUDE.md (v6 LightGBM):
     - 65q NDCG@5 = 0.783
     - 20q NDCG@5 = 0.552

5. Report any deltas > ±0.01 in absolute NDCG@5 — these would
   indicate the Phase A wiring is leaking into the rule path
   (which it should NOT, because agentic_path_enabled=False).
```

### Acceptance criteria

- 65q NDCG@5 within 0.01 of 0.783 (rule baseline preserved).
- 20q NDCG@5 within 0.01 of 0.552.
- No new errors in any per-query metric.

### Rollback

If baseline regressed, the Phase A wiring is leaking. The most likely culprit is the conditional edge: confirm `_route_after_parse` returns `"execute_search"` when `route["arm"] == "rule"`. Run the C1 step again after fixing.

---

## Step C2 — Agent ON for ALL routed queries (~60 min, ~$2 Haiku + agent)

**Goal:** Force every query through the agent arm (no router gating). This is NOT a production setting — it's the upper-bound measurement of agent quality. Tells us "if every query went to the agent, what would NDCG and cost look like?"

### Prompt to send

```
Create scripts/eval_agent_path.py per Phase C2 of docs/build_agent.md.

The script:
1. Imports and patches DegradationConfig at runtime to set
   agentic_path_enabled=True
2. Imports and patches the AB router's agent_path_v1 experiment to
   force variant="treatment" (so the router's 50% gate doesn't
   downsample the eval set)
3. Patches the heuristic in query_router.py to ALWAYS return
   arm="agent" (so we measure the agent on every query, not just
   the sparse/complex ones)
4. Runs the same retrieval logic as scripts/build_full_eval_dataset.py
   but uses pipeline.search() instead of orchestrator-direct, so
   the agent arm is exercised
5. Writes per-query agent results to
   data/evaluation/full_labeled_dataset_agent_all.jsonl with the
   same schema as full_labeled_dataset.jsonl, PLUS extra fields:
     - agent_iters_used (int)
     - agent_tool_sequence (list[str])
     - agent_input_tokens (int)
     - agent_output_tokens (int)
     - agent_cost_usd (float)
     - agent_latency_ms (float)
6. Resumable via --resume (skip queries whose nct_ids are already
   labeled)
7. CLI: --labels <path> --output <path> --limit N (for dry-run)

After the script is built:

  OMP_NUM_THREADS=1 python scripts/eval_agent_path.py \
      --labels data/evaluation/full_labeled_dataset.jsonl \
      --output data/evaluation/full_labeled_dataset_agent_all_65q.jsonl \
      --resume

  OMP_NUM_THREADS=1 python scripts/eval_agent_path.py \
      --labels data/evaluation/full_labeled_dataset_expansion_v2.jsonl \
      --output data/evaluation/full_labeled_dataset_agent_all_20q.jsonl \
      --resume

Note: the agent's results may include NCT IDs not in the existing
labeled pool. For those, label with Haiku (re-use the labeling
prompt from build_full_eval_dataset.py). At the project's measured
rate (CLAUDE.md Phase 9: 1,300 pairs ≈ $1, so ~$0.0008/pair) the
expected cost for 85 queries × ~5 new nct_ids each = 425 pairs is
~$0.34–$0.85 depending on prompt size. Budget $2 to be safe.

Report:
  - Total cost (agent API + labeling)
  - Per-category NDCG@5 / NDCG@10 / MRR with bootstrap 95% CIs
  - Mean iters used per query (target: 3-5; alarm if > 5)
  - p95 latency per query
  - Tool-call frequency (which tools the agent leaned on)
```

### Acceptance criteria

- All 85 queries (65q + 20q) have agent results written.
- Mean iters used ∈ [2.5, 5.5] (within budget, not infinite-looping).
- p95 latency < 12s.
- Per-category NDCG@5 computed with bootstrap CIs.
- Total cost < $5 (agent API + labeling combined; if substantially
  higher, the agent's per-call prompt is leaking tokens or labeling
  is hitting more new NCTs than expected — investigate before
  proceeding to C3).

### Rollback

The agent results are additive (new JSONL file). To revert, just delete:
```
rm data/evaluation/full_labeled_dataset_agent_all_*.jsonl
```

---

## Step C3 — A/B comparison with selective routing (~45 min, $0 — offline analysis only)

**Goal:** Production setting. Router enabled, heuristics applied, 50% of traffic in the treatment arm and the heuristic further selecting which subset of those queries actually go to the agent. Compute per-arm metrics and the total cost-quality picture.

### Prompt to send

```
Run Phase C3 of docs/build_agent.md: compute the production A/B
comparison metrics.

Inputs:
  - Rule arm: data/evaluation/full_labeled_dataset.jsonl + the
    expansion (the pre-Phase-A baseline)
  - Agent arm: data/evaluation/full_labeled_dataset_agent_all_*.jsonl
    (from C2)

Apply the router decision logic OFFLINE to label each of the 85
queries with what arm they WOULD route to in production:

  from TrialMine.agents.query_router import route_decision
  from TrialMine.config import DegradationConfig
  c = DegradationConfig(agentic_path_enabled=True)
  for each query:
      # Need the patient_profile from the query; this is in the
      # labeled JSONL as either parsed or raw
      route = route_decision(profile, c)
      query_routing[qid] = route.arm

Then compute, for each category (common, rare, pediatric, complex,
geographic, vague, treatment, existing, rare_explicit):
  - NDCG@5 of the rule arm on queries routed to rule
  - NDCG@5 of the agent arm on queries routed to agent
  - "Production" NDCG@5: mixture of rule-arm-on-rule-queries +
    agent-arm-on-agent-queries (this is what users would see)
  - Δ Production vs Rule-only baseline (with bootstrap 95% CI)
  - Routing rate (fraction routed to agent)
  - Mean cost/query (rule queries: ~$0.0017; agent queries: their
    actual measured cost from C2)
  - Mean latency/query

Write to data/evaluation/agent_vs_rule_v1.json with this schema:
  {
    "summary": {
      "n_queries": 85,
      "routing_rate": 0.XX,
      "production_ndcg_at_5": 0.XX,
      "production_ndcg_at_5_ci": [lo, hi],
      "delta_vs_baseline": 0.XX,
      "delta_ci": [lo, hi],
      "mean_cost_per_query_usd": 0.XXXX,
      "p95_latency_ms": XXXX
    },
    "per_category": { <cat>: {<same fields>} },
    "per_query": [ <85 entries> ]
  }

Also generate a markdown table summary suitable for pasting into
docs/evaluation-report.md §12 (Phase D).
```

### Acceptance criteria

- All 85 queries assigned to an arm (no nulls).
- Production NDCG@5 computed with bootstrap CI half-width < 0.08.
- Per-category breakdown for all 9 categories.
- JSON output well-formed and inspectable.

### Rollback

```
rm data/evaluation/agent_vs_rule_v1.json
```

---

## Phase C3b — Tune the three known issues + re-evaluate (~2 hr, ~$2-3)

> **When to run this phase.** C3's A/B output (`agent_vs_rule_v1.json`)
> showed the agent path **regressed** production NDCG@5 under the strict
> reading (−0.067 [−0.137, −0.007], CI excludes 0) and **lifted** it
> under the fallback reading (+0.048 [+0.024, +0.076], CI also excludes
> 0). The gap between strict and fallback is entirely explained by three
> known issues that surfaced during C2:
>
> 1. **The agent's degraded `result_dict["error"]` doesn't bubble up to
>    `state["error"]`** — so the A5 retry-as-rule edge that was supposed
>    to absorb agent failures never fires. The 18 of 85 queries that hit
>    `agent_did_not_terminate` returned 0 results to the user instead of
>    falling back to rule.
> 2. **No-terminator failure rate is too high on hard slices** — 50 %
>    of complex queries and 75 % of vague queries fail to call
>    `submit_final_results` within 5 cycles, because Haiku makes
>    parallel tool-call bursts (3 `get_trial_details` in one cycle) and
>    "5 cycles" doesn't equal "5 tool calls". A8's Rollback section
>    documented this; we never applied the fix.
> 3. **Per-query cost is 11× the rule baseline** because Anthropic
>    prompt caching isn't engaging — the ~2200-token system prompt is
>    just above the 1024-token cacheable-prefix floor but we don't pass
>    the `cache_control` field through `langchain_anthropic`.
>
> All three fixes are small and reversible. Together they should flip
> the C4 decision from HOLD → SHIP. **If the user explicitly skips this
> phase or the C3 numbers were already passing, jump to C4.**
>
> Predicted post-fix numbers (based on the C3 per-category breakdown):
> agent success rate **57.5 % → ~85 %**, strict NDCG@5 **0.725 → ~0.84**,
> mean cost per query **$0.0195 → ~$0.008**. If the actual numbers fall
> short of these projections by > 0.05 NDCG or > 1.5× cost, do NOT
> proceed to C4 — investigate which fix didn't engage and re-run C3b.4
> before spending the full C3b.5 budget.

---

## Step C3b.1 — Fix the degraded-result bubble in `execute_agent_search` (~15 min, $0)

**Goal:** When `AgenticSearchAgent.search()` returns a *degraded*
`result_dict` (with an embedded `error` field) instead of raising, the
LangGraph `execute_agent_search` node must elevate that error to
`state["error"]` so `_route_after_search` triggers the existing
retry-as-rule edge. This is a small fix to a Phase A5 design gap that
C2 surfaced.

### Why this is the highest-leverage fix

C2's 18 failed queries all followed the same path:

```
agent.search() → asyncio.wait_for caught no exception
              → degraded result_dict["error"]="agent_did_not_terminate"
              → execute_agent_search returned {"search_results": result_dict, ...}
              → state["error"] = None  (cleared by A5 success-path code)
              → _route_after_search saw error=None → routed to __end__
              → user got search_results.results=[] (0 trials)
```

The A5 retry-as-rule edge **exists** but never fires for this code path.
After the fix:

```
agent.search() → degraded result_dict["error"]="agent_did_not_terminate"
              → execute_agent_search detects inner error
              → returns {"error": "agent_arm_failed: agent_did_not_terminate", ...}
              → _route_after_search sees agent_arm_failed: prefix → routes to execute_search
              → rule arm produces 10 results
              → user gets the rule fallback
```

The "fallback" column of `agent_vs_rule_v1.json` is what production
NDCG@5 actually becomes after this fix.

### Prompt to send

```
Update src/TrialMine/agents/pipeline.py per Phase C3b.1 of
docs/build_agent.md. In execute_agent_search, AFTER agent.search()
returns successfully (i.e. the try/except didn't catch), detect a
degraded result_dict and elevate the inner error to state["error"]
using the same `agent_arm_failed:` prefix the exception branch
uses — that way the existing _route_after_search edge to
execute_search fires unchanged.

Concretely, replace this success-return block at the end of
execute_agent_search:

    return {
        "search_results": result_dict,
        "agent_trace": trace_entries,
        "error": None,
    }

with:

    inner_error = (result_dict or {}).get("error")
    if inner_error:
        # The agent ran without raising but returned a degraded
        # result_dict (timeout / no-terminator / parse failure
        # inside agent.search()). The A4.2 contract stashes these
        # in result_dict["error"] rather than raising. Mirror the
        # exception-path shape so _route_after_search routes to
        # execute_search for the retry-as-rule fallback.
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "execute_agent_search degraded (%s); routing to "
            "retry-as-rule",
            inner_error,
        )
        record_agent_failure("execute_agent_search_degraded")
        return {
            "error": f"agent_arm_failed: {inner_error}",
            "agent_trace": trace_entries + [
                {
                    "step": "execute_agent_search_degraded",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {"inner_error": inner_error},
                }
            ],
        }
    return {
        "search_results": result_dict,
        "agent_trace": trace_entries,
        "error": None,
    }

CRITICAL gotcha #1 — the bubble-fix return path MUST NOT write
``search_results``. The retry (execute_search) will overwrite it
anyway via LangGraph's state-merge semantics, but leaving the
agent's empty result_dict on state pre-retry would be confusing
for any future node that reads state mid-run. Explicitly omitting
search_results in the bubble-fix branch keeps the contract clean.

CRITICAL gotcha #2 — DO NOT change the error-string format. The
existing _route_after_search check is:

    if error.startswith("agent_arm_failed:"):
        return "execute_search"

So the prefix `agent_arm_failed:` is load-bearing — without it, the
retry edge won't fire and the fix is a no-op.

CRITICAL gotcha #3 — match the trace_entries pattern. The existing
exception branch appends a single `execute_agent_search_error`
entry to trace_entries. The bubble-fix branch should append
`execute_agent_search_degraded` (NOT `_error` — different step
name lets Grafana/dashboards distinguish "agent raised" from
"agent gave up cleanly"). The trace_entries from the agent itself
(which document the iter/tool sequence) are preserved by
concatenating.

Add an integration test in tests/integration/test_route_node.py:

  async def test_degraded_result_dict_routes_to_rule_retry(...)
      # Mock AgenticSearchAgent so it returns a degraded result
      # (results=[], error="agent_did_not_terminate") and ALSO
      # mock execute_search to return a synthetic rule result.
      # Verify:
      #   - state["error"] flips to "agent_arm_failed:..."
      #   - execute_search ran (the retry fired)
      #   - final response has rule-arm results, not empty
      #   - agent_trace contains BOTH the agent's no-terminator
      #     entries AND the rule arm's execute_search entry

Show me the diff.
```

### Verify

```
# 1. Unit + integration tests
python -m pytest tests/integration/test_route_node.py -v

# 2. End-to-end: pick a previously-failing qid and run a single
#    eval. With the fix, the agent's no-terminator should bubble
#    up → retry as rule → 10 rule-arm rows written.
OMP_NUM_THREADS=1 python scripts/eval_agent_path.py \
    --labels data/evaluation/full_labeled_dataset.jsonl \
    --output /tmp/c3b1_smoke.jsonl \
    --limit 1
# Pick a query that failed in C2 (qid=17 is the easiest — "chronic
# lymphocytic leukemia ibrutinib failure"). Use --limit 1 and run
# it against the FAILED qid by editing the script's qid filter
# temporarily, OR run on all 18 missing qids by re-running with
# --resume against the existing output file.

# 3. Inspect /tmp/c3b1_smoke.jsonl — should have 10 rows for qid=17
#    with relevance + reason populated AND `agent_iters_used` should
#    show the agent's failed run (not 0).
```

### Acceptance criteria

- Integration test passes (6 existing + 1 new degraded-result test).
- Smoke writes 10 rows for the previously-failing qid 17.
- The 10 rows have rule-arm provenance (look at `agent_iters_used` —
  it should reflect the agent's last attempt, but the trials should
  be rule-arm picks; `agent_error` field carries the inner error).
- No regression on `scripts/test_pipeline.py` (5 rule queries still
  succeed end-to-end with no degraded results).

### Rollback

```
git diff src/TrialMine/agents/pipeline.py | git apply -R
```

If the fix introduced a new failure mode (e.g., every agent run is
now classified as degraded), the symptom is: A8 smoke shows all 5
queries running through `execute_agent_search_degraded` even when
they should succeed. Common cause: a benign `result_dict.error`
field set on the success path that we shouldn't be acting on. The
fix: check the agent's success path for any non-None error writes —
the A4.2 contract says the error field MUST be None on success.

---

## Step C3b.2 — Strengthen prompt + bump `agentic_max_iters` to 6 (~25 min, $0)

**Goal:** Reduce the no-terminator failure rate from 42 % (18 of 85
agent runs in C2) to ≤ 15 %. **Two coupled changes** — neither alone
solves the problem, both together do.

### Why C2's data demands BOTH changes, not just one

Looking at C2's failed queries: the no-terminator rate split by tool
call count tells the story.

- **Failures with 5-7 tool calls** (12 of 18): the agent reached cycle
  5 but those cycles included parallel-tool bursts (3 `get_trial_details`
  in one cycle counts as 3 tool calls but 1 cycle). The prompt's "5
  cycles" wording was clear, but Haiku interpreted parallel calls as
  "free" — and ran out of cycles before submitting. A pure prompt fix
  ("submit by cycle 5") helps here.

- **Failures with 8+ tool calls** (6 of 18): the agent was actively
  exploring — 4 details + 3 eligibility checks + 1 lookup spread across
  5 cycles. Even with a prompt directive, the agent reasonably needs
  one more cycle to wrap up. A `max_iters=6` bump gives the buffer.

Applying #2 alone (bump only): doesn't discipline the prompt — Haiku
uses the extra cycle for more exploration, not for submitting.

Applying #1 alone (prompt only): doesn't help the legitimately-busy
queries that need the extra cycle.

Together: prompt discipline + 1-cycle buffer = ≥ 85 % success rate.

### The exact prompt change

In `src/TrialMine/agents/react_agent.py`, inside `AGENT_SYSTEM_PROMPT`,
find the HARD CONSTRAINTS section. Locate this block:

```
* MAXIMUM 5 tool-use cycles total. Count before each call. By cycle 4
  you should be reading details + checking eligibility; cycle 5 must
  be `submit_final_results`.
```

Replace with:

```
* MAXIMUM 6 tool-use cycles total. Count before each call.
* TERMINATION DEADLINE: by cycle ≥ 5 you MUST be calling
  submit_final_results, even if eligibility checks are incomplete.
  A partial-but-submitted result (3–5 trials) is ALWAYS better than
  no result. Do NOT continue exploring past cycle 5 — the loop will
  hit its budget cap and your work will be DISCARDED entirely.
* If you have already issued ≥ 7 tool calls (regardless of cycle
  number — counting EACH tool in a parallel batch), prefer wrapping
  up over launching another batch. Parallel batches consume the
  cycle budget the same as sequential calls.
```

### The config change

In `src/TrialMine/config.py`, change the default of `agentic_max_iters`:

```python
agentic_max_iters: int = Field(
    default=6,  # was 5; bumped per Phase C3b.2 to absorb parallel-
                # tool-call bursts of 3+ tools in one cycle (see C2
                # failure analysis). The recursion_limit calculation
                # `max_iters * 2 + 2` in react_agent.py scales
                # automatically.
    ge=1,
    le=10,
    ...
)
```

### CRITICAL gotchas

**1. `recursion_limit` scales but verify.** `react_agent.py:search()`
computes `recursion_limit = max_iters * 2 + 2`. With max_iters=6 that's
14 — well under LangGraph's hard cap (25 by default). Verify this is
the actual value used at runtime:

```
python -c "
from TrialMine.config import DegradationConfig
c = DegradationConfig()
print(f'max_iters={c.agentic_max_iters}, '
      f'recursion_limit={c.agentic_max_iters * 2 + 2}')
"
# Expect: max_iters=6, recursion_limit=14
```

**2. The Prometheus iteration histogram buckets are (1, 2, 3, 4, 5, 6, 7, 8).**
With max_iters=6, the `outcome="submitted"` distribution can still land
in any bucket. Mean iters should hover at 3.5–4.5 (i.e., the budget
bump is precaution, not the new normal). Watch for mean > 5 in the A8
smoke — that's a sign the prompt change made the agent treat the new
cap as the target.

**3. DO NOT also bump `agentic_per_iter_timeout_s`.** A8 set this to
12.0s for the smoke environment; production default is 6.0s. The
total budget = `max_iters × per_iter_timeout_s` = 6 × 6 = 36s. The
outer `skip_agent_if_slow_s` should accommodate this; if you bumped
it to 90s in the smoke, leave it. In production with prompt caching
(C3b.3) the per-cycle latency drops to 1-2s anyway, making the per-iter
cap mostly cosmetic.

**4. The system prompt token count goes UP, not down.** The new HARD
CONSTRAINTS block is ~3 additional sentences (~100 tokens). Total prompt
size grows from ~2200 → ~2300 tokens. Still comfortably above the
1024-token Haiku cache floor. **No impact on caching** (C3b.3).

### Verify

```
# 1. Tests still pass
python -m pytest tests/unit/test_query_router.py \
                 tests/unit/test_degradation.py \
                 tests/integration/test_route_node.py -v

# 2. A8 smoke re-run — 5 hand-picked queries, focus on
#    no-terminator rate
OMP_NUM_THREADS=1 python scripts/test_agent_path.py
# Expect:
#   - ≥ 4 of 5 agent queries reach submit_final_results
#   - Mean iters (over successful runs) in [3.0, 4.5]
#   - Total cost ≤ $0.20 (lower than A8's $0.14 because cache
#     should help — but cache is C3b.3, so don't be alarmed if
#     it's still ~$0.20)

# 3. Sanity-check prompt size
python -c "
from TrialMine.agents.react_agent import AGENT_SYSTEM_PROMPT
n = len(AGENT_SYSTEM_PROMPT)
print(f'chars={n} (~{n//4} tokens, target 2200-2500)')
"
```

### Acceptance criteria

- A8 smoke: ≥ 4 of 5 agent queries call `submit_final_results`
  (was 3 of 4 in A8 with max_iters=5).
- Mean iters (submitted runs) in [3.0, 4.5]. If > 5, the bump
  became the target — roll back and try a more aggressive prompt
  directive ("you must submit by cycle 4").
- Pre-existing rule-arm latency / NDCG unchanged.

### Rollback

```
git checkout src/TrialMine/agents/react_agent.py src/TrialMine/config.py
```

Both files revert. If only one of the two changes was problematic,
revert that one selectively — but the design assumes both ship
together.

---

## Step C3b.3 — Enable Anthropic prompt caching on `AGENT_SYSTEM_PROMPT` (~25 min, $0)

**Goal:** Drop the per-agent-query input cost from ~$0.035 to ~$0.005
by enabling Anthropic prompt caching on the ~2300-token system prompt.
**This is the smallest fix and the highest cost-leverage one — but it
has the trickiest verification path because `langchain_anthropic`'s
caching API is a known fragile area.**

### Why caching cuts cost so much for THIS loop specifically

Anthropic Haiku 4.5 input rate is $1/M tokens. Each ReAct cycle re-sends:
- The system prompt (~2300 tokens after C3b.2)
- The 5 tool definitions (~500 tokens)
- The growing message history (~500-2000 tokens by cycle 5)

So by cycle 5, a single query has consumed ~12-15K input tokens, of
which ~10K is the system prompt repeated 5 times. With caching, the
prompt prefix bills at the cached rate ($0.10/M for cache reads) instead
of the full rate ($1/M). Net per-query saving: ~$0.025–$0.035 → ~$0.005.

This is the cost projection's biggest unknown. **If caching doesn't
engage, the C3b.5 budget overshoots by ~$1.50.** Verify carefully.

### The known-fragile part

The raw Anthropic SDK accepts `cache_control` directly inside the
`system` field (see `query_parser.py:248` for the production pattern):

```python
system=[
    {"type": "text", "text": SYSTEM_PROMPT,
     "cache_control": {"type": "ephemeral"}},
],
```

But we're using `langchain_anthropic.ChatAnthropic` via
`langgraph.prebuilt.create_react_agent(prompt=AGENT_SYSTEM_PROMPT)`,
which doesn't pass arbitrary kwargs through to the SDK's `system` field.
Three paths to try, in order of preference:

**Path (a) — Pass a `SystemMessage` with `additional_kwargs`** (cleanest
if it works in the installed langchain_anthropic version):

```python
from langchain_core.messages import SystemMessage

self._agent = create_react_agent(
    self._llm,
    tools=self._tools,
    prompt=SystemMessage(
        content=AGENT_SYSTEM_PROMPT,
        additional_kwargs={
            "cache_control": {"type": "ephemeral"},
        },
    ),
)
```

**Path (b) — Use `langchain_anthropic`'s `model_kwargs` to set
`extra_headers`** (forces the `anthropic-beta` header):

```python
self._llm = ChatAnthropic(
    model=model,
    anthropic_api_key=api_key,
    temperature=0.0,
    max_tokens=2048,
    timeout=max(15.0, max_iters * per_iter_timeout_s),
    model_kwargs={
        "extra_headers": {
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
    },
)
```
(Still need cache_control somewhere — path (b) just enables the header;
the system prompt still needs the cache_control annotation.)

**Path (c) — Drop langchain_anthropic, call the raw Anthropic SDK
inline in react_agent.py** (most robust, most code; defer to a
follow-up if (a) and (b) both fail).

### CRITICAL — verification is mandatory

Cache headers are only useful if they FIRE. After the change, run a
single multi-cycle agent query and inspect `cache_read_input_tokens`
on the SECOND LLM call within the loop.

- First call: cache MISS (creation) — `cache_creation_input_tokens`
  ≈ 2300, `cache_read_input_tokens` = 0.
- Subsequent calls in the same conversation: cache HIT —
  `cache_read_input_tokens` ≈ 2300, `cache_creation_input_tokens` = 0.

If `cache_read_input_tokens` stays at 0 across iterations, caching
**did not engage**. This means path (a) failed silently — langchain
swallowed the cache_control field. Move to path (b) or (c).

### CRITICAL — the cost calculation in metrics.py needs updating

The A6 `_walk_messages` in `react_agent.py` sums `usage.input_tokens`
and `usage.output_tokens`. With caching enabled, `input_tokens` STILL
includes cache reads (Anthropic charges $0.10/M for them). The total
cost calculation in `monitoring/metrics.py:record_agent_cost()` needs
to apply the discounted rate to the cached portion:

```python
def record_agent_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,    # NEW
    cache_creation_tokens: int = 0,  # NEW
) -> None:
    ...
    in_rate, out_rate = rates
    # Cache reads bill at 10% of the input rate (Haiku 4.5 spec).
    # Cache creation bills at 125% of the input rate.
    cache_read_rate = in_rate * 0.10
    cache_creation_rate = in_rate * 1.25
    # input_tokens includes cache reads; subtract them out so
    # we don't double-charge.
    uncached_input = max(0, input_tokens - cache_read_tokens
                                       - cache_creation_tokens)
    cost = (uncached_input * in_rate +
            cache_read_tokens * cache_read_rate +
            cache_creation_tokens * cache_creation_rate +
            output_tokens * out_rate)
    AGENT_COST_USD.labels(model=model).inc(cost)
```

And update `react_agent.py:_walk_messages()` to also extract:
```python
cache_read = int(usage.get("cache_read_input_tokens") or 0)
cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
```

And pass these to `record_agent_cost()` at termination.

### Prompt to send

```
Apply Phase C3b.3 per docs/build_agent.md. Three coupled changes:

1. In src/TrialMine/agents/react_agent.py, wrap AGENT_SYSTEM_PROMPT
   in a SystemMessage with cache_control set (path (a) from the
   runbook). If that doesn't engage at verify time, fall back to
   path (b) — extra_headers via model_kwargs.

2. In src/TrialMine/agents/react_agent.py:_walk_messages(), extract
   cache_read_input_tokens and cache_creation_input_tokens from the
   AIMessage usage_metadata (or response_metadata.usage fallback).
   Include them in the (input_tokens_total, output_tokens_total)
   tuple's adjacent fields. Update the agent_submit trace entry's
   decisions dict to include `cache_read_input_tokens` and
   `cache_creation_input_tokens`.

3. In src/TrialMine/monitoring/metrics.py:record_agent_cost(), add
   optional cache_read_tokens and cache_creation_tokens kwargs and
   apply the discounted rates (cache reads at 10% of input rate,
   cache creation at 125%). Call sites in react_agent.py need to
   pass them through.

Show me all three diffs.
```

### Verify

```
# 1. Tests still pass
python -m pytest tests/unit/test_trace_store.py -v

# 2. Single-query cache probe — most important check
OMP_NUM_THREADS=1 python -c "
import asyncio, json
import TrialMine.config as cfg_mod
cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
    agentic_path_enabled=True,
    agentic_per_iter_timeout_s=12.0,
    skip_agent_if_slow_s=90.0,
)
import TrialMine.agents.pipeline as pl
pl._agent_singleton = None
from TrialMine.agents.react_agent import AgenticSearchAgent
from TrialMine.agents.query_parser import PatientProfile
agent = AgenticSearchAgent()
profile = PatientProfile(
    raw_query='BRCA1 ovarian cancer PARP inhibitor failure',
    condition='ovarian cancer',
    biomarkers=['BRCA1'],
    prior_treatments=['PARP inhibitor'],
)
result, trace = asyncio.run(agent.search(profile))
submit = next(e for e in trace if e.get('step') == 'agent_submit')
d = submit['decisions']
print('input_tokens:', d.get('input_tokens'))
print('output_tokens:', d.get('output_tokens'))
print('cache_read:', d.get('cache_read_input_tokens'))
print('cache_creation:', d.get('cache_creation_input_tokens'))
"

# CRITICAL — expect cache_read > 1000 (the system prompt size)
# on iter ≥ 2. If cache_read is 0, caching DID NOT engage.
```

If cache_read is 0, debug before proceeding:
```
# (a) Verify the model_kwargs / extra_headers are reaching the SDK:
python -c "
from TrialMine.agents.react_agent import AgenticSearchAgent
agent = AgenticSearchAgent()
print('llm config:', agent._llm.model_kwargs)
"
# Expect: anthropic-beta header present.

# (b) If the header is there but cache still 0, the cache_control
#     field isn't propagating through SystemMessage. Switch to
#     path (b) or (c).
```

### Acceptance criteria

- After warm-up (1 successful agent query), the NEXT query's
  `cache_read_input_tokens` ≥ 1800 on iter ≥ 2.
- The 5-query A8 smoke's cumulative cost drops to ~$0.04 (from
  ~$0.14 pre-fix). If still ~$0.10+, caching didn't engage on
  iter ≥ 2 of multi-cycle queries.
- `agent_submit` trace entries contain the new `cache_read_input_tokens`
  + `cache_creation_input_tokens` fields.
- record_agent_cost test (if any exists) still passes.

### Rollback

```
git checkout src/TrialMine/agents/react_agent.py \
              src/TrialMine/monitoring/metrics.py
```

Both files revert. If only the metrics update broke (rare), revert
metrics.py alone — the react_agent caching change is independent.

---

## Step C3b.4 — Sanity-test the fixes on the 18 previously-failed queries (~15 min, ~$0.40)

**Goal:** Before committing to the full $2-3 re-eval in C3b.5, verify
the three fixes actually moved the no-terminator rate on the queries
they were designed to fix. **This is a gate — if recovery rate is
< 78 %, do NOT proceed to C3b.5.**

### Identifying the 18 queries (extract from C2 outputs)

The list is deterministic from C2's output files. Run this to extract:

```bash
python -c "
import json
all_qids = set()
for path in ['data/evaluation/full_labeled_dataset.jsonl',
             'data/evaluation/full_labeled_dataset_expansion_v2.jsonl']:
    with open(path) as f:
        all_qids |= {json.loads(l)['query_id'] for l in f}
done_qids = set()
for path in ['data/evaluation/full_labeled_dataset_agent_all_65q.jsonl',
             'data/evaluation/full_labeled_dataset_agent_all_20q.jsonl']:
    with open(path) as f:
        done_qids |= {json.loads(l)['query_id'] for l in f}
missing = sorted(all_qids - done_qids)
print(f'{len(missing)} missing qids:', missing)
"
# Expect: 18 missing qids: [17, 413, 415, 417, 420, 421, 424, 425,
#                            602, 603, 604, 606, 610, 611, 613, 615,
#                            616, 619]
```

### Why these specific 18, not a random sample

The 18 are NOT a random subset — they're the WORST cases (the queries
where the agent burned its budget without submitting). If the fixes
work on these, the success rate on the easier queries (which already
worked in C2) will only improve. **Recovering ≥ 14 of 18 is high
signal that the full re-eval will succeed.**

If recovery is < 14:
- The prompt change didn't discipline the agent enough — try a more
  aggressive directive ("you MUST submit by cycle 4 — cycle 5 is the
  budget cap, NOT the target").
- OR the bubble fix isn't routing — verify with the per-query trace
  that `execute_agent_search_degraded` is firing.
- OR `max_iters=6` wasn't picked up — verify `get_default_degradation()`.

### Prompt to send

```
Build scripts/c3b_sanity.py per Phase C3b.4 of docs/build_agent.md.

The script:
1. Loads the 18 missing qids by diffing the C2 output files against
   the input label files (see the extraction snippet in the runbook).
2. Reuses scripts/eval_agent_path.py's machinery (force agent,
   patches AB router, etc.) but operates ONLY on those 18 qids
   (filter the queries list before the main loop).
3. Writes to /tmp/c3b_sanity_65q.jsonl and /tmp/c3b_sanity_20q.jsonl
   (separate files for the two source label sets — easier to
   diagnose if the recovery rate splits 65q-better vs 20q-better).
4. At the end, prints:
   - newly-successful qids (which ones recovered)
   - mean iters / cost / latency for the recovered queries
   - no-terminator rate (= 1 − recovery_rate)
   - explicit diagnostic for the remaining failures: which step
     (route_decision / execute_agent_search / agent_submit) was
     last in the trace before failure

CLI: --limit N for testing on the first N of 18.

Show me the script.
```

### Verify

```
OMP_NUM_THREADS=1 python scripts/c3b_sanity.py
```

Watch the per-query output for two things:
1. Trace shows `execute_agent_search_degraded` followed by
   `execute_search` for the no-terminator path (proves C3b.1 fixed).
2. Mean iters in the [3.0, 4.5] window (proves C3b.2 didn't make
   the agent treat 6 as the new target).
3. `cache_read_input_tokens` > 0 on iter ≥ 2 of multi-cycle queries
   (proves C3b.3 engaged).

### Acceptance criteria

| Signal | Acceptable | Yellow flag | Red flag (do NOT proceed) |
|---|---|---|---|
| Recovery rate (qids with ≥ 1 row) | ≥ 78 % (14 / 18) | 50–78 % | < 50 % |
| Mean iters of recovered queries | 3.0 – 4.5 | 4.5 – 5.5 | > 5.5 (cycle cap is new target) |
| Cumulative cost | ≤ $0.80 | $0.80 – $1.50 | > $1.50 (caching didn't engage) |
| `cache_read_input_tokens` > 0 on iter ≥ 2 | yes on most queries | yes on some | always 0 (caching dead) |

### Decision branch

- **Green on all 4 signals** → proceed to C3b.5.
- **Recovery rate 50–78 %** → diagnose the residual failures. If
  they're concentrated in one slice (e.g., all 4 remaining failures
  are vague queries), it's a slice-specific issue and worth
  documenting; you can still proceed to C3b.5 but flag in C4.
- **Red on any signal** → roll back C3b.1–3 selectively (figure out
  which fix didn't engage) and retry C3b.4. Do NOT spend the C3b.5
  budget until C3b.4 is green.

### Rollback

```
rm /tmp/c3b_sanity_*.jsonl
git diff src/TrialMine/agents/ src/TrialMine/config.py | git apply -R
```

The original C2 outputs are untouched (we wrote to /tmp/), so even
on full rollback the data files are preserved.

---

## Step C3b.5 — Full C2 + C3 re-run + before/after comparison (~1 hr, ~$2)

**Goal:** Re-run C2 against all 85 queries (with C3b.1–3 fixes in
place), then re-run C3 to produce `agent_vs_rule_v2.json`. Build a
side-by-side comparison vs C3's v1 output so the C4 decision gate
has concrete before/after numbers.

### Prerequisites — DO NOT skip

C3b.4 must have passed all four acceptance signals before running
C3b.5. The ~$2 spend is not recoverable; if the fixes aren't
landing, you'll burn the budget on a re-run that produces the same
disappointing v2 numbers as v1.

### Prompt to send

```
Run Phase C3b.5 of docs/build_agent.md:

1. Run C2 against the 65q label file with the fixes from C3b.1–3
   in place (no --resume — fresh output):

     OMP_NUM_THREADS=1 python scripts/eval_agent_path.py \
         --labels data/evaluation/full_labeled_dataset.jsonl \
         --output data/evaluation/full_labeled_dataset_agent_all_v2_65q.jsonl

2. Run C2 against the 20q expansion:

     OMP_NUM_THREADS=1 python scripts/eval_agent_path.py \
         --labels data/evaluation/full_labeled_dataset_expansion_v2.jsonl \
         --output data/evaluation/full_labeled_dataset_agent_all_v2_20q.jsonl

3. Re-run C3 against the NEW outputs (note --agent-65q / --agent-20q
   are now the _v2 files):

     python scripts/c3_compare.py \
         --agent-65q data/evaluation/full_labeled_dataset_agent_all_v2_65q.jsonl \
         --agent-20q data/evaluation/full_labeled_dataset_agent_all_v2_20q.jsonl \
         --output data/evaluation/agent_vs_rule_v2.json \
         --markdown data/evaluation/agent_vs_rule_v2.md

4. Build a side-by-side comparison table. Create
   scripts/c3b_before_after.py that loads v1 and v2 JSONs, prints
   per-category before/after deltas, and writes
   data/evaluation/c3b_before_after.md with:
   - overall NDCG@5 v1 → v2 (strict + fallback)
   - per-category routing rate v1 vs v2
   - per-category agent success rate v1 vs v2
   - per-category fallback NDCG@5 v1 vs v2
   - per-query cost v1 vs v2 (with cache_read tokens distribution)
   - which qids recovered (failed v1, succeeded v2) — explicit list
   - which qids regressed (succeeded v1, failed v2) — explicit list;
     should be 0 or near-0, but call them out

Show me both the v2 JSON's `summary` block AND the c3b_before_after.md
table. Format it for direct paste into C4's decision-gate review.
```

### Verify

```
# Confirm v2 outputs exist + non-empty
ls -la data/evaluation/*v2*.jsonl data/evaluation/agent_vs_rule_v2.{json,md} \
       data/evaluation/c3b_before_after.md
```

### Acceptance criteria (for C4's decision-gate input)

| Signal | v1 (C3) result | C3b.5 target | C4 decision impact |
|---|---|---|---|
| Agent success rate | 57.5 % | ≥ 80 % | < 70 % → HOLD; ≥ 80 % → SHIP-eligible |
| Strict production NDCG@5 vs baseline 0.792 | 0.725 (Δ −0.067) | ≥ 0.79 (Δ ≥ −0.005) | Δ < −0.02 → HOLD |
| Fallback production NDCG@5 vs baseline | 0.840 (Δ +0.048, CI excl. 0) | ≥ 0.84 (Δ ≥ +0.05) | Δ CI overlapping 0 → TUNE |
| Complex slice fallback NDCG@5 lift | +0.177 | ≥ +0.10 | < +0.05 → HOLD (the slice the agent was built for) |
| Mean cost / query | $0.0195 | ≤ $0.012 | > $0.020 → HOLD (cost too high) |
| p95 latency | 60 s | ≤ 30 s | > 60 s → HOLD (no improvement) |

### Decision branch (input to C4)

- **All 6 criteria pass** → C4 SHIPS with high confidence.
- **5 of 6 pass** → C4 ships with a documented caveat about the
  failing one.
- **3–4 of 6 pass** → C4 verdict TUNE: one more C3b cycle targeting
  whichever criterion failed.
- **≤ 2 of 6 pass** → C4 verdict HOLD: the three fixes weren't
  enough. Document the remaining work in `docs/things_can_be_fixed.md`
  and run Phase D2 (revert path) — the agent stays default-OFF.

### Rollback

The v2 eval data is additive — it doesn't replace v1 files. To clear
the C3b artifacts:

```
rm data/evaluation/full_labeled_dataset_agent_all_v2_*.jsonl
rm data/evaluation/agent_vs_rule_v2.{json,md}
rm data/evaluation/c3b_before_after.md
```

The C3b.1–3 code fixes are separate — those stay in until you
explicitly roll them back via `git checkout`. If C3b.5's verdict was
HOLD, the recommendation is: keep the code fixes (they're improvements
regardless), keep `agentic_path_enabled=False` in production, and
document what to try next.

---

## Step C4 — Decision gate (manual review, no code)

**Goal:** Read the C3 output and make the ship/tune/hold/revert decision.

### What to look at

Open `data/evaluation/agent_vs_rule_v1.json` and inspect:

| Signal | What we want | What it means |
|---|---|---|
| Production NDCG@5 vs baseline | ≥ baseline, CI overlapping or above | No regression on the headline |
| Complex slice agent-arm NDCG@5 | ≥ 0.45 (was ~0.34 in rule arm) | Real lift on the targeted slice |
| Complex slice CI for the delta | excludes 0 | Statistically detectable lift |
| Vague slice agent-arm NDCG@5 | ≥ rule-arm by ≥ +0.03 | Modest lift expected |
| Routing rate | 15-35 % | Sane heuristic |
| Mean iters used | 3-5 | Loop is converging, not capping |
| p95 latency | ≤ 12 s | Within budget |
| Mean cost/query | ≤ $0.005 | At 25 % routing, cost premium ~3-4× |
| Tool-call success rate | ≥ 95 % | Tools are well-defined |

### Decision matrix

| Picture | Decision |
|---|---|
| Production NDCG ≥ baseline AND complex slice lift CI excludes 0 AND cost within projected envelope | **SHIP** (Phase D1) |
| Production NDCG ≥ baseline BUT complex slice lift is tied (CI overlaps 0) | **TUNE** — try lowering routing threshold; re-run C2/C3 |
| Production NDCG regressed by > 0.02 OR cost > 2× projected OR iter mean > 5.5 | **HOLD** — diagnose; consider Sonnet swap or prompt revision |
| Multiple signals bad AND no clear fix | **REVERT** (Phase D2) — keep code, leave toggle OFF |

### Output of this step

Write the decision + rationale into a short note that becomes the basis for Phase D's CLAUDE.md entry. Format:

```
Decision: SHIP | TUNE | HOLD | REVERT
Signals reviewed (paste table):
  - Production NDCG@5: X (Δ vs baseline +Y, CI [lo, hi])
  - Complex slice: agent X vs rule Y (Δ +Z, CI [lo, hi])
  - Routing rate: X%
  - Mean iters: X
  - p95 latency: Xms
  - Cost/query: $X
  - Tool success rate: X%
Why this decision: <one paragraph>
What we'd watch in production: <one paragraph>
```

---

# Phase D — Ship or Revert (~1 hr)

## Step D1 — SHIP path (if Phase C4 decision was SHIP)

### Prompt to send

```
Ship the agent path per Phase D1 of docs/build_agent.md.

1. Flip the production default in src/TrialMine/config.py:
   - DegradationConfig.agentic_path_enabled: bool = True
     (was False through Phases A-C)
   - DegradationConfig.skip_agent_if_slow_s: float = 25.0
     (was 10.0). The agent loop's realistic budget is ~15-20s
     (5 iters with 4s CE reranks inside search_trials); the old
     10s outer cap would cancel ~half of agent runs before they
     complete. The rule path is unaffected because the rule
     orchestrator's hot path is unchanged and finishes well under
     10s anyway — the outer cap is a safety net, not the typical
     run time.

2. Enable the experiment in ABTestRouter:
   - In src/TrialMine/experiments/ab_test.py, get_default_router(),
     change the agent_path_v1 Experiment(enabled=False) to True.

3. Update docs/evaluation-report.md — add §12 "Phase 14 — Agentic
   search path" containing:
   - Motivation (62.5% complex-slice misses are rerank-bound; model
     retraining hasn't moved this; iterative refinement is the next
     intervention candidate)
   - Architecture (Haiku 4.5 ReAct loop on ALL_TOOLS + submit_final_results
     terminator; LangGraph conditional routing; AB router gating;
     SQLite trace store + Grafana panels)
   - Methodology (force-all eval in C2, production-routing eval in C3,
     holistic decision gate in C4 mirroring Decision 36 framing)
   - Headline tables from data/evaluation/agent_vs_rule_v1.json
   - Honest "what this fixed" (complex slice lift, observability) and
     "what this didn't fix" (rule arm still catches most queries; agent
     adds latency and cost; tied on easy slices by design)

4. Update CLAUDE.md:
   - Bump the "Phase" line at the top of "Current State" to "Phase 14"
     with a one-paragraph summary of the agent ship
   - Add Decision 43: "Agent path with Haiku 4.5 ReAct loop ships
     behind 50/50 AB router for queries matching sparse-profile or
     complex-pattern heuristics. Why: ..., How to apply: ...,
     What this does NOT solve: ..." — same structure as Decisions 36 / 39 / 42
   - Add a "What's working" entry for the agent path
   - Move "agent path" from "What's next" / planned to shipped

5. Commit on a feature branch with 3 logical commits:
   (a) Code (config flip + AB router flip)
   (b) Eval artefacts (data/evaluation/agent_vs_rule_v1.json + any
       Haiku labels from C2)
   (c) Docs (evaluation-report.md + CLAUDE.md)

6. Open a PR titled "feat(agent): ship Haiku ReAct agent path for
   hard queries (+<delta> NDCG@5 on complex slice)" with the C4
   decision-gate signals in the description.

Show me each diff before committing.
```

### Acceptance criteria

- `agentic_path_enabled` default is `True`.
- `agent_path_v1` experiment is `enabled=True`.
- `docs/evaluation-report.md` §12 written.
- `CLAUDE.md` Phase line bumped + Decision 43 added.
- 3 commits on a feature branch.
- PR opened.

---

## Step D2 — REVERT path (if Phase C4 decision was REVERT)

### Prompt to send

```
Revert (hold at default-OFF) per Phase D2 of docs/build_agent.md.

1. Leave src/TrialMine/config.py as-is (default still OFF — production
   was never affected).

2. Leave src/TrialMine/experiments/ab_test.py as-is (experiment still
   disabled).

3. Write a "lessons learned" entry in docs/things_can_be_fixed.md:
   - What was tried (Haiku ReAct agent, 50% AB gate, sparse+complex
     routing heuristics)
   - What didn't lift (specific metrics from C3 JSON)
   - What we'd try next (Sonnet swap? different routing heuristic?
     different tool surface? give up on this lever and try multi-vector
     encoding from Issue #4?)

4. Update CLAUDE.md:
   - Add Decision 43 documenting the HOLD: "Agent path built but
     held at default-OFF. Why: <holistic signal summary from C4>.
     Re-engagement = ..."
   - Note in "What's next" that the agent code is preserved in the
     repo and can be enabled by one config flip + AB router flip
     (cite the file:line refs from the build).

5. The code stays in the repo. We do NOT git revert. Same precedent
   as Decision 41 (UMLS) was a full revert because that touched
   production; this path was always default-OFF so nothing to revert,
   just don't flip the flags. The repo carrying the inert agent
   infrastructure is acceptable here because (a) it's behind a
   working AB router with a public-facing experiment shape that's
   useful regardless, (b) the trace observability from Phase B is
   load-bearing for the rule arm too and stays valuable independently.

   The alternative (full git revert of the agent code) is overkill
   given the small inert-code footprint and the high re-engagement
   value.

Show me each diff.
```

### Acceptance criteria

- Production behaviour byte-identical to pre-Phase-A (since the toggle never flipped).
- Lessons-learned entry written.
- Decision 43 documents the HOLD with re-engagement instructions.

---

# Appendix — Failure modes to watch for

| Symptom | Likely cause | Fix |
|---|---|---|
| `A3` unit tests fail with "agent_path_enabled invariant violated" | Routing logic doesn't short-circuit on `False` | Re-check `route_decision`: the FIRST rule must be the disabled-path short-circuit |
| `A4` smoke loops past 5 iters | Prompt isn't pushing `submit_final_results` hard enough | Add stronger language; consider an explicit step-counter sentence in the prompt |
| `A4` smoke: `submit_final_results` never called | Haiku's tool selection drifting | Move `submit_final_results` to be the most-prominent tool description in the system prompt; add explicit "you must end your turn by calling submit_final_results" |
| `A4` smoke: agent loops past `submit_final_results` (calls it but keeps going) | The tool was added WITHOUT `return_direct=True`, OR submit_final_results was called in the same turn as another tool | (1) Verify `submit_final_results.return_direct is True` in tools.py. (2) Strengthen prompt: submit_final_results MUST be called alone — `return_direct` only terminates the loop when ALL tool_calls in a single AIMessage are return_direct |
| `A4` smoke: ChatAnthropic raises "model" not recognized | langchain-anthropic older version doesn't accept the bare alias | Use the canonical kwarg name `model_name="claude-haiku-4-5"` instead of `model=` |
| `A5` `prompt=` kwarg rejected by create_react_agent | langgraph < 0.2 used `state_modifier=` | Upgrade `langgraph>=0.2` (pyproject pins `>=0.1`, may need bumping) or use `state_modifier=`; verify against the installed version |
| `A5` pipeline compile error: cycle detected | Adding `"execute_search": "execute_search"` in the conditional dict — but if execute_search itself can also retry, we get an unbounded cycle | Bound the retry: add a counter to SearchState, cap at 1 retry |
| `A8` cost > $1 for 5 queries | Agent is calling tools with massive prompts (e.g., dumping full eligibility text into every iteration's context) | Truncate tool outputs more aggressively in `tools.py:_serialize_search_result` and friends |
| `A8` every agent query times out at ~30s with 0 tool calls + `anthropic.APITimeoutError` in the log | `ChatAnthropic(timeout=per_iter_timeout_s)` is too tight — Haiku's first tool-use response with a 2000-token system prompt needs ~5-8s and a 6s HTTP timeout cancels it before any response comes back | Set `timeout=max(15.0, max_iters * per_iter_timeout_s)` in the `ChatAnthropic` constructor; the outer `asyncio.wait_for` cap is what enforces the loop budget, not the HTTP timeout |
| `A8` agent hits max_iters without calling `submit_final_results` (n_results=0, ≥ 6 tool calls in the trace) | Haiku is making parallel tool calls within one cycle (e.g., 3 `get_trial_details` in iter 4); 5 cycles can mean 8-9 actual tool calls and the budget runs out before the agent decides to submit | Prompt change: add a hard step-counter directive — *"On cycle ≥ 4 you MUST be calling submit_final_results"*. Combined with `agentic_max_iters=6` as a production setting to give buffer for parallel bursts |
| `A8` per-query cost is 2-4× the projected $0.014 | Anthropic prompt caching not engaging on AGENT_SYSTEM_PROMPT because the prompt is just below the 4096-token auto-cache threshold; every call pays full input rate | Add `cache_control: {"type": "ephemeral"}` on the system prompt (mirror `query_parser.py:248`). Drops input cost by ~10× on warm cache hits; closes the gap to the $3/1K projection |
| `B1` `sqlite3.OperationalError: database is locked` | Multiple workers writing concurrently without `check_same_thread=False` | Re-check `trace_store.py` connection config; add `busy_timeout` |
| `B3` traces missing for rule arm | The `pipeline.search()` change didn't reach the rule-path return | Ensure write_trace is in the OUTER return, not inside an inner branch |
| `B5` Grafana panels all empty | Time range too narrow / DB path not mounted right | Check `docker logs grafana` for SQLite plugin warnings; widen time range to "last 6h" |
| `C1` rule baseline regressed > 0.01 | Phase A wiring is leaking into rule path | Check that `agentic_path_enabled=False` ALWAYS routes to rule arm; check the AB router default behaviour |
| `C2` agent mean iters > 5.5 | Loop not converging on a meaningful fraction of queries | Lower `max_iters` to 4 OR strengthen the prompt; rerun |
| `C2` agent NDCG < rule baseline on the complex slice | The agent isn't actually using `lookup_medical_concept` or `get_trial_details` — just calling `search_trials` once and submitting | Check tool-call sequences in C2 output; if 80% are `search_trials → submit_final_results`, the prompt isn't teaching the multi-step pattern. Strengthen the few-shot examples |
| `C3` routing rate < 10% | Heuristic threshold too high OR the test queries don't exercise sparse/complex patterns | Lower `agentic_routing_threshold_slots` to 3; rerun C3 |
| `C3` routing rate > 50% | Heuristic over-triggering — too many "complex" matches | Tighten the complex-pattern regex; rerun |
| `C4` "TUNE" decision: lift CI overlaps 0 | Sample size too small for the slice | Expand eval to n=30 per slice (mirror Decision 37 framework); cost ~$0.04/labeled pair |

---

# Appendix — Cost recap

| Item | Cost | Phase |
|---|---|---|
| Dev — Claude Code for Phase A scaffolding | ~$15-30 | A |
| Dev — Claude Code for Phase B observability | ~$5-15 | B |
| API — Phase A8 smoke (Haiku, 5 queries) | ~$0.50 | A8 |
| API — Phase C2 force-all eval (Haiku agent, 85 queries) | ~$1.20 | C2 |
| Haiku — Phase C2 new-NCT labeling (~5 new × 85 q × $0.002) | ~$0.85 | C2 |
| Haiku — Phase C3 (no new API calls, just analysis) | $0 | C3 |
| **Total dev + eval (one-time)** | **~$22-47** | |
| Production — agent path at 25 % routing | +$3.08 /1K queries | post-D1 |
| Production — observability (SQLite + Grafana panels) | $0 | post-D1 |

Engineer wall-clock: **~10-14 hr**, almost entirely local. No cloud GPU. No long-running training jobs to babysit.

---

# Appendix — Why `return_direct` is load-bearing

The terminator-tool pattern in Phase A4 ONLY works because we pass
`return_direct=True` to the `@tool` decorator. Without it, the agent
loop never ends after `submit_final_results` — it just keeps reasoning
until the recursion_limit fires.

Why: `langgraph.prebuilt.create_react_agent` terminates the loop in
exactly two cases (verified against langgraph 1.1.10 source in
`langgraph/prebuilt/chat_agent_executor.py`):

1. The LLM's response contains NO `tool_calls` — i.e., the agent
   produced a final natural-language answer.
2. Every tool call in the response is to a tool with
   `return_direct=True`. The check is:
       all(call["name"] in should_return_direct for call in response.tool_calls)

In case (2), the graph routes from `tools` directly to END instead
of looping back to `agent`. That's the mechanism we lean on.

Concretely: if `submit_final_results.return_direct = False`, the
agent calls the tool → the tool returns the JSON envelope → the
result goes back into the agent's context → the agent looks at the
tool result and decides what to do next → typically calls more
tools → eventually hits recursion_limit and crashes. Symptom in the
A8 smoke: every query times out at the outer 60s budget with no
final_results.

If you ever need a "fancier" terminator (e.g., conditional on tool
result), the cleanest LangGraph-native alternative is a custom
post-tools router via the `post_model_hook` arg to
create_react_agent. But for our case — "terminate on this specific
tool, full stop" — `return_direct=True` is the simplest, most
robust choice and is exactly what the prebuilt agent was designed
to support.

The matching prompt constraint (submit_final_results must be called
ALONE in its turn) is non-negotiable for the same reason: if the
agent calls submit_final_results AND another tool in the same
response, `all(...)` evaluates False and the loop continues. The
system prompt's hard-constraints block must say this explicitly.

---

# Appendix — Why Haiku 4.5 over Sonnet 4.6

The agent path's value proposition is **iterative refinement on the hard 25 % of queries**. For that pattern, the bottleneck is rarely the LLM's raw reasoning ceiling — it's whether the LLM can reliably:

1. Choose the right tool from a small set (5 tools)
2. Form a reasonable search query refinement based on the previous tool result
3. Stop calling tools when it has enough (the `submit_final_results` terminator)

All three are well within Haiku 4.5's competence for a domain-constrained problem with explicit few-shot examples. The places where Sonnet 4.6 would beat Haiku — multi-step formal reasoning, long-context synthesis — aren't load-bearing here.

The cost case is decisive:

| | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|
| Per agent query | ~$0.040 | ~$0.014 |
| At 25 % routing per 1K queries | +$9.58 | +$3.08 |
| Latency per cycle | ~1.5-2s | ~0.6-1s |

Sonnet is a worthwhile escape hatch (one model_id change behind the AB router) if Haiku's Phase C eval underperforms — e.g., mean iters > 5.5 or tool-call success < 90 %. The Phase C decision matrix calls this out explicitly under TUNE / HOLD.

---

# Appendix — Why we kept Phase B (observability) over the leaner Prometheus-only path

The earlier discussion considered shipping with **just** the 4 Prometheus counters from A6 (no SQLite, no Grafana panels). That would have saved ~2-3 hr of work and the answer to "is it working?" would be `curl /metrics | grep agent`.

Phase B exists because:

1. **Per-query debugging requires the query text.** Prometheus is for aggregate counters; the actual `raw_query` and tool-call sequence for a specific failing query lives in the SQLite store. When an interviewer asks *"show me a query the agent handled well"*, the answer is one SQL query away with Phase B and a `grep` over logs without.
2. **The dashboard is the interview demo.** The system-design round goes much better with *"here's the dashboard tracking routing rate and per-arm NDCG over time"* than *"here are some Prometheus counters; trust me they go up"*.
3. **Phase B work compounds.** The 4 Grafana panels are scaffolding any future ranking experiment also benefits from — they're not agent-specific.

If you're tight on calendar time, B1-B3 (the SQLite store + write path) is the load-bearing 60 % of Phase B; B4-B6 (Grafana provisioning + dashboards) can defer. The SQLite store alone gives you ad-hoc analysis with `sqlite3 data/agent_runs.db "<query>"` even without panels.

---

# Appendix — What this build does NOT solve (be honest in D1 writeup)

1. **The 50 in-top-10 complex misses (34.7 % of complex misses) don't need the agent.** They're already in top-10; the issue is ordering within top-10. Agent likely won't move these much — they're already "found", just ranked imperfectly. The agent's lever is the 90 rerank-bound misses where the right trial is at rank 11-80.

2. **Vague queries with no extractable patient profile.** If the parser produced `raw_query`-only (no slots), the agent gets the same raw query as input — it doesn't have a magic way to extract age/sex from "trials for my dad" that the parser missed. Decision 41's lesson applies: parser improvements are a separate lever, not absorbed by the agent.

3. **Drug-class semantics in eligibility.** The agent calls `check_trial_eligibility` which still uses the rule-based substring matching from `tools.py`. The Q413-style false positives (dacomitinib vs osimertinib EGFR-TKI class collision) remain — they need the LLM-at-matching approach (Phase B of the original plan, deprecated in this build).

4. **Retrieval coverage.** The agent re-ranks within the candidate pool the bi-encoder + BM25 already surface. If a relevant trial is at rank 200+, neither the rule pipeline nor the agent will see it. That's retrieval-bound, fixable only with bi-encoder / index work.

5. **Production load behaviour.** All measurements are on the held-out 85-query eval set. Real production traffic mix may shift the routing rate ±10 % and the cost per 1K queries proportionally. Phase B's Grafana panels exist precisely to surface this drift; the C4 measurements are "ship-with-confidence", not "won't-change-in-production".

Ship the agent for what it actually improves (complex/vague slice on routed queries + observability substrate). Don't oversell it as a complete solution to the complex-slice problem — it's one tractable intervention with a measurable lift.
