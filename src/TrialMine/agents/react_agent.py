"""Haiku 4.5 ReAct agent for hard / vague queries.

Phase A4 of ``docs/build_agent.md``. This module wraps
``langgraph.prebuilt.create_react_agent`` around the existing 5-tool
surface in :mod:`TrialMine.agents.tools` and exposes the same
``(result_dict, trace_entries)`` contract as
:class:`TrialMine.agents.orchestrator.SearchOrchestrator`, so the
LangGraph pipeline (Phase A5) can route to either arm without diverging
on downstream consumers.

Two load-bearing invariants:

* The loop terminates via ``submit_final_results`` (the only tool in
  ``ALL_TOOLS`` with ``return_direct=True`` — see ``tools.py``). Without
  that flag the loop never exits after the terminator call.
* The agent NEVER raises into the LangGraph node. Timeouts, cap-hits,
  and parse failures are returned as a degraded ``result_dict`` carrying
  an ``error`` field so the pipeline node (Phase A5) can decide whether
  to fall back to the rule arm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from TrialMine.agents.query_parser import PatientProfile
from TrialMine.agents.tools import (
    ALL_TOOLS,
    _db_conn,
    _load_trial_row,
)
from TrialMine.monitoring import (
    record_agent_cost,
    record_agent_iterations,
    record_agent_tool_call,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #


DEFAULT_MODEL = "claude-haiku-4-5"

# Sentinel error string. Phase A5's ``execute_agent_search`` node may
# inspect ``result_dict["error"]`` to decide between fallback paths.
_ERR_NO_TERMINATE = "agent_did_not_terminate"

# Hard caps on what we put into the trace + final result so a misbehaving
# agent can't blow up the trace store row.
_TRACE_ARG_MAX_CHARS = 200
_REASONING_EXCERPT_MAX_CHARS = 240
_MAX_FINAL_RESULTS = 20


# --------------------------------------------------------------------------- #
# System prompt                                                               #
# --------------------------------------------------------------------------- #


AGENT_SYSTEM_PROMPT = """You are a clinical-trial search expert helping a patient find the most
relevant clinical trials. Your job is to take a STRUCTURED PATIENT
PROFILE (condition, biomarkers, prior treatments, etc.) and produce a
ranked list of 5–10 trials with one-line explanations.

You operate inside a tool-using ReAct loop. You have FIVE tools. Call
them one at a time, in the order the recipe suggests, unless the next
step is obviously wasteful.

============================================================
TOOLS
============================================================

1. search_trials(query: str, top_k: int = 10, filters: dict | None = None)
   Run the full TrialMine retrieval pipeline (BM25 + bi-encoder semantic
   + cross-encoder rerank + LightGBM blender) over ~140K oncology trials.
   Returns a ranked list of candidates with nct_id, title, conditions,
   phase, status, score, and a `source` tag. THIS IS YOUR FIRST MOVE on
   any query. Compose `query` from the patient's condition + biomarkers
   + stage; do NOT just paste the raw query unless the profile is empty.
   Use `filters={"status": "RECRUITING"}` for actively-enrolling trials
   (the default behavior).

2. lookup_medical_concept(query: str)
   Expand a lay term, a drug name, or a specific resistance pattern into
   broader medical vocabulary. Example: "stomach cancer" → "gastric
   neoplasm", "osimertinib resistance" → "EGFR C797S / fourth-generation
   EGFR TKI / amivantamab-based regimens". USE WHEN: search_trials
   returns off-topic results, OR the patient profile mentions a specific
   prior-treatment failure (e.g., "failed osimertinib") and you want
   to expand to post-failure modalities.

3. get_trial_details(nct_id: str)
   Fetch the full record for ONE trial: title, brief_summary (truncated
   to 1500 chars), conditions, interventions, eligibility_criteria
   (truncated to 2000 chars), phase, status, enrollment. USE WHEN: you
   want to verify clinical context for the top 2–3 candidates before
   ranking them. Reading brief_summary catches off-topic matches that
   the title alone hides.

4. check_trial_eligibility(nct_id: str, patient_age: float | None,
                            patient_sex: str | None,
                            patient_condition: str | None,
                            prior_treatments: str | None)
   Score the patient against the trial's parsed eligibility criteria.
   Returns Met / Unmet / Unknown verdict + per-criterion breakdown. USE
   ONLY on the top 2–3 candidates after detail review — eligibility
   checks are slow (~50–150 ms each) and noisy on free-text criteria.
   Pass `prior_treatments` as a single comma-joined string.

5. submit_final_results(results: list[dict], reasoning: str)
   TERMINATE the loop. You MUST call this EXACTLY ONCE to end your turn,
   and it MUST be the ONLY tool called in that turn. Each entry in
   `results` is a dict with at least an `nct_id` (str) and optionally
   an `explanation` (str, one sentence per trial). `reasoning` is a
   one-paragraph rationale for the overall ordering (≥10 chars, used in
   the trace, not shown to the user). 1–20 results allowed.

============================================================
RECIPE (recommended order; deviate only if a step is obviously redundant)
============================================================

(a) Read the patient profile (`condition`, `biomarkers`,
    `prior_treatments`, `condition_stage`, `age`, `sex`, etc.). Note any
    vague phrasing or specific drug failures.

(b) Call `search_trials` with the patient's primary terms. Good first
    queries combine the condition + biomarker + stage (e.g., "EGFR
    T790M NSCLC stage IV"). Skip filters unless the patient explicitly
    asked for non-recruiting trials.

(c) If the top results look off-topic, OR the patient mentions a prior
    failure or progression, call `lookup_medical_concept` to expand
    vocabulary (e.g., "osimertinib resistance" → 4th-gen EGFR TKIs).
    Then call `search_trials` again with the expanded query.

(d) Pick the top 2–3 candidates by retrieval score. For each, call
    `get_trial_details` to read brief_summary + eligibility_criteria.
    Drop any that are clearly off-topic.

(e) For the surviving 2–3 candidates, call `check_trial_eligibility`
    using the patient's `age`, `sex`, `condition`, and a comma-joined
    `prior_treatments`. Trials with verdict "Unmet" should usually be
    dropped or down-ranked unless the criterion is one the patient
    might still satisfy (e.g., "no prior chemo within 6 months" when
    the patient finished chemo a year ago).

(f) Call `submit_final_results` with 5–10 ranked trials (highest
    relevance first) and a one-paragraph reasoning.

============================================================
HARD CONSTRAINTS — violations break the loop
============================================================

* You MUST call `submit_final_results` exactly once to end your turn.
* `submit_final_results` MUST be the ONLY tool called in its turn. Do
  NOT mix it with another tool in the same response — the terminator
  only fires when every tool call in a response is `return_direct`,
  and mixing defeats that check.
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
* If a tool returns `{"error": "<message>"}`, do NOT retry the same
  call with the same args. Either adjust the args (e.g., reformulate
  the query) or move on to a different tool.
* NEVER narrow the patient's condition to a subtype they did not name.
  If the profile says "lung cancer" without a subtype, do NOT search
  for "non-small cell lung cancer" — search for "lung cancer".
* If the retrieval returns 0 results, expand the query with
  `lookup_medical_concept` and retry once. If still 0, call
  `submit_final_results` with an empty list and a reasoning that
  explains the no-match.

============================================================
EXAMPLE 1 — complex query
============================================================

Input profile:
  raw_query: "failed osimertinib EGFR T790M NSCLC"
  condition: "non-small cell lung cancer"
  biomarkers: ["EGFR T790M"]
  prior_treatments: ["osimertinib"]

Tool sequence:
  iter 1: search_trials(
              query="EGFR T790M NSCLC post-osimertinib",
              top_k=10,
          )
          → returns a mix of trials; some include osimertinib as study
            drug (already failed by this patient).
  iter 2: lookup_medical_concept(query="osimertinib resistance")
          → returns "EGFR C797S / fourth-generation EGFR TKI /
            amivantamab / chemo + EGFR-TKI combos".
  iter 3: search_trials(
              query="EGFR-mutant NSCLC fourth-generation TKI amivantamab "
                    "post-osimertinib progression",
              top_k=10,
          )
          → returns 4th-line trials excluding prior osimertinib.
  iter 4: get_trial_details(nct_id="NCTxxxxxxxx")
          → confirms the top candidate is a post-osimertinib study.
  iter 5: submit_final_results(
              results=[
                {"nct_id": "NCTxxxxxxxx", "explanation": "4th-gen EGFR
                 TKI for post-osimertinib progression."},
                ...
              ],
              reasoning="Ranked by post-osimertinib relevance: top trials
              target T790M+C797S resistance with amivantamab or 4th-gen
              TKIs; lower-ranked trials are general EGFR-mutant NSCLC.",
          )

============================================================
EXAMPLE 2 — vague query
============================================================

Input profile:
  raw_query: "trials for my mom with breast cancer"
  condition: "breast cancer"
  (other slots empty)

Tool sequence:
  iter 1: search_trials(query="breast cancer", top_k=10)
          → returns mixed phase 2 / 3 breast cancer trials.
  iter 2: get_trial_details(nct_id="NCTxxxxxxxx")
          → top candidate is a HER2+ trial; confirm it's actively
            recruiting and the indication is breast cancer broadly.
  iter 3: submit_final_results(
              results=[
                {"nct_id": "NCTxxxxxxxx", "explanation": "Phase 3
                 recruiting trial for HER2+ breast cancer."},
                ...
              ],
              reasoning="Patient profile is sparse — only the cancer
              type was given. Ranked by retrieval score and recruiting
              status; HER2+ trials surfaced first because they're the
              most common breast-cancer subtype in the corpus.",
          )

Now read the patient profile and proceed."""


# --------------------------------------------------------------------------- #
# Public class                                                                #
# --------------------------------------------------------------------------- #


class AgenticSearchAgent:
    """Haiku 4.5 ReAct agent over the 5-tool surface in ``tools.py``.

    Phase A4 of the agent build. The agent is constructed once per
    process (the pipeline holds a lazy singleton); each :meth:`search`
    call runs an independent loop bounded by ``max_iters *
    per_iter_timeout_s``. Failures are returned as a degraded
    ``result_dict`` with an ``error`` field; the agent never raises.
    """

    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        model: str = DEFAULT_MODEL,
        max_iters: int = 5,
        per_iter_timeout_s: float = 6.0,
        api_key: str | None = None,
    ) -> None:
        """Build the LangGraph ReAct agent bound to Haiku 4.5.

        Args:
            tools: Override the default ``ALL_TOOLS`` list (useful for
                tests that inject mocked tools). Production should
                always use ``ALL_TOOLS`` — its order is load-bearing
                because ``submit_final_results`` carries
                ``return_direct=True`` and must be the loop terminator.
            model: Anthropic model id. Default ``claude-haiku-4-5``;
                see runbook Appendix "Why Haiku 4.5 over Sonnet 4.6"
                for the cost-vs-quality tradeoff.
            max_iters: Hard cap on ReAct cycles. Matches
                ``DegradationConfig.agentic_max_iters``.
            per_iter_timeout_s: Per-iter wall-clock; multiplied by
                ``max_iters`` to form the total agent budget. Matches
                ``DegradationConfig.agentic_per_iter_timeout_s``.
            api_key: Override ``ANTHROPIC_API_KEY``; raises
                ``ValueError`` if both are absent (mirrors
                :class:`QueryParserAgent`).

        Raises:
            ValueError: When no Anthropic API key is available.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY missing. Set the env var or pass api_key=.")

        self.model = model
        self.max_iters = max_iters
        self.per_iter_timeout_s = per_iter_timeout_s
        self._tools: list[BaseTool] = list(tools) if tools is not None else list(ALL_TOOLS)

        # ChatAnthropic is the langchain_anthropic wrapper that
        # create_react_agent expects. temperature=0 keeps tool-call
        # selection deterministic across runs; max_tokens=2048 is enough
        # for ~10 trials of submit_final_results args without truncation.
        #
        # ``timeout`` is the per-HTTP-request cap, NOT the agent-loop
        # budget. Sizing to the total budget (max_iters × per_iter) so
        # a single ~5-8s Haiku call with our 2k-token prompt + tool
        # schemas can complete; the outer ``asyncio.wait_for`` in
        # :meth:`search` still bounds the whole loop's wall-clock at
        # the same value. The ``max(15.0, …)`` floor guards against a
        # caller passing pathologically small per-iter values that
        # would cancel even a single warm LLM call.
        self._llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            timeout=max(15.0, max_iters * per_iter_timeout_s),
        )
        # Phase C3b.3 — enable Anthropic prompt caching on the system
        # prompt. Wrap AGENT_SYSTEM_PROMPT in a SystemMessage with
        # ``cache_control: ephemeral`` so the ~2300-token prefix bills
        # at the cached rate ($0.10/M for reads) on iters ≥ 2 instead
        # of the full input rate ($1/M) for every ReAct cycle. Net cost
        # per agent query drops from ~$0.035 to ~$0.005.
        #
        # ``prompt=`` is the langgraph 1.x kwarg; ``state_modifier=``
        # was the deprecated 0.1 form and silently ignored on newer
        # versions. Path (a) from the runbook spec: pass a
        # SystemMessage with ``additional_kwargs.cache_control``.
        # If verification (cache_read_input_tokens > 0 on iter ≥ 2)
        # shows this didn't engage, fall back to path (b) — set
        # ``model_kwargs.extra_headers`` on ChatAnthropic.
        # Path (a) — ``SystemMessage(content=str, additional_kwargs=...)`` —
        # was tried first but didn't engage on the installed
        # langchain_anthropic (cache_read_input_tokens stayed 0 across
        # iterations in the C3b.3 verify probe). Falling back to the
        # content-block form below, which mirrors the raw Anthropic SDK
        # shape: a list of text parts each with its own cache_control
        # annotation. langchain_anthropic forwards this list verbatim
        # to the SDK so caching engages.
        self._agent = create_react_agent(
            self._llm,
            tools=self._tools,
            prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": AGENT_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
        )

    # ------------------------------------------------------------------ #
    # Public entry point                                                  #
    # ------------------------------------------------------------------ #

    async def search(self, profile: PatientProfile) -> tuple[dict, list[dict]]:
        """Run the agent loop end-to-end.

        Args:
            profile: Parsed patient profile. ``raw_query`` is always
                populated; structured slots may be sparse.

        Returns:
            ``(result_dict, trace_entries)`` matching the orchestrator
            contract. ``result_dict`` carries the four standard keys
            ``results``, ``query_used``, ``filters``,
            ``normalized_condition``; on degraded paths it also carries
            ``error`` for the Phase A5 pipeline node to inspect.
        """
        t_start = time.perf_counter()
        trace_entries: list[dict] = [
            {
                "step": "agent_start",
                "duration_ms": 0.0,
                "decisions": {
                    "model": self.model,
                    "max_iters": self.max_iters,
                    "per_iter_timeout_s": self.per_iter_timeout_s,
                },
            }
        ]

        initial_state = {
            "messages": [HumanMessage(content=self._format_initial_message(profile))],
        }
        # LangGraph counts state transitions, not LLM calls. A tool-use
        # cycle is two transitions (agent → tools → agent); +2 covers
        # the entry/exit. Without +2 a max_iters-budget run gets cancelled
        # right at the last iteration.
        recursion_limit = self.max_iters * 2 + 2
        total_budget_s = self.max_iters * self.per_iter_timeout_s

        try:
            final_state = await asyncio.wait_for(
                self._agent.ainvoke(
                    initial_state,
                    config={"recursion_limit": recursion_limit},
                ),
                timeout=total_budget_s,
            )
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.warning(
                "AgenticSearchAgent timed out after %.1fs (budget=%.1fs)",
                elapsed_ms / 1000,
                total_budget_s,
            )
            trace_entries.append(
                {
                    "step": "agent_timeout",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "budget_s": total_budget_s,
                        "error": _ERR_NO_TERMINATE,
                    },
                }
            )
            # Iteration count is unknown on timeout (no final_state) — record
            # 0 so the histogram still shows the "capped" outcome.
            record_agent_iterations(0, "capped")
            return self._degraded_result(_ERR_NO_TERMINATE, profile), trace_entries
        except Exception as exc:  # noqa: BLE001 — surface non-fatally
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.exception("AgenticSearchAgent invocation failed")
            trace_entries.append(
                {
                    "step": "agent_error",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:_TRACE_ARG_MAX_CHARS],
                    },
                }
            )
            record_agent_iterations(0, "errored")
            return (
                self._degraded_result(
                    f"agent_invocation_error: {type(exc).__name__}",
                    profile,
                ),
                trace_entries,
            )

        # Walk the final message list to (a) reconstruct the per-tool
        # trace, (b) capture the agent's last search / lookup args for
        # the result_dict, (c) build a nct_id → eligibility lookup,
        # (d) extract submit_final_results' final args, (e) collect
        # per-tool status, and (f) sum token usage for cost reporting.
        messages: list = final_state.get("messages") or []
        (
            submit_args,
            last_search_query,
            last_lookup_input,
            eligibility_by_nct,
            iters_used,
            tool_call_entries,
            tool_call_status,
            (
                input_tokens_total,
                output_tokens_total,
                cache_read_total,
                cache_creation_total,
            ),
        ) = self._walk_messages(messages)
        trace_entries.extend(tool_call_entries)

        # Fan per-tool status into Prometheus. Done here (not inside the
        # walk) so the helper call sits at the same termination level as
        # record_agent_iterations / record_agent_cost.
        for name, ok in tool_call_status:
            record_agent_tool_call(name, ok=ok)

        # Cost is recorded on every path that has token info, including
        # the no-terminator path below — the spend has already been
        # incurred even if the agent didn't finish cleanly. C3b.3 added
        # the cache-aware kwargs so reads bill at $0.10/M (10% of input
        # rate) instead of the full rate.
        if input_tokens_total or output_tokens_total:
            record_agent_cost(
                self.model,
                input_tokens_total,
                output_tokens_total,
                cache_read_tokens=cache_read_total,
                cache_creation_tokens=cache_creation_total,
            )

        if submit_args is None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.warning(
                "AgenticSearchAgent finished %d iters without calling submit_final_results",
                iters_used,
            )
            trace_entries.append(
                {
                    "step": "agent_no_terminator",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "iters_used": iters_used,
                        "error": _ERR_NO_TERMINATE,
                    },
                }
            )
            record_agent_iterations(iters_used, "capped")
            return self._degraded_result(_ERR_NO_TERMINATE, profile), trace_entries

        # Synthesize the final result list: SQLite lookup per nct_id,
        # merged with the agent's per-trial explanation + last eligibility.
        final_results = self._synthesize_results(
            submit_args.get("results") or [], eligibility_by_nct
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        reasoning = str(submit_args.get("reasoning") or "")
        trace_entries.append(
            {
                "step": "agent_submit",
                "duration_ms": round(elapsed_ms, 2),
                "decisions": {
                    "n_results": len(final_results),
                    "iters_used": iters_used,
                    "reasoning_excerpt": _trim_str(reasoning, _REASONING_EXCERPT_MAX_CHARS),
                    "input_tokens": input_tokens_total,
                    "output_tokens": output_tokens_total,
                    # Phase C3b.3 cache observability. Non-zero
                    # cache_read on iter ≥ 2 means caching engaged.
                    "cache_read_input_tokens": cache_read_total,
                    "cache_creation_input_tokens": cache_creation_total,
                },
            }
        )
        record_agent_iterations(iters_used, "submitted")
        logger.info(
            "AgenticSearchAgent completed in %.0fms (iters=%d, results=%d, in=%d, out=%d)",
            elapsed_ms,
            iters_used,
            len(final_results),
            input_tokens_total,
            output_tokens_total,
        )

        return (
            {
                "results": final_results,
                "query_used": last_search_query,
                "filters": {},
                "normalized_condition": last_lookup_input,
            },
            trace_entries,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_initial_message(profile: PatientProfile) -> str:
        """Render the patient profile as a HumanMessage payload.

        The agent receives the raw query plus a compact JSON sidecar of
        the extracted slots so it can read the structured fields without
        re-parsing.
        """
        extracted = {
            "condition": profile.condition,
            "condition_stage": profile.condition_stage,
            "age": profile.age,
            "sex": profile.sex,
            "biomarkers": profile.biomarkers,
            "prior_treatments": profile.prior_treatments,
            "preferences": profile.preferences,
            "location": profile.location,
        }
        return (
            f"Patient query: {profile.raw_query}\n\n"
            f"Extracted profile fields:\n"
            f"{json.dumps(extracted, indent=2, default=str)}\n\n"
            "Find clinical trials that match this profile."
        )

    @staticmethod
    def _walk_messages(
        messages: list,
    ) -> tuple[
        dict | None,
        str | None,
        str | None,
        dict[str, dict],
        int,
        list[dict],
        list[tuple[str, bool]],
        tuple[int, int, int, int],
    ]:
        """Reconstruct trace + key signals from the final message list.

        Returns (in order):

        * ``submit_args`` — args of the LAST submit_final_results tool
          call, or ``None`` if the terminator was never invoked.
        * ``last_search_query`` — query arg of the last ``search_trials``.
        * ``last_lookup_input`` — query arg of the last
          ``lookup_medical_concept``.
        * ``eligibility_by_nct`` — nct_id → parsed eligibility dict from
          the last ``check_trial_eligibility`` for that trial.
        * ``iters_used`` — count of AIMessages that issued any tool call.
        * ``tool_call_entries`` — trace entries (one per tool call).
        * ``tool_call_status`` — list of ``(tool_name, ok)`` pairs in
          execution order, used to fan into ``AGENT_TOOL_CALLS``.
          ``ok=False`` when the matched ``ToolMessage`` returned the
          ``{"error": "..."}`` envelope; ``ok=True`` otherwise. Tool
          calls without a matched ``ToolMessage`` (e.g. the loop exited
          mid-cycle) are recorded with ``ok=False`` so unmatched failures
          show up.
        * ``cumulative_usage`` — 4-tuple
          ``(input_tokens, output_tokens, cache_read_tokens,
          cache_creation_tokens)`` summed across every AIMessage.
          ``input_tokens`` is the LangChain-normalised count (which
          INCLUDES cache reads); the two cache fields are pulled from
          the raw Anthropic metadata so the cost calculation can apply
          the discounted cached-read rate. Fed into
          ``record_agent_cost`` exactly once at termination.
        """
        submit_args: dict | None = None
        last_search_query: str | None = None
        last_lookup_input: str | None = None
        eligibility_by_nct: dict[str, dict] = {}
        iters_used = 0
        tool_call_entries: list[dict] = []
        # tool_call_id → (tool_name, index_in_status_list) so a ToolMessage
        # arriving later can flip the status from True (default-ok) to
        # False when its JSON content has an "error" key.
        tool_calls_by_id: dict[str, tuple[str, int]] = {}
        tool_call_status: list[tuple[str, bool]] = []
        input_tokens_total = 0
        output_tokens_total = 0
        # Phase C3b.3 — track cache hits / cache creation tokens
        # separately so the cost calculation can apply the discounted
        # cached-read rate ($0.10/M on Haiku 4.5 — 10% of the input
        # rate) instead of double-charging at the full input rate.
        cache_read_total = 0
        cache_creation_total = 0

        for msg in messages:
            if isinstance(msg, AIMessage):
                # Token accounting — every AIMessage carries an
                # ``usage_metadata`` dict from langchain_anthropic in the
                # standardised shape {"input_tokens", "output_tokens",
                # "total_tokens"}. Falls back to the raw Anthropic shape
                # under ``response_metadata.usage`` for older SDKs.
                # Anthropic also reports ``cache_read_input_tokens`` and
                # ``cache_creation_input_tokens`` for cached prefixes
                # (see ``query_parser.py:248`` for the analogous raw-SDK
                # pattern); langchain_anthropic surfaces them via
                # ``response_metadata.usage`` rather than the normalised
                # ``usage_metadata`` dict.
                usage = getattr(msg, "usage_metadata", None) or {}
                if not usage:
                    raw = getattr(msg, "response_metadata", None) or {}
                    usage = raw.get("usage") if isinstance(raw, dict) else {}
                # Pull cache fields from the raw metadata even when the
                # normalised dict was non-empty (langchain doesn't echo
                # them into usage_metadata yet).
                raw_meta = getattr(msg, "response_metadata", None) or {}
                raw_usage = raw_meta.get("usage") if isinstance(raw_meta, dict) else {}
                if isinstance(usage, dict):
                    input_tokens_total += int(usage.get("input_tokens") or 0)
                    output_tokens_total += int(usage.get("output_tokens") or 0)
                if isinstance(raw_usage, dict):
                    cache_read_total += int(raw_usage.get("cache_read_input_tokens") or 0)
                    cache_creation_total += int(raw_usage.get("cache_creation_input_tokens") or 0)

                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    continue
                iters_used += 1
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args") or {}
                    if name == "search_trials":
                        q = args.get("query")
                        if isinstance(q, str):
                            last_search_query = q
                    elif name == "lookup_medical_concept":
                        q = args.get("query")
                        if isinstance(q, str):
                            last_lookup_input = q
                    elif name == "submit_final_results":
                        # Last call wins if the agent (incorrectly)
                        # called it twice — defensive only; the prompt
                        # forbids it.
                        submit_args = args
                    tool_call_entries.append(
                        {
                            "step": "tool_call",
                            "duration_ms": 0.0,  # post-hoc walk — per-tool wall-clock not captured
                            "decisions": {
                                "tool": name,
                                "iter": iters_used,
                                "args_summary": _truncate_args(args),
                            },
                        }
                    )
                    tool_call_status.append((name, True))
                    tc_id = tc.get("id")
                    if tc_id:
                        tool_calls_by_id[tc_id] = (name, len(tool_call_status) - 1)
            elif isinstance(msg, ToolMessage):
                content = msg.content
                # Best-effort: try to detect the {"error": "..."} envelope
                # produced by our tools. If parsing fails we treat the
                # call as ok — JSON parse failure shouldn't double-count
                # as an error.
                is_error = False
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and "error" in parsed:
                        is_error = True
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in tool_calls_by_id:
                    name, idx = tool_calls_by_id[tc_id]
                    tool_call_status[idx] = (name, not is_error)

                if msg.name == "check_trial_eligibility":
                    try:
                        elig = json.loads(content) if isinstance(content, str) else None
                    except (json.JSONDecodeError, TypeError):
                        elig = None
                    if isinstance(elig, dict):
                        nct = (elig.get("nct_id") or "").strip().upper()
                        if nct:
                            eligibility_by_nct[nct] = elig

        return (
            submit_args,
            last_search_query,
            last_lookup_input,
            eligibility_by_nct,
            iters_used,
            tool_call_entries,
            tool_call_status,
            (
                input_tokens_total,
                output_tokens_total,
                cache_read_total,
                cache_creation_total,
            ),
        )

    @staticmethod
    def _synthesize_results(
        raw_results: list[dict],
        eligibility_by_nct: dict[str, dict],
    ) -> list[dict]:
        """Look up each agent-ranked nct_id in SQLite and enrich.

        Per-result shape matches ``SearchOrchestrator``'s Step 6 contract
        (see ``orchestrator.py``): ``nct_id``, ``title``, ``phase``,
        ``status``, ``enrollment``, ``conditions``, ``score``, ``source``,
        ``explanation``, ``warnings``, ``eligibility``, ``url``.

        Unknown / missing nct_ids are dropped with a warning. Hard cap
        of :data:`_MAX_FINAL_RESULTS` so a misbehaving agent that returns
        100 trials can't blow up downstream consumers.
        """
        enriched: list[dict] = []
        if not raw_results:
            return enriched

        try:
            con = _db_conn()
        except Exception:
            logger.exception("synthesize_results: SQLite unavailable; returning empty list")
            return enriched

        try:
            for entry in raw_results[:_MAX_FINAL_RESULTS]:
                if not isinstance(entry, dict):
                    continue
                nct = str(entry.get("nct_id") or "").strip().upper()
                if not nct:
                    continue
                explanation = str(entry.get("explanation") or "")
                row = _safe_load_trial_row(con, nct)
                if row is None:
                    logger.warning("synthesize_results: %s not found in SQLite", nct)
                    continue
                enriched.append(_row_to_result(row, nct, explanation, eligibility_by_nct.get(nct)))
        finally:
            con.close()

        return enriched

    @staticmethod
    def _degraded_result(error: str, profile: PatientProfile) -> dict:
        """Build a result_dict for a degraded path.

        Has the four standard keys (so downstream consumers don't break
        on a missing field) plus an ``error`` field that Phase A5's
        pipeline node can read to decide on fallback.
        """
        return {
            "results": [],
            "query_used": profile.raw_query,
            "filters": {},
            "normalized_condition": None,
            "error": error,
        }


# --------------------------------------------------------------------------- #
# Module-level helpers                                                        #
# --------------------------------------------------------------------------- #


def _truncate_args(args: dict) -> dict:
    """Per-value truncate so the trace stays compact (and JSON-safe)."""
    out: dict[str, Any] = {}
    for key, value in (args or {}).items():
        if isinstance(value, str) and len(value) > _TRACE_ARG_MAX_CHARS:
            out[key] = value[:_TRACE_ARG_MAX_CHARS] + "…"
        elif isinstance(value, (list, dict)):
            # Stringify-and-truncate so the trace row stays one-line on dashboards.
            blob = json.dumps(value, default=str)[: _TRACE_ARG_MAX_CHARS + 1]
            out[key] = (
                (blob[:_TRACE_ARG_MAX_CHARS] + "…") if len(blob) > _TRACE_ARG_MAX_CHARS else blob
            )
        else:
            out[key] = value
    return out


def _trim_str(text: str, max_chars: int) -> str:
    """Soft trim; preserve original if shorter than max_chars."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _safe_load_trial_row(con: sqlite3.Connection, nct_id: str) -> sqlite3.Row | None:
    """``_load_trial_row`` with a try/except so a single bad nct doesn't
    abort the whole synthesis."""
    try:
        return _load_trial_row(con, nct_id)
    except sqlite3.Error:
        logger.exception("_safe_load_trial_row: SQLite error for %s", nct_id)
        return None


def _row_to_result(
    row: sqlite3.Row,
    nct_id: str,
    explanation: str,
    eligibility: dict | None,
) -> dict:
    """Project a SQLite trial row into the orchestrator's per-result shape.

    ``conditions`` is stored as a JSON array in SQLite but the orchestrator
    contract is a semicolon-joined string; convert to keep callers (UI,
    explanation builders) consistent across arms.
    """
    # SIM401's row.get("conditions") suggestion doesn't apply — sqlite3.Row
    # has no .get() method; falling back to ``in`` + ``[]`` is the API.
    raw_conditions = row["conditions"] if "conditions" in row else None  # noqa: SIM401
    conditions_str: str
    try:
        parsed = json.loads(raw_conditions or "[]")
        conditions_str = (
            "; ".join(str(c) for c in parsed) if isinstance(parsed, list) else str(parsed)
        )
    except (json.JSONDecodeError, TypeError):
        conditions_str = str(raw_conditions or "")

    return {
        "nct_id": row["nct_id"],
        "title": row["title"] or "",
        "phase": row["phase"],
        "status": row["status"],
        "enrollment": row["enrollment"],
        "conditions": conditions_str,
        # The agent path has no LightGBM blender; ranking comes from
        # the agent's own ordering, which we represent as score=None.
        "score": None,
        "source": "agent",
        "explanation": explanation,
        "warnings": [],
        "eligibility": eligibility,
        "url": f"https://clinicaltrials.gov/study/{row['nct_id']}",
    }


__all__ = ["AgenticSearchAgent", "AGENT_SYSTEM_PROMPT", "DEFAULT_MODEL"]
