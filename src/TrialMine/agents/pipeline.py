# ruff: noqa: E501
"""End-to-end LangGraph pipeline: QueryParser → route_decision → (rule | agent) → fallback.

The Phase A5 topology (the agent arm is built but inert until Phase D
flips ``DegradationConfig.agentic_path_enabled``):

    START → parse_query → route_decision ─┬─ "rule"  ──→ execute_search ─┬─→ END
                                          │                              └─→ fallback_search → END
                                          └─ "agent" ──→ execute_agent_search ─┬─→ END
                                                                               ├─→ execute_search   (retry as rule on agent_arm_failed:)
                                                                               └─→ fallback_search  (other errors)

The graph carries a :class:`SearchState` ``TypedDict``. ``agent_trace`` is the
primary observability surface — every node appends one or more structured
entries via the ``operator.add`` reducer. ``route`` uses override semantics
so the last writer wins.

Wall-clock budget is enforced by :func:`search` via ``asyncio.wait_for``; on
timeout we return a degraded response rather than letting the caller hang.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import os
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from TrialMine.agents.orchestrator import SearchOrchestrator
from TrialMine.agents.query_parser import PatientProfile, QueryParserAgent
from TrialMine.config import get_default_degradation
from TrialMine.monitoring import (
    record_agent_failure,
    record_agent_routing,
    time_stage_cm,
)

logger = logging.getLogger(__name__)


# Pulled from DegradationConfig at import time so callers that don't pass
# an explicit timeout get the configured budget. The runtime
# get_default_degradation() singleton can still be flipped per-process by
# tests / future control planes — pass an explicit timeout to override.
DEFAULT_TIMEOUT = get_default_degradation().skip_agent_if_slow_s
DEFAULT_FALLBACK_TOP_K = 20

# Phase B3 — SQLite trace persistence. Default ON so every pipeline
# invocation lands a row in ``data/agent_runs.db`` for Grafana / ad-hoc
# SQL analysis. Set ``TRIALMINE_TRACE_PERSIST=0`` in tests or in any
# process that should remain trace-free (e.g. ``scripts/`` runs that
# don't want to pollute the prod trace store with synthetic queries).
# The trace_store module owns its own lazy ``_initialized`` guard, so
# the DB file is only created on the first write — not at import time.
TRACE_PERSIST_ENABLED: bool = os.getenv("TRIALMINE_TRACE_PERSIST", "1") == "1"


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #


class SearchState(TypedDict):
    """LangGraph state for the search pipeline.

    ``agent_trace`` uses ``operator.add`` so each node returns a list of new
    entries that the reducer concatenates onto the running trace. Other
    fields use override semantics (the latest write wins).
    """

    raw_query: str
    patient_profile: dict | None
    search_results: dict | None
    agent_trace: Annotated[list[dict], operator.add]
    error: str | None
    used_fallback: bool
    # ``route`` is written by the ``route_decision`` node:
    # ``{"arm": "rule"|"agent", "reason": str, "signals": dict}``. Override
    # semantics — only one node writes it per request.
    route: dict | None


# --------------------------------------------------------------------------- #
# Lazy singletons                                                             #
# --------------------------------------------------------------------------- #

_parser_singleton: QueryParserAgent | None = None
_orchestrator_singleton: SearchOrchestrator | None = None
# Forward-typed (string) so we don't import :mod:`react_agent` at module-load
# time. ``react_agent`` pulls in langchain_anthropic + langgraph.prebuilt,
# which we'd rather not pay for on every pipeline import — especially while
# the agent path is default-OFF through Phase A–C.
_agent_singleton: AgenticSearchAgent | None = None  # type: ignore[name-defined]  # noqa: F821


def _get_parser() -> QueryParserAgent:
    """Return a cached :class:`QueryParserAgent` (Haiku 4.5 via the Anthropic SDK)."""
    global _parser_singleton
    if _parser_singleton is None:
        _parser_singleton = QueryParserAgent()
    return _parser_singleton


def _get_orchestrator() -> SearchOrchestrator:
    """Return a cached :class:`SearchOrchestrator` sharing tools.py singletons."""
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = SearchOrchestrator()
    return _orchestrator_singleton


def _get_agent() -> Any:
    """Return a cached :class:`AgenticSearchAgent`.

    Pulls ``max_iters`` and ``per_iter_timeout_s`` from
    :func:`get_default_degradation`, so a control-plane that flips the
    config at runtime is honoured at first agent invocation. Import is
    deferred to first call so module load doesn't pull in
    ``langchain_anthropic`` when the agent path is default-OFF.
    """
    global _agent_singleton
    if _agent_singleton is None:
        from TrialMine.agents.react_agent import AgenticSearchAgent

        config = get_default_degradation()
        _agent_singleton = AgenticSearchAgent(
            max_iters=config.agentic_max_iters,
            per_iter_timeout_s=config.agentic_per_iter_timeout_s,
        )
    return _agent_singleton


# --------------------------------------------------------------------------- #
# Nodes                                                                        #
# --------------------------------------------------------------------------- #


async def parse_query(state: SearchState) -> dict:
    """Parse the raw query into a structured :class:`PatientProfile`.

    The underlying SDK call is sync, so we run it in a worker thread to
    avoid blocking the event loop. The parser swallows its own errors and
    always returns a profile (with only ``raw_query`` populated when
    extraction fails) — so this node never raises.
    """
    t0 = time.perf_counter()
    parser = _get_parser()
    profile: PatientProfile = await asyncio.to_thread(parser.parse, state["raw_query"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    populated = (
        sum(
            1
            for v in (
                profile.condition,
                profile.condition_stage,
                profile.age,
                profile.sex,
                profile.location,
            )
            if v is not None
        )
        + len(profile.prior_treatments)
        + len(profile.biomarkers)
        + len(profile.preferences)
    )

    return {
        "patient_profile": profile.model_dump(),
        "agent_trace": [
            {
                "step": "parse_query",
                "duration_ms": round(elapsed_ms, 2),
                "decisions": {
                    "condition": profile.condition,
                    "condition_stage": profile.condition_stage,
                    "age": profile.age,
                    "sex": profile.sex,
                    "biomarkers": profile.biomarkers,
                    "preferences": profile.preferences,
                    "location": profile.location,
                    "n_slots_populated": populated,
                },
            }
        ],
    }


async def route_decision(state: SearchState) -> dict:
    """Pick the rule arm or the agent arm for this query.

    Two layers compose here:

    1. **AB-router gate** (Phase A7) — when
       ``agentic_path_enabled=True``, the ``agent_path_v1`` experiment
       buckets traffic 50/50 by hashing ``raw_query``. The control half
       short-circuits to the rule arm with ``reason="ab_router_control"``;
       the treatment half falls through to layer 2.
    2. **Pure heuristic** — :func:`TrialMine.agents.query_router.route_decision`
       (Phase A3 — no IO, no LLM) decides agent vs rule based on
       ``populated_slots`` and the complex-pattern regex.

    With ``agentic_path_enabled=False`` (the Phase A–C default) layer 1
    is skipped entirely and the pure heuristic returns
    ``arm="rule"`` / ``reason="agentic_path_disabled"`` — production
    behaviour is unchanged.

    Production agent coverage is roughly
    ``0.5 × P(sparse OR complex) ≈ 12–25 %`` of traffic depending on
    query mix.
    """
    t0 = time.perf_counter()
    profile = PatientProfile.model_validate(
        state.get("patient_profile") or {"raw_query": state["raw_query"]}
    )
    config = get_default_degradation()

    # Layer 1 — AB-router gate. Lazy imports keep ``hashlib`` and the
    # experiments package off the rule-only import path when the agent
    # is hard-disabled.
    if config.agentic_path_enabled:
        import hashlib

        from TrialMine.experiments.ab_test import get_default_router

        subject_id = hashlib.sha256((profile.raw_query or "").encode()).hexdigest()[:16]
        router = get_default_router()
        variant = router.route(subject_id=subject_id, experiment_name="agent_path_v1")
        router.log_exposure(subject_id, "agent_path_v1", variant)
        if variant == "control":
            elapsed_ms = (time.perf_counter() - t0) * 1000
            record_agent_routing("rule", "ab_router_control")
            return {
                "route": {
                    "arm": "rule",
                    "reason": "ab_router_control",
                    "signals": {
                        "subject_id": subject_id,
                        "variant": variant,
                    },
                },
                "agent_trace": [
                    {
                        "step": "route_decision",
                        "duration_ms": round(elapsed_ms, 2),
                        "decisions": {
                            "arm": "rule",
                            "reason": "ab_router_control",
                            "subject_id": subject_id,
                            "variant": variant,
                        },
                    }
                ],
            }
        # variant == "treatment": fall through to the heuristic.

    # Layer 2 — pure A3 heuristic. Aliased import avoids the obvious
    # name collision with this node.
    from TrialMine.agents.query_router import route_decision as _route

    decision = _route(profile, config)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    record_agent_routing(decision.arm, decision.reason)
    return {
        "route": {
            "arm": decision.arm,
            "reason": decision.reason,
            "signals": decision.signals,
        },
        "agent_trace": [
            {
                "step": "route_decision",
                "duration_ms": round(elapsed_ms, 2),
                "decisions": {
                    "arm": decision.arm,
                    "reason": decision.reason,
                    **decision.signals,
                },
            }
        ],
    }


async def execute_agent_search(state: SearchState) -> dict:
    """Run the Haiku 4.5 ReAct agent on the parsed profile.

    Mirrors :func:`execute_search`'s try/except shape but signals failure
    with the ``agent_arm_failed:`` prefix so :func:`_route_after_search`
    routes the retry through the rule arm rather than the fallback. The
    agent itself swallows timeouts and parse failures into a degraded
    ``result_dict`` (see :class:`AgenticSearchAgent`), so this except
    branch only fires on unexpected Python exceptions (e.g. SQLite
    unavailable mid-synthesis).
    """
    t0 = time.perf_counter()
    agent = _get_agent()

    profile_dict = state.get("patient_profile") or {"raw_query": state["raw_query"]}
    try:
        profile = PatientProfile.model_validate(profile_dict)
    except Exception as exc:
        logger.warning("Could not validate patient_profile dict: %s", exc)
        profile = PatientProfile(raw_query=state["raw_query"])

    try:
        with time_stage_cm("agent_react_search"):
            result_dict, trace_entries = await agent.search(profile)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("execute_agent_search failed; will retry as rule")
        record_agent_failure("execute_agent_search")
        # Same rationale as the degraded path below — the rule arm is
        # about to retry, so flag the fallback in state for the trace
        # store.
        return {
            "error": f"agent_arm_failed: {type(exc).__name__}: {exc}",
            "used_fallback": True,
            "agent_trace": [
                {
                    "step": "execute_agent_search_error",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
            ],
        }

    # Phase C3b.1 — detect the agent's degraded result_dict and elevate
    # the inner error to state["error"] so _route_after_search triggers
    # the retry-as-rule edge wired in A5. Without this, the 18-of-85
    # ``agent_did_not_terminate`` runs from C2 returned 0 results to
    # the user instead of falling back to the rule arm.
    #
    # The error prefix MUST be ``agent_arm_failed:`` — that's what the
    # existing _route_after_search check looks for; any other prefix
    # routes to fallback_search instead. The trace step name is
    # ``execute_agent_search_degraded`` (NOT _error, used by the
    # exception path) so Grafana can distinguish "agent raised" from
    # "agent gave up cleanly". search_results is INTENTIONALLY omitted
    # from this return — the retry's execute_search will overwrite it,
    # and explicitly omitting keeps the contract clean.
    inner_error = (result_dict or {}).get("error")
    if inner_error:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "execute_agent_search degraded (%s); routing to retry-as-rule",
            inner_error,
        )
        record_agent_failure("execute_agent_search_degraded")
        # ``used_fallback=True`` here is honest: the agent didn't produce
        # a usable result, the rule arm is about to retry, and the
        # trace-store row should reflect that a fallback occurred (was
        # silently 0 for the first 231 agent runs before this fix).
        return {
            "error": f"agent_arm_failed: {inner_error}",
            "used_fallback": True,
            "agent_trace": trace_entries
            + [
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
        # Clear any stale error from upstream; harmless when there isn't one.
        "error": None,
    }


async def execute_search(state: SearchState) -> dict:
    """Run the rule-based :class:`SearchOrchestrator` on the parsed profile.

    Wraps the orchestrator call in try/except so any failure (ES down,
    embedder OOM, unexpected data shape) sets ``state["error"]`` and lets
    the conditional edge route to ``fallback_search``. Trace entries
    produced before the failure point are still preserved.
    """
    t0 = time.perf_counter()
    orchestrator = _get_orchestrator()

    profile_dict = state.get("patient_profile") or {"raw_query": state["raw_query"]}
    try:
        profile = PatientProfile.model_validate(profile_dict)
    except Exception as exc:
        logger.warning("Could not validate patient_profile dict: %s", exc)
        profile = PatientProfile(raw_query=state["raw_query"])

    try:
        # Orchestrator.search is async — heavy steps internally use
        # asyncio.to_thread + asyncio.gather, so we await directly. The
        # context manager records SEARCH_STAGE_LATENCY[stage="agent_search"]
        # on both success and exception paths.
        with time_stage_cm("agent_search"):
            result_dict, trace_entries = await orchestrator.search(profile)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("execute_search failed; routing to fallback")
        record_agent_failure("execute_search")
        return {
            "error": f"execute_search failed: {type(exc).__name__}: {exc}",
            "agent_trace": [
                {
                    "step": "execute_search_error",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
            ],
        }

    return {
        "search_results": result_dict,
        "agent_trace": trace_entries,
        # Clear a stale ``agent_arm_failed:`` error left over from the
        # agent arm — the rule retry succeeded, so the pipeline is in a
        # healthy state again. Without this, ``_route_after_search``
        # would see the stale prefix and try to retry again, blowing up
        # the rule-arm conditional dict (which intentionally has no
        # ``execute_search`` edge — see :func:`build_pipeline`).
        "error": None,
    }


async def _run_fallback_search_for_query(raw_query: str, reason: str) -> dict:
    """Pure async helper — runs the fallback hybrid search.

    Extracted from the ``fallback_search`` LangGraph node so the outer
    ``search()`` exit handlers (TimeoutError, generic Exception) can
    invoke the same recovery path the inner bubble-fix already uses via
    LangGraph's conditional edge. Without this helper, outer failures
    short-circuit to ``results=[]`` while inner failures get rule-arm
    retry — same conceptual event, asymmetric user experience.

    Returns the same state-merge dict shape ``fallback_search`` returns:
    ``search_results``, ``used_fallback`` (always True), ``agent_trace``,
    and ``error`` on the second-level failure branch.
    """
    t0 = time.perf_counter()
    try:
        from TrialMine.agents.tools import _get_hybrid

        retriever = _get_hybrid()
        ranked = await asyncio.to_thread(retriever.search, raw_query, DEFAULT_FALLBACK_TOP_K, None)
        results = [
            {
                "nct_id": r.get("nct_id"),
                "title": r.get("title", ""),
                "phase": r.get("phase"),
                "status": r.get("status"),
                "enrollment": r.get("enrollment"),
                "conditions": r.get("conditions", ""),
                "score": r.get("score"),
                "source": r.get("source"),
                "explanation": "Returned by fallback hybrid search "
                "(no eligibility check, no re-ranking).",
                "warnings": ["fallback path"],
                "eligibility": None,
                "url": f"https://clinicaltrials.gov/study/{r['nct_id']}"
                if r.get("nct_id")
                else None,
            }
            for r in ranked
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "search_results": {
                "results": results,
                "query_used": raw_query,
                "filters": {},
                "normalized_condition": None,
            },
            "used_fallback": True,
            "agent_trace": [
                {
                    "step": "fallback_search",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "reason": reason,
                        "n_results": len(results),
                    },
                }
            ],
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("fallback_search itself failed")
        record_agent_failure("fallback_search")
        return {
            "search_results": {
                "results": [],
                "query_used": raw_query,
                "filters": {},
                "normalized_condition": None,
            },
            "used_fallback": True,
            "error": (
                f"both paths failed — primary: {reason}; fallback: {type(exc).__name__}: {exc}"
            ),
            "agent_trace": [
                {
                    "step": "fallback_search_error",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "primary_reason": reason,
                        "fallback_error": str(exc),
                    },
                }
            ],
        }


async def fallback_search(state: SearchState) -> dict:
    """Degraded path: plain hybrid search via :meth:`HybridRetriever.search`.

    No re-ranking, no eligibility check, no template explanations — just
    BM25 + semantic + RRF. Wraps :func:`_run_fallback_search_for_query`
    so the same recovery logic is reachable from outside the LangGraph
    state machine (e.g. the outer wall-clock timeout handler).
    """
    raw_query = state["raw_query"]
    reason = state.get("error") or "execute_search failed"
    return await _run_fallback_search_for_query(raw_query, reason)


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #


def _extract_pipeline_kind(trace: list[dict]) -> str | None:
    """Pull ``pipeline_kind`` from the rule arm's ``retrieve`` step.

    The orchestrator (rule arm) writes a ``retrieve`` trace entry whose
    ``decisions.pipeline`` is one of ``full_pipeline`` /
    ``hybrid_only_disabled`` / ``hybrid_only_after_timeout``. The agent
    arm doesn't emit a ``retrieve`` step (it goes through the LangGraph
    tools directly), so this returns ``None`` for agent-routed traces —
    that distinction is itself a useful signal in the trace store.
    """
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        if entry.get("step") == "retrieve":
            decisions = entry.get("decisions") or {}
            kind = decisions.get("pipeline")
            return kind if isinstance(kind, str) else None
    return None


def _persist_trace_safely(**kwargs: Any) -> None:
    """Fan one ``pipeline.search()`` invocation into the SQLite trace store.

    Single seam for all three return paths in :func:`search` (success,
    timeout, exception). Swallows every exception so a trace-store
    failure can NEVER break the search response — this is additive
    instrumentation, not a load-bearing leg of the pipeline. The
    ``TRACE_PERSIST_ENABLED`` flag short-circuits before the import so
    opt-out runs don't even load the trace_store module.

    Note: this is ADDITIVE to the existing ``record_agent_trace`` call
    in ``api/routes.py`` which fans the same trace into Prometheus's
    SEARCH_STAGE_LATENCY histograms. Different surface (counters vs
    rows), different purpose (aggregate dashboards vs per-query
    debugging) — both stay.
    """
    if not TRACE_PERSIST_ENABLED:
        return
    try:
        from TrialMine.monitoring.trace_store import write_trace

        write_trace(**kwargs)
    except Exception:  # noqa: BLE001 — trace failures must never propagate
        logger.exception("trace persist failed; not blocking response")


def _route_after_parse(
    state: SearchState,
) -> Literal["execute_search", "execute_agent_search"]:
    """Conditional edge after ``route_decision``: dispatch to the chosen arm."""
    route = state.get("route") or {}
    return "execute_agent_search" if route.get("arm") == "agent" else "execute_search"


def _route_after_search(
    state: SearchState,
) -> Literal["fallback_search", "execute_search", "__end__"]:
    """Conditional edge after either search node.

    Three branches:

    * ``"agent_arm_failed:"`` prefix → retry as rule (``execute_search``).
      This path is only reachable from ``execute_agent_search``; the rule
      arm cannot produce this prefix, so the edge is registered only on
      the agent node's conditional in :func:`build_pipeline`.
    * Any other error → degrade to ``fallback_search``.
    * No error → ``END``.
    """
    error = state.get("error") or ""
    if error.startswith("agent_arm_failed:"):
        return "execute_search"
    if error:
        return "fallback_search"
    return "__end__"


# --------------------------------------------------------------------------- #
# Graph construction                                                           #
# --------------------------------------------------------------------------- #


def build_pipeline() -> Any:
    """Build and compile the search pipeline.

    Phase A5 topology — see module docstring for the ASCII diagram. The
    agent arm is inert until ``DegradationConfig.agentic_path_enabled``
    is flipped to True in Phase D; the ``route_decision`` node returns
    ``arm="rule"`` for every query until then.

    Returns:
        A compiled LangGraph StateGraph ready to ``ainvoke()``.
    """
    g: StateGraph = StateGraph(SearchState)
    g.add_node("parse_query", parse_query)
    g.add_node("route_decision", route_decision)
    g.add_node("execute_search", execute_search)
    g.add_node("execute_agent_search", execute_agent_search)
    g.add_node("fallback_search", fallback_search)

    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "route_decision")
    g.add_conditional_edges(
        "route_decision",
        _route_after_parse,
        {
            "execute_search": "execute_search",
            "execute_agent_search": "execute_agent_search",
        },
    )
    # Rule-arm conditional: rule errors → fallback; success → END.
    # The rule arm cannot produce an ``agent_arm_failed:`` error, so we
    # do NOT register the retry-as-rule edge here — it would be dead
    # code and introduce a confusing self-cycle.
    g.add_conditional_edges(
        "execute_search",
        _route_after_search,
        {"fallback_search": "fallback_search", "__end__": END},
    )
    # Agent-arm conditional: agent_arm_failed → retry as rule via
    # execute_search; other errors → fallback; success → END.
    g.add_conditional_edges(
        "execute_agent_search",
        _route_after_search,
        {
            "execute_search": "execute_search",
            "fallback_search": "fallback_search",
            "__end__": END,
        },
    )
    g.add_edge("fallback_search", END)

    return g.compile()


# --------------------------------------------------------------------------- #
# Public async entry point                                                     #
# --------------------------------------------------------------------------- #


async def search(
    patient_description: str,
    pipeline: Any,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Run the pipeline on a patient query with a wall-clock budget.

    Args:
        patient_description: Free-text patient query.
        pipeline: Compiled graph from :func:`build_pipeline`.
        timeout: Wall-clock cap in seconds. On timeout we return a
            degraded response, never raise.

    Returns:
        Dict with keys ``patient_profile``, ``search_results``,
        ``agent_trace``, ``used_fallback``, ``error``, ``elapsed_ms``,
        and ``route`` (``{"arm", "reason", "signals"}`` from the
        :func:`route_decision` node — ``None`` if the pipeline never
        reached that node, e.g. on the outer timeout / exception paths).
    """
    initial: SearchState = {
        "raw_query": patient_description,
        "patient_profile": None,
        "search_results": None,
        "agent_trace": [],
        "error": None,
        "used_fallback": False,
        "route": None,
    }

    t0 = time.perf_counter()
    try:
        final = await asyncio.wait_for(pipeline.ainvoke(initial), timeout=timeout)
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timeout_reason = f"pipeline exceeded {timeout}s budget"
        logger.warning("%s — invoking outer fallback", timeout_reason)
        record_agent_failure("timeout")

        # Funnel into the same recovery the inner bubble-fix uses. The
        # outer wait_for has cancelled the LangGraph task, so we can't
        # re-enter the graph — invoke the fallback helper directly. If
        # the fallback itself fails, ``_run_fallback_search_for_query``
        # already returns a graceful ``results=[]`` + error envelope.
        fb = await _run_fallback_search_for_query(patient_description, timeout_reason)
        n_results = len((fb.get("search_results") or {}).get("results") or [])
        timeout_trace = [
            {
                "step": "timeout",
                "duration_ms": round(elapsed_ms, 2),
                "decisions": {"timeout_s": timeout},
            }
        ] + (fb.get("agent_trace") or [])

        _persist_trace_safely(
            query_id=str(uuid.uuid4()),
            raw_query=patient_description,
            # Pipeline never reached ``route_decision``, so we have no
            # arm assignment yet. ``unknown`` is the catch-all label.
            arm="unknown",
            route_reason=None,
            route_signals=None,
            pipeline_kind="outer_timeout_fallback",
            used_fallback=True,
            total_ms=round(elapsed_ms, 2),
            n_results=n_results,
            error=timeout_reason,
            stages=timeout_trace,
        )
        return {
            "patient_profile": None,
            "search_results": fb["search_results"],
            "agent_trace": timeout_trace,
            "used_fallback": True,
            "error": timeout_reason,
            "elapsed_ms": round(elapsed_ms, 2),
            "route": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("Pipeline raised unexpectedly")
        record_agent_failure("pipeline")
        error_str = f"{type(exc).__name__}: {exc}"

        # Mirror the timeout path: try the outer fallback so the user
        # still gets results when an unexpected exception escapes the
        # LangGraph. Same rationale, same helper.
        fb = await _run_fallback_search_for_query(patient_description, error_str)
        n_results = len((fb.get("search_results") or {}).get("results") or [])
        error_trace = [
            {
                "step": "pipeline_error",
                "duration_ms": round(elapsed_ms, 2),
                "decisions": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
        ] + (fb.get("agent_trace") or [])

        _persist_trace_safely(
            query_id=str(uuid.uuid4()),
            raw_query=patient_description,
            arm="unknown",
            route_reason=None,
            route_signals=None,
            pipeline_kind="outer_exception_fallback",
            used_fallback=True,
            total_ms=round(elapsed_ms, 2),
            n_results=n_results,
            error=error_str,
            stages=error_trace,
        )
        return {
            "patient_profile": None,
            "search_results": fb["search_results"],
            "agent_trace": error_trace,
            "used_fallback": True,
            "error": error_str,
            "elapsed_ms": round(elapsed_ms, 2),
            "route": None,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    response = {
        "patient_profile": final.get("patient_profile"),
        "search_results": final.get("search_results"),
        "agent_trace": final.get("agent_trace") or [],
        "used_fallback": final.get("used_fallback", False),
        "error": final.get("error"),
        "elapsed_ms": round(elapsed_ms, 2),
        "route": final.get("route"),
    }

    _persist_trace_safely(
        query_id=str(uuid.uuid4()),
        raw_query=patient_description,
        arm=(response["route"] or {}).get("arm", "unknown"),
        route_reason=(response["route"] or {}).get("reason"),
        route_signals=(response["route"] or {}).get("signals"),
        pipeline_kind=_extract_pipeline_kind(response["agent_trace"]),
        used_fallback=response["used_fallback"],
        total_ms=round(elapsed_ms, 2),
        n_results=len(((response["search_results"] or {}).get("results")) or []),
        error=response["error"],
        stages=response["agent_trace"],
    )
    return response


__all__ = [
    "SearchState",
    "build_pipeline",
    "search",
    "DEFAULT_TIMEOUT",
    "DEFAULT_FALLBACK_TOP_K",
]
