"""End-to-end LangGraph pipeline: QueryParser → SearchOrchestrator → fallback.

A small, deterministic graph:

    START → parse_query → execute_search ─┬─→ END                (success)
                                          └─→ fallback_search → END  (on exception)

The graph carries a :class:`SearchState` ``TypedDict``. ``agent_trace`` is the
primary observability surface — every node appends one or more structured
entries via the ``operator.add`` reducer.

Wall-clock budget is enforced by :func:`search` via ``asyncio.wait_for``; on
timeout we return a degraded response rather than letting the caller hang.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from TrialMine.agents.orchestrator import SearchOrchestrator
from TrialMine.agents.query_parser import PatientProfile, QueryParserAgent
from TrialMine.monitoring import record_agent_failure, time_stage_cm

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 15.0
DEFAULT_FALLBACK_TOP_K = 20


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


# --------------------------------------------------------------------------- #
# Lazy singletons                                                             #
# --------------------------------------------------------------------------- #

_parser_singleton: QueryParserAgent | None = None
_orchestrator_singleton: SearchOrchestrator | None = None


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
    }


async def fallback_search(state: SearchState) -> dict:
    """Degraded path: plain hybrid search via :meth:`HybridRetriever.search`.

    No re-ranking, no eligibility check, no template explanations — just
    BM25 + semantic + RRF. Itself wrapped in try/except so even an ES
    outage doesn't propagate; the caller gets ``results=[]`` plus a clear
    ``error`` reason.
    """
    t0 = time.perf_counter()
    raw_query = state["raw_query"]
    reason = state.get("error") or "execute_search failed"

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


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #


def _route_after_search(
    state: SearchState,
) -> Literal["fallback_search", "__end__"]:
    """Conditional edge: route to fallback_search when execute_search set an error."""
    if state.get("error"):
        return "fallback_search"
    return "__end__"


# --------------------------------------------------------------------------- #
# Graph construction                                                           #
# --------------------------------------------------------------------------- #


def build_pipeline() -> Any:
    """Build and compile the search pipeline.

    Returns:
        A compiled LangGraph StateGraph ready to ``ainvoke()``.
    """
    g: StateGraph = StateGraph(SearchState)
    g.add_node("parse_query", parse_query)
    g.add_node("execute_search", execute_search)
    g.add_node("fallback_search", fallback_search)

    g.set_entry_point("parse_query")
    g.add_edge("parse_query", "execute_search")
    g.add_conditional_edges(
        "execute_search",
        _route_after_search,
        {"fallback_search": "fallback_search", "__end__": END},
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
        ``agent_trace``, ``used_fallback``, ``error``, ``elapsed_ms``.
    """
    initial: SearchState = {
        "raw_query": patient_description,
        "patient_profile": None,
        "search_results": None,
        "agent_trace": [],
        "error": None,
        "used_fallback": False,
    }

    t0 = time.perf_counter()
    try:
        final = await asyncio.wait_for(pipeline.ainvoke(initial), timeout=timeout)
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning("Pipeline exceeded %.1fs budget — returning degraded response", timeout)
        record_agent_failure("timeout")
        return {
            "patient_profile": None,
            "search_results": {
                "results": [],
                "query_used": patient_description,
                "filters": {},
                "normalized_condition": None,
            },
            "agent_trace": [
                {
                    "step": "timeout",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {"timeout_s": timeout},
                }
            ],
            "used_fallback": True,
            "error": f"pipeline exceeded {timeout}s budget",
            "elapsed_ms": round(elapsed_ms, 2),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("Pipeline raised unexpectedly")
        record_agent_failure("pipeline")
        return {
            "patient_profile": None,
            "search_results": {
                "results": [],
                "query_used": patient_description,
                "filters": {},
                "normalized_condition": None,
            },
            "agent_trace": [
                {
                    "step": "pipeline_error",
                    "duration_ms": round(elapsed_ms, 2),
                    "decisions": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
            ],
            "used_fallback": True,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round(elapsed_ms, 2),
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "patient_profile": final.get("patient_profile"),
        "search_results": final.get("search_results"),
        "agent_trace": final.get("agent_trace") or [],
        "used_fallback": final.get("used_fallback", False),
        "error": final.get("error"),
        "elapsed_ms": round(elapsed_ms, 2),
    }


__all__ = [
    "SearchState",
    "build_pipeline",
    "search",
    "DEFAULT_TIMEOUT",
    "DEFAULT_FALLBACK_TOP_K",
]
