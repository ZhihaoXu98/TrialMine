"""Integration tests for the LangGraph ``route_decision`` node.

The pure routing function lives in :mod:`TrialMine.agents.query_router`
and its unit tests stay in :mod:`tests.unit.test_query_router` — those
tests pin behaviour assuming no IO. The Phase-A7 node composes the pure
function with the AB router (which has in-memory state + an exposure
log), so its tests live here rather than in the unit suite.

Three invariants pinned:

* AB-router gate is skipped when ``agentic_path_enabled=False`` — the
  Phase A–C default. Pure A3 heuristic answers, production behaviour
  byte-identical to pre-Phase-A.
* With ``agentic_path_enabled=True`` AND the experiment in its catalog
  default (``enabled=False``), every subject lands in ``"control"`` →
  ``arm="rule"`` / ``reason="ab_router_control"``. Calling the node
  twice with the same query yields the same arm (the runbook's
  "determinism" check).
* With the experiment forcibly enabled the hash bucketing is exercised
  — two calls with the same ``raw_query`` still yield the same variant.
"""

from __future__ import annotations

import asyncio

import pytest

import TrialMine.config as cfg_mod
import TrialMine.experiments.ab_test as ab
from TrialMine.agents.pipeline import route_decision
from TrialMine.agents.query_parser import PatientProfile


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    """Reset module-level singletons + degradation so tests are isolated."""
    saved_router = ab._default_router
    saved_degradation = cfg_mod._DEFAULT_DEGRADATION
    ab._default_router = None
    yield
    ab._default_router = saved_router
    cfg_mod._DEFAULT_DEGRADATION = saved_degradation


def _state(profile: PatientProfile) -> dict:
    """Build a minimal SearchState dict the node accepts."""
    return {
        "raw_query": profile.raw_query,
        "patient_profile": profile.model_dump(),
    }


# --------------------------------------------------------------------------- #
# Layer 1 skipped — Phase A invariant under agentic_path_enabled=False        #
# --------------------------------------------------------------------------- #


async def test_agent_path_disabled_falls_through_to_pure_router() -> None:
    """Default config: AB gate is skipped, pure A3 returns rule arm with
    reason="agentic_path_disabled". Pinning this guarantees production
    behaviour is byte-identical to pre-Phase-A through Phase C."""
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=False,
    )
    profile = PatientProfile(
        raw_query="failed osimertinib EGFR T790M",
        prior_treatments=["osimertinib"],
        biomarkers=["EGFR T790M"],
        condition="non-small cell lung cancer",
    )
    result = await route_decision(_state(profile))  # type: ignore[arg-type]
    assert result["route"]["arm"] == "rule"
    assert result["route"]["reason"] == "agentic_path_disabled"
    # No ab-router signals when the gate is skipped.
    assert "subject_id" not in result["route"]["signals"]
    assert "variant" not in result["route"]["signals"]


# --------------------------------------------------------------------------- #
# Layer 1 fires, experiment disabled — Phase A default for the AB router     #
# --------------------------------------------------------------------------- #


async def test_ab_router_control_returns_rule_with_correct_reason() -> None:
    """Runbook test #2. With ``agentic_path_enabled=True`` but the
    experiment in its catalog default (``enabled=False``), every subject
    lands in the first variant ("control") and the node short-circuits
    to ``arm="rule"`` / ``reason="ab_router_control"``."""
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=True,
    )
    profile = PatientProfile(raw_query="trials for my dad")
    result = await route_decision(_state(profile))  # type: ignore[arg-type]
    assert result["route"]["arm"] == "rule"
    assert result["route"]["reason"] == "ab_router_control"
    sig = result["route"]["signals"]
    assert sig["variant"] == "control"
    assert isinstance(sig["subject_id"], str) and len(sig["subject_id"]) == 16


async def test_same_query_yields_same_arm_across_two_calls_disabled_experiment() -> None:
    """Runbook test #1 — disabled-experiment variant. Two calls with the
    same query must yield the same arm + reason + signals. Trivially
    true under the disabled-short-circuit, but worth pinning because a
    future change to the AB-router internals could break it."""
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=True,
    )
    profile = PatientProfile(raw_query="very specific deterministic query")
    r1 = await route_decision(_state(profile))  # type: ignore[arg-type]
    r2 = await route_decision(_state(profile))  # type: ignore[arg-type]
    assert r1["route"]["arm"] == r2["route"]["arm"]
    assert r1["route"]["reason"] == r2["route"]["reason"]
    assert r1["route"]["signals"] == r2["route"]["signals"]


# --------------------------------------------------------------------------- #
# Layer 1 fires AND the experiment is enabled — hash-bucketing determinism   #
# --------------------------------------------------------------------------- #


async def test_same_query_deterministic_under_enabled_experiment() -> None:
    """Runbook test #1 — enabled-experiment variant. Force the
    experiment on and confirm hash bucketing is deterministic across
    two calls with the same ``raw_query``. Variant arm might be either
    'control' or 'treatment' depending on the bucket; the contract is
    that it's the SAME across calls."""
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=True,
    )
    # Build the singleton then force the experiment on for this test.
    router = ab.get_default_router()
    router._catalog["agent_path_v1"].enabled = True

    profile = PatientProfile(raw_query="hash determinism probe query")
    r1 = await route_decision(_state(profile))  # type: ignore[arg-type]
    r2 = await route_decision(_state(profile))  # type: ignore[arg-type]
    assert r1["route"]["arm"] == r2["route"]["arm"]
    assert r1["route"]["reason"] == r2["route"]["reason"]


async def test_exposure_log_records_every_ab_gated_call() -> None:
    """The exposure log is the join key for outcomes analysis. Each
    call that reaches the AB-router gate must append exactly one
    exposure record. (Calls under ``agentic_path_enabled=False`` skip
    the gate entirely and should NOT log.)"""
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=True,
    )
    profile = PatientProfile(raw_query="exposure probe")
    router_before = ab.get_default_router()
    n_before = len(router_before.exposures)
    await route_decision(_state(profile))  # type: ignore[arg-type]
    await route_decision(_state(profile))  # type: ignore[arg-type]
    n_after = len(router_before.exposures)
    assert n_after - n_before == 2, "expected exactly one exposure per call"


async def test_skipped_gate_does_not_pollute_exposure_log() -> None:
    """Under ``agentic_path_enabled=False`` the AB-router gate is
    skipped — no exposures should be recorded."""
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=False,
    )
    profile = PatientProfile(raw_query="should not log exposure")
    router_before = ab.get_default_router()
    n_before = len(router_before.exposures)
    await route_decision(_state(profile))  # type: ignore[arg-type]
    n_after = len(router_before.exposures)
    assert n_after == n_before, "AB gate was skipped — no exposure expected"


# --------------------------------------------------------------------------- #
# Phase C3b.1 — degraded agent result_dict routes to rule retry              #
# --------------------------------------------------------------------------- #


async def test_degraded_agent_result_dict_routes_to_rule_retry() -> None:
    """When AgenticSearchAgent.search() returns a degraded result_dict
    (results=[], error="agent_did_not_terminate") instead of raising,
    the bubble-fix in execute_agent_search must elevate the inner error
    to ``state["error"]`` with the ``agent_arm_failed:`` prefix so the
    retry-as-rule edge fires and the rule arm produces results.

    Before the fix, this path returned ``search_results.results=[]``
    to the user — exactly the 18 dropped queries in C2.
    """
    from unittest.mock import patch

    import TrialMine.agents.pipeline as pl

    # Force the agent path on so route_decision dispatches to the agent.
    cfg_mod._DEFAULT_DEGRADATION = cfg_mod.DegradationConfig(
        agentic_path_enabled=True,
    )

    # Force AB router to 100% treatment so we don't accidentally bucket
    # into control and skip the agent arm entirely.
    ab._default_router = None
    router = ab.get_default_router()
    router._catalog["agent_path_v1"].enabled = True
    router._catalog["agent_path_v1"].variants[0].weight = 0.0
    router._catalog["agent_path_v1"].variants[1].weight = 1.0

    # Mock agent: returns a degraded result_dict (the A4.2 contract for
    # timeout / no-terminator / parse failure inside search()).
    class _DegradedAgent:
        async def search(self, profile):
            return (
                {
                    "results": [],
                    "query_used": profile.raw_query,
                    "filters": {},
                    "normalized_condition": None,
                    "error": "agent_did_not_terminate",
                },
                [
                    {
                        "step": "agent_start",
                        "duration_ms": 0.0,
                        "decisions": {"model": "mock", "max_iters": 5},
                    },
                    {
                        "step": "agent_no_terminator",
                        "duration_ms": 30_000.0,
                        "decisions": {
                            "iters_used": 5,
                            "error": "agent_did_not_terminate",
                        },
                    },
                ],
            )

    pl._agent_singleton = _DegradedAgent()

    # Mock rule arm (execute_search) so we can detect that the retry
    # fired AND verify the user gets non-empty results.
    async def _fake_rule_search(state: dict) -> dict:
        return {
            "search_results": {
                "results": [{"nct_id": "NCT-RULE-RETRY-OK"}],
                "query_used": state["raw_query"],
                "filters": {},
                "normalized_condition": None,
            },
            "agent_trace": [
                {
                    "step": "execute_search",
                    "duration_ms": 5.0,
                    "decisions": {"pipeline": "rule_retry_stub"},
                }
            ],
            "error": None,
        }

    # Mock parse_query so the test doesn't hit the real Anthropic API.
    async def _fake_parse(state: dict) -> dict:
        return {
            "patient_profile": {"raw_query": state["raw_query"]},
            "agent_trace": [
                {
                    "step": "parse_query",
                    "duration_ms": 0.5,
                    "decisions": {"n_slots_populated": 0},
                }
            ],
        }

    async def _fake_fallback(state: dict) -> dict:
        return {
            "search_results": {
                "results": [{"nct_id": "NCT-FALLBACK"}],
                "query_used": state["raw_query"],
                "filters": {},
                "normalized_condition": None,
            },
            "used_fallback": True,
            "agent_trace": [
                {
                    "step": "fallback_search",
                    "duration_ms": 1.0,
                    "decisions": {"reason": "should NOT be reached on degraded path"},
                }
            ],
        }

    with (
        patch.object(pl, "parse_query", _fake_parse),
        patch.object(pl, "execute_search", _fake_rule_search),
        patch.object(pl, "fallback_search", _fake_fallback),
    ):
        graph = pl.build_pipeline()
        final = await graph.ainvoke(
            {
                "raw_query": "test no-terminator routes to rule retry",
                "patient_profile": None,
                "search_results": None,
                "agent_trace": [],
                "error": None,
                "used_fallback": False,
                "route": None,
            }
        )

    # 1. The retry-as-rule path must have fired — final results come from
    #    the rule arm's execute_search, NOT from the agent's empty list
    #    AND NOT from fallback_search (which is for non-agent errors).
    results = (final.get("search_results") or {}).get("results") or []
    assert len(results) == 1
    assert results[0]["nct_id"] == "NCT-RULE-RETRY-OK", (
        "expected rule-arm retry result, not "
        f"{results[0]['nct_id']!r} — bubble fix did not route correctly"
    )
    # PR-3 update: bubble-fix now sets used_fallback=True so the trace
    # store row honestly reflects that the agent failed and the rule
    # arm became the fallback. The fallback_search NODE still doesn't
    # fire (verified by trace_steps below) — the flag is broader than
    # "fallback_search ran" now; it means "some recovery path engaged".
    assert final.get("used_fallback") is True, (
        "bubble-fix retry must mark used_fallback=True so the trace "
        "store distinguishes agent-failed-rule-recovered runs from "
        "clean agent successes"
    )

    # 2. The agent's no-terminator trace + the bubble-fix's
    #    execute_agent_search_degraded entry + the rule retry's
    #    execute_search entry must all be present in agent_trace.
    trace_steps = [e.get("step") for e in (final.get("agent_trace") or [])]
    assert "agent_no_terminator" in trace_steps, f"agent's own trace lost; got steps: {trace_steps}"
    assert "execute_agent_search_degraded" in trace_steps, (
        f"bubble-fix trace entry missing; got steps: {trace_steps}"
    )
    assert "execute_search" in trace_steps, (
        f"rule retry trace entry missing; got steps: {trace_steps}"
    )

    # 3. The final state's error should be CLEARED (rule arm cleared it
    #    on its success return per B3). Otherwise the conditional edge
    #    would re-loop forever.
    assert final.get("error") is None, (
        f"state.error should be cleared after rule retry; got {final.get('error')!r}"
    )


# --------------------------------------------------------------------------- #
# PR — outer-timeout + outer-exception recovery (asymmetry fix)               #
# --------------------------------------------------------------------------- #


async def test_outer_timeout_returns_fallback_results_not_empty() -> None:
    """When the outer ``asyncio.wait_for`` cap fires in pipeline.search(),
    the handler must call the outer fallback helper so the user gets
    rule-arm results instead of ``results=[]``. Pins the asymmetry-fix
    delivered alongside the bubble-fix ``used_fallback`` change.

    Before the fix, the timeout handler short-circuited to an empty
    result list with ``error: 'pipeline exceeded ...s budget'``, which
    matched the on-paper UX promise but never honored it — the inner
    bubble-fix path retried-as-rule, the outer-timeout path didn't.
    """
    from unittest.mock import patch

    import TrialMine.agents.pipeline as pl

    # Stub the fallback helper so we don't need ES/FAISS. The real
    # helper is exercised in production tests; here we pin the wiring.
    async def _stub_fallback(raw_query: str, reason: str) -> dict:
        return {
            "search_results": {
                "results": [{"nct_id": "NCT-OUTER-TIMEOUT-FALLBACK"}],
                "query_used": raw_query,
                "filters": {},
                "normalized_condition": None,
            },
            "used_fallback": True,
            "agent_trace": [
                {
                    "step": "fallback_search",
                    "duration_ms": 1.0,
                    "decisions": {"reason": reason, "n_results": 1},
                }
            ],
        }

    # Build a fake pipeline object whose ainvoke just hangs past the cap.
    class _HangingPipeline:
        async def ainvoke(self, state, config=None):
            await asyncio.sleep(10)  # well past the 0.05s test cap
            return state

    with patch.object(pl, "_run_fallback_search_for_query", _stub_fallback):
        result = await pl.search(
            "test outer timeout falls back",
            _HangingPipeline(),
            timeout=0.05,  # forces TimeoutError quickly
        )

    results = (result.get("search_results") or {}).get("results") or []
    assert len(results) == 1, f"expected 1 fallback result, got {len(results)}: {results}"
    assert results[0]["nct_id"] == "NCT-OUTER-TIMEOUT-FALLBACK", (
        f"outer-timeout handler must call the fallback helper; got {results[0]!r}"
    )
    assert result.get("used_fallback") is True
    assert "pipeline exceeded" in (result.get("error") or ""), (
        f"error should preserve the timeout reason; got {result.get('error')!r}"
    )
    # The outer-fallback fingerprint in the trace stream.
    trace_steps = [e.get("step") for e in (result.get("agent_trace") or [])]
    assert "timeout" in trace_steps
    assert "fallback_search" in trace_steps, (
        f"fallback trace must be stitched onto outer-timeout trace; got {trace_steps}"
    )


async def test_outer_exception_returns_fallback_results_not_empty() -> None:
    """Mirror of the outer-timeout test for the generic-Exception branch.
    Same recovery contract: an unexpected error escaping the LangGraph
    must funnel into the outer fallback helper so the user still gets
    rule-arm results.
    """
    from unittest.mock import patch

    import TrialMine.agents.pipeline as pl

    async def _stub_fallback(raw_query: str, reason: str) -> dict:
        return {
            "search_results": {
                "results": [{"nct_id": "NCT-OUTER-EXCEPTION-FALLBACK"}],
                "query_used": raw_query,
                "filters": {},
                "normalized_condition": None,
            },
            "used_fallback": True,
            "agent_trace": [
                {
                    "step": "fallback_search",
                    "duration_ms": 1.0,
                    "decisions": {"reason": reason, "n_results": 1},
                }
            ],
        }

    class _ExplodingPipeline:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("synthetic pipeline blow-up")

    with patch.object(pl, "_run_fallback_search_for_query", _stub_fallback):
        result = await pl.search(
            "test outer exception falls back",
            _ExplodingPipeline(),
            timeout=5.0,
        )

    results = (result.get("search_results") or {}).get("results") or []
    assert len(results) == 1
    assert results[0]["nct_id"] == "NCT-OUTER-EXCEPTION-FALLBACK"
    assert result.get("used_fallback") is True
    err = result.get("error") or ""
    assert "synthetic pipeline blow-up" in err, (
        f"error should preserve the inner exception message; got {err!r}"
    )
    trace_steps = [e.get("step") for e in (result.get("agent_trace") or [])]
    assert "pipeline_error" in trace_steps
    assert "fallback_search" in trace_steps
