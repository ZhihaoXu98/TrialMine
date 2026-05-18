"""Phase A8 smoke — end-to-end agent path on the live Haiku 4.5 API.

Drives :func:`TrialMine.agents.pipeline.search` on five hard-coded
agent-targeted queries with the AB router's *treatment* variant forced
to 100 % weight (so every query reaches the A3 heuristic + agent loop
where applicable, regardless of hash bucketing). Reports per-query arm,
iter count, tool sequence, latency, cost, and rolls up aggregate stats.

Why the AB router is force-pinned to treatment: the production catalog
splits 50/50 between control (rule arm) and treatment (heuristic →
maybe agent). With only five queries that random split would leave us
testing the agent on 1–3 of them — not enough to validate the loop
shape. We override weights to (control=0, treatment=1) so the heuristic
fires on every query; the agent vs rule decision is then made purely on
profile shape.

Reproducibility::

    docker start trialmine-es && sleep 5
    OMP_NUM_THREADS=1 python scripts/test_agent_path.py

Cost budget: ≤$1.00 for 5 queries at Haiku 4.5 published rates
($1/M input, $5/M output). Real measurement target is ~$0.05–0.15.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

# Load ANTHROPIC_API_KEY from .env BEFORE the heavy imports — the agent's
# constructor reads it at instantiation, and pipeline.py's lazy
# ``_get_agent()`` reaches for ``os.environ[...]``, not for Settings().
from TrialMine.config import DegradationConfig, Settings

_settings = Settings()  # type: ignore[call-arg]
os.environ.setdefault("ANTHROPIC_API_KEY", _settings.anthropic_api_key)

import TrialMine.config as cfg_mod  # noqa: E402

# Phase A8 — agent path ON, threshold defaults, outer pipeline budget
# widened so the agent has 60s of wall-clock (vs the 10s production
# default that would cancel ~half of agent runs mid-loop).
cfg_mod._DEFAULT_DEGRADATION = DegradationConfig(
    agentic_path_enabled=True,
    agentic_routing_threshold_slots=2,
    agentic_complex_pattern_enabled=True,
    agentic_max_iters=5,
    # Bumped from the production 6.0s default — for the smoke we want
    # the FULL 5-cycle loop to fit without prematurely cancelling. The
    # production default works because (a) Anthropic responses are
    # typically <6s when the prompt is prompt-cached, and (b) the
    # outer skip_agent_if_slow_s cap is the binding constraint in
    # production anyway.
    agentic_per_iter_timeout_s=12.0,
    skip_agent_if_slow_s=90.0,
)

# Force every query to the heuristic by giving "treatment" 100 % weight.
# Mutating the catalog before ``build_pipeline()`` so the first
# ``route_decision`` call already sees the override.
import TrialMine.experiments.ab_test as ab  # noqa: E402

ab._default_router = None  # discard whatever a prior import built
_router = ab.get_default_router()
_router._catalog["agent_path_v1"].enabled = True
_router._catalog["agent_path_v1"].variants[0].weight = 0.0  # control
_router._catalog["agent_path_v1"].variants[1].weight = 1.0  # treatment

# Pipeline holds its own lazy agent singleton — reset so the first
# ``_get_agent()`` call picks up the new degradation config.
import TrialMine.agents.pipeline as pl  # noqa: E402

pl._agent_singleton = None

from TrialMine.agents.pipeline import build_pipeline, search  # noqa: E402

# --------------------------------------------------------------------------- #
# Five canonical A8 queries (a–e per the runbook)                             #
# --------------------------------------------------------------------------- #

QUERIES: list[str] = [
    "failed osimertinib EGFR T790M NSCLC, looking for trials",        # a — complex 3a + 3b
    "trials for my dad with prostate cancer",                          # b — vague, sparse
    "BRCA1 mutation, ovarian cancer, prior PARP inhibitor",            # c — complex 3b
    "metastatic HER2-positive breast cancer after trastuzumab",        # d — complex 3a
    "pediatric acute lymphoblastic leukemia trials",                   # e — single-axis
]

# Haiku 4.5 published rates — duplicated here so the per-query cost
# attribution in this script is self-contained (the canonical rates
# live in monitoring/metrics.py:_MODEL_PRICING_USD_PER_TOKEN).
_HAIKU_IN_RATE = 1.0 / 1_000_000
_HAIKU_OUT_RATE = 5.0 / 1_000_000

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _extract(trace: list[dict], step: str) -> dict | None:
    for entry in trace:
        if entry.get("step") == step:
            return entry
    return None


def _tool_sequence(trace: list[dict]) -> list[str]:
    return [
        entry["decisions"]["tool"]
        for entry in trace
        if entry.get("step") == "tool_call"
        and isinstance(entry.get("decisions"), dict)
        and "tool" in entry["decisions"]
    ]


async def _run_one(graph, query: str) -> dict:
    t0 = time.perf_counter()
    final = await search(query, graph, timeout=90.0)
    wall_ms = (time.perf_counter() - t0) * 1000

    trace = final.get("agent_trace") or []
    route = final.get("route") or {}
    submit = _extract(trace, "agent_submit") or {}
    submit_d = submit.get("decisions") or {}

    in_tok = int(submit_d.get("input_tokens") or 0)
    out_tok = int(submit_d.get("output_tokens") or 0)
    cost = in_tok * _HAIKU_IN_RATE + out_tok * _HAIKU_OUT_RATE

    return {
        "query": query,
        "arm": route.get("arm"),
        "reason": route.get("reason"),
        "n_results": len(((final.get("search_results") or {}).get("results")) or []),
        "iters_used": submit_d.get("iters_used"),
        "tool_sequence": _tool_sequence(trace),
        "latency_ms": round(wall_ms, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "error": final.get("error"),
        "used_fallback": final.get("used_fallback"),
        "agent_trace_steps": [e.get("step") for e in trace],
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    graph = build_pipeline()
    print(f"\n{'=' * 78}")
    print("Phase A8 — agent path smoke (5 queries, live Haiku 4.5 API)")
    print(f"{'=' * 78}")
    cfg = cfg_mod.get_default_degradation()
    print(
        f"Config: agentic_path_enabled={cfg.agentic_path_enabled}  "
        f"max_iters={cfg.agentic_max_iters}  "
        f"per_iter_timeout_s={cfg.agentic_per_iter_timeout_s}  "
        f"skip_agent_if_slow_s={cfg.skip_agent_if_slow_s}"
    )
    print("AB router: experiment forced to 100% treatment.")
    print()

    results: list[dict] = []
    for i, q in enumerate(QUERIES, start=1):
        print(f"--- Query {i}: {q!r}")
        try:
            r = asyncio.run(_run_one(graph, q))
        except Exception as exc:  # noqa: BLE001 — surface uncaught exc for the smoke
            logger.exception("query %d raised", i)
            r = {
                "query": q,
                "arm": None,
                "reason": None,
                "n_results": 0,
                "iters_used": None,
                "tool_sequence": [],
                "latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "error": f"uncaught {type(exc).__name__}: {exc}",
                "used_fallback": False,
                "agent_trace_steps": [],
            }
        results.append(r)
        print(f"    arm={r['arm']!r}  reason={r['reason']!r}")
        print(
            f"    n_results={r['n_results']}  "
            f"iters={r['iters_used']}  "
            f"latency={r['latency_ms']:.0f}ms"
        )
        print(f"    tools={r['tool_sequence']}")
        print(
            f"    tokens: in={r['input_tokens']}  out={r['output_tokens']}  "
            f"cost=${r['cost_usd']:.4f}"
        )
        if r["error"]:
            print(f"    ⚠ ERROR: {r['error']}")
        if r["used_fallback"]:
            print("    ⚠ used_fallback=True (degraded path)")
        print()

    # Aggregate
    agent_count = sum(1 for r in results if r["arm"] == "agent")
    rule_count = sum(1 for r in results if r["arm"] == "rule")
    iters = [r["iters_used"] for r in results if isinstance(r["iters_used"], int)]
    mean_iters = sum(iters) / len(iters) if iters else 0.0
    total_cost = sum(r["cost_usd"] for r in results)
    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)

    print(f"{'=' * 78}")
    print("AGGREGATE")
    print(f"{'=' * 78}")
    print(f"  agent_arm:           {agent_count}/{len(QUERIES)}")
    print(f"  rule_arm:            {rule_count}/{len(QUERIES)}")
    print(f"  mean iters (agent):  {mean_iters:.1f}")
    print(f"  total tokens:        in={total_in:,}  out={total_out:,}")
    print(f"  cumulative cost:     ${total_cost:.4f}")

    # Anomalies — fail the smoke if any uncaught exceptions, hit caps,
    # or empty agent-arm responses. The runbook's acceptance criteria
    # also require ≥3 agent-arm queries (a, c, d).
    anoms: list[str] = []
    for r in results:
        if r["error"]:
            anoms.append(f"  - {r['query']!r}: error={r['error']}")
        if r["used_fallback"]:
            anoms.append(f"  - {r['query']!r}: used_fallback=True")
        if (
            r["arm"] == "agent"
            and isinstance(r["iters_used"], int)
            and r["iters_used"] >= 5
        ):
            anoms.append(f"  - {r['query']!r}: hit max iters")
        if r["arm"] == "agent" and not r["n_results"]:
            anoms.append(f"  - {r['query']!r}: agent returned 0 results")
    if agent_count < 3:
        anoms.append(
            f"  - only {agent_count}/5 routed to agent — "
            "runbook expects ≥3 (queries a, c, d are designed to trigger heuristics)"
        )

    print("\nANOMALIES:" if anoms else "\nANOMALIES: none")
    for a in anoms:
        print(a)

    return 0 if not anoms else 1


if __name__ == "__main__":
    raise SystemExit(main())
