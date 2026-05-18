"""Phase B6 — observability smoke: 20 mixed queries to populate the dashboard.

Unlike A8 (which force-routed 100 % of traffic to the agent arm to
validate the loop end-to-end), B6 enables the AB router with its
**natural 50/50 weights** so the dashboard shows realistic
production-shaped data: ~half the queries land in ``control`` (rule
arm) and the other half land in ``treatment`` where the heuristic
further filters to agent vs rule based on profile shape.

The 20 queries split intentionally:

* 10 "vague / complex" — sparse profiles or multi-constraint phrasings
  that the A3 heuristic routes to the agent arm if they hit treatment.
* 10 "easy explicit" — single-axis cancer-type queries that the
  heuristic routes to rule even on treatment (no sparse / no
  complex_pattern trigger).

Expected dashboard outcome:

* Panel 1 shows both arms — split roughly 50/50 on a histogram basis
  (within ±2 due to hash-bucket variance over n=20).
* Panel 3 populates with both rule-arm stages AND agent-arm stages
  (``agent_start`` / ``tool_call`` / ``agent_submit``).
* Panel 4 populates with ≥3 distinct tool names, ok_count > 0.
* Panel 2 stays near 0 (we shouldn't be falling back).

Cost budget: ~$0.20–$0.60 (5–10 agent queries × ~$0.05 each;
rule queries free).

Reproducibility::

    docker start trialmine-es && sleep 5
    OMP_NUM_THREADS=1 python scripts/observability_smoke.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

# Load ANTHROPIC_API_KEY from .env BEFORE the heavy imports.
from TrialMine.config import DegradationConfig, Settings

_settings = Settings()  # type: ignore[call-arg]
os.environ.setdefault("ANTHROPIC_API_KEY", _settings.anthropic_api_key)

import TrialMine.config as cfg_mod  # noqa: E402

# Agent path ON + the A8-tuned wider per-iter timeout and outer budget.
cfg_mod._DEFAULT_DEGRADATION = DegradationConfig(
    agentic_path_enabled=True,
    agentic_routing_threshold_slots=2,
    agentic_complex_pattern_enabled=True,
    agentic_max_iters=5,
    agentic_per_iter_timeout_s=12.0,
    skip_agent_if_slow_s=90.0,
)

# Enable the AB router experiment at its **natural 50/50 weights** so
# the dashboard exhibits a realistic arm mix (not the A8 force-treatment
# pattern). Hash-bucketing of raw_query determines which queries land
# in which bucket.
import TrialMine.experiments.ab_test as ab  # noqa: E402

ab._default_router = None
_router = ab.get_default_router()
_router._catalog["agent_path_v1"].enabled = True
# Leaving variants at the default (control=0.5, treatment=0.5).

# Reset pipeline's agent singleton so the new config is picked up.
import TrialMine.agents.pipeline as pl  # noqa: E402

pl._agent_singleton = None

from TrialMine.agents.pipeline import build_pipeline, search  # noqa: E402

# --------------------------------------------------------------------------- #
# Queries — 10 agent-targeted + 10 rule-targeted                              #
# --------------------------------------------------------------------------- #

AGENT_TARGETED: list[str] = [
    # Vague (sparse_profile candidates) ----------------------------------
    "trials for my dad",
    "trials for my mom with breast cancer",
    "looking for trials for grandfather",
    "any trials available?",
    "studies for my partner",
    # Complex (failed_X_after_Y or multi_constraint candidates) ----------
    "failed osimertinib EGFR T790M NSCLC",
    "BRCA1 mutation, ovarian cancer, prior PARP inhibitor",
    "metastatic HER2-positive breast cancer after trastuzumab",
    "stage IV pancreatic cancer post-FOLFIRINOX",
    "MSI-high colorectal cancer refractory to pembrolizumab",
]

RULE_TARGETED: list[str] = [
    "lung cancer trials",
    "breast cancer trials in Boston",
    "prostate cancer clinical trials",
    "pediatric leukemia trials",
    "melanoma immunotherapy trials",
    "glioblastoma trials phase 3",
    "lymphoma trials recruiting",
    "ovarian cancer phase 2 trials",
    "colorectal cancer trials phase 2",
    "kidney cancer immunotherapy",
]

QUERIES: list[str] = AGENT_TARGETED + RULE_TARGETED


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


async def _run_one(graph, query: str) -> dict:
    t0 = time.perf_counter()
    final = await search(query, graph, timeout=90.0)
    wall_ms = (time.perf_counter() - t0) * 1000

    route = final.get("route") or {}
    return {
        "query": query,
        "arm": route.get("arm"),
        "reason": route.get("reason"),
        "n_results": len(
            (final.get("search_results") or {}).get("results") or []
        ),
        "latency_ms": round(wall_ms, 1),
        "used_fallback": final.get("used_fallback"),
        "error": final.get("error"),
    }


def main() -> int:
    graph = build_pipeline()
    print(f"\n{'=' * 78}")
    print("Phase B6 — observability smoke (20 mixed queries)")
    print(f"{'=' * 78}")
    cfg = cfg_mod.get_default_degradation()
    print(
        f"Config: agentic_path_enabled={cfg.agentic_path_enabled}  "
        f"max_iters={cfg.agentic_max_iters}  "
        f"per_iter_timeout_s={cfg.agentic_per_iter_timeout_s}"
    )
    print("AB router: experiment enabled at natural 50/50 weights.\n")

    results: list[dict] = []
    for i, q in enumerate(QUERIES, start=1):
        category = "agent-targeted" if q in AGENT_TARGETED else "rule-targeted"
        print(f"[{i:2d}/20] ({category:14s}) {q!r}")
        try:
            r = asyncio.run(_run_one(graph, q))
        except Exception as exc:  # noqa: BLE001
            logger.exception("query %d raised", i)
            r = {
                "query": q,
                "arm": None,
                "reason": None,
                "n_results": 0,
                "latency_ms": 0.0,
                "used_fallback": False,
                "error": f"uncaught {type(exc).__name__}: {exc}",
            }
        results.append(r)
        print(
            f"        → arm={r['arm']:<6} reason={r['reason']!r:<28}  "
            f"n_results={r['n_results']:<3} latency={r['latency_ms']:.0f}ms"
        )
        if r["error"]:
            print(f"        ⚠ {r['error']}")

    # Aggregate
    n_agent = sum(1 for r in results if r["arm"] == "agent")
    n_rule = sum(1 for r in results if r["arm"] == "rule")
    n_fallback = sum(1 for r in results if r["used_fallback"])
    n_err = sum(1 for r in results if r["error"])

    print(f"\n{'=' * 78}")
    print("AGGREGATE")
    print(f"{'=' * 78}")
    print(f"  agent_arm:     {n_agent}/20")
    print(f"  rule_arm:      {n_rule}/20")
    print(f"  fallback hits: {n_fallback}/20")
    print(f"  errors:        {n_err}/20")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
