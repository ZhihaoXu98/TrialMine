# ruff: noqa: E501  # Markdown table cells + JSON-key strings frequently exceed 100 chars; not load-bearing for analysis scripts.
"""Phase C3 — production A/B comparison: rule arm vs agent arm.

Reads the C2 outputs (agent arm) + the pre-Phase-A labels (rule arm),
applies the A3 routing heuristic offline to each of the 85 held-out
queries, and produces ``data/evaluation/agent_vs_rule_v1.json`` plus a
markdown summary table.

Key choices that the runbook leaves open and we pin here:

* **Routing rate is the *heuristic* rate** (what the A3 pure function
  decides). The AB router's 50/50 gate is *not* applied — that would
  halve the rate in production. Both numbers are surfaced in the
  markdown so the reader doesn't conflate them.
* **The 18 agent-arm failures** (no_terminator) are handled two ways
  and reported both:
   1. ``production_strict``: failed-agent qids score NDCG@5 = 0
      (honest accounting of the cost of agent failure).
   2. ``production_with_fallback``: failed-agent qids fall back to
      the rule arm's score (matches the A5 retry-as-rule design once
      the agent's degraded result_dict bug is fixed).
  C4 should look at both — they bracket the true production NDCG.
* **Bootstrap CIs** use 1000 samples with replacement at the
  query level (mirrors ``scripts/evaluate.py``'s convention).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Make src/ importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Load .env (for ANTHROPIC_API_KEY, used by QueryParser).
from TrialMine.config import DegradationConfig, Settings  # noqa: E402

_settings = Settings()  # type: ignore[call-arg]
os.environ.setdefault("ANTHROPIC_API_KEY", _settings.anthropic_api_key)

from TrialMine.agents.query_parser import PatientProfile, QueryParserAgent  # noqa: E402
from TrialMine.agents.query_router import route_decision  # noqa: E402
from TrialMine.evaluation.metrics import mrr, ndcg_at_k  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Per-query Haiku QueryParser cost is the floor for rule-routed queries
# (the only LLM call in the rule pipeline is the parser).
_RULE_COST_PER_QUERY_USD = 0.0017

# Failed agent runs that hit the 60s no-terminator cap still consumed
# tokens — they just didn't reach submit_final_results. We attribute the
# average successful-agent cost as the failed-run estimate so the cost
# numbers honestly reflect the agent's overhead even when it produces
# nothing useful.
# (Resolved at runtime from the actual successful-run mean.)

# Bootstrap config — matches scripts/evaluate.py.
_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def group_by_qid(rows: list[dict]) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        qid = r.get("query_id")
        if qid is None:
            continue
        by[int(qid)].append(r)
    return by


def ndcg_at_k_for_query(rows: list[dict], k: int) -> float:
    """NDCG@k from a query's rows (rank + relevance fields)."""
    if not rows:
        return 0.0
    ranked = sorted(rows, key=lambda r: int(r["rank"]))
    result_ids = [str(r["nct_id"]) for r in ranked]
    relevance_scores = {str(r["nct_id"]): float(r["relevance"]) for r in ranked}
    return ndcg_at_k(result_ids, relevance_scores, k=k)


def mrr_for_query(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    ranked = sorted(rows, key=lambda r: int(r["rank"]))
    result_ids = [str(r["nct_id"]) for r in ranked]
    relevant_ids = {str(r["nct_id"]) for r in ranked if int(r["relevance"]) >= 2}
    return mrr(result_ids, relevant_ids)


def bootstrap_mean_ci(
    values: list[float],
    n_samples: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap mean + 95 % CI. Returns ``(mean, lo, hi)``."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_paired_delta_ci(
    a_values: list[float],
    b_values: list[float],
    n_samples: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap mean(a - b) with paired sampling. Returns ``(delta, lo, hi)``."""
    if not a_values or len(a_values) != len(b_values):
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr_a = np.asarray(a_values, dtype=float)
    arr_b = np.asarray(b_values, dtype=float)
    diffs = arr_a - arr_b
    n = len(diffs)
    means = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    return float(diffs.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values), pct))


# --------------------------------------------------------------------------- #
# Routing                                                                     #
# --------------------------------------------------------------------------- #


def parse_all_queries(
    queries: dict[int, tuple[str, str]],
    parser: QueryParserAgent,
) -> dict[int, PatientProfile]:
    """Parse each query into a PatientProfile via QueryParserAgent.

    The parser is the same one the production pipeline uses, so the
    routing decision matches what would happen at inference time.
    """
    profiles: dict[int, PatientProfile] = {}
    for i, (qid, (query, _cat)) in enumerate(sorted(queries.items()), start=1):
        if i % 10 == 0 or i == 1 or i == len(queries):
            logger.info("parsing %d/%d (qid=%d)", i, len(queries), qid)
        profiles[qid] = parser.parse(query)
    return profiles


def route_all(
    profiles: dict[int, PatientProfile],
    config: DegradationConfig,
) -> dict[int, dict]:
    """Apply the A3 pure router to each profile. Returns ``{qid: {arm, reason, signals}}``."""
    out: dict[int, dict] = {}
    for qid, profile in profiles.items():
        decision = route_decision(profile, config)
        out[qid] = {
            "arm": decision.arm,
            "reason": decision.reason,
            "signals": decision.signals,
        }
    return out


# --------------------------------------------------------------------------- #
# Per-query metric assembly                                                   #
# --------------------------------------------------------------------------- #


def build_per_query_table(
    queries: dict[int, tuple[str, str]],
    routing: dict[int, dict],
    rule_by_qid: dict[int, list[dict]],
    agent_by_qid: dict[int, list[dict]],
    failed_agent_cost_estimate: float,
) -> list[dict]:
    """For every qid, build a row of rule + agent + production metrics."""
    rows: list[dict] = []
    for qid in sorted(queries.keys()):
        query, category = queries[qid]
        arm_decision = routing[qid]
        rule_rows = rule_by_qid.get(qid, [])
        agent_rows = agent_by_qid.get(qid, [])
        agent_succeeded = bool(agent_rows)

        rule_ndcg5 = ndcg_at_k_for_query(rule_rows, k=5)
        rule_ndcg10 = ndcg_at_k_for_query(rule_rows, k=10)
        rule_mrr = mrr_for_query(rule_rows)

        if agent_succeeded:
            agent_ndcg5 = ndcg_at_k_for_query(agent_rows, k=5)
            agent_ndcg10 = ndcg_at_k_for_query(agent_rows, k=10)
            agent_mrr_v = mrr_for_query(agent_rows)
            first = agent_rows[0]
            agent_cost = float(first.get("agent_cost_usd") or 0.0)
            agent_latency = float(first.get("agent_latency_ms") or 0.0)
            agent_iters = int(first.get("agent_iters_used") or 0)
        else:
            agent_ndcg5 = 0.0
            agent_ndcg10 = 0.0
            agent_mrr_v = 0.0
            agent_cost = failed_agent_cost_estimate
            agent_latency = 60_000.0  # 60s budget cap
            agent_iters = 5  # capped

        production_arm = arm_decision["arm"]

        if production_arm == "rule":
            # Easy case: would have used rule arm in production.
            prod_ndcg5_strict = rule_ndcg5
            prod_ndcg5_fallback = rule_ndcg5
            prod_ndcg10_strict = rule_ndcg10
            prod_ndcg10_fallback = rule_ndcg10
            prod_mrr_strict = rule_mrr
            prod_mrr_fallback = rule_mrr
            prod_cost = _RULE_COST_PER_QUERY_USD
            prod_latency = 6_000.0  # rule p50 ~6s on smoke
        else:
            # Agent-routed in production.
            prod_ndcg5_strict = agent_ndcg5  # 0 if agent failed
            prod_ndcg5_fallback = agent_ndcg5 if agent_succeeded else rule_ndcg5
            prod_ndcg10_strict = agent_ndcg10
            prod_ndcg10_fallback = agent_ndcg10 if agent_succeeded else rule_ndcg10
            prod_mrr_strict = agent_mrr_v
            prod_mrr_fallback = agent_mrr_v if agent_succeeded else rule_mrr
            prod_cost = agent_cost  # successful: real; failed: estimate
            prod_latency = agent_latency

        rows.append(
            {
                "query_id": qid,
                "query": query,
                "category": category,
                "routed_arm": production_arm,
                "routed_reason": arm_decision["reason"],
                "agent_succeeded": agent_succeeded,
                "rule_ndcg5": rule_ndcg5,
                "rule_ndcg10": rule_ndcg10,
                "rule_mrr": rule_mrr,
                "agent_ndcg5": agent_ndcg5,
                "agent_ndcg10": agent_ndcg10,
                "agent_mrr": agent_mrr_v,
                "agent_iters_used": agent_iters,
                "agent_cost_usd": agent_cost,
                "agent_latency_ms": agent_latency,
                "production_strict_ndcg5": prod_ndcg5_strict,
                "production_strict_ndcg10": prod_ndcg10_strict,
                "production_strict_mrr": prod_mrr_strict,
                "production_with_fallback_ndcg5": prod_ndcg5_fallback,
                "production_with_fallback_ndcg10": prod_ndcg10_fallback,
                "production_with_fallback_mrr": prod_mrr_fallback,
                "production_cost_usd": prod_cost,
                "production_latency_ms": prod_latency,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Roll up a row collection into the summary schema."""
    n = len(rows)
    if n == 0:
        return {
            "n_queries": 0,
            "routing_rate": 0.0,
            "rule_ndcg_at_5": 0.0,
            "agent_ndcg_at_5_where_ran": 0.0,
            "production_strict_ndcg_at_5": 0.0,
            "production_strict_ndcg_at_5_ci": [0.0, 0.0],
            "production_with_fallback_ndcg_at_5": 0.0,
            "production_with_fallback_ndcg_at_5_ci": [0.0, 0.0],
            "delta_strict_vs_baseline": 0.0,
            "delta_strict_ci": [0.0, 0.0],
            "delta_fallback_vs_baseline": 0.0,
            "delta_fallback_ci": [0.0, 0.0],
            "agent_success_rate": 0.0,
            "mean_cost_per_query_usd": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    agent_routed = [r for r in rows if r["routed_arm"] == "agent"]
    rule_routed = [r for r in rows if r["routed_arm"] == "rule"]
    n_agent_routed = len(agent_routed)

    rule_only_ndcg5 = [r["rule_ndcg5"] for r in rows]
    prod_strict_ndcg5 = [r["production_strict_ndcg5"] for r in rows]
    prod_fallback_ndcg5 = [r["production_with_fallback_ndcg5"] for r in rows]

    rule_mean = sum(rule_only_ndcg5) / n
    prod_strict_mean, prod_strict_lo, prod_strict_hi = bootstrap_mean_ci(prod_strict_ndcg5)
    prod_fallback_mean, prod_fallback_lo, prod_fallback_hi = bootstrap_mean_ci(prod_fallback_ndcg5)

    delta_strict, delta_strict_lo, delta_strict_hi = bootstrap_paired_delta_ci(
        prod_strict_ndcg5, rule_only_ndcg5
    )
    delta_fallback, delta_fallback_lo, delta_fallback_hi = bootstrap_paired_delta_ci(
        prod_fallback_ndcg5, rule_only_ndcg5
    )

    agent_ndcg5_where_ran = [r["agent_ndcg5"] for r in agent_routed if r["agent_succeeded"]]
    agent_ndcg5_mean = (
        sum(agent_ndcg5_where_ran) / len(agent_ndcg5_where_ran) if agent_ndcg5_where_ran else 0.0
    )
    agent_success_rate = (
        sum(1 for r in agent_routed if r["agent_succeeded"]) / n_agent_routed
        if n_agent_routed
        else 0.0
    )

    return {
        "n_queries": n,
        "n_agent_routed": n_agent_routed,
        "n_rule_routed": len(rule_routed),
        "routing_rate": n_agent_routed / n,
        "rule_ndcg_at_5": rule_mean,
        "agent_ndcg_at_5_where_ran": agent_ndcg5_mean,
        "agent_success_rate": agent_success_rate,
        "production_strict_ndcg_at_5": prod_strict_mean,
        "production_strict_ndcg_at_5_ci": [prod_strict_lo, prod_strict_hi],
        "production_with_fallback_ndcg_at_5": prod_fallback_mean,
        "production_with_fallback_ndcg_at_5_ci": [
            prod_fallback_lo,
            prod_fallback_hi,
        ],
        "delta_strict_vs_baseline": delta_strict,
        "delta_strict_ci": [delta_strict_lo, delta_strict_hi],
        "delta_fallback_vs_baseline": delta_fallback,
        "delta_fallback_ci": [delta_fallback_lo, delta_fallback_hi],
        "mean_cost_per_query_usd": sum(r["production_cost_usd"] for r in rows) / n,
        "p50_latency_ms": percentile([r["production_latency_ms"] for r in rows], 0.50),
        "p95_latency_ms": percentile([r["production_latency_ms"] for r in rows], 0.95),
    }


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_dollar(x: float) -> str:
    return f"${x:.4f}"


def write_markdown(report_path: Path, summary: dict, per_category: dict) -> None:
    """Generate a docs/evaluation-report.md §12-ready markdown table."""
    lines: list[str] = []
    lines.append("## Phase C3 — Agent-arm A/B comparison (offline)\n")
    lines.append(
        "Production NDCG is shown two ways: ``strict`` (agent failures "
        "score 0 — honest cost of failure) and ``with_fallback`` (failures "
        "fall back to rule, matching the A5 retry-as-rule design once the "
        "agent's degraded-result bubble bug is fixed).\n"
    )

    # Headline
    s = summary
    lines.append("### Headline")
    lines.append(
        f"- **Queries**: {s['n_queries']} "
        f"(routed to agent: {s['n_agent_routed']}, "
        f"to rule: {s['n_rule_routed']})"
    )
    lines.append(f"- **Routing rate (heuristic)**: {fmt_pct(s['routing_rate'])}")
    lines.append(f"  (Production with the 50/50 AB router: {fmt_pct(s['routing_rate'] * 0.5)})")
    lines.append(f"- **Agent success rate** (of agent-routed): {fmt_pct(s['agent_success_rate'])}")
    lines.append(f"- **Rule-only baseline NDCG@5**: {s['rule_ndcg_at_5']:.3f}")
    lines.append(
        f"- **Production NDCG@5 (strict)**: {s['production_strict_ndcg_at_5']:.3f} "
        f"[{s['production_strict_ndcg_at_5_ci'][0]:.3f}, "
        f"{s['production_strict_ndcg_at_5_ci'][1]:.3f}] "
        f"→ Δ = {s['delta_strict_vs_baseline']:+.3f} "
        f"[{s['delta_strict_ci'][0]:+.3f}, {s['delta_strict_ci'][1]:+.3f}]"
    )
    lines.append(
        f"- **Production NDCG@5 (with rule fallback)**: "
        f"{s['production_with_fallback_ndcg_at_5']:.3f} "
        f"[{s['production_with_fallback_ndcg_at_5_ci'][0]:.3f}, "
        f"{s['production_with_fallback_ndcg_at_5_ci'][1]:.3f}] "
        f"→ Δ = {s['delta_fallback_vs_baseline']:+.3f} "
        f"[{s['delta_fallback_ci'][0]:+.3f}, {s['delta_fallback_ci'][1]:+.3f}]"
    )
    lines.append(
        f"- **Mean cost/query**: {fmt_dollar(s['mean_cost_per_query_usd'])}  "
        f"(rule-only baseline: {fmt_dollar(_RULE_COST_PER_QUERY_USD)})"
    )
    lines.append(
        f"- **p50 / p95 latency**: {s['p50_latency_ms']:.0f} ms / {s['p95_latency_ms']:.0f} ms\n"
    )

    # Per-category
    lines.append("### Per-category")
    lines.append("")
    lines.append(
        "| Category | n | Routed to agent | Agent success | "
        "Rule NDCG@5 | Agent NDCG@5 (where ran) | "
        "Prod NDCG@5 (strict) | Prod NDCG@5 (fallback) | "
        "Δ strict vs rule | Δ fallback vs rule |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in sorted(per_category):
        ps = per_category[cat]
        if ps["n_queries"] == 0:
            continue
        lines.append(
            f"| {cat} | {ps['n_queries']} | "
            f"{ps['n_agent_routed']} | "
            f"{fmt_pct(ps['agent_success_rate'])} | "
            f"{ps['rule_ndcg_at_5']:.3f} | "
            f"{ps['agent_ndcg_at_5_where_ran']:.3f} | "
            f"{ps['production_strict_ndcg_at_5']:.3f} | "
            f"{ps['production_with_fallback_ndcg_at_5']:.3f} | "
            f"{ps['delta_strict_vs_baseline']:+.3f} | "
            f"{ps['delta_fallback_vs_baseline']:+.3f} |"
        )
    lines.append("")

    # Cost / latency table
    lines.append("### Cost + latency per category")
    lines.append("")
    lines.append("| Category | Mean cost / query | p50 latency | p95 latency |")
    lines.append("|---|---:|---:|---:|")
    for cat in sorted(per_category):
        ps = per_category[cat]
        if ps["n_queries"] == 0:
            continue
        lines.append(
            f"| {cat} | {fmt_dollar(ps['mean_cost_per_query_usd'])} | "
            f"{ps['p50_latency_ms']:.0f} ms | "
            f"{ps['p95_latency_ms']:.0f} ms |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines))
    logger.info("wrote markdown summary to %s", report_path)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C3 — A/B comparison.")
    p.add_argument(
        "--rule-65q",
        type=Path,
        default=Path("data/evaluation/full_labeled_dataset.jsonl"),
    )
    p.add_argument(
        "--rule-20q",
        type=Path,
        default=Path("data/evaluation/full_labeled_dataset_expansion_v2.jsonl"),
    )
    p.add_argument(
        "--agent-65q",
        type=Path,
        default=Path("data/evaluation/full_labeled_dataset_agent_all_65q.jsonl"),
    )
    p.add_argument(
        "--agent-20q",
        type=Path,
        default=Path("data/evaluation/full_labeled_dataset_agent_all_20q.jsonl"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/agent_vs_rule_v1.json"),
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=Path("data/evaluation/agent_vs_rule_v1.md"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load all four input files.
    rule_rows = load_jsonl(args.rule_65q) + load_jsonl(args.rule_20q)
    agent_rows = load_jsonl(args.agent_65q) + load_jsonl(args.agent_20q)

    rule_by_qid = group_by_qid(rule_rows)
    agent_by_qid = group_by_qid(agent_rows)
    logger.info(
        "loaded: rule rows=%d (qids=%d), agent rows=%d (qids=%d)",
        len(rule_rows),
        len(rule_by_qid),
        len(agent_rows),
        len(agent_by_qid),
    )

    # Build the canonical {qid: (query, category)} from the rule arm
    # (the rule arm has full coverage of all 85 queries).
    queries: dict[int, tuple[str, str]] = {}
    for r in rule_rows:
        qid = int(r["query_id"])
        if qid not in queries:
            queries[qid] = (r["query"], r.get("category", "unknown"))
    logger.info("total queries to process: %d", len(queries))

    # Parse each query → PatientProfile so the A3 router can decide.
    parser = QueryParserAgent()
    t0 = time.perf_counter()
    profiles = parse_all_queries(queries, parser)
    logger.info("parsed %d queries in %.1fs", len(profiles), time.perf_counter() - t0)

    # Apply the routing decision with agentic_path_enabled=True.
    cfg = DegradationConfig(agentic_path_enabled=True)
    routing = route_all(profiles, cfg)
    agent_routed_count = sum(1 for r in routing.values() if r["arm"] == "agent")
    logger.info(
        "heuristic routing: %d of %d → agent (%.1f%%)",
        agent_routed_count,
        len(routing),
        100 * agent_routed_count / len(routing),
    )

    # Estimate the cost of a failed agent run as the mean successful cost.
    successful_agent_costs = [
        float(rows[0]["agent_cost_usd"])
        for qid, rows in agent_by_qid.items()
        if rows and rows[0].get("agent_cost_usd") is not None
    ]
    failed_agent_cost_estimate = (
        statistics.mean(successful_agent_costs) if successful_agent_costs else 0.05
    )
    logger.info(
        "failed-agent cost estimate (mean of successful): $%.4f",
        failed_agent_cost_estimate,
    )

    # Per-query metric table.
    per_query = build_per_query_table(
        queries, routing, rule_by_qid, agent_by_qid, failed_agent_cost_estimate
    )

    # Aggregate overall + per-category.
    overall = aggregate(per_query)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_query:
        by_cat[r["category"]].append(r)
    per_category = {cat: aggregate(rows) for cat, rows in by_cat.items()}

    # Write JSON output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": overall,
        "per_category": per_category,
        "per_query": per_query,
        "notes": {
            "routing_rate_interpretation": "heuristic-rate; multiply by 0.5 for the AB router production rate",
            "production_strict": "agent failures score NDCG@5=0 (honest)",
            "production_with_fallback": "agent failures fall back to rule arm (matches A5 design with the bubble bug fixed)",
            "rule_cost_per_query_usd": _RULE_COST_PER_QUERY_USD,
            "failed_agent_cost_estimate_usd": failed_agent_cost_estimate,
            "n_bootstrap": _N_BOOTSTRAP,
            "bootstrap_seed": _BOOTSTRAP_SEED,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote JSON output to %s", args.output)

    # Write markdown.
    write_markdown(args.markdown, overall, per_category)

    # Quick console summary.
    print(f"\n{'=' * 78}")
    print("PHASE C3 — A/B comparison summary")
    print(f"{'=' * 78}")
    print(f"  n_queries:                       {overall['n_queries']}")
    print(f"  routing rate (heuristic):        {fmt_pct(overall['routing_rate'])}")
    print(f"  agent success rate:              {fmt_pct(overall['agent_success_rate'])}")
    print(f"  rule-only baseline NDCG@5:       {overall['rule_ndcg_at_5']:.3f}")
    print(
        f"  production NDCG@5 (strict):      "
        f"{overall['production_strict_ndcg_at_5']:.3f} "
        f"[{overall['production_strict_ndcg_at_5_ci'][0]:.3f}, "
        f"{overall['production_strict_ndcg_at_5_ci'][1]:.3f}]"
    )
    print(
        f"  production NDCG@5 (w/ fallback): "
        f"{overall['production_with_fallback_ndcg_at_5']:.3f} "
        f"[{overall['production_with_fallback_ndcg_at_5_ci'][0]:.3f}, "
        f"{overall['production_with_fallback_ndcg_at_5_ci'][1]:.3f}]"
    )
    print(
        f"  Δ strict vs baseline:            "
        f"{overall['delta_strict_vs_baseline']:+.3f} "
        f"[{overall['delta_strict_ci'][0]:+.3f}, "
        f"{overall['delta_strict_ci'][1]:+.3f}]"
    )
    print(
        f"  Δ fallback vs baseline:          "
        f"{overall['delta_fallback_vs_baseline']:+.3f} "
        f"[{overall['delta_fallback_ci'][0]:+.3f}, "
        f"{overall['delta_fallback_ci'][1]:+.3f}]"
    )
    print(f"  mean cost / query:               {fmt_dollar(overall['mean_cost_per_query_usd'])}")
    print(
        f"  p50 / p95 latency:               {overall['p50_latency_ms']:.0f} / {overall['p95_latency_ms']:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
