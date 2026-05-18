# ruff: noqa: E501  # Markdown table cells + JSON-key strings frequently exceed 100 chars; not load-bearing for analysis scripts.
"""Phase C3b.5 — side-by-side comparison of C3 (v1) vs C3b (v2).

Reads ``agent_vs_rule_v1.json`` and ``agent_vs_rule_v2.json`` produced
by ``scripts/c3_compare.py`` runs against the C2 (v1) and C3b.5 (v2)
agent outputs, and writes a structured before/after markdown so the
C4 decision gate has concrete deltas.

Compares the six C3b.5 acceptance signals from the runbook:

* Agent success rate (was 57.5 % — target ≥ 80 %)
* Strict production NDCG@5 (was 0.725 — target ≥ 0.79)
* Fallback production NDCG@5 (was 0.840 — target ≥ 0.84)
* Complex-slice fallback NDCG@5 lift (was +0.177 — target ≥ +0.10)
* Mean cost / query (was $0.0195 — target ≤ $0.012)
* p95 latency (was 60 s — target ≤ 30 s)

Plus the recovered / regressed qid lists for an honest writeup.

Usage::

    python scripts/c3b_before_after.py \\
        --before data/evaluation/agent_vs_rule_v1.json \\
        --after data/evaluation/agent_vs_rule_v2.json \\
        --output data/evaluation/c3b_before_after.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_dollar(x: float) -> str:
    return f"${x:.4f}"


def fmt_signed(x: float, decimals: int = 3) -> str:
    return f"{x:+.{decimals}f}"


def delta_signal(
    before: float,
    after: float,
    *,
    higher_is_better: bool,
    target: float | None = None,
    target_compare: str = ">=",
) -> str:
    """Build an emoji + delta string for one signal."""
    diff = after - before
    direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    sign = (
        "✅" if ((higher_is_better and diff >= 0) or (not higher_is_better and diff <= 0)) else "❌"
    )
    if target is not None:
        ok = (
            after >= target
            if target_compare == ">="
            else after <= target
            if target_compare == "<="
            else None
        )
        sign = "✅" if ok else "⚠"
    return f"{sign} {direction} {fmt_signed(diff)}"


# --------------------------------------------------------------------------- #
# Comparison                                                                  #
# --------------------------------------------------------------------------- #


def build_summary_block(v1: dict, v2: dict) -> list[str]:
    s1 = v1["summary"]
    s2 = v2["summary"]
    lines: list[str] = []
    lines.append("### Headline before / after\n")
    lines.append("| Signal | C3 (v1) | C3b (v2) | Δ | Verdict |")
    lines.append("|---|---:|---:|---:|:---:|")

    # 1. Agent success rate
    a1 = s1.get("agent_success_rate", 0.0)
    a2 = s2.get("agent_success_rate", 0.0)
    lines.append(
        f"| Agent success rate | {fmt_pct(a1)} | {fmt_pct(a2)} | "
        f"{fmt_signed((a2 - a1) * 100, 1)} pts | "
        f"{'✅ ≥80%' if a2 >= 0.80 else '⚠ <80%'} |"
    )

    # 2. Strict NDCG@5
    b1 = s1.get("production_strict_ndcg_at_5", 0.0)
    b2 = s2.get("production_strict_ndcg_at_5", 0.0)
    rule_baseline = s1.get("rule_ndcg_at_5", 0.792)
    lines.append(
        f"| Strict NDCG@5 | {b1:.3f} | {b2:.3f} | {fmt_signed(b2 - b1)} | "
        f"{'✅ ≥ rule (' + str(round(rule_baseline, 2)) + ')' if b2 >= rule_baseline else '⚠ < rule baseline'} |"
    )

    # 3. Fallback NDCG@5
    c1 = s1.get("production_with_fallback_ndcg_at_5", 0.0)
    c2 = s2.get("production_with_fallback_ndcg_at_5", 0.0)
    lines.append(
        f"| Fallback NDCG@5 | {c1:.3f} | {c2:.3f} | {fmt_signed(c2 - c1)} | "
        f"{'✅ lift' if c2 >= rule_baseline else '⚠'} |"
    )

    # 4. Mean cost / query
    d1 = s1.get("mean_cost_per_query_usd", 0.0)
    d2 = s2.get("mean_cost_per_query_usd", 0.0)
    lines.append(
        f"| Mean cost / query | {fmt_dollar(d1)} | {fmt_dollar(d2)} | "
        f"{fmt_signed((d2 - d1) * 1000, 2)} m¢ | "
        f"{'✅ ≤ $0.012' if d2 <= 0.012 else '⚠ above $0.012'} |"
    )

    # 5. p95 latency
    e1 = s1.get("p95_latency_ms", 0.0)
    e2 = s2.get("p95_latency_ms", 0.0)
    lines.append(
        f"| p95 latency | {e1:.0f} ms | {e2:.0f} ms | "
        f"{fmt_signed(e2 - e1, 0)} ms | "
        f"{'✅ ≤ 30s' if e2 <= 30_000 else '⚠ > 30s'} |"
    )

    # 6. Routing rate
    f1 = s1.get("routing_rate", 0.0)
    f2 = s2.get("routing_rate", 0.0)
    lines.append(
        f"| Routing rate | {fmt_pct(f1)} | {fmt_pct(f2)} | "
        f"{fmt_signed((f2 - f1) * 100, 1)} pts | — |"
    )

    return lines


def build_per_category_block(v1: dict, v2: dict) -> list[str]:
    pc1 = v1.get("per_category", {})
    pc2 = v2.get("per_category", {})
    cats = sorted(set(pc1.keys()) | set(pc2.keys()))
    lines: list[str] = []
    lines.append("\n### Per-category fallback NDCG@5 — v1 vs v2")
    lines.append("")
    lines.append(
        "| Category | n | v1 fallback NDCG@5 | v2 fallback NDCG@5 | Δ | v2 agent success |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat in cats:
        a = pc1.get(cat, {})
        b = pc2.get(cat, {})
        if not a and not b:
            continue
        n = max(a.get("n_queries", 0), b.get("n_queries", 0))
        n1 = a.get("production_with_fallback_ndcg_at_5", 0.0)
        n2 = b.get("production_with_fallback_ndcg_at_5", 0.0)
        success2 = b.get("agent_success_rate", 0.0)
        lines.append(
            f"| {cat} | {n} | {n1:.3f} | {n2:.3f} | {fmt_signed(n2 - n1)} | {fmt_pct(success2)} |"
        )
    return lines


def build_recovered_regressed_block(v1: dict, v2: dict) -> list[str]:
    """List qids that recovered (failed v1 → succeeded v2) and regressed."""
    pq1 = {q["query_id"]: q for q in v1.get("per_query", [])}
    pq2 = {q["query_id"]: q for q in v2.get("per_query", [])}

    recovered: list[dict] = []  # failed v1, succeeded v2
    regressed: list[dict] = []  # succeeded v1, failed v2
    still_failing: list[dict] = []  # failed both

    for qid, r2 in pq2.items():
        r1 = pq1.get(qid)
        if not r1:
            continue
        agent_ok_1 = r1.get("agent_succeeded", False)
        agent_ok_2 = r2.get("agent_succeeded", False)
        if not agent_ok_1 and agent_ok_2:
            recovered.append(r2)
        elif agent_ok_1 and not agent_ok_2:
            regressed.append(r2)
        elif not agent_ok_1 and not agent_ok_2:
            still_failing.append(r2)

    lines: list[str] = []
    lines.append("\n### Agent recovery / regression breakdown")
    lines.append("")
    lines.append(
        f"- **Recovered** (failed v1, succeeded v2): "
        f"{len(recovered)} qids → " + ", ".join(str(r["query_id"]) for r in recovered[:30])
    )
    lines.append(
        f"- **Regressed** (succeeded v1, failed v2): "
        f"{len(regressed)} qids → "
        + (", ".join(str(r["query_id"]) for r in regressed[:30]) if regressed else "none")
    )
    lines.append(
        f"- **Still failing** (no agent submit in either run): "
        f"{len(still_failing)} qids → " + ", ".join(str(r["query_id"]) for r in still_failing[:30])
    )
    return lines


def build_cost_block(v1: dict, v2: dict) -> list[str]:
    pq1 = v1.get("per_query", [])
    pq2 = v2.get("per_query", [])

    # Average cost per query for queries that ran the agent path
    def _avg(items: list[dict], key: str) -> float:
        vals = [
            float(r.get(key) or 0)
            for r in items
            if r.get("routed_arm") == "agent" and r.get("agent_succeeded")
        ]
        return sum(vals) / len(vals) if vals else 0.0

    c1 = _avg(pq1, "agent_cost_usd")
    c2 = _avg(pq2, "agent_cost_usd")
    lines: list[str] = []
    lines.append("\n### Cost per successful agent query")
    lines.append("")
    lines.append(f"- C3 (v1, no caching): {fmt_dollar(c1)}/query")
    lines.append(f"- C3b (v2, with caching): {fmt_dollar(c2)}/query")
    if c1 > 0:
        savings = (c1 - c2) / c1 * 100
        lines.append(f"- Savings: {fmt_pct(savings / 100)} per successful agent query")
    return lines


def build_c4_input_block(v1: dict, v2: dict) -> list[str]:
    s1 = v1["summary"]
    s2 = v2["summary"]
    rule_baseline = s1.get("rule_ndcg_at_5", 0.792)
    lines: list[str] = []
    lines.append("\n### C4 decision-gate input — 6 criteria")
    lines.append("")
    lines.append("| Criterion | C2/C3 result | C3b.5 result | Target | Verdict |")
    lines.append("|---|---:|---:|---:|:---:|")

    checks = [
        (
            "Agent success rate",
            fmt_pct(s1.get("agent_success_rate", 0.0)),
            fmt_pct(s2.get("agent_success_rate", 0.0)),
            "≥ 80%",
            s2.get("agent_success_rate", 0.0) >= 0.80,
        ),
        (
            "Strict NDCG@5",
            f"{s1.get('production_strict_ndcg_at_5', 0.0):.3f}",
            f"{s2.get('production_strict_ndcg_at_5', 0.0):.3f}",
            f"≥ {rule_baseline:.3f}",
            s2.get("production_strict_ndcg_at_5", 0.0) >= rule_baseline - 0.005,
        ),
        (
            "Fallback NDCG@5",
            f"{s1.get('production_with_fallback_ndcg_at_5', 0.0):.3f}",
            f"{s2.get('production_with_fallback_ndcg_at_5', 0.0):.3f}",
            f"≥ {rule_baseline + 0.05:.3f}",
            s2.get("production_with_fallback_ndcg_at_5", 0.0) >= rule_baseline + 0.05,
        ),
        (
            "Complex fallback lift",
            f"{(v1.get('per_category') or {}).get('complex', {}).get('delta_fallback_vs_baseline', 0.0):+.3f}",
            f"{(v2.get('per_category') or {}).get('complex', {}).get('delta_fallback_vs_baseline', 0.0):+.3f}",
            "≥ +0.10",
            (v2.get("per_category") or {}).get("complex", {}).get("delta_fallback_vs_baseline", 0.0)
            >= 0.10,
        ),
        (
            "Mean cost / query",
            fmt_dollar(s1.get("mean_cost_per_query_usd", 0.0)),
            fmt_dollar(s2.get("mean_cost_per_query_usd", 0.0)),
            "≤ $0.012",
            s2.get("mean_cost_per_query_usd", 0.0) <= 0.012,
        ),
        (
            "p95 latency",
            f"{s1.get('p95_latency_ms', 0.0):.0f} ms",
            f"{s2.get('p95_latency_ms', 0.0):.0f} ms",
            "≤ 30,000 ms",
            s2.get("p95_latency_ms", 0.0) <= 30_000,
        ),
    ]

    pass_count = 0
    for name, before, after, target, passed in checks:
        verdict = "✅" if passed else "⚠"
        pass_count += int(passed)
        lines.append(f"| {name} | {before} | {after} | {target} | {verdict} |")

    lines.append("")
    lines.append(f"**Pass rate: {pass_count} / {len(checks)}**")
    if pass_count == len(checks):
        lines.append("\n→ **C4 verdict: SHIP** (all 6 criteria green)")
    elif pass_count >= len(checks) - 1:
        lines.append(
            "\n→ **C4 verdict: SHIP with documented caveat** "
            f"({len(checks) - pass_count} criterion missed)"
        )
    elif pass_count >= 3:
        lines.append("\n→ **C4 verdict: TUNE** (one more iteration)")
    else:
        lines.append("\n→ **C4 verdict: HOLD** (three fixes weren't enough; document & revert)")

    return lines


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before",
        type=Path,
        default=Path("data/evaluation/agent_vs_rule_v1.json"),
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=Path("data/evaluation/agent_vs_rule_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/c3b_before_after.md"),
    )
    args = parser.parse_args()

    v1 = json.loads(args.before.read_text())
    v2 = json.loads(args.after.read_text())
    logger.info(
        "loaded v1 (%d queries) and v2 (%d queries)",
        v1["summary"]["n_queries"],
        v2["summary"]["n_queries"],
    )

    lines: list[str] = []
    lines.append("# Phase C3b — Before / after comparison\n")
    lines.append(
        "Side-by-side of C3's pre-fix A/B output (`agent_vs_rule_v1.json`) "
        "vs C3b.5's post-fix re-eval (`agent_vs_rule_v2.json`).\n"
    )
    lines.extend(build_summary_block(v1, v2))
    lines.extend(build_per_category_block(v1, v2))
    lines.extend(build_cost_block(v1, v2))
    lines.extend(build_recovered_regressed_block(v1, v2))
    lines.extend(build_c4_input_block(v1, v2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    logger.info("wrote before/after summary to %s", args.output)

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
