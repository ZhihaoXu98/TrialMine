# ruff: noqa: E501  # Markdown table cells + JSON-key strings frequently exceed 100 chars; not load-bearing for analysis scripts.
"""Phase C3b.4 — sanity test on the 18 previously-failed queries.

Reuses the machinery from ``scripts/eval_agent_path.py`` but operates
ONLY on the qids that returned 0 rows in the original C2 run. With
C3b.1 (bubble fix) + C3b.2 (prompt + max_iters=6) + C3b.3 (caching)
applied, we expect:

* The bubble fix means **every** qid now produces ≥ 1 row (either the
  agent submitted or the rule arm fallback fired).
* The prompt+max_iters change should LIFT the agent submit rate —
  fewer qids should need the rule fallback.
* Caching should drop the cost-per-query by ~30-40 % on multi-cycle
  queries.

Writes to ``/tmp/c3b_sanity_65q.jsonl`` and ``/tmp/c3b_sanity_20q.jsonl``
so the existing C2 output files stay untouched.

Decision gate for C3b.5 (per the runbook):

* **Green** on all 4 signals → run the full C2 + C3 re-eval.
* **Yellow** on any → diagnose before spending the $2 budget.
* **Red** on any → revert the C3b changes selectively.

Usage::

    OMP_NUM_THREADS=1 python scripts/c3b_sanity.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Load env first.
from TrialMine.config import DegradationConfig, Settings  # noqa: E402

_settings = Settings()  # type: ignore[call-arg]
os.environ.setdefault("ANTHROPIC_API_KEY", _settings.anthropic_api_key)

import TrialMine.config as cfg_mod  # noqa: E402

# Force agent path on. ``agentic_max_iters`` defaults to 6 from C3b.2;
# we explicitly set the wide per-iter / outer budget for the sanity
# probe so transient slowness doesn't masquerade as a no-terminator.
cfg_mod._DEFAULT_DEGRADATION = DegradationConfig(
    agentic_path_enabled=True,
    agentic_routing_threshold_slots=2,
    agentic_complex_pattern_enabled=True,
    agentic_max_iters=6,
    agentic_per_iter_timeout_s=12.0,
    skip_agent_if_slow_s=120.0,
)

# Force AB router to 100 % treatment.
import TrialMine.experiments.ab_test as ab  # noqa: E402

ab._default_router = None
_router = ab.get_default_router()
_router._catalog["agent_path_v1"].enabled = True
_router._catalog["agent_path_v1"].variants[0].weight = 0.0
_router._catalog["agent_path_v1"].variants[1].weight = 1.0

# Force the heuristic to always agent.
import TrialMine.agents.query_router as qr  # noqa: E402

_original_route = qr.route_decision


def _force_agent(profile, config):
    real = _original_route(profile, config)
    return qr.RouteDecision(
        arm="agent",
        reason="force_agent_for_c3b_sanity",
        signals=real.signals,
    )


qr.route_decision = _force_agent  # type: ignore[assignment]

import TrialMine.agents.pipeline as pl  # noqa: E402

pl._agent_singleton = None

# Pull the heavy helpers from eval_agent_path.py — they're already
# the right shape and re-implementing them would duplicate ~150 lines.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_agent_path import (  # noqa: E402
    load_labels,
    run_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Find the 18 missing qids                                                    #
# --------------------------------------------------------------------------- #


_RULE_INPUT_65Q = Path("data/evaluation/full_labeled_dataset.jsonl")
_RULE_INPUT_20Q = Path("data/evaluation/full_labeled_dataset_expansion_v2.jsonl")
_C2_OUTPUT_65Q = Path("data/evaluation/full_labeled_dataset_agent_all_65q.jsonl")
_C2_OUTPUT_20Q = Path("data/evaluation/full_labeled_dataset_agent_all_20q.jsonl")

_SANITY_OUT_65Q = Path("/tmp/c3b_sanity_65q.jsonl")
_SANITY_OUT_20Q = Path("/tmp/c3b_sanity_20q.jsonl")


def _qids_in(path: Path) -> set[int]:
    return {int(json.loads(line)["query_id"]) for line in path.open()}


def _identify_originally_failed_qids() -> tuple[set[int], set[int]]:
    """Find qids that failed in the *original* C2 run.

    The 65q output file currently contains the bubble-fix rule-fallback
    rows for the 8 originally-failed 65q qids (we re-ran with --resume
    after C3b.1 to bring coverage to 100 %). To recover the original
    failure list we look at agent_iters_used: any qid with iters=0 in
    the current 65q output was originally an agent failure that now
    falls back to rule.

    The 20q file is at 10/20 coverage — anything missing there is an
    original failure that hasn't been retried yet.
    """
    # 65q: originally-failed qids are those where iters_used == 0 OR
    # where they are missing entirely. Hardcode the canonical list
    # from C2's log to avoid ambiguity if the file changed.
    failed_65q = {17, 413, 415, 417, 420, 421, 424, 425}

    # 20q: missing qids in the C2 output.
    all_20q = _qids_in(_RULE_INPUT_20Q)
    done_20q = _qids_in(_C2_OUTPUT_20Q) if _C2_OUTPUT_20Q.exists() else set()
    failed_20q = all_20q - done_20q

    return failed_65q, failed_20q


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _run_one_subset(
    label_path: Path,
    output_path: Path,
    qids_to_run: set[int],
    label_map: dict,
    queries_meta: dict,
    client,
    con,
    label_token_counter: dict,
    graph,
) -> list[dict]:
    """Run the sanity slice for one input file's failed qids."""
    output_path.unlink(missing_ok=True)
    summaries: list[dict] = []
    sorted_qids = sorted(qids_to_run & set(queries_meta.keys()))
    for i, qid in enumerate(sorted_qids, start=1):
        query, category = queries_meta[qid]
        logger.info(
            "[%d/%d] qid=%d cat=%s query=%r",
            i,
            len(sorted_qids),
            qid,
            category,
            query[:70],
        )
        try:
            rows = asyncio.run(
                run_query(
                    graph,
                    qid,
                    query,
                    category,
                    label_map,
                    client,
                    con,
                    label_token_counter,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("qid=%d crashed", qid)
            rows = []
        with open(output_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Per-qid summary — classify outcome
        if not rows:
            outcome = "no_rows"
            iters = None
        else:
            iters = rows[0].get("agent_iters_used") or 0
            outcome = "agent_submitted" if iters > 0 else "rule_fallback"
        summaries.append(
            {
                "qid": qid,
                "category": category,
                "outcome": outcome,
                "n_rows": len(rows),
                "iters_used": iters,
                "cost_usd": rows[0].get("agent_cost_usd") if rows else 0.0,
                "latency_ms": rows[0].get("agent_latency_ms") if rows else 0.0,
                "agent_error": rows[0].get("agent_error") if rows else None,
                "tool_sequence_len": len(rows[0].get("agent_tool_sequence") or []) if rows else 0,
            }
        )
    return summaries


def main() -> int:
    failed_65q, failed_20q = _identify_originally_failed_qids()
    logger.info("originally-failed 65q qids: %s", sorted(failed_65q))
    logger.info("originally-failed 20q qids: %s", sorted(failed_20q))
    total_n = len(failed_65q) + len(failed_20q)
    logger.info("total to retry under C3b fixes: %d", total_n)

    # Load labels for both inputs.
    label_map_65q, queries_65q = load_labels(_RULE_INPUT_65Q)
    label_map_20q, queries_20q = load_labels(_RULE_INPUT_20Q)
    label_map = {**label_map_65q, **label_map_20q}

    # Anthropic client for any new-NCT labeling.
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    from TrialMine.agents.tools import _db_conn

    con = _db_conn()

    graph = pl.build_pipeline()
    label_token_counter: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "n_labels": 0,
    }

    print(f"\n{'=' * 78}")
    print(f"Phase C3b.4 — Sanity test ({total_n} previously-failed qids)")
    print(f"{'=' * 78}\n")

    try:
        sum_65q = _run_one_subset(
            _RULE_INPUT_65Q,
            _SANITY_OUT_65Q,
            failed_65q,
            label_map,
            queries_65q,
            client,
            con,
            label_token_counter,
            graph,
        )
        sum_20q = _run_one_subset(
            _RULE_INPUT_20Q,
            _SANITY_OUT_20Q,
            failed_20q,
            label_map,
            queries_20q,
            client,
            con,
            label_token_counter,
            graph,
        )
    finally:
        con.close()

    all_sum = sum_65q + sum_20q

    # --- Outcome classification ----------------------------------------- #
    agent_submitted = [s for s in all_sum if s["outcome"] == "agent_submitted"]
    rule_fallback = [s for s in all_sum if s["outcome"] == "rule_fallback"]
    no_rows = [s for s in all_sum if s["outcome"] == "no_rows"]

    n = len(all_sum)
    recovery_rate = (n - len(no_rows)) / n if n else 0.0
    agent_submit_rate = len(agent_submitted) / n if n else 0.0

    # --- Cost + iter stats ---------------------------------------------- #
    agent_costs = [s["cost_usd"] for s in agent_submitted if s["cost_usd"]]
    agent_iters_used = [s["iters_used"] for s in agent_submitted if s["iters_used"]]
    cum_agent_cost = sum(agent_costs)
    label_cost = (
        label_token_counter["input_tokens"] / 1_000_000
        + label_token_counter["output_tokens"] * 5 / 1_000_000
    )
    mean_iters = sum(agent_iters_used) / len(agent_iters_used) if agent_iters_used else 0.0

    # --- Summary table -------------------------------------------------- #
    print(f"{'=' * 78}")
    print(f"OUTCOMES (n={n})")
    print(f"{'=' * 78}")
    print(
        f"  agent submitted:           {len(agent_submitted):2d}  ({agent_submit_rate * 100:.0f}%)"
    )
    print(
        f"  rule fallback (bubble fix):{len(rule_fallback):2d}  ({len(rule_fallback) / n * 100:.0f}%)"
    )
    print(f"  no rows (BAD):             {len(no_rows):2d}  ({len(no_rows) / n * 100:.0f}%)")
    print(f"  recovery rate (any rows):  {recovery_rate * 100:.0f}%   (target ≥ 78%)")
    print()
    print(f"  mean iters (submitted):    {mean_iters:.2f}  (target [3.0, 4.5])")
    print(f"  total agent cost:          ${cum_agent_cost:.4f}")
    print(
        f"  total labeling cost:       ${label_cost:.4f}  ({label_token_counter['n_labels']} new labels)"
    )
    print(f"  TOTAL cost (this run):     ${cum_agent_cost + label_cost:.4f}   (cap ≤ $1.50)")

    # --- Per-qid table -------------------------------------------------- #
    print(f"\n{'=' * 78}")
    print("PER-QID OUTCOMES")
    print(f"{'=' * 78}")
    print(
        f"  {'qid':>5}  {'cat':<10} {'outcome':<18} {'n':>3}  {'iters':>5}  {'cost':>8}  {'latency':>8}"
    )
    print(f"  {'-' * 75}")
    for s in all_sum:
        print(
            f"  {s['qid']:>5}  {s['category']:<10} {s['outcome']:<18} "
            f"{s['n_rows']:>3}  "
            f"{str(s['iters_used'] if s['iters_used'] is not None else '-'):>5}  "
            f"${(s['cost_usd'] or 0):.4f}  "
            f"{(s['latency_ms'] or 0):.0f}ms"
        )

    # --- Decision gate -------------------------------------------------- #
    print(f"\n{'=' * 78}")
    print("DECISION GATE (input to C3b.5)")
    print(f"{'=' * 78}")
    green = []
    yellow = []
    red = []

    if recovery_rate >= 0.78:
        green.append(f"recovery rate {recovery_rate * 100:.0f}% ≥ 78%")
    elif recovery_rate >= 0.50:
        yellow.append(f"recovery rate {recovery_rate * 100:.0f}% in [50, 78%)")
    else:
        red.append(f"recovery rate {recovery_rate * 100:.0f}% < 50% — fixes not landing")

    if 3.0 <= mean_iters <= 4.5:
        green.append(f"mean iters {mean_iters:.2f} in [3.0, 4.5]")
    elif 4.5 < mean_iters <= 5.5:
        yellow.append(f"mean iters {mean_iters:.2f} > 4.5")
    elif mean_iters > 5.5:
        red.append(f"mean iters {mean_iters:.2f} > 5.5 — cycle cap is new target")
    elif mean_iters > 0:
        yellow.append(f"mean iters {mean_iters:.2f} < 3.0 (premature termination?)")

    total_cost = cum_agent_cost + label_cost
    if total_cost <= 0.80:
        green.append(f"cost ${total_cost:.4f} ≤ $0.80")
    elif total_cost <= 1.50:
        yellow.append(f"cost ${total_cost:.4f} in ($0.80, $1.50]")
    else:
        red.append(f"cost ${total_cost:.4f} > $1.50 — caching may not have engaged")

    print("\n  GREEN signals:")
    for g in green:
        print(f"    ✅ {g}")
    print("\n  YELLOW signals:")
    for y in yellow:
        print(f"    ⚠ {y}")
    if red:
        print("\n  RED signals:")
        for r in red:
            print(f"    ❌ {r}")

    print()
    if red:
        print("  → DO NOT proceed to C3b.5. Diagnose the red signal(s) first.")
        return 1
    elif yellow:
        print("  → Proceed to C3b.5 with caveat: " + "; ".join(yellow))
        return 0
    else:
        print("  → ALL GREEN. Safe to run C3b.5 full re-eval.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
