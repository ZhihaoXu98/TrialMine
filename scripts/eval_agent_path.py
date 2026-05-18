"""Phase C2 — eval the agent arm on every routed query.

Forces the LangGraph pipeline through the agent arm for every query in
the input label set, captures per-query agent metrics, and emits a
labeled JSONL that mirrors the input schema plus six agent-specific
fields. NCTs that already have a label in the input pool reuse it;
new NCTs surfaced by the agent are labeled by Claude Haiku 4.5 with
the same 0-3 rubric ``build_full_eval_dataset.py`` uses (so the two
files are directly comparable for the C3 A/B analysis).

Three patches make the agent fire on every query:

1. ``DegradationConfig.agentic_path_enabled = True`` so the pipeline's
   route-decision node consults the AB router.
2. The AB router's ``agent_path_v1`` experiment is forced to weight
   ``treatment=1.0`` so the 50/50 hash gate doesn't downsample the
   eval set.
3. The pure :func:`TrialMine.agents.query_router.route_decision`
   function is monkey-patched to always return ``arm="agent"`` so the
   sparse/complex heuristic doesn't down-select the queries either.

Usage::

    OMP_NUM_THREADS=1 python scripts/eval_agent_path.py \\
        --labels data/evaluation/full_labeled_dataset.jsonl \\
        --output data/evaluation/full_labeled_dataset_agent_all_65q.jsonl \\
        --resume

The ``--resume`` flag skips ``query_id``s that already have rows in
the output file — agent runs are slow + expensive, so a partial run
should not have to redo work.

Cost budget (per the runbook):
    Agent API:    ~$0.05 per query × 85 queries = ~$4.25
    Haiku labels: ~$0.0008 per new (query, trial) pair × ~425 new = ~$0.35
    Total:        ~$5 worst-case. Use ``--limit N`` for a sanity slice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Make the src/ package importable when running as a script (mirrors the
# pattern in scripts/parse_eligibility.py and friends).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Load ANTHROPIC_API_KEY from .env BEFORE the heavy imports — the agent
# constructor reads it at instantiation, and Haiku labeling also needs it.
from TrialMine.config import DegradationConfig, Settings  # noqa: E402

_settings = Settings()  # type: ignore[call-arg]
os.environ.setdefault("ANTHROPIC_API_KEY", _settings.anthropic_api_key)

# Patch #1 — force agent path on with generous budgets. The A8-tuned
# per-iter (12s) and outer budget (120s) give complex queries room to
# finish their 4–5 cycle loop.
import TrialMine.config as cfg_mod  # noqa: E402

cfg_mod._DEFAULT_DEGRADATION = DegradationConfig(
    agentic_path_enabled=True,
    agentic_routing_threshold_slots=2,
    agentic_complex_pattern_enabled=True,
    agentic_max_iters=5,
    agentic_per_iter_timeout_s=12.0,
    skip_agent_if_slow_s=120.0,
)

# Patch #2 — force the AB router to 100 % treatment.
import TrialMine.experiments.ab_test as ab  # noqa: E402

ab._default_router = None
_router = ab.get_default_router()
_router._catalog["agent_path_v1"].enabled = True
_router._catalog["agent_path_v1"].variants[0].weight = 0.0  # control
_router._catalog["agent_path_v1"].variants[1].weight = 1.0  # treatment

# Patch #3 — force the heuristic to ALWAYS say agent. We preserve the
# real signals (populated_slots, matched_pattern, etc.) so the trace
# stays informative, but override arm + reason.
import TrialMine.agents.query_router as qr  # noqa: E402

_original_route = qr.route_decision


def _force_agent(profile, config):
    real = _original_route(profile, config)
    return qr.RouteDecision(
        arm="agent",
        reason="force_agent_for_eval",
        signals=real.signals,
    )


qr.route_decision = _force_agent  # type: ignore[assignment]

# Reset pipeline singletons so the new config takes effect on first call.
import TrialMine.agents.pipeline as pl  # noqa: E402

pl._agent_singleton = None

from TrialMine.agents.pipeline import build_pipeline, search  # noqa: E402
from TrialMine.evaluation.metrics import mrr, ndcg_at_k  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Haiku 4.5 published rates (matches monitoring/metrics.py).
_HAIKU_IN_RATE = 1.0 / 1_000_000
_HAIKU_OUT_RATE = 5.0 / 1_000_000

# Default per-query wall-clock budget; the inner agent has its own
# (max_iters × per_iter), but the outer wait_for here is the hard cap.
_PER_QUERY_TIMEOUT_S = 120.0

# Labeling prompt copied from ``build_full_eval_dataset.py`` (Phase 9)
# so the C2 labels are directly comparable with the existing eval set.
LABELING_PROMPT = """Rate the relevance of this clinical trial to this patient's search query.

Patient query: {query}

Trial title: {trial_title}
Trial conditions: {conditions}
Trial phase: {phase}
Trial status: {status}
Trial eligibility (first 500 chars): {eligibility}

Rate on a scale of 0-3:
0 = Completely irrelevant — wrong cancer type entirely
1 = Marginally relevant — same general cancer area but wrong specifics
2 = Relevant — matches condition, patient could potentially be eligible
3 = Highly relevant — strong match for condition, treatment type, and eligibility

Respond with ONLY a JSON object: {{"score": X, "reason": "brief 1-sentence explanation"}}"""

LABELER_MODEL = "claude-haiku-4-5-20251001"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def load_labels(path: Path) -> tuple[dict, dict]:
    """Load the input JSONL into ``({(qid, nct): label}, {qid: (query, category)})``."""
    label_map: dict[tuple[int, str], dict] = {}
    queries_meta: dict[int, tuple[str, str]] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qid = d.get("query_id")
            nct = d.get("nct_id")
            if qid is None or nct is None:
                continue
            label_map[(int(qid), str(nct))] = {
                "relevance": int(d["relevance"]),
                "reason": d.get("reason", ""),
                "labeler": d.get("labeler", "claude-haiku-4-5"),
            }
            if qid not in queries_meta:
                queries_meta[int(qid)] = (
                    d.get("query") or "",
                    d.get("category") or "unknown",
                )
    return label_map, queries_meta


def load_done_qids(output_path: Path) -> set[int]:
    """Read already-written query_ids from ``output_path`` for --resume."""
    done: set[int] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = d.get("query_id")
            if qid is not None:
                done.add(int(qid))
    return done


def load_trial(con, nct_id: str) -> dict | None:
    """Project a SQLite trial row into the label-prompt's expected shape."""
    from TrialMine.agents.tools import _load_trial_row

    row = _load_trial_row(con, nct_id)
    if row is None:
        return None
    raw_conditions = row["conditions"]
    try:
        cond_list = json.loads(raw_conditions or "[]")
        conditions = (
            "; ".join(str(c) for c in cond_list)
            if isinstance(cond_list, list)
            else str(raw_conditions or "")
        )
    except (json.JSONDecodeError, TypeError):
        conditions = str(raw_conditions or "")
    return {
        "title": row["title"] or "",
        "conditions": conditions,
        "phase": row["phase"] or "",
        "status": row["status"] or "",
        "eligibility": row["eligibility_criteria"] or "",
    }


def label_pair(client, query: str, trial: dict) -> tuple[int, str, int, int]:
    """Score one (query, trial) pair. Returns (score, reason, in_tok, out_tok).

    On any API / parse failure, returns ``(0, "label_failed: …", 0, 0)``
    so a single bad call does not stop the run.
    """
    import anthropic  # noqa: F401

    prompt = LABELING_PROMPT.format(
        query=query,
        trial_title=trial.get("title") or "N/A",
        conditions=trial.get("conditions") or "N/A",
        phase=trial.get("phase") or "N/A",
        status=trial.get("status") or "N/A",
        eligibility=(trial.get("eligibility") or "")[:500],
    )
    try:
        response = client.messages.create(
            model=LABELER_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        score = int(result["score"])
        reason = result.get("reason", "")
        in_tok = int(getattr(response.usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(response.usage, "output_tokens", 0) or 0)
        return max(0, min(3, score)), reason, in_tok, out_tok
    except Exception as exc:  # noqa: BLE001
        logger.warning("label_pair failed: %s", exc)
        return 0, f"label_failed: {type(exc).__name__}: {exc}", 0, 0


def extract_agent_metrics(final: dict) -> dict[str, Any]:
    """Pull per-query agent metrics out of the pipeline.search() response.

    Phase C3b.3 added ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens`` to the ``agent_submit`` trace entry.
    The cost formula below applies the Anthropic prompt-caching pricing:
    cache reads bill at 10 % of the input rate, cache creation at 125 %.
    ``input_tokens`` from langchain INCLUDES the cached portion, so we
    subtract them out before applying the full rate (otherwise the
    cached portion would be double-charged).
    """
    trace = final.get("agent_trace") or []
    submit = next((e for e in trace if e.get("step") == "agent_submit"), None)
    tools = [
        e["decisions"]["tool"]
        for e in trace
        if e.get("step") == "tool_call"
        and isinstance(e.get("decisions"), dict)
        and "tool" in e["decisions"]
    ]
    if submit:
        sd = submit.get("decisions") or {}
        iters = int(sd.get("iters_used") or 0)
        in_tok = int(sd.get("input_tokens") or 0)
        out_tok = int(sd.get("output_tokens") or 0)
        cache_r = int(sd.get("cache_read_input_tokens") or 0)
        cache_c = int(sd.get("cache_creation_input_tokens") or 0)
    else:
        iters = 0
        in_tok = 0
        out_tok = 0
        cache_r = 0
        cache_c = 0
    # Cache-aware cost (matches monitoring/metrics.py:record_agent_cost).
    uncached_in = max(0, in_tok - cache_r - cache_c)
    cost_usd = (
        uncached_in * _HAIKU_IN_RATE
        + cache_r * (_HAIKU_IN_RATE * 0.10)
        + cache_c * (_HAIKU_IN_RATE * 1.25)
        + out_tok * _HAIKU_OUT_RATE
    )
    return {
        "iters": iters,
        "tools": tools,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_read_input_tokens": cache_r,
        "cache_creation_input_tokens": cache_c,
        "cost_usd": cost_usd,
    }


async def run_query(
    graph,
    qid: int,
    query: str,
    category: str,
    label_map: dict,
    client,
    con,
    label_token_counter: dict,
) -> list[dict]:
    """Run the agent on one query, emit one row per top result.

    Reuses existing labels when (qid, nct) is in ``label_map``; otherwise
    calls Haiku to score the new pair. New-label tokens are summed into
    ``label_token_counter`` so we can report total labeling spend later.
    """
    t0 = time.perf_counter()
    try:
        final = await asyncio.wait_for(
            search(query, graph, timeout=_PER_QUERY_TIMEOUT_S),
            timeout=_PER_QUERY_TIMEOUT_S + 5.0,
        )
    except TimeoutError:
        logger.warning("qid=%d outer timeout", qid)
        return []
    wall_ms = (time.perf_counter() - t0) * 1000
    meta = extract_agent_metrics(final)
    meta["latency_ms"] = round(wall_ms, 1)
    # ``final.error`` is cleared by execute_search's success path after
    # the C3b.1 bubble-fix's retry-as-rule fires — so the top-level
    # error field is None even when the agent originally failed. Look
    # at the trace's ``execute_agent_search_degraded`` step instead,
    # which preserves the agent's inner_error for observability.
    error_str = final.get("error")
    if not error_str:
        for entry in final.get("agent_trace") or []:
            if entry.get("step") == "execute_agent_search_degraded":
                inner = (entry.get("decisions") or {}).get("inner_error")
                if inner:
                    error_str = f"agent_arm_failed: {inner}"
                break

    results = (final.get("search_results") or {}).get("results") or []
    rows: list[dict] = []
    seen_ncts: set[str] = set()
    # Skip rows when the agent returned the degraded shape — i.e. the
    # invocation failed inside react_agent and produced an empty result
    # with an error stashed in the result_dict. (A5's pipeline node
    # doesn't elevate this to state["error"], so we surface it here so
    # the row isn't written as a real agent ranking.)
    inner_error = (final.get("search_results") or {}).get("error")
    if inner_error and not results:
        logger.warning(
            "qid=%d agent path returned no results due to: %s",
            qid,
            inner_error,
        )
        return []
    for rank, r in enumerate(results, start=1):
        nct = r.get("nct_id")
        if not nct:
            continue
        nct = str(nct).strip().upper()
        if nct in seen_ncts:
            # Same trial submitted twice by the agent — relevance score
            # is meaningless past the first appearance, and a duplicate
            # NCT inflates DCG vs IDCG (DCG counts the relevance at every
            # rank position; IDCG sorts the dedup'd values). Skip.
            logger.warning(
                "qid=%d nct=%s appeared more than once in agent results "
                "(rank=%d) — keeping only the first occurrence",
                qid,
                nct,
                rank,
            )
            continue
        seen_ncts.add(nct)
        key = (qid, nct)
        trial_record = load_trial(con, nct)
        if trial_record is None:
            logger.warning("qid=%d nct=%s not found in SQLite — skipping row", qid, nct)
            continue

        if key in label_map:
            lbl = label_map[key]
            relevance = lbl["relevance"]
            reason = lbl["reason"]
            labeler = lbl["labeler"]
        else:
            relevance, reason, in_tok, out_tok = label_pair(client, query, trial_record)
            labeler = LABELER_MODEL
            label_token_counter["input_tokens"] += in_tok
            label_token_counter["output_tokens"] += out_tok
            label_token_counter["n_labels"] += 1
            # Cache the new label so later queries with the same (qid, nct)
            # within this run re-use it for free (rare but possible).
            label_map[key] = {
                "relevance": relevance,
                "reason": reason,
                "labeler": labeler,
            }

        rows.append(
            {
                "query_id": qid,
                "query": query,
                "category": category,
                "rank": rank,
                "nct_id": nct,
                "trial_title": trial_record.get("title"),
                "trial_conditions": trial_record.get("conditions"),
                "trial_phase": trial_record.get("phase"),
                "trial_status": trial_record.get("status"),
                "pipeline_score": None,  # agent has no blender score
                "cross_encoder_score": None,  # not exposed by agent path
                "relevance": relevance,
                "reason": reason,
                "labeler": labeler,
                # Per-query agent metrics (duplicated per row for analysis ease).
                "agent_iters_used": meta["iters"],
                "agent_tool_sequence": meta["tools"],
                "agent_input_tokens": meta["input_tokens"],
                "agent_output_tokens": meta["output_tokens"],
                "agent_cache_read_input_tokens": meta.get("cache_read_input_tokens", 0),
                "agent_cache_creation_input_tokens": meta.get("cache_creation_input_tokens", 0),
                "agent_cost_usd": meta["cost_usd"],
                "agent_latency_ms": meta["latency_ms"],
                "agent_error": error_str,
            }
        )
    return rows


def compute_per_query_metrics(rows: list[dict]) -> dict[str, float]:
    """NDCG@5, NDCG@10, MRR for one query's rows.

    Uses the project's metrics API which takes (ordered ids, relevance
    dict). MRR's "relevant" threshold is ``relevance >= 2`` — matches the
    rubric where 2 = "matches condition" and 3 = "highly relevant".
    """
    ranked = sorted(rows, key=lambda r: int(r["rank"]))
    result_ids = [str(r["nct_id"]) for r in ranked]
    relevance_scores = {str(r["nct_id"]): float(r["relevance"]) for r in ranked}
    relevant_ids = {r["nct_id"] for r in ranked if int(r["relevance"]) >= 2}
    return {
        "ndcg5": ndcg_at_k(result_ids, relevance_scores, k=5),
        "ndcg10": ndcg_at_k(result_ids, relevance_scores, k=10),
        "mrr": mrr(result_ids, relevant_ids),
    }


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[k]


def print_summary(
    per_query: list[dict],
    label_tokens: dict,
    total_queries_in_run: int,
) -> None:
    if not per_query:
        print("No queries ran in this invocation.")
        return

    n = len(per_query)
    total_agent_cost = sum(m["cost_usd"] for m in per_query)
    label_cost = (
        label_tokens["input_tokens"] * _HAIKU_IN_RATE
        + label_tokens["output_tokens"] * _HAIKU_OUT_RATE
    )

    iters = [m["iters"] for m in per_query if m["iters"] > 0]
    mean_iters = sum(iters) / len(iters) if iters else 0.0
    p95_latency = percentile([m["latency_ms"] for m in per_query], 0.95)
    p50_latency = percentile([m["latency_ms"] for m in per_query], 0.50)

    tool_counter: Counter = Counter()
    for m in per_query:
        for t in m["tools"]:
            tool_counter[t] += 1

    print(f"\n{'=' * 78}")
    print("PHASE C2 — agent eval summary (this invocation)")
    print(f"{'=' * 78}")
    print(f"  queries run:                {n} / {total_queries_in_run}")
    print(f"  total agent API cost:       ${total_agent_cost:.4f}")
    print(
        f"  total labeling cost:        ${label_cost:.4f}  "
        f"({label_tokens['n_labels']} new (q,t) labels)"
    )
    print(f"  TOTAL cost (this run):      ${total_agent_cost + label_cost:.4f}")
    print(
        f"  mean iters / query (agent): {mean_iters:.2f}  "
        f"({'ok' if 3 <= mean_iters <= 5 else 'ALARM — target 3–5'})"
    )
    print(f"  p50 latency:                {p50_latency:.0f} ms")
    print(f"  p95 latency:                {p95_latency:.0f} ms")
    print("\n  tool-call frequency:")
    for tool, count in tool_counter.most_common():
        print(f"    {tool:<28} {count}")

    # Per-category NDCG + MRR
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in per_query:
        by_cat[m["category"]].append(m)

    print("\n  per-category NDCG@5 / NDCG@10 / MRR:")
    print(f"    {'category':<14} {'n':>3}  {'NDCG@5':>8}  {'NDCG@10':>8}  {'MRR':>8}")
    for cat in sorted(by_cat):
        ms = by_cat[cat]
        n_cat = len(ms)
        if n_cat == 0:
            continue
        ndcg5 = sum(x["ndcg5"] for x in ms) / n_cat
        ndcg10 = sum(x["ndcg10"] for x in ms) / n_cat
        mrr = sum(x["mrr"] for x in ms) / n_cat
        print(f"    {cat:<14} {n_cat:>3}  {ndcg5:>8.3f}  {ndcg10:>8.3f}  {mrr:>8.3f}")

    overall_n5 = sum(x["ndcg5"] for x in per_query) / n
    overall_n10 = sum(x["ndcg10"] for x in per_query) / n
    overall_mrr = sum(x["mrr"] for x in per_query) / n
    print(f"\n  OVERALL (mean over {n} queries):")
    print(f"    NDCG@5  = {overall_n5:.3f}")
    print(f"    NDCG@10 = {overall_n10:.3f}")
    print(f"    MRR     = {overall_mrr:.3f}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C2 — eval the agent arm on every query in a labeled set."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Input labeled JSONL (e.g. data/evaluation/full_labeled_dataset.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL — one row per (query, top result).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N queries (sanity slice).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip query_ids already present in --output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    label_map, queries_meta = load_labels(args.labels)
    logger.info(
        "loaded %d (qid, nct) labels from %s across %d queries",
        len(label_map),
        args.labels,
        len(queries_meta),
    )

    queries = sorted(queries_meta.items())
    if args.limit:
        queries = queries[: args.limit]
        logger.info("--limit %d → %d queries", args.limit, len(queries))

    done = load_done_qids(args.output) if args.resume else set()
    if done:
        logger.info("resume: skipping %d qids already in %s", len(done), args.output)
    to_run = [(qid, q, cat) for qid, (q, cat) in queries if qid not in done]
    logger.info("running %d / %d queries", len(to_run), len(queries))

    if not to_run:
        logger.info("nothing to do — all queries already done")
        return 0

    # Anthropic client for new-NCT labeling (separate from the agent's own
    # Haiku client; reused across all label_pair calls in this run).
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    from TrialMine.agents.tools import _db_conn

    con = _db_conn()

    graph = build_pipeline()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    label_token_counter: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "n_labels": 0,
    }
    per_query_metrics: list[dict] = []

    try:
        with open(args.output, "a") as out_f:
            for i, (qid, query, category) in enumerate(to_run, start=1):
                logger.info(
                    "[%d/%d] qid=%d cat=%s query=%r",
                    i,
                    len(to_run),
                    qid,
                    category,
                    query[:80],
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

                if not rows:
                    logger.warning("qid=%d returned 0 rows", qid)
                    continue

                for r in rows:
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()

                # Per-query metric snapshot (re-uses the agent_* fields
                # that are duplicated across rows for this qid).
                first = rows[0]
                metrics = compute_per_query_metrics(rows)
                per_query_metrics.append(
                    {
                        "qid": qid,
                        "category": category,
                        "iters": first["agent_iters_used"],
                        "tools": first["agent_tool_sequence"],
                        "cost_usd": first["agent_cost_usd"],
                        "latency_ms": first["agent_latency_ms"],
                        "n_results": len(rows),
                        **metrics,
                    }
                )
                logger.info(
                    "  → arm=agent n_results=%d iters=%d cost=$%.4f "
                    "ndcg@5=%.3f ndcg@10=%.3f mrr=%.3f",
                    len(rows),
                    first["agent_iters_used"],
                    first["agent_cost_usd"],
                    metrics["ndcg5"],
                    metrics["ndcg10"],
                    metrics["mrr"],
                )
    finally:
        con.close()

    print_summary(per_query_metrics, label_token_counter, len(queries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
