"""Smoke-test the LangGraph search pipeline end-to-end.

Runs five representative queries — including a garbage input — and prints
the parsed PatientProfile, the full agent_trace step by step, the top
results with template explanations, and timings. Designed to surface both
happy-path behavior (with ES + models up) and the fallback path (with ES
down). Always exits cleanly even when a query produces empty results.

Run::

    docker start es                       # for happy path
    OMP_NUM_THREADS=1 python scripts/test_pipeline.py

Stop ES (`docker stop es`) and re-run to exercise the fallback edge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Quiet third-party logging chatter that would drown out the trace output.
for noisy in (
    "httpx",
    "httpcore",
    "elasticsearch",
    "elastic_transport",
    "elastic_transport.transport",
    "urllib3",
):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
)

from TrialMine.agents.pipeline import build_pipeline, search  # noqa: E402

QUERIES: list[str] = [
    "I'm 62 with stage 3 non-small cell lung cancer, tried carboplatin, looking for immunotherapy near Boston",
    "breast cancer trials",
    "advanced melanoma that has spread to the brain",
    "stage 4 pancreatic cancer?",
    "asdfghjkl",
]


def _profile_summary(profile: dict | None) -> str:
    """Render a single-line summary of the parsed PatientProfile."""
    if not profile:
        return "(none)"
    bits = []
    for key in (
        "condition",
        "condition_stage",
        "age",
        "sex",
        "biomarkers",
        "preferences",
        "prior_treatments",
        "location",
    ):
        v = profile.get(key)
        if v not in (None, "", []):
            bits.append(f"{key}={v!r}")
    if not bits:
        bits.append(f"raw_query={profile.get('raw_query', '')!r}")
    return ", ".join(bits)


def _print_trace(trace: list[dict]) -> None:
    """Print the agent_trace one entry per line."""
    for i, entry in enumerate(trace, 1):
        step = entry.get("step", "?")
        ms = entry.get("duration_ms", "?")
        decisions = entry.get("decisions") or {}
        decisions_str = json.dumps(decisions, default=str)
        if len(decisions_str) > 240:
            decisions_str = decisions_str[:237] + "..."
        print(f"  {i:>2}. [{step:<24}] {ms:>7} ms  {decisions_str}")


def _print_top_results(results: list[dict], n: int = 3) -> None:
    """Print the top N results with their template explanations."""
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results[:n], 1):
        nct = r.get("nct_id", "?")
        title = (r.get("title") or "")[:90]
        score = r.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
        print(f"  {i}. {nct} (score={score_str})  {title}")
        print(f"      → {r.get('explanation', '')}")


async def run_one(pipeline, query: str) -> dict:
    """Run the pipeline on one query and print a structured report."""
    print("=" * 78)
    print(f"QUERY: {query}")
    print("-" * 78)
    result = await search(query, pipeline, timeout=20.0)

    print(f"PROFILE: {_profile_summary(result.get('patient_profile'))}")
    used_fallback = result.get("used_fallback", False)
    err = result.get("error")
    elapsed = result.get("elapsed_ms")
    flags = []
    if used_fallback:
        flags.append("FALLBACK")
    if err:
        flags.append(f"ERROR={err[:80]}")
    print(
        f"OUTCOME: elapsed={elapsed} ms"
        + (f"  [{' | '.join(flags)}]" if flags else "")
    )

    sr = result.get("search_results") or {}
    print(
        f"SEARCH: query_used={sr.get('query_used')!r}  "
        f"filters={sr.get('filters')}  "
        f"normalized_condition={sr.get('normalized_condition')!r}"
    )

    print("TRACE:")
    _print_trace(result.get("agent_trace") or [])

    print("TOP RESULTS:")
    _print_top_results(sr.get("results") or [])
    print()

    return result


async def main() -> int:
    print("Building pipeline (lazy: heavy resources load on first query)...\n")
    pipeline = build_pipeline()

    n_failures = 0
    n_fallback = 0
    n_empty = 0
    for q in QUERIES:
        try:
            r = await run_one(pipeline, q)
        except Exception as exc:
            print(f"  UNCAUGHT EXCEPTION on query {q!r}: {type(exc).__name__}: {exc}\n")
            n_failures += 1
            continue
        if r.get("used_fallback"):
            n_fallback += 1
        if not (r.get("search_results") or {}).get("results"):
            n_empty += 1

    print("=" * 78)
    print(
        f"SUMMARY: {len(QUERIES)} queries, "
        f"uncaught_exceptions={n_failures}, "
        f"used_fallback={n_fallback}, "
        f"empty_results={n_empty}"
    )

    # Pass criterion: no uncaught exceptions. Empty results / fallback are
    # acceptable outcomes (depend on ES being up and the query being well-formed).
    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
