"""Run the eligibility parser on 20 real trials and print a results table.

Usage:
    python scripts/demo_eligibility_parser.py [--limit 20]

The 20 trials are sampled to span the format space:
- 5 short (<1k chars)
- 10 medium (1k-5k chars)
- 5 long (>5k chars, including ones with escape leaks)
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Make the src/ package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from TrialMine.features.eligibility import EligibilityParser  # noqa: E402

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "trials.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("demo_eligibility_parser")


def sample_trials(db_path: Path, total: int = 20) -> list[dict]:
    """Pull a mix of short / medium / long trials with non-trivial eligibility."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        n_short = max(1, total // 4)
        n_long = max(1, total // 4)
        n_medium = total - n_short - n_long
        rows: list[dict] = []
        for length_clause, n in [
            ("length(eligibility_criteria) BETWEEN 200 AND 1000", n_short),
            ("length(eligibility_criteria) BETWEEN 1000 AND 5000", n_medium),
            ("length(eligibility_criteria) > 5000", n_long),
        ]:
            cur = conn.execute(
                f"""
                SELECT nct_id, eligibility_criteria, min_age, max_age, sex
                FROM trials
                WHERE eligibility_criteria IS NOT NULL
                  AND {length_clause}
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (n,),
            )
            rows.extend(dict(r) for r in cur.fetchall())
        return rows
    finally:
        conn.close()


def format_table(rows: list[dict]) -> str:
    """Render the results as a fixed-width table."""
    header = (
        f"{'nct_id':<12} | {'min_age':>8} | {'max_age':>8} | {'sex':<6} | "
        f"{'cond':>5} | {'excl_c':>6} | {'treat':>5} | {'excl_t':>6} | "
        f"{'src':<13} | {'conf':>5}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['nct_id']:<12} | "
            f"{_fmt_num(r['min_age_years']):>8} | "
            f"{_fmt_num(r['max_age_years']):>8} | "
            f"{(r['sex'] or '-'):<6} | "
            f"{len(r['required_conditions']):>5} | "
            f"{len(r['excluded_conditions']):>6} | "
            f"{len(r['required_prior_treatments']):>5} | "
            f"{len(r['excluded_prior_treatments']):>6} | "
            f"{r['section_source']:<13} | "
            f"{r['parse_confidence']:>5.2f}"
        )
    return "\n".join(lines)


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "None"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.2f}"


def aggregate_stats(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "trials": n,
        "min_age_populated_pct": 100.0 * sum(r["min_age_years"] is not None for r in rows) / n,
        "max_age_populated_pct": 100.0 * sum(r["max_age_years"] is not None for r in rows) / n,
        "mean_required_conditions": sum(len(r["required_conditions"]) for r in rows) / n,
        "mean_excluded_conditions": sum(len(r["excluded_conditions"]) for r in rows) / n,
        "mean_required_treatments": sum(
            len(r["required_prior_treatments"]) for r in rows
        ) / n,
        "section_source": dict(Counter(r["section_source"] for r in rows)),
        "mean_confidence": sum(r["parse_confidence"] for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20, help="Number of trials to sample")
    ap.add_argument(
        "--show-buckets", action="store_true", help="Print full bucket contents per trial"
    )
    args = ap.parse_args()

    logger.info("Sampling %d trials from %s", args.limit, DB_PATH)
    trials = sample_trials(DB_PATH, total=args.limit)
    logger.info("Got %d trials", len(trials))

    logger.info("Loading SciSpacy model (one-time, ~3s)...")
    parser = EligibilityParser(model_name="en_core_sci_lg")
    parser._load_nlp()  # force load now so timing of parse() is clean

    results: list[dict] = []
    for t in trials:
        profile = parser.parse(
            t["eligibility_criteria"],
            min_age_col=t["min_age"],
            max_age_col=t["max_age"],
            sex_col=t["sex"],
        )
        d = profile.model_dump()
        d["nct_id"] = t["nct_id"]
        results.append(d)

    print()
    print(format_table(results))
    print()
    stats = aggregate_stats(results)
    print("Aggregate stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.2f}")
        else:
            print(f"  {k:<28} {v}")

    if args.show_buckets:
        print("\nBuckets per trial:")
        for r in results:
            print(f"\n--- {r['nct_id']} ({r['section_source']}, conf={r['parse_confidence']:.2f}) ---")
            print(f"  required_conditions ({len(r['required_conditions'])}): "
                  f"{r['required_conditions'][:8]}")
            print(f"  excluded_conditions ({len(r['excluded_conditions'])}): "
                  f"{r['excluded_conditions'][:8]}")
            print(f"  required_prior_treatments ({len(r['required_prior_treatments'])}): "
                  f"{r['required_prior_treatments'][:8]}")
            print(f"  excluded_prior_treatments ({len(r['excluded_prior_treatments'])}): "
                  f"{r['excluded_prior_treatments'][:8]}")


if __name__ == "__main__":
    main()
