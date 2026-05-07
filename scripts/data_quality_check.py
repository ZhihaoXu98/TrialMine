"""Data quality report for the indexed clinical-trial corpus.

Reads ``data/trials.db`` (the parsed SQLite store written by
``scripts/download_data.py``) and emits:

* **Field coverage** — % of rows where each canonical field is non-null
  / non-empty. Catches silent drop-outs from the downloader (a parser
  regression that suddenly stops capturing ``brief_summary`` shows up
  here as a coverage cliff).
* **Freshness** — most recent ``start_date`` and ``completion_date`` in
  the corpus, plus the file mtime of the SQLite database itself. The
  pipeline expects fresh ClinicalTrials.gov data; staleness here means
  the downloader hasn't run.
* **Suspicious records** — counts (and a few NCT examples) for each
  pattern below. These are heuristics, not bugs in themselves; rising
  counts indicate upstream drift.

Output: pretty stdout summary plus
``data/data_quality_report.json`` with the same data structured for
machine consumers (CI alerts, dashboards). Returns a non-zero exit
code only when the SQLite database is missing — quality findings are
informational, not gating.

Usage:

    python scripts/data_quality_check.py
    python scripts/data_quality_check.py --db data/trials.db --out data/quality.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("data_quality_check")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "trials.db"
DEFAULT_OUT = REPO_ROOT / "data" / "data_quality_report.json"

# Fields we expect populated on every row (per CLAUDE.md schema).
COVERAGE_FIELDS: tuple[str, ...] = (
    "title",
    "brief_summary",
    "conditions",
    "interventions",
    "eligibility_criteria",
    "min_age",
    "max_age",
    "sex",
    "phase",
    "status",
    "enrollment",
    "start_date",
    "completion_date",
    "sponsor",
    "locations",
)

# Suspicious-record query templates. ``{table}`` is filled at runtime so
# the same script can target a swapped-out fixtures DB in tests.
SUSPICIOUS_QUERIES: tuple[tuple[str, str, str], ...] = (
    (
        "missing_title",
        "Trials with empty / null title",
        "SELECT nct_id FROM {table} WHERE title IS NULL OR trim(title) = ''",
    ),
    (
        "empty_conditions",
        "Trials with empty conditions field (parser likely failed)",
        "SELECT nct_id FROM {table} WHERE conditions IS NULL OR trim(conditions) = ''",
    ),
    (
        "recruiting_no_eligibility",
        "RECRUITING trials with no eligibility criteria text",
        (
            "SELECT nct_id FROM {table} "
            "WHERE status = 'RECRUITING' "
            "AND (eligibility_criteria IS NULL OR length(eligibility_criteria) < 20)"
        ),
    ),
    (
        "missing_age_bounds",
        "Both min_age and max_age are null (likely parser miss)",
        "SELECT nct_id FROM {table} WHERE min_age IS NULL AND max_age IS NULL",
    ),
    (
        "implausible_enrollment",
        "Enrollment outside [1, 100000]",
        (
            "SELECT nct_id FROM {table} "
            "WHERE enrollment IS NOT NULL "
            "AND (enrollment < 1 OR enrollment > 100000)"
        ),
    ),
    (
        "unknown_status",
        "Status is not one of the known ClinicalTrials.gov values",
        (
            "SELECT nct_id FROM {table} WHERE status NOT IN ("
            "'RECRUITING', 'COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED', "
            "'NOT_YET_RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION', "
            "'AVAILABLE', 'NO_LONGER_AVAILABLE', 'TEMPORARILY_NOT_AVAILABLE', "
            "'APPROVED_FOR_MARKETING', 'UNKNOWN', 'UNKNOWN_STATUS')"
        ),
    ),
)


def _coverage(conn: sqlite3.Connection, table: str, total: int) -> dict[str, dict[str, Any]]:
    """Return ``{field: {non_null: N, coverage: 0..1}}`` for each tracked field."""
    out: dict[str, dict[str, Any]] = {}
    for field in COVERAGE_FIELDS:
        # ``trim(...)``+empty-string check catches the placeholder rows
        # the downloader writes when an upstream field is the empty string.
        sql = (
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE {field} IS NOT NULL "
            f"AND ({field} != '' OR typeof({field}) != 'text')"
        )
        non_null = conn.execute(sql).fetchone()[0]
        out[field] = {
            "non_null": int(non_null),
            "coverage": (non_null / total) if total else 0.0,
        }
    return out


def _freshness(conn: sqlite3.Connection, table: str, db_path: Path) -> dict[str, str | None]:
    """Latest dates in the corpus + DB file mtime."""
    def _max(col: str) -> str | None:
        row = conn.execute(
            f"SELECT MAX({col}) FROM {table} WHERE {col} IS NOT NULL"
        ).fetchone()
        return row[0] if row and row[0] else None

    return {
        "max_start_date": _max("start_date"),
        "max_completion_date": _max("completion_date"),
        "db_mtime_utc": datetime.fromtimestamp(
            db_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _suspicious(
    conn: sqlite3.Connection, table: str, examples_per_pattern: int = 3
) -> dict[str, dict[str, Any]]:
    """Run each suspicious-record query and capture count + a few example NCT ids."""
    out: dict[str, dict[str, Any]] = {}
    for key, description, sql in SUSPICIOUS_QUERIES:
        rendered = sql.format(table=table)
        count = conn.execute(
            f"SELECT COUNT(*) FROM ({rendered}) AS sub"
        ).fetchone()[0]
        examples = [
            row[0]
            for row in conn.execute(
                f"{rendered} LIMIT {examples_per_pattern}"
            ).fetchall()
        ]
        out[key] = {
            "description": description,
            "count": int(count),
            "examples": examples,
        }
    return out


def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print the report to stdout."""
    total = report["total_rows"]
    print()
    print(f"=== TrialMine data quality report ({total:,} rows) ===")
    print()

    print("Field coverage:")
    for field, stats in sorted(
        report["field_coverage"].items(), key=lambda x: x[1]["coverage"]
    ):
        bar = "█" * int(stats["coverage"] * 20)
        print(
            f"  {field:<22} {stats['coverage']:>6.1%}  "
            f"{bar:<20}  ({stats['non_null']:,} non-null)"
        )
    print()

    fresh = report["freshness"]
    print("Freshness:")
    print(f"  latest start_date      = {fresh['max_start_date']}")
    print(f"  latest completion_date = {fresh['max_completion_date']}")
    print(f"  trials.db mtime (UTC)  = {fresh['db_mtime_utc']}")
    print()

    print("Suspicious records:")
    for key, payload in report["suspicious_records"].items():
        marker = "⚠" if payload["count"] > 0 else "✓"
        print(
            f"  {marker} {key:<28} {payload['count']:>8,}  "
            f"({payload['description']})"
        )
        if payload["examples"]:
            joined = ", ".join(payload["examples"])
            print(f"      examples: {joined}")
    print()


def build_report(
    db_path: Path = DEFAULT_DB, table: str = "trials"
) -> dict[str, Any]:
    """Build the quality report dict (pure function — no I/O outside SQLite)."""
    if not db_path.exists():
        raise FileNotFoundError(f"trials database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        total_row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        total = int(total_row[0]) if total_row else 0

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "table": table,
            "total_rows": total,
            "field_coverage": _coverage(conn, table, total),
            "freshness": _freshness(conn, table, db_path),
            "suspicious_records": _suspicious(conn, table),
        }
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to trials.db (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write the JSON report (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--table", default="trials", help="SQLite table name (default: trials)"
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()

    try:
        report = build_report(db_path=args.db, table=args.table)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
