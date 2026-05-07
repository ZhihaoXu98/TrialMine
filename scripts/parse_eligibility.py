"""Batch-parse eligibility_criteria for all trials and persist to SQLite.

Reads from the existing ``trials`` table, runs each trial through
:class:`EligibilityParser`, and writes the structured profile into a new
``parsed_eligibility`` table. Failures are logged per-trial and never crash
the run.

Usage:
    python scripts/parse_eligibility.py [--limit N] [--resume] [--db PATH]

Examples:
    python scripts/parse_eligibility.py --limit 1000        # quick demo
    python scripts/parse_eligibility.py --resume            # full run, skip done
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

# Make the src/ package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from TrialMine.features.eligibility import EligibilityParser  # noqa: E402

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "trials.db"
PARSER_VERSION = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("parse_eligibility")


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS parsed_eligibility (
    nct_id                      TEXT PRIMARY KEY,
    min_age_years               REAL,
    max_age_years               REAL,
    sex                         TEXT,
    required_conditions         TEXT,
    excluded_conditions         TEXT,
    required_prior_treatments   TEXT,
    excluded_prior_treatments   TEXT,
    raw_inclusion               TEXT,
    raw_exclusion               TEXT,
    parse_confidence            REAL,
    section_source              TEXT,
    parser_version              TEXT,
    parsed_at                   TEXT
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO parsed_eligibility VALUES (
    :nct_id, :min_age_years, :max_age_years, :sex,
    :required_conditions, :excluded_conditions,
    :required_prior_treatments, :excluded_prior_treatments,
    :raw_inclusion, :raw_exclusion,
    :parse_confidence, :section_source,
    :parser_version, :parsed_at
)
"""


def fetch_trials(
    conn: sqlite3.Connection, *, limit: int | None, skip_ids: set[str]
) -> list[sqlite3.Row]:
    """Fetch trials with non-null eligibility, optionally skipping resumed ids."""
    query = """
        SELECT nct_id, eligibility_criteria, min_age, max_age, sex
        FROM trials
        WHERE eligibility_criteria IS NOT NULL
          AND length(eligibility_criteria) > 0
    """
    params: list = []
    if skip_ids:
        # SQLite has a parameter limit; build IN clause carefully for small skip sets.
        # For large skip sets we filter in Python.
        pass
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    if skip_ids:
        rows = [r for r in rows if r["nct_id"] not in skip_ids]
    return rows


def existing_ids(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT nct_id FROM parsed_eligibility")
    return {r[0] for r in cur.fetchall()}


def parse_one(parser: EligibilityParser, row: sqlite3.Row) -> dict:
    """Parse one trial → dict ready for INSERT. Always returns; never raises."""
    nct_id = row["nct_id"]
    profile = parser.parse(
        row["eligibility_criteria"],
        min_age_col=row["min_age"],
        max_age_col=row["max_age"],
        sex_col=row["sex"],
    )
    return {
        "nct_id": nct_id,
        "min_age_years": profile.min_age_years,
        "max_age_years": profile.max_age_years,
        "sex": profile.sex,
        "required_conditions": json.dumps(profile.required_conditions),
        "excluded_conditions": json.dumps(profile.excluded_conditions),
        "required_prior_treatments": json.dumps(profile.required_prior_treatments),
        "excluded_prior_treatments": json.dumps(profile.excluded_prior_treatments),
        "raw_inclusion": profile.raw_inclusion,
        "raw_exclusion": profile.raw_exclusion,
        "parse_confidence": profile.parse_confidence,
        "section_source": profile.section_source,
        "parser_version": PARSER_VERSION,
        "parsed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def summarize(conn: sqlite3.Connection) -> None:
    """Print summary stats from the parsed_eligibility table."""
    cur = conn.execute(
        """
        SELECT
            COUNT(*)                                         AS total,
            SUM(min_age_years IS NOT NULL)                   AS n_min_age,
            SUM(max_age_years IS NOT NULL)                   AS n_max_age,
            SUM(json_array_length(required_conditions) > 0)  AS n_with_cond,
            SUM(json_array_length(required_prior_treatments) > 0) AS n_with_treat,
            AVG(parse_confidence)                            AS avg_conf,
            AVG(json_array_length(required_conditions))      AS avg_n_cond,
            AVG(json_array_length(excluded_conditions))      AS avg_n_excl_cond,
            AVG(json_array_length(required_prior_treatments)) AS avg_n_treat
        FROM parsed_eligibility
        """
    )
    row = cur.fetchone()
    total = row[0] or 1  # avoid div by zero in the print line below
    src_counts = dict(
        conn.execute(
            "SELECT section_source, COUNT(*) FROM parsed_eligibility GROUP BY section_source"
        ).fetchall()
    )

    print("\n=== parsed_eligibility summary ===")
    print(f"  total parsed              : {row[0]}")
    print(f"  with min_age              : {row[1]}  ({100.0 * row[1] / total:.1f}%)")
    print(f"  with max_age              : {row[2]}  ({100.0 * row[2] / total:.1f}%)")
    print(f"  with required_conditions  : {row[3]}  ({100.0 * row[3] / total:.1f}%)")
    print(f"  with required_treatments  : {row[4]}  ({100.0 * row[4] / total:.1f}%)")
    print(f"  avg parse_confidence      : {row[5]:.3f}")
    print(f"  avg n_conditions          : {row[6]:.1f}")
    print(f"  avg n_excluded_conditions : {row[7]:.1f}")
    print(f"  avg n_treatments          : {row[8]:.1f}")
    print(f"  section_source histogram  : {src_counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DB_PATH, help="Path to trials.db")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of trials to parse")
    ap.add_argument(
        "--resume", action="store_true", help="Skip nct_ids already in parsed_eligibility"
    )
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()

        skip = existing_ids(conn) if args.resume else set()
        if skip:
            logger.info("Resume mode: skipping %d already-parsed trials", len(skip))

        rows = fetch_trials(conn, limit=args.limit, skip_ids=skip)
        logger.info("Fetched %d trials to parse", len(rows))
        if not rows:
            summarize(conn)
            return

        logger.info("Loading SciSpacy model...")
        parser = EligibilityParser(model_name="en_core_sci_lg")
        parser._load_nlp()

        n_ok = 0
        n_fail = 0
        t0 = time.time()
        commit_every = 200
        pending: list[dict] = []
        for row in tqdm(rows, desc="parsing", unit="trial"):
            try:
                pending.append(parse_one(parser, row))
                n_ok += 1
            except Exception as exc:  # never crash the run
                n_fail += 1
                logger.warning("Failed on %s: %s", row["nct_id"], exc)

            if len(pending) >= commit_every:
                conn.executemany(INSERT_SQL, pending)
                conn.commit()
                pending = []

        if pending:
            conn.executemany(INSERT_SQL, pending)
            conn.commit()

        dt = time.time() - t0
        logger.info(
            "Done. ok=%d fail=%d, %.1f trials/sec (%.1f sec total)",
            n_ok,
            n_fail,
            n_ok / max(dt, 1e-3),
            dt,
        )

        summarize(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
