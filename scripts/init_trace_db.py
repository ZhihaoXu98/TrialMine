"""Initialise the SQLite trace store at ``data/agent_runs.db``.

Phase B1 of ``docs/build_agent.md`` — creates the three tables
(``agent_runs``, ``agent_stages``, ``agent_tool_calls``) plus their
indexes if they don't already exist. The script is **idempotent**: every
``CREATE`` carries ``IF NOT EXISTS``, so re-running on an already-built
database is a no-op (and only prints the current row counts).

Schema is documented inline in :data:`SCHEMA`. The file is intentionally
separate from ``trials.db`` so it can be rotated, dropped, or copied
independently for offline analysis.

Usage::

    python scripts/init_trace_db.py
    python scripts/init_trace_db.py --db /tmp/agent_runs.db   # override path
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths + logging                                                              #
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "agent_runs.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #


SCHEMA: str = """
CREATE TABLE IF NOT EXISTS agent_runs (
    query_id          TEXT PRIMARY KEY,    -- uuid4 generated per call
    raw_query         TEXT NOT NULL,
    ts                INTEGER NOT NULL,    -- unix ms
    arm               TEXT NOT NULL,       -- 'rule' | 'agent' | 'fallback'
    route_reason      TEXT,                -- from route_decision
    pipeline_kind     TEXT,                -- from rule arm's retrieve step
    used_fallback     INTEGER NOT NULL DEFAULT 0,
    total_ms          REAL,
    n_results         INTEGER,
    error             TEXT,                -- nullable
    route_signals     TEXT                 -- JSON blob of routing signals
);
CREATE INDEX IF NOT EXISTS idx_runs_ts   ON agent_runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_arm  ON agent_runs(arm);
CREATE INDEX IF NOT EXISTS idx_runs_kind ON agent_runs(pipeline_kind);

CREATE TABLE IF NOT EXISTS agent_stages (
    query_id     TEXT NOT NULL,
    stage        TEXT NOT NULL,            -- e.g. 'parse_query', 'tool_call'
    duration_ms  REAL NOT NULL,
    seq          INTEGER NOT NULL,         -- order within the trace
    FOREIGN KEY (query_id) REFERENCES agent_runs(query_id)
);
CREATE INDEX IF NOT EXISTS idx_stages_qid   ON agent_stages(query_id);
CREATE INDEX IF NOT EXISTS idx_stages_stage ON agent_stages(stage);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    query_id     TEXT NOT NULL,
    iter_n       INTEGER NOT NULL,         -- 1-indexed cycle in the loop
    tool_name    TEXT NOT NULL,
    ok           INTEGER NOT NULL,         -- 0/1
    duration_ms  REAL,
    args_summary TEXT,                     -- truncated JSON
    FOREIGN KEY (query_id) REFERENCES agent_runs(query_id)
);
CREATE INDEX IF NOT EXISTS idx_tools_qid  ON agent_tool_calls(query_id);
CREATE INDEX IF NOT EXISTS idx_tools_name ON agent_tool_calls(tool_name);
""".strip()


_TABLES: tuple[str, ...] = ("agent_runs", "agent_stages", "agent_tool_calls")


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def init_db(db_path: Path = DEFAULT_DB_PATH) -> dict[str, int]:
    """Create tables + indexes if missing, return row counts per table.

    Wrapped in a single ``with con:`` transaction so partial schema
    creation is atomic — either every CREATE in :data:`SCHEMA` runs or
    none of them do. ``IF NOT EXISTS`` makes the script safe to re-run.

    Args:
        db_path: SQLite file. Created (with parent dir) if absent.

    Returns:
        Mapping ``{table_name: row_count}`` for the three trace tables.

    Raises:
        sqlite3.Error: If schema execution fails (the transaction
            rolls back automatically before this propagates).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    was_new = not db_path.exists()
    logger.info(
        "Initialising trace DB at %s (new=%s)",
        db_path,
        was_new,
    )

    con = sqlite3.connect(db_path)
    try:
        # ``executescript`` commits any pending transaction implicitly,
        # so a leading BEGIN here would be a no-op. The semicolon-split
        # explicit execute path lets us run inside the ``with con:``
        # context manager, which commits on success and rolls back on
        # exception — atomic schema creation.
        statements = [
            stmt.strip() for stmt in SCHEMA.split(";") if stmt.strip()
        ]
        with con:
            for stmt in statements:
                con.execute(stmt)

        counts: dict[str, int] = {}
        for table in _TABLES:
            row = con.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 — table whitelist
            ).fetchone()
            counts[table] = int(row[0])
    finally:
        con.close()

    logger.info(
        "Schema ready: %s",
        ", ".join(f"{t}={n}" for t, n in counts.items()),
    )
    return counts


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise the TrialMine agent trace SQLite DB.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite file (default: {DEFAULT_DB_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = init_db(args.db)
    print(f"Trace DB ready at {args.db}")
    for table, n in counts.items():
        print(f"  {table:<20} rows={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
