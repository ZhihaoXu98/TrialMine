"""SQLite trace writer for the LangGraph agent pipeline.

Phase B2 of ``docs/build_agent.md`` — persists one row per
``pipeline.search()`` invocation into ``data/agent_runs.db`` so an
operator can answer per-query questions ("show me the slowest agent
queries last hour", "what tools did this query call?") that the
Prometheus counters can't.

The writer is **synchronous + transactional + non-raising**:

* synchronous because SQLite writes a handful of rows in well under a
  millisecond; an async queue is over-engineered for this scale.
* transactional via ``with con:`` so partial inserts are impossible —
  either the run + every stage + every tool_call land together or none
  do.
* non-raising: any ``sqlite3.Error`` is caught and logged at warning
  level. The trace store must never break the search response.

The schema is duplicated from ``scripts/init_trace_db.py`` rather than
imported — scripts/ isn't on the import path, and CREATE IF NOT EXISTS
makes accidental drift a no-op. See B1 for the canonical schema source.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths + module state                                                         #
# --------------------------------------------------------------------------- #


DEFAULT_DB_PATH: Path = Path(os.getenv("TRIALMINE_TRACE_DB", "data/agent_runs.db"))

# One-shot init guard so the FIRST write per process triggers the
# schema check, not every write. Pipeline.search() (Phase B3) lazily
# calls write_trace, so paying the schema check once per process —
# not per import — also avoids polluting random CWDs with empty DB
# files in test / script runs.
_initialized: bool = False
_init_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Schema (duplicated from scripts/init_trace_db.py — see module docstring)    #
# --------------------------------------------------------------------------- #


_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS agent_runs (
    query_id          TEXT PRIMARY KEY,
    raw_query         TEXT NOT NULL,
    ts                INTEGER NOT NULL,
    arm               TEXT NOT NULL,
    route_reason      TEXT,
    pipeline_kind     TEXT,
    used_fallback     INTEGER NOT NULL DEFAULT 0,
    total_ms          REAL,
    n_results         INTEGER,
    error             TEXT,
    route_signals     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts   ON agent_runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_arm  ON agent_runs(arm);
CREATE INDEX IF NOT EXISTS idx_runs_kind ON agent_runs(pipeline_kind);

CREATE TABLE IF NOT EXISTS agent_stages (
    query_id     TEXT NOT NULL,
    stage        TEXT NOT NULL,
    duration_ms  REAL NOT NULL,
    seq          INTEGER NOT NULL,
    FOREIGN KEY (query_id) REFERENCES agent_runs(query_id)
);
CREATE INDEX IF NOT EXISTS idx_stages_qid   ON agent_stages(query_id);
CREATE INDEX IF NOT EXISTS idx_stages_stage ON agent_stages(stage);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    query_id     TEXT NOT NULL,
    iter_n       INTEGER NOT NULL,
    tool_name    TEXT NOT NULL,
    ok           INTEGER NOT NULL,
    duration_ms  REAL,
    args_summary TEXT,
    FOREIGN KEY (query_id) REFERENCES agent_runs(query_id)
);
CREATE INDEX IF NOT EXISTS idx_tools_qid  ON agent_tool_calls(query_id);
CREATE INDEX IF NOT EXISTS idx_tools_name ON agent_tool_calls(tool_name);
""".strip()


# Cap on the args_summary JSON payload to keep trace rows compact.
_ARGS_SUMMARY_MAX_CHARS = 2000


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def init_db_if_missing(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create the trace DB + schema if absent. Idempotent.

    Safe to call multiple times: every CREATE carries IF NOT EXISTS, and
    the parent directory is mkdir'd with ``exist_ok=True``. Intended to
    be called once at startup by :func:`write_trace`'s lazy guard (see
    Phase B3 of the runbook for why we don't fire this at module import
    time).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    try:
        statements = [s.strip() for s in _SCHEMA.split(";") if s.strip()]
        with con:
            for stmt in statements:
                con.execute(stmt)
    finally:
        con.close()


def write_trace(
    query_id: str,
    raw_query: str,
    arm: str,
    route_reason: str | None,
    route_signals: dict | None,
    pipeline_kind: str | None,
    used_fallback: bool,
    total_ms: float,
    n_results: int,
    error: str | None,
    stages: list[dict],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Insert one ``agent_runs`` row + N ``agent_stages`` + M ``agent_tool_calls``.

    Synchronous. Single transaction via ``with con:`` so partial writes
    aren't possible. Any :class:`sqlite3.Error` (DB locked beyond
    busy_timeout, missing schema, etc.) is caught, logged at WARNING
    level, and swallowed — the trace store must never break the search
    response.

    Args:
        query_id: UUID generated per call by :func:`pipeline.search`.
        raw_query: The patient's free-text query.
        arm: ``"rule"`` / ``"agent"`` / ``"fallback"`` / ``"unknown"``.
        route_reason: From the ``route_decision`` node's signals.
        route_signals: Routing signals dict; JSON-encoded into the
            ``route_signals`` column. ``None`` is stored as SQL NULL.
        pipeline_kind: From the rule arm's ``retrieve`` step
            (``"full_pipeline"`` / ``"hybrid_only_disabled"`` /
            ``"hybrid_only_after_timeout"``); ``None`` for agent / fallback.
        used_fallback: Whether the pipeline degraded to ``fallback_search``.
        total_ms: Wall-clock for the whole ``pipeline.search()`` call.
        n_results: Length of the final results list.
        error: Top-level pipeline error string; ``None`` on success.
        stages: The raw ``agent_trace`` list from ``pipeline.search()``.
            Each entry should be a dict with ``step`` / ``duration_ms`` /
            ``decisions`` keys; malformed entries are skipped.
        db_path: SQLite file. Defaults to :data:`DEFAULT_DB_PATH`.
    """
    # Lazy schema init on the first write per process. Cheap after the
    # flag flips; protected by a lock so concurrent first-writes don't
    # both try to create the schema (the lock is rare-path; subsequent
    # writes bypass it via the ``_initialized`` short-circuit).
    global _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:
                try:
                    init_db_if_missing(db_path)
                    _initialized = True
                except sqlite3.Error:
                    logger.warning(
                        "trace_store: schema init failed for %s; this trace will be dropped",
                        db_path,
                        exc_info=True,
                    )
                    return

    if not isinstance(stages, list):
        logger.warning(
            "trace_store: stages must be a list, got %s; writing run row only without stages",
            type(stages).__name__,
        )
        stages = []

    try:
        # check_same_thread=False because the pipeline runs from
        # FastAPI's worker thread pool; busy_timeout handles the SQLite
        # SQLITE_BUSY case when two workers contend for the write lock.
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            con.execute("PRAGMA busy_timeout = 1000")
            with con:
                _insert_run(
                    con,
                    query_id=query_id,
                    raw_query=raw_query,
                    arm=arm,
                    route_reason=route_reason,
                    route_signals=route_signals,
                    pipeline_kind=pipeline_kind,
                    used_fallback=used_fallback,
                    total_ms=total_ms,
                    n_results=n_results,
                    error=error,
                )
                _insert_stages_and_tools(con, query_id=query_id, stages=stages)
        finally:
            con.close()
    except sqlite3.Error:
        logger.warning(
            "trace_store: write_trace failed for query_id=%s; search response not affected",
            query_id,
            exc_info=True,
        )
        return


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #


def _insert_run(
    con: sqlite3.Connection,
    *,
    query_id: str,
    raw_query: str,
    arm: str,
    route_reason: str | None,
    route_signals: dict | None,
    pipeline_kind: str | None,
    used_fallback: bool,
    total_ms: float,
    n_results: int,
    error: str | None,
) -> None:
    """Insert one row into ``agent_runs``. Timestamp is unix ms (UTC)."""
    import time as _time

    ts_ms = int(_time.time() * 1000)
    signals_json: str | None = None
    if route_signals is not None:
        try:
            signals_json = json.dumps(route_signals, default=str)
        except (TypeError, ValueError) as exc:
            # Don't drop the whole trace just because signals are weird.
            logger.warning(
                "trace_store: route_signals not JSON-serialisable (%s); storing as NULL",
                exc,
            )
            signals_json = None

    con.execute(
        """
        INSERT OR REPLACE INTO agent_runs (
            query_id, raw_query, ts, arm, route_reason, pipeline_kind,
            used_fallback, total_ms, n_results, error, route_signals
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            raw_query,
            ts_ms,
            arm,
            route_reason,
            pipeline_kind,
            int(bool(used_fallback)),
            float(total_ms) if total_ms is not None else None,
            int(n_results) if n_results is not None else None,
            error,
            signals_json,
        ),
    )


def _insert_stages_and_tools(
    con: sqlite3.Connection,
    *,
    query_id: str,
    stages: list[dict],
) -> None:
    """Fan ``stages`` into ``agent_stages`` rows + ``agent_tool_calls`` rows.

    Each well-formed stage entry yields ONE ``agent_stages`` row.
    Entries with ``step == "tool_call"`` additionally yield one
    ``agent_tool_calls`` row. Malformed entries (non-dict, missing
    required keys) are skipped silently — a single bad entry can't
    take down the whole trace.
    """
    seq = 0
    for raw in stages:
        if not isinstance(raw, dict):
            continue
        step = raw.get("step")
        duration = raw.get("duration_ms")
        if not isinstance(step, str) or not isinstance(duration, int | float):
            continue
        seq += 1
        con.execute(
            "INSERT INTO agent_stages (query_id, stage, duration_ms, seq) VALUES (?, ?, ?, ?)",
            (query_id, step, float(duration), seq),
        )

        if step != "tool_call":
            continue
        decisions = raw.get("decisions")
        if not isinstance(decisions, dict):
            continue
        tool_name = decisions.get("tool")
        if not isinstance(tool_name, str):
            continue
        # ``ok`` may be absent in current traces (react_agent's status
        # signal is fed to Prometheus, not into decisions). Default to
        # 1 ("ok") rather than dropping the row — keeps the count
        # accurate even if status detail is missing.
        ok_raw = decisions.get("ok", 1)
        ok = int(bool(ok_raw))
        iter_n = int(decisions.get("iter") or 0)
        args_summary = decisions.get("args_summary")
        args_json: str | None
        if args_summary is None:
            args_json = None
        else:
            try:
                args_json = json.dumps(args_summary, default=str)
            except (TypeError, ValueError):
                args_json = str(args_summary)
        if args_json is not None and len(args_json) > _ARGS_SUMMARY_MAX_CHARS:
            args_json = args_json[:_ARGS_SUMMARY_MAX_CHARS] + "…"

        con.execute(
            "INSERT INTO agent_tool_calls "
            "(query_id, iter_n, tool_name, ok, duration_ms, args_summary) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                query_id,
                iter_n,
                tool_name,
                ok,
                float(duration),
                args_json,
            ),
        )


__all__ = ["DEFAULT_DB_PATH", "init_db_if_missing", "write_trace"]
