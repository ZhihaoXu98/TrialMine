"""Unit tests for :mod:`TrialMine.monitoring.trace_store`.

Phase B2 of ``docs/build_agent.md``. Tests pin the writer's contract:

* One ``agent_runs`` row + N ``agent_stages`` + M ``agent_tool_calls``
  per call, with the correct shape on the wire.
* Concurrent writes don't deadlock (FastAPI's worker pool is the
  intended caller surface, so threaded contention must not hang).
* Malformed inputs log and return without raising — the trace store
  must never break the search response.
* ``init_db_if_missing`` is idempotent across multiple calls and
  produces the canonical 3-table / 7-index schema.

All tests use a tmp_path-isolated DB and reset the module-level
``_initialized`` guard between tests so the lazy schema-init path is
exercised on every call.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

import TrialMine.monitoring.trace_store as ts


@pytest.fixture(autouse=True)
def _reset_init_flag() -> None:
    """Reset the module-level ``_initialized`` guard so each test
    exercises the first-write lazy-init path."""
    saved = ts._initialized
    ts._initialized = False
    yield
    ts._initialized = saved


# --------------------------------------------------------------------------- #
# Row-count contract                                                          #
# --------------------------------------------------------------------------- #


def test_write_trace_inserts_expected_row_counts(tmp_path: Path) -> None:
    """Happy path: one run + 3 stages (incl. 1 tool_call) + 1 tool row."""
    db = tmp_path / "trace.db"
    ts.write_trace(
        query_id="q-001",
        raw_query="failed osimertinib NSCLC",
        arm="agent",
        route_reason="complex_pattern",
        route_signals={"populated_slots": 3, "matched_pattern": "failed_X_after_Y"},
        pipeline_kind=None,
        used_fallback=False,
        total_ms=12_345.6,
        n_results=7,
        error=None,
        stages=[
            {"step": "parse_query", "duration_ms": 10.0, "decisions": {}},
            {
                "step": "tool_call",
                "duration_ms": 4_000.0,
                "decisions": {
                    "tool": "search_trials",
                    "iter": 1,
                    "ok": True,
                    "args_summary": {"query": "EGFR T790M NSCLC"},
                },
            },
            {"step": "agent_submit", "duration_ms": 20.0, "decisions": {}},
        ],
        db_path=db,
    )

    con = sqlite3.connect(db)
    try:
        runs = con.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
        stages = con.execute("SELECT COUNT(*) FROM agent_stages").fetchone()[0]
        tools = con.execute("SELECT COUNT(*) FROM agent_tool_calls").fetchone()[0]
        assert (runs, stages, tools) == (1, 3, 1)

        # Confirm the row content survives the round-trip
        row = con.execute(
            "SELECT raw_query, arm, route_reason, used_fallback, "
            "total_ms, n_results, route_signals FROM agent_runs"
        ).fetchone()
        assert row[0] == "failed osimertinib NSCLC"
        assert row[1] == "agent"
        assert row[2] == "complex_pattern"
        assert row[3] == 0
        assert row[4] == pytest.approx(12_345.6)
        assert row[5] == 7
        assert json.loads(row[6])["matched_pattern"] == "failed_X_after_Y"

        # The single tool_call should have iter=1, ok=1, search_trials
        tool_row = con.execute(
            "SELECT iter_n, tool_name, ok, duration_ms FROM agent_tool_calls"
        ).fetchone()
        assert tool_row == (1, "search_trials", 1, pytest.approx(4_000.0))
    finally:
        con.close()


def test_tool_call_ok_defaults_to_one_when_absent_in_decisions(
    tmp_path: Path,
) -> None:
    """Decisions dict without an ``ok`` key still produces a tool_call
    row with ``ok=1`` (preserves count; status fidelity is a later fix)."""
    db = tmp_path / "trace.db"
    ts.write_trace(
        query_id="q-ok",
        raw_query="x",
        arm="agent",
        route_reason=None,
        route_signals=None,
        pipeline_kind=None,
        used_fallback=False,
        total_ms=1.0,
        n_results=0,
        error=None,
        stages=[
            {
                "step": "tool_call",
                "duration_ms": 1.0,
                "decisions": {"tool": "search_trials", "iter": 1},
            }
        ],
        db_path=db,
    )
    con = sqlite3.connect(db)
    try:
        ok = con.execute("SELECT ok FROM agent_tool_calls").fetchone()[0]
        assert ok == 1
    finally:
        con.close()


def test_route_signals_none_stored_as_sql_null(tmp_path: Path) -> None:
    """A ``None`` route_signals must land as SQL NULL, not the string
    ``"null"`` — that distinction matters for the Grafana query that
    filters on signals presence."""
    db = tmp_path / "trace.db"
    ts.write_trace(
        query_id="q-null",
        raw_query="x",
        arm="rule",
        route_reason="agentic_path_disabled",
        route_signals=None,
        pipeline_kind=None,
        used_fallback=False,
        total_ms=1.0,
        n_results=0,
        error=None,
        stages=[],
        db_path=db,
    )
    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT route_signals FROM agent_runs").fetchone()
        assert row[0] is None
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Concurrency                                                                  #
# --------------------------------------------------------------------------- #


def test_concurrent_writes_do_not_deadlock(tmp_path: Path) -> None:
    """20 threads write simultaneously; every thread completes within
    the 10 s join timeout and every row lands. Pins the
    ``check_same_thread=False`` + ``busy_timeout`` behaviour."""
    db = tmp_path / "trace.db"
    n_writers = 20

    barrier = threading.Barrier(n_writers)

    def _writer(i: int) -> None:
        # Sync everyone at the barrier so writes actually contend
        # rather than serialising naturally.
        barrier.wait()
        ts.write_trace(
            query_id=f"q-{i:03d}",
            raw_query=f"thread-{i}",
            arm="rule",
            route_reason="ab_router_control",
            route_signals={"i": i},
            pipeline_kind="full_pipeline",
            used_fallback=False,
            total_ms=1.0,
            n_results=0,
            error=None,
            stages=[
                {"step": "parse_query", "duration_ms": 0.5, "decisions": {}},
            ],
            db_path=db,
        )

    threads = [threading.Thread(target=_writer, args=(i,), daemon=True) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Every thread must have finished within the timeout — no deadlock.
    for t in threads:
        assert not t.is_alive(), f"thread {t.name} deadlocked"

    con = sqlite3.connect(db)
    try:
        runs = con.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
        stages = con.execute("SELECT COUNT(*) FROM agent_stages").fetchone()[0]
        assert runs == n_writers
        assert stages == n_writers
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Malformed input + error swallowing                                          #
# --------------------------------------------------------------------------- #


def test_malformed_stages_logs_and_returns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``stages`` that isn't a list logs a warning and writes the run
    row with zero stages — must not raise."""
    db = tmp_path / "trace.db"
    with caplog.at_level("WARNING", logger="TrialMine.monitoring.trace_store"):
        ts.write_trace(
            query_id="q-bad",
            raw_query="x",
            arm="rule",
            route_reason=None,
            route_signals=None,
            pipeline_kind=None,
            used_fallback=False,
            total_ms=0.0,
            n_results=0,
            error=None,
            stages="not a list",  # type: ignore[arg-type]
            db_path=db,
        )
    assert any("stages must be a list" in rec.message for rec in caplog.records), (
        "expected a 'stages must be a list' warning"
    )

    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM agent_stages").fetchone()[0] == 0
    finally:
        con.close()


def test_malformed_individual_stage_entries_are_skipped(tmp_path: Path) -> None:
    """A non-dict entry (or one missing ``step``/``duration_ms``)
    is silently skipped — single bad entry does not abort the trace."""
    db = tmp_path / "trace.db"
    ts.write_trace(
        query_id="q-mixed",
        raw_query="x",
        arm="rule",
        route_reason=None,
        route_signals=None,
        pipeline_kind=None,
        used_fallback=False,
        total_ms=0.0,
        n_results=0,
        error=None,
        stages=[
            {"step": "parse_query", "duration_ms": 1.0, "decisions": {}},
            "junk",  # not a dict
            {"step": "noop"},  # missing duration_ms
            {"duration_ms": 1.0},  # missing step
            {"step": "tool_call", "duration_ms": 1.0, "decisions": "not-a-dict"},
            {
                "step": "tool_call",
                "duration_ms": 2.0,
                "decisions": {"tool": "search_trials", "iter": 1},
            },
        ],
        db_path=db,
    )

    con = sqlite3.connect(db)
    try:
        # 3 well-formed stages: parse_query + tool_call (dict-decisions-fail) + tool_call (good)
        # Wait — the "tool_call with non-dict decisions" entry: it has a
        # valid step + duration_ms so the stage row is inserted; only
        # the tool_call sub-insert is skipped. So expect 3 stages, 1 tool.
        stages = con.execute("SELECT COUNT(*) FROM agent_stages").fetchone()[0]
        tools = con.execute("SELECT COUNT(*) FROM agent_tool_calls").fetchone()[0]
        assert (stages, tools) == (3, 1)
    finally:
        con.close()


def test_sqlite_error_is_swallowed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SQLite error during the write logs a warning and returns —
    callers (pipeline.search) must not see the exception."""
    db = tmp_path / "trace.db"

    # Force write_trace's internal connection step to raise.
    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(ts.sqlite3, "connect", _boom)

    with caplog.at_level("WARNING", logger="TrialMine.monitoring.trace_store"):
        # Should NOT raise
        ts.write_trace(
            query_id="q-err",
            raw_query="x",
            arm="rule",
            route_reason=None,
            route_signals=None,
            pipeline_kind=None,
            used_fallback=False,
            total_ms=0.0,
            n_results=0,
            error=None,
            stages=[],
            db_path=db,
        )

    # The schema-init step OR the write step (whichever fired first)
    # should have logged a warning.
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


# --------------------------------------------------------------------------- #
# init_db_if_missing idempotency                                              #
# --------------------------------------------------------------------------- #


def test_init_db_if_missing_is_idempotent(tmp_path: Path) -> None:
    """Calling init three times in a row leaves the schema healthy and
    raises nothing. Confirms the IF NOT EXISTS guard chain works."""
    db = tmp_path / "trace.db"
    ts.init_db_if_missing(db)
    ts.init_db_if_missing(db)
    ts.init_db_if_missing(db)

    con = sqlite3.connect(db)
    try:
        names = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert names == {"agent_runs", "agent_stages", "agent_tool_calls"}

        idx_count = con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        assert idx_count == 7
    finally:
        con.close()


def test_init_creates_parent_directory(tmp_path: Path) -> None:
    """Fresh checkout case: ``data/`` may not exist yet. Init must
    create the parent dir before opening the DB."""
    db = tmp_path / "deeply" / "nested" / "trace.db"
    ts.init_db_if_missing(db)
    assert db.exists()
    assert db.parent.exists()
