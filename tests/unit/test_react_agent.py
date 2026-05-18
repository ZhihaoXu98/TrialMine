"""Unit tests for :class:`AgenticSearchAgent` internals.

The full agent loop end-to-end is exercised by integration tests + the
85-query held-out evaluation. This module pins the smaller pieces that
are easier (and cheaper) to test without LangGraph + the live LLM.

Currently covers:

* ``_AgentMetricsCallbackHandler`` — the LangChain callback that fires
  ``record_agent_tool_call`` synchronously per tool. PR-3 added this so
  cancelled agent runs still record their partial tool-call counts;
  before, end-of-run reconciliation lost those counts on every outer
  timeout.
"""

from __future__ import annotations

from unittest.mock import patch

from TrialMine.agents.react_agent import _AgentMetricsCallbackHandler


def test_callback_records_ok_on_normal_tool_output() -> None:
    """on_tool_end with a non-error JSON payload increments the counter
    with status='ok' for the matched tool name.
    """
    handler = _AgentMetricsCallbackHandler()

    with patch("TrialMine.agents.react_agent.record_agent_tool_call") as mock_record:
        handler.on_tool_start(
            serialized={"name": "search_trials"},
            input_str='{"query":"lung"}',
            run_id="run-1",
        )
        handler.on_tool_end(
            output='{"results":[{"nct_id":"NCT001"}]}',
            run_id="run-1",
        )

    mock_record.assert_called_once_with("search_trials", ok=True)


def test_callback_records_error_on_error_envelope() -> None:
    """Tools surface internal failures via ``{"error": "..."}`` JSON
    envelopes (not by raising). The handler must classify those as
    status='error' so the Prometheus label matches the trace-store row.
    """
    handler = _AgentMetricsCallbackHandler()

    with patch("TrialMine.agents.react_agent.record_agent_tool_call") as mock_record:
        handler.on_tool_start(
            serialized={"name": "get_trial_details"},
            input_str='{"nct_id":"NCT000"}',
            run_id="run-2",
        )
        handler.on_tool_end(
            output='{"error":"trial not found"}',
            run_id="run-2",
        )

    mock_record.assert_called_once_with("get_trial_details", ok=False)


def test_callback_records_error_on_exception() -> None:
    """When a tool raises (rather than returning an error envelope),
    LangChain dispatches ``on_tool_error`` instead of ``on_tool_end``.
    The handler must still mark the call as failed.
    """
    handler = _AgentMetricsCallbackHandler()

    with patch("TrialMine.agents.react_agent.record_agent_tool_call") as mock_record:
        handler.on_tool_start(
            serialized={"name": "lookup_medical_concept"},
            input_str='{"term":"x"}',
            run_id="run-3",
        )
        handler.on_tool_error(
            error=RuntimeError("network timeout"),
            run_id="run-3",
        )

    mock_record.assert_called_once_with("lookup_medical_concept", ok=False)


def test_callback_isolates_concurrent_tools_by_run_id() -> None:
    """Two tools running concurrently must not cross-contaminate. The
    handler tracks them by ``run_id`` so each on_tool_end resolves to
    the right name. This is the load-bearing invariant: LangGraph fires
    tool callbacks from the runtime's executor, so on_tool_start and
    on_tool_end for different tools can interleave.
    """
    handler = _AgentMetricsCallbackHandler()

    with patch("TrialMine.agents.react_agent.record_agent_tool_call") as mock_record:
        handler.on_tool_start(serialized={"name": "search_trials"}, input_str="{}", run_id="A")
        handler.on_tool_start(serialized={"name": "get_trial_details"}, input_str="{}", run_id="B")
        # Finish B first, then A — interleaved completion order.
        handler.on_tool_end(output='{"ok":1}', run_id="B")
        handler.on_tool_end(output='{"ok":1}', run_id="A")

    calls = mock_record.call_args_list
    assert len(calls) == 2
    # Order of completion: B then A
    assert calls[0].args == ("get_trial_details",)
    assert calls[0].kwargs == {"ok": True}
    assert calls[1].args == ("search_trials",)
    assert calls[1].kwargs == {"ok": True}


def test_callback_handles_missing_run_id_gracefully() -> None:
    """Some LangChain dispatch paths may omit ``run_id`` on legacy
    callbacks. The handler must not raise on those; degrade to an
    'unknown' name rather than blowing up the loop.
    """
    handler = _AgentMetricsCallbackHandler()

    with patch("TrialMine.agents.react_agent.record_agent_tool_call") as mock_record:
        handler.on_tool_end(output='{"results":[]}')  # no run_id, no prior start

    mock_record.assert_called_once_with("unknown", ok=True)
