"""Prometheus metrics for TrialMine.

Re-exports the public surface from :mod:`TrialMine.monitoring.metrics`
so existing imports — ``from TrialMine.monitoring import
metrics_middleware, record_agent_trace`` — keep working after the
single-file ``monitoring.py`` was split into a package.
"""

from TrialMine.monitoring.metrics import (
    AGENT_COST_USD,
    AGENT_FAILURES,
    AGENT_ITERATIONS,
    AGENT_ROUTING,
    AGENT_TOOL_CALLS,
    MODEL_INFERENCE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SEARCH_RESULTS_COUNT,
    SEARCH_STAGE_LATENCY,
    ZERO_RESULTS,
    metrics_app,
    metrics_middleware,
    record_agent_cost,
    record_agent_failure,
    record_agent_iterations,
    record_agent_routing,
    record_agent_tool_call,
    record_agent_trace,
    record_search_results,
    time_model,
    time_model_cm,
    time_stage,
    time_stage_cm,
)

__all__ = [
    "AGENT_COST_USD",
    "AGENT_FAILURES",
    "AGENT_ITERATIONS",
    "AGENT_ROUTING",
    "AGENT_TOOL_CALLS",
    "MODEL_INFERENCE",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "SEARCH_RESULTS_COUNT",
    "SEARCH_STAGE_LATENCY",
    "ZERO_RESULTS",
    "metrics_app",
    "metrics_middleware",
    "record_agent_cost",
    "record_agent_failure",
    "record_agent_iterations",
    "record_agent_routing",
    "record_agent_tool_call",
    "record_agent_trace",
    "record_search_results",
    "time_model",
    "time_model_cm",
    "time_stage",
    "time_stage_cm",
]
