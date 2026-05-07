"""Prometheus metrics for TrialMine.

Re-exports the public surface from :mod:`TrialMine.monitoring.metrics`
so existing imports — ``from TrialMine.monitoring import
metrics_middleware, record_agent_trace`` — keep working after the
single-file ``monitoring.py`` was split into a package.
"""

from TrialMine.monitoring.metrics import (
    AGENT_FAILURES,
    MODEL_INFERENCE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SEARCH_RESULTS_COUNT,
    SEARCH_STAGE_LATENCY,
    ZERO_RESULTS,
    metrics_app,
    metrics_middleware,
    record_agent_failure,
    record_agent_trace,
    record_search_results,
    time_model,
    time_model_cm,
    time_stage,
    time_stage_cm,
)

__all__ = [
    "AGENT_FAILURES",
    "MODEL_INFERENCE",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "SEARCH_RESULTS_COUNT",
    "SEARCH_STAGE_LATENCY",
    "ZERO_RESULTS",
    "metrics_app",
    "metrics_middleware",
    "record_agent_failure",
    "record_agent_trace",
    "record_search_results",
    "time_model",
    "time_model_cm",
    "time_stage",
    "time_stage_cm",
]
