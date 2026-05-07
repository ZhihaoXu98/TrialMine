"""Prometheus metrics instrumentation for TrialMine.

Exposes seven metrics, an HTTP middleware that records request count and
latency, and timing decorators / context managers for instrumenting pipeline
stages and model calls. The standard ``prometheus_client`` ASGI app is
mounted under ``/metrics``.

Wiring (see :mod:`TrialMine.api.app`):

* ``app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware)``
* ``app.mount("/metrics", metrics_app())``

Public surface (re-exported from :mod:`TrialMine.monitoring`):

* Counters / histograms
    - ``REQUEST_COUNT``        — Counter [endpoint, status]
    - ``REQUEST_LATENCY``      — Histogram [endpoint], seconds
    - ``SEARCH_STAGE_LATENCY`` — Histogram [stage], milliseconds
    - ``SEARCH_RESULTS_COUNT`` — Histogram (no labels), result-count buckets
    - ``ZERO_RESULTS``         — Counter (no labels)
    - ``AGENT_FAILURES``       — Counter [stage]
    - ``MODEL_INFERENCE``      — Histogram [model], seconds
* Helpers
    - ``metrics_middleware``   — Starlette middleware
    - ``metrics_app``          — prometheus_client ASGI app
    - ``record_agent_trace``   — fan agent_trace entries into SEARCH_STAGE_LATENCY
    - ``record_search_results`` — observe result-count + bump zero-results
    - ``record_agent_failure`` — bump AGENT_FAILURES under a stage label
    - ``time_stage`` / ``time_stage_cm`` — decorator + context manager
    - ``time_model`` / ``time_model_cm`` — decorator + context manager
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.requests import Request
from starlette.responses import Response

# --------------------------------------------------------------------------- #
# Metric definitions                                                           #
# --------------------------------------------------------------------------- #

REQUEST_COUNT = Counter(
    "trialmine_requests_total",
    "Total HTTP requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "trialmine_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0),
)

# Recorded in *milliseconds* — matches the duration_ms keys already produced
# by the agent_trace and the timings dict returned from
# HybridRetriever.full_pipeline. Stages explicitly observed by the pipeline:
#   bm25, semantic, merge, cross_encoder, blender, agent_parse, agent_search
# Other agent_trace steps (normalize, build_query, build_filters,
# check_eligibility, explain, fallback_search, ...) are also recorded under
# their own labels so the stage-breakdown panel is exhaustive.
SEARCH_STAGE_LATENCY = Histogram(
    "trialmine_search_stage_latency_ms",
    "Search pipeline stage latency in milliseconds",
    ["stage"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)

SEARCH_RESULTS_COUNT = Histogram(
    "trialmine_search_results_count",
    "Number of results returned by a search request",
    buckets=(0, 1, 5, 10, 20, 50),
)

ZERO_RESULTS = Counter(
    "trialmine_zero_results_total",
    "Search requests that returned zero results",
)

AGENT_FAILURES = Counter(
    "trialmine_agent_failures_total",
    "Agent pipeline failures (timeouts, exceptions, fallback triggers)",
    ["stage"],
)

MODEL_INFERENCE = Histogram(
    "trialmine_model_inference_seconds",
    "Model inference latency in seconds, by model name",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# --------------------------------------------------------------------------- #
# Middleware                                                                   #
# --------------------------------------------------------------------------- #

# Endpoints whose path includes a varying ID — collapse to a stable label so
# we don't blow up time-series cardinality (one series per NCT id is bad).
_NORMALISE_PREFIXES: tuple[tuple[str, str], ...] = (("/api/v1/trial/", "/api/v1/trial/{nct_id}"),)


def _endpoint_label(path: str) -> str:
    """Collapse high-cardinality path segments to a route template."""
    for prefix, template in _NORMALISE_PREFIXES:
        if path.startswith(prefix):
            return template
    return path


async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Record REQUEST_COUNT + REQUEST_LATENCY for every request other than
    the metrics scrape itself.

    Skipping ``/metrics`` keeps the scrape out of its own time-series —
    otherwise the counter ticks on every scrape interval and dashboards
    are dominated by self-traffic.
    """
    path = request.url.path
    if path == "/metrics" or path.startswith("/metrics/"):
        return await call_next(request)

    endpoint = _endpoint_label(path)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        raise

    REQUEST_COUNT.labels(endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    return response


# --------------------------------------------------------------------------- #
# Stage / model timing — decorators + context managers                         #
# --------------------------------------------------------------------------- #

F = TypeVar("F", bound=Callable[..., Any])


def _observe_stage(stage: str, elapsed_seconds: float) -> None:
    """Observe ``elapsed_seconds`` (seconds) into SEARCH_STAGE_LATENCY (ms)."""
    SEARCH_STAGE_LATENCY.labels(stage=stage).observe(elapsed_seconds * 1000.0)


def time_stage(stage: str) -> Callable[[F], F]:
    """Decorator: record wall-clock latency into SEARCH_STAGE_LATENCY[stage].

    Works on both sync and async functions. Records on success and on
    exception (so failures show up as long-tail observations rather than
    silently dropping out of the histogram).
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    _observe_stage(stage, time.perf_counter() - t0)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                _observe_stage(stage, time.perf_counter() - t0)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def time_stage_cm(stage: str) -> Iterator[None]:
    """Context-manager equivalent of :func:`time_stage` for inline blocks."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _observe_stage(stage, time.perf_counter() - t0)


def time_model(model: str) -> Callable[[F], F]:
    """Decorator: record wall-clock latency into MODEL_INFERENCE[model]."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    MODEL_INFERENCE.labels(model=model).observe(time.perf_counter() - t0)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                MODEL_INFERENCE.labels(model=model).observe(time.perf_counter() - t0)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def time_model_cm(model: str) -> Iterator[None]:
    """Context-manager equivalent of :func:`time_model` for inline blocks."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        MODEL_INFERENCE.labels(model=model).observe(time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Agent-trace + result-count helpers                                           #
# --------------------------------------------------------------------------- #

# Map agent_trace ``step`` values to canonical stage labels we expose in the
# Grafana panels. Steps not in this map are recorded under their raw name —
# extra detail is fine, the panel just queries the labels it cares about.
_TRACE_STEP_TO_STAGE: dict[str, str] = {
    "parse_query": "agent_parse",
}


def record_agent_trace(trace: list[dict] | None) -> None:
    """Fan agent_trace entries into ``SEARCH_STAGE_LATENCY``.

    For each trace entry we observe the entry's own ``duration_ms`` (mapped
    through ``_TRACE_STEP_TO_STAGE`` for canonical labels), and — if the
    entry carries ``decisions.retrieve_timings`` (the dict produced by
    :meth:`HybridRetriever.full_pipeline`) — also record each ``*_ms`` key
    under its sub-stage label (bm25, semantic, merge, cross_encoder, blender).

    Safe on empty / malformed traces — never raises into the request path.
    """
    if not trace:
        return
    for entry in trace:
        if not isinstance(entry, dict):
            continue

        step = entry.get("step")
        duration = entry.get("duration_ms")
        if isinstance(step, str) and isinstance(duration, int | float):
            label = _TRACE_STEP_TO_STAGE.get(step, step)
            SEARCH_STAGE_LATENCY.labels(stage=label).observe(float(duration))

        decisions = entry.get("decisions")
        if not isinstance(decisions, dict):
            continue
        retrieve_timings = decisions.get("retrieve_timings")
        if not isinstance(retrieve_timings, dict):
            continue
        for key, value in retrieve_timings.items():
            if not isinstance(value, int | float) or not isinstance(key, str):
                continue
            if not key.endswith("_ms"):
                continue
            sub_stage = key[:-3]
            if sub_stage in ("total", ""):
                continue
            SEARCH_STAGE_LATENCY.labels(stage=sub_stage).observe(float(value))


def record_search_results(n_results: int) -> None:
    """Observe a search result count and bump ``ZERO_RESULTS`` when empty."""
    n = max(0, int(n_results))
    SEARCH_RESULTS_COUNT.observe(n)
    if n == 0:
        ZERO_RESULTS.inc()


def record_agent_failure(stage: str) -> None:
    """Increment ``AGENT_FAILURES`` under ``stage``."""
    AGENT_FAILURES.labels(stage=stage).inc()


# --------------------------------------------------------------------------- #
# ASGI mount                                                                   #
# --------------------------------------------------------------------------- #


def metrics_app():
    """Return the prometheus_client ASGI app to mount under ``/metrics``."""
    return make_asgi_app()


__all__ = [
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "SEARCH_STAGE_LATENCY",
    "SEARCH_RESULTS_COUNT",
    "ZERO_RESULTS",
    "AGENT_FAILURES",
    "MODEL_INFERENCE",
    "metrics_middleware",
    "metrics_app",
    "record_agent_trace",
    "record_search_results",
    "record_agent_failure",
    "time_stage",
    "time_stage_cm",
    "time_model",
    "time_model_cm",
]
