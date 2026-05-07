"""Prometheus metrics instrumentation for TrialMine.

Exposes:
- Per-request counter labelled by method / endpoint / status
- Per-request duration histogram (seconds, with buckets tuned for the
  cross-encoder cold-start tail)
- Per-stage retrieval latency histogram (milliseconds), recorded from
  the agent_trace by ``record_agent_trace``

Wired in by :func:`TrialMine.api.app.create_app`:

* ``app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware)``
* ``app.mount("/metrics", make_asgi_app())``
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.requests import Request
from starlette.responses import Response

# --------------------------------------------------------------------------- #
# Metric definitions                                                           #
# --------------------------------------------------------------------------- #

REQUEST_COUNT = Counter(
    "trialmine_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "trialmine_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    # Buckets cover plain BM25 (~50 ms) up through the agent path's
    # CE cold-start tail (>5 s on first call).
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

STAGE_DURATION = Histogram(
    "trialmine_search_stage_duration_ms",
    "Search pipeline stage duration in milliseconds",
    ["stage"],
    # The agent stages span microseconds (normalize) through seconds
    # (retrieve / parse_query LLM call).
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)


# --------------------------------------------------------------------------- #
# Middleware                                                                   #
# --------------------------------------------------------------------------- #

# Routes whose path includes a varying ID — collapse to a stable label so we
# don't blow up cardinality (one time-series per NCT id would be terrible).
_NORMALISE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/api/v1/trial/", "/api/v1/trial/{nct_id}"),
)


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
    """Record counter + duration for every request other than ``/metrics``.

    Skipping ``/metrics`` keeps the scrape itself out of its own time-series
    (otherwise the Counter ticks every scrape interval and the dashboard is
    dominated by self-traffic).
    """
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)

    endpoint = _endpoint_label(path)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        # Record the failure as a 500 then re-raise so FastAPI's own error
        # handling still runs.
        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, status="500"
        ).inc()
        REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(
            time.perf_counter() - start
        )
        raise

    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=str(status)
    ).inc()
    REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(
        time.perf_counter() - start
    )
    return response


# --------------------------------------------------------------------------- #
# Stage-timing recorder                                                        #
# --------------------------------------------------------------------------- #


def record_agent_trace(trace: list[dict] | None) -> None:
    """Observe per-stage durations into ``STAGE_DURATION`` from the agent_trace.

    Safe on empty / malformed trace entries — never raises into the request path.
    """
    if not trace:
        return
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        duration = entry.get("duration_ms")
        if not step or not isinstance(duration, (int, float)):
            continue
        STAGE_DURATION.labels(stage=str(step)).observe(float(duration))


# --------------------------------------------------------------------------- #
# ASGI mount                                                                   #
# --------------------------------------------------------------------------- #


def metrics_app():
    """Return the prometheus_client ASGI app to mount under ``/metrics``."""
    return make_asgi_app()


__all__ = [
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "STAGE_DURATION",
    "metrics_middleware",
    "record_agent_trace",
    "metrics_app",
]
