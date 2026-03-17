"""Prometheus metrics instrumentation for TrialMine.

Exposes:
- Request latency histograms per endpoint
- Retrieval stage latencies (BM25, semantic, rerank)
- Cache hit/miss counters
- Active request gauge
"""

# TODO: define prometheus_client metrics (Counter, Histogram, Gauge)
# TODO: expose /metrics endpoint via prometheus_client.make_asgi_app()
# TODO: add middleware helper to record per-request latency
