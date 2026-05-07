"""Unit tests for the FastAPI surface.

Uses :class:`fastapi.testclient.TestClient` *without* the lifespan
context manager so external resources (Elasticsearch, FAISS, embedder)
are never constructed. Where a route depends on those, we install a
mock onto ``app.state`` directly.

Coverage:

1. ``GET /health`` — liveness probe.
2. ``POST /api/v1/search`` — happy path, agent pipeline mocked at the
   import boundary so no model loads.
3. Empty query → 4xx validation error (Pydantic ``min_length=1``).
4. ``GET /api/v1/trial/{nct_id}`` — 404 when ES has no record.
5. Response schema — ``SearchResponse`` round-trips back through the
   Pydantic model.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# QueryParserAgent's __init__ raises if ANTHROPIC_API_KEY isn't set, and the
# pipeline build path imports it. Set a placeholder *before* importing the
# app so module-level construction succeeds in CI.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-placeholder")

from TrialMine.api.app import create_app  # noqa: E402
from TrialMine.api.schemas import SearchResponse, TrialResult  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Bare TestClient — lifespan is *not* invoked, so no real ES / FAISS load."""
    app = create_app()
    # Default-state every resource to None; individual tests override what they need.
    app.state.es_index = None
    app.state.faiss_index = None
    app.state.embedder = None
    app.state.hybrid_retriever = None
    app.state.pipeline = None
    with TestClient(app) as c:  # noqa: SIM117 — context manager kept explicit for cleanup
        yield c


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    """``/health`` is the docker / k8s liveness probe — must always 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_via_agent_pipeline_returns_search_response(
    client: TestClient,
) -> None:
    """Mock the agent pipeline at the call site and assert the route plumbs
    the result through into a valid ``SearchResponse``."""
    # Stand in for the compiled LangGraph pipeline — routes only checks it
    # for "is not None"; the real call goes via ``pipeline_search``.
    client.app.state.pipeline = object()  # type: ignore[attr-defined]

    fake_agent_result = {
        "patient_profile": {"condition": "breast cancer", "raw_query": "breast cancer"},
        "search_results": {
            "results": [
                {
                    "nct_id": "NCT00000001",
                    "title": "A Phase 3 Trial of X for Breast Cancer",
                    "phase": "Phase 3",
                    "status": "RECRUITING",
                    "conditions": "Breast Cancer",
                    "score": 0.92,
                    "explanation": "Eligibility: likely a Match (5 met, 0 unmet, 1 unknown)",
                    "eligibility": {"verdict": "Met"},
                    "warnings": [],
                    "url": "https://clinicaltrials.gov/study/NCT00000001",
                }
            ],
            "query_used": "breast cancer",
            "filters": {"status": "RECRUITING"},
            "normalized_condition": "breast neoplasm",
        },
        "agent_trace": [{"step": "parse_query", "duration_ms": 1200.0, "decisions": {}}],
        "used_fallback": False,
        "error": None,
    }

    # Patch the pipeline-search coroutine the route awaits.
    import TrialMine.agents.pipeline as pipeline_mod

    real_search = pipeline_mod.search
    pipeline_mod.search = AsyncMock(return_value=fake_agent_result)
    try:
        resp = client.post(
            "/api/v1/search",
            json={"query": "breast cancer", "top_k": 5, "use_agent": True},
        )
    finally:
        pipeline_mod.search = real_search  # type: ignore[assignment]

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["search_method"] == "agent"
    assert body["total"] == 1
    assert body["results"][0]["nct_id"] == "NCT00000001"
    assert body["used_fallback"] is False


def test_search_empty_query_rejected_by_validation(client: TestClient) -> None:
    """An empty query must be rejected at the schema layer (Pydantic
    ``min_length=1``). FastAPI returns 422 by default for validation
    errors; either of {400, 422} is an acceptable client-error signal."""
    resp = client.post("/api/v1/search", json={"query": "", "top_k": 5})
    assert resp.status_code in (400, 422), resp.text


def test_get_trial_returns_404_when_not_in_es(client: TestClient) -> None:
    """Trial-detail route must surface ES "no such doc" as a structured 404."""
    fake_es = MagicMock()
    fake_es.get_trial = MagicMock(return_value=None)
    client.app.state.es_index = fake_es  # type: ignore[attr-defined]

    resp = client.get("/api/v1/trial/NCT99999999")
    assert resp.status_code == 404
    body = resp.json()
    assert "not found" in body["detail"].lower()


def test_search_response_round_trips_through_pydantic(client: TestClient) -> None:
    """Schema-validation: a successful search response can be re-parsed
    by the SearchResponse Pydantic model — guards against accidental
    field renames breaking external clients."""
    client.app.state.pipeline = object()  # type: ignore[attr-defined]
    fake_agent_result = {
        "patient_profile": None,
        "search_results": {
            "results": [
                {
                    "nct_id": "NCT00000002",
                    "title": "Trial X",
                    "phase": "Phase 2",
                    "status": "RECRUITING",
                    "conditions": "Lung Cancer",
                    "score": 0.5,
                }
            ],
            "query_used": "lung cancer",
            "filters": {},
            "normalized_condition": None,
        },
        "agent_trace": [],
        "used_fallback": False,
        "error": None,
    }
    import TrialMine.agents.pipeline as pipeline_mod

    real_search = pipeline_mod.search
    pipeline_mod.search = AsyncMock(return_value=fake_agent_result)
    try:
        resp = client.post(
            "/api/v1/search",
            json={"query": "lung cancer", "top_k": 5, "use_agent": True},
        )
    finally:
        pipeline_mod.search = real_search  # type: ignore[assignment]

    assert resp.status_code == 200
    parsed = SearchResponse.model_validate(resp.json())
    assert parsed.total == 1
    assert isinstance(parsed.results[0], TrialResult)
    assert parsed.results[0].nct_id == "NCT00000002"
