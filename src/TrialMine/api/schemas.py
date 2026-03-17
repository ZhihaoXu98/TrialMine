"""FastAPI request and response Pydantic models.

These are API-boundary types — keep them separate from data/models.py
so the internal representation can evolve independently of the API contract.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Incoming search request."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(20, ge=1, le=100)
    filters: dict | None = None  # e.g. {"status": "RECRUITING", "phase": "Phase 3"}


class TrialResult(BaseModel):
    """A single trial in search results."""

    nct_id: str
    title: str
    conditions: list[str]
    phase: str | None
    status: str | None
    score: float
    url: str | None


class SearchResponse(BaseModel):
    """Response from POST /api/v1/search."""

    results: list[TrialResult]
    total: int
    query: str
    search_time_ms: float


class TrialDetailResponse(BaseModel):
    """Full trial detail returned by GET /api/v1/trial/{nct_id}."""

    nct_id: str
    title: str | None
    brief_summary: str | None
    conditions: str | None
    interventions: str | None
    eligibility_criteria: str | None
    phase: str | None
    status: str | None
    enrollment: int | None


class ErrorResponse(BaseModel):
    """Structured error response — never return a raw 500."""

    error_code: str
    message: str
    detail: str | None = None
