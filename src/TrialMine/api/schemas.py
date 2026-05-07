"""FastAPI request and response Pydantic models.

These are API-boundary types — keep them separate from data/models.py
so the internal representation can evolve independently of the API contract.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Incoming search request.

    By default the request is routed through the agent pipeline
    (parse → orchestrate → fallback). Set ``use_agent=False`` to bypass
    and run plain BM25 / semantic / hybrid retrieval as before.
    """

    query: str = Field(..., min_length=1)
    top_k: int = Field(20, ge=1, le=100)
    filters: dict | None = None  # e.g. {"status": "RECRUITING", "phase": "Phase 3"}
    method: Literal["bm25", "semantic", "hybrid"] = "hybrid"
    use_agent: bool = True


class TrialResult(BaseModel):
    """A single trial in search results.

    Agent-only fields (``explanation``, ``eligibility``, ``warnings``) are
    populated when the request was routed through the pipeline; remain
    ``None`` / empty for the legacy bm25 / semantic / hybrid path.
    """

    nct_id: str
    title: str
    conditions: list[str]
    phase: str | None
    status: str | None
    score: float
    url: str | None
    source: str | None = None  # "bm25_only", "semantic_only", "both" (hybrid only)
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    # Agent-only fields
    explanation: str | None = None
    eligibility: dict | None = None
    warnings: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from POST /api/v1/search.

    Agent-only fields (``agent_trace``, ``used_fallback``,
    ``patient_profile``, ``error``) are populated when ``use_agent=True``;
    remain ``None`` / ``False`` / empty otherwise.
    """

    results: list[TrialResult]
    total: int
    query: str
    search_time_ms: float
    search_method: str
    timings: dict[str, float] | None = None
    # Agent-only fields
    agent_trace: list[dict] | None = None
    used_fallback: bool = False
    patient_profile: dict | None = None
    error: str | None = None


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
