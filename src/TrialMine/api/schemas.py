"""FastAPI request and response Pydantic models.

These are API-boundary types — keep them separate from data/models.py
so the internal representation can evolve independently of the API contract.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Incoming search request from a patient or clinician."""

    description: str = Field(..., min_length=10, description="Patient situation description")
    max_results: int = Field(10, ge=1, le=50)
    # TODO: add optional filters (location, phase, status)


class TrialSummary(BaseModel):
    """Summarised trial returned in search results."""

    nct_id: str
    title: str
    overall_status: str
    phases: list[str]
    relevance_score: float
    explanation: str  # patient-friendly explanation from the agent


class SearchResponse(BaseModel):
    """Response returned by POST /search."""

    query_id: str
    trials: list[TrialSummary]
    total_candidates_evaluated: int


class TrialDetailResponse(BaseModel):
    """Full trial detail returned by GET /trials/{nct_id}."""

    nct_id: str
    title: Optional[str]
    brief_summary: Optional[str]
    eligibility_criteria: Optional[str]
    phases: list[str]
    overall_status: Optional[str]
    enrollment_count: Optional[int]
    locations: list[dict]
    lead_sponsor: Optional[str]


class ErrorResponse(BaseModel):
    """Structured error response — never return a raw 500."""

    error_code: str
    message: str
    detail: Optional[str] = None
