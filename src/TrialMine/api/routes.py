"""FastAPI route definitions.

Endpoints:
- POST /search          — main trial search endpoint
- GET  /trials/{nct_id} — fetch full trial details
- GET  /health          — liveness probe
- GET  /metrics         — Prometheus metrics (mounted separately)
"""

import logging

from fastapi import APIRouter, HTTPException

from TrialMine.api.schemas import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    TrialDetailResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
async def search_trials(request: SearchRequest) -> SearchResponse:
    """Run the end-to-end trial search pipeline for a patient description.

    Args:
        request: SearchRequest with patient description and optional filters.

    Returns:
        SearchResponse with ranked trials and explanations.
    """
    # TODO: call agents.pipeline.search(), handle exceptions → ErrorResponse
    raise NotImplementedError


@router.get("/trials/{nct_id}", response_model=TrialDetailResponse)
async def get_trial(nct_id: str) -> TrialDetailResponse:
    """Fetch full details for a single trial by NCT ID.

    Args:
        nct_id: ClinicalTrials.gov unique identifier.

    Returns:
        TrialDetailResponse with full trial data.
    """
    # TODO: query DB, raise 404 if not found
    raise NotImplementedError
