"""FastAPI route definitions.

Endpoints:
- POST /api/v1/search       — BM25 trial search
- GET  /api/v1/trial/{nct_id} — single trial details
- GET  /health               — liveness probe
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from TrialMine.api.schemas import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    TrialDetailResponse,
    TrialResult,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/api/v1/search", response_model=SearchResponse)
async def search_trials(request: SearchRequest, req: Request) -> SearchResponse:
    """Run a BM25 search over indexed trials.

    Args:
        request: SearchRequest with query, top_k, and optional filters.
        req: FastAPI Request (carries app state).

    Returns:
        SearchResponse with ranked results and timing.
    """
    es_index = req.app.state.es_index

    try:
        t0 = time.perf_counter()
        raw_results = es_index.search(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        TrialResult(
            nct_id=r["nct_id"],
            title=r["title"],
            conditions=[c.strip() for c in (r.get("conditions") or "").split(";") if c.strip()],
            phase=r.get("phase"),
            status=r.get("status"),
            score=r["score"],
            url=f"https://clinicaltrials.gov/study/{r['nct_id']}",
        )
        for r in raw_results
    ]

    return SearchResponse(
        results=results,
        total=len(results),
        query=request.query,
        search_time_ms=round(elapsed_ms, 2),
    )


@router.get(
    "/api/v1/trial/{nct_id}",
    response_model=TrialDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_trial(nct_id: str, req: Request) -> TrialDetailResponse:
    """Fetch full details for a single trial by NCT ID.

    Args:
        nct_id: ClinicalTrials.gov unique identifier.
        req: FastAPI Request (carries app state).

    Returns:
        TrialDetailResponse with full trial data.
    """
    es_index = req.app.state.es_index
    doc = es_index.get_trial(nct_id.upper())

    if not doc:
        raise HTTPException(status_code=404, detail=f"Trial {nct_id} not found")

    return TrialDetailResponse(
        nct_id=doc["nct_id"],
        title=doc.get("title"),
        brief_summary=doc.get("brief_summary"),
        conditions=doc.get("conditions"),
        interventions=doc.get("interventions"),
        eligibility_criteria=doc.get("eligibility_criteria"),
        phase=doc.get("phase"),
        status=doc.get("status"),
        enrollment=doc.get("enrollment"),
    )
