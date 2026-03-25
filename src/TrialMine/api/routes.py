"""FastAPI route definitions.

Endpoints:
- POST /api/v1/search       — BM25, semantic, or hybrid trial search
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
    """Run a search over indexed trials using the specified method.

    Args:
        request: SearchRequest with query, top_k, filters, and method.
        req: FastAPI Request (carries app state).

    Returns:
        SearchResponse with ranked results, timing, and method used.
    """
    method = request.method

    # Validate that required components are loaded
    if method in ("semantic", "hybrid"):
        if req.app.state.faiss_index is None or req.app.state.embedder is None:
            raise HTTPException(
                status_code=503,
                detail=f"Semantic search unavailable — FAISS index not loaded. Use method='bm25'.",
            )

    try:
        t0 = time.perf_counter()

        if method == "bm25":
            raw_results = _search_bm25(request, req)
        elif method == "semantic":
            raw_results = _search_semantic(request, req)
        else:
            raw_results = _search_hybrid(request, req)

        elapsed_ms = (time.perf_counter() - t0) * 1000
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed (method=%s)", method)
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        TrialResult(
            nct_id=r["nct_id"],
            title=r.get("title", ""),
            conditions=[
                c.strip()
                for c in (r.get("conditions") or "").split(";")
                if c.strip()
            ],
            phase=r.get("phase"),
            status=r.get("status"),
            score=r.get("score", 0.0),
            url=f"https://clinicaltrials.gov/study/{r['nct_id']}",
            source=r.get("source"),
            bm25_rank=r.get("bm25_rank"),
            semantic_rank=r.get("semantic_rank"),
        )
        for r in raw_results
    ]

    return SearchResponse(
        results=results,
        total=len(results),
        query=request.query,
        search_time_ms=round(elapsed_ms, 2),
        search_method=method,
    )


def _search_bm25(request: SearchRequest, req: Request) -> list[dict]:
    """Run BM25 search via Elasticsearch."""
    es_index = req.app.state.es_index
    return es_index.search(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
    )


def _search_semantic(request: SearchRequest, req: Request) -> list[dict]:
    """Run semantic search via FAISS."""
    embedder = req.app.state.embedder
    faiss_index = req.app.state.faiss_index
    es_index = req.app.state.es_index

    query_embedding = embedder.embed_text(request.query)
    raw = faiss_index.search(query_embedding=query_embedding, top_k=request.top_k)

    # Enrich with metadata from Elasticsearch
    results = []
    for nct_id, score in raw:
        doc = es_index.get_trial(nct_id) or {}
        results.append(
            {
                "nct_id": nct_id,
                "title": doc.get("title", ""),
                "conditions": doc.get("conditions", ""),
                "phase": doc.get("phase"),
                "status": doc.get("status"),
                "enrollment": doc.get("enrollment"),
                "score": score,
            }
        )

    return results


def _search_hybrid(request: SearchRequest, req: Request) -> list[dict]:
    """Run hybrid search via HybridRetriever."""
    hybrid = req.app.state.hybrid_retriever
    if hybrid is None:
        raise HTTPException(
            status_code=503,
            detail="Hybrid retriever not initialised.",
        )
    return hybrid.search(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
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
