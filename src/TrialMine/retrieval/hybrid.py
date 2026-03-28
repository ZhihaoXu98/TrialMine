"""Hybrid retrieval: merge BM25 and semantic candidate sets via RRF.

Strategy: Reciprocal Rank Fusion (RRF) combines two ranked lists by
assigning each document a score of sum(1 / (k + rank)) across lists.
The merged candidate list can then be passed to re-ranking.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.semantic import FAISSIndex

if TYPE_CHECKING:
    from TrialMine.models.cross_encoder import CrossEncoderReranker
    from TrialMine.models.ranker import RankingBlender

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    semantic_results: list[tuple[str, float]],
    k: int = 60,
) -> list[dict]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    For each document appearing in either list:
        rrf_score = sum(1 / (k + rank)) for each list containing the document.

    Args:
        bm25_results: BM25 results — list of dicts with at least 'nct_id'.
        semantic_results: Semantic results — list of (nct_id, score) tuples.
        k: RRF smoothing constant (default 60 per Cormack et al., 2009).

    Returns:
        Merged list sorted by descending RRF score. Each entry is a dict with
        nct_id, rrf_score, bm25_rank, semantic_rank, and source.
    """
    scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    semantic_ranks: dict[str, int] = {}

    # BM25 contributions
    for rank, result in enumerate(bm25_results, start=1):
        nct_id = result["nct_id"]
        scores[nct_id] = scores.get(nct_id, 0.0) + 1.0 / (k + rank)
        bm25_ranks[nct_id] = rank

    # Semantic contributions
    for rank, (nct_id, _score) in enumerate(semantic_results, start=1):
        scores[nct_id] = scores.get(nct_id, 0.0) + 1.0 / (k + rank)
        semantic_ranks[nct_id] = rank

    # Build fused list
    fused = []
    for nct_id, rrf_score in scores.items():
        bm25_rank = bm25_ranks.get(nct_id)
        semantic_rank = semantic_ranks.get(nct_id)

        if bm25_rank is not None and semantic_rank is not None:
            source = "both"
        elif bm25_rank is not None:
            source = "bm25_only"
        else:
            source = "semantic_only"

        fused.append(
            {
                "nct_id": nct_id,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "source": source,
            }
        )

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


class HybridRetriever:
    """Orchestrates BM25 + semantic retrieval and fuses results."""

    def __init__(
        self,
        bm25: ElasticsearchIndex,
        semantic: FAISSIndex,
        embedder: TrialEmbedder,
    ) -> None:
        """Initialise with concrete retriever instances.

        Args:
            bm25: ElasticsearchIndex for BM25 keyword search.
            semantic: FAISSIndex for dense vector search.
            embedder: TrialEmbedder for encoding queries.
        """
        self.bm25 = bm25
        self.semantic = semantic
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 50,
        filters: dict | None = None,
        candidate_k: int = 200,
    ) -> list[dict]:
        """Hybrid search combining BM25 and semantic retrieval via RRF.

        Steps:
            1. Get top candidate_k from BM25 (with filters if provided).
            2. Embed query, get top candidate_k from FAISS.
            3. Merge using Reciprocal Rank Fusion (k=60).
            4. Enrich top_k results with metadata from BM25 results.

        Args:
            query: Patient or clinical search query.
            top_k: Number of final results to return.
            filters: Optional dict with 'status' and/or 'phase' filters.
            candidate_k: Number of candidates from each retriever.

        Returns:
            List of dicts with nct_id, title, conditions, phase, status,
            score (rrf_score), bm25_rank, semantic_rank, source.
        """
        # Step 1: BM25 retrieval
        t0 = time.perf_counter()
        bm25_results = self.bm25.search(
            query=query, filters=filters, top_k=candidate_k
        )
        bm25_ms = (time.perf_counter() - t0) * 1000

        # Step 2: Semantic retrieval
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.semantic.search(
            query_embedding=query_embedding, top_k=candidate_k
        )
        semantic_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Hybrid candidates: %d BM25 (%.0f ms) + %d semantic (%.0f ms)",
            len(bm25_results),
            bm25_ms,
            len(semantic_results),
            semantic_ms,
        )

        # Step 3: RRF fusion
        fused = reciprocal_rank_fusion(bm25_results, semantic_results)

        # Step 4: Enrich with metadata from BM25 results
        bm25_meta = {r["nct_id"]: r for r in bm25_results}
        enriched = []
        for item in fused[:top_k]:
            meta = bm25_meta.get(item["nct_id"], {})
            enriched.append(
                {
                    "nct_id": item["nct_id"],
                    "title": meta.get("title", ""),
                    "conditions": meta.get("conditions", ""),
                    "phase": meta.get("phase"),
                    "status": meta.get("status"),
                    "enrollment": meta.get("enrollment"),
                    "score": item["rrf_score"],
                    "bm25_rank": item["bm25_rank"],
                    "semantic_rank": item["semantic_rank"],
                    "source": item["source"],
                }
            )

        # For semantic-only results lacking metadata, fetch from ES
        for item in enriched:
            if not item["title"] and item["source"] == "semantic_only":
                doc = self.bm25.get_trial(item["nct_id"])
                if doc:
                    item["title"] = doc.get("title", "")
                    item["conditions"] = doc.get("conditions", "")
                    item["phase"] = doc.get("phase")
                    item["status"] = doc.get("status")
                    item["enrollment"] = doc.get("enrollment")

        return enriched

    def full_pipeline(
        self,
        query: str,
        reranker: CrossEncoderReranker,
        blender: RankingBlender | None = None,
        top_k: int = 20,
        rerank_top_k: int = 50,
        filters: dict | None = None,
    ) -> tuple[list[dict], dict[str, float]]:
        """Run the full retrieval + re-ranking + blending pipeline.

        Steps:
            1. Hybrid retrieval (BM25 + Semantic + RRF) → top rerank_top_k
            2. Cross-encoder scores top rerank_top_k candidates
            3. LightGBM blender (if provided) re-sorts using all features
            4. Return top_k results with all scores

        Args:
            query: Patient or clinical search query.
            reranker: CrossEncoderReranker for scoring candidates.
            blender: Optional RankingBlender for final re-ranking.
            top_k: Number of final results to return.
            rerank_top_k: Number of candidates for cross-encoder.
            filters: Optional dict with 'status' and/or 'phase' filters.

        Returns:
            Tuple of (ranked results, timings dict with per-stage ms).
        """
        timings: dict[str, float] = {}

        # Step 1: BM25 retrieval
        t0 = time.perf_counter()
        bm25_results = self.bm25.search(
            query=query, filters=filters, top_k=200
        )
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000
        bm25_score_map = {r["nct_id"]: r["score"] for r in bm25_results}

        # Step 2: Semantic retrieval
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.semantic.search(
            query_embedding=query_embedding, top_k=200
        )
        timings["semantic_ms"] = (time.perf_counter() - t0) * 1000
        semantic_score_map = {nct_id: s for nct_id, s in semantic_results}

        # Step 3: RRF merge
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_results, semantic_results)
        bm25_meta = {r["nct_id"]: r for r in bm25_results}
        candidates = []
        for item in fused[:rerank_top_k]:
            meta = bm25_meta.get(item["nct_id"], {})
            c = {
                "nct_id": item["nct_id"],
                "title": meta.get("title", ""),
                "conditions": meta.get("conditions", ""),
                "phase": meta.get("phase"),
                "status": meta.get("status"),
                "enrollment": meta.get("enrollment"),
                "score": item["rrf_score"],
                "bm25_rank": item["bm25_rank"],
                "semantic_rank": item["semantic_rank"],
                "source": item["source"],
                "bm25_score": bm25_score_map.get(item["nct_id"], 0.0),
                "semantic_score": semantic_score_map.get(item["nct_id"], 0.0),
            }
            # Fetch metadata for semantic-only results
            if not c["title"] and c["source"] == "semantic_only":
                doc = self.bm25.get_trial(c["nct_id"])
                if doc:
                    c["title"] = doc.get("title", "")
                    c["conditions"] = doc.get("conditions", "")
                    c["phase"] = doc.get("phase")
                    c["status"] = doc.get("status")
                    c["enrollment"] = doc.get("enrollment")
            candidates.append(c)
        timings["merge_ms"] = (time.perf_counter() - t0) * 1000

        # Step 4: Cross-encoder scoring
        t0 = time.perf_counter()
        for c in candidates:
            if not c.get("brief_summary"):
                doc = self.bm25.get_trial(c["nct_id"])
                if doc:
                    c["brief_summary"] = doc.get("brief_summary", "")
                    c["eligibility_criteria"] = doc.get("eligibility_criteria", "")
            parts = []
            if c.get("title"):
                parts.append(c["title"])
            if c.get("conditions"):
                parts.append(c["conditions"])
            if c.get("brief_summary"):
                parts.append(c["brief_summary"])
            c["trial_text"] = (" [SEP] ".join(parts) if parts else "")[:2048]

        texts = [c["trial_text"] for c in candidates]
        raw_scores = reranker.score(query, texts)
        for c, raw in zip(candidates, raw_scores):
            c["cross_encoder_score"] = 1 / (1 + math.exp(-raw))
        timings["cross_encoder_ms"] = (time.perf_counter() - t0) * 1000

        # Step 5: LightGBM blending or CE-blended fallback
        t0 = time.perf_counter()
        if blender:
            ranked = blender.rerank(query, candidates, top_k=top_k)
        else:
            # Fallback to CE-blended scoring
            rrf_scores = [c["score"] for c in candidates]
            rrf_max = max(rrf_scores) if rrf_scores else 1.0
            rrf_min = min(rrf_scores) if rrf_scores else 0.0
            rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0
            for c in candidates:
                rrf_norm = (c["score"] - rrf_min) / rrf_range
                c["blended_score"] = 0.7 * rrf_norm + 0.3 * c["cross_encoder_score"]
            ranked = sorted(
                candidates, key=lambda x: x["blended_score"], reverse=True
            )[:top_k]
        timings["blender_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = sum(timings.values())
        return ranked, timings
