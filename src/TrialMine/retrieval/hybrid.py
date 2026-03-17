"""Hybrid retrieval: merge BM25 and semantic candidate sets.

Strategy: Reciprocal Rank Fusion (RRF) by default.
The merged candidate list is then passed to the cross-encoder re-ranker.
"""

import logging
from dataclasses import dataclass

from TrialMine.retrieval.bm25 import BM25Result
from TrialMine.retrieval.semantic import SemanticResult

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A candidate after score fusion, before re-ranking."""

    nct_id: str
    rrf_score: float
    bm25_rank: int | None = None
    semantic_rank: int | None = None


def reciprocal_rank_fusion(
    bm25_results: list[BM25Result],
    semantic_results: list[SemanticResult],
    k: int = 60,
) -> list[HybridResult]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        bm25_results: Ranked BM25 candidates.
        semantic_results: Ranked semantic candidates.
        k: RRF smoothing constant (default 60 per literature).

    Returns:
        Merged list sorted by descending RRF score.
    """
    # TODO: implement RRF: score(d) = sum(1 / (k + rank(d)))
    raise NotImplementedError


class HybridRetriever:
    """Orchestrates BM25 + semantic retrieval and fuses results."""

    def __init__(self, bm25, semantic) -> None:  # type: ignore[no-untyped-def]
        """Initialise with concrete retriever instances.

        Args:
            bm25: BM25Retriever instance.
            semantic: SemanticRetriever instance.
        """
        self.bm25 = bm25
        self.semantic = semantic

    async def retrieve(self, query: str, top_k: int = 100) -> list[HybridResult]:
        """Run both retrievers in parallel and fuse results.

        Args:
            query: Patient query string.
            top_k: Number of fused candidates to return.

        Returns:
            Fused and sorted candidate list.
        """
        # TODO: asyncio.gather BM25 + semantic, call reciprocal_rank_fusion
        raise NotImplementedError
