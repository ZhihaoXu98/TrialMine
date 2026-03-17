"""Elasticsearch BM25 retrieval interface.

Responsibilities:
- Index clinical trials (title + summary + eligibility text)
- Execute keyword queries, returning ranked NCT IDs with scores
- Manage index lifecycle (create, delete, refresh)
"""

import logging
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

INDEX_MAPPING: dict = {
    # TODO: define field mappings (text fields with english analyser,
    #       keyword fields for status/phase/sex filtering)
}


@dataclass
class BM25Result:
    """A single BM25 retrieval result."""

    nct_id: str
    score: float


class BM25Retriever:
    """Wraps AsyncElasticsearch for trial retrieval."""

    def __init__(self, client: AsyncElasticsearch, index_name: str) -> None:
        """Initialise with an existing Elasticsearch client.

        Args:
            client: Async Elasticsearch client.
            index_name: Target index name.
        """
        self.client = client
        self.index_name = index_name

    async def index_trials(self, trials: list[dict]) -> None:
        """Bulk-index a list of trial dicts.

        Args:
            trials: List of dicts with at minimum nct_id, title, brief_summary.
        """
        # TODO: use helpers.async_bulk for efficiency
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 100) -> list[BM25Result]:
        """Run a BM25 query and return top_k results.

        Args:
            query: Natural-language or keyword query string.
            top_k: Number of results to return.

        Returns:
            List of BM25Result sorted by descending score.
        """
        # TODO: multi_match across title, brief_summary, eligibility_text
        raise NotImplementedError
