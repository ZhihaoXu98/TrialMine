"""Elasticsearch BM25 retrieval interface.

Responsibilities:
- Create an index with field mappings (text + keyword + boosting)
- Bulk-index trials from the Trial model
- Execute multi-match queries with optional phase/status filters
"""

import logging
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError

from TrialMine.data.models import Trial

logger = logging.getLogger(__name__)

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "english_custom": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stemmer", "english_stop"],
                },
            },
            "filter": {
                "english_stemmer": {"type": "stemmer", "language": "english"},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
            },
        },
    },
    "mappings": {
        "properties": {
            "nct_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english_custom"},
            "brief_summary": {"type": "text", "analyzer": "english_custom"},
            "conditions": {"type": "text", "analyzer": "english_custom"},
            "interventions": {"type": "text", "analyzer": "english_custom"},
            "eligibility_criteria": {"type": "text", "analyzer": "english_custom"},
            "all_text": {"type": "text", "analyzer": "english_custom"},
            "phase": {"type": "keyword"},
            "status": {"type": "keyword"},
            "enrollment": {"type": "integer"},
        }
    },
}


class ElasticsearchIndex:
    """Manages an Elasticsearch index of clinical trials."""

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "trials",
    ) -> None:
        """Connect to Elasticsearch.

        Args:
            es_url: Elasticsearch cluster URL.
            index_name: Name of the index to create / search.
        """
        self.es = Elasticsearch(es_url, request_timeout=60)
        self.index_name = index_name
        info = self.es.info()
        logger.info(
            "Connected to Elasticsearch %s at %s",
            info["version"]["number"],
            es_url,
        )

    def create_index(self, delete_existing: bool = True) -> None:
        """Create the trials index with mappings and analysers.

        Args:
            delete_existing: If True, drop the existing index first.
        """
        if delete_existing and self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logger.info("Deleted existing index '%s'", self.index_name)

        self.es.indices.create(index=self.index_name, body=INDEX_SETTINGS)
        logger.info("Created index '%s'", self.index_name)

    def index_trials(self, trials: list[Trial]) -> int:
        """Bulk-index a list of Trial objects.

        Args:
            trials: Trial objects to index.

        Returns:
            Number of successfully indexed documents.
        """
        total = len(trials)
        indexed = 0
        batch_size = 5000

        for start in range(0, total, batch_size):
            batch = trials[start : start + batch_size]
            actions = [self._trial_to_action(t) for t in batch]

            try:
                success, _ = bulk(self.es, actions, raise_on_error=False)
                indexed += success
            except BulkIndexError as exc:
                indexed += len(batch) - len(exc.errors)
                logger.warning("Bulk index had %d errors", len(exc.errors))

            logger.info("Indexed %d / %d trials...", indexed, total)

        self.es.indices.refresh(index=self.index_name)
        logger.info("Index refreshed. Total indexed: %d", indexed)
        return indexed

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 50,
    ) -> list[dict]:
        """Run a multi-match BM25 query with optional filters.

        Field boosting: title 3×, conditions 2×, others 1×.

        Args:
            query: Free-text search query.
            filters: Optional dict with 'status' and/or 'phase' keyword filters.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with 'nct_id', 'title', 'conditions', 'phase',
            'status', 'enrollment', 'score'.
        """
        must = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",
                        "conditions^2",
                        "interventions",
                        "brief_summary",
                        "eligibility_criteria",
                        "all_text",
                    ],
                    "type": "best_fields",
                    "tie_breaker": 0.3,
                }
            }
        ]

        filter_clauses = []
        if filters:
            if "status" in filters:
                filter_clauses.append({"term": {"status": filters["status"]}})
            if "phase" in filters:
                filter_clauses.append({"term": {"phase": filters["phase"]}})

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses,
                }
            },
        }

        resp = self.es.search(index=self.index_name, body=body)

        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                {
                    "nct_id": src["nct_id"],
                    "title": src.get("title", ""),
                    "conditions": src.get("conditions", ""),
                    "phase": src.get("phase"),
                    "status": src.get("status"),
                    "enrollment": src.get("enrollment"),
                    "score": hit["_score"],
                }
            )

        return results

    def get_trial(self, nct_id: str) -> dict | None:
        """Look up a single trial by nct_id.

        Args:
            nct_id: ClinicalTrials.gov identifier.

        Returns:
            Source dict, or None if not found.
        """
        resp = self.es.search(
            index=self.index_name,
            body={"query": {"term": {"nct_id": nct_id}}, "size": 1},
        )
        hits = resp["hits"]["hits"]
        return hits[0]["_source"] if hits else None

    def _trial_to_action(self, trial: Trial) -> dict:
        """Convert a Trial to an Elasticsearch bulk action dict."""
        conditions_str = " ; ".join(trial.conditions)
        interventions_str = " ; ".join(trial.interventions)

        all_text = " ".join(
            filter(
                None,
                [
                    trial.title,
                    conditions_str,
                    trial.brief_summary,
                    trial.eligibility_criteria,
                    interventions_str,
                ],
            )
        )

        return {
            "_index": self.index_name,
            "_id": trial.nct_id,
            "_source": {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "brief_summary": trial.brief_summary,
                "conditions": conditions_str,
                "interventions": interventions_str,
                "eligibility_criteria": trial.eligibility_criteria,
                "all_text": all_text,
                "phase": trial.phase,
                "status": trial.status,
                "enrollment": trial.enrollment,
            },
        }
