"""Integration tests against a live Elasticsearch.

Brings up a dedicated ``trials_test`` index, bulk-indexes the 50-trial
sample fixture, and exercises BM25 search + filters via
:class:`TrialMine.retrieval.bm25.ElasticsearchIndex`.

Skipped automatically when ES is unreachable so unit suites still pass
without infrastructure (CI brings up an ES service container; locally
the docker-compose dev stack provides one on port 9200).

Index lifecycle is session-scoped: created and populated once, dropped
on teardown. Each test gets the same already-populated index — they
don't mutate it, so this is safe and saves the ~5 s bulk-index hit per
test.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest

elasticsearch = pytest.importorskip("elasticsearch")
from elasticsearch.helpers import bulk  # noqa: E402

from TrialMine.retrieval.bm25 import INDEX_SETTINGS, ElasticsearchIndex  # noqa: E402

TEST_INDEX = "trials_test"


# --------------------------------------------------------------------------- #
# Session-scoped: connect, index, teardown                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def es_index(es_url: str, sample_trials: list[dict]) -> Iterator[ElasticsearchIndex]:
    """Provide a populated :class:`ElasticsearchIndex` against ``trials_test``.

    Skips the entire integration suite if ES is unreachable.
    """
    try:
        idx = ElasticsearchIndex(es_url=es_url, index_name=TEST_INDEX)
        idx.es.cluster.health(wait_for_status="yellow", timeout="30s")
    except Exception as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"Elasticsearch unreachable at {es_url}: {exc}")

    # Fresh index every session run.
    if idx.es.indices.exists(index=TEST_INDEX):
        idx.es.indices.delete(index=TEST_INDEX)
    idx.es.indices.create(index=TEST_INDEX, body=INDEX_SETTINGS)

    actions = []
    for t in sample_trials:
        # The fixture mirrors SQLite columns — ``conditions`` arrives as a
        # ``"; "``-separated string. The ES mapping treats it as text, so we
        # can index it verbatim. We synthesise ``all_text`` so multi-field
        # queries land on the same haystack the production indexer builds.
        all_text = " ".join(
            filter(
                None,
                [
                    t.get("title"),
                    t.get("conditions"),
                    t.get("brief_summary"),
                    t.get("eligibility_criteria"),
                    t.get("interventions"),
                ],
            )
        )
        actions.append(
            {
                "_index": TEST_INDEX,
                "_id": t["nct_id"],
                "_source": {
                    "nct_id": t["nct_id"],
                    "title": t.get("title") or "",
                    "brief_summary": t.get("brief_summary") or "",
                    "conditions": t.get("conditions") or "",
                    "interventions": t.get("interventions") or "",
                    "eligibility_criteria": t.get("eligibility_criteria") or "",
                    "all_text": all_text,
                    "phase": t.get("phase"),
                    "status": t.get("status"),
                    "enrollment": t.get("enrollment"),
                },
            }
        )

    success, _errors = bulk(idx.es, actions, raise_on_error=False)
    idx.es.indices.refresh(index=TEST_INDEX)
    assert success >= 45, f"only indexed {success}/50 trials"

    try:
        yield idx
    finally:
        with contextlib.suppress(Exception):
            idx.es.indices.delete(index=TEST_INDEX)
        idx.es.close()


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_index_built_with_expected_doc_count(
    es_index: ElasticsearchIndex,
) -> None:
    """All 50 sample trials are queryable post-bulk-index."""
    resp = es_index.es.count(index=TEST_INDEX)
    assert resp["count"] == 50


def test_search_returns_relevant_results_for_breast_cancer_query(
    es_index: ElasticsearchIndex,
) -> None:
    """Plain BM25 search retrieves at least one breast-cancer trial.

    We don't pin a specific NCT id (sampled trials drift over time) —
    we assert the structural contract: results are returned, the
    top hit's text mentions breast, and ``score`` is monotone-decreasing
    so we know the BM25 sort survived the trip.
    """
    results = es_index.search(query="breast cancer", top_k=10)
    assert len(results) > 0
    top = results[0]
    text_blob = f"{top['title']} {top['conditions']}".lower()
    assert "breast" in text_blob
    # Scores arrive sorted descending — sanity check.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_status_filter_excludes_completed_trials(
    es_index: ElasticsearchIndex,
) -> None:
    """``status=RECRUITING`` filter must drop COMPLETED / TERMINATED hits.

    Cross-check: an unfiltered search for breast cancer in this fixture
    surfaces both RECRUITING and COMPLETED documents (the fixture was
    stratified to include some of each). The filter must eliminate the
    non-recruiting ones.
    """
    unfiltered = es_index.search(query="breast cancer", top_k=20)
    statuses = {r["status"] for r in unfiltered}
    # If the fixture only ever surfaces one status here the assertion below
    # is meaningless — guard against that.
    assert "RECRUITING" in statuses, "fixture sanity: needs RECRUITING hits"

    filtered = es_index.search(query="breast cancer", filters={"status": "RECRUITING"}, top_k=20)
    assert len(filtered) > 0
    assert all(r["status"] == "RECRUITING" for r in filtered)


def test_phase_filter_narrows_to_requested_phase(
    es_index: ElasticsearchIndex,
) -> None:
    """``phase`` filter is exact-match on the keyword field."""
    results = es_index.search(query="cancer", filters={"phase": "Phase 3"}, top_k=20)
    assert len(results) > 0
    assert all(r["phase"] == "Phase 3" for r in results)


def test_get_trial_returns_indexed_document_by_nct_id(
    es_index: ElasticsearchIndex, sample_trials: list[dict]
) -> None:
    """Direct NCT lookup returns the source we indexed."""
    expected_id = sample_trials[0]["nct_id"]
    doc = es_index.get_trial(expected_id)
    assert doc is not None
    assert doc["nct_id"] == expected_id

    missing = es_index.get_trial("NCT00000000")  # not in fixture
    assert missing is None
