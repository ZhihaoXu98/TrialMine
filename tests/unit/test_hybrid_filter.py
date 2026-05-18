"""Unit tests for the post-RRF filter in :class:`HybridRetriever`.

Decision 28 (Phase 7) introduced a post-RRF filter to drop semantic-only
candidates whose ES doc doesn't match the caller's ``filters`` dict —
FAISS has no native filter primitive, so wrong-status / wrong-phase
trials otherwise leak through the semantic side of RRF.

The original fix landed only on :meth:`HybridRetriever.full_pipeline`;
the legacy non-agent :meth:`HybridRetriever.search` path was missed and
silently returned COMPLETED trials when callers passed
``filters={"status": "RECRUITING"}``. The backport ships in lockstep
with this test suite, which pins both methods against the same
expectation so the bug can't drift back into either path.
"""

from __future__ import annotations

from typing import Any

import pytest

from TrialMine.retrieval.hybrid import HybridRetriever

# --------------------------------------------------------------------------- #
# Stubs — minimal surface needed by HybridRetriever                            #
# --------------------------------------------------------------------------- #


class _StubBM25:
    """Stub for :class:`ElasticsearchIndex` honoring exact-match filters."""

    def __init__(self, trials: list[dict]) -> None:
        self.trials = {t["nct_id"]: t for t in trials}

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 50,
    ) -> list[dict]:
        results = list(self.trials.values())
        if filters:
            results = [
                t for t in results if all(str(t.get(k) or "") == str(v) for k, v in filters.items())
            ]
        # Real ES results carry a BM25 relevance score; full_pipeline reads
        # it into ``bm25_score_map`` as a downstream feature. Inject a
        # monotonic synthetic score so the stub matches the contract.
        return [dict(t, score=1.0 - i * 0.01) for i, t in enumerate(results[:top_k])]

    def get_trial(self, nct_id: str) -> dict | None:
        return self.trials.get(nct_id)


class _StubFAISS:
    """Stub for :class:`FAISSIndex` returning a fixed ranked list."""

    def __init__(self, ranked_nct_ids: list[str]) -> None:
        self.ranked_nct_ids = ranked_nct_ids

    def search(self, query_embedding: Any, top_k: int = 50) -> list[tuple[str, float]]:
        return [(nct, 1.0 - i * 0.01) for i, nct in enumerate(self.ranked_nct_ids[:top_k])]


class _StubEmbedder:
    """Stub for :class:`TrialEmbedder`. FAISS stub ignores the vector."""

    def embed_text(self, query: str) -> list[float]:
        return [0.0]


class _StubReranker:
    """Stub for :class:`CrossEncoderReranker` — returns a fixed score per text.

    The score doesn't matter for filter-correctness assertions; we only
    care that the filter ran before the reranker saw COMPLETED trials.
    """

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.5] * len(texts)


# --------------------------------------------------------------------------- #
# Shared fixture: 3 RECRUITING + 2 COMPLETED, FAISS prefers COMPLETED          #
# --------------------------------------------------------------------------- #


def _five_trials_three_recruiting() -> list[dict]:
    """Five trials where the COMPLETED two would dominate without the filter."""
    return [
        {
            "nct_id": "NCT01",
            "status": "RECRUITING",
            "phase": "Phase 2",
            "title": "Lung-R-1",
            "conditions": "lung cancer",
            "brief_summary": "Recruiting lung cancer trial 1",
            "eligibility_criteria": "Adults 18+",
        },
        {
            "nct_id": "NCT02",
            "status": "RECRUITING",
            "phase": "Phase 3",
            "title": "Lung-R-2",
            "conditions": "lung cancer",
            "brief_summary": "Recruiting lung cancer trial 2",
            "eligibility_criteria": "Adults 18+",
        },
        {
            "nct_id": "NCT03",
            "status": "RECRUITING",
            "phase": "Phase 1",
            "title": "Lung-R-3",
            "conditions": "lung cancer",
            "brief_summary": "Recruiting lung cancer trial 3",
            "eligibility_criteria": "Adults 18+",
        },
        {
            "nct_id": "NCT04",
            "status": "COMPLETED",
            "phase": "Phase 3",
            "title": "Lung-C-1",
            "conditions": "lung cancer",
            "brief_summary": "Completed lung cancer trial 1",
            "eligibility_criteria": "Adults 18+",
        },
        {
            "nct_id": "NCT05",
            "status": "COMPLETED",
            "phase": "Phase 2",
            "title": "Lung-C-2",
            "conditions": "lung cancer",
            "brief_summary": "Completed lung cancer trial 2",
            "eligibility_criteria": "Adults 18+",
        },
    ]


# FAISS ranks the COMPLETED trials in positions 2 and 4 — high enough that
# without the post-RRF filter they would end up in any reasonable top-K.
_FAISS_RANKING = ["NCT01", "NCT04", "NCT02", "NCT05", "NCT03"]


# --------------------------------------------------------------------------- #
# search() — the path the legacy non-agent API hits                            #
# --------------------------------------------------------------------------- #


def test_search_drops_completed_when_filter_recruiting() -> None:
    """Bug repro: FAISS returns COMPLETED in its top-K; without the
    backported filter, those leak into the search() output even when
    the caller passed ``filters={"status": "RECRUITING"}``.
    """
    trials = _five_trials_three_recruiting()
    retriever = HybridRetriever(
        bm25=_StubBM25(trials),
        semantic=_StubFAISS(_FAISS_RANKING),
        embedder=_StubEmbedder(),
    )

    results = retriever.search(
        query="lung cancer",
        filters={"status": "RECRUITING"},
        top_k=10,
    )

    statuses = [r["status"] for r in results]
    assert "COMPLETED" not in statuses, f"Expected no COMPLETED, got: {statuses}"
    assert {r["nct_id"] for r in results} == {"NCT01", "NCT02", "NCT03"}


def test_search_no_filter_returns_all_statuses() -> None:
    """Regression guard: with ``filters=None``, the fix must not change
    behavior — both RECRUITING and COMPLETED trials should still show
    up exactly as RRF ordered them.
    """
    trials = _five_trials_three_recruiting()
    retriever = HybridRetriever(
        bm25=_StubBM25(trials),
        semantic=_StubFAISS(_FAISS_RANKING),
        embedder=_StubEmbedder(),
    )

    results = retriever.search(query="lung cancer", filters=None, top_k=10)

    assert {r["status"] for r in results} == {"RECRUITING", "COMPLETED"}
    assert len(results) == 5


def test_search_multi_key_filter_requires_all_match() -> None:
    """Multi-key filter: both ``status`` AND ``phase`` must match. Only
    NCT01 (RECRUITING + Phase 2) should survive.
    """
    trials = _five_trials_three_recruiting()
    retriever = HybridRetriever(
        bm25=_StubBM25(trials),
        semantic=_StubFAISS(_FAISS_RANKING),
        embedder=_StubEmbedder(),
    )

    results = retriever.search(
        query="lung cancer",
        filters={"status": "RECRUITING", "phase": "Phase 2"},
        top_k=10,
    )

    assert [r["nct_id"] for r in results] == ["NCT01"]


# --------------------------------------------------------------------------- #
# full_pipeline() — the path the agent uses; pinned for parity                 #
# --------------------------------------------------------------------------- #


def test_full_pipeline_drops_completed_when_filter_recruiting() -> None:
    """Parity guard: :meth:`full_pipeline` and :meth:`search` must agree
    on filter semantics. If a future refactor breaks one but not the
    other, this test fails alongside its search() twin.
    """
    trials = _five_trials_three_recruiting()
    retriever = HybridRetriever(
        bm25=_StubBM25(trials),
        semantic=_StubFAISS(_FAISS_RANKING),
        embedder=_StubEmbedder(),
    )

    results, _timings = retriever.full_pipeline(
        query="lung cancer",
        reranker=_StubReranker(),
        blender=None,  # exercises the CE-blended fallback branch
        top_k=10,
        rerank_top_k=50,
        filters={"status": "RECRUITING"},
    )

    statuses = [r["status"] for r in results]
    assert "COMPLETED" not in statuses, f"Expected no COMPLETED, got: {statuses}"
    assert {r["nct_id"] for r in results} == {"NCT01", "NCT02", "NCT03"}


def test_full_pipeline_no_filter_returns_all_statuses() -> None:
    """Regression guard mirror of the search()-side test."""
    trials = _five_trials_three_recruiting()
    retriever = HybridRetriever(
        bm25=_StubBM25(trials),
        semantic=_StubFAISS(_FAISS_RANKING),
        embedder=_StubEmbedder(),
    )

    results, _timings = retriever.full_pipeline(
        query="lung cancer",
        reranker=_StubReranker(),
        blender=None,
        top_k=10,
        rerank_top_k=50,
        filters=None,
    )

    assert {r["status"] for r in results} == {"RECRUITING", "COMPLETED"}


# --------------------------------------------------------------------------- #
# Cross-method parity                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "filters",
    [
        {"status": "RECRUITING"},
        {"status": "RECRUITING", "phase": "Phase 3"},
        None,
    ],
)
def test_search_and_full_pipeline_return_same_filtered_id_set(
    filters: dict | None,
) -> None:
    """search() and full_pipeline() may diverge in scoring + ordering
    (full_pipeline reranks with CE + blender), but their POST-FILTER
    candidate sets must agree. This is the load-bearing guarantee
    Decision 28 ships: filter semantics are independent of which
    retrieval entry point you used.
    """
    trials = _five_trials_three_recruiting()

    def _build() -> HybridRetriever:
        return HybridRetriever(
            bm25=_StubBM25(trials),
            semantic=_StubFAISS(_FAISS_RANKING),
            embedder=_StubEmbedder(),
        )

    search_results = _build().search(query="lung cancer", filters=filters, top_k=10)
    pipeline_results, _ = _build().full_pipeline(
        query="lung cancer",
        reranker=_StubReranker(),
        blender=None,
        top_k=10,
        rerank_top_k=50,
        filters=filters,
    )

    assert {r["nct_id"] for r in search_results} == {r["nct_id"] for r in pipeline_results}
