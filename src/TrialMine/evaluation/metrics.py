"""Information retrieval evaluation metrics.

Provides standard IR metrics for evaluating search quality once
relevance labels are available. Each function operates on a single
query's result list and its known relevant documents.

These metrics will be used in Week 4 when we have human relevance
judgments for our test queries.
"""

import math


def precision_at_k(
    result_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Precision@K — fraction of top-k results that are relevant.

    Measures how many of the returned results are actually useful.
    High precision means fewer irrelevant results shown to the user.

    Args:
        result_ids: Ordered list of document IDs returned by the search system.
        relevant_ids: Set of document IDs known to be relevant for this query.
        k: Cutoff rank.

    Returns:
        Precision value in [0.0, 1.0].

    Example:
        >>> precision_at_k(["A", "B", "C", "D"], {"A", "C", "E"}, k=3)
        0.6667  # 2 relevant out of top 3
    """
    if k <= 0:
        return 0.0
    top_k = result_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(
    result_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall@K — fraction of all relevant documents found in top-k.

    Measures how many of the known relevant documents the system managed
    to retrieve. High recall means fewer relevant trials are missed.

    Args:
        result_ids: Ordered list of document IDs returned by the search system.
        relevant_ids: Set of document IDs known to be relevant for this query.
        k: Cutoff rank.

    Returns:
        Recall value in [0.0, 1.0]. Returns 0.0 if there are no relevant docs.

    Example:
        >>> recall_at_k(["A", "B", "C", "D"], {"A", "C", "E"}, k=3)
        0.6667  # found 2 of 3 relevant docs
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = result_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def ndcg_at_k(
    result_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """NDCG@K — Normalized Discounted Cumulative Gain at rank K.

    Measures ranking quality: rewards placing highly relevant documents
    at the top. Unlike precision, this metric is graded — a relevance
    score of 3 ("perfect match") is worth more than 1 ("partially relevant").

    Uses the standard formula:
        DCG@K  = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K

    Args:
        result_ids: Ordered list of document IDs returned by the search system.
        relevance_scores: Dict mapping document ID to graded relevance
            (e.g., 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant).
            Documents not in the dict are assumed to have relevance 0.
        k: Cutoff rank.

    Returns:
        NDCG value in [0.0, 1.0]. Returns 0.0 if no relevant docs exist.

    Example:
        >>> scores = {"A": 3, "B": 2, "C": 1, "D": 0}
        >>> ndcg_at_k(["D", "A", "B", "C"], scores, k=3)
        0.7866  # A and B are relevant but not in ideal order
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    # DCG for the actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(result_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += (2**rel - 1) / math.log2(i + 1)

    # IDCG: sort all relevance scores descending, take top k
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr(
    result_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """MRR — Mean Reciprocal Rank (for a single query).

    Returns 1/rank of the first relevant result. Measures how quickly
    the user finds something useful. MRR=1.0 means the first result
    is relevant; MRR=0.5 means the second result is the first relevant one.

    To compute MRR across multiple queries, average the per-query values.

    Args:
        result_ids: Ordered list of document IDs returned by the search system.
        relevant_ids: Set of document IDs known to be relevant for this query.

    Returns:
        Reciprocal rank in (0.0, 1.0], or 0.0 if no relevant doc is found.

    Example:
        >>> mrr(["X", "A", "B"], {"A", "C"})
        0.5  # first relevant doc ("A") is at rank 2
    """
    for i, doc_id in enumerate(result_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0
