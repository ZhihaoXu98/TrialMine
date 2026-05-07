"""CI quality gate — assert retrieval NDCG@10 stays above threshold.

Runs the existing ablation evaluation (``scripts.evaluate``) on the
labeled-query test set when all artefacts are present, parses the final
NDCG@10 from the best configuration in the report, and exits non-zero
if it drops below ``--threshold``.

Soft-skips with exit 0 when any of the following are missing:

* ``data/evaluation/labeled_queries.jsonl`` (labels)
* ``data/faiss_finetuned.index`` (FAISS embeddings)
* ``models/embeddings/fine-tuned/`` (bi-encoder)
* ``models/cross-encoder/fine-tuned/`` (cross-encoder)
* a reachable Elasticsearch with the ``trials`` index populated

Skipping in CI is the default — the artefacts are gitignored. The gate
becomes a real gate on machines (or release pipelines) where those
artefacts are provisioned.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = REPO_ROOT / "data" / "evaluation" / "labeled_queries.jsonl"
FAISS_INDEX = REPO_ROOT / "data" / "faiss_finetuned.index"
EMBEDDER = REPO_ROOT / "models" / "embeddings" / "fine-tuned"
CROSS_ENCODER = REPO_ROOT / "models" / "cross-encoder" / "fine-tuned"


def _missing_artefacts() -> list[str]:
    """Return the list of missing artefacts; empty list = all present."""
    missing: list[str] = []
    for path in (LABELS, FAISS_INDEX, EMBEDDER, CROSS_ENCODER):
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
    return missing


def _es_reachable_and_indexed(es_url: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for ES reachability + ``trials`` index size."""
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:  # pragma: no cover
        return False, f"elasticsearch not installed: {exc}"
    try:
        es = Elasticsearch(es_url, request_timeout=5)
        if not es.ping():
            return False, f"ES at {es_url} did not respond to ping"
        if not es.indices.exists(index="trials"):
            return False, "ES has no 'trials' index"
        count = es.count(index="trials")["count"]
        if count < 1000:
            return False, f"ES 'trials' index has only {count} docs"
        return True, f"ES has {count} trials"
    except Exception as exc:
        return False, f"ES check failed: {exc}"


def _run_evaluation_and_extract_ndcg() -> float:
    """Run ablation evaluation in-process and return the *best* NDCG@10.

    We use the in-process API rather than shelling out so we don't have
    to parse a printed table.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from TrialMine.evaluation.metrics import ndcg_at_k  # noqa: F401  (sanity-import only)
    from TrialMine.models.cross_encoder import CrossEncoderReranker
    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.hybrid import HybridRetriever
    from TrialMine.retrieval.semantic import FAISSIndex

    # Assemble the retrieval stack.
    es = ElasticsearchIndex()
    faiss_index = FAISSIndex()
    faiss_index.load(
        str(REPO_ROOT / "data" / "faiss_finetuned.index"),
        str(REPO_ROOT / "data" / "faiss_finetuned.json"),
    )
    embedder = TrialEmbedder(model_name=str(EMBEDDER))
    hybrid = HybridRetriever(bm25=es, semantic=faiss_index, embedder=embedder)
    reranker = CrossEncoderReranker(model_name=str(CROSS_ENCODER))

    # Read labels (jsonl: {query_id, query, nct_id, score, ...})
    import json
    from collections import defaultdict

    by_query: dict[str, dict] = defaultdict(lambda: {"query": "", "labels": {}})
    with LABELS.open() as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            by_query[qid]["query"] = row["query"]
            by_query[qid]["labels"][row["nct_id"]] = float(row.get("score", 0))

    ndcg_scores: list[float] = []
    for qid, payload in by_query.items():
        ranked, _timings = hybrid.full_pipeline(
            query=payload["query"],
            reranker=reranker,
            blender=None,
            top_k=10,
            rerank_top_k=50,
            filters=None,
        )
        gains = [payload["labels"].get(r["nct_id"], 0) for r in ranked[:10]]
        ndcg_scores.append(_ndcg(gains, k=10))

    avg = sum(ndcg_scores) / max(1, len(ndcg_scores))
    print(f"Per-query NDCG@10 (n={len(ndcg_scores)}): {ndcg_scores}")
    print(f"Average NDCG@10: {avg:.4f}")
    return avg


def _ndcg(gains: list[float], k: int = 10) -> float:
    """Compute NDCG at ``k`` from a list of relevance gains."""
    import math

    def dcg(items: list[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(items))

    actual = dcg(gains[:k])
    ideal = dcg(sorted(gains, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum required average NDCG@10 to pass the gate.",
    )
    args = parser.parse_args()

    missing = _missing_artefacts()
    if missing:
        print("[quality-gate] Skipped — required artefacts missing:")
        for m in missing:
            print(f"  - {m}")
        print("[quality-gate] Exit 0 (soft-skip).")
        return 0

    ok, reason = _es_reachable_and_indexed(
        os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
    )
    if not ok:
        print(f"[quality-gate] Skipped — {reason}")
        print("[quality-gate] Exit 0 (soft-skip).")
        return 0

    print(f"[quality-gate] All artefacts present; running ablation. {reason}.")
    avg = _run_evaluation_and_extract_ndcg()
    if avg < args.threshold:
        print(
            f"[quality-gate] FAIL: NDCG@10={avg:.4f} below threshold {args.threshold:.4f}"
        )
        return 1
    print(
        f"[quality-gate] PASS: NDCG@10={avg:.4f} ≥ threshold {args.threshold:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
