"""Evaluate a cross-encoder re-ranker on labeled queries.

Re-ranks hybrid retrieval candidates with a cross-encoder and measures
NDCG@5, NDCG@10, and MRR against graded relevance labels (0-3).
Compares to the hybrid-only baseline (no re-ranking).

Usage:
    # Off-the-shelf baseline
    python scripts/evaluate_cross_encoder.py

    # Fine-tuned model
    python scripts/evaluate_cross_encoder.py --model models/cross-encoder/fine-tuned

    # Custom rerank depth
    python scripts/evaluate_cross_encoder.py --rerank-top-k 100

Requirements:
    - Elasticsearch running with `trials` index
    - FAISS index: data/faiss_finetuned.index + .json
    - Labels: data/evaluation/labeled_queries.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.evaluation.metrics import mrr, ndcg_at_k
from TrialMine.models.cross_encoder import CrossEncoderReranker
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

LABELS_FILE = Path("data/evaluation/labeled_queries.jsonl")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "trialmind-cross-encoder"

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"


def load_labels(labels_file: Path) -> dict[int, dict]:
    """Load relevance labels grouped by query_id.

    Args:
        labels_file: Path to labeled_queries.jsonl.

    Returns:
        Dict mapping query_id to query text, graded relevance, and relevant IDs.
    """
    queries: dict[int, dict] = {}
    with open(labels_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            if qid not in queries:
                queries[qid] = {
                    "query": rec["query"],
                    "relevance": {},
                    "relevant_ids": set(),
                }
            score = rec["relevance"]
            if score < 0:
                continue
            queries[qid]["relevance"][rec["nct_id"]] = score
            if score >= 2:
                queries[qid]["relevant_ids"].add(rec["nct_id"])
    return queries


def build_trial_text(result: dict) -> str:
    """Build cross-encoder input text from a hybrid result dict.

    Matches the bi-encoder trial text format:
    ``title [SEP] conditions [SEP] brief_summary``

    Args:
        result: Hybrid retrieval result dict (enriched with brief_summary).

    Returns:
        Concatenated trial text string.
    """
    parts = []
    if result.get("title"):
        parts.append(result["title"])
    if result.get("conditions"):
        cond = result["conditions"]
        if isinstance(cond, list):
            cond = " ".join(cond)
        parts.append(cond)
    if result.get("brief_summary"):
        parts.append(result["brief_summary"])
    text = " [SEP] ".join(parts) if parts else ""
    # Match training truncation
    if len(text) > 2048:
        text = text[:2048]
    return text


def enrich_with_summary(
    candidates: list[dict],
    es_index: ElasticsearchIndex,
) -> None:
    """Fetch brief_summary from Elasticsearch for each candidate.

    Modifies candidates in-place.

    Args:
        candidates: List of hybrid retrieval result dicts.
        es_index: Elasticsearch index for lookups.
    """
    for c in candidates:
        if not c.get("brief_summary"):
            doc = es_index.get_trial(c["nct_id"])
            if doc:
                c["brief_summary"] = doc.get("brief_summary", "")


def evaluate(
    labels: dict[int, dict],
    hybrid: HybridRetriever,
    es_index: ElasticsearchIndex,
    reranker: CrossEncoderReranker | None,
    rerank_top_k: int = 50,
    final_top_k: int = 10,
) -> dict:
    """Evaluate hybrid search with optional cross-encoder re-ranking.

    Args:
        labels: Relevance labels from load_labels().
        hybrid: HybridRetriever instance.
        es_index: Elasticsearch index for fetching brief_summary.
        reranker: Optional CrossEncoderReranker. None = hybrid-only baseline.
        rerank_top_k: Number of hybrid candidates to re-rank.
        final_top_k: Number of final results to evaluate.

    Returns:
        Dict with aggregate metrics and per-query breakdown.
    """
    ndcg5_scores = []
    ndcg10_scores = []
    mrr_scores = []
    per_query = []
    rerank_times = []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]
        relevance = qdata["relevance"]
        relevant_ids = qdata["relevant_ids"]

        # Hybrid retrieval — get enough candidates for re-ranking
        fetch_k = rerank_top_k if reranker else final_top_k
        results = hybrid.search(query=query, top_k=fetch_k)

        if reranker:
            # Fetch brief_summary so trial text matches training format
            enrich_with_summary(results, es_index)

            # Add trial_text to each candidate for cross-encoder scoring
            for r in results:
                r["trial_text"] = build_trial_text(r)

            ranked, elapsed_ms = reranker.rerank_with_timing(
                query, results, top_k=final_top_k
            )
            rerank_times.append(elapsed_ms)
            result_ids = [r["nct_id"] for r in ranked]
        else:
            result_ids = [r["nct_id"] for r in results[:final_top_k]]

        # Compute metrics
        n5 = ndcg_at_k(result_ids, relevance, k=5)
        n10 = ndcg_at_k(result_ids, relevance, k=10)
        m = mrr(result_ids, relevant_ids)

        ndcg5_scores.append(n5)
        ndcg10_scores.append(n10)
        mrr_scores.append(m)
        per_query.append({
            "query_id": qid,
            "query": query,
            "ndcg5": round(n5, 4),
            "ndcg10": round(n10, 4),
            "mrr": round(m, 4),
        })

    avg_ndcg5 = sum(ndcg5_scores) / len(ndcg5_scores) if ndcg5_scores else 0.0
    avg_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores) if ndcg10_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_rerank_ms = (
        sum(rerank_times) / len(rerank_times) if rerank_times else 0.0
    )

    return {
        "avg_ndcg5": avg_ndcg5,
        "avg_ndcg10": avg_ndcg10,
        "avg_mrr": avg_mrr,
        "avg_rerank_ms": avg_rerank_ms,
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate cross-encoder re-ranker on labeled queries",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_CE_MODEL,
        help=f"Cross-encoder model name or path (default: {DEFAULT_CE_MODEL})",
    )
    parser.add_argument(
        "--rerank-top-k", type=int, default=50,
        help="Number of hybrid candidates to re-rank (default: 50)",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Run hybrid-only baseline (no cross-encoder)",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate cross-encoder re-ranking against hybrid-only baseline."""
    args = parse_args()

    # ── Check prerequisites ───────────────────────────────────────────────
    if not LABELS_FILE.exists():
        print(f"ERROR: Labels file not found: {LABELS_FILE}")
        print("Run: python scripts/build_eval_dataset.py first")
        sys.exit(1)

    if not Path(FAISS_INDEX).exists():
        print(f"ERROR: FAISS index not found: {FAISS_INDEX}")
        print("Run: python scripts/build_index.py --skip-bm25 --model fine-tuned")
        sys.exit(1)

    # ── Load labels ───────────────────────────────────────────────────────
    labels = load_labels(LABELS_FILE)
    total_labels = sum(len(q["relevance"]) for q in labels.values())
    total_relevant = sum(len(q["relevant_ids"]) for q in labels.values())
    print(f"Loaded {total_labels} relevance labels across {len(labels)} queries")
    print(f"  Relevant (score >= 2): {total_relevant}")

    # ── Setup retrieval pipeline ──────────────────────────────────────────
    print("\nLoading retrieval components...")
    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)

    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)

    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    # ── Evaluate hybrid-only baseline ─────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Evaluating: hybrid-only (no re-ranking)")
    print(f"{'='*62}")

    t0 = time.time()
    baseline = evaluate(labels, hybrid, es_index, reranker=None)
    baseline_time = time.time() - t0
    print(f"  Done in {baseline_time:.1f}s")

    if args.no_rerank:
        # Print baseline and exit
        print(f"\n{'='*62}")
        print("  HYBRID-ONLY BASELINE")
        print(f"{'='*62}")
        print(f"  NDCG@5:   {baseline['avg_ndcg5']:.3f}")
        print(f"  NDCG@10:  {baseline['avg_ndcg10']:.3f}")
        print(f"  MRR:      {baseline['avg_mrr']:.3f}")
        return

    # ── Load cross-encoder and evaluate ───────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Evaluating: {args.model}")
    print(f"  Re-ranking top {args.rerank_top_k} candidates")
    print(f"{'='*62}")

    reranker = CrossEncoderReranker(model_name=args.model)

    t0 = time.time()
    reranked = evaluate(
        labels, hybrid, es_index, reranker=reranker, rerank_top_k=args.rerank_top_k
    )
    reranked_time = time.time() - t0
    print(f"  Done in {reranked_time:.1f}s")

    # ── Comparison table ──────────────────────────────────────────────────
    def pct_diff(new: float, old: float) -> str:
        if old == 0:
            return "  N/A"
        diff = (new - old) / old * 100
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1f}%"

    print(f"\n{'='*62}")
    print("  CROSS-ENCODER RE-RANKING RESULTS")
    print(f"{'='*62}")
    print(f"  {'Metric':<12} {'Hybrid only':>14} {'+ Re-ranking':>14} {'Difference':>12}")
    print(f"  {'-'*52}")
    print(f"  {'NDCG@5':<12} {baseline['avg_ndcg5']:>14.3f} {reranked['avg_ndcg5']:>14.3f} {pct_diff(reranked['avg_ndcg5'], baseline['avg_ndcg5']):>12}")
    print(f"  {'NDCG@10':<12} {baseline['avg_ndcg10']:>14.3f} {reranked['avg_ndcg10']:>14.3f} {pct_diff(reranked['avg_ndcg10'], baseline['avg_ndcg10']):>12}")
    print(f"  {'MRR':<12} {baseline['avg_mrr']:>14.3f} {reranked['avg_mrr']:>14.3f} {pct_diff(reranked['avg_mrr'], baseline['avg_mrr']):>12}")
    print(f"{'='*62}")
    print(f"  Avg re-ranking latency: {reranked['avg_rerank_ms']:.0f} ms ({args.rerank_top_k} candidates)")

    # ── Per-query breakdown ───────────────────────────────────────────────
    print(f"\n  Per-query NDCG@10:")
    print(f"  {'Q#':<4} {'Hybrid':>8} {'+ CE':>8} {'Query':<45}")
    print(f"  {'-'*65}")

    wins = 0
    losses = 0
    ties = 0
    for bq, rq in zip(baseline["per_query"], reranked["per_query"]):
        if rq["ndcg10"] > bq["ndcg10"] + 0.001:
            marker = " ^"
            wins += 1
        elif rq["ndcg10"] < bq["ndcg10"] - 0.001:
            marker = " v"
            losses += 1
        else:
            marker = ""
            ties += 1
        print(f"  {bq['query_id']:<4} {bq['ndcg10']:>8.3f} {rq['ndcg10']:>8.3f} {bq['query'][:43]:<45}{marker}")

    print(f"\n  Wins: {wins}, Losses: {losses}, Ties: {ties}")

    # ── MLflow logging ────────────────────────────────────────────────────
    print(f"\nLogging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Determine tag based on model path
    if "fine-tuned" in args.model or "models/" in args.model:
        model_tag = "fine-tuned"
    else:
        model_tag = "off-the-shelf"

    # Log baseline
    with mlflow.start_run(run_name="hybrid_only_baseline") as run:
        mlflow.set_tag("method", "hybrid")
        mlflow.set_tag("stage", "cross_encoder_eval")
        mlflow.set_tag("reranker", "none")
        mlflow.log_param("num_queries", len(labels))
        mlflow.log_param("num_labels", total_labels)
        mlflow.log_metric("ndcg_at_5", round(baseline["avg_ndcg5"], 4))
        mlflow.log_metric("ndcg_at_10", round(baseline["avg_ndcg10"], 4))
        mlflow.log_metric("mrr", round(baseline["avg_mrr"], 4))
        print(f"  Logged baseline run (ID: {run.info.run_id})")

    # Log re-ranked
    with mlflow.start_run(run_name=f"reranked_{model_tag}") as run:
        mlflow.set_tag("method", "hybrid+reranking")
        mlflow.set_tag("stage", "cross_encoder_eval")
        mlflow.set_tag("reranker", model_tag)
        mlflow.log_param("cross_encoder_model", args.model)
        mlflow.log_param("rerank_top_k", args.rerank_top_k)
        mlflow.log_param("num_queries", len(labels))
        mlflow.log_param("num_labels", total_labels)
        mlflow.log_metric("ndcg_at_5", round(reranked["avg_ndcg5"], 4))
        mlflow.log_metric("ndcg_at_10", round(reranked["avg_ndcg10"], 4))
        mlflow.log_metric("mrr", round(reranked["avg_mrr"], 4))
        mlflow.log_metric("avg_rerank_ms", round(reranked["avg_rerank_ms"], 1))
        mlflow.log_metric("wins", wins)
        mlflow.log_metric("losses", losses)

        # Save per-query results
        artifact_path = Path(f"data/evaluation/per_query_reranked_{model_tag}.json")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(reranked["per_query"], f, indent=2)
        mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")

        print(f"  Logged re-ranked run (ID: {run.info.run_id})")

    print("\nDone. View results: make mlflow")


if __name__ == "__main__":
    main()
