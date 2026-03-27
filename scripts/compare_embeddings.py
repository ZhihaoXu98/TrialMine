"""Compare off-the-shelf vs fine-tuned embeddings using labeled eval data.

Runs hybrid search with both embedding models, computes NDCG@5, NDCG@10,
and MRR using relevance labels from build_eval_dataset.py, prints a
comparison table, and logs both runs to MLflow.

Requirements:
    - Elasticsearch running with `trials` index
    - data/faiss_offshelf.index (off-the-shelf BioLinkBERT)
    - data/faiss_finetuned.index (fine-tuned BioLinkBERT)
    - data/evaluation/labeled_queries.jsonl (relevance labels)

Usage:
    python scripts/compare_embeddings.py
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.evaluation.metrics import mrr, ndcg_at_k
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
MLFLOW_EXPERIMENT = "trialmind-retrieval"

MODEL_CONFIGS = {
    "off-the-shelf": {
        "model_name": "michiyasunaga/BioLinkBERT-base",
        "faiss_index": "data/faiss_offshelf.index",
        "faiss_mapping": "data/faiss_offshelf.json",
    },
    "fine-tuned": {
        "model_name": "models/embeddings/fine-tuned",
        "faiss_index": "data/faiss_finetuned.index",
        "faiss_mapping": "data/faiss_finetuned.json",
    },
}


def load_labels(labels_file: Path) -> dict[int, dict]:
    """Load relevance labels grouped by query_id.

    Args:
        labels_file: Path to labeled_queries.jsonl.

    Returns:
        Dict mapping query_id to {
            'query': str,
            'relevance': {nct_id: score},  # graded relevance
            'relevant_ids': set of nct_ids with score >= 2,
        }.
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
                continue  # skip errors
            queries[qid]["relevance"][rec["nct_id"]] = score
            if score >= 2:
                queries[qid]["relevant_ids"].add(rec["nct_id"])
    return queries


def evaluate_model(
    model_label: str,
    config: dict,
    es_index: ElasticsearchIndex,
    labels: dict[int, dict],
) -> dict[str, float]:
    """Run hybrid search with a given model and compute metrics.

    Args:
        model_label: Display name ('off-the-shelf' or 'fine-tuned').
        config: Model config dict with model_name, faiss_index, faiss_mapping.
        es_index: Shared Elasticsearch index.
        labels: Relevance labels from load_labels().

    Returns:
        Dict with avg_ndcg5, avg_ndcg10, avg_mrr, per_query results.
    """
    print(f"\n  Loading {model_label} model...")
    embedder = TrialEmbedder(model_name=config["model_name"])

    faiss_index = FAISSIndex()
    faiss_index.load(config["faiss_index"], config["faiss_mapping"])

    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    ndcg5_scores = []
    ndcg10_scores = []
    mrr_scores = []
    per_query = []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]
        relevance = qdata["relevance"]
        relevant_ids = qdata["relevant_ids"]

        # Run hybrid search
        results = hybrid.search(query=query, top_k=10)
        result_ids = [r["nct_id"] for r in results]

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
            "ndcg5": n5,
            "ndcg10": n10,
            "mrr": m,
        })

    avg_ndcg5 = sum(ndcg5_scores) / len(ndcg5_scores) if ndcg5_scores else 0.0
    avg_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores) if ndcg10_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    return {
        "avg_ndcg5": avg_ndcg5,
        "avg_ndcg10": avg_ndcg10,
        "avg_mrr": avg_mrr,
        "per_query": per_query,
    }


def main() -> None:
    """Compare both embedding models and log results."""
    # ── Check prerequisites ───────────────────────────────────────────────────
    if not LABELS_FILE.exists():
        print(f"ERROR: Labels file not found: {LABELS_FILE}")
        print("Run: python scripts/build_eval_dataset.py first")
        sys.exit(1)

    for label, cfg in MODEL_CONFIGS.items():
        if not Path(cfg["faiss_index"]).exists():
            print(f"ERROR: FAISS index not found for {label}: {cfg['faiss_index']}")
            print(f"Run: python scripts/build_index.py --model {label} --skip-bm25")
            sys.exit(1)

    # ── Load labels ───────────────────────────────────────────────────────────
    labels = load_labels(LABELS_FILE)
    total_labels = sum(len(q["relevance"]) for q in labels.values())
    total_relevant = sum(len(q["relevant_ids"]) for q in labels.values())
    print(f"Loaded {total_labels} relevance labels across {len(labels)} queries")
    print(f"  Relevant (score >= 2): {total_relevant}")

    # ── Setup shared components ───────────────────────────────────────────────
    print("\nLoading Elasticsearch index...")
    es_index = ElasticsearchIndex()

    # ── Evaluate both models ──────────────────────────────────────────────────
    results = {}
    for model_label, config in MODEL_CONFIGS.items():
        print(f"\n{'='*55}")
        print(f"  Evaluating: {model_label}")
        print(f"{'='*55}")
        t0 = time.time()
        results[model_label] = evaluate_model(model_label, config, es_index, labels)
        elapsed = time.time() - t0
        results[model_label]["eval_time_s"] = elapsed
        print(f"  Done in {elapsed:.1f}s")

    # ── Comparison table ──────────────────────────────────────────────────────
    ots = results["off-the-shelf"]
    ft = results["fine-tuned"]

    def pct_diff(new: float, old: float) -> str:
        if old == 0:
            return "  N/A"
        diff = (new - old) / old * 100
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1f}%"

    print(f"\n{'='*62}")
    print("  EMBEDDING COMPARISON RESULTS")
    print(f"{'='*62}")
    print(f"  {'Metric':<12} {'Off-the-shelf':>14} {'Fine-tuned':>12} {'Difference':>12}")
    print(f"  {'-'*50}")
    print(f"  {'NDCG@5':<12} {ots['avg_ndcg5']:>14.3f} {ft['avg_ndcg5']:>12.3f} {pct_diff(ft['avg_ndcg5'], ots['avg_ndcg5']):>12}")
    print(f"  {'NDCG@10':<12} {ots['avg_ndcg10']:>14.3f} {ft['avg_ndcg10']:>12.3f} {pct_diff(ft['avg_ndcg10'], ots['avg_ndcg10']):>12}")
    print(f"  {'MRR':<12} {ots['avg_mrr']:>14.3f} {ft['avg_mrr']:>12.3f} {pct_diff(ft['avg_mrr'], ots['avg_mrr']):>12}")
    print(f"{'='*62}")

    # ── Per-query breakdown ───────────────────────────────────────────────────
    print(f"\n  Per-query NDCG@10:")
    print(f"  {'Q#':<4} {'Off-shelf':>10} {'Fine-tuned':>11} {'Query':<45}")
    print(f"  {'-'*70}")
    for ots_q, ft_q in zip(ots["per_query"], ft["per_query"]):
        marker = " *" if ft_q["ndcg10"] > ots_q["ndcg10"] else ""
        print(f"  {ots_q['query_id']:<4} {ots_q['ndcg10']:>10.3f} {ft_q['ndcg10']:>11.3f} {ots_q['query'][:43]:<45}{marker}")

    # ── MLflow logging ────────────────────────────────────────────────────────
    print(f"\nLogging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for model_label in ("off-the-shelf", "fine-tuned"):
        r = results[model_label]
        cfg = MODEL_CONFIGS[model_label]
        with mlflow.start_run(run_name=f"eval_{model_label.replace('-', '_')}") as run:
            mlflow.set_tag("method", "hybrid")
            mlflow.set_tag("stage", "embedding_comparison")
            mlflow.set_tag("model_type", model_label)

            mlflow.log_param("model", cfg["model_name"])
            mlflow.log_param("faiss_index", cfg["faiss_index"])
            mlflow.log_param("num_queries", len(labels))
            mlflow.log_param("num_labels", total_labels)
            mlflow.log_param("labeler", "claude-haiku")

            mlflow.log_metric("ndcg_at_5", round(r["avg_ndcg5"], 4))
            mlflow.log_metric("ndcg_at_10", round(r["avg_ndcg10"], 4))
            mlflow.log_metric("mrr", round(r["avg_mrr"], 4))
            mlflow.log_metric("eval_time_s", round(r["eval_time_s"], 1))

            # Per-query metrics
            for q in r["per_query"]:
                mlflow.log_metric("ndcg10_per_query", round(q["ndcg10"], 4), step=q["query_id"])

            # Save per-query results as artifact
            artifact_path = Path(f"data/evaluation/per_query_{model_label.replace('-', '_')}.json")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with open(artifact_path, "w") as f:
                json.dump(r["per_query"], f, indent=2)
            mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")

            print(f"  Logged run '{run.info.run_name}' (ID: {run.info.run_id})")

    print("\nDone. View results: make mlflow → http://localhost:5001")


if __name__ == "__main__":
    main()
