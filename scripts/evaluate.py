"""Full ablation evaluation of the retrieval and ranking pipeline.

Evaluates all pipeline configurations on labeled queries and produces
THE ablation table with bootstrap confidence intervals.

Configurations evaluated:
  1. BM25 only
  2. Semantic only
  3. Hybrid (BM25 + Semantic)
  4. + Cross-Encoder Re-ranking
  5. + Metadata Blender (LightGBM)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --skip-blender   # skip LightGBM if not trained

Requirements:
    - Elasticsearch running with `trials` index
    - FAISS index: data/faiss_finetuned.index + .json
    - Labels: data/evaluation/labeled_queries.jsonl
    - Cross-encoder: models/cross-encoder/fine-tuned/
    - LightGBM ranker: models/ranker/v1/model.lgb
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.evaluation.metrics import mrr, ndcg_at_k
from TrialMine.models.cross_encoder import CrossEncoderReranker
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.models.ranker import (
    FEATURE_NAMES,
    RankingBlender,
    compute_features,
)
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

LABELS_FILE = Path("data/evaluation/labeled_queries.jsonl")
REPORT_FILE = Path("docs/evaluation-report.md")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "trialmind-ablation"

EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"
CE_MODEL = "models/cross-encoder/fine-tuned"
RANKER_MODEL = "models/ranker/v1/model.lgb"

N_BOOTSTRAP = 1000
EVAL_TOP_K = 10
CANDIDATE_K = 200


def load_labels(labels_file: Path) -> dict[int, dict]:
    """Load relevance labels grouped by query_id.

    Args:
        labels_file: Path to labeled_queries.jsonl.

    Returns:
        Dict mapping query_id to query text, relevance scores, and relevant IDs.
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


def build_trial_text(doc: dict) -> str:
    """Build cross-encoder input text from an ES document.

    Args:
        doc: Elasticsearch document dict.

    Returns:
        Concatenated trial text string.
    """
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("conditions"):
        parts.append(doc["conditions"])
    if doc.get("brief_summary"):
        parts.append(doc["brief_summary"])
    text = " [SEP] ".join(parts) if parts else ""
    return text[:2048]


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: List of per-query metric values.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        Tuple of (mean, ci_low, ci_high).
    """
    arr = np.array(values)
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - ci) / 2
    low = np.percentile(means, alpha * 100)
    high = np.percentile(means, (1 - alpha) * 100)
    return float(arr.mean()), float(low), float(high)


def evaluate_bm25_only(
    labels: dict[int, dict],
    es_index: ElasticsearchIndex,
    top_k: int = EVAL_TOP_K,
) -> dict:
    """Evaluate BM25-only retrieval.

    Args:
        labels: Relevance labels from load_labels().
        es_index: Elasticsearch index.
        top_k: Number of results to evaluate.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    ndcg5_list, ndcg10_list, mrr_list, latencies = [], [], [], []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]

        t0 = time.perf_counter()
        results = es_index.search(query=query, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        result_ids = [r["nct_id"] for r in results]
        ndcg5_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=5))
        ndcg10_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=10))
        mrr_list.append(mrr(result_ids, qdata["relevant_ids"]))

    return {
        "ndcg5": ndcg5_list,
        "ndcg10": ndcg10_list,
        "mrr": mrr_list,
        "latencies": latencies,
    }


def evaluate_semantic_only(
    labels: dict[int, dict],
    faiss_index: FAISSIndex,
    embedder: TrialEmbedder,
    top_k: int = EVAL_TOP_K,
) -> dict:
    """Evaluate semantic-only retrieval.

    Args:
        labels: Relevance labels from load_labels().
        faiss_index: FAISS index.
        embedder: Trial embedder.
        top_k: Number of results to evaluate.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    ndcg5_list, ndcg10_list, mrr_list, latencies = [], [], [], []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]

        t0 = time.perf_counter()
        query_emb = embedder.embed_text(query)
        results = faiss_index.search(query_embedding=query_emb, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        result_ids = [nct_id for nct_id, _ in results]
        ndcg5_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=5))
        ndcg10_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=10))
        mrr_list.append(mrr(result_ids, qdata["relevant_ids"]))

    return {
        "ndcg5": ndcg5_list,
        "ndcg10": ndcg10_list,
        "mrr": mrr_list,
        "latencies": latencies,
    }


def evaluate_hybrid(
    labels: dict[int, dict],
    hybrid: HybridRetriever,
    top_k: int = EVAL_TOP_K,
) -> dict:
    """Evaluate hybrid (BM25 + Semantic) retrieval.

    Args:
        labels: Relevance labels from load_labels().
        hybrid: HybridRetriever instance.
        top_k: Number of results to evaluate.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    ndcg5_list, ndcg10_list, mrr_list, latencies = [], [], [], []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]

        t0 = time.perf_counter()
        results = hybrid.search(query=query, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        result_ids = [r["nct_id"] for r in results]
        ndcg5_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=5))
        ndcg10_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=10))
        mrr_list.append(mrr(result_ids, qdata["relevant_ids"]))

    return {
        "ndcg5": ndcg5_list,
        "ndcg10": ndcg10_list,
        "mrr": mrr_list,
        "latencies": latencies,
    }


def evaluate_hybrid_plus_ce(
    labels: dict[int, dict],
    hybrid: HybridRetriever,
    es_index: ElasticsearchIndex,
    reranker: CrossEncoderReranker,
    rerank_top_k: int = 50,
    final_top_k: int = EVAL_TOP_K,
) -> dict:
    """Evaluate hybrid + cross-encoder blended re-ranking.

    Args:
        labels: Relevance labels from load_labels().
        hybrid: HybridRetriever instance.
        es_index: Elasticsearch index for fetching brief_summary.
        reranker: CrossEncoderReranker instance.
        rerank_top_k: Number of hybrid candidates to re-rank.
        final_top_k: Number of final results to evaluate.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    ndcg5_list, ndcg10_list, mrr_list, latencies = [], [], [], []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]

        t0 = time.perf_counter()
        candidates = hybrid.search(query=query, top_k=rerank_top_k)

        # Enrich with brief_summary for CE text matching
        for c in candidates:
            if not c.get("brief_summary"):
                doc = es_index.get_trial(c["nct_id"])
                if doc:
                    c["brief_summary"] = doc.get("brief_summary", "")
            c["trial_text"] = build_trial_text(c)

        ranked = reranker.rerank(query, candidates, top_k=final_top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        result_ids = [r["nct_id"] for r in ranked]
        ndcg5_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=5))
        ndcg10_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=10))
        mrr_list.append(mrr(result_ids, qdata["relevant_ids"]))

    return {
        "ndcg5": ndcg5_list,
        "ndcg10": ndcg10_list,
        "mrr": mrr_list,
        "latencies": latencies,
    }


def evaluate_full_pipeline(
    labels: dict[int, dict],
    hybrid: HybridRetriever,
    es_index: ElasticsearchIndex,
    reranker: CrossEncoderReranker,
    blender: RankingBlender,
    rerank_top_k: int = 50,
    final_top_k: int = EVAL_TOP_K,
) -> dict:
    """Evaluate full pipeline: hybrid + CE + LightGBM blender.

    Args:
        labels: Relevance labels from load_labels().
        hybrid: HybridRetriever instance.
        es_index: Elasticsearch index.
        reranker: CrossEncoderReranker instance.
        blender: RankingBlender instance.
        rerank_top_k: Number of candidates for CE re-ranking.
        final_top_k: Number of final results to evaluate.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    ndcg5_list, ndcg10_list, mrr_list, latencies = [], [], [], []

    for qid in sorted(labels.keys()):
        qdata = labels[qid]
        query = qdata["query"]

        t0 = time.perf_counter()
        candidates = hybrid.search(query=query, top_k=rerank_top_k)

        # Enrich candidates with full metadata + scores
        bm25_results = es_index.search(query=query, top_k=500)
        bm25_score_map = {r["nct_id"]: r["score"] for r in bm25_results}

        query_emb = hybrid.embedder.embed_text(query)
        semantic_results = hybrid.semantic.search(
            query_embedding=query_emb, top_k=500
        )
        semantic_score_map = {nct_id: s for nct_id, s in semantic_results}

        for c in candidates:
            nct_id = c["nct_id"]
            # Add individual scores
            c["bm25_score"] = bm25_score_map.get(nct_id, 0.0)
            c["semantic_score"] = semantic_score_map.get(nct_id, 0.0)

            # Fetch full doc for CE text + metadata features
            doc = es_index.get_trial(nct_id)
            if doc:
                c["brief_summary"] = doc.get("brief_summary", "")
                c["eligibility_criteria"] = doc.get("eligibility_criteria", "")
            c["trial_text"] = build_trial_text(c)

        # Cross-encoder scoring
        texts = [c["trial_text"] for c in candidates]
        raw_scores = reranker.score(query, texts)
        for c, raw in zip(candidates, raw_scores):
            c["cross_encoder_score"] = 1 / (1 + math.exp(-raw))

        # LightGBM blending
        ranked = blender.rerank(query, candidates, top_k=final_top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        result_ids = [r["nct_id"] for r in ranked]
        ndcg5_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=5))
        ndcg10_list.append(ndcg_at_k(result_ids, qdata["relevance"], k=10))
        mrr_list.append(mrr(result_ids, qdata["relevant_ids"]))

    return {
        "ndcg5": ndcg5_list,
        "ndcg10": ndcg10_list,
        "mrr": mrr_list,
        "latencies": latencies,
    }


def format_metric(values: list[float]) -> str:
    """Format metric with bootstrap CI.

    Args:
        values: Per-query metric values.

    Returns:
        Formatted string like "0.816+-0.05".
    """
    mean, low, high = bootstrap_ci(values)
    half_width = (high - low) / 2
    return f"{mean:.3f}\u00b1{half_width:.2f}"


def format_latency(latencies: list[float]) -> str:
    """Format median latency.

    Args:
        latencies: Per-query latency values in ms.

    Returns:
        Formatted string like "77ms".
    """
    median = np.median(latencies)
    return f"{median:.0f}ms"


def build_ablation_table(results: dict[str, dict]) -> str:
    """Build the ablation table as a markdown string.

    Args:
        results: Dict mapping method name to eval results.

    Returns:
        Markdown-formatted table string.
    """
    lines = []
    lines.append(
        f"| {'Method':<33} | {'NDCG@5':^13} | {'NDCG@10':^13} "
        f"| {'MRR':^13} | {'Median Latency':^14} |"
    )
    lines.append(
        f"|{'-'*35}|{'-'*15}|{'-'*15}|{'-'*15}|{'-'*16}|"
    )
    for method, res in results.items():
        n5 = format_metric(res["ndcg5"])
        n10 = format_metric(res["ndcg10"])
        m = format_metric(res["mrr"])
        lat = format_latency(res["latencies"])
        lines.append(
            f"| {method:<33} | {n5:^13} | {n10:^13} "
            f"| {m:^13} | {lat:^14} |"
        )
    return "\n".join(lines)


def save_report(table: str, results: dict[str, dict]) -> None:
    """Save evaluation report as markdown.

    Args:
        table: The ablation table string.
        results: Dict mapping method name to eval results.
    """
    report = f"""# TrialMine Ablation Evaluation Report

## Ablation Table

{table}

*Bootstrap 95% confidence intervals from {N_BOOTSTRAP} samples over {len(next(iter(results.values()))['ndcg5'])} queries.*

## Methodology

- **Labels**: 990 graded relevance labels (0-3 scale) from Claude Haiku, pooled from both embedding models
- **Queries**: 20 diverse oncology queries (clinical, patient-language, misspelled)
- **Bootstrap CI**: 95% confidence intervals from {N_BOOTSTRAP} bootstrap resamples of query-level metrics
- **Latency**: Median per-query wall-clock time including all pipeline stages
- **NDCG**: Uses graded relevance (0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant)

## Per-Method Details

### BM25 Only
Elasticsearch multi-match with field boosting (title 3x, conditions 2x).

### Semantic Only
Fine-tuned BioLinkBERT bi-encoder + FAISS IndexFlatIP (cosine similarity).

### Hybrid (BM25 + Semantic)
Reciprocal Rank Fusion (k=60) combining 200 candidates from each retriever.

### + Cross-Encoder Re-ranking
Fine-tuned BioLinkBERT cross-encoder scores top-50 hybrid candidates.
Blended scoring: 0.7 * RRF_normalized + 0.3 * CE_sigmoid.

### + Metadata Blender
LightGBM LambdaRank re-ranks using retrieval scores + metadata features
(phase, status, enrollment, condition match, title overlap, eligibility).
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", REPORT_FILE)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Full ablation evaluation of the retrieval pipeline",
    )
    parser.add_argument(
        "--skip-blender", action="store_true",
        help="Skip LightGBM blender evaluation",
    )
    parser.add_argument(
        "--skip-ce", action="store_true",
        help="Skip cross-encoder evaluation",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full ablation evaluation and print THE table."""
    args = parse_args()

    # ── Check prerequisites ───────────────────────────────────────────────
    if not LABELS_FILE.exists():
        logger.error("Labels file not found: %s", LABELS_FILE)
        sys.exit(1)
    if not Path(FAISS_INDEX).exists():
        logger.error("FAISS index not found: %s", FAISS_INDEX)
        sys.exit(1)

    # ── Load labels ───────────────────────────────────────────────────────
    labels = load_labels(LABELS_FILE)
    total_labels = sum(len(q["relevance"]) for q in labels.values())
    logger.info("Loaded %d labels across %d queries", total_labels, len(labels))

    # ── Load components ───────────────────────────────────────────────────
    logger.info("Loading retrieval components...")
    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    reranker = None
    if not args.skip_ce and Path(CE_MODEL).exists():
        reranker = CrossEncoderReranker(model_name=CE_MODEL)

    blender = None
    if not args.skip_blender and Path(RANKER_MODEL).exists():
        blender = RankingBlender(model_path=RANKER_MODEL)

    # ── Evaluate each configuration ───────────────────────────────────────
    results: dict[str, dict] = {}

    # 1. BM25 only
    print(f"\n{'='*62}")
    print(f"  Evaluating: BM25 only")
    print(f"{'='*62}")
    t0 = time.time()
    results["BM25 only"] = evaluate_bm25_only(labels, es_index)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 2. Semantic only
    print(f"\n{'='*62}")
    print(f"  Evaluating: Semantic only")
    print(f"{'='*62}")
    t0 = time.time()
    results["Semantic only"] = evaluate_semantic_only(labels, faiss_index, embedder)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 3. Hybrid
    print(f"\n{'='*62}")
    print(f"  Evaluating: Hybrid (BM25 + Semantic)")
    print(f"{'='*62}")
    t0 = time.time()
    results["Hybrid (BM25 + Semantic)"] = evaluate_hybrid(labels, hybrid)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 4. + Cross-Encoder
    if reranker:
        print(f"\n{'='*62}")
        print(f"  Evaluating: + Cross-Encoder Re-ranking")
        print(f"{'='*62}")
        t0 = time.time()
        results["+ Cross-Encoder Re-ranking"] = evaluate_hybrid_plus_ce(
            labels, hybrid, es_index, reranker
        )
        print(f"  Done in {time.time() - t0:.1f}s")

    # 5. + Metadata Blender
    if reranker and blender:
        print(f"\n{'='*62}")
        print(f"  Evaluating: + Metadata Blender")
        print(f"{'='*62}")
        t0 = time.time()
        results["+ Metadata Blender"] = evaluate_full_pipeline(
            labels, hybrid, es_index, reranker, blender
        )
        print(f"  Done in {time.time() - t0:.1f}s")

    # ── THE ABLATION TABLE ────────────────────────────────────────────────
    table = build_ablation_table(results)

    print(f"\n\n{'='*80}")
    print(f"  THE ABLATION TABLE")
    print(f"{'='*80}\n")
    print(table)
    print(f"\n{'='*80}")
    print(f"  Bootstrap 95% CI from {N_BOOTSTRAP} samples over {len(labels)} queries")
    print(f"{'='*80}")

    # ── Per-query breakdown ───────────────────────────────────────────────
    query_ids = sorted(labels.keys())
    print(f"\n  Per-query NDCG@10:")
    header_parts = [f"{'Q#':<4}"]
    methods = list(results.keys())
    for m in methods:
        short = m.replace("Hybrid (BM25 + Semantic)", "Hybrid").replace("+ Cross-Encoder Re-ranking", "+ CE").replace("+ Metadata Blender", "+ LGB")
        header_parts.append(f"{short:>8}")
    header_parts.append(f"  {'Query':<40}")
    print(f"  {'  '.join(header_parts)}")
    print(f"  {'-'*100}")

    for i, qid in enumerate(query_ids):
        parts = [f"{qid:<4}"]
        for m in methods:
            parts.append(f"{results[m]['ndcg10'][i]:>8.3f}")
        parts.append(f"  {labels[qid]['query'][:38]:<40}")
        print(f"  {'  '.join(parts)}")

    # ── Save report ───────────────────────────────────────────────────────
    save_report(table, results)

    # ── MLflow logging ────────────────────────────────────────────────────
    print(f"\nLogging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="final-ablation") as run:
        mlflow.set_tag("stage", "ablation")
        mlflow.log_param("n_queries", len(labels))
        mlflow.log_param("n_labels", total_labels)
        mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
        mlflow.log_param("methods", ",".join(methods))

        for method, res in results.items():
            prefix = (
                method.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("+", "plus")
            )
            n5_mean, _, _ = bootstrap_ci(res["ndcg5"])
            n10_mean, _, _ = bootstrap_ci(res["ndcg10"])
            m_mean, _, _ = bootstrap_ci(res["mrr"])
            mlflow.log_metric(f"{prefix}_ndcg5", round(n5_mean, 4))
            mlflow.log_metric(f"{prefix}_ndcg10", round(n10_mean, 4))
            mlflow.log_metric(f"{prefix}_mrr", round(m_mean, 4))
            mlflow.log_metric(
                f"{prefix}_latency_ms", round(float(np.median(res["latencies"])), 1)
            )

        mlflow.log_artifact(str(REPORT_FILE))
        print(f"  Logged run (ID: {run.info.run_id})")

    print(f"\nReport: {REPORT_FILE}")
    print(f"View results: make mlflow")


if __name__ == "__main__":
    main()
