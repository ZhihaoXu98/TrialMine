"""Train LightGBM LambdaRank ranker on labeled query-trial pairs.

Computes retrieval + metadata features for each labeled (query, trial) pair,
trains a LambdaRank model with leave-one-query-out cross-validation,
and saves the model + feature importance chart.

Usage:
    python scripts/train_ranker.py
    python scripts/train_ranker.py --features-only   # just build features CSV
    python scripts/train_ranker.py --no-ce            # skip cross-encoder features
    python scripts/train_ranker.py --load-features    # reuse cached features

Requirements:
    - Elasticsearch running with `trials` index
    - FAISS index: data/faiss_finetuned.index + .json
    - Labels: data/evaluation/labeled_queries.jsonl
    - Cross-encoder model: models/cross-encoder/fine-tuned/
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.models.cross_encoder import CrossEncoderReranker
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.models.ranker import (
    FEATURE_NAMES,
    compute_features,
)
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

LABELS_FILES = [
    Path("data/evaluation/labeled_queries.jsonl"),      # 20 queries (IDs 0-19)
    Path("data/evaluation/test_labels.jsonl"),           # 50 queries (IDs 100-149)
    Path("data/evaluation/train_labels_extra.jsonl"),    # 80 queries (IDs 200-279)
]
FEATURES_FILE = Path("data/evaluation/ranking_features_v2.csv")
MODEL_DIR = Path("models/ranker/v2")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "trialmind-ranker"

EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"
CE_MODEL = "models/cross-encoder/fine-tuned"


def load_labels(labels_files: list[Path]) -> list[dict]:
    """Load relevance labels from multiple JSONL files.

    Args:
        labels_files: Paths to label JSONL files.

    Returns:
        List of dicts with query_id, query, nct_id, relevance.
    """
    records = []
    for labels_file in labels_files:
        if not labels_file.exists():
            logger.warning("Labels file not found, skipping: %s", labels_file)
            continue
        with open(labels_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("relevance", -1) < 0:
                    continue
                records.append({
                    "query_id": rec["query_id"],
                    "query": rec["query"],
                    "nct_id": rec["nct_id"],
                    "relevance": rec["relevance"],
                })
        logger.info("Loaded %d labels from %s", len(records), labels_file.name)
    return records


def build_score_lookups(
    query: str,
    es_index: ElasticsearchIndex,
    faiss_index: FAISSIndex,
    embedder: TrialEmbedder,
    top_k: int = 500,
) -> tuple[dict[str, float], dict[str, float]]:
    """Get BM25 and semantic score lookups for a query.

    Args:
        query: Search query string.
        es_index: Elasticsearch index.
        faiss_index: FAISS index.
        embedder: Trial embedder for query encoding.
        top_k: Number of candidates to retrieve from each source.

    Returns:
        Tuple of (bm25_scores, semantic_scores) dicts mapping nct_id -> score.
    """
    bm25_results = es_index.search(query=query, top_k=top_k)
    bm25_scores = {r["nct_id"]: r["score"] for r in bm25_results}

    query_emb = embedder.embed_text(query)
    semantic_results = faiss_index.search(query_embedding=query_emb, top_k=top_k)
    semantic_scores = {nct_id: score for nct_id, score in semantic_results}

    return bm25_scores, semantic_scores


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


def build_features(
    labels: list[dict],
    es_index: ElasticsearchIndex,
    faiss_index: FAISSIndex,
    embedder: TrialEmbedder,
    reranker: CrossEncoderReranker | None,
) -> pd.DataFrame:
    """Compute ranking features for all labeled pairs.

    Args:
        labels: Labeled query-trial pairs from load_labels().
        es_index: Elasticsearch index.
        faiss_index: FAISS index.
        embedder: Trial embedder.
        reranker: Optional cross-encoder for CE scores.

    Returns:
        DataFrame with features + query_id + nct_id + relevance.
    """
    # Group labels by query
    queries: dict[int, dict] = {}
    for rec in labels:
        qid = rec["query_id"]
        if qid not in queries:
            queries[qid] = {"query": rec["query"], "items": []}
        queries[qid]["items"].append(rec)

    rows = []
    for qid in sorted(queries.keys()):
        qdata = queries[qid]
        query = qdata["query"]
        items = qdata["items"]

        logger.info(
            "Building features for query %d: %s (%d trials)",
            qid, query[:50], len(items),
        )

        # Get BM25 and semantic scores
        bm25_scores, semantic_scores = build_score_lookups(
            query, es_index, faiss_index, embedder
        )

        # Compute ranks for RRF
        bm25_ranks = {}
        for rank, (nct_id, _) in enumerate(
            sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            bm25_ranks[nct_id] = rank

        semantic_ranks = {}
        for rank, (nct_id, _) in enumerate(
            sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            semantic_ranks[nct_id] = rank

        # Cross-encoder scoring (batch per query)
        ce_scores_map: dict[str, float] = {}
        if reranker:
            nct_ids = [item["nct_id"] for item in items]
            trial_texts = []
            for nct_id in nct_ids:
                doc = es_index.get_trial(nct_id)
                trial_texts.append(build_trial_text(doc) if doc else "")
            raw_scores = reranker.score(query, trial_texts)
            for nct_id, raw in zip(nct_ids, raw_scores):
                ce_scores_map[nct_id] = 1 / (1 + math.exp(-raw))

        # Build feature rows
        for item in items:
            nct_id = item["nct_id"]
            doc = es_index.get_trial(nct_id)

            # Compute RRF score
            rrf = 0.0
            k = 60
            if nct_id in bm25_ranks:
                rrf += 1.0 / (k + bm25_ranks[nct_id])
            if nct_id in semantic_ranks:
                rrf += 1.0 / (k + semantic_ranks[nct_id])

            candidate = {
                "bm25_score": bm25_scores.get(nct_id, 0.0),
                "semantic_score": semantic_scores.get(nct_id, 0.0),
                "cross_encoder_score": ce_scores_map.get(nct_id, 0.5),
                "score": rrf,
            }
            if doc:
                candidate.update({
                    "phase": doc.get("phase"),
                    "status": doc.get("status"),
                    "enrollment": doc.get("enrollment"),
                    "conditions": doc.get("conditions", ""),
                    "title": doc.get("title", ""),
                    "eligibility_criteria": doc.get("eligibility_criteria", ""),
                })

            feats = compute_features(query, candidate, doc)
            feats["query_id"] = qid
            feats["nct_id"] = nct_id
            feats["relevance"] = item["relevance"]
            rows.append(feats)

    return pd.DataFrame(rows)


def train_lgbm(
    df: pd.DataFrame,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM LambdaRank with leave-one-query-out CV.

    Args:
        df: Feature DataFrame from build_features().

    Returns:
        Tuple of (model trained on all data, CV metrics dict).
    """
    query_ids = sorted(df["query_id"].unique())
    n_queries = len(query_ids)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    # Leave-one-query-out cross-validation
    fold_ndcg5 = []
    fold_ndcg10 = []

    logger.info("Running leave-one-query-out CV (%d folds)...", n_queries)
    for held_out_qid in query_ids:
        train_mask = df["query_id"] != held_out_qid
        val_mask = df["query_id"] == held_out_qid

        train_df = df[train_mask]
        val_df = df[val_mask]

        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["relevance"].values
        groups_train = train_df.groupby("query_id").size().values

        X_val = val_df[FEATURE_NAMES].values
        y_val = val_df["relevance"].values
        groups_val = val_df.groupby("query_id").size().values

        train_data = lgb.Dataset(
            X_train, label=y_train, group=groups_train,
            feature_name=FEATURE_NAMES, free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, group=groups_val,
            reference=train_data, free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        ndcg5 = model.best_score.get("val", {}).get("ndcg@5", 0.0)
        ndcg10 = model.best_score.get("val", {}).get("ndcg@10", 0.0)
        fold_ndcg5.append(ndcg5)
        fold_ndcg10.append(ndcg10)

    cv_metrics = {
        "cv_ndcg5_mean": float(np.mean(fold_ndcg5)),
        "cv_ndcg5_std": float(np.std(fold_ndcg5)),
        "cv_ndcg10_mean": float(np.mean(fold_ndcg10)),
        "cv_ndcg10_std": float(np.std(fold_ndcg10)),
        "cv_ndcg5_per_query": {int(qid): float(s) for qid, s in zip(query_ids, fold_ndcg5)},
        "cv_ndcg10_per_query": {int(qid): float(s) for qid, s in zip(query_ids, fold_ndcg10)},
    }

    logger.info(
        "CV results: NDCG@5=%.3f±%.3f, NDCG@10=%.3f±%.3f",
        cv_metrics["cv_ndcg5_mean"], cv_metrics["cv_ndcg5_std"],
        cv_metrics["cv_ndcg10_mean"], cv_metrics["cv_ndcg10_std"],
    )

    # Train final model on ALL data
    logger.info("Training final model on all %d queries...", n_queries)
    X_all = df[FEATURE_NAMES].values
    y_all = df["relevance"].values
    groups_all = df.groupby("query_id").size().values

    all_data = lgb.Dataset(
        X_all, label=y_all, group=groups_all,
        feature_name=FEATURE_NAMES, free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        all_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    return final_model, cv_metrics


def save_feature_importance(model: lgb.Booster, output_path: Path) -> None:
    """Save feature importance bar chart.

    Args:
        model: Trained LightGBM model.
        output_path: Path to save the PNG chart.
    """
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()

    sorted_idx = np.argsort(importance)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_names, sorted_importance, color="#2196F3")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("LightGBM Ranker — Feature Importance")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance chart saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train LightGBM LambdaRank ranker",
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Only build features CSV, skip training",
    )
    parser.add_argument(
        "--no-ce", action="store_true",
        help="Skip cross-encoder features (faster)",
    )
    parser.add_argument(
        "--load-features", action="store_true",
        help="Load features from CSV instead of recomputing",
    )
    return parser.parse_args()


def main() -> None:
    """Build features, train LambdaRank, save model + importance chart."""
    args = parse_args()

    # ── Check prerequisites ───────────────────────────────────────────────
    if not any(f.exists() for f in LABELS_FILES):
        logger.error("No labels files found: %s", LABELS_FILES)
        sys.exit(1)
    if not args.load_features and not Path(FAISS_INDEX).exists():
        logger.error("FAISS index not found: %s", FAISS_INDEX)
        sys.exit(1)

    if args.load_features and FEATURES_FILE.exists():
        logger.info("Loading features from %s", FEATURES_FILE)
        df = pd.read_csv(FEATURES_FILE)
    else:
        # ── Load labels ───────────────────────────────────────────────────
        labels = load_labels(LABELS_FILES)
        n_queries = len(set(r["query_id"] for r in labels))
        logger.info("Loaded %d labels across %d queries", len(labels), n_queries)

        # ── Setup components ──────────────────────────────────────────────
        logger.info("Loading retrieval components...")
        es_index = ElasticsearchIndex()
        embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
        faiss_idx = FAISSIndex()
        faiss_idx.load(FAISS_INDEX, FAISS_MAPPING)

        reranker = None
        if not args.no_ce and Path(CE_MODEL).exists():
            logger.info("Loading cross-encoder: %s", CE_MODEL)
            reranker = CrossEncoderReranker(model_name=CE_MODEL)
        elif not args.no_ce:
            logger.warning(
                "Cross-encoder not found at %s, skipping CE features", CE_MODEL
            )

        # ── Build features ────────────────────────────────────────────────
        logger.info("Building features...")
        t0 = time.time()
        df = build_features(labels, es_index, faiss_idx, embedder, reranker)
        elapsed = time.time() - t0
        logger.info(
            "Features built in %.1fs: %d rows, %d columns",
            elapsed, len(df), len(df.columns),
        )

        # Save features
        FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(FEATURES_FILE, index=False)
        logger.info("Saved features to %s", FEATURES_FILE)

    if args.features_only:
        print(f"\nFeatures: {len(df)} rows, {len(FEATURE_NAMES)} features")
        print(f"Queries: {df['query_id'].nunique()}")
        print(f"\nFeature stats:")
        print(df[FEATURE_NAMES].describe().round(3).to_string())
        return

    # ── Print feature summary ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Feature Engineering Summary")
    print(f"{'='*62}")
    print(f"  Rows: {len(df)}")
    print(f"  Queries: {df['query_id'].nunique()}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"\n  Feature coverage (non-zero %):")
    for feat in FEATURE_NAMES:
        pct = (df[feat] > 0).mean() * 100
        print(f"    {feat:<25} {pct:5.1f}%")
    print(f"\n  Relevance distribution:")
    for score in sorted(df["relevance"].unique()):
        count = (df["relevance"] == score).sum()
        print(f"    Score {score}: {count} ({count/len(df)*100:.1f}%)")

    # ── Train LightGBM ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Training LightGBM LambdaRank")
    print(f"{'='*62}")

    t0 = time.time()
    model, cv_metrics = train_lgbm(df)
    train_time = time.time() - t0

    print(f"\n  Leave-one-query-out CV results:")
    print(f"    NDCG@5:  {cv_metrics['cv_ndcg5_mean']:.3f} ± {cv_metrics['cv_ndcg5_std']:.3f}")
    print(f"    NDCG@10: {cv_metrics['cv_ndcg10_mean']:.3f} ± {cv_metrics['cv_ndcg10_std']:.3f}")
    print(f"  Training time: {train_time:.1f}s")

    # ── Save model ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.lgb"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    metadata = {
        "model_type": "lightgbm_lambdarank",
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "n_queries": int(df["query_id"].nunique()),
        "n_samples": len(df),
        "params": {
            "objective": "lambdarank",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "num_boost_round": 200,
        },
        "cv_metrics": cv_metrics,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Feature importance chart ──────────────────────────────────────────
    chart_path = Path("docs/feature_importance.png")
    save_feature_importance(model, chart_path)

    # Console table
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    sorted_pairs = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*62}")
    print(f"  Feature Importance (Gain)")
    print(f"{'='*62}")
    for name, imp in sorted_pairs:
        bar = "#" * int(imp / max(importance) * 30)
        print(f"  {name:<25} {imp:8.1f}  {bar}")

    # ── MLflow logging ────────────────────────────────────────────────────
    print(f"\nLogging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="lambdarank_v2") as run:
        mlflow.set_tag("method", "lightgbm_lambdarank")
        mlflow.set_tag("stage", "ranker_training")
        mlflow.log_param("n_features", len(FEATURE_NAMES))
        mlflow.log_param("n_queries", int(df["query_id"].nunique()))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("num_boost_round", 200)
        mlflow.log_param("num_leaves", 31)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("cv_ndcg5", round(cv_metrics["cv_ndcg5_mean"], 4))
        mlflow.log_metric("cv_ndcg10", round(cv_metrics["cv_ndcg10_mean"], 4))
        mlflow.log_metric("cv_ndcg5_std", round(cv_metrics["cv_ndcg5_std"], 4))
        mlflow.log_metric("cv_ndcg10_std", round(cv_metrics["cv_ndcg10_std"], 4))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(chart_path))
        mlflow.log_artifact(str(FEATURES_FILE))
        print(f"  Logged run (ID: {run.info.run_id})")

    print(f"\nDone. Model: {model_path}")
    print(f"View results: make mlflow")


if __name__ == "__main__":
    main()
