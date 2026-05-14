"""Phase C2 — pure-CE evaluation on the held-out 65q + 20q expansion sets.

Three systems × each labels file:
  1. Hybrid baseline — RRF only (no CE). Rank-based score from a fresh hybrid
     retrieval; candidates not in the new hybrid top_k get score 0.
  2. v1 binary CE — re-rank labeled pool by v1 CE logit (pure replacement,
     no blender, no LightGBM).
  3. v2 graded CE — re-rank labeled pool by v2 CE logit (pure replacement).

Metric pool per query is the labeled candidates from the input JSONL (~20/q
for the 65-query set, ~20/q for the expansion). Each system's score map
controls the ordering; NDCG@5/@10 + MRR are computed against the graded
relevance labels (0-3). Mean is reported with bootstrap 95% CI (1000
resamples over queries) overall and per-category.

Trial text matches the production CE inference format
(`HybridRetriever.full_pipeline`, src/TrialMine/retrieval/hybrid.py:292-299):
title + [SEP] + conditions + [SEP] + brief_summary, truncated to 2048 chars.

Usage:
    python scripts/c2_eval_pure_ce.py \
      --labels data/evaluation/full_labeled_dataset.jsonl \
      --dataset-name full_labeled

    python scripts/c2_eval_pure_ce.py \
      --labels data/evaluation/full_labeled_dataset_expansion_v2.jsonl \
      --dataset-name expansion_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentence_transformers import CrossEncoder

from TrialMine.evaluation.metrics import mrr, ndcg_at_k
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

TRIALS_DB = PROJECT_ROOT / "data" / "trials.db"
V1_CE_PATH = "models/cross-encoder/fine-tuned-v1"
V2_CE_PATH = "models/cross-encoder/fine-tuned-v2"
EMBEDDER_PATH = "models/embeddings/fine-tuned-v2"
FAISS_INDEX = "data/faiss_finetuned_v2.index"
FAISS_MAPPING = "data/faiss_finetuned_v2.json"

MAX_TEXT_CHARS = 2048
N_BOOTSTRAP = 1000
HYBRID_TOP_K = 200  # generous coverage for the labeled pool (which is top-20 of prod pipeline)
SYSTEMS = ("hybrid", "v1_ce", "v2_ce")


def load_labels(path: Path) -> dict[int, dict]:
    """Group labels by query_id → {qid: {query, category, candidates: [(nct, grade), ...]}}."""
    by_qid: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = int(r["query_id"])
        rec = by_qid.setdefault(
            qid,
            {
                "qid": qid,
                "query": str(r["query"]),
                "category": str(r.get("category", "all")),
                "candidates": [],
            },
        )
        rec["candidates"].append((str(r["nct_id"]), int(r["relevance"])))
    return by_qid


def fetch_trial_text(nct_ids: set[str], db_path: Path) -> dict[str, str]:
    """{nct_id: 'title [SEP] conditions [SEP] brief_summary'} truncated to 2048 chars."""
    if not nct_ids:
        return {}
    conn = sqlite3.connect(str(db_path))
    cache: dict[str, str] = {}
    try:
        ncts = sorted(nct_ids)
        chunk = 500
        for i in range(0, len(ncts), chunk):
            batch = ncts[i : i + chunk]
            placeholders = ",".join("?" * len(batch))
            cur = conn.execute(
                f"SELECT nct_id, title, conditions, brief_summary "
                f"FROM trials WHERE nct_id IN ({placeholders})",
                batch,
            )
            for nct, title, conds, summary in cur.fetchall():
                parts = []
                if title:
                    parts.append(title)
                if conds:
                    parts.append(conds)
                if summary:
                    parts.append(summary)
                text = " [SEP] ".join(parts) if parts else ""
                if len(text) > MAX_TEXT_CHARS:
                    text = text[:MAX_TEXT_CHARS]
                if text:
                    cache[nct] = text
    finally:
        conn.close()
    return cache


def hybrid_scores_for_pool(
    query: str,
    pool_ncts: list[str],
    hybrid: HybridRetriever,
    top_k: int = HYBRID_TOP_K,
) -> dict[str, float]:
    """Rank-based RRF-like score: higher = better. Candidates outside hybrid top_k get 0."""
    hits = hybrid.search(query, top_k=top_k)
    score_map = {h["nct_id"]: 1.0 / (i + 1) for i, h in enumerate(hits)}
    return {nct: score_map.get(nct, 0.0) for nct in pool_ncts}


def ce_scores_for_pool(
    query: str,
    pool_ncts: list[str],
    trial_text: dict[str, str],
    ce_model: CrossEncoder,
    batch_size: int = 32,
) -> dict[str, float]:
    """CE logit score per NCT in the pool. NCTs with no trial text get 0."""
    valid = [nct for nct in pool_ncts if nct in trial_text]
    if not valid:
        return {nct: 0.0 for nct in pool_ncts}
    pairs = [(query, trial_text[nct]) for nct in valid]
    raw = ce_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    out = {nct: float(s) for nct, s in zip(valid, raw, strict=False)}
    for nct in pool_ncts:
        out.setdefault(nct, 0.0)
    return out


def per_query_metrics(
    candidates: list[tuple[str, int]],
    scores: dict[str, float],
) -> dict[str, float]:
    """Sort by score desc, compute NDCG@5/@10 + MRR using graded labels."""
    ranked = sorted(candidates, key=lambda c: -scores.get(c[0], 0.0))
    result_ids = [nct for nct, _ in ranked]
    relevance_scores = {nct: float(grade) for nct, grade in candidates}
    relevant_ids = {nct for nct, grade in candidates if grade > 0}
    return {
        "ndcg5": ndcg_at_k(result_ids, relevance_scores, 5),
        "ndcg10": ndcg_at_k(result_ids, relevance_scores, 10),
        "mrr": mrr(result_ids, relevant_ids),
    }


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = 42) -> tuple[float, float, float]:
    """Mean + 95% bootstrap CI. Returns (mean, lower, upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boot_means = np.array(
        [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n)]
    )
    return float(arr.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def fmt_ci(m: float, lo: float, hi: float) -> str:
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def score_all_queries(
    labels: dict[int, dict],
    trial_text: dict[str, str],
    hybrid: HybridRetriever,
    v1_ce: CrossEncoder,
    v2_ce: CrossEncoder,
) -> dict[str, list[dict]]:
    """For each system, return [{qid, category, ndcg5, ndcg10, mrr}, ...] per query."""
    per_sys: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
    qids = sorted(labels)
    for i, qid in enumerate(qids, 1):
        q_data = labels[qid]
        pool_ncts = [nct for nct, _ in q_data["candidates"]]
        for sys_name, scores in (
            ("hybrid", hybrid_scores_for_pool(q_data["query"], pool_ncts, hybrid)),
            ("v1_ce", ce_scores_for_pool(q_data["query"], pool_ncts, trial_text, v1_ce)),
            ("v2_ce", ce_scores_for_pool(q_data["query"], pool_ncts, trial_text, v2_ce)),
        ):
            m = per_query_metrics(q_data["candidates"], scores)
            m["qid"] = qid
            m["category"] = q_data["category"]
            per_sys[sys_name].append(m)
        if i % 10 == 0:
            logger.info("Scored %d/%d queries", i, len(qids))
    return per_sys


def print_overall_table(per_sys: dict[str, list[dict]], n: int, name: str) -> None:
    print(f"\n## Overall — {name} (n={n} queries)\n")
    print("| System | NDCG@5 | NDCG@10 | MRR |")
    print("|---|---|---|---|")
    for s in SYSTEMS:
        ms = per_sys[s]
        n5 = bootstrap_ci([m["ndcg5"] for m in ms])
        n10 = bootstrap_ci([m["ndcg10"] for m in ms])
        mr = bootstrap_ci([m["mrr"] for m in ms])
        print(f"| {s} | {fmt_ci(*n5)} | {fmt_ci(*n10)} | {fmt_ci(*mr)} |")


def print_per_category(per_sys: dict[str, list[dict]], metric: str, name: str) -> None:
    cats = sorted({m["category"] for m in per_sys["v2_ce"]})
    print(f"\n## Per-category {metric.upper()} — {name}\n")
    print("| Category | n | hybrid | v1_ce | v2_ce |")
    print("|---|---|---|---|---|")
    for cat in cats:
        n = sum(1 for m in per_sys["v2_ce"] if m["category"] == cat)
        row = [cat, str(n)]
        for s in SYSTEMS:
            vals = [m[metric] for m in per_sys[s] if m["category"] == cat]
            if vals:
                mu, lo, hi = bootstrap_ci(vals)
                row.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                row.append("—")
        print("| " + " | ".join(row) + " |")


def flag_changes(per_sys: dict[str, list[dict]], name: str) -> None:
    """Highlight per-category v2_ce vs v1_ce drift (> 0.02) and v2_ce strength (> 0.80)."""
    print(f"\n## Flags — {name}\n")
    cats = sorted({m["category"] for m in per_sys["v2_ce"]})
    any_flag = False
    for cat in cats:
        v1 = [m["ndcg5"] for m in per_sys["v1_ce"] if m["category"] == cat]
        v2 = [m["ndcg5"] for m in per_sys["v2_ce"] if m["category"] == cat]
        if not v1 or not v2:
            continue
        v1m = sum(v1) / len(v1)
        v2m = sum(v2) / len(v2)
        delta = v2m - v1m
        if delta < -0.02:
            print(f"- ⚠️ REGRESSION: {cat} NDCG@5 v2={v2m:.3f} < v1={v1m:.3f} (Δ={delta:+.3f})")
            any_flag = True
        elif delta > 0.02:
            print(f"- ✓ LIFT: {cat} NDCG@5 v2={v2m:.3f} > v1={v1m:.3f} (Δ={delta:+.3f})")
            any_flag = True
        if v2m > 0.80:
            print(f"- 🌟 STRENGTH: {cat} NDCG@5 v2={v2m:.3f} > 0.80")
            any_flag = True
    if not any_flag:
        print("- (no per-category v2_ce vs v1_ce drift beyond ±0.02)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C2 — pure-CE evaluation")
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/full_labeled_dataset.jsonl",
    )
    parser.add_argument("--dataset-name", type=str, default="full_labeled")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump per-query metrics for downstream use",
    )
    args = parser.parse_args()

    logger.info("Loading labels from %s", args.labels)
    labels = load_labels(args.labels)
    logger.info("Loaded %d queries", len(labels))

    nct_ids = {nct for q in labels.values() for nct, _ in q["candidates"]}
    trial_text = fetch_trial_text(nct_ids, TRIALS_DB)
    logger.info(
        "Trial-text cache: %d / %d NCT IDs resolved", len(trial_text), len(nct_ids)
    )

    logger.info("Loading retrieval stack ...")
    es = ElasticsearchIndex(es_url="http://localhost:9200", index_name="trials")
    embedder = TrialEmbedder(EMBEDDER_PATH)
    faiss_idx = FAISSIndex(dimension=768)
    faiss_idx.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es, semantic=faiss_idx, embedder=embedder)

    logger.info("Loading v1 CE from %s ...", V1_CE_PATH)
    v1_ce = CrossEncoder(V1_CE_PATH, num_labels=1, max_length=512, device="cpu")
    logger.info("Loading v2 CE from %s ...", V2_CE_PATH)
    v2_ce = CrossEncoder(V2_CE_PATH, num_labels=1, max_length=512, device="cpu")

    logger.info("Scoring all queries with 3 systems ...")
    per_sys = score_all_queries(labels, trial_text, hybrid, v1_ce, v2_ce)

    print_overall_table(per_sys, len(labels), args.dataset_name)
    print_per_category(per_sys, "ndcg5", args.dataset_name)
    print_per_category(per_sys, "ndcg10", args.dataset_name)
    print_per_category(per_sys, "mrr", args.dataset_name)
    flag_changes(per_sys, args.dataset_name)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(per_sys, indent=2) + "\n")
        logger.info("Wrote per-query metrics → %s", args.output_json)


if __name__ == "__main__":
    main()
