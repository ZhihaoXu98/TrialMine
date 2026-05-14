"""Phase C3 — sweep blender α on the held-out 65-query set with the v2 CE.

The production blend (src/TrialMine/models/cross_encoder.py:97) is:

    blended = α * rrf_norm + (1 − α) * ce_sigmoid

with α = 0.7 today (RRF dominant) and a min-max RRF normalization computed
within the candidate pool. This script replicates that math externally and
sweeps α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} without touching production code, so we
can measure "with v2 CE wired in, what α maximizes NDCG@5?" before deciding
whether to bump the default in C5a.

For each (α, query): score the labeled pool by blended_score, compute NDCG@5/
@10 + MRR vs graded labels (0–3). Aggregate to mean with bootstrap 95% CI
(1000 resamples over queries). Pick winner by NDCG@5 (tiebreak NDCG@10).

Per-query CE / RRF scores are computed once and reused across the 5 α values
— the sweep itself is just arithmetic + re-sort, no model inference.

Output: a markdown table printed to stdout + a JSON dump of per-(α, query)
metrics for downstream use.

Usage:
    python scripts/c3_sweep_blender_alpha.py \
      --labels data/evaluation/full_labeled_dataset.jsonl \
      --output-json data/evaluation/c3_alpha_sweep_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
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
V2_CE_PATH = "models/cross-encoder/fine-tuned-v2"
EMBEDDER_PATH = "models/embeddings/fine-tuned-v2"
FAISS_INDEX = "data/faiss_finetuned_v2.index"
FAISS_MAPPING = "data/faiss_finetuned_v2.json"

ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7)
MAX_TEXT_CHARS = 2048
N_BOOTSTRAP = 1000
HYBRID_TOP_K = 200


def load_labels(path: Path) -> dict[int, dict]:
    """Group labels by query_id."""
    by_qid: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = int(r["query_id"])
        rec = by_qid.setdefault(
            qid,
            {"qid": qid, "query": str(r["query"]), "category": str(r["category"]), "candidates": []},
        )
        rec["candidates"].append((str(r["nct_id"]), int(r["relevance"])))
    return by_qid


def fetch_trial_text(nct_ids: set[str], db_path: Path) -> dict[str, str]:
    """{nct_id: 'title [SEP] conditions [SEP] brief_summary'} truncated to 2048 chars."""
    if not nct_ids:
        return {}
    cache: dict[str, str] = {}
    conn = sqlite3.connect(str(db_path))
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


def hybrid_rrf_for_pool(query: str, pool_ncts: list[str], hybrid: HybridRetriever) -> dict[str, float]:
    """Rank-based RRF-like score (1/rank). Candidates outside hybrid top_k get 0."""
    hits = hybrid.search(query, top_k=HYBRID_TOP_K)
    score_map = {h["nct_id"]: 1.0 / (i + 1) for i, h in enumerate(hits)}
    return {nct: score_map.get(nct, 0.0) for nct in pool_ncts}


def ce_logits_for_pool(
    query: str, pool_ncts: list[str], trial_text: dict[str, str], ce_model: CrossEncoder
) -> dict[str, float]:
    """CE logit per NCT in the pool. Missing trial text → 0 logit (sigmoid 0.5)."""
    valid = [nct for nct in pool_ncts if nct in trial_text]
    if not valid:
        return {nct: 0.0 for nct in pool_ncts}
    pairs = [(query, trial_text[nct]) for nct in valid]
    raw = ce_model.predict(pairs, batch_size=32, show_progress_bar=False)
    out = {nct: float(s) for nct, s in zip(valid, raw, strict=False)}
    for nct in pool_ncts:
        out.setdefault(nct, 0.0)
    return out


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """[0, 1] min-max normalization within the pool — matches production blender."""
    vals = list(scores.values())
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / rng for k, v in scores.items()}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blended_for_alpha(rrf_norm: dict[str, float], ce_sigmoid: dict[str, float], alpha: float) -> dict[str, float]:
    """blended = α * rrf_norm + (1 − α) * ce_sigmoid. Matches production blender."""
    keys = set(rrf_norm) | set(ce_sigmoid)
    return {k: alpha * rrf_norm.get(k, 0.0) + (1.0 - alpha) * ce_sigmoid.get(k, 0.0) for k in keys}


def per_query_metrics(candidates: list[tuple[str, int]], scores: dict[str, float]) -> dict[str, float]:
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
    """Mean + 95% bootstrap CI (mean, lower, upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n)])
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C3 — blender α sweep")
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/full_labeled_dataset.jsonl",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    logger.info("Loading labels from %s", args.labels)
    labels = load_labels(args.labels)
    logger.info("Loaded %d queries", len(labels))

    nct_ids = {nct for q in labels.values() for nct, _ in q["candidates"]}
    trial_text = fetch_trial_text(nct_ids, TRIALS_DB)
    logger.info("Trial-text cache: %d / %d NCT IDs resolved", len(trial_text), len(nct_ids))

    logger.info("Loading retrieval stack ...")
    es = ElasticsearchIndex(es_url="http://localhost:9200", index_name="trials")
    embedder = TrialEmbedder(EMBEDDER_PATH)
    faiss_idx = FAISSIndex(dimension=768)
    faiss_idx.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es, semantic=faiss_idx, embedder=embedder)

    logger.info("Loading v2 CE from %s ...", V2_CE_PATH)
    v2_ce = CrossEncoder(V2_CE_PATH, num_labels=1, max_length=512, device="cpu")

    # Compute RRF + CE scores once per query (the expensive step), then sweep α cheaply.
    logger.info("Computing per-query RRF and v2-CE scores once ...")
    per_query_rrf_ce: dict[int, dict] = {}
    qids = sorted(labels)
    for i, qid in enumerate(qids, 1):
        q_data = labels[qid]
        pool_ncts = [nct for nct, _ in q_data["candidates"]]
        rrf = hybrid_rrf_for_pool(q_data["query"], pool_ncts, hybrid)
        ce_logits = ce_logits_for_pool(q_data["query"], pool_ncts, trial_text, v2_ce)
        per_query_rrf_ce[qid] = {
            "rrf_norm": minmax_normalize(rrf),
            "ce_sigmoid": {k: sigmoid(v) for k, v in ce_logits.items()},
            "category": q_data["category"],
            "candidates": q_data["candidates"],
        }
        if i % 10 == 0:
            logger.info("  scored %d/%d queries", i, len(qids))

    # Sweep α — pure arithmetic per query
    sweep: dict[float, list[dict]] = {a: [] for a in ALPHAS}
    for alpha in ALPHAS:
        for qid in qids:
            d = per_query_rrf_ce[qid]
            blended = blended_for_alpha(d["rrf_norm"], d["ce_sigmoid"], alpha)
            m = per_query_metrics(d["candidates"], blended)
            m["qid"] = qid
            m["category"] = d["category"]
            sweep[alpha].append(m)

    # Headline table
    print(f"\n## Blender α sweep on full_labeled (n={len(labels)} queries, v2 CE wired in)\n")
    print("| α (RRF weight) | (1−α) CE weight | NDCG@5 | NDCG@10 | MRR |")
    print("|---|---|---|---|---|")
    rows = []
    for alpha in ALPHAS:
        ms = sweep[alpha]
        n5 = bootstrap_ci([m["ndcg5"] for m in ms])
        n10 = bootstrap_ci([m["ndcg10"] for m in ms])
        mr = bootstrap_ci([m["mrr"] for m in ms])
        rows.append({"alpha": alpha, "ndcg5": n5, "ndcg10": n10, "mrr": mr})
        marker = ""
        prod_marker = " (← production default)" if alpha == 0.7 else ""
        print(
            f"| {alpha:.1f}{prod_marker} | {1 - alpha:.1f} | "
            f"{n5[0]:.3f} [{n5[1]:.3f}, {n5[2]:.3f}] | "
            f"{n10[0]:.3f} [{n10[1]:.3f}, {n10[2]:.3f}] | "
            f"{mr[0]:.3f} [{mr[1]:.3f}, {mr[2]:.3f}] |{marker}"
        )

    # Pick winner — max NDCG@5, tiebreak NDCG@10
    rows.sort(key=lambda r: (-r["ndcg5"][0], -r["ndcg10"][0]))
    winner = rows[0]
    logger.info(
        "Winning α = %.1f (NDCG@5 = %.4f, NDCG@10 = %.4f, MRR = %.4f)",
        winner["alpha"],
        winner["ndcg5"][0],
        winner["ndcg10"][0],
        winner["mrr"][0],
    )
    if winner["alpha"] == 0.7:
        recommendation = "no change recommended"
    else:
        direction = "more CE weight" if winner["alpha"] < 0.7 else "more RRF weight"
        recommendation = f"sweep suggests shifting to α = {winner['alpha']:.1f} ({direction})"
    print(
        f"\n**Winner: α = {winner['alpha']:.1f}** "
        f"(NDCG@5 = {winner['ndcg5'][0]:.4f} [{winner['ndcg5'][1]:.3f}, {winner['ndcg5'][2]:.3f}], "
        f"NDCG@10 = {winner['ndcg10'][0]:.4f}, MRR = {winner['mrr'][0]:.4f}). "
        f"Production default is α = 0.7 (RRF dominant); {recommendation}."
    )

    # Per-category NDCG@5 across α
    cats = sorted({m["category"] for m in sweep[ALPHAS[0]]})
    print(f"\n## Per-category NDCG@5 across α\n")
    header = "| Category | n | " + " | ".join(f"α={a:.1f}" for a in ALPHAS) + " |"
    sep = "|---|---|" + "|".join("---" for _ in ALPHAS) + "|"
    print(header)
    print(sep)
    for cat in cats:
        n = sum(1 for m in sweep[ALPHAS[0]] if m["category"] == cat)
        row = [cat, str(n)]
        for a in ALPHAS:
            vals = [m["ndcg5"] for m in sweep[a] if m["category"] == cat]
            row.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
        print("| " + " | ".join(row) + " |")

    if args.output_json:
        out = {
            "alphas": list(ALPHAS),
            "n_queries": len(labels),
            "winner_alpha": winner["alpha"],
            "headline": {
                f"alpha={a:.1f}": {
                    "ndcg5_mean": bootstrap_ci([m["ndcg5"] for m in sweep[a]])[0],
                    "ndcg5_ci": list(bootstrap_ci([m["ndcg5"] for m in sweep[a]])[1:]),
                    "ndcg10_mean": bootstrap_ci([m["ndcg10"] for m in sweep[a]])[0],
                    "ndcg10_ci": list(bootstrap_ci([m["ndcg10"] for m in sweep[a]])[1:]),
                    "mrr_mean": bootstrap_ci([m["mrr"] for m in sweep[a]])[0],
                    "mrr_ci": list(bootstrap_ci([m["mrr"] for m in sweep[a]])[1:]),
                }
                for a in ALPHAS
            },
            "per_query": {f"alpha={a:.1f}": sweep[a] for a in ALPHAS},
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2) + "\n")
        logger.info("Wrote sweep results → %s", args.output_json)


if __name__ == "__main__":
    main()
