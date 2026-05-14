"""Phase R-E — v2 vs v3 LightGBM ranker held-out evaluation.

Per query: run hybrid retrieval (top 50) once, score candidates with v2 CE
once, compute features once. Then predict with v2 LGB AND v3 LGB on the
same feature matrix and compare NDCG@5/@10 + MRR. ~2× faster than running
scripts/evaluate.py twice.

Runs on the two held-out sets:
- data/evaluation/full_labeled_dataset.jsonl (65q)
- data/evaluation/full_labeled_dataset_expansion_v2.jsonl (20q)

Output: 4 rows (v2 vs v3 × 65q vs 20q) with bootstrap 95% CIs.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TrialMine.evaluation.metrics import mrr, ndcg_at_k
from TrialMine.models.cross_encoder import CrossEncoderReranker
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.models.ranker import FEATURE_NAMES, RankingBlender, compute_features
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

import os
V2_RANKER = os.environ.get("RANKER_A", "models/ranker/v2/model.lgb")
V3_RANKER = os.environ.get("RANKER_B", "models/ranker/v3/model.lgb")
CE_MODEL = "models/cross-encoder/fine-tuned-v2"
EMBEDDER = "models/embeddings/fine-tuned-v2"
FAISS_INDEX = "data/faiss_finetuned_v2.index"
FAISS_MAPPING = "data/faiss_finetuned_v2.json"
HYBRID_TOP_K = 50
N_BOOTSTRAP = 1000

DATASETS = [
    ("full_65q", PROJECT_ROOT / "data/evaluation/full_labeled_dataset.jsonl"),
    ("expansion_20q", PROJECT_ROOT / "data/evaluation/full_labeled_dataset_expansion_v2.jsonl"),
]


def load_labels(path: Path) -> dict[int, dict]:
    by_qid: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = int(r["query_id"])
        rec = by_qid.setdefault(
            qid,
            {"qid": qid, "query": str(r["query"]),
             "category": str(r.get("category", "all")),
             "relevance": {}, "relevant_ids": set()},
        )
        rec["relevance"][r["nct_id"]] = int(r["relevance"])
        if int(r["relevance"]) > 0:
            rec["relevant_ids"].add(r["nct_id"])
    return by_qid


def build_trial_text(doc: dict) -> str:
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("conditions"):
        parts.append(doc["conditions"] if isinstance(doc["conditions"], str) else " ".join(doc["conditions"]))
    if doc.get("brief_summary"):
        parts.append(doc["brief_summary"])
    return " [SEP] ".join(parts)[:2048] if parts else ""


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n)])
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    es = ElasticsearchIndex(es_url="http://localhost:9200", index_name="trials")
    embedder = TrialEmbedder(EMBEDDER)
    faiss = FAISSIndex(dimension=768); faiss.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es, semantic=faiss, embedder=embedder)
    ce = CrossEncoderReranker(model_name=CE_MODEL, device="cpu")
    v2_lgb = RankingBlender(model_path=V2_RANKER)
    v3_lgb = RankingBlender(model_path=V3_RANKER)
    logger.info("All models loaded")

    out: dict[str, dict] = {}
    for ds_name, labels_path in DATASETS:
        logger.info("=== %s: %s ===", ds_name, labels_path.name)
        labels = load_labels(labels_path)
        logger.info("Loaded %d queries", len(labels))

        per_query = {"v2": [], "v3": [], "pure_ce": []}
        per_cat = {"v2": defaultdict(list), "v3": defaultdict(list), "pure_ce": defaultdict(list)}

        for i, (qid, q_data) in enumerate(sorted(labels.items()), 1):
            # Hybrid retrieval once
            cands = hybrid.search(q_data["query"], top_k=HYBRID_TOP_K)
            # Enrich for CE + features (need trial_text + brief_summary + eligibility)
            for c in cands:
                if not c.get("brief_summary"):
                    doc = es.get_trial(c["nct_id"])
                    if doc:
                        c["brief_summary"] = doc.get("brief_summary", "")
                        c["eligibility_criteria"] = doc.get("eligibility_criteria", "")
                c["trial_text"] = build_trial_text(c)
            # CE score once
            texts = [c["trial_text"] for c in cands]
            ce_logits = ce.score(q_data["query"], texts)
            import math
            for c, logit in zip(cands, ce_logits, strict=False):
                c["cross_encoder_score"] = 1.0 / (1.0 + math.exp(-logit))
            rel_dict = {k: float(v) for k, v in q_data["relevance"].items()}
            # Predict with v2 + v3 LGB
            for lgb_name, lgb in [("v2", v2_lgb), ("v3", v3_lgb)]:
                # Make per-LGB copies because compute_features may mutate
                local = [dict(c) for c in cands]
                ranked = lgb.rerank(q_data["query"], local, top_k=HYBRID_TOP_K)
                ranked_ids = [c["nct_id"] for c in ranked]
                ndcg5 = ndcg_at_k(ranked_ids, rel_dict, 5)
                ndcg10 = ndcg_at_k(ranked_ids, rel_dict, 10)
                mrr_val = mrr(ranked_ids, q_data["relevant_ids"])
                per_query[lgb_name].append({"qid": qid, "category": q_data["category"],
                                             "ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr_val})
                per_cat[lgb_name][q_data["category"]].append(ndcg5)
            # Pure CE — sort cands by cross_encoder_score directly, no LGB
            ranked_ce = sorted(cands, key=lambda c: -c["cross_encoder_score"])
            ranked_ids_ce = [c["nct_id"] for c in ranked_ce]
            ndcg5_ce = ndcg_at_k(ranked_ids_ce, rel_dict, 5)
            ndcg10_ce = ndcg_at_k(ranked_ids_ce, rel_dict, 10)
            mrr_ce = mrr(ranked_ids_ce, q_data["relevant_ids"])
            per_query["pure_ce"].append({"qid": qid, "category": q_data["category"],
                                          "ndcg5": ndcg5_ce, "ndcg10": ndcg10_ce, "mrr": mrr_ce})
            per_cat["pure_ce"][q_data["category"]].append(ndcg5_ce)
            if i % 10 == 0:
                logger.info("  scored %d/%d queries", i, len(labels))

        out[ds_name] = {"per_query": per_query, "per_cat": per_cat, "n": len(labels)}

    # Report — 3-way comparison (v2 LGB, v3 LGB, pure CE on hybrid top-50)
    print("\n" + "=" * 80)
    print("HELD-OUT EVAL — v2 LGB vs v3 LGB vs PURE v2 CE (all on hybrid top-50)")
    print("=" * 80)
    print("\n| Dataset | n | Ranker | NDCG@5 | NDCG@10 | MRR |")
    print("|---|---|---|---|---|---|")
    for ds_name, data in out.items():
        # Compute means so we can mark winners
        means = {}
        for k in ("v2", "v3", "pure_ce"):
            means[k] = bootstrap_ci([m["ndcg5"] for m in data["per_query"][k]])[0]
        winner = max(means, key=means.get)
        for sys_name, label in [("v2", "v2 LGB"), ("v3", "v3-regularized LGB"), ("pure_ce", "Pure v2 CE (no LGB)")]:
            ms = data["per_query"][sys_name]
            n5 = bootstrap_ci([m["ndcg5"] for m in ms])
            n10 = bootstrap_ci([m["ndcg10"] for m in ms])
            mr = bootstrap_ci([m["mrr"] for m in ms])
            marker = " **← winner**" if sys_name == winner else ""
            print(f"| {ds_name} | {data['n']} | {label} | {n5[0]:.3f} [{n5[1]:.3f}, {n5[2]:.3f}] | {n10[0]:.3f} [{n10[1]:.3f}, {n10[2]:.3f}] | {mr[0]:.3f} [{mr[1]:.3f}, {mr[2]:.3f}] |{marker}")

    print("\n## All pairwise deltas (mean NDCG@5)")
    for ds_name, data in out.items():
        v2_mean = bootstrap_ci([m["ndcg5"] for m in data["per_query"]["v2"]])[0]
        v3_mean = bootstrap_ci([m["ndcg5"] for m in data["per_query"]["v3"]])[0]
        ce_mean = bootstrap_ci([m["ndcg5"] for m in data["per_query"]["pure_ce"]])[0]
        print(f"  {ds_name}:")
        print(f"    v3-regularized − v2 LGB      = {v3_mean - v2_mean:+.4f}")
        print(f"    pure CE       − v2 LGB      = {ce_mean - v2_mean:+.4f}")
        print(f"    pure CE       − v3-regularized LGB = {ce_mean - v3_mean:+.4f}")

    print("\n## Per-category NDCG@5 (full_65q) — 3-way")
    cats = sorted(out["full_65q"]["per_cat"]["v3"].keys())
    print("| Category | n | v2 LGB | v3-reg LGB | pure CE | Δ pureCE−v2 | Δ pureCE−v3reg |")
    print("|---|---|---|---|---|---|---|")
    for cat in cats:
        v2_vals = out["full_65q"]["per_cat"]["v2"][cat]
        v3_vals = out["full_65q"]["per_cat"]["v3"][cat]
        ce_vals = out["full_65q"]["per_cat"]["pure_ce"][cat]
        v2m = sum(v2_vals) / len(v2_vals)
        v3m = sum(v3_vals) / len(v3_vals)
        cem = sum(ce_vals) / len(ce_vals)
        print(f"| {cat} | {len(v3_vals)} | {v2m:.3f} | {v3m:.3f} | {cem:.3f} | {cem-v2m:+.3f} | {cem-v3m:+.3f} |")

    # Save JSON
    out_path = PROJECT_ROOT / "data/evaluation/r_e_lgb_v2_vs_v3.json"
    out_path.write_text(json.dumps({k: {"per_query": v["per_query"], "n": v["n"],
                                         "per_cat": {ln: {c: vals for c, vals in cd.items()}
                                                      for ln, cd in v["per_cat"].items()}}
                                     for k, v in out.items()}, indent=2, default=list) + "\n")
    logger.info("Wrote per-query results → %s", out_path)


if __name__ == "__main__":
    main()
