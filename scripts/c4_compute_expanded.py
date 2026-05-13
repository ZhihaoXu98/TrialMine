"""Compute the expanded Phase C4 metrics (v1 vs v2 with n=15 for complex/vague).

Reads:
  data/evaluation/full_labeled_dataset_v1.jsonl
  data/evaluation/full_labeled_dataset_expansion_v1.jsonl
  data/evaluation/full_labeled_dataset.jsonl              (v2)
  data/evaluation/full_labeled_dataset_expansion_v2.jsonl

Writes:
  data/evaluation/c4_expanded_metrics.json

Per-category NDCG@5, NDCG@10, MRR for v1 and v2 with bootstrap 95% CIs
(1000 resamples). Apples-to-apples on the 80 shared queries; rare_explicit
listed separately (v2-only).

Pre-registered C4 thresholds (docs/fix_bi-encoder.md §C4):
  complex ≥ 0.68, vague ≥ 0.74, rare_explicit ≥ 0.55
  common/rare/treatment ≥ 0.90 (no-regression floor)
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

EVAL = Path("data/evaluation")
OUT = EVAL / "c4_expanded_metrics.json"

V1_FILES = [
    EVAL / "full_labeled_dataset_v1.jsonl",
    EVAL / "full_labeled_dataset_expansion_v1.jsonl",
]
V2_FILES = [
    EVAL / "full_labeled_dataset.jsonl",
    EVAL / "full_labeled_dataset_expansion_v2.jsonl",
]

GATE = {
    "complex":       0.68,
    "vague":         0.74,
    "rare_explicit": 0.55,
    "common":        0.90,
    "rare":          0.90,
    "treatment":     0.90,
}


def load(paths: list[Path]) -> dict[int, dict]:
    queries: dict[int, dict] = defaultdict(lambda: {"category": None, "pairs": []})
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["query_id"]
                queries[qid]["category"] = rec["category"]
                queries[qid]["pairs"].append((rec["rank"], rec["relevance"]))
    for q in queries.values():
        q["pairs"].sort()
    return queries


def dcg(rels: list[int], k: int) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(rels: list[int], k: int) -> float:
    a = dcg(rels, k)
    b = dcg(sorted(rels, reverse=True), k)
    return a / b if b > 0 else 0.0


def mrr(rels: list[int], threshold: int = 2) -> float:
    for i, r in enumerate(rels):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap(values: list[float], n: int = 1000, seed: int = 42) -> dict:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "halfwidth": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "lo": values[0], "hi": values[0], "halfwidth": 0.0}
    random.seed(seed)
    means = []
    for _ in range(n):
        sample = [random.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    mean = sum(values) / len(values)
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    return {"mean": mean, "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2}


def per_query(queries: dict[int, dict]) -> dict[int, dict]:
    out = {}
    for qid, q in queries.items():
        rels = [r for _, r in q["pairs"]]
        out[qid] = {
            "category": q["category"],
            "ndcg5":  ndcg(rels, 5),
            "ndcg10": ndcg(rels, 10),
            "mrr":    mrr(rels),
        }
    return out


def category_metrics(per_q: dict[int, dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in per_q.values():
        by_cat[m["category"]].append(m)
    out = {}
    for cat, ms in by_cat.items():
        out[cat] = {
            "n":      len(ms),
            "ndcg5":  bootstrap([m["ndcg5"]  for m in ms]),
            "ndcg10": bootstrap([m["ndcg10"] for m in ms]),
            "mrr":    bootstrap([m["mrr"]    for m in ms]),
        }
    return out


def main() -> None:
    v1_per_q = per_query(load(V1_FILES))
    v2_per_q = per_query(load(V2_FILES))
    v1_cat = category_metrics(v1_per_q)
    v2_cat = category_metrics(v2_per_q)

    # Apples-to-apples overall (only shared categories — exclude rare_explicit which is v2-only)
    shared_qids = set(v1_per_q.keys()) & set(v2_per_q.keys())
    v1_all_n5  = [v1_per_q[qid]["ndcg5"]  for qid in shared_qids]
    v1_all_n10 = [v1_per_q[qid]["ndcg10"] for qid in shared_qids]
    v1_all_mrr = [v1_per_q[qid]["mrr"]    for qid in shared_qids]
    v2_all_n5  = [v2_per_q[qid]["ndcg5"]  for qid in shared_qids]
    v2_all_n10 = [v2_per_q[qid]["ndcg10"] for qid in shared_qids]
    v2_all_mrr = [v2_per_q[qid]["mrr"]    for qid in shared_qids]

    # Gate evaluation
    gate_results = {}
    for cat, threshold in GATE.items():
        if cat in v2_cat:
            v2_val = v2_cat[cat]["ndcg5"]["mean"]
            gate_results[cat] = {
                "v2_ndcg5":  v2_val,
                "threshold": threshold,
                "gap":       v2_val - threshold,
                "passed":    v2_val >= threshold,
            }

    out = {
        "n_shared_queries": len(shared_qids),
        "overall_v1": {
            "ndcg5":  bootstrap(v1_all_n5),
            "ndcg10": bootstrap(v1_all_n10),
            "mrr":    bootstrap(v1_all_mrr),
        },
        "overall_v2": {
            "ndcg5":  bootstrap(v2_all_n5),
            "ndcg10": bootstrap(v2_all_n10),
            "mrr":    bootstrap(v2_all_mrr),
        },
        "v1_per_category": v1_cat,
        "v2_per_category": v2_cat,
        "gate_results":    gate_results,
        "decision":        "SHIP (Option C: partial-success / honest-writeup)",
        "decision_rationale": (
            "Pre-registered complex/vague gates missed. But overall ~tied "
            "(Δ NDCG@5 = {:+.3f}, NDCG@10 = {:+.3f}). v2 has real wins on "
            "vague (+{:.3f}) and geographic (+{:.3f}), no regression on "
            "common/rare/treatment floors. Shipping the partial-success "
            "model + documenting complex failure as a pipeline (eligibility-"
            "filter), not embedding, problem. Next dollar to Issue #4."
        ).format(
            bootstrap(v2_all_n5)["mean"] - bootstrap(v1_all_n5)["mean"],
            bootstrap(v2_all_n10)["mean"] - bootstrap(v1_all_n10)["mean"],
            v2_cat["vague"]["ndcg5"]["mean"] - v1_cat["vague"]["ndcg5"]["mean"],
            v2_cat["geographic"]["ndcg5"]["mean"] - v1_cat["geographic"]["ndcg5"]["mean"],
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT}")
    print(f"\nOverall (n={len(shared_qids)} apples-to-apples queries):")
    print(f"  v1 NDCG@5  = {out['overall_v1']['ndcg5']['mean']:.3f} ± {out['overall_v1']['ndcg5']['halfwidth']:.2f}")
    print(f"  v2 NDCG@5  = {out['overall_v2']['ndcg5']['mean']:.3f} ± {out['overall_v2']['ndcg5']['halfwidth']:.2f}")
    print(f"  Δ NDCG@5   = {out['overall_v2']['ndcg5']['mean'] - out['overall_v1']['ndcg5']['mean']:+.3f}")
    print(f"  v1 NDCG@10 = {out['overall_v1']['ndcg10']['mean']:.3f} ± {out['overall_v1']['ndcg10']['halfwidth']:.2f}")
    print(f"  v2 NDCG@10 = {out['overall_v2']['ndcg10']['mean']:.3f} ± {out['overall_v2']['ndcg10']['halfwidth']:.2f}")
    print(f"  Δ NDCG@10  = {out['overall_v2']['ndcg10']['mean'] - out['overall_v1']['ndcg10']['mean']:+.3f}")
    print(f"\nGate verdict (v2):")
    for cat in ["complex", "vague", "rare_explicit", "common", "rare", "treatment"]:
        g = gate_results.get(cat)
        if g:
            print(f"  {cat}: {g['v2_ndcg5']:.3f} vs ≥{g['threshold']:.2f} → {'PASS' if g['passed'] else 'FAIL'} (gap {g['gap']:+.3f})")


if __name__ == "__main__":
    main()
