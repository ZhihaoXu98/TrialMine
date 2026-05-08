"""Compute LLM/human agreement on the full evaluation dataset.

Reads the human-labeled JSONL produced by `scripts/manual_label.py` and
computes:

  - Cohen's kappa (unweighted, linear-weighted, quadratic-weighted)
  - Bootstrap 95 % CI on each kappa
  - Raw agreement percentage
  - Off-by-one accuracy (within ±1 on the 0-3 scale)
  - 4×4 confusion matrix (LLM rows × human columns)
  - Pearson correlation (as a sanity check on the kappa)
  - Per-category breakdown (kappa + n)
  - Per-LLM-score breakdown (where does the LLM most disagree?)

Output: data/evaluation/agreement_analysis.json

Why three kappas:
    - Unweighted treats every disagreement equally — mismatches the 0-3 ordinal.
    - Linear-weighted: cost of disagreement scales linearly with distance.
    - Quadratic-weighted: cost scales with squared distance — the conventional
      choice for ordinal scales (e.g., COCO captioning, medical IRR studies).

Interpretation guide (Landis & Koch, 1977):
    kappa < 0.2  : poor
    0.2 - 0.4    : fair
    0.4 - 0.6    : moderate
    0.6 - 0.8    : substantial
    0.8 - 1.0    : almost perfect

Usage:
    python scripts/agreement_analysis.py
    python scripts/agreement_analysis.py --input data/evaluation/full_labeled_dataset_human.jsonl
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/evaluation/full_labeled_dataset_human.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/agreement_analysis.json")


def load_pairs(path: Path) -> list[dict]:
    """Load (LLM, human) labeled pairs, dropping any LLM=-1 errors."""
    pairs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["relevance_llm"] < 0 or rec["relevance_human"] < 0:
                continue
            pairs.append(rec)
    return pairs


def kappa_with_ci(
    llm: np.ndarray,
    human: np.ndarray,
    weights: str | None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute Cohen's kappa with a non-parametric bootstrap 95 % CI.

    Args:
        llm: 1D array of LLM scores.
        human: 1D array of human scores (same length as llm).
        weights: None | "linear" | "quadratic" — passed to sklearn.
        n_boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        dict with {kappa, ci_low, ci_high}. CIs collapse to the point
        estimate when n_boot == 0 or sample size < 5.
    """
    point = float(cohen_kappa_score(llm, human, weights=weights))
    n = len(llm)
    if n < 5 or n_boot <= 0:
        return {"kappa": point, "ci_low": point, "ci_high": point, "n": n}

    rng = np.random.RandomState(seed)
    boot_kappas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            k = cohen_kappa_score(llm[idx], human[idx], weights=weights)
        except ValueError:
            # All same class in the bootstrap sample → skip
            continue
        if not np.isnan(k):
            boot_kappas.append(k)
    if len(boot_kappas) < 10:
        return {"kappa": point, "ci_low": point, "ci_high": point, "n": n}
    low, high = np.percentile(boot_kappas, [2.5, 97.5])
    return {"kappa": point, "ci_low": float(low), "ci_high": float(high), "n": n}


def per_category_kappa(pairs: list[dict]) -> dict[str, dict]:
    """Compute quadratic-weighted kappa within each category."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_cat[p.get("category") or "uncategorised"].append(p)

    out = {}
    for cat, recs in by_cat.items():
        if len(recs) < 5:
            out[cat] = {"n": len(recs), "kappa_quadratic": None, "note": "n<5 — skipped"}
            continue
        llm = np.array([r["relevance_llm"] for r in recs])
        human = np.array([r["relevance_human"] for r in recs])
        out[cat] = kappa_with_ci(llm, human, weights="quadratic", n_boot=500)
    return out


def per_llm_score_breakdown(pairs: list[dict]) -> dict[str, dict]:
    """For each LLM score s, what did the human label?

    Useful for spotting systematic LLM biases — e.g., if LLM=3 maps to human=2
    in 60 % of cases, the LLM is over-confident in its top tier.
    """
    out: dict[str, dict] = {}
    for s in [0, 1, 2, 3]:
        recs = [p for p in pairs if p["relevance_llm"] == s]
        if not recs:
            out[str(s)] = {"n": 0}
            continue
        human_dist = Counter(p["relevance_human"] for p in recs)
        out[str(s)] = {
            "n": len(recs),
            "human_dist": {str(k): v for k, v in sorted(human_dist.items())},
            "mean_human": float(np.mean([p["relevance_human"] for p in recs])),
            "exact_agree_pct": round(human_dist.get(s, 0) / len(recs) * 100, 1),
        }
    return out


def disagreement_examples(pairs: list[dict], top_n: int = 10) -> list[dict]:
    """Return the largest |LLM - human| disagreements for inspection."""
    sorted_pairs = sorted(
        pairs,
        key=lambda p: abs(p["relevance_llm"] - p["relevance_human"]),
        reverse=True,
    )
    return [
        {
            "query": p["query"],
            "nct_id": p["nct_id"],
            "trial_title": p.get("trial_title", "")[:120],
            "category": p.get("category"),
            "llm": p["relevance_llm"],
            "human": p["relevance_human"],
            "delta": abs(p["relevance_llm"] - p["relevance_human"]),
            "llm_reason": p.get("llm_reason", "")[:200],
        }
        for p in sorted_pairs[:top_n]
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="Compute LLM/human agreement metrics")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to full_labeled_dataset_human.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Path to write agreement_analysis.json")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap replicates for CI (0 to skip)")
    return parser.parse_args()


def main() -> None:
    """Compute kappa + supporting agreement metrics, write JSON, print summary."""
    args = parse_args()
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run scripts/manual_label.py first to produce human labels.")
        return

    pairs = load_pairs(args.input)
    if len(pairs) < 5:
        print(f"ERROR: Only {len(pairs)} valid pairs in {args.input} — need at least 5 for kappa.")
        return

    llm = np.array([p["relevance_llm"] for p in pairs])
    human = np.array([p["relevance_human"] for p in pairs])

    # ── Core kappa metrics ────────────────────────────────────────────────────
    k_unweighted = kappa_with_ci(llm, human, weights=None, n_boot=args.n_boot)
    k_linear = kappa_with_ci(llm, human, weights="linear", n_boot=args.n_boot)
    k_quadratic = kappa_with_ci(llm, human, weights="quadratic", n_boot=args.n_boot)

    # ── Raw agreement & off-by-one ────────────────────────────────────────────
    exact_agree = float(np.mean(llm == human))
    off_by_one = float(np.mean(np.abs(llm - human) <= 1))

    # ── Pearson correlation ───────────────────────────────────────────────────
    pearson = float(np.corrcoef(llm, human)[0, 1]) if llm.std() > 0 and human.std() > 0 else 0.0

    # ── Confusion matrix (rows = LLM, cols = human) ───────────────────────────
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(llm, human, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist(),
        "row_label": "LLM score",
        "col_label": "Human score",
    }

    # ── Per-category & per-LLM-score breakdowns ──────────────────────────────
    by_cat = per_category_kappa(pairs)
    by_llm_score = per_llm_score_breakdown(pairs)
    biggest_disagreements = disagreement_examples(pairs, top_n=10)

    report = {
        "n_pairs": len(pairs),
        "exact_agreement": round(exact_agree, 4),
        "off_by_one_accuracy": round(off_by_one, 4),
        "pearson_r": round(pearson, 4),
        "kappa_unweighted": k_unweighted,
        "kappa_linear_weighted": k_linear,
        "kappa_quadratic_weighted": k_quadratic,
        "confusion_matrix": cm_dict,
        "per_category": by_cat,
        "per_llm_score": by_llm_score,
        "biggest_disagreements": biggest_disagreements,
        "interpretation_landis_koch": {
            "<0.2": "poor",
            "0.2-0.4": "fair",
            "0.4-0.6": "moderate",
            "0.6-0.8": "substantial",
            "0.8-1.0": "almost perfect",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # ── Pretty-print summary ──────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  LLM / HUMAN AGREEMENT ANALYSIS")
    print("=" * 68)
    print(f"  Input              : {args.input}")
    print(f"  N pairs            : {len(pairs)}")
    print(f"  Exact agreement    : {exact_agree:.1%}")
    print(f"  Off-by-one (±1)    : {off_by_one:.1%}")
    print(f"  Pearson r          : {pearson:.3f}")
    print()
    print(f"  {'Metric':<28} {'kappa':>7}    {'95% CI':>16}")
    print(f"  {'-'*28} {'-'*7}    {'-'*16}")
    for name, k in [
        ("Cohen kappa (unweighted)", k_unweighted),
        ("Cohen kappa (linear)", k_linear),
        ("Cohen kappa (quadratic)", k_quadratic),
    ]:
        print(f"  {name:<28} {k['kappa']:>7.3f}   "
              f"[{k['ci_low']:>5.3f}, {k['ci_high']:>5.3f}]")
    print()
    print("  Confusion matrix (rows = LLM, cols = human):")
    print(f"        {'h=0':>6} {'h=1':>6} {'h=2':>6} {'h=3':>6}")
    for i, row in enumerate(cm):
        print(f"  l={i}  " + " ".join(f"{v:>6d}" for v in row))
    print()
    print("  Per-category quadratic kappa:")
    for cat, k in by_cat.items():
        if k.get("kappa") is None:
            print(f"    {cat:<14} n={k['n']:>3}  (skipped — n<5)")
        else:
            print(f"    {cat:<14} n={k['n']:>3}  k={k['kappa']:>6.3f}  "
                  f"[{k['ci_low']:>5.3f}, {k['ci_high']:>5.3f}]")
    print()
    print("  Where does the LLM disagree most? (LLM score → human distribution)")
    for s, info in by_llm_score.items():
        if info["n"] == 0:
            continue
        print(f"    LLM={s} (n={info['n']:>3}):  "
              f"human dist {info['human_dist']}  "
              f"exact_agree={info['exact_agree_pct']}%")
    print()
    print(f"  Saved full report: {args.output}")
    print("=" * 68)


if __name__ == "__main__":
    main()
