"""Per-query failure-mode attribution on the 15 complex held-out queries.

For each query, traces every labeled-relevant trial (grade ≥ 2) through
the pipeline and identifies where it dropped:

  retrieval_bound   — never surfaced in hybrid top-50 (the slice rerank sees)
  rerank_bound      — in hybrid top-50 but not in CE+LGB top-10
  eligibility_bound — in CE+LGB top-10 but the hard filter dropped it
  in_top_10         — survived everything (no failure)

Also audits each query's QueryParserAgent output (condition / age / sex /
biomarkers / prior_treatments) so we can flag parser-bound failures
indirectly — if eligibility_bound dominates AND the parser is missing
load-bearing fields, the real bottleneck is upstream.

Inputs:
  - data/evaluation/full_labeled_dataset.jsonl              (Q413..Q417)
  - data/evaluation/full_labeled_dataset_expansion_v2.jsonl (Q600..Q609)

Output:
  - data/evaluation/complex_failure_attribution.json
  - Console summary table (per-stage drop counts, dominant failure mode)

No Haiku spend, no model retraining. Reuses the existing labels.

Usage:
    docker start es
    OMP_NUM_THREADS=1 PYTHONPATH=src \\
      /opt/miniconda3/envs/trialmine/bin/python \\
      scripts/diagnose_complex_failures.py 2>&1 | tee /tmp/diagnose.log
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
SOURCE_LABEL_FILES = (
    EVAL_DIR / "full_labeled_dataset.jsonl",
    EVAL_DIR / "full_labeled_dataset_expansion_v2.jsonl",
)
OUTPUT = EVAL_DIR / "complex_failure_attribution.json"

CATEGORY = "complex"
RELEVANT_GRADE_THRESHOLD = 2
RETRIEVAL_RECALL_K = 200
RERANK_TOP_K = 50
FINAL_TOP_K = 10
EXPECTED_QUERY_COUNT = 15


def load_complex_queries_with_labels() -> dict[int, dict[str, Any]]:
    """Group labels by qid for complex queries only.

    Returns:
        Dict ``{qid: {'qid', 'query', 'candidates': [(nct, grade), ...]}}``.
        Dedupes (qid, nct) — keeps the first occurrence's grade.
    """
    out: dict[int, dict[str, Any]] = {}
    seen_pairs: set[tuple[int, str]] = set()
    for src in SOURCE_LABEL_FILES:
        if not src.exists():
            raise FileNotFoundError(f"Source label file missing: {src}")
        with open(src) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("category") != CATEGORY:
                    continue
                qid = int(r["query_id"])
                nct = str(r["nct_id"])
                if (qid, nct) in seen_pairs:
                    continue
                seen_pairs.add((qid, nct))
                rec = out.setdefault(
                    qid,
                    {"qid": qid, "query": r["query"], "candidates": []},
                )
                rec["candidates"].append((nct, int(r["relevance"])))
    return out


def attribute_failure(
    nct: str,
    hybrid_top50: set[str],
    ce_lgb_top10: set[str],
    post_filter_top10: set[str],
) -> str:
    """Pre-registered attribution rules — return the failure stage label.

    Priority: post-filter (success) → eligibility (dropped from top-10) →
    rerank (was in hybrid top-50 but rerank pushed below top-10) →
    retrieval (never made it to rerank).
    """
    if nct in post_filter_top10:
        return "in_top_10"
    if nct in ce_lgb_top10:
        return "eligibility_bound"
    if nct in hybrid_top50:
        return "rerank_bound"
    return "retrieval_bound"


def run_pipeline_for_query(
    query: str,
    profile,
    hybrid,
    reranker,
    blender,
    check_trial_eligibility_fn,
    apply_hard_filter_fn,
) -> dict[str, Any]:
    """Run the full pipeline and capture intermediate rank sets.

    Returns:
        Dict with ``hybrid_ranks`` (top-200 → {nct: rank}),
        ``ce_lgb_ranks`` (top-10 → {nct: rank}),
        ``post_filter_ranks`` (final top-10 → {nct: rank}),
        ``n_dropped``, ``safety_net``.
    """
    # 1. Hybrid retrieval — recall-aware top-200
    hybrid_hits = hybrid.search(query, top_k=RETRIEVAL_RECALL_K)
    hybrid_ranks = {h["nct_id"]: i for i, h in enumerate(hybrid_hits)}

    # 2. Full pipeline: CE + LGB top-10 (no eligibility filter yet — orchestrator
    # applies it as a separate step, mirroring eval_parser_umls.py's pattern).
    ce_lgb_results, _ = hybrid.full_pipeline(
        query=query,
        reranker=reranker,
        blender=blender,
        top_k=FINAL_TOP_K,
        rerank_top_k=RERANK_TOP_K,
    )
    ce_lgb_ranks = {r["nct_id"]: i for i, r in enumerate(ce_lgb_results)}

    # 3. Run eligibility check on the top-10, apply hard filter
    prior_treatments_str: str | None = None
    if getattr(profile, "prior_treatments", None):
        prior_treatments_str = ", ".join(profile.prior_treatments)

    elig_list: list[dict] = []
    for trial in ce_lgb_results:
        try:
            elig_json = check_trial_eligibility_fn.invoke(
                {
                    "nct_id": trial["nct_id"],
                    "patient_age": (
                        float(profile.age) if getattr(profile, "age", None) is not None else None
                    ),
                    "patient_sex": getattr(profile, "sex", None),
                    "patient_condition": getattr(profile, "condition", None),
                    "prior_treatments": prior_treatments_str,
                }
            )
            elig_list.append(json.loads(elig_json))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Eligibility check failed for %s: %s", trial.get("nct_id"), exc)
            elig_list.append({"verdict": "Unknown", "error": str(exc)})

    filtered, _filtered_elig, n_dropped, safety_net = apply_hard_filter_fn(
        ce_lgb_results, elig_list
    )
    post_filter_ranks = {r["nct_id"]: i for i, r in enumerate(filtered)}

    return {
        "hybrid_ranks": hybrid_ranks,
        "ce_lgb_ranks": ce_lgb_ranks,
        "post_filter_ranks": post_filter_ranks,
        "n_dropped": n_dropped,
        "safety_net": safety_net,
    }


def main() -> None:
    queries = load_complex_queries_with_labels()
    logger.info("Loaded %d complex queries", len(queries))
    assert len(queries) == EXPECTED_QUERY_COUNT, (
        f"expected {EXPECTED_QUERY_COUNT} complex queries, got {len(queries)}"
    )

    # Heavy imports deferred
    from TrialMine.agents.query_parser import QueryParserAgent
    from TrialMine.agents.tools import check_trial_eligibility
    from TrialMine.features.eligibility_filter import apply_hard_filter
    from TrialMine.models.cross_encoder import CrossEncoderReranker
    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.models.ranker import RankingBlender
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.hybrid import HybridRetriever
    from TrialMine.retrieval.semantic import FAISSIndex

    logger.info("Loading pipeline components...")
    es = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name="models/embeddings/fine-tuned-v2")
    faiss = FAISSIndex()
    faiss.load("data/faiss_finetuned_v2.index", "data/faiss_finetuned_v2.json")
    hybrid = HybridRetriever(bm25=es, semantic=faiss, embedder=embedder)
    reranker = CrossEncoderReranker(model_name="models/cross-encoder/fine-tuned-v2")
    blender = RankingBlender(model_path="models/ranker/v3-regularized/model.lgb")
    parser = QueryParserAgent()
    logger.info("Pipeline ready. Running %d queries...", len(queries))

    per_query_results: list[dict[str, Any]] = []
    for qid in sorted(queries):
        q = queries[qid]
        query_text = q["query"]
        candidates = q["candidates"]
        relevant_ncts = [nct for nct, grade in candidates if grade >= RELEVANT_GRADE_THRESHOLD]

        if not relevant_ncts:
            logger.warning("Q%d has 0 relevant trials (grade ≥ %d); skipping attribution",
                           qid, RELEVANT_GRADE_THRESHOLD)
            per_query_results.append({
                "qid": qid, "query": query_text,
                "n_relevant": 0, "per_trial": [], "parser_profile": None,
            })
            continue

        t0 = time.perf_counter()
        try:
            profile = parser.parse(query_text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Parser failed for Q%d: %s", qid, exc)
            continue

        try:
            stages = run_pipeline_for_query(
                query=query_text,
                profile=profile,
                hybrid=hybrid,
                reranker=reranker,
                blender=blender,
                check_trial_eligibility_fn=check_trial_eligibility,
                apply_hard_filter_fn=apply_hard_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for Q%d: %s", qid, exc)
            continue

        hybrid_top50 = {nct for nct, rank in stages["hybrid_ranks"].items() if rank < RERANK_TOP_K}
        ce_lgb_top10 = set(stages["ce_lgb_ranks"].keys())
        post_filter_top10 = set(stages["post_filter_ranks"].keys())

        per_trial = []
        for nct in relevant_ncts:
            grade = next(g for n, g in candidates if n == nct)
            cat = attribute_failure(nct, hybrid_top50, ce_lgb_top10, post_filter_top10)
            per_trial.append({
                "nct_id": nct,
                "grade": grade,
                "hybrid_rank": stages["hybrid_ranks"].get(nct, -1),
                "ce_lgb_rank": stages["ce_lgb_ranks"].get(nct, -1),
                "post_filter_rank": stages["post_filter_ranks"].get(nct, -1),
                "failure": cat,
            })

        parser_audit = {
            "condition": getattr(profile, "condition", None),
            "age": getattr(profile, "age", None),
            "sex": getattr(profile, "sex", None),
            "biomarkers": list(getattr(profile, "biomarkers", None) or []),
            "prior_treatments": list(getattr(profile, "prior_treatments", None) or []),
        }

        dt = time.perf_counter() - t0
        local_counts = Counter(t["failure"] for t in per_trial)
        logger.info(
            "Q%d done in %.1fs — %d relevant, attribution: %s",
            qid, dt, len(relevant_ncts), dict(local_counts),
        )

        per_query_results.append({
            "qid": qid,
            "query": query_text,
            "n_relevant": len(relevant_ncts),
            "n_dropped_by_filter": stages["n_dropped"],
            "safety_net_triggered": stages["safety_net"],
            "parser_profile": parser_audit,
            "per_trial": per_trial,
        })

    # Aggregate failure distribution across all queries × all relevant trials
    failure_counts: Counter = Counter()
    for r in per_query_results:
        for t in r.get("per_trial", []):
            failure_counts[t["failure"]] += 1

    output = {
        "metadata": {
            "n_queries": len(queries),
            "n_relevant_total": sum(r["n_relevant"] for r in per_query_results),
            "relevant_grade_threshold": RELEVANT_GRADE_THRESHOLD,
            "retrieval_recall_k": RETRIEVAL_RECALL_K,
            "rerank_top_k": RERANK_TOP_K,
            "final_top_k": FINAL_TOP_K,
            "category": CATEGORY,
        },
        "failure_distribution": dict(failure_counts),
        "per_query": per_query_results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote attribution: %s", OUTPUT)

    # Console summary table
    print()
    print("=" * 80)
    print(
        f"Complex failure-mode attribution — {len(queries)} queries, "
        f"{output['metadata']['n_relevant_total']} relevant trials"
    )
    print("=" * 80)
    print(f"{'Failure mode':<22} {'Count':<8} {'%':<8}")
    print("-" * 80)
    total = sum(failure_counts.values()) or 1
    # Order: in_top_10 first (success), then failures
    order = ["in_top_10", "retrieval_bound", "rerank_bound", "eligibility_bound"]
    for fail in order:
        count = failure_counts.get(fail, 0)
        pct = 100.0 * count / total
        print(f"{fail:<22} {count:<8} {pct:>5.1f}%")
    print("=" * 80)

    # Per-query breakdown
    print(f"\n{'QID':<6} {'#rel':<5} {'in10':<5} {'ret':<4} {'rerank':<7} {'elig':<5} Query")
    print("-" * 80)
    for r in per_query_results:
        local = Counter(t["failure"] for t in r.get("per_trial", []))
        print(
            f"{r['qid']:<6} {r['n_relevant']:<5} "
            f"{local.get('in_top_10',0):<5} "
            f"{local.get('retrieval_bound',0):<4} "
            f"{local.get('rerank_bound',0):<7} "
            f"{local.get('eligibility_bound',0):<5} "
            f"{r['query'][:48]}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
