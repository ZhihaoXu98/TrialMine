"""Phase B1 of docs/fix_parser_umls.md — focused UMLS toggle re-eval on the
30 complex+vague held-out queries.

For each query, runs the full production pipeline (retrieval + CE + LightGBM +
eligibility check + hard-filter) TWICE — once with
``DegradationConfig.umls_drug_class_matching_enabled = False`` (current
production behavior) and once with it ``True`` (the Phase 13A path). The
retrieval / CE / blender stage is cached across the two passes; only the
eligibility-check + filter phase is rerun, since the toggle only affects
``_match_required_treatments`` and ``_match_excluded_treatments`` in
``TrialMine.agents.tools``.

Captures per query, per toggle state:
  - Top-K filtered NCT IDs (the orchestrator output users would see)
  - Verdict counts for required_prior_treatments + excluded_prior_treatments
    on the SAME pre-filter pool (so "newly Unmet" is well-defined)
  - Number of trials dropped by the hard filter (today this fires only on
    age / sex / excluded_prior_treatments — UMLS-aware
    excluded_prior_treatments is the path that can newly drop trials when
    the toggle is on)
  - Wall-clock latency of the eligibility phase

Re-labels any NEW top-K NCT (not already in the two source label files or in
prior output) via Claude Haiku with the same prompt as
``build_full_eval_dataset.py`` (the canonical 0-3 grader). Resumable: a
progress checkpoint is written after every 5 queries; if the script is
killed, re-running with ``--resume`` skips queries already in the
checkpoint and pairs already in the output JSONL.

Outputs:
  - data/evaluation/full_labeled_complex_vague_umls.jsonl
    NEW (qid, nct_id) pairs labelled by Haiku — appended.
  - data/evaluation/umls_eval_metrics.json
    Per-category NDCG@5/10 + MRR with bootstrap 95% CIs for OFF and ON.
  - data/evaluation/umls_eval_progress.json
    Resume checkpoint with per-query top-K NCTs, verdict counts, drops, latency.

Reuses ``bootstrap_ci`` + ``per_query_metrics`` from
``scripts/c2_eval_pure_ce.py`` and ``LABELING_PROMPT`` + ``label_pair`` +
``load_existing_labels`` from ``scripts/build_full_eval_dataset.py`` —
no reimplementation per docs/fix_parser_umls.md B1.

Usage:
    docker start es                                 # Elasticsearch up
    set -a; source .env; set +a                     # ANTHROPIC_API_KEY
    OMP_NUM_THREADS=1 PYTHONPATH=src \
      python scripts/eval_parser_umls.py 2>&1 | tee /tmp/umls_eval.log

    OMP_NUM_THREADS=1 PYTHONPATH=src \
      python scripts/eval_parser_umls.py --resume    # resume after kill
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# Make src/ and scripts/ both importable. Adding scripts/ lets us reuse
# helpers from build_full_eval_dataset.py + c2_eval_pure_ce.py without
# duplicating their code (per docs/fix_parser_umls.md B1).
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

# Re-exported helpers. Importing these modules also triggers their
# top-level heavy imports (sentence_transformers, FAISS, etc.) — that's
# fine because this script needs them anyway for the pipeline.
from build_full_eval_dataset import (  # noqa: E402
    LABELING_PROMPT,  # noqa: F401  exposed for traceability
    label_pair,
    load_existing_labels,
)
from c2_eval_pure_ce import bootstrap_ci, per_query_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
SOURCE_LABEL_FILES: tuple[Path, ...] = (
    EVAL_DIR / "full_labeled_dataset.jsonl",
    EVAL_DIR / "full_labeled_dataset_expansion_v2.jsonl",
)
OUTPUT_LABELS = EVAL_DIR / "full_labeled_complex_vague_umls.jsonl"
OUTPUT_METRICS = EVAL_DIR / "umls_eval_metrics.json"
PROGRESS_CHECKPOINT = EVAL_DIR / "umls_eval_progress.json"

# Path defaults — env-overridable so the script can be pointed at v1
# components without code edits. Mirrors build_full_eval_dataset.py:200-204.
import os  # noqa: E402  (local to const definitions)

EMBEDDER_MODEL = os.environ.get("TRIALMINE_EMBEDDER", "models/embeddings/fine-tuned-v2")
FAISS_INDEX = os.environ.get("TRIALMINE_FAISS_INDEX", "data/faiss_finetuned_v2.index")
FAISS_MAPPING = os.environ.get("TRIALMINE_FAISS_MAPPING", "data/faiss_finetuned_v2.json")
CE_MODEL = os.environ.get("TRIALMINE_CROSS_ENCODER", "models/cross-encoder/fine-tuned-v2")
RANKER_MODEL = os.environ.get("TRIALMINE_RANKER", "models/ranker/v3-regularized/model.lgb")

TOP_K = 10
RERANK_TOP_K = 50
CATEGORIES_OF_INTEREST = ("complex", "vague")
EXPECTED_QUERY_COUNT = 30
CHECKPOINT_EVERY_N = 5  # save progress after every N queries


# --------------------------------------------------------------------------- #
# Toggle context manager                                                       #
# --------------------------------------------------------------------------- #


@contextmanager
def umls_toggle(enabled: bool):
    """Temporarily flip ``umls_drug_class_matching_enabled``; reset on exit.

    Mutates the process-wide DegradationConfig singleton. ``try/finally``
    so a failed pipeline step doesn't leak the toggle state to the next
    query. Lazy import keeps the script importable without TrialMine on
    the path (useful for --list-queries quick checks).
    """
    from TrialMine.config import get_default_degradation

    cfg = get_default_degradation()
    prev = cfg.umls_drug_class_matching_enabled
    cfg.umls_drug_class_matching_enabled = enabled
    try:
        yield
    finally:
        cfg.umls_drug_class_matching_enabled = prev


# --------------------------------------------------------------------------- #
# Query loading                                                                #
# --------------------------------------------------------------------------- #


def load_complex_vague_queries() -> list[tuple[int, str, str]]:
    """Load queries with category in {complex, vague} from the two source files.

    Deduplicates by ``query_id`` (the same qid appears multiple times in
    the labels JSONL — one row per (qid, nct_id) pair). Asserts the final
    count is exactly :data:`EXPECTED_QUERY_COUNT` so we never silently
    eval against fewer queries than the runbook specifies.

    Returns:
        Sorted list of ``(qid, category, query)`` tuples.
    """
    seen: dict[int, tuple[int, str, str]] = {}
    for src in SOURCE_LABEL_FILES:
        if not src.exists():
            raise FileNotFoundError(f"Source label file missing: {src}")
        with open(src) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("category") not in CATEGORIES_OF_INTEREST:
                    continue
                qid = int(rec["query_id"])
                if qid in seen:
                    continue
                seen[qid] = (qid, rec["category"], rec["query"])

    queries = sorted(seen.values(), key=lambda t: t[0])
    n_complex = sum(1 for _, c, _ in queries if c == "complex")
    n_vague = sum(1 for _, c, _ in queries if c == "vague")
    logger.info(
        "Loaded %d queries (%d complex + %d vague) from %d source files",
        len(queries),
        n_complex,
        n_vague,
        len(SOURCE_LABEL_FILES),
    )
    assert len(queries) == EXPECTED_QUERY_COUNT, (
        f"expected {EXPECTED_QUERY_COUNT} queries, got {len(queries)} "
        f"({n_complex} complex + {n_vague} vague). Check that the two "
        f"source files include the 5 complex + 5 vague qids in the 65q "
        f"and the 10 complex + 10 vague qids in the 20q expansion."
    )
    return queries


# --------------------------------------------------------------------------- #
# Progress checkpoint                                                          #
# --------------------------------------------------------------------------- #


def load_progress() -> dict[int, dict[str, Any]]:
    """Load per-query pipeline output from the progress checkpoint."""
    if not PROGRESS_CHECKPOINT.exists():
        return {}
    with open(PROGRESS_CHECKPOINT) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get("per_query", {}).items()}


def save_progress(progress: dict[int, dict[str, Any]]) -> None:
    """Atomically write per-query pipeline output to the progress checkpoint."""
    PROGRESS_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_CHECKPOINT.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(
            {"per_query": {str(k): v for k, v in progress.items()}},
            f,
            indent=2,
        )
    tmp.replace(PROGRESS_CHECKPOINT)


# --------------------------------------------------------------------------- #
# Eligibility verdict accounting                                               #
# --------------------------------------------------------------------------- #


def count_verdicts(elig_list: list[dict]) -> dict[str, dict[str, int]]:
    """Return ``{criterion: {Met|Unmet|Unknown: count}}`` for the two
    treatment criteria.

    These are the only two criteria whose verdict logic flips with the
    ``umls_drug_class_matching_enabled`` toggle. Condition matchers
    (``required_conditions`` / ``excluded_conditions``) use substring
    regardless of the toggle, so their counts would be identical
    OFF vs ON — not worth recording here.
    """
    out: dict[str, dict[str, int]] = {
        "required_prior_treatments": {"Met": 0, "Unmet": 0, "Unknown": 0},
        "excluded_prior_treatments": {"Met": 0, "Unmet": 0, "Unknown": 0},
    }
    for elig in elig_list:
        if not isinstance(elig, dict) or "criteria" not in elig:
            continue
        for criterion in out:
            verdict = elig["criteria"].get(criterion, {}).get("verdict", "Unknown")
            if verdict not in out[criterion]:
                out[criterion][verdict] = 0
            out[criterion][verdict] += 1
    return out


def count_newly_unmet(off_elig: list[dict], on_elig: list[dict]) -> dict[str, int]:
    """For each treatment criterion, count trials where the verdict flipped
    from non-Unmet (OFF) to Unmet (ON).

    Both lists must be aligned by index (same pre-filter trial order).
    """
    out: dict[str, int] = {
        "required_prior_treatments": 0,
        "excluded_prior_treatments": 0,
    }
    for i in range(min(len(off_elig), len(on_elig))):
        off_e = off_elig[i] if isinstance(off_elig[i], dict) else {}
        on_e = on_elig[i] if isinstance(on_elig[i], dict) else {}
        off_c = off_e.get("criteria", {})
        on_c = on_e.get("criteria", {})
        for criterion in out:
            off_v = off_c.get(criterion, {}).get("verdict", "Unknown")
            on_v = on_c.get(criterion, {}).get("verdict", "Unknown")
            if off_v != "Unmet" and on_v == "Unmet":
                out[criterion] += 1
    return out


# --------------------------------------------------------------------------- #
# Pipeline run (eligibility phase only — retrieval cached)                     #
# --------------------------------------------------------------------------- #


def run_eligibility_phase(
    raw_results: list[dict],
    profile,
    check_trial_eligibility_fn,
    apply_hard_filter_fn,
) -> tuple[list[dict], list[dict], int, bool, float]:
    """Run eligibility check + hard filter on a cached retrieval result.

    Returns:
        (filtered, elig_list, n_dropped, safety_net, latency_ms)
        - filtered: post-filter trials, ranked
        - elig_list: per-trial eligibility verdicts on the PRE-filter pool
          (aligned by index with ``raw_results``)
        - n_dropped: number of trials dropped by the hard filter
        - safety_net: True if the filter would have emptied results and
          the unfiltered list is returned instead
        - latency_ms: wall-clock for the eligibility + filter step
    """
    t0 = time.perf_counter()
    elig_list: list[dict] = []
    prior_treatments_str: str | None = None
    if getattr(profile, "prior_treatments", None):
        prior_treatments_str = ", ".join(profile.prior_treatments)
    for trial in raw_results:
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

    filtered, _filtered_elig, n_dropped, safety_net = apply_hard_filter_fn(raw_results, elig_list)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return filtered, elig_list, n_dropped, safety_net, latency_ms


# --------------------------------------------------------------------------- #
# Labelling new pairs                                                          #
# --------------------------------------------------------------------------- #


def fetch_trial_metadata(conn: sqlite3.Connection, nct_ids: list[str]) -> dict[str, dict]:
    """Bulk-fetch (title, conditions, phase, status, eligibility) for label_pair."""
    if not nct_ids:
        return {}
    out: dict[str, dict] = {}
    chunk = 500
    for i in range(0, len(nct_ids), chunk):
        batch = nct_ids[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        cur = conn.execute(
            f"SELECT nct_id, title, conditions, phase, status, eligibility_criteria "
            f"FROM trials WHERE nct_id IN ({placeholders})",
            batch,
        )
        for row in cur.fetchall():
            nct, title, conds, phase, status, elig = row
            out[nct] = {
                "title": title or "",
                "conditions": conds or "",
                "phase": phase,
                "status": status,
                "eligibility_criteria": elig or "",
            }
    return out


def label_missing_pairs(
    queries: list[tuple[int, str, str]],
    progress: dict[int, dict[str, Any]],
    existing_labels: set[tuple[int, str]],
    db_conn: sqlite3.Connection,
    client,
    output_file: Path,
    rate_limit_s: float,
) -> tuple[int, dict[int, int]]:
    """Label any (qid, nct) in either toggle's top-K not already labeled.

    Pairs are deduped before the API loop. Records are appended one line
    at a time and flushed so the file is resumable if the script is killed.
    """
    new_pairs: list[tuple[int, str, str, str]] = []  # (qid, category, query, nct)
    for qid, category, query in queries:
        per_q = progress.get(qid)
        if per_q is None:
            continue
        ncts = set(per_q["off"]["top_k_ncts"]) | set(per_q["on"]["top_k_ncts"])
        for nct in ncts:
            if (qid, nct) not in existing_labels:
                new_pairs.append((qid, category, query, nct))

    if not new_pairs:
        logger.info("No new (qid, nct) pairs to label.")
        return 0, {}

    logger.info("Labelling %d new (qid, nct) pairs via Haiku...", len(new_pairs))

    unique_ncts = sorted({nct for _, _, _, nct in new_pairs})
    trial_meta = fetch_trial_metadata(db_conn, unique_ncts)

    score_dist: dict[int, int] = {}
    labeled = 0
    mode = "a" if output_file.exists() else "w"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode) as f:
        for qid, category, query, nct in new_pairs:
            meta = trial_meta.get(nct)
            if meta is None:
                logger.warning("Trial %s not in DB; skipping label for q%d", nct, qid)
                continue
            trial = {
                "nct_id": nct,
                "title": meta["title"],
                "conditions": meta["conditions"],
                "phase": meta["phase"],
                "status": meta["status"],
            }
            score, reason = label_pair(client, query, trial, meta["eligibility_criteria"])
            record = {
                "query_id": qid,
                "query": query,
                "category": category,
                "nct_id": nct,
                "trial_title": trial["title"],
                "trial_conditions": trial["conditions"],
                "trial_phase": trial["phase"],
                "trial_status": trial["status"],
                "relevance": score,
                "reason": reason,
                "labeler": "claude-haiku-4-5",
                "source": "B1_umls_eval",
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            existing_labels.add((qid, nct))
            score_dist[score] = score_dist.get(score, 0) + 1
            labeled += 1
            time.sleep(rate_limit_s)

    logger.info("Labelled %d new pairs (output: %s)", labeled, output_file)
    return labeled, score_dist


# --------------------------------------------------------------------------- #
# Metric computation                                                           #
# --------------------------------------------------------------------------- #


def load_full_label_pool(*paths: Path) -> dict[int, dict[str, int]]:
    """Build ``{qid: {nct_id: relevance}}`` from all label files.

    Drops rows with ``relevance < 0`` (Haiku parse / API errors).
    """
    pool: dict[int, dict[str, int]] = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                grade = int(rec["relevance"])
                if grade < 0:
                    continue
                qid = int(rec["query_id"])
                nct = str(rec["nct_id"])
                pool.setdefault(qid, {})[nct] = grade
    return pool


def compute_per_query_metrics(
    queries: list[tuple[int, str, str]],
    progress: dict[int, dict[str, Any]],
    label_pool: dict[int, dict[str, int]],
    toggle_key: str,
) -> dict[str, list[dict]]:
    """For one toggle state, compute NDCG@5/10 + MRR per query, grouped by category.

    Unlabelled NCTs in the top-K are scored 0 (irrelevant) — by design we
    re-label the entire top-K, so this should never fire after the labelling
    pass; if it does, log a warning so the operator notices.
    """
    by_cat: dict[str, list[dict]] = {"complex": [], "vague": []}
    aggregate: list[dict] = []
    for qid, category, _query in queries:
        per_q = progress.get(qid)
        if per_q is None:
            continue
        top_ncts: list[str] = per_q[toggle_key]["top_k_ncts"]
        grades = label_pool.get(qid, {})
        unlabelled = [nct for nct in top_ncts if nct not in grades]
        if unlabelled:
            logger.warning(
                "qid=%d toggle=%s has %d unlabelled top-K NCTs (scoring as 0): %s",
                qid,
                toggle_key,
                len(unlabelled),
                unlabelled[:3],
            )
        candidates = [(nct, grades.get(nct, 0)) for nct in top_ncts]
        # Rank-based score: position 0 highest, position k-1 lowest.
        scores = {nct: 1.0 / (i + 1) for i, nct in enumerate(top_ncts)}
        m = per_query_metrics(candidates, scores)
        row = {"qid": qid, "category": category, **m}
        by_cat[category].append(row)
        aggregate.append(row)
    by_cat["aggregate"] = aggregate
    return by_cat


def summarize(per_cat: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """For each category + ``aggregate``, mean + 95% bootstrap CI per metric."""
    out: dict[str, dict[str, dict]] = {}
    for cat, rows in per_cat.items():
        out[cat] = {"n": len(rows)}
        for metric in ("ndcg5", "ndcg10", "mrr"):
            values = [r[metric] for r in rows]
            mean, lo, hi = bootstrap_ci(values)
            out[cat][metric] = {"mean": mean, "lo": lo, "hi": hi}
    return out


# --------------------------------------------------------------------------- #
# Console summary                                                              #
# --------------------------------------------------------------------------- #


def print_summary(
    off_summary: dict[str, dict[str, dict]],
    on_summary: dict[str, dict[str, dict]],
    progress: dict[int, dict[str, Any]],
    queries: list[tuple[int, str, str]],
) -> None:
    """Render the per-category OFF/ON/Delta table the runbook asked for."""
    print()
    print("=" * 88)
    print("Phase B1 UMLS toggle re-eval — 30 complex+vague held-out queries")
    print("=" * 88)
    print(
        f"{'Category':<12} {'OFF NDCG@5':<22} {'ON NDCG@5':<22} {'Delta':<10} "
        f"{'OFF #drops':<12} {'ON #drops':<12}"
    )
    print("-" * 88)
    for cat in ("complex", "vague", "aggregate"):
        if cat not in off_summary:
            continue
        off_m = off_summary[cat]["ndcg5"]
        on_m = on_summary[cat]["ndcg5"]
        delta = on_m["mean"] - off_m["mean"]
        cat_qids = [q for q, c, _ in queries if (cat == "aggregate" or c == cat)]
        drops_off = [progress[q]["off"]["n_dropped"] for q in cat_qids if q in progress]
        drops_on = [progress[q]["on"]["n_dropped"] for q in cat_qids if q in progress]
        avg_drops_off = sum(drops_off) / len(drops_off) if drops_off else 0.0
        avg_drops_on = sum(drops_on) / len(drops_on) if drops_on else 0.0
        off_str = f"{off_m['mean']:.3f} [{off_m['lo']:.3f},{off_m['hi']:.3f}]"
        on_str = f"{on_m['mean']:.3f} [{on_m['lo']:.3f},{on_m['hi']:.3f}]"
        print(
            f"{cat:<12} {off_str:<22} {on_str:<22} {delta:+.3f}    "
            f"{avg_drops_off:<12.2f} {avg_drops_on:<12.2f}"
        )
    print("=" * 88)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="Cap queries (0=all 30)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip qids already in the progress checkpoint and (qid,nct) pairs already labelled",
    )
    p.add_argument("--db", default="data/trials.db", help="SQLite DB path for trial metadata")
    p.add_argument("--top-k", type=int, default=TOP_K, help="Pipeline top-K (default 10)")
    p.add_argument(
        "--rerank-top-k", type=int, default=RERANK_TOP_K, help="CE rerank top-K (default 50)"
    )
    p.add_argument(
        "--rate-limit-s", type=float, default=0.1, help="Sleep between Haiku label calls"
    )
    p.add_argument(
        "--skip-labelling",
        action="store_true",
        help=(
            "Run the pipeline pass + write progress checkpoint, but skip the "
            "Haiku labelling pass. Useful for a dry-run."
        ),
    )
    p.add_argument(
        "--skip-pipeline",
        action="store_true",
        help=(
            "Skip the pipeline pass (assumes progress checkpoint already "
            "exists). Useful for re-labelling only or re-computing metrics "
            "after fixing labels."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    queries = load_complex_vague_queries()
    if args.limit > 0:
        queries = queries[: args.limit]
        logger.info("--limit %d → running %d queries", args.limit, len(queries))

    progress = load_progress() if args.resume else {}
    if args.resume and progress:
        logger.info("Resuming from progress checkpoint: %d qids already done", len(progress))

    # ── Pipeline pass ─────────────────────────────────────────────────────────
    if not args.skip_pipeline:
        # Heavy imports deferred — only loaded when actually running the pipeline.
        from TrialMine.agents.query_parser import QueryParserAgent
        from TrialMine.agents.tools import check_trial_eligibility
        from TrialMine.features.eligibility_filter import apply_hard_filter
        from TrialMine.models.cross_encoder import CrossEncoderReranker
        from TrialMine.models.embeddings import TrialEmbedder
        from TrialMine.models.ranker import RankingBlender
        from TrialMine.retrieval.bm25 import ElasticsearchIndex
        from TrialMine.retrieval.hybrid import HybridRetriever
        from TrialMine.retrieval.semantic import FAISSIndex

        logger.info("Loading retrieval components...")
        es_index = ElasticsearchIndex()
        embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
        faiss_index = FAISSIndex()
        faiss_index.load(FAISS_INDEX, FAISS_MAPPING)
        hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

        if not Path(CE_MODEL).exists():
            logger.error("Cross-encoder not found: %s", CE_MODEL)
            sys.exit(1)
        reranker = CrossEncoderReranker(model_name=CE_MODEL)

        blender = None
        if Path(RANKER_MODEL).exists():
            blender = RankingBlender(model_path=RANKER_MODEL)
        else:
            logger.warning("LightGBM ranker missing at %s — CE-blended fallback", RANKER_MODEL)

        query_parser = QueryParserAgent()
        logger.info("Pipeline ready. Running %d queries × 2 toggle states...", len(queries))

        for i, (qid, category, query) in enumerate(queries):
            if args.resume and qid in progress:
                logger.info("  Q%d [%s] cached — skipping", qid, category)
                continue

            # Parse query → PatientProfile ONCE (UMLS toggle doesn't affect parsing).
            try:
                profile = query_parser.parse(query)
            except Exception as exc:  # noqa: BLE001
                logger.error("QueryParser failed for q%d: %s", qid, exc)
                continue

            # Retrieval / CE / blender ONCE (toggle doesn't affect this either).
            try:
                raw_results, _timings = hybrid.full_pipeline(
                    query=query,
                    reranker=reranker,
                    blender=blender,
                    top_k=args.top_k,
                    rerank_top_k=args.rerank_top_k,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Pipeline failed for q%d: %s", qid, exc)
                continue

            per_q: dict[str, Any] = {
                "query": query,
                "category": category,
                "n_retrieved": len(raw_results),
            }
            for toggle_state, key in ((False, "off"), (True, "on")):
                with umls_toggle(toggle_state):
                    filtered, elig_list, n_dropped, safety_net, elig_ms = run_eligibility_phase(
                        raw_results=raw_results,
                        profile=profile,
                        check_trial_eligibility_fn=check_trial_eligibility,
                        apply_hard_filter_fn=apply_hard_filter,
                    )
                per_q[key] = {
                    "top_k_ncts": [r["nct_id"] for r in filtered[: args.top_k]],
                    "verdict_counts": count_verdicts(elig_list),
                    "n_dropped": n_dropped,
                    "safety_net": safety_net,
                    "elig_phase_latency_ms": round(elig_ms, 1),
                }
                # Stash the raw verdicts on the off side too, so we can diff later.
                if key == "off":
                    per_q["_off_elig_for_diff"] = elig_list
                elif key == "on":
                    per_q["newly_unmet"] = count_newly_unmet(
                        per_q.pop("_off_elig_for_diff"), elig_list
                    )

            progress[qid] = per_q
            logger.info(
                "  Q%d [%s] off_drops=%d on_drops=%d newly_unmet_req=%d newly_unmet_excl=%d",
                qid,
                category,
                per_q["off"]["n_dropped"],
                per_q["on"]["n_dropped"],
                per_q["newly_unmet"]["required_prior_treatments"],
                per_q["newly_unmet"]["excluded_prior_treatments"],
            )

            if (i + 1) % CHECKPOINT_EVERY_N == 0:
                save_progress(progress)
                logger.info("  → checkpoint saved (%d / %d queries done)", i + 1, len(queries))

        save_progress(progress)
        logger.info("Pipeline pass complete (%d queries in checkpoint)", len(progress))

    # ── Labelling pass ────────────────────────────────────────────────────────
    if args.skip_labelling:
        logger.info("--skip-labelling set — exiting before Haiku pass")
        return

    if not progress:
        progress = load_progress()
        if not progress:
            logger.error(
                "No progress checkpoint found and --skip-pipeline was set; "
                "nothing to label"
            )
            sys.exit(1)

    import anthropic  # noqa: E402  deferred

    client = anthropic.Anthropic()
    existing_labels: set[tuple[int, str]] = set()
    for src in SOURCE_LABEL_FILES:
        existing_labels |= load_existing_labels(src)
    existing_labels |= load_existing_labels(OUTPUT_LABELS)
    logger.info("Existing labels in scope: %d (qid, nct) pairs", len(existing_labels))

    db_conn = sqlite3.connect(args.db)
    try:
        labeled, score_dist = label_missing_pairs(
            queries=queries,
            progress=progress,
            existing_labels=existing_labels,
            db_conn=db_conn,
            client=client,
            output_file=OUTPUT_LABELS,
            rate_limit_s=args.rate_limit_s,
        )
    finally:
        db_conn.close()

    if labeled:
        logger.info("New label score distribution: %s", score_dist)

    # ── Metric computation ────────────────────────────────────────────────────
    label_pool = load_full_label_pool(*SOURCE_LABEL_FILES, OUTPUT_LABELS)
    off_per_q = compute_per_query_metrics(queries, progress, label_pool, toggle_key="off")
    on_per_q = compute_per_query_metrics(queries, progress, label_pool, toggle_key="on")
    off_summary = summarize(off_per_q)
    on_summary = summarize(on_per_q)

    # Per-query newly-unmet counts for the JSON output
    newly_unmet_per_q = {
        str(qid): progress[qid].get(
            "newly_unmet", {"required_prior_treatments": 0, "excluded_prior_treatments": 0}
        )
        for qid in progress
    }

    metrics_out = {
        "metadata": {
            "n_queries": len(queries),
            "n_complex": sum(1 for _, c, _ in queries if c == "complex"),
            "n_vague": sum(1 for _, c, _ in queries if c == "vague"),
            "top_k": args.top_k,
            "rerank_top_k": args.rerank_top_k,
            "source_label_files": [str(p) for p in SOURCE_LABEL_FILES],
            "new_labels_file": str(OUTPUT_LABELS),
            "new_labels_added": labeled,
        },
        "umls_off": {
            "summary": off_summary,
            "per_query": off_per_q,
        },
        "umls_on": {
            "summary": on_summary,
            "per_query": on_per_q,
        },
        "delta": {
            cat: {
                m: on_summary[cat][m]["mean"] - off_summary[cat][m]["mean"]
                for m in ("ndcg5", "ndcg10", "mrr")
            }
            for cat in off_summary
        },
        "newly_unmet_per_query": newly_unmet_per_q,
    }

    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Wrote metrics: %s", OUTPUT_METRICS)

    print_summary(off_summary, on_summary, progress, queries)


if __name__ == "__main__":
    main()
