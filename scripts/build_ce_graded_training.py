"""Build graded (query, positive, negative, margin) training pairs for the CE retrain.

Phase A2 of docs/fix_CE.md.

Reads 4 source Haiku-labelled JSONL files, excludes 85 held-out query_ids
(the 65-query benchmark + the 20-query Phase C4 expansion), deduplicates on
(query_id, nct_id) by first-occurrence-in-file-order, splits queries 80/20
with random.Random(42), and emits ordered margin-ranked pairs whose label
is grade(A) - grade(B).

Trial text matches the CE's production inference path
(`HybridRetriever.full_pipeline` in src/TrialMine/retrieval/hybrid.py:292-299):
title + [SEP] + conditions + [SEP] + brief_summary, truncated to 2048 chars.
This is the same 3-part format used by `TrialEmbedder.prepare_trial_text`
in src/TrialMine/models/embeddings.py:116-146. The runbook's "title + [SEP]
+ brief_summary" spec is a 2-part typo of its own stated intent ("matches
prepare_trial_text"); we follow production behaviour so training-time text
and inference-time text are identical.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from collections.abc import Iterable
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_LABEL_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "evaluation" / "labeled_queries.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "test_labels.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "train_labels_extra.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "test_labels_v2.jsonl",
)
HELD_OUT_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "evaluation" / "full_labeled_dataset.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "full_labeled_dataset_expansion_v2.jsonl",
)
TRIALS_DB = PROJECT_ROOT / "data" / "trials.db"

OUTPUT_TRAIN = PROJECT_ROOT / "data" / "training" / "ce_graded_train.jsonl"
OUTPUT_VAL = PROJECT_ROOT / "data" / "training" / "ce_graded_val.jsonl"
OUTPUT_METADATA = PROJECT_ROOT / "data" / "training" / "ce_graded_metadata.json"

SEED = 42
SPLIT_RATIO = 0.80
MAX_TEXT_CHARS = 2048  # matches the production CE inference truncation
SQLITE_PARAM_CHUNK = 500  # safely under SQLite's default 999-parameter cap


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        level=logging.INFO,
    )


def _validate_inputs() -> None:
    missing = [p for p in (*SOURCE_LABEL_FILES, *HELD_OUT_FILES, TRIALS_DB) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input files missing:\n  " + "\n  ".join(str(p) for p in missing)
        )


def load_held_out_qids(paths: Iterable[Path]) -> set[int]:
    """Union the `query_id` field across the held-out JSONLs."""
    held_out: set[int] = set()
    for path in paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                held_out.add(int(json.loads(line)["query_id"]))
    return held_out


def load_source_rows(
    paths: Iterable[Path],
    held_out: set[int],
) -> tuple[list[dict], int, int]:
    """Concatenate the 4 source files in order, drop held-out qids, dedupe on (qid, nct).

    Returns (rows, n_dropped_held_out, n_deduped). Each row carries the minimal
    fields the downstream pair-builder needs: query_id, query, nct_id, grade.
    """
    seen: set[tuple[int, str]] = set()
    rows: list[dict] = []
    n_dropped = 0
    n_deduped = 0
    for path in paths:
        path_rows = 0
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                src = json.loads(line)
                qid = int(src["query_id"])
                nct = str(src["nct_id"])
                if qid in held_out:
                    n_dropped += 1
                    continue
                key = (qid, nct)
                if key in seen:
                    n_deduped += 1
                    continue
                seen.add(key)
                rows.append(
                    {
                        "query_id": qid,
                        "query": str(src["query"]),
                        "nct_id": nct,
                        "grade": int(src["relevance"]),
                    }
                )
                path_rows += 1
        logger.info("  %s → %d rows (post-filter)", path.name, path_rows)
    return rows, n_dropped, n_deduped


def build_trial_text_cache(nct_ids: set[str], db_path: Path) -> dict[str, str]:
    """One bulk SQL fetch into an in-memory dict — keeps total runtime under ~1 min."""
    if not nct_ids:
        return {}
    nct_list = sorted(nct_ids)
    cache: dict[str, str] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        for i in range(0, len(nct_list), SQLITE_PARAM_CHUNK):
            batch = nct_list[i : i + SQLITE_PARAM_CHUNK]
            placeholders = ",".join("?" * len(batch))
            cur.execute(
                f"SELECT nct_id, title, conditions, brief_summary "
                f"FROM trials WHERE nct_id IN ({placeholders})",
                batch,
            )
            for nct_id, title, conditions, brief_summary in cur.fetchall():
                parts: list[str] = []
                if title:
                    parts.append(title)
                if conditions:
                    parts.append(conditions)
                if brief_summary:
                    parts.append(brief_summary)
                text = " [SEP] ".join(parts) if parts else ""
                if len(text) > MAX_TEXT_CHARS:
                    text = text[:MAX_TEXT_CHARS]
                if text:
                    cache[nct_id] = text
    finally:
        conn.close()
    return cache


def split_queries(
    qids: Iterable[int],
    seed: int,
    ratio: float,
) -> tuple[list[int], list[int]]:
    """Deterministic 80/20 query-level split (sort → seeded shuffle → slice)."""
    qids_sorted = sorted(set(qids))
    rng = random.Random(seed)
    rng.shuffle(qids_sorted)
    n_train = int(len(qids_sorted) * ratio)
    return qids_sorted[:n_train], qids_sorted[n_train:]


def group_by_qid(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["query_id"], []).append(r)
    return out


def build_pairs_for_query(
    candidates: list[dict],
    trial_text: dict[str, str],
) -> Iterable[dict]:
    """Emit one row per ordered pair (A, B) where grade(A) > grade(B).

    Pairs whose nct_id is missing from `trial_text` (SQL miss) are dropped
    upfront. Pairs where grade(A) == grade(B) are skipped (no ranking signal).
    """
    resolved = [c for c in candidates if c["nct_id"] in trial_text]
    for a in resolved:
        for b in resolved:
            if a["nct_id"] == b["nct_id"]:
                continue
            if a["grade"] <= b["grade"]:
                continue
            yield {
                "query_id": a["query_id"],
                "query": a["query"],
                "positive": trial_text[a["nct_id"]],
                "negative": trial_text[b["nct_id"]],
                "label": float(a["grade"] - b["grade"]),
                "doc_pos_nct_id": a["nct_id"],
                "doc_neg_nct_id": b["nct_id"],
                "grade_pos": a["grade"],
                "grade_neg": b["grade"],
            }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def main() -> None:
    _setup_logging()
    _validate_inputs()

    logger.info("Loading held-out query_ids from %d files", len(HELD_OUT_FILES))
    held_out = load_held_out_qids(HELD_OUT_FILES)
    logger.info("Held-out qid count: %d", len(held_out))

    logger.info("Loading source labels from %d files", len(SOURCE_LABEL_FILES))
    rows, n_dropped, n_deduped = load_source_rows(SOURCE_LABEL_FILES, held_out)
    logger.info(
        "Source pool: %d unique (qid, nct_id) rows  (dropped %d held-out, %d duplicates)",
        len(rows),
        n_dropped,
        n_deduped,
    )

    nct_ids = {r["nct_id"] for r in rows}
    logger.info("Fetching trial text for %d unique nct_ids", len(nct_ids))
    trial_text = build_trial_text_cache(nct_ids, TRIALS_DB)
    hit_rate = 100 * len(trial_text) / max(len(nct_ids), 1)
    logger.info(
        "Trial-text cache: %d / %d nct_ids resolved (%.1f%% hit rate)",
        len(trial_text),
        len(nct_ids),
        hit_rate,
    )

    qids = {r["query_id"] for r in rows}
    train_qids, val_qids = split_queries(qids, SEED, SPLIT_RATIO)
    logger.info(
        "Query split: %d train + %d val  (seed=%d, ratio=%.2f, total=%d)",
        len(train_qids),
        len(val_qids),
        SEED,
        SPLIT_RATIO,
        len(qids),
    )

    grouped = group_by_qid(rows)

    train_pairs = [
        pair
        for qid in train_qids
        for pair in build_pairs_for_query(grouped[qid], trial_text)
    ]
    val_pairs = [
        pair
        for qid in val_qids
        for pair in build_pairs_for_query(grouped[qid], trial_text)
    ]

    n_train = write_jsonl(OUTPUT_TRAIN, train_pairs)
    n_val = write_jsonl(OUTPUT_VAL, val_pairs)
    logger.info("Wrote %d train pairs → %s", n_train, OUTPUT_TRAIN.relative_to(PROJECT_ROOT))
    logger.info("Wrote %d val pairs   → %s", n_val, OUTPUT_VAL.relative_to(PROJECT_ROOT))

    label_dist = Counter(int(r["label"]) for r in train_pairs)
    train_qids_with_pairs = {r["query_id"] for r in train_pairs}
    val_qids_with_pairs = {r["query_id"] for r in val_pairs}

    metadata = {
        "n_train_queries": len(train_qids_with_pairs),
        "n_val_queries": len(val_qids_with_pairs),
        "n_train_pairs": n_train,
        "n_val_pairs": n_val,
        "label_distribution_train": {str(k): label_dist[k] for k in sorted(label_dist)},
        "held_out_qid_count": len(held_out),
        "trial_text_lookup_cache_size": len(trial_text),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Wrote metadata → %s", OUTPUT_METADATA.relative_to(PROJECT_ROOT))
    logger.info("Summary:\n%s", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
