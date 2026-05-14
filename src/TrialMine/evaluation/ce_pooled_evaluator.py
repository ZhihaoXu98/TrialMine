"""Pooled cross-encoder evaluator — production-shaped NDCG@5/@10 + MRR.

Phase A5 of docs/fix_CE.md.

Replaces the v1 CERerankingEvaluator's 1-positive / 1-negative setup
(which saturated NDCG@10 at 0.993 on the binary CE and was non-predictive
of production ranking quality) with a per-query pooled ranking: for each
val query, take the first `pool_top_k` graded candidates from the source
label files (in build-order, matching the dedup rule of
scripts/build_ce_graded_training.py), score them with the model, sort by
score, and compute NDCG@5/@10 + MRR over the full pool.

The evaluator is data-self-contained: trial text is recovered from
ce_graded_val.jsonl (which already carries the trial text the builder
extracted from trials.db), so the cloud training instance does not need
trials.db rsynced.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

# Use the new (non-deprecated) import path. v5.5+ exposes the base classes
# at this location; the legacy `sentence_transformers.evaluation` path
# triggers a DeprecationWarning at import time.
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator

from TrialMine.evaluation.metrics import mrr, ndcg_at_k

logger = logging.getLogger(__name__)


class CEPooledEvaluator(SentenceEvaluator):
    """Evaluate a CrossEncoder with a pooled NDCG / MRR over graded candidates.

    For each val query, pools up to `pool_top_k` candidates (in source-file
    build-order, dedup by (qid, nct_id) first-occurrence — same rule the
    training-pair builder uses, so labels and pool order are aligned).
    Trial text comes from the val triplet file's `positive` / `negative`
    columns (the builder pre-extracted text from trials.db), so this
    evaluator needs no SQLite at runtime.

    Returns metrics with keys `{name}_ndcg@5`, `{name}_ndcg@10`, `{name}_mrr`
    so a top-level `metric_for_best_model: "{name}_ndcg@10"` in the training
    config picks the best checkpoint by the right metric.
    """

    def __init__(
        self,
        val_pair_file: str | Path,
        source_label_files: Iterable[str | Path],
        pool_top_k: int = 20,
        max_val_pairs: int | None = None,
        batch_size: int = 64,
        name: str = "pooled",
    ) -> None:
        super().__init__()
        self.name = name
        self.pool_top_k = pool_top_k
        self.batch_size = batch_size

        val_pair_path = Path(val_pair_file)
        source_paths = [Path(p) for p in source_label_files]
        missing = [p for p in (val_pair_path, *source_paths) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "CEPooledEvaluator inputs missing:\n  " + "\n  ".join(str(p) for p in missing)
            )

        val_qids, text_map = self._load_val_text(val_pair_path)
        logger.info(
            "Pooled evaluator: val_pair_file=%s  val_qids=%d  text_map=%d entries",
            val_pair_path.name,
            len(val_qids),
            len(text_map),
        )

        per_qid_query, per_qid_candidates = self._load_source_candidates(source_paths, val_qids)
        logger.info(
            "Pooled evaluator: collected candidate lists for %d / %d val qids",
            len(per_qid_candidates),
            len(val_qids),
        )

        self.queries = self._resolve_queries(
            val_qids, per_qid_query, per_qid_candidates, text_map, pool_top_k
        )

        if max_val_pairs is not None:
            self._apply_max_pairs_cap(max_val_pairs)

        total_pairs = sum(len(q["candidates"]) for q in self.queries)
        logger.info(
            "Pooled evaluator ready: %d queries  •  %d (q, doc) pairs  •  "
            "pool_top_k=%d  •  max_val_pairs=%s  •  metric prefix=%s_",
            len(self.queries),
            total_pairs,
            pool_top_k,
            max_val_pairs,
            self.name,
        )

    # ── construction helpers ─────────────────────────────────────────────

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[dict]:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)

    def _load_val_text(self, val_pair_path: Path) -> tuple[set[int], dict[tuple[int, str], str]]:
        """Extract val qids and (qid, nct_id) → trial_text from the pair file."""
        val_qids: set[int] = set()
        text_map: dict[tuple[int, str], str] = {}
        for row in self._iter_jsonl(val_pair_path):
            qid = int(row["query_id"])
            val_qids.add(qid)
            # Both positive and negative trial texts go into the lookup.
            # First-occurrence wins; the text should be identical for the
            # same (qid, nct_id) across pairs, but being explicit avoids
            # a silent dependency on row order.
            for nct_key, text_key in (
                ("doc_pos_nct_id", "positive"),
                ("doc_neg_nct_id", "negative"),
            ):
                nct = str(row[nct_key])
                if (qid, nct) not in text_map:
                    text_map[(qid, nct)] = str(row[text_key])
        return val_qids, text_map

    def _load_source_candidates(
        self,
        source_paths: list[Path],
        val_qids: set[int],
    ) -> tuple[dict[int, str], dict[int, list[tuple[str, int]]]]:
        """Walk source label files in given order; collect per-qid candidates."""
        per_qid_query: dict[int, str] = {}
        per_qid_candidates: dict[int, list[tuple[str, int]]] = {}
        seen: set[tuple[int, str]] = set()
        for path in source_paths:
            for row in self._iter_jsonl(path):
                qid = int(row["query_id"])
                if qid not in val_qids:
                    continue
                nct = str(row["nct_id"])
                key = (qid, nct)
                if key in seen:
                    continue
                seen.add(key)
                per_qid_query.setdefault(qid, str(row["query"]))
                per_qid_candidates.setdefault(qid, []).append((nct, int(row["relevance"])))
        return per_qid_query, per_qid_candidates

    def _resolve_queries(
        self,
        val_qids: set[int],
        per_qid_query: dict[int, str],
        per_qid_candidates: dict[int, list[tuple[str, int]]],
        text_map: dict[tuple[int, str], str],
        pool_top_k: int,
    ) -> list[dict]:
        """Materialise per-query pool: keep only candidates whose text we have."""
        queries: list[dict] = []
        skipped_no_text = 0
        for qid in sorted(val_qids):
            candidates = per_qid_candidates.get(qid)
            if not candidates or qid not in per_qid_query:
                continue
            resolved: list[dict] = []
            for nct, grade in candidates:
                text = text_map.get((qid, nct))
                if text is None:
                    skipped_no_text += 1
                    continue
                resolved.append({"nct_id": nct, "grade": grade, "text": text})
                if len(resolved) >= pool_top_k:
                    break
            if not resolved:
                continue
            queries.append(
                {
                    "query_id": qid,
                    "query": per_qid_query[qid],
                    "candidates": resolved,
                }
            )
        if skipped_no_text:
            logger.info(
                "Pooled evaluator: %d (qid, nct) candidates skipped — no trial text "
                "(candidate not in any val pair)",
                skipped_no_text,
            )
        return queries

    def _apply_max_pairs_cap(self, max_val_pairs: int) -> None:
        """Trim full queries from the end until total pairs ≤ max_val_pairs."""
        total = sum(len(q["candidates"]) for q in self.queries)
        if total <= max_val_pairs:
            return
        # Deterministic: queries are already sorted by qid; drop highest-qid first.
        kept: list[dict] = []
        running = 0
        for q in self.queries:
            if running + len(q["candidates"]) > max_val_pairs:
                break
            kept.append(q)
            running += len(q["candidates"])
        logger.info(
            "Pooled evaluator: max_val_pairs=%d cap applied — kept %d/%d queries (%d/%d pairs)",
            max_val_pairs,
            len(kept),
            len(self.queries),
            running,
            total,
        )
        self.queries = kept

    # ── evaluator interface ──────────────────────────────────────────────

    def __call__(
        self,
        model,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        """Score each query's pool with the model; return mean NDCG/MRR."""
        if not self.queries:
            zeros = {
                f"{self.name}_ndcg@5": 0.0,
                f"{self.name}_ndcg@10": 0.0,
                f"{self.name}_mrr": 0.0,
            }
            logger.warning("Pooled evaluator has no queries — returning zeros: %s", zeros)
            return zeros

        ndcg5s: list[float] = []
        ndcg10s: list[float] = []
        mrrs: list[float] = []
        for q in self.queries:
            pairs = [(q["query"], c["text"]) for c in q["candidates"]]
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scored = sorted(
                zip(q["candidates"], scores, strict=False),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            result_ids = [c["nct_id"] for c, _ in scored]
            relevance_scores: dict[str, float] = {
                c["nct_id"]: float(c["grade"]) for c in q["candidates"]
            }
            relevant_ids: set[str] = {nct for nct, grade in relevance_scores.items() if grade > 0}
            ndcg5s.append(ndcg_at_k(result_ids, relevance_scores, 5))
            ndcg10s.append(ndcg_at_k(result_ids, relevance_scores, 10))
            mrrs.append(mrr(result_ids, relevant_ids))

        metrics = {
            f"{self.name}_ndcg@5": float(np.mean(ndcg5s)),
            f"{self.name}_ndcg@10": float(np.mean(ndcg10s)),
            f"{self.name}_mrr": float(np.mean(mrrs)),
        }
        logger.info(
            "Pooled evaluator @ epoch=%s steps=%s  •  ndcg@5=%.4f  ndcg@10=%.4f  mrr=%.4f",
            epoch,
            steps,
            metrics[f"{self.name}_ndcg@5"],
            metrics[f"{self.name}_ndcg@10"],
            metrics[f"{self.name}_mrr"],
        )
        return metrics
