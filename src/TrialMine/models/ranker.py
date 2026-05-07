"""LightGBM LambdaRank re-ranker for clinical trial search.

Combines retrieval scores (BM25, semantic, cross-encoder) with trial
metadata features to produce a final ranking score. Trained with
LambdaRank to optimize NDCG directly.
"""

import logging
import math

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

PHASE_MAP = {
    "PHASE1": 1.0,
    "Phase 1": 1.0,
    "PHASE2": 2.0,
    "Phase 2": 2.0,
    "PHASE3": 3.0,
    "Phase 3": 3.0,
    "PHASE4": 4.0,
    "Phase 4": 4.0,
    "EARLY_PHASE1": 0.5,
    "Early Phase 1": 0.5,
    "NA": 0.0,
    "NOT_APPLICABLE": 0.0,
}

RECRUITING_STATUSES = {"RECRUITING"}
ACTIVE_STATUSES = {"ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"}

FEATURE_NAMES = [
    "bm25_score",
    "semantic_score",
    "cross_encoder_score",
    "rrf_score",
    "phase_numeric",
    "is_recruiting",
    "is_active",
    "enrollment_log",
    "condition_exact_match",
    "title_query_overlap",
    "has_eligibility",
]


def phase_to_numeric(phase: str | None) -> float:
    """Convert phase string to numeric value.

    Args:
        phase: Phase string from ClinicalTrials.gov (e.g., "Phase 3", "PHASE3").

    Returns:
        Numeric phase value. 0.0 for unknown/None.
    """
    if phase is None:
        return 0.0
    # Handle combined phases like "Phase 1/Phase 2"
    if "/" in phase:
        parts = phase.split("/")
        values = [PHASE_MAP.get(p.strip(), 0.0) for p in parts]
        return sum(values) / len(values) if values else 0.0
    return PHASE_MAP.get(phase, 0.0)


def compute_features(
    query: str,
    candidate: dict,
    trial_doc: dict | None = None,
) -> dict[str, float]:
    """Compute ranking features for a single (query, candidate) pair.

    Args:
        query: Search query string.
        candidate: Candidate dict with retrieval scores and metadata.
        trial_doc: Optional full trial document from Elasticsearch.

    Returns:
        Dict of feature name -> value.
    """
    query_words = set(query.lower().split())

    # Retrieval scores
    bm25_score = candidate.get("bm25_score", 0.0)
    semantic_score = candidate.get("semantic_score", 0.0)
    ce_score = candidate.get("cross_encoder_score", 0.0)
    rrf_score = candidate.get("score", candidate.get("rrf_score", 0.0))

    # Metadata — prefer trial_doc if available
    doc = trial_doc or candidate
    phase = doc.get("phase")
    status = doc.get("status", "")
    enrollment = doc.get("enrollment") or 0
    conditions = doc.get("conditions", "")
    title = doc.get("title", "")
    eligibility = doc.get("eligibility_criteria", "")

    # Derived features
    cond_words = set(conditions.lower().split()) if conditions else set()
    title_words = set(title.lower().split()) if title else set()
    condition_match = 1.0 if query_words & cond_words else 0.0
    title_overlap = len(query_words & title_words) / len(query_words) if query_words else 0.0

    return {
        "bm25_score": bm25_score,
        "semantic_score": semantic_score,
        "cross_encoder_score": ce_score,
        "rrf_score": rrf_score,
        "phase_numeric": phase_to_numeric(phase),
        "is_recruiting": 1.0 if status in RECRUITING_STATUSES else 0.0,
        "is_active": 1.0 if status in ACTIVE_STATUSES else 0.0,
        "enrollment_log": math.log1p(enrollment),
        "condition_exact_match": condition_match,
        "title_query_overlap": title_overlap,
        "has_eligibility": 1.0 if eligibility and len(eligibility) > 10 else 0.0,
    }


class RankingBlender:
    """LightGBM LambdaRank model for final re-ranking."""

    def __init__(self, model_path: str | None = None) -> None:
        """Load a trained LightGBM model.

        Args:
            model_path: Path to the saved .lgb model file.
                        If None, the model must be set via load().
        """
        self.model: lgb.Booster | None = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load a LightGBM model from disk.

        Args:
            model_path: Path to the .lgb model file.
        """
        self.model = lgb.Booster(model_file=model_path)
        logger.info("Loaded LightGBM ranker from %s", model_path)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict ranking scores for a feature matrix.

        Args:
            features: 2D array of shape (n_candidates, n_features).

        Returns:
            1D array of ranking scores.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return np.asarray(self.model.predict(features))

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """Re-rank candidates using the LightGBM model.

        Args:
            query: Search query string.
            candidates: List of candidate dicts with retrieval scores + metadata.
            top_k: Number of top candidates to return.

        Returns:
            Re-ranked list of candidates with ``blender_score`` added.
        """
        if not candidates:
            return []

        feature_rows = []
        for c in candidates:
            feats = compute_features(query, c)
            feature_rows.append([feats[name] for name in FEATURE_NAMES])

        features = np.array(feature_rows, dtype=np.float32)
        scores = self.predict(features)

        for candidate, score in zip(candidates, scores, strict=False):
            candidate["blender_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["blender_score"], reverse=True)
        return ranked[:top_k]
