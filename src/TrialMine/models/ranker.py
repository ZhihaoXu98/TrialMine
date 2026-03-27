"""LightGBM metadata blender / final ranker.

Takes cross-encoder scores + structured metadata features and produces
a final relevance score for each candidate trial.

Feature groups:
- Retrieval signals: BM25 rank/score, semantic rank/score, RRF score
- Cross-encoder: relevance score
- Trial metadata: phase, status, enrollment size, sponsor type
- Eligibility match: age match, sex match, concept overlap
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MetadataRanker:
    """LightGBM-based pointwise ranker combining retrieval and metadata signals."""

    def __init__(self, model_path: Path) -> None:
        """Load a trained LightGBM model from disk.

        Args:
            model_path: Path to a saved .lgbm model file.
        """
        # TODO: lgb.Booster(model_file=str(model_path))
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Produce relevance scores for a feature matrix.

        Args:
            features: Float array of shape (n_candidates, n_features).

        Returns:
            Score array of shape (n_candidates,).
        """
        # TODO: return self.model.predict(features)
        raise NotImplementedError

    def rank(self, candidates: list[dict]) -> list[dict]:
        """Build feature matrix from candidates, score, and sort.

        Args:
            candidates: List of candidate dicts with retrieval signals
                        and metadata already populated.

        Returns:
            Same list sorted by descending predicted relevance score.
        """
        # TODO: extract_features → predict → argsort → reorder
        raise NotImplementedError
