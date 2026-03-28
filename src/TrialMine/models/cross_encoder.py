"""Cross-encoder re-ranker for (query, trial) pairs.

Uses a fine-tuned BioLinkBERT cross-encoder to score relevance.
Receives the top-k candidates from hybrid retrieval and produces
a refined relevance score for the LightGBM metadata blender.
"""

import logging
import time

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Scores (query, trial_text) pairs with a cross-encoder model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model name or local checkpoint path.
            device: Torch device string.
        """
        logger.info("Loading cross-encoder '%s' on %s ...", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Cross-encoder loaded.")

    def score(self, query: str, trial_texts: list[str]) -> list[float]:
        """Score a batch of (query, trial_text) pairs.

        Args:
            query: Patient query string.
            trial_texts: List of trial text representations to score.

        Returns:
            List of relevance scores aligned with trial_texts order.
        """
        if not trial_texts:
            return []
        pairs = [(query, t) for t in trial_texts]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        return scores.tolist()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 20,
        text_key: str = "trial_text",
    ) -> list[dict]:
        """Re-rank candidates by cross-encoder score.

        Scores each candidate with the cross-encoder, attaches
        ``cross_encoder_score`` to each dict, sorts descending,
        and returns the top-k.

        Args:
            query: Patient query string.
            candidates: List of candidate dicts from hybrid retriever.
            top_k: Number of top candidates to return.
            text_key: Key in candidate dict containing the trial text.

        Returns:
            Re-ranked list of candidates with ``cross_encoder_score`` added.
        """
        if not candidates:
            return []

        texts = [c[text_key] for c in candidates]
        ce_scores = self.score(query, texts)

        # Normalize CE scores to [0, 1] via sigmoid (already logits)
        import math
        for candidate, ce in zip(candidates, ce_scores):
            candidate["cross_encoder_score"] = 1 / (1 + math.exp(-ce))

        # Normalize RRF scores to [0, 1] for blending
        rrf_scores = [c.get("score", 0.0) for c in candidates]
        rrf_max = max(rrf_scores) if rrf_scores else 1.0
        rrf_min = min(rrf_scores) if rrf_scores else 0.0
        rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

        for candidate in candidates:
            rrf_norm = (candidate.get("score", 0.0) - rrf_min) / rrf_range
            ce_norm = candidate["cross_encoder_score"]
            # Blend: keep RRF as primary signal, CE as boost
            candidate["blended_score"] = 0.7 * rrf_norm + 0.3 * ce_norm

        ranked = sorted(
            candidates, key=lambda x: x["blended_score"], reverse=True
        )
        return ranked[:top_k]

    def rerank_with_timing(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 20,
        text_key: str = "trial_text",
    ) -> tuple[list[dict], float]:
        """Re-rank candidates and return elapsed time.

        Args:
            query: Patient query string.
            candidates: List of candidate dicts from hybrid retriever.
            top_k: Number of top candidates to return.
            text_key: Key in candidate dict containing the trial text.

        Returns:
            Tuple of (re-ranked candidates, elapsed_ms).
        """
        t0 = time.perf_counter()
        ranked = self.rerank(query, candidates, top_k=top_k, text_key=text_key)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ranked, elapsed_ms
