"""Cross-encoder re-ranker for (query, trial) pairs.

Uses a fine-tuned BioLinkBERT cross-encoder to score relevance.
Receives the top-k candidates from hybrid retrieval and produces
a refined relevance score for the LightGBM metadata blender.
"""

import logging
import time

from sentence_transformers import CrossEncoder

from TrialMine.monitoring import time_model_cm

logger = logging.getLogger(__name__)

# Logical model name surfaced to MODEL_INFERENCE — kept stable across HF
# checkpoint paths so the Grafana panel doesn't re-bucket on a model swap.
_CE_MODEL_LABEL = "biolinkbert-cross-encoder"


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
        with time_model_cm(_CE_MODEL_LABEL):
            scores = self.model.predict(pairs, convert_to_numpy=True)
        return scores.tolist()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 20,
        text_key: str = "trial_text",
        rrf_weight: float | None = None,
    ) -> list[dict]:
        """Re-rank candidates by a blend of RRF rank and cross-encoder score.

        Scores each candidate with the cross-encoder, attaches
        ``cross_encoder_score`` (sigmoid of the logit) to each dict, and
        produces a ``blended_score`` of::

            blended = rrf_weight * rrf_norm + (1 − rrf_weight) * ce_sigmoid

        where ``rrf_norm`` is min-max normalised within the candidate pool.
        Sorts by blended_score descending and returns the top-k.

        Default ``rrf_weight = 0.3`` (CE-dominant) was selected by the Phase
        C3 blender sweep on the held-out 65-query benchmark — NDCG@5 0.861
        at α=0.3 vs 0.842 at the previous α=0.7 (RRF-dominant). The flip
        only became safe with the v2 (graded MarginMSELoss) CE; v1's binary
        CE was a worse ranker than RRF alone and required the 0.7/0.3 RRF
        floor as a quality-preserving cap (see Decision 39 + §10 of
        docs/evaluation-report.md). Pure-CE re-ranking (`rrf_weight=0.0`)
        hits NDCG@5=0.898 on the same benchmark — an even larger lift,
        deferred to a future architectural step.

        Args:
            query: Patient query string.
            candidates: List of candidate dicts from hybrid retriever.
                Each must carry the trial text under ``text_key`` and the
                RRF ``score`` (the hybrid retriever's per-candidate score).
            top_k: Number of top candidates to return.
            text_key: Key in candidate dict containing the trial text.
            rrf_weight: Optional override for the blend weight (α). If
                None, uses the production default (0.3, CE-dominant).

        Returns:
            Re-ranked list of candidates with ``cross_encoder_score`` +
            ``blended_score`` fields added.
        """
        if not candidates:
            return []

        if rrf_weight is None:
            rrf_weight = 0.3
        if not 0.0 <= rrf_weight <= 1.0:
            raise ValueError(f"rrf_weight must be in [0, 1]; got {rrf_weight!r}")

        texts = [c[text_key] for c in candidates]
        ce_scores = self.score(query, texts)

        # Normalize CE scores to [0, 1] via sigmoid (already logits)
        import math

        for candidate, ce in zip(candidates, ce_scores, strict=False):
            candidate["cross_encoder_score"] = 1 / (1 + math.exp(-ce))

        # Normalize RRF scores to [0, 1] for blending
        rrf_scores = [c.get("score", 0.0) for c in candidates]
        rrf_max = max(rrf_scores) if rrf_scores else 1.0
        rrf_min = min(rrf_scores) if rrf_scores else 0.0
        rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

        ce_weight = 1.0 - rrf_weight
        for candidate in candidates:
            rrf_norm = (candidate.get("score", 0.0) - rrf_min) / rrf_range
            ce_norm = candidate["cross_encoder_score"]
            candidate["blended_score"] = rrf_weight * rrf_norm + ce_weight * ce_norm

        ranked = sorted(candidates, key=lambda x: x["blended_score"], reverse=True)
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
