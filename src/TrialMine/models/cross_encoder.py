"""Cross-encoder re-ranker for (query, trial) pairs.

Uses a fine-tuned BioLinkBERT cross-encoder to score relevance.
Receives the top-k candidates from hybrid retrieval and produces
a refined relevance score for the LightGBM metadata blender.
"""

import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Scores (query, trial_text) pairs with a cross-encoder model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model name or local checkpoint path.
            device: Torch device string.
        """
        # TODO: load CrossEncoder from sentence_transformers
        raise NotImplementedError

    def score(self, query: str, trial_texts: list[str]) -> list[float]:
        """Score a batch of (query, trial_text) pairs.

        Args:
            query: Patient query string.
            trial_texts: List of trial text representations to score.

        Returns:
            List of relevance scores aligned with trial_texts order.
        """
        # TODO: call model.predict([(query, t) for t in trial_texts])
        raise NotImplementedError
