"""Sentence embedding model wrapper for trial and query encoding.

Uses BioLinkBERT via sentence-transformers for biomedical text encoding.
Supports batch encoding with progress tracking.
"""

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from TrialMine.data.models import Trial

logger = logging.getLogger(__name__)


class TrialEmbedder:
    """Encodes trial text and patient queries into dense vectors."""

    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-base",
        device: str = "cpu",
    ) -> None:
        """Load the sentence-transformers model.

        Args:
            model_name: HuggingFace model name or local checkpoint path.
            device: Torch device string ('cpu', 'cuda', 'mps').
        """
        logger.info("Loading embedding model '%s' on %s ...", model_name, device)
        if self._needs_explicit_modules(model_name):
            logger.info("No modules.json found — loading with explicit Transformer+Pooling ...")
            from sentence_transformers.models import Pooling, Transformer

            word_model = Transformer(model_name)
            pooling = Pooling(
                word_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
            )
            self.model = SentenceTransformer(modules=[word_model, pooling], device=device)
        else:
            self.model = SentenceTransformer(model_name, device=device)

        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dimension: %d", self.dimension)

    @staticmethod
    def _needs_explicit_modules(model_name: str) -> bool:
        """Check whether a model lacks sentence-transformers config.

        Models without ``modules.json`` (e.g. raw HuggingFace BERT checkpoints)
        need explicit Transformer + Pooling wiring to avoid a SIGSEGV when
        sentence-transformers auto-detects and silently misconfigures them.

        Args:
            model_name: HuggingFace model name or local path.

        Returns:
            True if explicit module wiring is needed.
        """
        local_path = Path(model_name)
        if local_path.is_dir():
            return not (local_path / "modules.json").exists()
        # Remote model — check HuggingFace cache
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=model_name, filename="modules.json")
            return False  # modules.json exists on the hub
        except Exception:
            return True  # not a sentence-transformers model

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single text into a normalised embedding.

        Args:
            text: Input string to encode.

        Returns:
            Float32 numpy array of shape (embedding_dim,).
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode many texts efficiently with batching.

        Args:
            texts: Input strings to encode.
            batch_size: Encoding batch size.
            show_progress: Whether to show a progress bar.

        Returns:
            Float32 numpy array of shape (len(texts), embedding_dim).
        """
        logger.info("Encoding %d texts (batch_size=%d) ...", len(texts), batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def prepare_trial_text(self, trial: Trial) -> str:
        """Build a single text string from a trial for embedding.

        Concatenates title, conditions, and summary separated by [SEP].
        Handles None fields gracefully.

        Args:
            trial: A Trial object.

        Returns:
            Concatenated text string (max ~512 tokens worth of text).
        """
        parts = []

        if trial.title:
            parts.append(trial.title)

        if trial.conditions:
            parts.append(" ".join(trial.conditions))

        if trial.brief_summary:
            parts.append(trial.brief_summary)

        text = " [SEP] ".join(parts) if parts else ""

        # Rough truncation: ~4 chars per token, 512 tokens ≈ 2048 chars
        max_chars = 2048
        if len(text) > max_chars:
            text = text[:max_chars]

        return text
