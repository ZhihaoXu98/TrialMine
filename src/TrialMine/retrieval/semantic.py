"""FAISS-based semantic retrieval interface.

Responsibilities:
- Load a fine-tuned BioLinkBERT sentence-transformers model
- Build and persist a FAISS IVFFlat or HNSW index over trial embeddings
- Query the index by embedding a user query and returning nearest neighbours
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """A single semantic retrieval result."""

    nct_id: str
    score: float  # cosine similarity


class SemanticRetriever:
    """Wraps a SentenceTransformer model and FAISS index for ANN search."""

    def __init__(self, model_name: str, index_path: Path) -> None:
        """Load model and index from disk.

        Args:
            model_name: HuggingFace model name or local path.
            index_path: Path to a persisted FAISS index file.
        """
        # TODO: load SentenceTransformer, load faiss index, load nct_id mapping
        raise NotImplementedError

    def build_index(self, embeddings: np.ndarray, nct_ids: list[str]) -> None:
        """Build and save a FAISS index from pre-computed embeddings.

        Args:
            embeddings: Float32 array of shape (n_trials, embedding_dim).
            nct_ids: Ordered list of NCT IDs corresponding to each row.
        """
        # TODO: create IVFFlat or HNSW index, train, add, save
        raise NotImplementedError

    def search(self, query: str, top_k: int = 100) -> list[SemanticResult]:
        """Embed query and return top_k approximate nearest neighbours.

        Args:
            query: Natural-language patient description.
            top_k: Number of candidates to return.

        Returns:
            List of SemanticResult sorted by descending similarity.
        """
        # TODO: encode query, faiss search, map indices to nct_ids
        raise NotImplementedError
