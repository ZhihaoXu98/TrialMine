"""FAISS-based semantic retrieval interface.

Responsibilities:
- Build and persist a FAISS IndexFlatIP (inner product ≈ cosine on normalised vectors)
- Map FAISS integer positions back to NCT IDs
- Query the index and return ranked results
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Manages a FAISS index of trial embeddings for semantic search."""

    def __init__(self, dimension: int = 768) -> None:
        """Initialise an empty FAISS index.

        Uses IndexFlatIP (inner product). When vectors are L2-normalised,
        inner product equals cosine similarity.

        Args:
            dimension: Embedding dimension (must match the embedder).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.trial_ids: list[str] = []

    def build(self, embeddings: np.ndarray, trial_ids: list[str]) -> None:
        """Build the index from pre-computed embeddings.

        Embeddings are L2-normalised before adding so that inner product
        scores equal cosine similarity.

        Args:
            embeddings: Float32 array of shape (n_trials, dimension).
            trial_ids: Ordered list of NCT IDs (one per row).
        """
        if embeddings.shape[0] != len(trial_ids):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(trial_ids)} trial IDs"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: got {embeddings.shape[1]}, expected {self.dimension}"
            )

        # Normalise to unit vectors (in-place)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.trial_ids = list(trial_ids)

        logger.info("FAISS index built with %d vectors (dim=%d)", self.index.ntotal, self.dimension)

    def search(self, query_embedding: np.ndarray, top_k: int = 200) -> list[tuple[str, float]]:
        """Search the index for the nearest neighbours of a query.

        Args:
            query_embedding: Float32 array of shape (dimension,) — should be normalised.
            top_k: Number of results to return.

        Returns:
            List of (nct_id, cosine_similarity_score) tuples, descending by score.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, returning no results")
            return []

        # Reshape to (1, dim) and normalise
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self.trial_ids[idx], float(score)))

        return results

    def save(self, index_path: str, mapping_path: str | None = None) -> None:
        """Save the FAISS index and trial ID mapping to disk.

        Args:
            index_path: Path for the FAISS index file (e.g. data/trial_embeddings.faiss).
            mapping_path: Path for the trial ID JSON mapping.
                          Defaults to index_path with .json extension.
        """
        if mapping_path is None:
            mapping_path = str(Path(index_path).with_suffix(".json"))

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(mapping_path, "w") as f:
            json.dump(self.trial_ids, f)

        logger.info(
            "Saved FAISS index (%d vectors) to %s and mapping to %s",
            self.index.ntotal,
            index_path,
            mapping_path,
        )

    def load(self, index_path: str, mapping_path: str | None = None) -> None:
        """Load a FAISS index and trial ID mapping from disk.

        Args:
            index_path: Path to the FAISS index file.
            mapping_path: Path to the trial ID JSON mapping.
                          Defaults to index_path with .json extension.
        """
        if mapping_path is None:
            mapping_path = str(Path(index_path).with_suffix(".json"))

        self.index = faiss.read_index(str(index_path))
        with open(mapping_path) as f:
            self.trial_ids = json.load(f)

        self.dimension = self.index.d
        logger.info(
            "Loaded FAISS index: %d vectors, dim=%d from %s",
            self.index.ntotal,
            self.dimension,
            index_path,
        )
