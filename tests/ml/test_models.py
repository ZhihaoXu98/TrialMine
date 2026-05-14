"""Smoke tests for the trained ML artefacts.

Each test gates on the presence of the corresponding model directory /
file via ``pytest.mark.skipif`` so the suite passes in environments that
don't have the (~1 GB) model artefacts (CI by default; fresh checkouts).

These are not full integration tests — they cover the contract that
matters: the artefact loads, the output shape is right, and basic
semantic / scoring behaviour is sane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDER_PATH = REPO_ROOT / "models" / "embeddings" / "fine-tuned"
CROSS_ENCODER_PATH = REPO_ROOT / "models" / "cross-encoder" / "fine-tuned"
RANKER_PATH = REPO_ROOT / "models" / "ranker" / "v3-regularized" / "model.lgb"

# Skip-gate predicates. We check for a *load-bearing* file, not just the
# directory — gitignore now lets ``metadata.json`` through (so the
# directories may exist in CI without weights), and ``Path.exists()`` on
# the bare dir would falsely advertise a usable model.
HAS_EMBEDDER = (EMBEDDER_PATH / "config.json").exists()
HAS_CROSS_ENCODER = (CROSS_ENCODER_PATH / "config.json").exists()
HAS_RANKER = RANKER_PATH.exists()


# --------------------------------------------------------------------------- #
# Embedder                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not HAS_EMBEDDER,
    reason="Fine-tuned embedder not present at models/embeddings/fine-tuned",
)
def test_embedder_loads() -> None:
    """Bi-encoder loads from disk and exposes an ``embed_text`` method."""
    from TrialMine.models.embeddings import TrialEmbedder

    embedder = TrialEmbedder(model_name=str(EMBEDDER_PATH))
    assert hasattr(embedder, "embed_text")


@pytest.mark.skipif(
    not HAS_EMBEDDER,
    reason="Fine-tuned embedder not present at models/embeddings/fine-tuned",
)
def test_embedder_outputs_768_dimensional_vector() -> None:
    """BioLinkBERT-base hidden size is 768; that's the contract FAISS expects."""
    from TrialMine.models.embeddings import TrialEmbedder

    embedder = TrialEmbedder(model_name=str(EMBEDDER_PATH))
    vec = np.asarray(embedder.embed_text("metastatic prostate cancer"))
    assert vec.shape == (768,), f"expected (768,), got {vec.shape}"
    # L2 norm should be close to 1 — embed_text mean-pools then normalises.
    norm = float(np.linalg.norm(vec))
    assert 0.95 <= norm <= 1.05, f"expected ~unit-normalised vector, got |v|={norm:.3f}"


@pytest.mark.skipif(
    not HAS_EMBEDDER,
    reason="Fine-tuned embedder not present at models/embeddings/fine-tuned",
)
def test_embedder_places_cancer_closer_to_tumor_than_to_bicycle() -> None:
    """Sanity check on the semantic space: clinical synonyms cluster."""
    from TrialMine.models.embeddings import TrialEmbedder

    embedder = TrialEmbedder(model_name=str(EMBEDDER_PATH))
    cancer = np.asarray(embedder.embed_text("cancer"))
    tumor = np.asarray(embedder.embed_text("tumor"))
    bicycle = np.asarray(embedder.embed_text("bicycle"))

    sim_cancer_tumor = float(np.dot(cancer, tumor))
    sim_cancer_bicycle = float(np.dot(cancer, bicycle))
    assert sim_cancer_tumor > sim_cancer_bicycle, (
        f"cancer↔tumor ({sim_cancer_tumor:.3f}) should beat "
        f"cancer↔bicycle ({sim_cancer_bicycle:.3f})"
    )


# --------------------------------------------------------------------------- #
# Cross-encoder                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not HAS_CROSS_ENCODER,
    reason="Fine-tuned cross-encoder not present at models/cross-encoder/fine-tuned",
)
def test_cross_encoder_loads_and_scores_pairs() -> None:
    """CE checkpoint loads and ``score`` returns one float per (query, text) pair."""
    from TrialMine.models.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name=str(CROSS_ENCODER_PATH))
    scores = reranker.score(
        query="metastatic prostate cancer",
        trial_texts=[
            "Phase 3 trial of enzalutamide in metastatic prostate cancer",
            "A study of bicycles in adolescents",
        ],
    )
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


# --------------------------------------------------------------------------- #
# LightGBM ranker                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not HAS_RANKER,
    reason="LightGBM ranker not present at models/ranker/v3-regularized/model.lgb",
)
def test_lightgbm_ranker_loads_and_predicts() -> None:
    """LightGBM model loads and produces a score for a single feature row."""
    from TrialMine.models.ranker import FEATURE_NAMES, RankingBlender

    blender = RankingBlender(model_path=str(RANKER_PATH))
    assert blender.model is not None
    n_feats = blender.model.num_feature()
    assert n_feats == len(FEATURE_NAMES), (
        f"Booster expects {n_feats} features, FEATURE_NAMES has "
        f"{len(FEATURE_NAMES)} — version drift."
    )
    # Predict on a single zeroed row — just confirms the predict path is wired.
    out = blender.predict(np.zeros((1, n_feats), dtype=np.float32))
    assert out.shape == (1,)
    assert np.isfinite(out).all()
