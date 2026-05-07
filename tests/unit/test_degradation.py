"""Unit tests for :class:`TrialMine.config.DegradationConfig` and its
wiring into the search orchestrator.

Two layers:

* Plain Pydantic-level validation (defaults, bounds, immutable types).
* Behaviour: the orchestrator reads the policy on every ``search()``
  call. We feed it a tiny stub retriever so the test runs without
  Elasticsearch / FAISS / cross-encoder.
"""

from __future__ import annotations

import asyncio

import pytest

from TrialMine.agents.orchestrator import SearchOrchestrator
from TrialMine.agents.query_parser import PatientProfile
from TrialMine.config import DegradationConfig, get_default_degradation

# --------------------------------------------------------------------------- #
# DegradationConfig — pure validation                                         #
# --------------------------------------------------------------------------- #


def test_degradation_defaults_match_documented_budgets() -> None:
    """Defaults are part of the contract — pin them so accidental edits
    that loosen the agent budget are caught here."""
    cfg = DegradationConfig()
    assert cfg.cross_encoder_enabled is True
    assert cfg.eligibility_check_enabled is True
    assert cfg.skip_cross_encoder_if_slow_s == 5.0
    assert cfg.skip_agent_if_slow_s == 10.0


def test_degradation_rejects_non_positive_budgets() -> None:
    """A zero / negative budget would mean ``always degrade``; that's a
    misconfiguration, surface it as a validation error."""
    with pytest.raises(ValueError):
        DegradationConfig(skip_cross_encoder_if_slow_s=0.0)
    with pytest.raises(ValueError):
        DegradationConfig(skip_agent_if_slow_s=-1.0)


def test_get_default_degradation_returns_singleton() -> None:
    """Same instance per process — call sites can compare by identity if needed."""
    assert get_default_degradation() is get_default_degradation()


# --------------------------------------------------------------------------- #
# Orchestrator wiring                                                         #
# --------------------------------------------------------------------------- #


class _StubRetriever:
    """Minimal HybridRetriever-shaped stub: tracks which paths were taken."""

    def __init__(self) -> None:
        self.search_calls: list[dict] = []
        self.full_pipeline_calls: list[dict] = []

    def search(self, query: str, top_k: int = 10, filters=None) -> list[dict]:
        self.search_calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [
            {
                "nct_id": f"NCT{idx:08d}",
                "title": f"Stub trial {idx}",
                "phase": "Phase 2",
                "status": "RECRUITING",
                "score": 1.0 - idx * 0.1,
            }
            for idx in range(1, top_k + 1)
        ]

    def full_pipeline(self, query, reranker, blender, top_k, rerank_top_k, filters):
        self.full_pipeline_calls.append({"query": query})
        # Should never be called when CE is disabled — the test asserts
        # that absence rather than relying on this body.
        return self.search(query, top_k=top_k, filters=filters), {"bm25_ms": 1.0}


class _StubNormalizer:
    def expand_query(self, q: str) -> list[str]:
        return [q]


def _orchestrator_with(degradation: DegradationConfig) -> tuple[SearchOrchestrator, _StubRetriever]:
    """Build an orchestrator that won't touch ES / FAISS / CE."""
    retriever = _StubRetriever()
    orch = SearchOrchestrator(
        retriever=retriever,  # type: ignore[arg-type]
        concept_normalizer=_StubNormalizer(),  # type: ignore[arg-type]
        top_k=3,
        eligibility_top_k=2,
        degradation=degradation,
    )
    return orch, retriever


def test_cross_encoder_disabled_skips_full_pipeline() -> None:
    """``cross_encoder_enabled=False`` must short-circuit to hybrid-only."""
    cfg = DegradationConfig(cross_encoder_enabled=False)
    orch, retriever = _orchestrator_with(cfg)

    profile = PatientProfile(raw_query="lung cancer", condition="lung cancer")
    result_dict, trace = asyncio.run(orch.search(profile))

    # Full pipeline (CE) must not have been called.
    assert retriever.full_pipeline_calls == []
    # Hybrid search must have been called instead.
    assert len(retriever.search_calls) >= 1

    # The retrieve trace step records that we degraded by toggle.
    retrieve_step = next(s for s in trace if s["step"] == "retrieve")
    assert retrieve_step["decisions"]["pipeline"] == "hybrid_only_disabled"
    assert len(result_dict["results"]) == 3


def test_eligibility_disabled_skips_eligibility_step() -> None:
    """``eligibility_check_enabled=False`` must skip the parallel SQLite reads."""
    cfg = DegradationConfig(cross_encoder_enabled=False, eligibility_check_enabled=False)
    orch, _ = _orchestrator_with(cfg)
    profile = PatientProfile(raw_query="metastatic prostate cancer")
    _, trace = asyncio.run(orch.search(profile))

    elig = next(s for s in trace if s["step"] == "check_eligibility")
    assert elig["decisions"]["n_checked"] == 0
    assert elig["decisions"]["skipped_by_degradation"] is True
