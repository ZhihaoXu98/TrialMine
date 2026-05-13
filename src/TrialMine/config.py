"""Application configuration loaded from environment variables and YAML.

Two distinct config surfaces live here:

* :class:`Settings` — environment-driven app configuration (``.env`` →
  process env → defaults). Read at startup; restart to change.
* :class:`DegradationConfig` — runtime degradation policy (which heavy
  components to skip, per-step time budgets). Read on every request,
  so it can be flipped without a restart by future control planes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Values come from .env, then environment, then defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # External services
    anthropic_api_key: str = Field(..., description="Anthropic API key for LLM agents")
    umls_api_key: str = Field("", description="UMLS API key for concept normalization")
    elasticsearch_url: str = Field("http://localhost:9200")

    # Paths
    db_path: str = Field("data/processed/trialmine.db")
    faiss_index_path: str = Field("data/processed/faiss.index")

    # TODO: add fields for model paths, retrieval top-k values, MLflow URI, etc.
    # Load these from configs/{env}.yaml rather than duplicating here.


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    # TODO: implement lru_cache or use FastAPI Depends
    return Settings()  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Graceful-degradation policy                                                  #
# --------------------------------------------------------------------------- #


class DegradationConfig(BaseModel):
    """Runtime policy for skipping heavy components when they're slow or broken.

    Two distinct controls per component:

    * **Hard toggle** (``*_enabled``) — flip to ``False`` to skip a
      component unconditionally. Useful for incident response (cross
      encoder is OOM-looping → skip it entirely until the next deploy)
      and for A/B experiments that turn a component off for a fraction
      of traffic.
    * **Soft budget** (``skip_*_if_slow_s``) — per-call wall-clock cap.
      When the budget is exceeded, the orchestrator gives up waiting,
      logs the event, and degrades to the next-best output. The thread
      may keep running in the background; we accept the leak rather
      than block the user.

    Defaults are tuned to the production budgets called out in
    CLAUDE.md (15 s overall agent budget, ~4 s typical cross-encoder
    inference). The two skip-if-slow values together leave roughly
    ``skip_agent_if_slow_s - skip_cross_encoder_if_slow_s`` seconds of
    slack for the rest of the pipeline (parser, eligibility,
    explanation).
    """

    cross_encoder_enabled: bool = Field(
        default=True,
        description=(
            "Hard toggle: when False, the orchestrator skips the cross-"
            "encoder step entirely and falls back to plain hybrid (BM25+"
            "semantic+RRF) ranking."
        ),
    )
    eligibility_check_enabled: bool = Field(
        default=True,
        description=(
            "Hard toggle: when False, results are returned without the "
            "Met/Unmet/Unknown verdicts. Saves the parallel SQLite reads "
            "+ regex parses (~50–150 ms typical)."
        ),
    )
    eligibility_hard_filter_enabled: bool = Field(
        default=True,
        description=(
            "Hard toggle: when True (default), trials whose eligibility "
            "verdict is 'Unmet' are dropped from the result list. This is "
            "the Phase C4 fix for `complex` queries — without it, trials "
            "that require a treatment the patient has already failed, or "
            "are restricted to a sex/age range the patient doesn't match, "
            "still appear in the ranking. Requires eligibility_check_enabled "
            "to also be True. Safety net: if the filter would drop ALL "
            "results, the orchestrator keeps the unfiltered list (logged). "
            "Performance cost: eligibility checks run on all ranked results "
            "instead of just the top eligibility_top_k, adding ~50–150 ms "
            "to the eligibility step."
        ),
    )
    skip_cross_encoder_if_slow_s: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Per-call cross-encoder budget in seconds. If retrieval+CE "
            "exceeds this, the orchestrator times out and falls back to "
            "plain hybrid ranking. Tune above the warm CE p95 — too "
            "tight and we degrade healthy traffic."
        ),
    )
    skip_agent_if_slow_s: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Outer agent-pipeline wall-clock budget in seconds. Replaces "
            "the legacy 15s default. On timeout the pipeline returns a "
            "structured degraded response with empty results and "
            "``error='timeout'``."
        ),
    )


_DEFAULT_DEGRADATION = DegradationConfig()


def get_default_degradation() -> DegradationConfig:
    """Return the process-wide :class:`DegradationConfig` singleton.

    Today this is a hard-coded default; a future revision will make it
    configurable per-environment via ``configs/degradation.yaml`` and
    swappable at runtime by a control-plane signal.
    """
    return _DEFAULT_DEGRADATION


__all__ = [
    "Settings",
    "get_settings",
    "DegradationConfig",
    "get_default_degradation",
]
