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
        default=60.0,
        gt=0.0,
        description=(
            "Outer agent-pipeline wall-clock budget in seconds. Phase D1 "
            "bumped from 10.0 to 25.0; post-ship trace-store telemetry "
            "showed agent p95 = 61.4 s, so 25 s was below the real p95 "
            "and ~25 % of agent runs hit the outer cap and returned "
            "empty results. Bumped to 60 s to match measured p95 plus "
            "small margin. Aligns with ``agentic_max_iters * "
            "per_iter_timeout_s`` (6 × ~8 s ≈ 48 s real-world). "
            "The rule path is unaffected — rule-arm wall-clock typically "
            "finishes under 10 s so the outer cap is still a safety net "
            "in the common case. On timeout the pipeline now funnels "
            "into the outer fallback (plain hybrid) rather than "
            "returning empty results — see :func:`pipeline.search` "
            "TimeoutError handler."
        ),
    )
    agentic_path_enabled: bool = Field(
        default=True,
        description=(
            "Hard toggle for the Haiku 4.5 ReAct agent path. When False, "
            "the pipeline always routes to the rule-based "
            "SearchOrchestrator; the agent code is wired but inert. "
            "Phase D1 flipped this to True after the C4 decision gate "
            "(Decision 43) cleared all three SHIP criteria: production "
            "NDCG@5 +0.046 [+0.009, +0.079] CI excludes 0; complex-slice "
            "lift +0.226 absolute; cost $0.0066/query under the $0.012 "
            "envelope. The AB router's ``agent_path_v1`` experiment "
            "gates the actual traffic at 50/50, so the net agent share "
            "is ~21 % of production queries."
        ),
    )
    agentic_routing_threshold_slots: int = Field(
        default=2,
        ge=1,
        le=8,
        description=(
            "Route to the agent when the parsed ``PatientProfile`` has "
            "FEWER than this many populated slots (condition, stage, age, "
            "sex, biomarkers, prior_treatments, preferences, location). "
            "Sparse profiles are typically vague queries that benefit "
            "from iterative refinement. Range: 1 to 8."
        ),
    )
    agentic_complex_pattern_enabled: bool = Field(
        default=True,
        description=(
            "When True, also route to the agent if the raw query matches "
            "the complex-pattern heuristic (multi-constraint phrasings "
            "like 'failed X' + 'after Y', or 3+ medical entities in one "
            "query) even when the slot count is high. Separate toggle "
            "from ``agentic_routing_threshold_slots`` so the two signals "
            "can be A/B'd independently."
        ),
    )
    agentic_max_iters: int = Field(
        default=6,
        ge=1,
        le=10,
        description=(
            "Hard cap on the ReAct loop. On reaching the cap, the agent "
            "must call ``submit_final_results`` with whatever it has — or "
            "the pipeline falls back to the rule arm. Range: 1 to 10. "
            "Bumped from 5 to 6 in Phase C3b.2 to absorb parallel-tool-call "
            "bursts (Haiku frequently fires 3+ tools in one cycle); paired "
            "with the system-prompt termination-deadline directive so the "
            "extra cycle is buffer, not a new target. ``recursion_limit`` "
            "in :mod:`react_agent` scales automatically via "
            "``max_iters * 2 + 2``. NOTE: this cap and "
            "``skip_agent_if_slow_s`` move together — total inner budget "
            "(``max_iters * per_iter_timeout_s`` ≈ 36 s, real-world ~48 s "
            "with CE warmup) must fit under the outer wall-clock cap or "
            "the agent's last iteration gets cancelled mid-flight."
        ),
    )
    agentic_per_iter_timeout_s: float = Field(
        default=6.0,
        gt=0.0,
        description=(
            "Per-iteration wall-clock budget for ONE LLM call + tool "
            "execution. Sized for the worst-case tool — ``search_trials`` "
            "runs the full retrieval pipeline (BM25 + semantic + ~4s CE "
            "rerank), so a per-iter under 5s would cancel almost every "
            "iteration that calls ``search_trials``. 6s leaves 2s slack "
            "on top of CE's warm p95. WARNING: the multiplied cap "
            "(``agentic_max_iters * agentic_per_iter_timeout_s`` = 30s "
            "by default) is HIGHER than the outer pipeline budget "
            "``skip_agent_if_slow_s`` (10s), which means the outer "
            "``wait_for`` in ``pipeline.search()`` will cancel the agent "
            "long before this inner per-iter timeout fires. This is "
            "intentional for the Phase A–C build (agent is default-OFF, "
            "so the outer cap never matters in practice). At Phase D "
            "ship, raise ``skip_agent_if_slow_s`` to ~25s if you want "
            "the inner timeout to be the binding constraint; otherwise "
            "accept that the outer budget will be the active cap and "
            "document it in D1's CLAUDE.md decision entry."
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
