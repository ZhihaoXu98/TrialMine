"""Rule-based search orchestrator.

Runs a deterministic sequence over an extracted ``PatientProfile``:

    1. Normalize condition via :class:`ConceptNormalizer` (lay → medical).
    2. Build a single retrieval query string from condition + biomarkers + stage.
    3. Compute filters — default ``status=RECRUITING``; honour explicit phase
       preferences and the "completed" override.
    4. Run :meth:`HybridRetriever.full_pipeline` (BM25 + semantic + cross-encoder
       + LightGBM) → top-K candidates, falling back gracefully if CE / blender
       are unavailable.
    5. For the top ``eligibility_top_k`` candidates, call the
       :func:`check_trial_eligibility` tool to attach Met / Unmet / Unknown
       verdicts (per design Decision 19).
    6. Generate a template-based explanation per result — no LLM in this hot
       path. The only LLM call upstream is :class:`QueryParserAgent` (Haiku 4.5).

Design rationale: clinical trial search is a deterministic pipeline, not
adaptive reasoning. A rule-based orchestrator keeps latency predictable
(<3 s typical), cost near zero per query, and the agent trace fully
inspectable for debugging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from TrialMine.agents.query_parser import PatientProfile

if TYPE_CHECKING:
    from TrialMine.features.concepts import ConceptNormalizer
    from TrialMine.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 10
DEFAULT_ELIGIBILITY_TOP_K = 5
DEFAULT_RERANK_TOP_K = 50

# Phase preference keywords → ES stored phase value. Order matters: more
# specific multi-phase strings come first so 'phase 1/2' isn't shadowed by
# the bare 'phase 1' / 'phase 2' entries.
_PHASE_MAP: list[tuple[tuple[str, ...], str]] = [
    (("phase 1/2", "phase 1/phase 2", "phase i/ii"), "Phase 1/Phase 2"),
    (("phase 2/3", "phase 2/phase 3", "phase ii/iii"), "Phase 2/Phase 3"),
    (("early phase 1", "early phase i"), "Early Phase 1"),
    (("phase 4", "phase iv"), "Phase 4"),
    (("phase 3", "phase iii"), "Phase 3"),
    (("phase 2", "phase ii"), "Phase 2"),
    (("phase 1", "phase i"), "Phase 1"),
]

# When any of these appear in the raw query or preferences, drop the default
# RECRUITING filter — the patient explicitly wants completed / closed trials.
_COMPLETED_KEYWORDS: tuple[str, ...] = ("completed", "closed", "finished trial")


class SearchOrchestrator:
    """Rule-based search pipeline. No LLM in the hot path.

    Construct with optional ``retriever`` and ``concept_normalizer`` for
    dependency injection / testing; defaults pull lazily from the
    module-level singletons in :mod:`TrialMine.agents.tools` so the same
    Elasticsearch / FAISS / cross-encoder / LightGBM instances are shared
    across the agent surface.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        concept_normalizer: ConceptNormalizer | None = None,
        top_k: int = DEFAULT_TOP_K,
        eligibility_top_k: int = DEFAULT_ELIGIBILITY_TOP_K,
    ) -> None:
        """Initialise the orchestrator with shared retrieval resources.

        Args:
            retriever: Pre-built :class:`HybridRetriever`. If ``None``,
                resolves to ``tools._get_hybrid()`` lazily on first use.
            concept_normalizer: Pre-built :class:`ConceptNormalizer`.
                If ``None``, resolves to ``tools._get_normalizer()`` lazily.
            top_k: Number of trials to return after retrieval.
            eligibility_top_k: Number of *top* trials to run eligibility
                matching against (must be ≤ ``top_k``).
        """
        if eligibility_top_k > top_k:
            raise ValueError(
                f"eligibility_top_k ({eligibility_top_k}) cannot exceed top_k ({top_k})"
            )
        self._retriever = retriever
        self._concept_normalizer = concept_normalizer
        self.top_k = top_k
        self.eligibility_top_k = eligibility_top_k

    # ------------------------------------------------------------------ #
    # Lazy dependency resolution                                          #
    # ------------------------------------------------------------------ #

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            from TrialMine.agents.tools import _get_hybrid

            self._retriever = _get_hybrid()
        return self._retriever

    @property
    def concept_normalizer(self) -> ConceptNormalizer:
        if self._concept_normalizer is None:
            from TrialMine.agents.tools import _get_normalizer

            self._concept_normalizer = _get_normalizer()
        return self._concept_normalizer

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def search(self, profile: PatientProfile) -> tuple[dict, list[dict]]:
        """Run the full deterministic search flow.

        Async because the two heavy steps — retrieval (BM25 + semantic +
        cross-encoder; multi-second) and the per-result eligibility checks
        (each a SQLite read) — are off-loaded to the default thread pool
        via :func:`asyncio.to_thread`. The eligibility checks then run
        concurrently via :func:`asyncio.gather`, turning a sequential
        ``N × ~50 ms`` loop into a single ``~50 ms`` wall-clock step.

        Args:
            profile: Output of :meth:`QueryParserAgent.parse`. Always has
                ``raw_query``; may have only that field if the parser
                produced a fallback profile.

        Returns:
            Tuple of (result_dict, trace_entries):
                * ``result_dict`` — keys ``results``, ``query_used``,
                  ``filters``, ``normalized_condition``.
                * ``trace_entries`` — list of structured per-step log dicts
                  ready for the LangGraph state's ``agent_trace`` reducer.
        """
        trace: list[dict] = []

        # --- Step 1: normalize condition (microseconds) ---
        t0 = time.perf_counter()
        normalized = self._normalize_condition(profile)
        trace.append(
            {
                "step": "normalize",
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "decisions": {
                    "input_condition": profile.condition,
                    "fell_back_to_raw_query": profile.condition is None,
                    "normalized": normalized,
                },
            }
        )

        # --- Step 2: build query (microseconds) ---
        t0 = time.perf_counter()
        query = self._build_query(profile, normalized)
        trace.append(
            {
                "step": "build_query",
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "decisions": {
                    "query": query,
                    "components": {
                        "normalized_condition": normalized,
                        "stage": profile.condition_stage,
                        "biomarkers": profile.biomarkers,
                    },
                },
            }
        )

        # --- Step 3: build filters (microseconds) ---
        t0 = time.perf_counter()
        filters = self._build_filters(profile)
        trace.append(
            {
                "step": "build_filters",
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "decisions": {"filters": filters, "preferences": profile.preferences},
            }
        )

        # --- Step 4: retrieve (heavy — off-load to thread pool) ---
        t0 = time.perf_counter()
        ranked, retrieve_timings, pipeline_kind = await asyncio.to_thread(
            self._do_retrieve, query, filters
        )
        trace.append(
            {
                "step": "retrieve",
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "decisions": {
                    "pipeline": pipeline_kind,
                    "n_results": len(ranked),
                    "retrieve_timings": retrieve_timings,
                    "top_nct_ids": [r.get("nct_id") for r in ranked[:5]],
                },
            }
        )

        # --- Step 5: parallel eligibility checks on top eligibility_top_k ---
        # Each call hits SQLite (concurrent reads are safe). asyncio.gather
        # over asyncio.to_thread runs them in the default thread pool —
        # wall-clock time becomes max(per-call latency), not sum().
        t0 = time.perf_counter()
        n_to_check = min(len(ranked), self.eligibility_top_k)
        eligibility_results: list[dict | None] = [None] * len(ranked)
        if n_to_check > 0:
            tasks = [
                asyncio.to_thread(self._check_eligibility, ranked[i]["nct_id"], profile)
                for i in range(n_to_check)
            ]
            partial = await asyncio.gather(*tasks)
            for i, result in enumerate(partial):
                eligibility_results[i] = result
        elig_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace.append(
            {
                "step": "check_eligibility",
                "duration_ms": elig_elapsed_ms,
                "decisions": {
                    "n_checked": n_to_check,
                    "n_total_results": len(ranked),
                    "concurrent": True,
                    "verdicts": [
                        (eligibility_results[i] or {}).get("verdict") for i in range(n_to_check)
                    ],
                },
            }
        )

        # --- Step 6: template explanations ---
        t0 = time.perf_counter()
        results: list[dict] = []
        for i, r in enumerate(ranked):
            elig = eligibility_results[i]
            explanation, warnings = self._build_explanation(r, elig)
            enriched = {
                "nct_id": r.get("nct_id"),
                "title": r.get("title", ""),
                "phase": r.get("phase"),
                "status": r.get("status"),
                "enrollment": r.get("enrollment"),
                "conditions": r.get("conditions", ""),
                "score": r.get("blender_score") or r.get("blended_score") or r.get("score"),
                "source": r.get("source"),
                "explanation": explanation,
                "warnings": warnings,
                "eligibility": elig,
                "url": f"https://clinicaltrials.gov/study/{r['nct_id']}",
            }
            results.append(enriched)
        trace.append(
            {
                "step": "explain",
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "decisions": {"n_explained": len(results)},
            }
        )

        return (
            {
                "results": results,
                "query_used": query,
                "filters": filters,
                "normalized_condition": normalized,
            },
            trace,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _do_retrieve(
        self,
        query: str,
        filters: dict[str, str],
    ) -> tuple[list[dict], dict[str, float] | None, str]:
        """Synchronous retrieve — runs the full pipeline or a fallback.

        Returns ``(ranked, retrieve_timings, pipeline_kind)``. Called from
        :meth:`search` via :func:`asyncio.to_thread` so it can do its
        seconds-long CE inference without blocking the event loop.
        """
        from TrialMine.agents.tools import (
            _get_blender_or_none,
            _get_reranker_or_none,
        )

        reranker = _get_reranker_or_none()
        if reranker is not None:
            blender = _get_blender_or_none()
            ranked, retrieve_timings = self.retriever.full_pipeline(
                query=query,
                reranker=reranker,
                blender=blender,
                top_k=self.top_k,
                rerank_top_k=DEFAULT_RERANK_TOP_K,
                filters=filters or None,
            )
            kind = "full_pipeline" if blender is not None else "ce_blended"
            return ranked, retrieve_timings, kind

        ranked = self.retriever.search(query=query, top_k=self.top_k, filters=filters or None)
        return ranked, None, "hybrid_only"

    def _normalize_condition(self, profile: PatientProfile) -> str:
        """Return the medical-equivalent form of the patient's condition.

        ``ConceptNormalizer.expand_query`` returns ``[original]`` or
        ``[original, expanded]``. We pick the last entry — the medical
        form when one exists, the original otherwise.
        """
        source = profile.condition or profile.raw_query or ""
        if not source.strip():
            return ""
        expanded = self.concept_normalizer.expand_query(source)
        return expanded[-1]

    def _build_query(self, profile: PatientProfile, normalized: str) -> str:
        """Compose a single retrieval query string from the parsed slots.

        Excludes preferences (those drive filters) and location (no geo
        retrieval). Falls back to raw_query if no slot produced any tokens.
        """
        parts: list[str] = []
        if normalized:
            parts.append(normalized)
        if profile.condition_stage:
            parts.append(profile.condition_stage)
        parts.extend(profile.biomarkers)
        composed = " ".join(p for p in parts if p).strip()
        return composed or profile.raw_query

    def _build_filters(self, profile: PatientProfile) -> dict[str, str]:
        """Default to ``status=RECRUITING``; honour phase preferences and the
        "completed" override.
        """
        filters: dict[str, str] = {}

        haystack = " ".join([profile.raw_query or "", *profile.preferences]).lower()
        if not any(kw in haystack for kw in _COMPLETED_KEYWORDS):
            filters["status"] = "RECRUITING"

        prefs_lower = " ".join(profile.preferences).lower()
        for keywords, value in _PHASE_MAP:
            if any(kw in prefs_lower for kw in keywords):
                filters["phase"] = value
                break

        return filters

    def _check_eligibility(self, nct_id: str, profile: PatientProfile) -> dict:
        """Call the ``check_trial_eligibility`` tool and parse its JSON output.

        Returns a dict with at least ``verdict``. On any failure (tool error,
        JSON parse failure) returns ``{"verdict": "Unknown", "error": "..."}``.
        """
        from TrialMine.agents.tools import check_trial_eligibility

        try:
            result_json = check_trial_eligibility.invoke(
                {
                    "nct_id": nct_id,
                    "patient_age": (float(profile.age) if profile.age is not None else None),
                    "patient_sex": profile.sex,
                    "patient_condition": profile.condition,
                    "prior_treatments": (
                        ", ".join(profile.prior_treatments) if profile.prior_treatments else None
                    ),
                }
            )
            return json.loads(result_json)
        except Exception as exc:
            logger.warning("Eligibility check failed for %s: %s", nct_id, exc)
            return {"verdict": "Unknown", "error": f"check failed: {exc}"}

    def _build_explanation(
        self,
        trial: dict,
        eligibility: dict | None,
    ) -> tuple[str, list[str]]:
        """Build a template-based explanation + structured warnings list.

        No LLM. Combines retrieval metadata (phase, status, conditions)
        with the eligibility verdict counts.
        """
        phase = trial.get("phase") or "unspecified-phase"
        status = trial.get("status") or "unknown-status"
        enrollment = trial.get("enrollment")
        conditions_str = trial.get("conditions", "")
        cond_list = [c.strip() for c in conditions_str.split(";") if c.strip()][:2]
        cond_text = "; ".join(cond_list) if cond_list else "the listed condition"

        warnings: list[str] = []

        if eligibility is None:
            match_info = "Eligibility: not checked (outside top candidates)"
        elif "error" in eligibility and "verdict" not in eligibility:
            match_info = f"Eligibility: check failed ({eligibility['error']})"
        else:
            verdict = eligibility.get("verdict", "Unknown")
            criteria = eligibility.get("criteria", {}) or {}
            n_met = sum(1 for c in criteria.values() if c.get("verdict") == "Met")
            n_unmet = sum(1 for c in criteria.values() if c.get("verdict") == "Unmet")
            n_unknown = sum(1 for c in criteria.values() if c.get("verdict") == "Unknown")
            counts = f"({n_met} met, {n_unmet} unmet, {n_unknown} unknown)"
            if verdict == "Met":
                match_info = f"Eligibility: likely a Match {counts}"
            elif verdict == "Unmet":
                match_info = f"Eligibility: likely NOT eligible {counts}"
            else:
                match_info = f"Eligibility: needs review {counts}"
            if eligibility.get("parse_confidence", 1.0) < 0.5:
                warnings.append("low parse confidence — verdict may be unreliable")

        enrollment_text = f", enrollment {enrollment}" if enrollment else ""
        explanation = (
            f"This {phase} trial ({status}{enrollment_text}) for {cond_text}. {match_info}."
        )
        if warnings:
            explanation += " " + " ".join(f"[{w}]" for w in warnings)
        return explanation, warnings


__all__ = [
    "SearchOrchestrator",
    "DEFAULT_TOP_K",
    "DEFAULT_ELIGIBILITY_TOP_K",
    "DEFAULT_RERANK_TOP_K",
]
