"""LangChain tools for the SearchOrchestrator agent.

Four ``@tool`` functions wrap the retrieval / eligibility / concept layers
so a LangGraph ReAct-style agent can invoke them:

* :func:`search_trials` — full retrieval pipeline (BM25 + semantic + RRF
  + cross-encoder + LightGBM) over the indexed corpus.
* :func:`lookup_medical_concept` — lay → medical synonym expansion via the
  hand-built :class:`ConceptNormalizer`.
* :func:`check_trial_eligibility` — Met / Unmet / Unknown patient-vs-trial
  matching (per design Decision 19).
* :func:`get_trial_details` — fetch the full trial record by NCT ID.

All tools return JSON strings (LangChain tool messages must be strings)
and catch exceptions to return ``{"error": "..."}`` rather than raising
into the agent loop.

Heavy resources — Elasticsearch client, FAISS index, BioLinkBERT
bi-encoder, cross-encoder, LightGBM blender — are lazy-loaded into
module-level singletons on first call and cached for the process
lifetime. Override defaults via environment variables; see :data:`_PATHS`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from TrialMine.features.concepts import ConceptNormalizer
from TrialMine.features.eligibility import EligibilityParser

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# Override any path via environment variables; otherwise sensible defaults
# rooted at the project working directory.
_PATHS: dict[str, str] = {
    "es_url": os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
    "es_index": os.getenv("TRIALMINE_ES_INDEX", "trials"),
    "faiss_index": os.getenv("TRIALMINE_FAISS_INDEX", "data/faiss_finetuned.index"),
    "faiss_mapping": os.getenv("TRIALMINE_FAISS_MAPPING", "data/faiss_finetuned.json"),
    "embedder": os.getenv("TRIALMINE_EMBEDDER", "models/embeddings/fine-tuned"),
    "cross_encoder": os.getenv(
        "TRIALMINE_CROSS_ENCODER", "models/cross-encoder/fine-tuned"
    ),
    "ranker": os.getenv("TRIALMINE_RANKER", "models/ranker/v2/model.lgb"),
    "db_path": os.getenv("TRIALMINE_DB_PATH", "data/trials.db"),
}


# --------------------------------------------------------------------------- #
# Lazy singletons                                                             #
# --------------------------------------------------------------------------- #

_singletons: dict[str, Any] = {}
_load_failures: set[str] = set()  # cache loaders that failed once — skip retry


def _get_es():
    """Lazy-load and cache an :class:`ElasticsearchIndex` handle."""
    if "es" in _singletons:
        return _singletons["es"]
    from TrialMine.retrieval.bm25 import ElasticsearchIndex

    _singletons["es"] = ElasticsearchIndex(
        es_url=_PATHS["es_url"], index_name=_PATHS["es_index"]
    )
    return _singletons["es"]


def _get_faiss():
    """Lazy-load and cache the FAISS index from disk."""
    if "faiss" in _singletons:
        return _singletons["faiss"]
    from TrialMine.retrieval.semantic import FAISSIndex

    idx = FAISSIndex()
    idx.load(_PATHS["faiss_index"], _PATHS["faiss_mapping"])
    _singletons["faiss"] = idx
    return idx


def _get_embedder():
    """Lazy-load and cache the BioLinkBERT bi-encoder."""
    if "embedder" in _singletons:
        return _singletons["embedder"]
    from TrialMine.models.embeddings import TrialEmbedder

    _singletons["embedder"] = TrialEmbedder(model_name=_PATHS["embedder"], device="cpu")
    return _singletons["embedder"]


def _get_hybrid():
    """Lazy-build and cache the :class:`HybridRetriever`."""
    if "hybrid" in _singletons:
        return _singletons["hybrid"]
    from TrialMine.retrieval.hybrid import HybridRetriever

    _singletons["hybrid"] = HybridRetriever(
        bm25=_get_es(), semantic=_get_faiss(), embedder=_get_embedder()
    )
    return _singletons["hybrid"]


def _get_reranker_or_none():
    """Try to load the cross-encoder; cache ``None`` on failure to avoid retry."""
    if "reranker" in _singletons:
        return _singletons["reranker"]
    if "reranker" in _load_failures:
        return None
    path = _PATHS["cross_encoder"]
    if not Path(path).exists():
        logger.warning(
            "Cross-encoder missing at %s — search_trials will fall back to hybrid only",
            path,
        )
        _load_failures.add("reranker")
        return None
    try:
        from TrialMine.models.cross_encoder import CrossEncoderReranker

        _singletons["reranker"] = CrossEncoderReranker(model_name=path, device="cpu")
        return _singletons["reranker"]
    except Exception as exc:
        logger.warning("Cross-encoder load failed: %s", exc)
        _load_failures.add("reranker")
        return None


def _get_blender_or_none():
    """Try to load the LightGBM ranker; cache ``None`` on failure."""
    if "blender" in _singletons:
        return _singletons["blender"]
    if "blender" in _load_failures:
        return None
    path = _PATHS["ranker"]
    if not Path(path).exists():
        logger.warning(
            "LightGBM ranker missing at %s — pipeline will use CE-blended fallback",
            path,
        )
        _load_failures.add("blender")
        return None
    try:
        from TrialMine.models.ranker import RankingBlender

        _singletons["blender"] = RankingBlender(model_path=path)
        return _singletons["blender"]
    except Exception as exc:
        logger.warning("LightGBM ranker load failed: %s", exc)
        _load_failures.add("blender")
        return None


def _get_normalizer() -> ConceptNormalizer:
    """Cheap; the synonym map is in-memory."""
    if "normalizer" not in _singletons:
        _singletons["normalizer"] = ConceptNormalizer()
    return _singletons["normalizer"]


def _get_eligibility_parser() -> EligibilityParser:
    """Return a regex-only :class:`EligibilityParser` (SciSpacy not loaded).

    Live re-parsing fills in age/sex from regex; condition/treatment buckets
    fall back to empty when the parsed_eligibility row is missing. That's
    acceptable for the tool because age/sex are the high-value match fields.
    """
    if "eligibility_parser" not in _singletons:
        _singletons["eligibility_parser"] = EligibilityParser()
    return _singletons["eligibility_parser"]


def _db_conn() -> sqlite3.Connection:
    """Open a fresh SQLite connection with row factory; callers must close()."""
    db = _PATHS["db_path"]
    if not Path(db).exists():
        raise FileNotFoundError(f"SQLite database not found: {db}")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _err(message: str) -> str:
    """Render a uniform error JSON envelope."""
    return json.dumps({"error": message})


def _split_conditions(raw: str | None) -> list[str]:
    """Split the BM25 conditions field (semicolon-joined) into a list."""
    if not raw:
        return []
    return [c.strip() for c in raw.split(";") if c.strip()]


def _trim(text: str | None, max_chars: int = 400) -> str:
    """Hard-truncate free-text fields so tool output stays compact."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


def _normalize_filters(phase: str | None, status: str | None) -> dict[str, str]:
    """Normalise filter values to ClinicalTrials.gov stored conventions.

    Status keywords are uppercase (e.g. ``RECRUITING``) with underscores;
    phase is title-cased with a space (e.g. ``Phase 3``). The LLM may pass
    either casing — we coerce leniently so the filter actually matches.
    """
    filters: dict[str, str] = {}
    if status:
        filters["status"] = status.strip().upper().replace(" ", "_")
    if phase:
        p = phase.strip()
        # "phase 3" → "Phase 3"; leave already-stored forms ("Phase 1/Phase 2") alone.
        if p.lower().startswith("phase ") and "/" not in p:
            p = "Phase " + p.split()[-1]
        filters["phase"] = p
    return filters


def _serialize_search_result(r: dict) -> dict:
    """Strip the candidate dict to a small set of LLM-friendly fields."""
    score = (
        r.get("blender_score")
        if r.get("blender_score") is not None
        else r.get("blended_score")
        if r.get("blended_score") is not None
        else r.get("score") or 0.0
    )
    return {
        "nct_id": r.get("nct_id"),
        "title": r.get("title", ""),
        "conditions": _split_conditions(r.get("conditions")),
        "phase": r.get("phase"),
        "status": r.get("status"),
        "enrollment": r.get("enrollment"),
        "score": round(float(score), 4),
        "source": r.get("source"),
        "url": (
            f"https://clinicaltrials.gov/study/{r['nct_id']}" if r.get("nct_id") else None
        ),
    }


# --------------------------------------------------------------------------- #
# Tool: search_trials                                                         #
# --------------------------------------------------------------------------- #


class SearchTrialsArgs(BaseModel):
    """Arguments for :func:`search_trials`."""

    query: str = Field(
        ...,
        description=(
            "Free-text clinical trial search query. Should describe the patient's "
            "condition (e.g. 'metastatic HER2-positive breast cancer'), an "
            "intervention (e.g. 'CAR-T for relapsed lymphoma'), or a combination."
        ),
    )
    phase: str | None = Field(
        None,
        description=(
            "Optional phase filter. Allowed values follow ClinicalTrials.gov: "
            "'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 1/Phase 2', "
            "'Early Phase 1', 'N/A'."
        ),
    )
    status: str | None = Field(
        None,
        description=(
            "Optional status filter. Allowed values (uppercase, underscored): "
            "'RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION', "
            "'COMPLETED', 'TERMINATED', 'WITHDRAWN', 'NOT_YET_RECRUITING'."
        ),
    )
    top_k: int = Field(
        20,
        ge=1,
        le=50,
        description="Number of top results to return after re-ranking. Default 20, max 50.",
    )


@tool("search_trials", args_schema=SearchTrialsArgs)
def search_trials(
    query: str,
    phase: str | None = None,
    status: str | None = None,
    top_k: int = 20,
) -> str:
    """Search the ClinicalTrials.gov corpus for trials relevant to a query.

    Runs the full retrieval pipeline:
        BM25 + semantic (BioLinkBERT) → RRF → cross-encoder → LightGBM blender.

    If the cross-encoder or LightGBM model is unavailable, the pipeline
    gracefully degrades (CE-blended fallback or hybrid only). The
    ``pipeline`` field in the response indicates which path ran.

    Use this tool to find candidate trials for a patient's condition,
    biomarker, or treatment interest. For a known NCT ID, use
    ``get_trial_details`` instead.

    Returns JSON: ``{"query", "filters", "pipeline", "total", "results",
    "latency_ms"}`` where each ``result`` carries ``nct_id``, ``title``,
    ``conditions`` (list), ``phase``, ``status``, ``enrollment``, ``score``,
    ``source``, ``url``.

    On error: ``{"error": "<message>"}``.
    """
    try:
        if not query or not query.strip():
            return _err("query is empty")

        filters = _normalize_filters(phase, status)
        hybrid = _get_hybrid()
        reranker = _get_reranker_or_none()

        if reranker is not None:
            blender = _get_blender_or_none()
            ranked, timings = hybrid.full_pipeline(
                query=query,
                reranker=reranker,
                blender=blender,
                top_k=top_k,
                filters=filters or None,
            )
            pipeline = "full_pipeline" if blender is not None else "ce_blended"
        else:
            ranked = hybrid.search(query=query, top_k=top_k, filters=filters or None)
            timings = None
            pipeline = "hybrid_only"

        return json.dumps(
            {
                "query": query,
                "filters": filters,
                "pipeline": pipeline,
                "total": len(ranked),
                "results": [_serialize_search_result(r) for r in ranked],
                "latency_ms": (
                    round(timings["total_ms"], 1) if timings is not None else None
                ),
            }
        )
    except Exception as exc:
        logger.exception("search_trials failed")
        return _err(f"search_trials failed: {exc}")


# --------------------------------------------------------------------------- #
# Tool: lookup_medical_concept                                                #
# --------------------------------------------------------------------------- #


class ConceptArgs(BaseModel):
    """Arguments for :func:`lookup_medical_concept`."""

    term: str = Field(
        ...,
        description=(
            "A lay or informal phrase to map to standardised medical "
            "terminology (e.g. 'stomach cancer', 'cancer that spread', "
            "'chemo')."
        ),
    )


@tool("lookup_medical_concept", args_schema=ConceptArgs)
def lookup_medical_concept(term: str) -> str:
    """Map informal patient language to standardised medical terminology.

    Uses a curated lay→medical synonym map (~45 oncology entries). Returns
    both an exact-phrase canonical form (e.g. 'stomach cancer' → 'gastric
    neoplasm') and an in-line query expansion that can be sent to BM25 as
    an OR-style alternate phrasing.

    Use this *before* :func:`search_trials` to broaden vocabulary coverage.
    Unknown phrases pass through unchanged.

    Returns JSON: ``{"original": str, "normalized": str, "expanded": list[str]}``.
    On error: ``{"error": "<message>"}``.
    """
    try:
        if not term or not term.strip():
            return _err("term is empty")
        normalizer = _get_normalizer()
        return json.dumps(
            {
                "original": term,
                "normalized": normalizer.normalize(term),
                "expanded": normalizer.expand_query(term),
            }
        )
    except Exception as exc:
        logger.exception("lookup_medical_concept failed")
        return _err(f"lookup_medical_concept failed: {exc}")


# --------------------------------------------------------------------------- #
# Tool: check_trial_eligibility                                               #
# --------------------------------------------------------------------------- #


class EligibilityArgs(BaseModel):
    """Arguments for :func:`check_trial_eligibility`."""

    nct_id: str = Field(
        ..., description="ClinicalTrials.gov ID (e.g. 'NCT04802759')."
    )
    patient_age: float | None = Field(
        None,
        ge=0,
        le=130,
        description="Patient age in years. Floats permitted (e.g. 0.5 for 6 months).",
    )
    patient_sex: str | None = Field(
        None,
        description="Patient sex: 'Male', 'Female', or omit if unknown.",
    )
    patient_condition: str | None = Field(
        None,
        description=(
            "Free-text patient diagnosis to compare against the trial's "
            "required/excluded conditions (e.g. 'HER2-positive breast cancer')."
        ),
    )
    prior_treatments: str | None = Field(
        None,
        description=(
            "Comma-separated or free-text list of prior treatments "
            "(e.g. 'endocrine therapy, fulvestrant')."
        ),
    )


def _rollup(verdicts: list[str]) -> str:
    """Trial-level rollup per Decision 19.

    Any hard ``Unmet`` → ``Unmet``; all ``Met`` → ``Met``; otherwise ``Unknown``.
    """
    if "Unmet" in verdicts:
        return "Unmet"
    if verdicts and all(v == "Met" for v in verdicts):
        return "Met"
    return "Unknown"


def _match_age(
    trial_min: float | None, trial_max: float | None, age: float | None
) -> dict:
    if age is None:
        return {"verdict": "Unknown", "reason": "patient_age not provided"}
    if trial_min is None and trial_max is None:
        return {
            "verdict": "Unknown",
            "trial_min": None,
            "trial_max": None,
            "patient": age,
            "reason": "trial age not specified",
        }
    if trial_min is not None and age < trial_min:
        return {
            "verdict": "Unmet",
            "trial_min": trial_min,
            "trial_max": trial_max,
            "patient": age,
            "reason": f"patient age {age} below trial minimum {trial_min}",
        }
    if trial_max is not None and age > trial_max:
        return {
            "verdict": "Unmet",
            "trial_min": trial_min,
            "trial_max": trial_max,
            "patient": age,
            "reason": f"patient age {age} above trial maximum {trial_max}",
        }
    return {
        "verdict": "Met",
        "trial_min": trial_min,
        "trial_max": trial_max,
        "patient": age,
    }


def _match_sex(trial_sex: str | None, patient_sex: str | None) -> dict:
    if patient_sex is None:
        return {
            "verdict": "Unknown",
            "trial": trial_sex,
            "patient": None,
            "reason": "patient_sex not provided",
        }
    p = patient_sex.strip().capitalize()
    if trial_sex in (None, "All"):
        return {"verdict": "Met", "trial": trial_sex or "All", "patient": p}
    if trial_sex == p:
        return {"verdict": "Met", "trial": trial_sex, "patient": p}
    return {
        "verdict": "Unmet",
        "trial": trial_sex,
        "patient": p,
        "reason": f"trial restricts to {trial_sex}, patient is {p}",
    }


def _any_overlap(
    needles: list[str], haystack: str | None
) -> tuple[bool, list[str]]:
    """Case-insensitive substring scan; returns (found, matched needles)."""
    if not haystack or not needles:
        return False, []
    h = haystack.lower()
    matches = [n for n in needles if n and n.lower() in h]
    return bool(matches), matches


def _match_required_conditions(
    required: list[str], patient_condition: str | None
) -> dict:
    if not required:
        return {
            "verdict": "Unknown",
            "trial": [],
            "patient": patient_condition,
            "reason": "no required conditions parsed for trial",
        }
    if not patient_condition:
        return {
            "verdict": "Unknown",
            "trial": required,
            "patient": None,
            "reason": "patient_condition not provided",
        }
    found, matches = _any_overlap(required, patient_condition)
    if found:
        return {
            "verdict": "Met",
            "trial": required,
            "patient": patient_condition,
            "matches": matches,
        }
    return {
        "verdict": "Unmet",
        "trial": required,
        "patient": patient_condition,
        "reason": "patient_condition does not mention any required trial condition",
    }


def _match_excluded_conditions(
    excluded: list[str], patient_condition: str | None
) -> dict:
    if not excluded:
        return {
            "verdict": "Met",
            "trial": [],
            "patient": patient_condition,
            "reason": "no exclusion conditions",
        }
    if not patient_condition:
        return {
            "verdict": "Unknown",
            "trial": excluded,
            "patient": None,
            "reason": "patient_condition not provided",
        }
    found, matches = _any_overlap(excluded, patient_condition)
    if found:
        return {
            "verdict": "Unmet",
            "trial": excluded,
            "patient": patient_condition,
            "matches": matches,
            "reason": "patient_condition matches an exclusion criterion",
        }
    return {"verdict": "Met", "trial": excluded, "patient": patient_condition}


def _match_required_treatments(
    required: list[str], prior_treatments: str | None
) -> dict:
    if not required:
        return {
            "verdict": "Met",
            "trial": [],
            "patient": prior_treatments,
            "reason": "no required prior treatments",
        }
    if not prior_treatments:
        return {
            "verdict": "Unknown",
            "trial": required,
            "patient": None,
            "reason": "prior_treatments not provided",
        }
    found, matches = _any_overlap(required, prior_treatments)
    if found:
        return {
            "verdict": "Met",
            "trial": required,
            "patient": prior_treatments,
            "matches": matches,
        }
    return {
        "verdict": "Unmet",
        "trial": required,
        "patient": prior_treatments,
        "reason": "patient has not received any required prior treatment",
    }


def _match_excluded_treatments(
    excluded: list[str], prior_treatments: str | None
) -> dict:
    if not excluded:
        return {
            "verdict": "Met",
            "trial": [],
            "patient": prior_treatments,
            "reason": "no excluded prior treatments",
        }
    if not prior_treatments:
        return {
            "verdict": "Unknown",
            "trial": excluded,
            "patient": None,
            "reason": "prior_treatments not provided",
        }
    found, matches = _any_overlap(excluded, prior_treatments)
    if found:
        return {
            "verdict": "Unmet",
            "trial": excluded,
            "patient": prior_treatments,
            "matches": matches,
            "reason": "patient prior treatment matches an exclusion criterion",
        }
    return {"verdict": "Met", "trial": excluded, "patient": prior_treatments}


def _load_trial_row(con: sqlite3.Connection, nct_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM trials WHERE nct_id = ?", (nct_id,)
    ).fetchone()


def _load_eligibility_row(
    con: sqlite3.Connection, nct_id: str
) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM parsed_eligibility WHERE nct_id = ?", (nct_id,)
    ).fetchone()


@tool("check_trial_eligibility", args_schema=EligibilityArgs)
def check_trial_eligibility(
    nct_id: str,
    patient_age: float | None = None,
    patient_sex: str | None = None,
    patient_condition: str | None = None,
    prior_treatments: str | None = None,
) -> str:
    """Check whether a patient could be eligible for a specific trial.

    Compares patient attributes against the trial's structured eligibility
    profile. Looks up the cached ``parsed_eligibility`` row first; falls
    back to live regex-only parsing of the free-text criteria when no
    cache row exists. Each criterion is judged independently as
    ``Met`` / ``Unmet`` / ``Unknown``.

    Trial-level rollup (per design Decision 19): any hard ``Unmet`` makes
    the trial ``Unmet``; all ``Met`` is ``Met``; otherwise ``Unknown``.
    Distinguish between *patient-unknown* (a missing ``patient_*`` arg)
    and *parser-unknown* (low ``parse_confidence``) — both surface as
    ``Unknown`` but with different reason strings.

    Use this tool *after* :func:`search_trials` to score the top candidates
    against a patient profile. Pass only the fields you have; missing
    fields produce ``Unknown`` rather than blocking the verdict.

    Returns JSON::

        {
            "nct_id": str,
            "verdict": "Met" | "Unmet" | "Unknown",
            "match_score": float | null,    // Met / (Met + Unmet); null if all Unknown
            "criteria": { age, sex, required_conditions, excluded_conditions,
                          required_prior_treatments, excluded_prior_treatments },
            "parse_confidence": float,
            "profile_source": "parsed_eligibility_table" | "live_regex_parse",
            "explanation": str
        }

    On error: ``{"error": "<message>"}``.
    """
    try:
        nct = nct_id.strip().upper()

        try:
            con = _db_conn()
        except Exception as exc:
            return _err(f"database unavailable: {exc}")

        try:
            trial = _load_trial_row(con, nct)
            if trial is None:
                return _err(f"trial {nct} not found")
            elig_row = _load_eligibility_row(con, nct)
        finally:
            con.close()

        # Build profile: prefer cached parsed table, else live regex parse.
        if elig_row is not None:
            profile = {
                "min_age_years": elig_row["min_age_years"],
                "max_age_years": elig_row["max_age_years"],
                "sex": elig_row["sex"],
                "required_conditions": json.loads(
                    elig_row["required_conditions"] or "[]"
                ),
                "excluded_conditions": json.loads(
                    elig_row["excluded_conditions"] or "[]"
                ),
                "required_prior_treatments": json.loads(
                    elig_row["required_prior_treatments"] or "[]"
                ),
                "excluded_prior_treatments": json.loads(
                    elig_row["excluded_prior_treatments"] or "[]"
                ),
                "parse_confidence": float(elig_row["parse_confidence"] or 0.0),
                "section_source": elig_row["section_source"],
                "source": "parsed_eligibility_table",
            }
        else:
            parser = _get_eligibility_parser()
            parsed = parser.parse(
                trial["eligibility_criteria"],
                min_age_col=trial["min_age"],
                max_age_col=trial["max_age"],
                sex_col=trial["sex"],
            )
            profile = parsed.model_dump()
            profile["source"] = "live_regex_parse"

        criteria = {
            "age": _match_age(
                profile["min_age_years"], profile["max_age_years"], patient_age
            ),
            "sex": _match_sex(profile["sex"], patient_sex),
            "required_conditions": _match_required_conditions(
                profile.get("required_conditions") or [], patient_condition
            ),
            "excluded_conditions": _match_excluded_conditions(
                profile.get("excluded_conditions") or [], patient_condition
            ),
            "required_prior_treatments": _match_required_treatments(
                profile.get("required_prior_treatments") or [], prior_treatments
            ),
            "excluded_prior_treatments": _match_excluded_treatments(
                profile.get("excluded_prior_treatments") or [], prior_treatments
            ),
        }

        verdicts = [v["verdict"] for v in criteria.values()]
        n_met = sum(1 for v in verdicts if v == "Met")
        n_unmet = sum(1 for v in verdicts if v == "Unmet")
        n_unknown = sum(1 for v in verdicts if v == "Unknown")
        score: float | None = (
            n_met / (n_met + n_unmet) if (n_met + n_unmet) > 0 else None
        )
        rollup = _rollup(verdicts)

        explanation_bits = [
            f"{n_met} criteria met",
            f"{n_unmet} unmet",
            f"{n_unknown} unknown",
        ]
        if profile.get("parse_confidence", 0) < 0.5:
            explanation_bits.append(
                "low parse confidence — verdict may be unreliable"
            )

        return json.dumps(
            {
                "nct_id": nct,
                "verdict": rollup,
                "match_score": round(score, 3) if score is not None else None,
                "criteria": criteria,
                "parse_confidence": profile.get("parse_confidence", 0.0),
                "profile_source": profile["source"],
                "explanation": "; ".join(explanation_bits),
            }
        )
    except Exception as exc:
        logger.exception("check_trial_eligibility failed")
        return _err(f"check_trial_eligibility failed: {exc}")


# --------------------------------------------------------------------------- #
# Tool: get_trial_details                                                     #
# --------------------------------------------------------------------------- #


class TrialDetailsArgs(BaseModel):
    """Arguments for :func:`get_trial_details`."""

    nct_id: str = Field(
        ..., description="ClinicalTrials.gov ID (e.g. 'NCT04802759')."
    )


@tool("get_trial_details", args_schema=TrialDetailsArgs)
def get_trial_details(nct_id: str) -> str:
    """Fetch the full record for a single clinical trial by NCT ID.

    Reads from the local SQLite store (140K parsed trials). Returns
    title, brief_summary (truncated to 1500 chars), conditions,
    interventions, eligibility_criteria (truncated to 2000 chars), age /
    sex columns, phase, status, enrollment, dates, sponsor, and the
    canonical ClinicalTrials.gov URL.

    Use this to surface details to the patient after they pick a trial,
    or to read eligibility text before calling :func:`check_trial_eligibility`.

    Returns JSON of the trial record. On error: ``{"error": "<message>"}``.
    """
    try:
        nct = nct_id.strip().upper()

        try:
            con = _db_conn()
        except Exception as exc:
            return _err(f"database unavailable: {exc}")
        try:
            row = _load_trial_row(con, nct)
        finally:
            con.close()

        if row is None:
            return _err(f"trial {nct} not found")

        return json.dumps(
            {
                "nct_id": row["nct_id"],
                "title": row["title"],
                "brief_summary": _trim(row["brief_summary"], 1500),
                "conditions": json.loads(row["conditions"] or "[]"),
                "interventions": json.loads(row["interventions"] or "[]"),
                "eligibility_criteria": _trim(row["eligibility_criteria"], 2000),
                "min_age": row["min_age"],
                "max_age": row["max_age"],
                "sex": row["sex"],
                "phase": row["phase"],
                "status": row["status"],
                "enrollment": row["enrollment"],
                "start_date": row["start_date"],
                "completion_date": row["completion_date"],
                "sponsor": row["sponsor"],
                "url": row["url"]
                or f"https://clinicaltrials.gov/study/{row['nct_id']}",
            }
        )
    except Exception as exc:
        logger.exception("get_trial_details failed")
        return _err(f"get_trial_details failed: {exc}")


# --------------------------------------------------------------------------- #
# Tool registry                                                               #
# --------------------------------------------------------------------------- #

ALL_TOOLS = [
    search_trials,
    lookup_medical_concept,
    check_trial_eligibility,
    get_trial_details,
]

__all__ = [
    "ALL_TOOLS",
    "search_trials",
    "lookup_medical_concept",
    "check_trial_eligibility",
    "get_trial_details",
    # Pydantic arg schemas exported for typed callers / tests
    "SearchTrialsArgs",
    "ConceptArgs",
    "EligibilityArgs",
    "TrialDetailsArgs",
]
