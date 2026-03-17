"""LangGraph tool definitions available to the SearchOrchestrator agent.

Tools:
- search_trials: Run hybrid retrieval for a query string
- get_trial_details: Fetch full trial record by NCT ID
- check_eligibility: Run structured eligibility matching for a patient profile
- explain_trial: Generate a patient-friendly explanation of a trial
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def search_trials(query: str, top_k: int = 20) -> list[dict]:
    """Search for clinical trials matching a query.

    Args:
        query: Reformulated search query from QueryParser.
        top_k: Number of candidates to return after re-ranking.

    Returns:
        List of trial dicts with nct_id, title, score fields.
    """
    # TODO: call HybridRetriever → CrossEncoderReranker → MetadataRanker
    raise NotImplementedError


def get_trial_details(nct_id: str) -> dict:
    """Retrieve the full ClinicalTrial record for a given NCT ID.

    Args:
        nct_id: ClinicalTrials.gov unique identifier.

    Returns:
        Full trial dict from the database.
    """
    # TODO: query SQLite / SQLAlchemy by nct_id
    raise NotImplementedError


def check_eligibility(nct_id: str, patient_profile: dict) -> dict:
    """Check whether a patient profile meets a trial's eligibility criteria.

    Args:
        nct_id: Target trial identifier.
        patient_profile: Dict with age, sex, diagnosis, biomarkers, etc.

    Returns:
        Dict with eligible (bool), met_criteria, unmet_criteria, uncertain.
    """
    # TODO: implement rule-based + LLM eligibility check
    raise NotImplementedError


def explain_trial(nct_id: str, patient_profile: dict) -> str:
    """Generate a patient-friendly plain-English explanation of a trial.

    Args:
        nct_id: Target trial identifier.
        patient_profile: Patient context for personalising the explanation.

    Returns:
        Plain-English explanation string.
    """
    # TODO: call LLM with trial details and patient context
    raise NotImplementedError
