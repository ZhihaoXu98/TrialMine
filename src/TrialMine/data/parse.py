"""Parse raw ClinicalTrials.gov API responses into ClinicalTrial models.

Handles:
- Extracting nested protocolSection fields (see CLAUDE.md for key paths)
- Normalising age strings ("18 Years" → 18)
- Splitting free-text eligibility into inclusion/exclusion lists
"""

import logging
from typing import Any

from TrialMine.data.models import ClinicalTrial, EligibilityCriteria, Intervention, Location

logger = logging.getLogger(__name__)


def parse_age(age_str: str | None) -> int | None:
    """Convert an age string like '18 Years' or '6 Months' to an integer year value.

    Args:
        age_str: Raw age string from the API, or None.

    Returns:
        Age in years as an integer, or None if unparseable.
    """
    # TODO: handle Years / Months / Weeks / Days conversions
    raise NotImplementedError


def parse_eligibility(raw: dict[str, Any]) -> EligibilityCriteria:
    """Parse the eligibilityModule dict into an EligibilityCriteria model.

    Args:
        raw: eligibilityModule dict from the API response.

    Returns:
        Populated EligibilityCriteria.
    """
    # TODO: extract eligibilityCriteria text, minimumAge, maximumAge, sex
    raise NotImplementedError


def parse_interventions(raw: list[dict[str, Any]]) -> list[Intervention]:
    """Parse the interventions list from armsInterventionsModule.

    Args:
        raw: List of intervention dicts from the API.

    Returns:
        List of Intervention models.
    """
    # TODO: map type, name, description fields
    raise NotImplementedError


def parse_locations(raw: list[dict[str, Any]]) -> list[Location]:
    """Parse the locations list from contactsLocationsModule.

    Args:
        raw: List of location dicts from the API.

    Returns:
        List of Location models.
    """
    # TODO: map facility, city, state, country, status fields
    raise NotImplementedError


def parse_study(raw: dict[str, Any]) -> ClinicalTrial:
    """Convert a raw API study dict to a validated ClinicalTrial model.

    Args:
        raw: One study dict from the ClinicalTrials.gov API v2 response.

    Returns:
        Fully populated and validated ClinicalTrial.
    """
    # TODO: navigate protocolSection.* paths per CLAUDE.md and call helpers above
    raise NotImplementedError
