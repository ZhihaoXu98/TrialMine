"""Pydantic models mirroring the ClinicalTrials.gov API v2 response structure.

These are the canonical data structures used throughout the pipeline.
Downstream consumers (parse.py, DB layer, retrieval) should all use these types.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Intervention(BaseModel):
    """A single intervention from armsInterventionsModule."""

    type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class Location(BaseModel):
    """A trial site location from contactsLocationsModule."""

    facility: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    status: Optional[str] = None


class EligibilityCriteria(BaseModel):
    """Parsed eligibility from eligibilityModule."""

    raw_text: str = ""
    min_age: Optional[str] = None
    max_age: Optional[str] = None
    sex: Optional[str] = None
    # TODO: add structured inclusion/exclusion lists after NLP parsing


class ClinicalTrial(BaseModel):
    """Canonical representation of a single clinical trial."""

    nct_id: str = Field(..., description="Unique ClinicalTrials.gov identifier")
    title: Optional[str] = None
    brief_summary: Optional[str] = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    eligibility: EligibilityCriteria = Field(default_factory=EligibilityCriteria)
    phases: list[str] = Field(default_factory=list)
    overall_status: Optional[str] = None
    enrollment_count: Optional[int] = None
    locations: list[Location] = Field(default_factory=list)
    lead_sponsor: Optional[str] = None

    # TODO: add computed fields (age_min_years, age_max_years) after parsing


class TrialPage(BaseModel):
    """A single page of results from the ClinicalTrials.gov API v2."""

    trials: list[ClinicalTrial]
    next_page_token: Optional[str] = None
    total_count: Optional[int] = None
