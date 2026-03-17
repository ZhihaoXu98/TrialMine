"""Pydantic models for clinical trial data.

These are the canonical types used throughout the pipeline.
parse.py → these models → store.py → SQLite → retrieval layer.
"""

from pydantic import BaseModel, Field


class Location(BaseModel):
    """A single trial site."""

    facility: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    zip_code: str | None = None


class Trial(BaseModel):
    """Canonical representation of one clinical trial."""

    nct_id: str = Field(..., description="ClinicalTrials.gov unique identifier")
    title: str = Field(default="")
    brief_summary: str | None = None
    detailed_description: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)  # intervention names only
    eligibility_criteria: str | None = None
    min_age: str | None = None   # raw string, e.g. "18 Years"
    max_age: str | None = None
    sex: str | None = None       # "ALL", "FEMALE", "MALE"
    phase: str | None = None     # e.g. "Phase 1", "Phase 1/Phase 2"
    status: str | None = None    # e.g. "RECRUITING", "COMPLETED"
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    sponsor: str | None = None
    locations: list[Location] = Field(default_factory=list)
    url: str | None = None
