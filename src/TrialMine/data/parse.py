"""Parse raw ClinicalTrials.gov API page files into Trial objects.

Each page file contains the full API response JSON with a 'studies' array.
Fields are extracted from protocolSection.* paths (see CLAUDE.md for the map).
All missing fields default to None — this module never raises on bad data.
"""

import json
import logging
from pathlib import Path
from typing import Any

from TrialMine.data.models import Location, Trial

logger = logging.getLogger(__name__)

_TRIAL_URL_BASE = "https://clinicaltrials.gov/study/"

_PHASE_MAP = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "N/A",
}


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested dict access: _get(d, 'a', 'b', 'c') == d.get('a',{}).get('b',{}).get('c')."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})  # type: ignore[assignment]
    return d if d != {} else default


def parse_locations(raw: list[dict[str, Any]]) -> list[Location]:
    """Parse the locations list from contactsLocationsModule.

    Args:
        raw: List of location dicts from the API.

    Returns:
        List of Location models.
    """
    locations = []
    for loc in raw:
        locations.append(
            Location(
                facility=loc.get("facility"),
                city=loc.get("city"),
                state=loc.get("state"),
                country=loc.get("country"),
                zip_code=loc.get("zip"),
            )
        )
    return locations


def parse_phase(phases: list[str]) -> str | None:
    """Convert a list of API phase codes to a readable string.

    Args:
        phases: List like ["PHASE1", "PHASE2"].

    Returns:
        Human-readable phase string, e.g. "Phase 1/Phase 2", or None.
    """
    if not phases:
        return None
    readable = [_PHASE_MAP.get(p, p) for p in phases]
    return "/".join(readable)


def parse_study(raw: dict[str, Any]) -> Trial | None:
    """Convert one raw API study dict to a Trial model.

    Args:
        raw: One element from the API 'studies' array.

    Returns:
        Populated Trial, or None if nct_id is missing.
    """
    ps = raw.get("protocolSection", {})

    id_mod = ps.get("identificationModule", {})
    nct_id = id_mod.get("nctId")
    if not nct_id:
        return None

    title = id_mod.get("officialTitle") or id_mod.get("briefTitle") or ""

    desc_mod = ps.get("descriptionModule", {})
    brief_summary = desc_mod.get("briefSummary")
    detailed_description = desc_mod.get("detailedDescription")

    conditions: list[str] = ps.get("conditionsModule", {}).get("conditions", [])

    interventions_raw = _get(ps, "armsInterventionsModule", "interventions", default=[])
    interventions: list[str] = [
        i["name"] for i in (interventions_raw or []) if i.get("name")
    ]

    elig_mod = ps.get("eligibilityModule", {})
    eligibility_criteria = elig_mod.get("eligibilityCriteria")
    min_age = elig_mod.get("minimumAge")
    max_age = elig_mod.get("maximumAge")
    sex = elig_mod.get("sex")

    design_mod = ps.get("designModule", {})
    phase = parse_phase(design_mod.get("phases", []))
    enrollment = _get(design_mod, "enrollmentInfo", "count", default=None)

    status_mod = ps.get("statusModule", {})
    status = status_mod.get("overallStatus")
    start_date = _get(status_mod, "startDateStruct", "date", default=None)
    completion_date = _get(status_mod, "completionDateStruct", "date", default=None)

    sponsor = _get(ps, "sponsorCollaboratorsModule", "leadSponsor", "name", default=None)

    locations_raw = _get(ps, "contactsLocationsModule", "locations", default=[])
    locations = parse_locations(locations_raw or [])

    return Trial(
        nct_id=nct_id,
        title=title,
        brief_summary=brief_summary,
        detailed_description=detailed_description,
        conditions=conditions,
        interventions=interventions,
        eligibility_criteria=eligibility_criteria,
        min_age=min_age,
        max_age=max_age,
        sex=sex,
        phase=phase,
        status=status,
        enrollment=enrollment,
        start_date=start_date,
        completion_date=completion_date,
        sponsor=sponsor,
        locations=locations,
        url=f"{_TRIAL_URL_BASE}{nct_id}",
    )


def parse_raw_files(raw_dir: Path) -> list[Trial]:
    """Read all page_*.json files and parse them into Trial objects.

    Args:
        raw_dir: Directory containing page_NNNN.json files from download.py.

    Returns:
        List of all successfully parsed Trial objects.
    """
    page_files = sorted(raw_dir.glob("page_*.json"))
    if not page_files:
        logger.warning("No page files found in %s", raw_dir)
        return []

    trials: list[Trial] = []
    missing_eligibility = 0
    missing_conditions = 0
    parse_errors = 0

    for page_file in page_files:
        try:
            data = json.loads(page_file.read_text())
        except Exception:
            logger.exception("Failed to read %s", page_file)
            parse_errors += 1
            continue

        for raw_study in data.get("studies", []):
            trial = parse_study(raw_study)
            if trial is None:
                parse_errors += 1
                continue

            if not trial.eligibility_criteria:
                missing_eligibility += 1
            if not trial.conditions:
                missing_conditions += 1

            trials.append(trial)

    logger.info(
        "Parsed %d trials from %d pages | missing eligibility: %d | "
        "missing conditions: %d | parse errors: %d",
        len(trials),
        len(page_files),
        missing_eligibility,
        missing_conditions,
        parse_errors,
    )
    return trials
