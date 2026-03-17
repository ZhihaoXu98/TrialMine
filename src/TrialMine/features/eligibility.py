"""Structured eligibility feature extraction for the LightGBM ranker.

Converts free-text eligibility criteria + patient profile into binary/numeric
features indicating inclusion/exclusion criterion matches.

Features produced:
- age_match: 1 if patient age within [min_age, max_age]
- sex_match: 1 if patient sex matches trial requirement
- prior_therapy_mention: 1 if patient's prior therapies overlap with criteria
- biomarker_match: fraction of mentioned biomarkers matched
"""

import logging

logger = logging.getLogger(__name__)


def compute_eligibility_features(trial: dict, patient_profile: dict) -> dict:
    """Compute eligibility match features for a (trial, patient) pair.

    Args:
        trial: Trial dict including eligibility sub-dict.
        patient_profile: Patient profile with age, sex, biomarkers, etc.

    Returns:
        Dict of feature_name → numeric value for the ranker.
    """
    # TODO: implement age, sex, biomarker, and prior therapy matching
    raise NotImplementedError
