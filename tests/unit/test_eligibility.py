"""Tests for the eligibility parser.

Three layers of test:
1. Pure-regex tests use ``EligibilityParser(nlp=None)`` so SciSpacy never loads.
2. ``parse_age_string`` unit tests on column values.
3. Full-parse integration tests on real DB fixtures load the SciSpacy model
   once via a session-scoped fixture and are marked ``slow``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from TrialMine.features.eligibility import (
    EligibilityParser,
    EligibilityProfile,
    parse_age_string,
)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "trials.db"


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def parser_no_nlp() -> EligibilityParser:
    """Parser without SciSpacy — used for regex-only tests."""
    return EligibilityParser(nlp=None)


@pytest.fixture(scope="session")
def parser_with_nlp() -> EligibilityParser:
    """Parser with SciSpacy loaded once per session. Skips on import failure."""
    try:
        import spacy

        nlp = spacy.load(
            "en_core_sci_lg",
            disable=["tagger", "lemmatizer", "attribute_ruler", "parser"],
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"en_core_sci_lg not loadable: {exc}")
    return EligibilityParser(nlp=nlp)


def _load_real_trial(nct_id: str) -> dict:
    if not DB_PATH.exists():
        pytest.skip(f"DB not found at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT nct_id, eligibility_criteria, min_age, max_age, sex "
            "FROM trials WHERE nct_id = ?",
            (nct_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip(f"{nct_id} not in DB")
    return dict(row)


# --------------------------------------------------------------------------- #
# parse_age_string                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("18 Years", 18.0),
        ("1 Year", 1.0),
        ("6 Months", 0.5),
        ("12 Months", 1.0),
        ("70 years", 70.0),
        (None, None),
        ("", None),
        ("garbage", None),
        ("eighteen years", None),
    ],
)
def test_parse_age_string(value: str | None, expected: float | None) -> None:
    assert parse_age_string(value) == expected


# --------------------------------------------------------------------------- #
# Age extraction (regex fallback only — column not provided)                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Age >= 18 years", (18.0, None)),
        ("18 years and older", (18.0, None)),
        ("at least 18 years of age", (18.0, None)),
        ("between 21 and 75 years", (21.0, 75.0)),
        ("Patients 18-75 years are eligible", (18.0, 75.0)),
        ("must be younger than 65", (None, 65.0)),
        ("≥ 18 years", (18.0, None)),
        (">= 18", (18.0, None)),
        ("Age \\>= 18 years", (18.0, None)),  # escape leak from DB
        ("18 to 75 years of age", (18.0, 75.0)),
        ("Adults (18+)", (18.0, None)),
        ("6 months to 18 years", (0.5, 18.0)),
    ],
)
def test_extract_age_regex_patterns(
    parser_no_nlp: EligibilityParser,
    text: str,
    expected: tuple[float | None, float | None],
) -> None:
    min_age, max_age, conf = parser_no_nlp._extract_age(text, None, None)
    assert (min_age, max_age) == expected
    assert conf == pytest.approx(0.9)


def test_extract_age_column_first_overrides_text(parser_no_nlp: EligibilityParser) -> None:
    """When min_age column is set, the regex fallback is ignored."""
    text = "Patients aged 50-80 years"
    min_age, max_age, conf = parser_no_nlp._extract_age(text, "18 Years", "65 Years")
    assert min_age == 18.0
    assert max_age == 65.0
    assert conf == 1.0


def test_extract_age_keyword_fallback(parser_no_nlp: EligibilityParser) -> None:
    """Bare keyword 'adult' is the lowest-precision fallback."""
    min_age, max_age, conf = parser_no_nlp._extract_age("Healthy adult volunteers", None, None)
    assert min_age == 18.0
    assert max_age is None
    assert conf == pytest.approx(0.6)


def test_extract_age_missing(parser_no_nlp: EligibilityParser) -> None:
    min_age, max_age, conf = parser_no_nlp._extract_age("", None, None)
    assert min_age is None
    assert max_age is None
    assert conf == 0.0


# --------------------------------------------------------------------------- #
# Section split                                                               #
# --------------------------------------------------------------------------- #


def test_split_sections_both_headers(parser_no_nlp: EligibilityParser) -> None:
    text = (
        "Inclusion Criteria:\n* age 18+\n* documented disease\n\n"
        "Exclusion Criteria:\n* prior chemotherapy"
    )
    inclusion, exclusion, source = parser_no_nlp._split_sections(text)
    assert source == "headers"
    assert "documented disease" in inclusion
    assert "prior chemotherapy" in exclusion
    assert "Exclusion" not in inclusion


def test_split_sections_single_header(parser_no_nlp: EligibilityParser) -> None:
    text = "Inclusion Criteria:\n* age 18+\n* documented disease"
    inclusion, exclusion, source = parser_no_nlp._split_sections(text)
    assert source == "single_header"
    assert "documented disease" in inclusion
    assert exclusion == ""


def test_split_sections_no_headers_fallback(parser_no_nlp: EligibilityParser) -> None:
    text = "Children and adolescents with leukemia, both genders."
    inclusion, exclusion, source = parser_no_nlp._split_sections(text)
    assert source == "fallback"
    assert inclusion == text.strip()
    assert exclusion == ""


def test_split_sections_variant_header(parser_no_nlp: EligibilityParser) -> None:
    text = "DISEASE CHARACTERISTICS: Histologically confirmed lung cancer"
    inclusion, exclusion, source = parser_no_nlp._split_sections(text)
    assert source == "variant"
    assert "lung cancer" in inclusion


# --------------------------------------------------------------------------- #
# Sex extraction                                                              #
# --------------------------------------------------------------------------- #


def test_extract_sex_from_column(parser_no_nlp: EligibilityParser) -> None:
    sex, conf = parser_no_nlp._extract_sex("any text", "FEMALE")
    assert sex == "Female"
    assert conf == 1.0


def test_extract_sex_from_text_fallback(parser_no_nlp: EligibilityParser) -> None:
    sex, conf = parser_no_nlp._extract_sex("This is a study of female participants only", None)
    assert sex == "Female"
    assert conf == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Stop-list and typing (require SciSpacy NER)                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_stop_list_filters_boilerplate(parser_with_nlp: EligibilityParser) -> None:
    text = (
        "Inclusion Criteria: Subjects must have histologically confirmed "
        "non-small cell lung cancer. Patients must have signed informed consent."
    )
    profile = parser_with_nlp.parse(text, sex_col="ALL")
    bucket = {item.lower() for item in profile.required_conditions}
    # Boilerplate should not appear
    for boilerplate in ("subjects", "patients", "inclusion criteria", "informed consent"):
        assert boilerplate not in bucket, (
            f"'{boilerplate}' leaked into required_conditions: {profile.required_conditions}"
        )


@pytest.mark.slow
def test_typing_routes_treatment_keywords(parser_with_nlp: EligibilityParser) -> None:
    """Spans matching treatment keywords go to treatments; diseases go to conditions."""
    text = (
        "Inclusion Criteria: Histologically confirmed non-small cell lung cancer.\n\n"
        "Exclusion Criteria: Prior chemotherapy or radiation therapy within 4 weeks."
    )
    profile = parser_with_nlp.parse(text, sex_col="ALL")
    cond_lower = " | ".join(profile.required_conditions).lower()
    assert "non-small cell lung cancer" in cond_lower

    excl_treat_lower = " | ".join(profile.excluded_prior_treatments).lower()
    assert "chemotherapy" in excl_treat_lower or "radiation therapy" in excl_treat_lower


# --------------------------------------------------------------------------- #
# Full-parse integration tests on real DB trials                              #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_full_parse_short_real_trial(parser_with_nlp: EligibilityParser) -> None:
    """NCT06073769: short (607 char) PMS trial with std headers."""
    row = _load_real_trial("NCT06073769")
    profile = parser_with_nlp.parse(
        row["eligibility_criteria"],
        min_age_col=row["min_age"],
        max_age_col=row["max_age"],
        sex_col=row["sex"],
    )
    assert isinstance(profile, EligibilityProfile)
    assert profile.section_source == "headers"
    assert profile.min_age_years == 19.0  # column "19 Years"
    assert profile.sex == "All"
    assert profile.parse_confidence > 0.5
    assert "azacitidine" in (
        " | ".join(profile.required_conditions + profile.required_prior_treatments).lower()
    )


@pytest.mark.slow
def test_full_parse_medium_real_trial(parser_with_nlp: EligibilityParser) -> None:
    """NCT05535569: 3254 char gastric trial with std headers."""
    row = _load_real_trial("NCT05535569")
    profile = parser_with_nlp.parse(
        row["eligibility_criteria"],
        min_age_col=row["min_age"],
        max_age_col=row["max_age"],
        sex_col=row["sex"],
    )
    assert profile.section_source == "headers"
    assert profile.min_age_years == 19.0
    assert profile.sex == "All"
    cond_text = " | ".join(profile.required_conditions).lower()
    assert "gastric adenocarcinoma" in cond_text or "adenocarcinoma" in cond_text
    # Both inclusion and exclusion content present
    assert profile.raw_inclusion
    assert profile.raw_exclusion


@pytest.mark.slow
def test_full_parse_long_real_trial_with_escape_leak(
    parser_with_nlp: EligibilityParser,
) -> None:
    """NCT03085069: 7724 char NSCLC trial. Should not crash on any escape leaks."""
    row = _load_real_trial("NCT03085069")
    profile = parser_with_nlp.parse(
        row["eligibility_criteria"],
        min_age_col=row["min_age"],
        max_age_col=row["max_age"],
        sex_col=row["sex"],
    )
    assert profile.section_source == "headers"
    assert profile.min_age_years == 18.0
    assert profile.max_age_years == 80.0
    assert profile.sex == "All"
    assert profile.parse_confidence > 0.7
    # Long trial should yield many entities in both buckets
    assert len(profile.required_conditions) > 5
    assert len(profile.excluded_conditions) > 5


# --------------------------------------------------------------------------- #
# Robustness                                                                  #
# --------------------------------------------------------------------------- #


def test_parse_handles_none_text(parser_no_nlp: EligibilityParser) -> None:
    profile = parser_no_nlp.parse(None, min_age_col="18 Years", sex_col="ALL")
    assert profile.section_source == "empty"
    assert profile.min_age_years == 18.0
    assert profile.sex == "All"
    assert profile.required_conditions == []


def test_parse_handles_empty_text(parser_no_nlp: EligibilityParser) -> None:
    profile = parser_no_nlp.parse("", sex_col="ALL")
    assert profile.section_source == "empty"
    assert profile.required_conditions == []


# --------------------------------------------------------------------------- #
# Phase A6: UMLS-CUI toggle behavior for prior-treatment matching.            #
# Q413 / Q416 canonical cases gated on the                                    #
# DegradationConfig.umls_drug_class_matching_enabled toggle.                  #
# --------------------------------------------------------------------------- #

import pathlib  # noqa: E402  (kept local to A6 block for clarity)
from contextlib import contextmanager  # noqa: E402

# Filesystem gate for tests that need scispacy's UMLS KB. scispacy 0.6.2's
# UmlsLinkerPaths.is_locally_cached() is unreliable (returns False even
# when the KB is fully cached at ~/.scispacy/datasets/, confirmed in the
# Phase A2 run on 2026-05-14), so we check the cache dir directly.
_SCISPACY_KB_DIR = pathlib.Path.home() / ".scispacy" / "datasets"
_KB_CACHED = _SCISPACY_KB_DIR.exists() and any(_SCISPACY_KB_DIR.iterdir())

_requires_kb = pytest.mark.skipif(
    not _KB_CACHED,
    reason=("scispacy UMLS KB not downloaded; run Phase A2 of docs/fix_parser_umls.md first"),
)


@contextmanager
def _umls_toggle(enabled: bool):
    """Temporarily flip ``umls_drug_class_matching_enabled``; reset on exit.

    Mutates the process-wide DegradationConfig singleton. Uses
    try/finally so a failed assertion doesn't leak the toggle state
    into the next test in the module.
    """
    from TrialMine.config import get_default_degradation

    cfg = get_default_degradation()
    prev = cfg.umls_drug_class_matching_enabled
    cfg.umls_drug_class_matching_enabled = enabled
    try:
        yield
    finally:
        cfg.umls_drug_class_matching_enabled = prev


@pytest.mark.slow
@_requires_kb
def test_q413_failed_osimertinib_egfr_tki_excluded_trial() -> None:
    """Q413 canonical: patient had osimertinib (an EGFR TKI). A trial
    that excludes prior EGFR-TKI-exposed patients must return Unmet
    on the ``excluded_prior_treatments`` criterion when the UMLS
    toggle is on.

    Direct substring matching ("EGFR tyrosine kinase inhibitor" vs
    "osimertinib") fails because the two strings share no characters.
    UMLS CUI matching resolves osimertinib -> C4058811 -> EGFR-TKI
    class CUIs (per :data:`DRUG_TO_CLASS_CUIS`), which overlap the
    class CUIs scispacy returns for "EGFR tyrosine kinase inhibitor".
    """
    from TrialMine.agents.tools import _match_excluded_treatments

    with _umls_toggle(True):
        result = _match_excluded_treatments(
            excluded=["EGFR tyrosine kinase inhibitor"],
            prior_treatments="osimertinib",
        )
    assert result["verdict"] == "Unmet", result
    # Sanity: the matched needle list should mention the trial's
    # required exclusion phrase so callers can render an explanation.
    assert "matches" in result and result["matches"], result


@pytest.mark.slow
@_requires_kb
def test_q416_post_trastuzumab_her2_excluded_trial() -> None:
    """Q416 canonical: patient had trastuzumab. A trial excluding
    prior HER2-directed therapy must return Unmet on the
    ``excluded_prior_treatments`` criterion when the UMLS toggle is on.

    Uses the "HER2 inhibitor" phrasing — "HER2-targeted therapy" does
    NOT resolve to a drug-type CUI in scispacy's 2022 KB
    (docs/fix_parser_umls.md A4 Finding #2).
    """
    from TrialMine.agents.tools import _match_excluded_treatments

    with _umls_toggle(True):
        result = _match_excluded_treatments(
            excluded=["HER2 inhibitor"],
            prior_treatments="trastuzumab",
        )
    assert result["verdict"] == "Unmet", result
    assert "matches" in result and result["matches"], result


@pytest.mark.slow
@_requires_kb
def test_toggle_off_uses_literal_match() -> None:
    """Control: same Q413 inputs with the UMLS toggle OFF must NOT
    return Unmet — literal substring match doesn't bridge
    "EGFR tyrosine kinase inhibitor" and "osimertinib".

    Per :func:`_match_excluded_treatments`, when the substring check
    fails, the verdict is ``Met`` (the trial's exclusion is judged
    not to apply). The contract for this control test is just
    ``verdict != "Unmet"``; we accept either ``Met`` or ``Unknown``.
    """
    from TrialMine.agents.tools import _match_excluded_treatments

    with _umls_toggle(False):
        result = _match_excluded_treatments(
            excluded=["EGFR tyrosine kinase inhibitor"],
            prior_treatments="osimertinib",
        )
    assert result["verdict"] != "Unmet", result
    # In practice the current code returns "Met" here, but any non-Unmet
    # verdict satisfies the contract.
    assert result["verdict"] in ("Met", "Unknown"), result
