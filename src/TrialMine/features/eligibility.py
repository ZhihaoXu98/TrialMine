"""Structured eligibility parser for clinical trial criteria text.

Converts free-text ``eligibility_criteria`` plus the structured ``min_age`` /
``max_age`` / ``sex`` columns into an :class:`EligibilityProfile` Pydantic
model with typed condition / treatment buckets and a single heuristic
``parse_confidence``.

Pipeline:
1. Split inclusion/exclusion sections via a four-tier regex fallback.
2. Extract age (column-first, regex fallback) and sex (column-first).
3. Run SciSpacy ``en_core_sci_lg`` on each section, filter boilerplate via a
   stop-list, then bucket the surviving spans into ``conditions`` vs
   ``prior_treatments`` using a keyword heuristic (~70% precision).

The ``parse_confidence`` is a heuristic average of section / age / sex
sub-confidences and is **not** a calibrated probability. Downstream
consumers should treat it as a relative quality signal only.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pydantic output model                                                       #
# --------------------------------------------------------------------------- #


class EligibilityProfile(BaseModel):
    """Structured representation of one trial's eligibility criteria.

    Ages are floats so months survive: ``"6 Months"`` → ``0.5``.

    ``parse_confidence`` is a heuristic mean of three sub-confidences
    (section split, age extraction, sex extraction) on a 0.0–1.0 scale.
    It is *not* a calibrated probability.
    """

    min_age_years: float | None = None
    max_age_years: float | None = None
    sex: Literal["All", "Male", "Female"] | None = None
    required_conditions: list[str] = Field(default_factory=list)
    excluded_conditions: list[str] = Field(default_factory=list)
    required_prior_treatments: list[str] = Field(default_factory=list)
    excluded_prior_treatments: list[str] = Field(default_factory=list)
    raw_inclusion: str = ""
    raw_exclusion: str = ""
    parse_confidence: float = 0.0
    section_source: Literal[
        "headers", "single_header", "variant", "fallback", "empty"
    ] = "fallback"


# --------------------------------------------------------------------------- #
# Regex constants                                                             #
# --------------------------------------------------------------------------- #

# Section headers — both must be found for tier 1.
_INCLUSION_HEADER_RE = re.compile(r"(?i)\binclusion\s+criteria\s*:?")
_EXCLUSION_HEADER_RE = re.compile(r"(?i)\bexclusion\s+criteria\s*:?")

# Variant headers — tier 3 (any one match qualifies).
_VARIANT_HEADER_RES = [
    re.compile(r"(?i)\bdisease\s+characteristics\s*:"),
    re.compile(r"(?i)\bpatient\s+characteristics\s*:"),
    re.compile(r"(?i)\beligibility\s*:"),
    re.compile(r"(?i)\bkey\s+inclusion\b"),
    re.compile(r"(?i)\bpatients\s+are\s+ineligible\s+if\b"),
    re.compile(r"(?i)\bpatients\s+must\s+have\b"),
]

# Age — single value parser used for column ("18 Years", "6 Months", "1 Year").
_AGE_STRING_RE = re.compile(
    r"(?i)^\s*(\d+(?:\.\d+)?)\s*(year|years|month|months|week|weeks|day|days)\s*$"
)

# Range patterns that yield (min, max) directly. Order matters: more specific first.
# Sub-confidence 0.9.
_AGE_RANGE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "between 21 and 75 years"
    (
        re.compile(r"(?i)\bbetween\s+(\d+)\s+and\s+(\d+)\s*(?:years?|y(?:rs?)?)\b"),
        "years",
    ),
    # "6 months to 18 years" — months on the left
    (
        re.compile(r"(?i)\b(\d+)\s*months?\s+to\s+(\d+)\s*years?\b"),
        "months_to_years",
    ),
    # "18 to 75 years" / "18 to 75 years of age"
    (
        re.compile(r"(?i)\b(\d+)\s+to\s+(\d+)\s*(?:years?|y(?:rs?)?)\b"),
        "years",
    ),
    # "Age: 18-65" / "18-75 years"
    (
        re.compile(
            r"(?i)(?:\bage\s*[:\-]?\s*)?\b(\d+)\s*-\s*(\d+)\s*(?:years?|y(?:rs?)?)\b"
        ),
        "years",
    ),
]

# Min-age patterns. Sub-confidence 0.9.
_AGE_MIN_PATTERNS = [
    # ">= 18 years", "≥ 18 years", "\>= 18 years" (escape leak), "Age >= 18"
    re.compile(
        r"(?i)(?:age\s*)?(?:\\)?(?:>=|≥)\s*(\d+)\s*(?:years?|y(?:rs?)?)?"
    ),
    # "at least 18 years"
    re.compile(r"(?i)\bat\s+least\s+(\d+)\s*(?:years?|y(?:rs?)?)\b"),
    # "minimum age 18" / "minimum age of 18"
    re.compile(r"(?i)\bminimum\s+age\s+(?:of\s+)?(\d+)"),
    # "18 years and older" / "18 years or older" / "18 years and above"
    re.compile(
        r"(?i)\b(\d+)\s*(?:years?|y(?:rs?)?)\s*(?:and|or)\s+(?:older|above|greater)"
    ),
    # "Adults (18+)" / "age 18+"
    re.compile(r"(?i)\b(?:adults?|age)\s*\(?(\d+)\+\)?"),
]

# Max-age patterns. Sub-confidence 0.9.
_AGE_MAX_PATTERNS = [
    # "younger than 65"
    re.compile(r"(?i)\byounger\s+than\s+(\d+)"),
    # "<= 65 years", "≤ 65 years", "\<= 65"
    re.compile(
        r"(?i)(?:age\s*)?(?:\\)?(?:<=|≤)\s*(\d+)\s*(?:years?|y(?:rs?)?)?"
    ),
    # "maximum age 75"
    re.compile(r"(?i)\bmaximum\s+age\s+(?:of\s+)?(\d+)"),
]

# Low-precision keyword fallback. Sub-confidence 0.6.
_KEYWORD_ADULT_RE = re.compile(r"(?i)\badults?\b")
_KEYWORD_PEDIATRIC_RE = re.compile(r"(?i)\bpediatric\b|\bchildren\b")

# Sex regex fallback (when sex column is missing).
_SEX_FEMALE_RE = re.compile(
    r"(?i)\b(?:female|women)\s+only\b|\b(?:females?|women)\s+participants?\b"
)
_SEX_MALE_RE = re.compile(
    r"(?i)\b(?:male|men)\s+only\b|\b(?:males?|men)\s+participants?\b"
)

# Treatment-typing heuristic — span text matches → bucket as treatment.
_TREATMENT_RE = re.compile(
    r"(?i)("
    r"therap|chemo|radiat|surger|transplant|inhibit|antibod|vaccin|immunother|"
    r"monoclonal|cytotox|adjuvant|neoadjuvant|infusion|chemotherapy|radiotherapy|"
    r"-mab\b|-nib\b|-inib\b|-zumab\b|-ciclib\b"
    r")"
)

# Boilerplate spans to drop after SciSpacy NER. Lowercased, exact-match.
_STOP_SPANS: frozenset[str] = frozenset(
    {
        # Section headers
        "inclusion criteria",
        "exclusion criteria",
        "criteria",
        # People
        "subject",
        "subjects",
        "patient",
        "patients",
        "participant",
        "participants",
        "investigator",
        "investigators",
        "investigator's",
        "subinvestigator's",
        "male",
        "female",
        "men",
        "women",
        "adult",
        "adults",
        "child",
        "children",
        "infant",
        "infants",
        # Time
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "hour",
        "hours",
        # Study terms
        "study",
        "studies",
        "trial",
        "trials",
        "treatment",
        "treatments",
        "therapy",
        "therapies",
        "drug",
        "drugs",
        "agent",
        "agents",
        "medication",
        "medications",
        "intervention",
        "interventions",
        "regimen",
        # Status / lifecycle
        "active",
        "stable",
        "eligible",
        "ineligible",
        "enrolled",
        "enrollment",
        "history",
        "current",
        "prior",
        "previous",
        "ongoing",
        "completion",
        # Process
        "screening",
        "baseline",
        "informed consent",
        "consent",
        "randomization",
        "randomized",
        "evaluation",
        "assessment",
        "diagnosis",
        "documented",
        "confirmed",
        "approved",
        "prescribed",
        "received",
        "scheduled",
        "visits",
        "visit",
        # Bio process / sample
        "specimens",
        "specimen",
        "tissue",
        "sample",
        "samples",
        "blood",
        "serum",
        "biopsy",
        # Generic
        "age",
        "older",
        "younger",
        "use",
        "uses",
        "used",
        "label",
        "site",
        "sites",
        "study site",
        "site of disease",
        "ulnᅟ",
        "uln",
        "opinion",
        "reasons",
        "consent form",
        "informed consent form",
        "irb/iec",
        "study results",
        "study treatment",
        "laboratory",
        "laboratory tests",
        "laboratory values",
        "requirements",
        "protocol",
        "procedures",
        "regulatory",
        "institutional guidelines",
        "central lab",
        "regimen",
        "marketing authorization",
        "ministry",
        "food",
        "drug safety",
        "korea",
        "korean",
        "prescribing information",
        "indications",
        "therapeutic",
        "indications",
    }
)


# --------------------------------------------------------------------------- #
# Pure functions                                                              #
# --------------------------------------------------------------------------- #


def parse_age_string(value: str | None) -> float | None:
    """Convert a column value like ``"18 Years"`` or ``"6 Months"`` to float years.

    Returns ``None`` if the input is empty or not parseable.

    Examples:
        >>> parse_age_string("18 Years")
        18.0
        >>> parse_age_string("6 Months")
        0.5
        >>> parse_age_string("1 Year")
        1.0
        >>> parse_age_string(None) is None
        True
        >>> parse_age_string("garbage") is None
        True
    """
    if not value:
        return None
    match = _AGE_STRING_RE.match(value)
    if not match:
        return None
    num = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("year"):
        return num
    if unit.startswith("month"):
        return num / 12.0
    if unit.startswith("week"):
        return num / 52.0
    if unit.startswith("day"):
        return num / 365.0
    return None


# --------------------------------------------------------------------------- #
# Parser class                                                                #
# --------------------------------------------------------------------------- #


class EligibilityParser:
    """Stateful parser that lazy-loads SciSpacy on first use.

    Construct once and reuse — the SciSpacy model is ~700 MB and loads in
    ~3 s. For unit tests, pass ``nlp=None`` to skip entity extraction.
    """

    DEFAULT_MODEL = "en_core_sci_lg"

    def __init__(self, nlp: object | None = None, model_name: str | None = None) -> None:
        """Initialise the parser.

        Args:
            nlp: A pre-loaded ``spacy.Language`` instance. If provided, used as-is.
                 If ``None`` and ``model_name`` is also ``None``, entity
                 extraction is disabled and only regex/column-based fields are
                 populated. Useful for fast unit tests.
            model_name: SciSpacy model name to lazy-load if ``nlp`` is None.
                Defaults to ``en_core_sci_lg`` if explicitly enabled.
        """
        self._nlp = nlp
        self._model_name = model_name

    def _load_nlp(self) -> object | None:
        """Lazy-load the SciSpacy model if requested."""
        if self._nlp is not None:
            return self._nlp
        if self._model_name is None:
            return None
        try:
            import spacy  # noqa: WPS433 (lazy import is intentional)

            logger.info("Loading SciSpacy model: %s", self._model_name)
            self._nlp = spacy.load(
                self._model_name,
                disable=["tagger", "lemmatizer", "attribute_ruler", "parser"],
            )
        except Exception as exc:  # broad catch: model load failures shouldn't crash callers
            logger.warning("Failed to load SciSpacy model %s: %s", self._model_name, exc)
            self._nlp = None
        return self._nlp

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(
        self,
        raw_text: str | None,
        *,
        min_age_col: str | None = None,
        max_age_col: str | None = None,
        sex_col: str | None = None,
    ) -> EligibilityProfile:
        """Parse one trial's eligibility text into an :class:`EligibilityProfile`.

        Args:
            raw_text: The ``eligibility_criteria`` free-text from the trial.
                May be ``None`` or empty.
            min_age_col: Value of the ``Trial.min_age`` column (e.g. ``"18 Years"``).
            max_age_col: Value of the ``Trial.max_age`` column.
            sex_col: Value of the ``Trial.sex`` column. Expected
                ``"ALL"`` / ``"MALE"`` / ``"FEMALE"`` (case-insensitive).

        Returns:
            An :class:`EligibilityProfile` with all fields populated as best
            the parser can manage. Always returns a profile, never raises.
        """
        if not raw_text or not raw_text.strip():
            min_age, max_age, age_conf = self._extract_age("", min_age_col, max_age_col)
            sex, sex_conf = self._extract_sex("", sex_col)
            section_conf = 0.0
            return EligibilityProfile(
                min_age_years=min_age,
                max_age_years=max_age,
                sex=sex,
                raw_inclusion="",
                raw_exclusion="",
                section_source="empty",
                parse_confidence=(section_conf + age_conf + sex_conf) / 3.0,
            )

        inclusion, exclusion, section_source = self._split_sections(raw_text)
        section_conf = _SECTION_CONF[section_source]

        min_age, max_age, age_conf = self._extract_age(
            inclusion, min_age_col, max_age_col
        )
        sex, sex_conf = self._extract_sex(raw_text, sex_col)

        req_cond, req_treat = self._extract_entities(inclusion)
        excl_cond, excl_treat = self._extract_entities(exclusion)

        confidence = (section_conf + age_conf + sex_conf) / 3.0

        return EligibilityProfile(
            min_age_years=min_age,
            max_age_years=max_age,
            sex=sex,
            required_conditions=req_cond,
            excluded_conditions=excl_cond,
            required_prior_treatments=req_treat,
            excluded_prior_treatments=excl_treat,
            raw_inclusion=inclusion,
            raw_exclusion=exclusion,
            parse_confidence=round(confidence, 3),
            section_source=section_source,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _split_sections(
        self, text: str
    ) -> tuple[str, str, Literal["headers", "single_header", "variant", "fallback"]]:
        """Split free text into (inclusion, exclusion, source).

        Priority:
          1. Both ``Inclusion Criteria:`` and ``Exclusion Criteria:`` headers → headers.
          2. One of them → single_header (other side is empty).
          3. Variant header (e.g. ``DISEASE CHARACTERISTICS:``) → variant
             (entire text → inclusion).
          4. Otherwise → fallback (entire text → inclusion).
        """
        inc_match = _INCLUSION_HEADER_RE.search(text)
        exc_match = _EXCLUSION_HEADER_RE.search(text)

        if inc_match and exc_match:
            inc_start = inc_match.end()
            exc_start = exc_match.start()
            if inc_start < exc_start:
                inclusion = text[inc_start:exc_start].strip()
                exclusion = text[exc_match.end() :].strip()
            else:
                # Unusual ordering: exclusion before inclusion. Take both segments.
                inclusion = text[inc_start:].strip()
                exclusion = text[exc_match.end() : inc_match.start()].strip()
            return inclusion, exclusion, "headers"

        if inc_match:
            inclusion = text[inc_match.end() :].strip()
            return inclusion, "", "single_header"

        if exc_match:
            # Inclusion is whatever precedes the exclusion header.
            inclusion = text[: exc_match.start()].strip()
            exclusion = text[exc_match.end() :].strip()
            return inclusion, exclusion, "single_header"

        for variant_re in _VARIANT_HEADER_RES:
            if variant_re.search(text):
                return text.strip(), "", "variant"

        return text.strip(), "", "fallback"

    def _extract_age(
        self,
        inclusion: str,
        min_age_col: str | None,
        max_age_col: str | None,
    ) -> tuple[float | None, float | None, float]:
        """Return (min_years, max_years, sub_confidence)."""
        col_min = parse_age_string(min_age_col)
        col_max = parse_age_string(max_age_col)

        if col_min is not None or col_max is not None:
            return col_min, col_max, 1.0

        if not inclusion:
            return None, None, 0.0

        # Try range patterns first — they yield both min and max.
        for pattern, kind in _AGE_RANGE_PATTERNS:
            m = pattern.search(inclusion)
            if m:
                lo = float(m.group(1))
                hi = float(m.group(2))
                if kind == "months_to_years":
                    lo = lo / 12.0
                return lo, hi, 0.9

        min_age: float | None = None
        max_age: float | None = None

        for pattern in _AGE_MIN_PATTERNS:
            m = pattern.search(inclusion)
            if m:
                min_age = float(m.group(1))
                break

        for pattern in _AGE_MAX_PATTERNS:
            m = pattern.search(inclusion)
            if m:
                max_age = float(m.group(1))
                break

        if min_age is not None or max_age is not None:
            return min_age, max_age, 0.9

        # Low-precision keyword fallback.
        if _KEYWORD_PEDIATRIC_RE.search(inclusion):
            return None, 18.0, 0.6
        if _KEYWORD_ADULT_RE.search(inclusion):
            return 18.0, None, 0.6

        return None, None, 0.0

    def _extract_sex(
        self, text: str, sex_col: str | None
    ) -> tuple[Literal["All", "Male", "Female"] | None, float]:
        """Return (sex, sub_confidence). Trusts the column when present."""
        if sex_col:
            normalised = sex_col.strip().upper()
            if normalised == "ALL":
                return "All", 1.0
            if normalised == "MALE":
                return "Male", 1.0
            if normalised == "FEMALE":
                return "Female", 1.0

        if _SEX_FEMALE_RE.search(text):
            return "Female", 0.8
        if _SEX_MALE_RE.search(text):
            return "Male", 0.8

        return "All", 0.5

    def _extract_entities(self, section_text: str) -> tuple[list[str], list[str]]:
        """Run NER on a section, filter noise, type into (conditions, treatments).

        Returns (conditions, prior_treatments). Both empty if NER is disabled
        or the section text is blank.
        """
        if not section_text or not section_text.strip():
            return [], []
        nlp = self._load_nlp()
        if nlp is None:
            return [], []

        try:
            doc = nlp(section_text)
        except Exception as exc:  # NER failures shouldn't bring the parser down
            logger.warning("SciSpacy NER failed: %s", exc)
            return [], []

        conditions: list[str] = []
        treatments: list[str] = []
        seen: set[str] = set()

        for ent in doc.ents:
            text = ent.text.strip().rstrip(".,;:")
            if not (2 <= len(text) <= 60):
                continue
            lowered = text.lower()
            if lowered in _STOP_SPANS:
                continue
            if all((not c.isalpha()) for c in text):
                continue
            if all(token.is_stop for token in ent if hasattr(token, "is_stop")):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)

            if _TREATMENT_RE.search(text):
                treatments.append(text)
            else:
                conditions.append(text)

        return conditions, treatments


# Confidence weights for section split tiers.
_SECTION_CONF: dict[str, float] = {
    "headers": 1.0,
    "single_header": 0.8,
    "variant": 0.6,
    "fallback": 0.3,
    "empty": 0.0,
}


# --------------------------------------------------------------------------- #
# Backwards-compatible compute_eligibility_features stub                      #
# --------------------------------------------------------------------------- #


def compute_eligibility_features(trial: dict, patient_profile: dict) -> dict:
    """Compute eligibility match features for a (trial, patient) pair.

    Not implemented in v1 — patient-vs-trial matching is a separate downstream
    layer. This stub remains so callers and ranker integration can stub it.
    """
    raise NotImplementedError("Patient-vs-trial matching layer not yet implemented")
