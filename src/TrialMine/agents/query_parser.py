"""LangGraph node: QueryParser.

Transforms a free-text patient description into a structured
:class:`PatientProfile` using Claude with native structured output.

Uses the Anthropic SDK's ``messages.parse()`` for reliable Pydantic-typed
slot extraction. On any failure (API error, schema validation, timeout,
refusal) returns a profile carrying only ``raw_query`` so the downstream
retrieval pipeline never breaks.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Cheapest current Haiku — fast and plenty accurate for slot extraction.
# Older Haiku 3 (claude-3-haiku-20240307) retires April 2026 and should not be used.
DEFAULT_MODEL = "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# Output schema                                                               #
# --------------------------------------------------------------------------- #


class PatientProfile(BaseModel):
    """Structured patient query — output of :class:`QueryParserAgent`.

    All slots default to ``None`` / empty so partial extraction works.
    ``raw_query`` is always populated; downstream retrieval can fall back
    to BM25 over the original text if structured fields are sparse.
    """

    condition: str | None = Field(
        None,
        description=(
            "Primary cancer / disease in standard medical terminology. "
            "Apply lay→medical mapping when obvious "
            "('stomach cancer' → 'gastric cancer'). "
            "Do NOT narrow to subtypes the patient did not name."
        ),
    )
    condition_stage: str | None = Field(
        None,
        description=(
            "Disease stage as explicitly stated: 'stage 3', 'stage IV', "
            "'metastatic', 'advanced', 'recurrent', 'early-stage'. "
            "Null otherwise."
        ),
    )
    age: int | None = Field(
        None,
        ge=0,
        le=130,
        description=(
            "Patient age in years; only when a specific number is given."
        ),
    )
    sex: Literal["Male", "Female"] | None = Field(
        None,
        description=(
            "Patient sex: 'Male' or 'Female'. Set only when explicitly "
            "identified or named via a clearly gendered noun (e.g. "
            "'my daughter' → Female)."
        ),
    )
    prior_treatments: list[str] = Field(
        default_factory=list,
        description=(
            "Specific drugs, named regimens, or therapy classes already "
            "received (e.g. 'carboplatin', 'pembrolizumab', 'radiation', "
            "'CAR-T', 'tamoxifen', 'chemotherapy')."
        ),
    )
    biomarkers: list[str] = Field(
        default_factory=list,
        description=(
            "Mutations, receptors, or molecular markers explicitly named "
            "(e.g. 'MSI-high', 'HER2-positive', 'EGFR mutation', 'BRCA1', "
            "'PD-L1 high', 'triple negative')."
        ),
    )
    preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Trial features the patient prefers (e.g. 'immunotherapy', "
            "'phase 3', 'pembrolizumab trials', 'targeted therapy', "
            "'near me')."
        ),
    )
    location: str | None = Field(
        None,
        description=(
            "Geographic preference if explicitly named: city, state, "
            "region, or country. 'Near me' is a preference, not a location."
        ),
    )
    raw_query: str = Field(
        ...,
        description=(
            "Original patient query, verbatim. Always populated by the "
            "agent — the LLM does not need to echo this back."
        ),
    )


class _ExtractedFields(BaseModel):
    """Internal LLM-output schema (everything except ``raw_query``).

    Splitting this off keeps the LLM from echoing the input back through
    its output tokens. The agent attaches ``raw_query`` after parsing.
    """

    condition: str | None = None
    condition_stage: str | None = None
    age: int | None = Field(None, ge=0, le=130)
    sex: Literal["Male", "Female"] | None = None
    prior_treatments: list[str] = Field(default_factory=list)
    biomarkers: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    location: str | None = None


# --------------------------------------------------------------------------- #
# System prompt                                                               #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a medical query parser for a clinical trial search engine.
Extract structured information from a patient's free-text query.

RULES (strict — disagreements resolve toward null/empty, not toward inference):

1. ONLY extract what is EXPLICITLY stated. Never assume or speculate.
2. If a slot is not mentioned, return null (or empty list for list-typed slots).
3. 'condition': use standard medical terminology when the patient's word has
   an unambiguous medical equivalent (e.g. 'stomach cancer' → 'gastric cancer';
   'ALL leukemia' → 'acute lymphoblastic leukemia'). Common typos may be
   corrected ('ostate cancer' → 'prostate cancer'). Do NOT narrow to subtypes
   the patient did NOT name (do NOT map 'liver cancer' → 'hepatocellular
   carcinoma'; do NOT map 'lung cancer' → 'NSCLC').
4. 'condition_stage': only when explicit. Counts: 'stage 3', 'stage IV',
   'metastatic', 'advanced', 'recurrent', 'early-stage'. Does NOT count:
   'serious', 'aggressive', 'late', 'bad'.
5. 'age': only when a specific number is given. 'In my 60s' is not specific.
6. 'sex': only when the patient is explicitly identified, OR when a clearly
   gendered noun names the patient ('my daughter is 8' → Female, age 8;
   'my son was diagnosed' → Male). Stand-alone pronouns 'he'/'she' on a
   vague subject do NOT qualify.
7. 'prior_treatments': specific drugs, named regimens, or therapy classes
   ('carboplatin', 'pembrolizumab', 'radiation', 'CAR-T', 'tamoxifen',
   'chemotherapy', 'finished chemo' → 'chemotherapy'). Generic words alone
   like 'treatment' or 'medicine' do NOT qualify.
8. 'biomarkers': molecular markers, mutations, receptors explicitly named
   ('MSI-high', 'HER2-positive', 'EGFR mutation', 'BRCA1', 'PD-L1 high',
   'triple negative'). Empty list if none stated.
9. 'preferences': trial features the patient says they want
   ('immunotherapy', 'phase 3', 'pembrolizumab trials', 'targeted therapy',
   'CAR-T', 'near me').
10. 'location': city, state, region, or country, only if explicitly named
    ('Boston', 'Texas', 'New England'). 'Near me' is a preference, not a
    location.

When in doubt, prefer null over guessing."""


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class QueryParserAgent:
    """Parse free-text patient queries into :class:`PatientProfile`.

    Uses the Anthropic SDK's ``messages.parse()`` with a Pydantic schema
    for reliable typed extraction. On any failure — API error, schema
    validation, timeout, refusal — returns a profile containing only the
    raw query so downstream retrieval can still operate.
    """

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_tokens: int = 1024,
    ) -> None:
        """Initialise the parser with an Anthropic client.

        Args:
            model: Anthropic model ID. Defaults to ``claude-haiku-4-5`` —
                cheap and fast for slot extraction.
            api_key: Override for ``ANTHROPIC_API_KEY``; reads env var if None.
            timeout: Per-request timeout in seconds.
            max_tokens: Output token cap. Slot extraction is short; 1024
                is generous and avoids ``stop_reason='max_tokens'`` truncation.

        Raises:
            ValueError: If no API key is available.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY missing. Set the env var or pass api_key=."
            )

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def parse(self, raw_query: str) -> PatientProfile:
        """Parse a free-text query into a :class:`PatientProfile`.

        Always returns a profile; ``raw_query`` is always populated. On
        any error, returns a profile with only ``raw_query`` set.

        Args:
            raw_query: Free-text patient query.

        Returns:
            :class:`PatientProfile` with as many slots populated as the
            LLM was able to extract.
        """
        if not raw_query or not raw_query.strip():
            logger.info("QueryParser: empty input, returning skeleton profile")
            return PatientProfile(raw_query=raw_query or "")

        t0 = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        # No-op until the prompt grows past the model's
                        # cache-prefix threshold, but harmless to mark.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": raw_query}],
                output_format=_ExtractedFields,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            extracted: _ExtractedFields = response.parsed_output
            profile = PatientProfile(
                **extracted.model_dump(),
                raw_query=raw_query,
            )

            logger.info(
                "QueryParser parsed in %.0f ms (in=%d, out=%d, cache_read=%d): "
                "condition=%r stage=%r age=%s sex=%s",
                elapsed_ms,
                response.usage.input_tokens,
                response.usage.output_tokens,
                getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                profile.condition,
                profile.condition_stage,
                profile.age,
                profile.sex,
            )
            return profile

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "QueryParser failed in %.0f ms (%s: %s); "
                "returning raw_query-only profile",
                elapsed_ms,
                type(exc).__name__,
                exc,
            )
            return PatientProfile(raw_query=raw_query)


__all__ = [
    "PatientProfile",
    "QueryParserAgent",
    "DEFAULT_MODEL",
    "SYSTEM_PROMPT",
]
