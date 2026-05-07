"""Unit tests for :class:`TrialMine.agents.query_parser.QueryParserAgent`.

The Anthropic SDK call is mocked so these tests run without an API key
and without network. Validates:

1. Successful parse with all fields populated.
2. Successful parse with only some fields populated; missing slots default
   to ``None`` / ``[]``.
3. Empty query → skeleton profile (no API call attempted).
4. SDK raises (e.g. malformed JSON, validation error, API error) → graceful
   fallback profile carrying only ``raw_query``.
5. Construction without ``ANTHROPIC_API_KEY`` raises a clear ``ValueError``
   so misconfiguration surfaces immediately rather than at first request.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from TrialMine.agents.query_parser import (
    PatientProfile,
    QueryParserAgent,
    _ExtractedFields,
)


def _make_parser_with_mocked_client(
    parsed: Any | None,
    raise_exc: BaseException | None = None,
) -> QueryParserAgent:
    """Construct a parser whose ``_client.messages.parse`` is a Mock.

    Args:
        parsed: Object to return as ``response.parsed_output`` (typically an
            ``_ExtractedFields`` instance). Ignored when ``raise_exc`` is set.
        raise_exc: When set, the mock raises this exception instead of returning.
    """
    parser = QueryParserAgent(api_key="sk-test-not-used")
    fake_response = MagicMock()
    fake_response.parsed_output = parsed
    fake_response.usage.input_tokens = 1370
    fake_response.usage.output_tokens = 50
    fake_response.usage.cache_read_input_tokens = 0
    if raise_exc is not None:
        parser._client.messages.parse = MagicMock(side_effect=raise_exc)
    else:
        parser._client.messages.parse = MagicMock(return_value=fake_response)
    return parser


def test_parse_with_all_fields_populated() -> None:
    """A fully-populated extraction round-trips into the public PatientProfile."""
    extracted = _ExtractedFields(
        condition="metastatic prostate cancer",
        condition_stage="metastatic",
        age=68,
        sex="Male",
        prior_treatments=["docetaxel", "abiraterone"],
        biomarkers=["AR-V7"],
        preferences=["immunotherapy"],
        location="Boston",
    )
    parser = _make_parser_with_mocked_client(parsed=extracted)
    profile = parser.parse("68 yo man with metastatic prostate cancer in Boston")

    assert isinstance(profile, PatientProfile)
    assert profile.condition == "metastatic prostate cancer"
    assert profile.age == 68
    assert profile.sex == "Male"
    assert profile.prior_treatments == ["docetaxel", "abiraterone"]
    assert profile.biomarkers == ["AR-V7"]
    assert profile.preferences == ["immunotherapy"]
    assert profile.location == "Boston"
    assert profile.raw_query.startswith("68 yo man")


def test_parse_with_missing_fields_defaults_remaining_to_none() -> None:
    """Sparse extraction → unmentioned slots stay None / empty."""
    extracted = _ExtractedFields(condition="lung cancer")  # everything else default
    parser = _make_parser_with_mocked_client(parsed=extracted)
    profile = parser.parse("lung cancer trials")

    assert profile.condition == "lung cancer"
    assert profile.condition_stage is None
    assert profile.age is None
    assert profile.sex is None
    assert profile.prior_treatments == []
    assert profile.biomarkers == []
    assert profile.preferences == []
    assert profile.location is None
    assert profile.raw_query == "lung cancer trials"


def test_parse_empty_query_returns_skeleton_without_calling_api() -> None:
    """Empty / whitespace-only query short-circuits before the API call."""
    parser = _make_parser_with_mocked_client(parsed=None)
    profile = parser.parse("")
    assert profile.raw_query == ""
    assert profile.condition is None
    parser._client.messages.parse.assert_not_called()  # type: ignore[attr-defined]

    profile2 = parser.parse("   \t\n  ")
    assert profile2.condition is None
    parser._client.messages.parse.assert_not_called()  # type: ignore[attr-defined]


def test_parse_sdk_failure_falls_back_to_raw_query_only() -> None:
    """Any SDK error (malformed JSON, validation, API outage) must NOT raise.

    Contract: the agent pipeline downstream must always receive a
    PatientProfile; the parser swallows its own errors and returns one
    populated only with ``raw_query``.
    """
    # Simulate the kind of failure messages.parse raises on bad JSON / refusal
    parser = _make_parser_with_mocked_client(
        parsed=None, raise_exc=ValueError("bad JSON: trailing comma")
    )
    profile = parser.parse("breast cancer trials")
    assert isinstance(profile, PatientProfile)
    assert profile.raw_query == "breast cancer trials"
    assert profile.condition is None
    assert profile.prior_treatments == []


def test_construction_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``ANTHROPIC_API_KEY`` env var with no override → ValueError.

    This is a fail-fast contract: the parser is instantiated at app startup,
    and we'd rather break loudly there than have every search return a
    silent fallback profile because the key was never set.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY missing"):
        QueryParserAgent()
