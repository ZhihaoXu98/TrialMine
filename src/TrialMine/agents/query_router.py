"""Pure routing function: pick the rule arm or the Haiku ReAct agent arm.

Phase A3 of ``docs/build_agent.md`` requires this module to stay PURE —
no IO, no LLM, no module-level state. The LangGraph ``route_decision``
node in :mod:`TrialMine.agents.pipeline` composes this function with the
AB router (Phase A7); the AB-router integration deliberately lives at
the node, not here, so this surface remains trivially testable.

Decision contract (first match wins for ``arm`` / ``reason``):

1. ``config.agentic_path_enabled`` is False
   → ``arm="rule"``, ``reason="agentic_path_disabled"``.
2. ``populated_slots < threshold``
   → ``arm="agent"``, ``reason="sparse_profile"``.
3. complex-pattern heuristic fired (rule 3a OR 3b)
   → ``arm="agent"``, ``reason="complex_pattern"``.
4. Otherwise
   → ``arm="rule"``, ``reason="rule_path_sufficient"``.

``signals`` always carries four keys for the trace store:
``populated_slots`` (int), ``is_multi_constraint`` (bool, profile fact),
``matched_pattern`` (``"failed_X_after_Y"`` / ``"multi_constraint"`` /
``None``), and ``threshold_slots`` (echo of the config).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from TrialMine.agents.query_parser import PatientProfile
from TrialMine.config import DegradationConfig

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #


# Trigger phrasings for the prior-treatment-failure heuristic (rule 3a).
# Word-boundaries on both sides so ``postscript`` / ``aftermath`` do not
# match. ``post[- ]`` covers ``post-`` (e.g. ``post-chemotherapy``) and
# ``post `` (e.g. ``post chemotherapy``); the trailing ``\b`` ensures the
# separator is followed by a word character.
_FAILURE_REGEX = re.compile(
    r"\b(failed|progressed on|refractory to|after|post[- ])\b",
    re.IGNORECASE,
)

# Sub-rule names recorded in ``signals["matched_pattern"]``. These exact
# strings are pinned by ``tests/unit/test_query_router.py`` and by future
# Grafana queries — do NOT rename without updating both.
_PATTERN_FAILURE = "failed_X_after_Y"
_PATTERN_MULTI = "multi_constraint"


# --------------------------------------------------------------------------- #
# Public surface                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RouteDecision:
    """Result of :func:`route_decision`.

    Attributes:
        arm: ``"agent"`` or ``"rule"``.
        reason: One of ``"agentic_path_disabled"``, ``"sparse_profile"``,
            ``"complex_pattern"``, ``"rule_path_sufficient"``.
        signals: Observability dict, always containing
            ``populated_slots`` (int), ``is_multi_constraint`` (bool),
            ``matched_pattern`` (str | None), ``threshold_slots`` (int).
            ``matched_pattern`` records the complex-pattern detection
            even when a different ``reason`` wins the arm decision.
    """

    arm: Literal["agent", "rule"]
    reason: str
    signals: dict


def route_decision(
    profile: PatientProfile,
    config: DegradationConfig,
) -> RouteDecision:
    """Pure routing function. No IO. No LLM.

    Args:
        profile: Parsed patient profile from ``QueryParserAgent``.
        config: Live degradation policy (read fresh each call so the
            outer pipeline can flip toggles without a process restart).

    Returns:
        A :class:`RouteDecision` carrying the arm, reason, and trace
        signals. The signal dict is always populated with the four
        keys documented on :class:`RouteDecision`.
    """
    populated_slots = _count_populated_slots(profile)
    # ``is_multi_constraint`` is a profile fact, not a rule-firing flag —
    # we surface it for trace analysis even when the heuristic toggle is
    # off, because the question "did this query have 3+ constraints?" is
    # answerable from the profile alone.
    is_multi_constraint = _is_multi_constraint(profile)
    threshold = config.agentic_routing_threshold_slots

    # Rule 1 — short-circuit when the agent path is hard-disabled. The
    # signal dict still ships the required four keys so trace consumers
    # have a uniform shape regardless of which arm fires.
    if not config.agentic_path_enabled:
        return RouteDecision(
            arm="rule",
            reason="agentic_path_disabled",
            signals={
                "populated_slots": populated_slots,
                "is_multi_constraint": is_multi_constraint,
                "matched_pattern": None,
                "threshold_slots": threshold,
            },
        )

    # Rule 3 — complex-pattern heuristic. Rule 3a (failure phrasing)
    # takes precedence over rule 3b (multi-constraint) for the
    # ``matched_pattern`` signal when both are eligible.
    matched_pattern: str | None = None
    if config.agentic_complex_pattern_enabled:
        if _failure_phrasing_match(profile):
            matched_pattern = _PATTERN_FAILURE
        elif is_multi_constraint:
            matched_pattern = _PATTERN_MULTI

    signals: dict = {
        "populated_slots": populated_slots,
        "is_multi_constraint": is_multi_constraint,
        "matched_pattern": matched_pattern,
        "threshold_slots": threshold,
    }

    # Rule 4a — sparse profile. Wins the ``reason`` slot over rule 4b
    # when both would fire (cheaper signal, more common trigger).
    if populated_slots < threshold:
        return RouteDecision(arm="agent", reason="sparse_profile", signals=signals)

    # Rule 4b — complex-pattern.
    if matched_pattern is not None:
        return RouteDecision(arm="agent", reason="complex_pattern", signals=signals)

    # Rule 5 — fallthrough to rule arm.
    return RouteDecision(arm="rule", reason="rule_path_sufficient", signals=signals)


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _count_populated_slots(profile: PatientProfile) -> int:
    """Count the 8 ``PatientProfile`` slots that carry a value.

    The 8 slots are the five scalar fields (``condition``,
    ``condition_stage``, ``age``, ``sex``, ``location``) populated when
    non-``None``, plus the three list fields (``prior_treatments``,
    ``biomarkers``, ``preferences``) populated when non-empty.
    ``raw_query`` is required by the parser and always populated, so it
    is NOT counted.
    """
    return sum(
        (
            profile.condition is not None,
            profile.condition_stage is not None,
            profile.age is not None,
            profile.sex is not None,
            profile.location is not None,
            bool(profile.prior_treatments),
            bool(profile.biomarkers),
            bool(profile.preferences),
        )
    )


def _is_multi_constraint(profile: PatientProfile) -> bool:
    """Rule 3b predicate: 3+ of the four 'constraint' slots populated.

    The four constraint slots — ``condition``, ``condition_stage``,
    ``biomarkers``, ``prior_treatments`` — are the ones that
    meaningfully narrow retrieval. Demographic / preference slots
    (``age`` / ``sex`` / ``location`` / ``preferences``) don't count
    because they typically map to soft filters rather than retrieval
    constraints.
    """
    return (
        sum(
            (
                profile.condition is not None,
                profile.condition_stage is not None,
                bool(profile.biomarkers),
                bool(profile.prior_treatments),
            )
        )
        >= 3
    )


def _failure_phrasing_match(profile: PatientProfile) -> bool:
    """Rule 3a predicate: failure regex hits AND ``prior_treatments`` non-empty.

    Both halves are required — ``"failed something"`` on its own is too
    weak a signal without a parsed treatment to anchor it.
    """
    if not profile.prior_treatments:
        return False
    return _FAILURE_REGEX.search(profile.raw_query or "") is not None


__all__ = ["RouteDecision", "route_decision"]
