"""Unit tests for the pure routing function in ``query_router.py``.

The runbook (Phase A3 of ``docs/build_agent.md``) requires the routing
module to stay PURE — no IO, no LLM, no module-level state. These tests
pin the public surface (``RouteDecision`` shape + the four reason
strings) so the AB-router LangGraph node added in Phase A7 can compose
this function safely.

Each test maps to one of the seven cases the runbook enumerates, plus a
handful of guard-rails for edge cases (threshold boundary, regex
case-insensitivity, signal shape under every branch).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from TrialMine.agents.query_parser import PatientProfile
from TrialMine.agents.query_router import route_decision
from TrialMine.config import DegradationConfig

# Reusable enabled-path config so individual tests don't have to repeat
# the boilerplate. Tests that exercise the disabled-path invariant build
# their own ``DegradationConfig()`` to be explicit about the default.
_ENABLED = DegradationConfig(agentic_path_enabled=True)


def _profile(**kwargs: object) -> PatientProfile:
    """Build a :class:`PatientProfile` with ``raw_query`` defaulted to ''."""
    kwargs.setdefault("raw_query", "")
    return PatientProfile(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Rule 1 — Phase A invariant                                                  #
# --------------------------------------------------------------------------- #


def test_agentic_path_disabled_forces_rule_arm() -> None:
    """Runbook case #1. With ``agentic_path_enabled=False`` (explicit),
    NO heuristic may fire — the Phase-A-through-D0 invariant. The
    profile below would otherwise hit both sparse_profile and
    complex_pattern; this test pins the short-circuit. Phase D1 shipped
    with ``agentic_path_enabled=True`` as the new production default,
    so the test now explicitly disables it to exercise the
    short-circuit branch."""
    profile = _profile(
        raw_query="failed osimertinib EGFR T790M",
        prior_treatments=["osimertinib"],
        biomarkers=["EGFR T790M"],
        condition="non-small cell lung cancer",
    )
    config = DegradationConfig(agentic_path_enabled=False)
    decision = route_decision(profile, config)
    assert decision.arm == "rule"
    assert decision.reason == "agentic_path_disabled"


# --------------------------------------------------------------------------- #
# Rule 4a — sparse profile                                                    #
# --------------------------------------------------------------------------- #


def test_zero_populated_slots_routes_to_agent_sparse() -> None:
    """Runbook case #2. Empty profile (only ``raw_query`` set) has 0
    populated slots; with default threshold 2, 0 < 2 → agent /
    sparse_profile."""
    profile = _profile(raw_query="trials for my dad")
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "agent"
    assert decision.reason == "sparse_profile"
    assert decision.signals["populated_slots"] == 0
    assert decision.signals["threshold_slots"] == 2


def test_custom_threshold_boundary_is_strict_less_than() -> None:
    """At threshold=4 with exactly 4 populated slots, sparse must NOT
    fire (the rule is strict ``<``, not ``<=``)."""
    profile = _profile(
        raw_query="lung cancer trials in Boston",
        condition="lung cancer",
        age=60,
        sex="Male",
        location="Boston",
    )  # 4 populated slots; only 1 of 4 constraint slots
    config = DegradationConfig(
        agentic_path_enabled=True,
        agentic_routing_threshold_slots=4,
    )
    decision = route_decision(profile, config)
    assert decision.arm == "rule"
    assert decision.reason == "rule_path_sufficient"
    assert decision.signals["populated_slots"] == 4


# --------------------------------------------------------------------------- #
# Rule 3 / 4b — complex-pattern heuristic                                     #
# --------------------------------------------------------------------------- #


def test_failure_phrasing_with_prior_treatment_routes_to_agent() -> None:
    """Runbook case #4. ``failed osimertinib`` + ``prior_treatments``
    non-empty satisfies rule 3a. The extra slots keep ``populated_slots``
    above the threshold so sparse_profile doesn't pre-empt the test."""
    profile = _profile(
        raw_query="failed osimertinib EGFR T790M",
        condition="non-small cell lung cancer",
        prior_treatments=["osimertinib"],
    )  # 2 slots populated (not sparse with default threshold=2)
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "agent"
    assert decision.reason == "complex_pattern"
    assert decision.signals["matched_pattern"] == "failed_X_after_Y"


def test_failure_phrasing_without_prior_treatments_does_not_match_3a() -> None:
    """Rule 3a needs BOTH the regex match AND non-empty
    ``prior_treatments``. ``failed something`` alone — with no parsed
    treatment to anchor the failure to — does not fire."""
    profile = _profile(
        raw_query="failed something",
        condition="lung cancer",
        age=65,
        sex="Male",
        location="Boston",
    )  # 4 slots (not sparse), no prior_treatments
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "rule"
    assert decision.reason == "rule_path_sufficient"
    assert decision.signals["matched_pattern"] is None


def test_multi_constraint_routes_to_agent() -> None:
    """Runbook case #5. 3+ of {condition, condition_stage, biomarkers,
    prior_treatments} populated, no failure phrasing → rule 3b fires."""
    profile = _profile(
        raw_query="EGFR T790M NSCLC with prior chemo",
        condition="non-small cell lung cancer",
        biomarkers=["EGFR T790M"],
        prior_treatments=["chemotherapy"],
    )  # 3 of 4 constraint slots populated; raw_query has no failure regex
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "agent"
    assert decision.reason == "complex_pattern"
    assert decision.signals["matched_pattern"] == "multi_constraint"
    assert decision.signals["is_multi_constraint"] is True


def test_failure_phrasing_wins_over_multi_constraint_for_matched_pattern() -> None:
    """When BOTH rule 3a and 3b are eligible, ``matched_pattern``
    records ``failed_X_after_Y`` (3a takes precedence in the signal),
    while ``is_multi_constraint`` independently reports the 3b fact."""
    profile = _profile(
        raw_query="failed osimertinib, EGFR T790M, prior chemo",
        condition="NSCLC",
        biomarkers=["EGFR T790M"],
        prior_treatments=["osimertinib", "chemotherapy"],
    )
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "agent"
    assert decision.reason == "complex_pattern"
    assert decision.signals["matched_pattern"] == "failed_X_after_Y"
    assert decision.signals["is_multi_constraint"] is True


@pytest.mark.parametrize(
    "phrase",
    [
        "failed osimertinib",
        "Failed Osimertinib",  # case-insensitive
        "progressed on Keytruda",
        "refractory to FOLFOX",
        "after trastuzumab",
        "post-chemotherapy regimen",
        "post chemo regimen",
    ],
)
def test_all_failure_phrasings_match_regex(phrase: str) -> None:
    """Every trigger phrasing in the runbook regex hits, case-insensitively.
    A parsed ``prior_treatments`` is required so rule 3a (not just the
    regex) fires end-to-end."""
    profile = _profile(
        raw_query=phrase,
        condition="cancer",
        prior_treatments=["something"],
    )
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "agent"
    assert decision.signals["matched_pattern"] == "failed_X_after_Y"


def test_eight_slots_no_complex_pattern_routes_to_rule() -> None:
    """Runbook case #3. With ``complex_pattern_enabled=False`` AND no
    sparse trigger, even a fully-populated profile routes to rule. (With
    the default toggle on, 8 populated slots would always satisfy rule
    3b — so 'no complex pattern' here means 'the complex heuristic is
    off'.)"""
    profile = _profile(
        raw_query="lung cancer trials, recently diagnosed",
        condition="non-small cell lung cancer",
        condition_stage="stage IV",
        age=64,
        sex="Male",
        location="Boston",
        prior_treatments=["chemotherapy"],
        biomarkers=["EGFR T790M"],
        preferences=["immunotherapy"],
    )
    config = DegradationConfig(
        agentic_path_enabled=True,
        agentic_complex_pattern_enabled=False,
    )
    decision = route_decision(profile, config)
    assert decision.arm == "rule"
    assert decision.reason == "rule_path_sufficient"
    assert decision.signals["populated_slots"] == 8
    assert decision.signals["matched_pattern"] is None


# --------------------------------------------------------------------------- #
# Rule 4 — preference ordering                                                #
# --------------------------------------------------------------------------- #


def test_sparse_preferred_over_complex_pattern_when_both_fire() -> None:
    """Runbook tie-break: if BOTH sparse_profile and complex_pattern
    would fire, sparse wins for ``reason`` (it's cheaper to compute and
    the more common trigger). ``matched_pattern`` still records the
    complex-pattern signal — the heuristic detection is independent of
    which arm-reason ultimately wins."""
    profile = _profile(
        raw_query="failed osimertinib",
        prior_treatments=["osimertinib"],
    )  # 1 slot populated → sparse fires; failure phrasing → 3a also fires
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "agent"
    assert decision.reason == "sparse_profile"
    assert decision.signals["matched_pattern"] == "failed_X_after_Y"


# --------------------------------------------------------------------------- #
# Rule 5 — rule-arm fallthrough                                               #
# --------------------------------------------------------------------------- #


def test_no_signal_fires_routes_to_rule_arm() -> None:
    """No sparse, no failure phrasing, only 1 of 4 constraint slots
    populated → none of the heuristics fire → rule arm."""
    profile = _profile(
        raw_query="breast cancer trials in Boston for women in their 50s",
        condition="breast cancer",
        age=55,
        sex="Female",
        location="Boston",
        preferences=["immunotherapy"],
    )  # 5 slots; only condition is in the constraint set
    decision = route_decision(profile, _ENABLED)
    assert decision.arm == "rule"
    assert decision.reason == "rule_path_sufficient"
    assert decision.signals["matched_pattern"] is None
    assert decision.signals["is_multi_constraint"] is False


def test_complex_pattern_disabled_blocks_only_complex_heuristic() -> None:
    """Runbook case #6. Disabling ``agentic_complex_pattern_enabled``
    blocks rule 3 but must not affect sparse_profile or false-trigger
    routing when the profile is rich. ``is_multi_constraint`` still
    reports the profile fact — it's an observability signal independent
    of whether the heuristic fired."""
    profile = _profile(
        raw_query="failed osimertinib EGFR T790M",
        condition="NSCLC",
        condition_stage="stage IV",
        biomarkers=["EGFR T790M"],
        prior_treatments=["osimertinib"],
        age=60,
        sex="Male",
        location="Boston",
        preferences=["targeted therapy"],
    )  # 8 populated slots — definitely not sparse
    config = DegradationConfig(
        agentic_path_enabled=True,
        agentic_complex_pattern_enabled=False,
    )
    decision = route_decision(profile, config)
    assert decision.arm == "rule"
    assert decision.reason == "rule_path_sufficient"
    assert decision.signals["matched_pattern"] is None
    # Profile fact is reported regardless of toggle state.
    assert decision.signals["is_multi_constraint"] is True


# --------------------------------------------------------------------------- #
# Signal-shape contract (runbook case #7)                                     #
# --------------------------------------------------------------------------- #


_EXPECTED_SIGNAL_KEYS = {
    "populated_slots",
    "is_multi_constraint",
    "matched_pattern",
    "threshold_slots",
}


@pytest.mark.parametrize(
    "label, profile, config",
    [
        (
            "disabled",
            PatientProfile(raw_query="x"),
            DegradationConfig(),  # rule 1
        ),
        (
            "sparse",
            PatientProfile(raw_query="x"),
            DegradationConfig(agentic_path_enabled=True),  # rule 4a
        ),
        (
            "complex_3a",
            PatientProfile(
                raw_query="failed osimertinib",
                condition="lung",
                prior_treatments=["osimertinib"],
            ),
            DegradationConfig(agentic_path_enabled=True),
        ),
        (
            "complex_3b",
            PatientProfile(
                raw_query="x",
                condition="lung",
                biomarkers=["EGFR"],
                prior_treatments=["chemo"],
            ),
            DegradationConfig(agentic_path_enabled=True),
        ),
        (
            "rule_fallthrough",
            PatientProfile(
                raw_query="x",
                condition="lung",
                age=60,
                sex="Male",
                location="Boston",
            ),
            DegradationConfig(agentic_path_enabled=True),
        ),
    ],
)
def test_signals_always_contain_required_keys(
    label: str,
    profile: PatientProfile,
    config: DegradationConfig,
) -> None:
    """Every decision (regardless of arm or short-circuit) MUST expose
    the four signal keys with their declared types — the trace store
    indexes every row by this shape."""
    decision = route_decision(profile, config)
    assert set(decision.signals.keys()) >= _EXPECTED_SIGNAL_KEYS, label
    assert isinstance(decision.signals["populated_slots"], int), label
    assert isinstance(decision.signals["is_multi_constraint"], bool), label
    assert isinstance(decision.signals["threshold_slots"], int), label
    mp = decision.signals["matched_pattern"]
    assert mp is None or isinstance(mp, str), label


# --------------------------------------------------------------------------- #
# Dataclass-level invariants                                                  #
# --------------------------------------------------------------------------- #


def test_route_decision_is_frozen() -> None:
    """``RouteDecision`` is ``frozen=True`` per the runbook surface so
    accidental mutation downstream is caught at write-time."""
    decision = route_decision(_profile(raw_query="x"), DegradationConfig())
    with pytest.raises((FrozenInstanceError, AttributeError)):
        decision.arm = "agent"  # type: ignore[misc]


def test_route_decision_arm_literal_values() -> None:
    """Sanity: arm is always one of the two declared Literals."""
    decision = route_decision(_profile(raw_query="x"), DegradationConfig())
    assert decision.arm in {"agent", "rule"}


# --------------------------------------------------------------------------- #
# Module purity (no IO / no side effects)                                     #
# --------------------------------------------------------------------------- #


def test_route_decision_module_has_no_module_level_state() -> None:
    """The runbook's Phase A invariant: the pure function may not
    accumulate state across calls. Two identical inputs must yield
    structurally-equal RouteDecisions."""
    profile = _profile(raw_query="x", condition="lung cancer")
    d1 = route_decision(profile, _ENABLED)
    d2 = route_decision(profile, _ENABLED)
    assert d1 == d2
