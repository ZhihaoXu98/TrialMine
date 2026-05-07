"""Unit tests for :mod:`TrialMine.experiments.ab_test`.

Pure-function tests — no I/O, no models. Validates the deterministic
contract that ranking and agent code will rely on once experiments are
wired in.
"""

from __future__ import annotations

import pytest

from TrialMine.experiments import ABTestRouter, Experiment, ExperimentVariant


def _two_variant_router(weight_b: float = 0.5) -> ABTestRouter:
    """50/50 router by default; weight_b shifts the boundary."""
    return ABTestRouter(
        [
            Experiment(
                name="exp",
                variants=[
                    ExperimentVariant(name="A", weight=1.0 - weight_b),
                    ExperimentVariant(name="B", weight=weight_b),
                ],
            )
        ]
    )


def test_route_is_deterministic_for_same_subject() -> None:
    """Same subject + same router → same variant on every call."""
    router = _two_variant_router()
    first = router.route("user-42", "exp")
    for _ in range(5):
        assert router.route("user-42", "exp") == first


def test_route_distribution_matches_weights() -> None:
    """Over many subjects the assignment mass should track the weights.

    1000 subjects × 80/20 split: assert variant B lands within ±5 pp of
    its 0.20 target. The hash is uniform on its input distribution; this
    is a sanity check, not a statistical proof.
    """
    router = _two_variant_router(weight_b=0.20)
    counts = {"A": 0, "B": 0}
    for i in range(1000):
        counts[router.route(f"user-{i}", "exp")] += 1
    share_b = counts["B"] / 1000
    assert 0.15 <= share_b <= 0.25, f"variant B share = {share_b:.3f}"


def test_disabled_experiment_falls_to_first_variant() -> None:
    """``enabled=False`` short-circuits routing — every subject lands in variant 0."""
    router = ABTestRouter(
        [
            Experiment(
                name="off",
                enabled=False,
                variants=[
                    ExperimentVariant(name="control", weight=0.5),
                    ExperimentVariant(name="treatment", weight=0.5),
                ],
            )
        ]
    )
    for i in range(50):
        assert router.route(f"user-{i}", "off") == "control"


def test_log_exposure_records_event_and_returns_copy() -> None:
    """log_exposure must persist the (subject, experiment, variant) tuple."""
    router = _two_variant_router()
    router.log_exposure("user-1", "exp", "A")
    router.log_exposure("user-2", "exp", "B")
    exposures = router.exposures
    assert len(exposures) == 2
    assert exposures[0] == {"subject_id": "user-1", "experiment": "exp", "variant": "A"}
    assert exposures[1] == {"subject_id": "user-2", "experiment": "exp", "variant": "B"}
    # ``exposures`` returns a defensive copy — mutating it must not alter state.
    exposures.clear()
    assert len(router.exposures) == 2


def test_unknown_experiment_in_route_raises_keyerror() -> None:
    """Unknown experiment name in route() is a programming error — surface it loudly."""
    router = _two_variant_router()
    with pytest.raises(KeyError):
        router.route("u", "missing")


def test_duplicate_experiment_name_rejected_at_construction() -> None:
    """Two experiments sharing a name would silently shadow — reject at build time."""
    with pytest.raises(ValueError, match="duplicate experiment name"):
        ABTestRouter(
            [
                Experiment(
                    name="dup",
                    variants=[ExperimentVariant(name="A", weight=1.0)],
                ),
                Experiment(
                    name="dup",
                    variants=[ExperimentVariant(name="A", weight=1.0)],
                ),
            ]
        )
