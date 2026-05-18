"""A/B test router — deterministic variant assignment + exposure logging.

Stub implementation that gives the ranking + agent layers a stable
contract to lean on when we start running real experiments. The
*interface* is the load-bearing part:

* :meth:`ABTestRouter.route` — given a stable subject identifier and an
  experiment name, return one of the experiment's variants. Routing is
  deterministic (same subject → same variant for the lifetime of the
  experiment) and proportional to ``ExperimentVariant.weight``.
* :meth:`ABTestRouter.log_exposure` — record that ``subject_id`` was
  shown ``variant`` for ``experiment_name``. Exposure is what an
  analysis pipeline joins to outcomes to compute lift.

Today the router is in-memory + log-only. The next iteration plugs a
real telemetry sink (Datadog, BigQuery, or a Kafka topic) into
``log_exposure`` without changing how the call sites look.

Design choices
--------------

* **Hash-based bucketing.** ``route`` hashes ``f"{experiment}:{subject_id}"``
  with SHA-256 and maps the first 8 bytes into ``[0, 1)``. The result is
  compared against cumulative variant weights. This makes routing both
  deterministic *and* independent across experiments — the same subject
  lands in different variants of two different experiments, so we don't
  accidentally couple traffic.
* **No mutable global state.** Construct a router with the experiment
  catalog explicitly. Tests can inject a freshly-built router without
  module-level resets.
* **Variant weights need not sum to 1.** They're normalised on
  construction. This avoids subtle bugs where adding a new variant
  silently changes the bucket boundaries of existing ones.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #


class ExperimentVariant(BaseModel):
    """One arm of an experiment.

    ``weight`` is relative; the router normalises across the variants of
    the same experiment.
    """

    name: str = Field(..., min_length=1)
    weight: float = Field(..., ge=0.0)


class Experiment(BaseModel):
    """An experiment definition.

    ``enabled=False`` short-circuits to the first variant — useful for
    killing an experiment without redeploying call sites.
    """

    name: str = Field(..., min_length=1)
    variants: list[ExperimentVariant] = Field(..., min_length=1)
    enabled: bool = True

    @field_validator("variants")
    @classmethod
    def _variant_names_unique(cls, value: list[ExperimentVariant]) -> list[ExperimentVariant]:
        names = [v.name for v in value]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variant names in experiment: {names}")
        return value


# --------------------------------------------------------------------------- #
# Router                                                                       #
# --------------------------------------------------------------------------- #


class ABTestRouter:
    """Deterministic A/B router + in-memory exposure log.

    Construct with the experiment catalog:

        router = ABTestRouter([
            Experiment(name="cross_encoder_blend",
                       variants=[ExperimentVariant(name="control", weight=0.5),
                                 ExperimentVariant(name="ce_only", weight=0.5)]),
        ])
        variant = router.route(subject_id="user-123", experiment_name="cross_encoder_blend")
        router.log_exposure(subject_id="user-123",
                            experiment_name="cross_encoder_blend",
                            variant=variant)

    The router is read-only after construction — adding/removing
    experiments at runtime would invalidate already-assigned subjects.
    Build a new router instead.
    """

    def __init__(self, experiments: Iterable[Experiment]) -> None:
        """Build a router from an experiment catalog.

        Args:
            experiments: The full set of experiments this router will
                serve. Names must be unique across the catalog.

        Raises:
            ValueError: If two experiments share a name, or if any
                experiment has zero total variant weight.
        """
        self._catalog: dict[str, Experiment] = {}
        for exp in experiments:
            if exp.name in self._catalog:
                raise ValueError(f"duplicate experiment name: {exp.name!r}")
            total = sum(v.weight for v in exp.variants)
            if total <= 0.0:
                raise ValueError(f"experiment {exp.name!r} has zero total variant weight")
            self._catalog[exp.name] = exp
        # Local exposure log — replaced by a real sink in production.
        self._exposures: list[dict[str, str]] = []

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def route(self, subject_id: str, experiment_name: str) -> str:
        """Return the variant assigned to ``subject_id`` for ``experiment_name``.

        Routing is deterministic: the same ``(experiment, subject)`` pair
        always returns the same variant for the life of this router. When
        the experiment is disabled, every subject lands in the *first*
        variant (the control by convention).

        Args:
            subject_id: Stable identifier for the unit being randomised
                (typically a user id; for anonymous traffic, a session
                or device id is acceptable as long as it's stable).
            experiment_name: One of the names supplied at construction.

        Returns:
            The chosen :attr:`ExperimentVariant.name`.

        Raises:
            KeyError: If ``experiment_name`` is not in the catalog.
        """
        exp = self._catalog[experiment_name]
        if not exp.enabled:
            return exp.variants[0].name

        bucket = self._bucket(experiment_name, subject_id)
        total = sum(v.weight for v in exp.variants)
        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.weight / total
            if bucket < cumulative:
                return variant.name
        # Float drift — bucket was 0.999... and never crossed the last
        # boundary. Fall through to the last variant.
        return exp.variants[-1].name

    def log_exposure(
        self,
        subject_id: str,
        experiment_name: str,
        variant: str,
    ) -> None:
        """Record that ``subject_id`` was shown ``variant`` for the experiment.

        Stub implementation: appends to an in-memory list and emits a
        structured log line. Production should fan this to a durable
        store (BigQuery / Snowflake / a Kafka topic) so the analysis
        pipeline can join exposures to outcomes.

        Args:
            subject_id: The subject that was exposed.
            experiment_name: The experiment they were exposed to.
            variant: The variant they saw — typically the value just
                returned by :meth:`route`, but call sites may pass an
                override (e.g. if a manual force-list assigned them
                outside the random bucket).
        """
        if experiment_name not in self._catalog:
            # Don't raise — exposure logging must be cheap and total.
            logger.warning(
                "log_exposure called for unknown experiment %r — recorded anyway",
                experiment_name,
            )
        record = {
            "subject_id": subject_id,
            "experiment": experiment_name,
            "variant": variant,
        }
        self._exposures.append(record)
        logger.info("ab_exposure %s", record)

    # ------------------------------------------------------------------ #
    # Test / inspection helpers                                           #
    # ------------------------------------------------------------------ #

    @property
    def exposures(self) -> list[dict[str, str]]:
        """Return a copy of the in-memory exposure log (test inspection)."""
        return list(self._exposures)

    @property
    def experiment_names(self) -> list[str]:
        """All experiment names this router knows about."""
        return list(self._catalog.keys())

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bucket(experiment_name: str, subject_id: str) -> float:
        """Hash ``(experiment, subject)`` into ``[0, 1)``.

        SHA-256 over ``f"{experiment}:{subject}"``; first 8 bytes as a
        big-endian unsigned int divided by ``2**64``. The experiment
        prefix is the salt — same subject lands in different buckets
        across experiments, so traffic is independent.
        """
        raw = hashlib.sha256(f"{experiment_name}:{subject_id}".encode()).digest()
        n = int.from_bytes(raw[:8], byteorder="big", signed=False)
        return n / (1 << 64)


# --------------------------------------------------------------------------- #
# Process-wide singleton — Phase A7                                            #
# --------------------------------------------------------------------------- #


_default_router: ABTestRouter | None = None


def get_default_router() -> ABTestRouter:
    """Return the process-wide :class:`ABTestRouter` for TrialMine.

    Lazy-initialised on first call. The router's catalog lists the
    experiments TrialMine actively runs; today that's just
    ``agent_path_v1`` (registered ``enabled=False`` until Phase D flips
    the toggle), which means every subject lands in the first variant
    (``"control"``) regardless of hash bucketing. Adding a new experiment
    is a one-line catalog edit here; no call site changes are needed.

    Tests that need an isolated router can either build one explicitly
    or reset the module-level ``_default_router`` to ``None`` after the
    test to force a fresh build on next call.
    """
    global _default_router
    if _default_router is None:
        _default_router = ABTestRouter(
            [
                Experiment(
                    name="agent_path_v1",
                    variants=[
                        ExperimentVariant(name="control", weight=0.5),
                        ExperimentVariant(name="treatment", weight=0.5),
                    ],
                    # Phase D1 — flipped on after the C4 decision gate
                    # cleared the SHIP criteria (Decision 43). 50/50 split
                    # gates the agent path to roughly 21 % of production
                    # traffic (heuristic eligibility × 50 % treatment).
                    # Widen only after the no-terminator rate on hard
                    # slices drops below ~10 % (currently ~50 % on
                    # complex / vague — bubble fix catches them but
                    # agent contribution = 0 on those queries).
                    enabled=True,
                ),
            ]
        )
    return _default_router


__all__ = [
    "ABTestRouter",
    "Experiment",
    "ExperimentVariant",
    "get_default_router",
]
