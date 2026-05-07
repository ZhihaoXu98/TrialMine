"""Experimentation surface — A/B testing, feature flags.

This is a thin scaffold so the rest of the system has a place to land
when we wire experiments into ranking / agent decisions. Today only
``ABTestRouter`` exists; future additions should keep router + storage
behind a small interface so a real experimentation backend (Statsig,
LaunchDarkly, in-house) can replace the in-memory implementation
without touching call sites.
"""

from TrialMine.experiments.ab_test import (
    ABTestRouter,
    Experiment,
    ExperimentVariant,
)

__all__ = ["ABTestRouter", "Experiment", "ExperimentVariant"]
