"""Shared pytest fixtures.

Lives at the package root so every sub-suite (unit, integration, ml,
features) inherits from one place.

Notable conventions:

* ``sample_trials`` — 50-trial stratified fixture loaded once per session
  from ``tests/fixtures/sample_trials.json``. Used by integration tests
  to bulk-index ES and by unit tests that want realistic record shapes.
* ``es_url`` — controlled by the ``ELASTICSEARCH_URL`` env var (matches
  the API's own contract). In CI the workflow points it at the ES
  service container; locally it falls back to ``http://localhost:9200``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_trials.json"


@pytest.fixture(scope="session")
def sample_trials() -> list[dict]:
    """Load the 50-trial stratified sample fixture."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"sample_trials.json not present at {FIXTURE_PATH}")
    with FIXTURE_PATH.open() as f:
        trials: list[dict] = json.load(f)
    assert len(trials) == 50, f"expected 50 trials, got {len(trials)}"
    return trials


@pytest.fixture(scope="session")
def es_url() -> str:
    """ES URL used by integration tests. Env-driven so CI can point at the
    service container (``elasticsearch:9200``) and local runs default to
    ``localhost:9200`` matching the docker-compose dev stack."""
    return os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
