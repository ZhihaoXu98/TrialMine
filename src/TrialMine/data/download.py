"""Download clinical trial data from ClinicalTrials.gov API v2.

Pagination uses pageToken (NOT page numbers). Max pageSize is 1000.
A 0.5s delay is inserted between requests to respect rate limits.
Downloaded pages are saved as newline-delimited JSON to data/raw/.
"""

import logging
import time
from pathlib import Path

import httpx

from TrialMine.data.models import ClinicalTrial, TrialPage

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 1000
REQUEST_DELAY = 0.5  # seconds between requests


def fetch_page(client: httpx.Client, params: dict) -> TrialPage:
    """Fetch a single page of trials from the API.

    Args:
        client: Shared httpx client.
        params: Query parameters including optional pageToken.

    Returns:
        Parsed TrialPage with trials and next page token.
    """
    # TODO: implement request, map JSON → TrialPage
    raise NotImplementedError


def parse_trial(raw: dict) -> ClinicalTrial:
    """Map a single raw API study dict to a ClinicalTrial model.

    Args:
        raw: One element from the API 'studies' array.

    Returns:
        Populated ClinicalTrial instance.
    """
    # TODO: extract fields via protocolSection paths defined in CLAUDE.md
    raise NotImplementedError


def download_oncology_trials(output_dir: Path, query: str = "cancer") -> int:
    """Download all matching trials and write them to output_dir as NDJSON.

    Args:
        output_dir: Directory for raw data files.
        query: ClinicalTrials search query string.

    Returns:
        Total number of trials downloaded.
    """
    # TODO: paginate until nextPageToken is None, respect REQUEST_DELAY
    raise NotImplementedError


def main() -> None:
    """Entry point for `trialmine-ingest` and `make download`."""
    # TODO: parse CLI args (output dir, query, log level)
    raise NotImplementedError
