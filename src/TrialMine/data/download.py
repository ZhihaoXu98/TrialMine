"""Download clinical trial data from ClinicalTrials.gov API v2.

Pagination uses opaque pageToken (NOT page numbers). Max pageSize = 1000.
Each page is saved as data/raw/page_{n:04d}.json.
A state file at data/raw/.download_state.json enables resume after interruption.
"""

import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 1000
REQUEST_DELAY = 0.5  # seconds — respect rate limits

ONCOLOGY_QUERY = (
    "cancer OR oncology OR tumor OR carcinoma OR "
    "lymphoma OR leukemia OR melanoma OR sarcoma"
)

_STATE_FILE = ".download_state.json"


def _load_state(output_dir: Path) -> dict:
    state_path = output_dir / _STATE_FILE
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"next_page_token": None, "pages_saved": 0, "trials_downloaded": 0}


def _save_state(output_dir: Path, state: dict) -> None:
    (output_dir / _STATE_FILE).write_text(json.dumps(state))


def _page_path(output_dir: Path, page_num: int) -> Path:
    return output_dir / f"page_{page_num:04d}.json"


def fetch_page(client: httpx.Client, params: dict) -> dict:
    """Fetch one page from the API and return the raw response dict.

    Args:
        client: Shared httpx client.
        params: Query parameters (query.term, pageSize, optional pageToken).

    Returns:
        Raw API response dict with 'studies', 'nextPageToken', 'totalCount'.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses after retries.
    """
    for attempt in range(3):
        try:
            response = client.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            if attempt == 2:
                raise
            wait = 2 ** attempt * 2
            logger.warning("Request failed (%s), retrying in %ds...", exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_oncology_trials(
    output_dir: Path,
    query: str = ONCOLOGY_QUERY,
) -> int:
    """Download all matching trials and save each page as JSON.

    Supports resume: already-downloaded pages are skipped based on the
    state file. Run again after interruption to continue from where it stopped.

    Args:
        output_dir: Directory where page files and state file are written.
        query: ClinicalTrials.gov free-text query string.

    Returns:
        Total number of trials downloaded in this run (cumulative if resumed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(output_dir)

    page_num = state["pages_saved"]
    total = state["trials_downloaded"]
    next_token = state["next_page_token"]

    if page_num > 0:
        logger.info("Resuming from page %d (%d trials already saved)", page_num, total)

    base_params: dict = {"query.term": query, "pageSize": PAGE_SIZE, "format": "json"}

    with httpx.Client(headers={"Accept": "application/json"}) as client:
        while True:
            params = dict(base_params)
            if next_token:
                params["pageToken"] = next_token

            try:
                data = fetch_page(client, params)
            except Exception:
                logger.exception("Failed to fetch page %d — progress saved, safe to retry", page_num)
                break

            studies = data.get("studies", [])
            if not studies:
                break

            _page_path(output_dir, page_num).write_text(json.dumps(data))
            page_num += 1
            total += len(studies)
            next_token = data.get("nextPageToken")

            state = {
                "next_page_token": next_token,
                "pages_saved": page_num,
                "trials_downloaded": total,
            }
            _save_state(output_dir, state)

            if total % 1000 == 0 or not next_token:
                logger.info(
                    "Downloaded %d trials (page %d, total in DB: %s)",
                    total,
                    page_num,
                    data.get("totalCount", "?"),
                )

            if not next_token:
                logger.info("Download complete. Total trials: %d", total)
                break

            time.sleep(REQUEST_DELAY)

    return total


def main() -> None:
    """Entry point for `trialmine-ingest` and `make download`."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download ClinicalTrials.gov oncology data")
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw page files")
    parser.add_argument("--query", default=ONCOLOGY_QUERY, help="Search query string")
    args = parser.parse_args()

    total = download_oncology_trials(Path(args.output_dir), query=args.query)
    print(f"\nDone. {total:,} trials saved to {args.output_dir}/")
