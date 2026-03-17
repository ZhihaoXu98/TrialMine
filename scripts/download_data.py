"""Download oncology trials from ClinicalTrials.gov API v2.

Usage:
    python scripts/download_data.py [--query QUERY] [--output-dir PATH]

Writes newline-delimited JSON to data/raw/trials_{timestamp}.ndjson.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ClinicalTrials.gov data")
    parser.add_argument("--query", default="cancer", help="Search query")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: import and call TrialMine.data.download.download_oncology_trials()
    logger.info("download_data.py not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
