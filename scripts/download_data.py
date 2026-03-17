"""End-to-end data pipeline: download → parse → store → summarise.

Usage:
    python scripts/download_data.py [--raw-dir PATH] [--db PATH] [--skip-download]
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def print_summary(trials) -> None:  # type: ignore[no-untyped-def]
    """Print summary statistics to stdout."""
    total = len(trials)
    print(f"\n{'='*55}")
    print(f"  TRIALMINE DATA SUMMARY")
    print(f"{'='*55}")
    print(f"  Total trials stored : {total:,}")

    # By status
    status_counts = Counter(t.status or "Unknown" for t in trials)
    print(f"\n  By status (top 10):")
    for status, count in status_counts.most_common(10):
        print(f"    {status:<35} {count:>7,}")

    # By phase
    phase_counts = Counter(t.phase or "Not specified" for t in trials)
    print(f"\n  By phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"    {phase:<35} {count:>7,}")

    # Eligibility coverage
    with_eligibility = sum(1 for t in trials if t.eligibility_criteria)
    print(f"\n  With eligibility criteria : {with_eligibility:,} ({with_eligibility/total*100:.1f}%)")
    print(f"  Without eligibility       : {total - with_eligibility:,}")

    # Top 20 conditions
    condition_counts: Counter = Counter()
    for t in trials:
        condition_counts.update(t.conditions)
    print(f"\n  Top 20 conditions:")
    for condition, count in condition_counts.most_common(20):
        print(f"    {condition:<45} {count:>6,}")

    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrialMine data pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for raw page files")
    parser.add_argument("--db", default="data/trials.db", help="SQLite database path")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (parse and store existing raw files only)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    db_path = Path(args.db)

    # Step 1: Download
    if not args.skip_download:
        from TrialMine.data.download import download_oncology_trials
        logger.info("=== Step 1: Download ===")
        n_downloaded = download_oncology_trials(raw_dir)
        logger.info("Download complete: %d trials", n_downloaded)
    else:
        logger.info("Skipping download (--skip-download)")

    # Step 2: Parse
    from TrialMine.data.parse import parse_raw_files
    logger.info("=== Step 2: Parse ===")
    trials = parse_raw_files(raw_dir)
    logger.info("Parsed %d trials", len(trials))

    if not trials:
        logger.error("No trials parsed — nothing to store.")
        return

    # Step 3: Store
    from TrialMine.data.store import store_trials
    logger.info("=== Step 3: Store ===")
    stored = store_trials(trials, db_path)
    logger.info("Stored %d new trials to %s", stored, db_path)

    # Step 4: Summary
    from TrialMine.data.store import load_trials
    all_trials = load_trials(db_path)
    print_summary(all_trials)


if __name__ == "__main__":
    main()
