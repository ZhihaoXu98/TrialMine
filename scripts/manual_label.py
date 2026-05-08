"""Manual labeling CLI for human/LLM agreement analysis.

Reads the LLM-labeled dataset (built by build_full_eval_dataset.py) and walks
the user through each (query, trial) pair, showing the LLM's label + reason
and prompting for the human's label (0-3) or Enter to accept the LLM.

Output (always-on, both labels per record):
    data/evaluation/full_labeled_dataset_human.jsonl

Resume:
    Re-running the script picks up where the user quit. Already-labeled pairs
    (matched by (query_id, nct_id)) are skipped.

Sampling:
    --sample N picks a random subset by deterministic seed (default: full set).
    Useful when the user wants to label, e.g., 200 pairs out of ~1200.

Usage:
    python scripts/manual_label.py                  # walk through full set
    python scripts/manual_label.py --sample 200     # 200-pair random sample
    python scripts/manual_label.py --sample 200 --seed 42
    python scripts/manual_label.py --category complex   # filter to one category
    python scripts/manual_label.py --skip-agreed    # only show LLM=??? edge cases (not yet labeled)

During labeling:
    - Type 0, 1, 2, 3 to set your own label.
    - Press Enter (empty input) to accept the LLM's label.
    - Type 'q' or Ctrl-D to quit (state is saved as you go).
    - Type 's' to skip without labeling (won't appear next run).
"""

import argparse
import json
import logging
import random
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/evaluation/full_labeled_dataset.jsonl")
OUTPUT_FILE = Path("data/evaluation/full_labeled_dataset_human.jsonl")
SKIP_FILE = Path("data/evaluation/full_labeled_dataset_skipped.jsonl")

VALID_INPUTS = {"0", "1", "2", "3", "", "q", "s"}


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts; return [] if missing."""
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_eligibility_lookup(db_path: str) -> dict[str, str]:
    """Load eligibility_criteria from SQLite, keyed by nct_id."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT nct_id, eligibility_criteria FROM trials").fetchall()
    conn.close()
    return {nct_id: (elig or "") for nct_id, elig in rows}


def render_pair(record: dict, eligibility: str, idx: int, total: int) -> None:
    """Print one (query, trial) pair to the terminal in a readable format."""
    print()
    print("=" * 78)
    print(f"  [{idx}/{total}]  query_id={record['query_id']}  rank={record.get('rank', '?')}  "
          f"category={record.get('category', '?')}")
    print("=" * 78)
    print(f"\n  QUERY: {record['query']}\n")
    print(f"  TRIAL: {record['nct_id']}")
    print(f"    Title:      {record.get('trial_title', '')[:160]}")
    print(f"    Conditions: {record.get('trial_conditions', '')[:160]}")
    print(f"    Phase:      {record.get('trial_phase', '?')}    "
          f"Status: {record.get('trial_status', '?')}")
    if eligibility:
        elig_short = eligibility.replace("\n", " ").strip()[:400]
        print(f"    Eligibility (first 400 chars):")
        print(f"      {elig_short}")
    print()
    print(f"  LLM LABEL: {record['relevance']}    "
          f"(reason: {record.get('reason', '')[:160]})")
    print()
    print("  Scale: 0=irrelevant  1=marginal  2=relevant  3=highly relevant")
    print("  Press 0/1/2/3, Enter to accept LLM, 's' to skip, 'q' to quit.")


def select_pairs(
    records: list[dict],
    sample: int,
    seed: int,
    category: str | None,
) -> list[dict]:
    """Apply category filter and optional random sampling.

    Sampling is deterministic with the given seed so re-running with the same
    flags yields the same pair order — important so the user's progress
    (42/200) refers to the same set on resume.
    """
    if category:
        records = [r for r in records if r.get("category") == category]
    if sample > 0 and sample < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, sample)
        # Keep deterministic order: sort by (query_id, rank) within the sample
        records.sort(key=lambda r: (r["query_id"], r.get("rank", 0)))
    else:
        records.sort(key=lambda r: (r["query_id"], r.get("rank", 0)))
    return records


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="Manual labeling CLI for kappa analysis")
    parser.add_argument("--sample", type=int, default=0,
                        help="Random sample size (0 = label all pairs)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (for reproducibility)")
    parser.add_argument("--category", default=None,
                        help="Filter to one category (common, rare, pediatric, complex, "
                             "geographic, vague, treatment, existing)")
    parser.add_argument("--db", default="data/trials.db", help="SQLite path for eligibility lookup")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip pairs where LLM returned -1 (parse/API errors)")
    return parser.parse_args()


def main() -> None:
    """Run the interactive labeling loop and append results to the output file."""
    args = parse_args()

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run scripts/build_full_eval_dataset.py first to generate LLM labels.")
        sys.exit(1)

    llm_records = load_jsonl(INPUT_FILE)
    if args.skip_errors:
        llm_records = [r for r in llm_records if r["relevance"] >= 0]

    # Already-human-labeled (resume)
    human_records = load_jsonl(OUTPUT_FILE)
    already_labeled = {(r["query_id"], r["nct_id"]) for r in human_records}

    # Skipped pairs — won't be shown again
    skipped_records = load_jsonl(SKIP_FILE)
    already_skipped = {(r["query_id"], r["nct_id"]) for r in skipped_records}

    # Apply sample / category filter to LLM records
    candidate_records = select_pairs(llm_records, args.sample, args.seed, args.category)

    # Drop ones we've already labeled or skipped
    queue = [r for r in candidate_records
             if (r["query_id"], r["nct_id"]) not in already_labeled
             and (r["query_id"], r["nct_id"]) not in already_skipped]

    total_in_target = len(candidate_records)
    print()
    print(f"  LLM-labeled pairs in scope     : {total_in_target}")
    print(f"  Already human-labeled (resume) : {len(already_labeled & {(r['query_id'], r['nct_id']) for r in candidate_records})}")
    print(f"  Already skipped (resume)       : {len(already_skipped & {(r['query_id'], r['nct_id']) for r in candidate_records})}")
    print(f"  To label this session          : {len(queue)}")
    print()

    if not queue:
        print("All pairs in scope already labeled or skipped. Nothing to do.")
        return

    # Eligibility lookup for richer rendering
    print("Loading eligibility lookup from SQLite...")
    elig_lookup = load_eligibility_lookup(args.db)
    print(f"  Loaded eligibility for {len(elig_lookup):,} trials\n")

    # ── Interactive loop ──────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.touch(exist_ok=True)
    SKIP_FILE.touch(exist_ok=True)

    labeled_this_session = 0
    accepted_llm = 0
    overrode_llm = 0
    quit_early = False

    out_f = open(OUTPUT_FILE, "a")
    skip_f = open(SKIP_FILE, "a")
    try:
        for i, record in enumerate(queue, start=1):
            eligibility = elig_lookup.get(record["nct_id"], "")
            render_pair(record, eligibility, idx=i, total=len(queue))

            try:
                raw = input("  your label > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  [quit]")
                quit_early = True
                break

            if raw not in VALID_INPUTS:
                print(f"  -> '{raw}' not recognised. Skipping render; try again next run.")
                continue

            if raw == "q":
                quit_early = True
                break

            if raw == "s":
                skip_rec = {"query_id": record["query_id"], "nct_id": record["nct_id"]}
                skip_f.write(json.dumps(skip_rec) + "\n")
                skip_f.flush()
                continue

            llm_score = record["relevance"]
            if raw == "":
                human_score = llm_score
                accepted_llm += 1
            else:
                human_score = int(raw)
                if human_score != llm_score:
                    overrode_llm += 1
                else:
                    accepted_llm += 1

            out_record = {
                "query_id": record["query_id"],
                "query": record["query"],
                "category": record.get("category"),
                "nct_id": record["nct_id"],
                "rank": record.get("rank"),
                "trial_title": record.get("trial_title", ""),
                "relevance_llm": llm_score,
                "relevance_human": human_score,
                "agree": human_score == llm_score,
                "llm_reason": record.get("reason", ""),
                "labeler_human": "user",
                "labeler_llm": record.get("labeler", "claude-haiku-4-5"),
            }
            out_f.write(json.dumps(out_record) + "\n")
            out_f.flush()
            labeled_this_session += 1
    finally:
        out_f.close()
        skip_f.close()

    # ── Session summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Labeled this session : {labeled_this_session}")
    print(f"    Accepted LLM       : {accepted_llm}")
    print(f"    Overrode LLM       : {overrode_llm}")
    if labeled_this_session:
        agree_pct = accepted_llm / labeled_this_session * 100
        print(f"    Raw agreement      : {agree_pct:.1f}%")
    print(f"  Output file          : {OUTPUT_FILE}")
    if quit_early:
        remaining = len(queue) - labeled_this_session
        print(f"\n  {remaining} pairs remain — re-run to resume.")
    else:
        print("\n  All pairs in scope labeled.")
        print("  Next: python scripts/agreement_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
