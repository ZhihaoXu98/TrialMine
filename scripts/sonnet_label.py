"""Label a stratified sample of pairs with Claude Sonnet 4.6 as the gold judge.

Reads the Haiku-labeled dataset from build_full_eval_dataset.py and labels a
stratified sample with Sonnet — treated as the "human" judge for kappa
analysis. Uses the IDENTICAL prompt as Haiku so the resulting kappa measures
pure model-capability drift, not prompt drift.

Stratification:
    Default sample = 100 pairs distributed across the 8 categories. Existing
    queries get a larger share because there are 600 of them; the small
    weak-slice categories (pediatric, geographic) get at least 10 so kappa is
    meaningful within them.

Output: data/evaluation/full_labeled_dataset_human.jsonl
    Schema matches what agreement_analysis.py expects:
      relevance_llm    = Haiku label (the system under test)
      relevance_human  = Sonnet label (the gold judge — note that "human" is
                         the field name for the analysis script's API; the
                         actual labeler_human field is "claude-sonnet-4-6")

Usage:
    python scripts/sonnet_label.py                # full 100-pair stratified run
    python scripts/sonnet_label.py --sample 200   # bigger sample
    python scripts/sonnet_label.py --resume       # skip already-Sonnet-labeled
    python scripts/sonnet_label.py --limit 5      # smoke test
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/evaluation/full_labeled_dataset.jsonl")
OUTPUT_FILE = Path("data/evaluation/full_labeled_dataset_human.jsonl")

SONNET_MODEL = "claude-sonnet-4-6"

# Default per-category quotas — sums to 100. Existing gets the largest share
# because n=600; weak slices (pediatric, geographic) capped at 10 since their
# pool is small but we still want a meaningful kappa within them.
DEFAULT_QUOTAS = {
    "existing": 20,
    "common": 12,
    "rare": 12,
    "pediatric": 10,
    "complex": 12,
    "geographic": 10,
    "vague": 12,
    "treatment": 12,
}

# Identical prompt to build_full_eval_dataset.py so kappa isolates model effect.
LABELING_PROMPT = """Rate the relevance of this clinical trial to this patient's search query.

Patient query: {query}

Trial title: {trial_title}
Trial conditions: {conditions}
Trial phase: {phase}
Trial status: {status}
Trial eligibility (first 500 chars): {eligibility}

Rate on a scale of 0-3:
0 = Completely irrelevant — wrong cancer type entirely
1 = Marginally relevant — same general cancer area but wrong specifics
2 = Relevant — matches condition, patient could potentially be eligible
3 = Highly relevant — strong match for condition, treatment type, and eligibility

Respond with ONLY a JSON object: {{"score": X, "reason": "brief 1-sentence explanation"}}"""


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, returning [] if missing."""
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def stratified_sample(
    records: list[dict],
    quotas: dict[str, int],
    seed: int,
) -> list[dict]:
    """Pick a stratified random sample.

    Args:
        records: All Haiku-labeled records (filtered to relevance >= 0).
        quotas: {category: n_to_pick}. If a category has fewer than n_to_pick
            records, takes all of them and logs a warning.
        seed: RNG seed.

    Returns:
        Sorted by (query_id, rank) for reproducible labeling order.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "uncategorised")].append(r)

    sampled: list[dict] = []
    for cat, n in quotas.items():
        pool = by_cat.get(cat, [])
        if not pool:
            logger.warning("Category '%s' has 0 records — skipping", cat)
            continue
        if len(pool) < n:
            logger.warning("Category '%s' has only %d records (wanted %d) — taking all",
                           cat, len(pool), n)
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n))

    sampled.sort(key=lambda r: (r["query_id"], r.get("rank", 0)))
    return sampled


def get_eligibility_lookup(db_path: str) -> dict[str, str]:
    """Pull eligibility_criteria from SQLite, keyed by nct_id."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT nct_id, eligibility_criteria FROM trials").fetchall()
    conn.close()
    return {nct_id: (elig or "") for nct_id, elig in rows}


def label_with_sonnet(client, query: str, record: dict, eligibility: str) -> tuple[int, str]:
    """Send one (query, trial) to Sonnet and parse the JSON score+reason.

    Returns (-1, "<error>") on parse / API failure so the caller can filter.
    """
    prompt = LABELING_PROMPT.format(
        query=query,
        trial_title=record.get("trial_title", "N/A"),
        conditions=record.get("trial_conditions", "N/A"),
        phase=record.get("trial_phase", "N/A"),
        status=record.get("trial_status", "N/A"),
        eligibility=eligibility[:500],
    )
    text = ""
    try:
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return int(result["score"]), result.get("reason", "")
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Parse error for %s: %s — raw: %s",
                       record.get("nct_id"), exc, text[:120])
        return -1, f"parse_error: {str(exc)[:100]}"
    except Exception as exc:
        logger.error("API error for %s: %s", record.get("nct_id"), exc)
        return -1, f"api_error: {str(exc)[:100]}"


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="Sonnet-as-judge labeling for kappa")
    parser.add_argument("--sample", type=int, default=100,
                        help="Total sample size (default: 100, distributed via quotas)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed (deterministic across runs)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap labels written this run (0 = full sample)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already in the output file")
    parser.add_argument("--db", default="data/trials.db", help="SQLite path for eligibility")
    parser.add_argument("--rate-limit-s", type=float, default=0.1,
                        help="Sleep between API calls")
    return parser.parse_args()


def scale_quotas(quotas: dict[str, int], target: int) -> dict[str, int]:
    """Scale per-category quotas proportionally to hit a custom total."""
    current = sum(quotas.values())
    if current == target:
        return quotas
    scaled = {k: max(1, round(v * target / current)) for k, v in quotas.items()}
    # Adjust drift from rounding
    drift = target - sum(scaled.values())
    if drift:
        # Add/remove to/from the largest bucket
        biggest = max(scaled, key=scaled.get)
        scaled[biggest] += drift
    return scaled


def main() -> None:
    """Sample, label with Sonnet, write the human-format JSONL."""
    import anthropic

    args = parse_args()
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run build_full_eval_dataset.py first.")
        sys.exit(1)

    haiku_records = [r for r in load_jsonl(INPUT_FILE) if r.get("relevance", -1) >= 0]
    logger.info("Loaded %d Haiku-labeled records", len(haiku_records))

    quotas = scale_quotas(DEFAULT_QUOTAS, args.sample) if args.sample != 100 else DEFAULT_QUOTAS
    logger.info("Per-category quotas (target n=%d): %s", sum(quotas.values()), quotas)

    sample = stratified_sample(haiku_records, quotas, args.seed)
    logger.info("Sampled %d pairs", len(sample))

    # Resume support
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_pairs: set[tuple[int, str]] = set()
    if args.resume:
        for r in load_jsonl(OUTPUT_FILE):
            if r.get("labeler_human") == SONNET_MODEL:
                existing_pairs.add((r["query_id"], r["nct_id"]))
        logger.info("Resuming: %d Sonnet-labeled pairs already written", len(existing_pairs))

    queue = [r for r in sample if (r["query_id"], r["nct_id"]) not in existing_pairs]
    if args.limit > 0:
        queue = queue[: args.limit]
    if not queue:
        print("Nothing to label — sample already complete in output file.")
        return

    logger.info("Loading eligibility lookup from %s", args.db)
    elig_lookup = get_eligibility_lookup(args.db)

    client = anthropic.Anthropic()
    mode = "a" if (args.resume or OUTPUT_FILE.exists()) else "w"

    score_dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
    agree_count = 0
    written = 0

    with open(OUTPUT_FILE, mode) as f:
        for i, record in enumerate(queue, start=1):
            eligibility = elig_lookup.get(record["nct_id"], "")
            sonnet_score, sonnet_reason = label_with_sonnet(
                client, record["query"], record, eligibility,
            )

            haiku_score = record["relevance"]
            agree = (sonnet_score == haiku_score) if sonnet_score >= 0 else False
            if agree:
                agree_count += 1

            out_record = {
                "query_id": record["query_id"],
                "query": record["query"],
                "category": record.get("category"),
                "nct_id": record["nct_id"],
                "rank": record.get("rank"),
                "trial_title": record.get("trial_title", ""),
                "relevance_llm": haiku_score,
                "relevance_human": sonnet_score,
                "agree": agree,
                "llm_reason": record.get("reason", ""),
                "human_reason": sonnet_reason,
                "labeler_llm": record.get("labeler", "claude-haiku-4-5"),
                "labeler_human": SONNET_MODEL,
            }
            f.write(json.dumps(out_record) + "\n")
            f.flush()

            score_dist[sonnet_score] = score_dist.get(sonnet_score, 0) + 1
            written += 1

            if written % 10 == 0:
                running_agree = agree_count / written * 100
                logger.info("  Labeled %d/%d  raw_agree=%.1f%%", written, len(queue), running_agree)

            time.sleep(args.rate_limit_s)

    print()
    print("=" * 60)
    print("  SONNET LABELING COMPLETE")
    print("=" * 60)
    print(f"  Pairs labeled       : {written}")
    print(f"  Output              : {OUTPUT_FILE}")
    print(f"  Raw agreement       : {agree_count / max(1, written):.1%}")
    print()
    print("  Sonnet score distribution:")
    for s in [0, 1, 2, 3]:
        c = score_dist.get(s, 0)
        pct = c / max(1, written) * 100
        bar = "#" * int(pct / 2)
        print(f"    {s}: {c:>3} ({pct:5.1f}%) {bar}")
    if score_dist.get(-1, 0):
        print(f"   -1: {score_dist[-1]:>3} (parse/API errors)")
    print()
    print("  Next: python scripts/agreement_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
