"""Build evaluation dataset with LLM-as-judge relevance labels.

For each of 20 test queries, retrieves top 30 hybrid results (fine-tuned
embeddings) and asks Claude Haiku to rate relevance on a 0-3 scale.

Output: data/evaluation/labeled_queries.jsonl

Usage:
    python scripts/build_eval_dataset.py [--limit N] [--resume]
    python scripts/build_eval_dataset.py --limit 10   # preview first 10 pairs
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Test queries (same as compare_methods.py) ────────────────────────────────
QUERIES = [
    "breast cancer hormone receptor positive phase 3",
    "immunotherapy for non-small cell lung cancer",
    "CAR-T cell therapy for pediatric leukemia",
    "clinical trial for glioblastoma that has come back",
    "melanomt with checkpoint inhibitors",
    "prostate cancer trials for men over 65",
    "pancreatic cancer that has spread to the liver",
    "triple negative breast cancer neoadjuvant",
    "colorectal cancer with microsatellite instability",
    "ovarian cancer PARP inhibitor maintenance",
    "stage 4 kidney cancer immunotherapy combination",
    "newly diagnosed multiple myeloma treatment",
    "HPV related head and neck cancer",
    "sarcoma clinical trials for young adults",
    "targeted therapy for EGFR mutated lung cancer",
    "liver cancer hepatocellular carcinoma systemic therapy",
    "bladder cancer BCG unresponsive",
    "chronic lymphocytic leukemia ibrutinib",
    "neuroblastoma high risk children",
    "mesothelioma immunotherapy combination",
]

OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "labeled_queries.jsonl"

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


def get_eligibility_lookup(db_path: str) -> dict[str, str]:
    """Load eligibility criteria from SQLite into a dict keyed by nct_id.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Dict mapping nct_id to eligibility_criteria text.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT nct_id, eligibility_criteria FROM trials").fetchall()
    conn.close()
    return {nct_id: (elig or "") for nct_id, elig in rows}


def load_existing_labels(output_file: Path) -> set[tuple[int, str]]:
    """Load already-labeled (query_id, nct_id) pairs for resume support.

    Args:
        output_file: Path to the JSONL output file.

    Returns:
        Set of (query_id, nct_id) tuples already labeled.
    """
    existing = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing.add((rec["query_id"], rec["nct_id"]))
    return existing


def label_pair(
    client: "anthropic.Anthropic",
    query: str,
    trial: dict,
    eligibility: str,
) -> tuple[int, str]:
    """Call Claude Haiku to rate a single (query, trial) pair.

    Args:
        client: Anthropic API client.
        query: Patient search query.
        trial: Dict with trial metadata (title, conditions, phase, status).
        eligibility: Eligibility criteria text.

    Returns:
        Tuple of (relevance_score, reason_string).
    """
    prompt = LABELING_PROMPT.format(
        query=query,
        trial_title=trial.get("title", "N/A"),
        conditions=trial.get("conditions", "N/A"),
        phase=trial.get("phase", "N/A"),
        status=trial.get("status", "N/A"),
        eligibility=eligibility[:500],
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Parse JSON from response (handle possible markdown wrapping)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return int(result["score"]), result.get("reason", "")
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Failed to parse response for %s: %s — raw: %s", trial.get("nct_id"), exc, text)
        return -1, f"parse_error: {text[:100]}"
    except Exception as exc:
        logger.error("API error for %s: %s", trial.get("nct_id"), exc)
        return -1, f"api_error: {exc}"


def main() -> None:
    """Retrieve hybrid results and label with Claude Haiku."""
    import anthropic

    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.hybrid import HybridRetriever
    from TrialMine.retrieval.semantic import FAISSIndex

    parser = argparse.ArgumentParser(description="Build evaluation dataset with LLM labels")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit total pairs to label (0 = all). Use --limit 10 for preview.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-labeled pairs (append mode).",
    )
    parser.add_argument("--db", default="data/trials.db", help="SQLite database path")
    parser.add_argument("--top-k", type=int, default=30, help="Results per query")
    args = parser.parse_args()

    # ── Setup retrieval ───────────────────────────────────────────────────────
    print("Loading search components...")
    es_index = ElasticsearchIndex()
    faiss_index = FAISSIndex()
    faiss_index.load("data/faiss_finetuned.index", "data/faiss_finetuned.json")
    embedder = TrialEmbedder(model_name="models/embeddings/fine-tuned")
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    # ── Load eligibility from SQLite ──────────────────────────────────────────
    print("Loading eligibility criteria from SQLite...")
    elig_lookup = get_eligibility_lookup(args.db)
    print(f"  Loaded eligibility for {len(elig_lookup):,} trials")

    # ── Retrieve top-k for each query ─────────────────────────────────────────
    print(f"\nRetrieving top {args.top_k} hybrid results for {len(QUERIES)} queries...")
    query_results: list[list[dict]] = []
    for i, query in enumerate(QUERIES):
        results = hybrid.search(query=query, top_k=args.top_k)
        query_results.append(results)
        print(f"  Query {i}: {len(results)} results")

    total_pairs = sum(len(r) for r in query_results)
    print(f"\nTotal (query, trial) pairs: {total_pairs}")

    # ── Resume support ────────────────────────────────────────────────────────
    existing = set()
    if args.resume:
        existing = load_existing_labels(OUTPUT_FILE)
        print(f"Resuming: {len(existing)} pairs already labeled")

    # ── Label with Claude Haiku ───────────────────────────────────────────────
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    labeled_count = 0
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}

    with open(OUTPUT_FILE, mode) as f:
        for query_id, (query, results) in enumerate(zip(QUERIES, query_results)):
            for trial in results:
                nct_id = trial["nct_id"]

                # Skip if already labeled
                if (query_id, nct_id) in existing:
                    continue

                # Check limit
                if args.limit > 0 and labeled_count >= args.limit:
                    break

                eligibility = elig_lookup.get(nct_id, "")
                score, reason = label_pair(client, query, trial, eligibility)

                record = {
                    "query_id": query_id,
                    "query": query,
                    "nct_id": nct_id,
                    "trial_title": trial.get("title", ""),
                    "relevance": score,
                    "reason": reason,
                    "labeler": "claude-haiku",
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                score_dist[score] = score_dist.get(score, 0) + 1
                labeled_count += 1

                if labeled_count % 10 == 0:
                    print(f"  Labeled {labeled_count} pairs...")

                # Rate limiting: ~0.1s between calls
                time.sleep(0.1)

            if args.limit > 0 and labeled_count >= args.limit:
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  LABELING COMPLETE")
    print(f"{'='*55}")
    print(f"  Total labeled   : {labeled_count}")
    print(f"  Output file     : {OUTPUT_FILE}")
    print(f"\n  Score distribution:")
    for score in [0, 1, 2, 3]:
        count = score_dist.get(score, 0)
        pct = count / max(labeled_count, 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {score}: {count:>4} ({pct:5.1f}%) {bar}")
    errors = score_dist.get(-1, 0)
    if errors:
        print(f"   -1: {errors:>4} (parse/API errors)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
