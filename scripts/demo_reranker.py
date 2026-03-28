"""Demo cross-encoder re-ranking: before vs after comparison.

Runs 3 diverse queries through hybrid retrieval, re-ranks with the
cross-encoder, and shows which trials moved up/down with position deltas.

Usage:
    python scripts/demo_reranker.py
    python scripts/demo_reranker.py --model models/cross-encoder/fine-tuned
    python scripts/demo_reranker.py --rerank-top-k 100

Requirements:
    - Elasticsearch running with `trials` index
    - FAISS index: data/faiss_finetuned.index + .json
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.models.cross_encoder import CrossEncoderReranker
from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"

DEMO_QUERIES = [
    "EGFR mutation non-small cell lung cancer targeted therapy",
    "breast cancer hormone receptor positive phase 3",
    "immunotherapy for advanced melanoma PD-1 checkpoint inhibitor",
]


def build_trial_text(result: dict) -> str:
    """Build cross-encoder input text from a hybrid result dict.

    Matches the training format: title [SEP] conditions [SEP] brief_summary.

    Args:
        result: Hybrid retrieval result dict (enriched with brief_summary).

    Returns:
        Concatenated trial text string.
    """
    parts = []
    if result.get("title"):
        parts.append(result["title"])
    if result.get("conditions"):
        cond = result["conditions"]
        if isinstance(cond, list):
            cond = " ".join(cond)
        parts.append(cond)
    if result.get("brief_summary"):
        parts.append(result["brief_summary"])
    text = " [SEP] ".join(parts) if parts else ""
    if len(text) > 2048:
        text = text[:2048]
    return text


def print_result(rank: int, result: dict, delta: str = "") -> None:
    """Print a single result line.

    Args:
        rank: Display rank (1-indexed).
        result: Result dict.
        delta: Position change string (e.g., "+3", "-2", "=").
    """
    title = result.get("title", "???")[:65]
    nct_id = result["nct_id"]
    source = result.get("source", "")
    ce_score = result.get("cross_encoder_score")
    score_str = f"  CE: {ce_score:.3f}" if ce_score is not None else ""
    delta_str = f"  ({delta})" if delta else ""
    print(f"    {rank}. [{nct_id}] {title}{score_str}{delta_str}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Demo cross-encoder re-ranking: before vs after",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_CE_MODEL,
        help=f"Cross-encoder model (default: {DEFAULT_CE_MODEL})",
    )
    parser.add_argument(
        "--rerank-top-k", type=int, default=50,
        help="Number of hybrid candidates to re-rank (default: 50)",
    )
    parser.add_argument(
        "--show-top", type=int, default=5,
        help="Number of top results to display (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    """Run before/after re-ranking demo on 3 queries."""
    args = parse_args()

    # ── Check prerequisites ───────────────────────────────────────────────
    if not Path(FAISS_INDEX).exists():
        print(f"ERROR: FAISS index not found: {FAISS_INDEX}")
        print("Run: python scripts/build_index.py --skip-bm25 --model fine-tuned")
        sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────────────
    print("Loading retrieval components...")
    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    print(f"Loading cross-encoder: {args.model}")
    reranker = CrossEncoderReranker(model_name=args.model)

    # ── Run demo queries ──────────────────────────────────────────────────
    for qi, query in enumerate(DEMO_QUERIES):
        print(f"\n{'='*75}")
        print(f"  QUERY {qi + 1}: {query}")
        print(f"{'='*75}")

        # Hybrid retrieval
        candidates = hybrid.search(query=query, top_k=args.rerank_top_k)

        # Fetch brief_summary so trial text matches training format
        for c in candidates:
            if not c.get("brief_summary"):
                doc = es_index.get_trial(c["nct_id"])
                if doc:
                    c["brief_summary"] = doc.get("brief_summary", "")
            c["trial_text"] = build_trial_text(c)

        # Before: top results by RRF score
        before_ids = [c["nct_id"] for c in candidates]
        before_top = candidates[:args.show_top]

        # Re-rank with cross-encoder
        reranked, elapsed_ms = reranker.rerank_with_timing(
            query, candidates, top_k=args.rerank_top_k
        )
        after_ids = [c["nct_id"] for c in reranked]
        after_top = reranked[:args.show_top]

        # Compute position deltas
        before_rank_map = {nct_id: i + 1 for i, nct_id in enumerate(before_ids)}
        after_rank_map = {nct_id: i + 1 for i, nct_id in enumerate(after_ids)}

        # Print BEFORE
        print(f"\n  BEFORE (hybrid RRF, top {args.show_top}):")
        for i, r in enumerate(before_top):
            print_result(i + 1, r)

        # Print AFTER
        print(f"\n  AFTER (+ cross-encoder re-ranking, top {args.show_top}):")
        for i, r in enumerate(after_top):
            old_rank = before_rank_map.get(r["nct_id"], "?")
            if old_rank == "?":
                delta = "new"
            elif old_rank == i + 1:
                delta = "="
            elif old_rank > i + 1:
                delta = f"^{old_rank - (i + 1)}"
            else:
                delta = f"v{(i + 1) - old_rank}"
            print_result(i + 1, r, delta=delta)

        # Score distribution
        all_scores = [c["cross_encoder_score"] for c in reranked if "cross_encoder_score" in c]
        if all_scores:
            scores_arr = np.array(all_scores)
            print(f"\n  Cross-encoder scores ({len(all_scores)} candidates):")
            print(f"    min={scores_arr.min():.3f}  mean={scores_arr.mean():.3f}  max={scores_arr.max():.3f}  std={scores_arr.std():.3f}")

        print(f"  Re-ranking latency: {elapsed_ms:.0f} ms ({len(candidates)} candidates)")

    print(f"\n{'='*75}")
    print("  Demo complete.")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
