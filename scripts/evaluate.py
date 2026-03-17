"""Evaluate the end-to-end retrieval and ranking pipeline.

Usage:
    python scripts/evaluate.py [--config PATH] [--split {dev,test}]

Reads held-out queries from data/evaluation/, runs the full pipeline,
computes NDCG@10, MRR, Recall@100, and saves results to docs/evaluation/.
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument("--config", default="configs/development.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: load evaluation queries and relevance judgements
    # TODO: run pipeline for each query
    # TODO: compute NDCG@10, MRR, Recall@100 per stage (BM25, semantic, hybrid, reranked)
    # TODO: print summary table (tabulate or rich)
    # TODO: save results JSON to docs/evaluation/{split}_{timestamp}.json
    logger.info("evaluate.py not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
