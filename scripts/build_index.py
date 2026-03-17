"""Build Elasticsearch BM25 index and FAISS semantic index from parsed trial data.

Usage:
    python scripts/build_index.py [--config PATH] [--input-dir PATH]

Reads parsed NDJSON from data/processed/, indexes into ES, builds FAISS index.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--config", default="configs/development.yaml")
    parser.add_argument("--input-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: load config YAML
    # TODO: call BM25Retriever.index_trials()
    # TODO: call TrialEmbedder.encode() then SemanticRetriever.build_index()
    logger.info("build_index.py not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
