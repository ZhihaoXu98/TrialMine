"""Build the Elasticsearch BM25 index from trials stored in SQLite.

Usage:
    python scripts/build_index.py [--db PATH] [--es-url URL] [--index NAME]
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Elasticsearch index")
    parser.add_argument("--db", default="data/trials.db", help="SQLite database path")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index", default="trials", help="Elasticsearch index name")
    args = parser.parse_args()

    # Step 1: Load trials from SQLite
    from TrialMine.data.store import load_trials

    logger.info("Loading trials from %s ...", args.db)
    t0 = time.time()
    trials = load_trials(Path(args.db))
    logger.info("Loaded %d trials in %.1fs", len(trials), time.time() - t0)

    # Step 2: Create index and bulk-index trials
    from TrialMine.retrieval.bm25 import ElasticsearchIndex

    es_index = ElasticsearchIndex(es_url=args.es_url, index_name=args.index)
    es_index.create_index(delete_existing=True)

    t0 = time.time()
    indexed = es_index.index_trials(trials)
    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print(f"  INDEX COMPLETE")
    print(f"{'='*55}")
    print(f"  Total indexed : {indexed:,}")
    print(f"  Time taken    : {elapsed:.1f}s")
    print(f"  Rate          : {indexed / elapsed:,.0f} docs/sec")
    print(f"{'='*55}")

    # Step 3: Test query
    print(f"\n  Test query: 'breast cancer'\n")
    results = es_index.search("breast cancer", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['nct_id']}] (score={r['score']:.2f})")
        print(f"     {r['title'][:90]}")
        print(f"     phase={r['phase']}  status={r['status']}")
        print()


if __name__ == "__main__":
    main()
