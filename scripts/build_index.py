"""Build Elasticsearch BM25 index and FAISS semantic index from SQLite.

Usage:
    python scripts/build_index.py [--db PATH] [--es-url URL] [--index NAME]
                                  [--faiss-path PATH] [--model NAME]
                                  [--batch-size N] [--skip-bm25] [--skip-semantic]
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


def build_bm25_index(trials: list, es_url: str, index_name: str) -> None:
    """Create and populate the Elasticsearch BM25 index.

    Args:
        trials: List of Trial objects.
        es_url: Elasticsearch URL.
        index_name: Elasticsearch index name.
    """
    from TrialMine.retrieval.bm25 import ElasticsearchIndex

    es_index = ElasticsearchIndex(es_url=es_url, index_name=index_name)
    es_index.create_index(delete_existing=True)

    t0 = time.time()
    indexed = es_index.index_trials(trials)
    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print("  BM25 INDEX COMPLETE")
    print(f"{'='*55}")
    print(f"  Total indexed : {indexed:,}")
    print(f"  Time taken    : {elapsed:.1f}s")
    print(f"  Rate          : {indexed / elapsed:,.0f} docs/sec")
    print(f"{'='*55}")

    # Test query
    print("\n  BM25 test query: 'breast cancer'\n")
    results = es_index.search("breast cancer", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['nct_id']}] (score={r['score']:.2f})")
        print(f"     {r['title'][:90]}")
        print(f"     phase={r['phase']}  status={r['status']}")
        print()


def build_semantic_index(
    db_path: Path,
    faiss_path: str,
    model_name: str,
    batch_size: int,
) -> None:
    """Generate embeddings and build the FAISS semantic index.

    Queries SQLite directly for only the columns needed for embedding,
    avoiding loading full Trial objects to minimize memory usage.

    Args:
        db_path: Path to the SQLite database.
        faiss_path: Output path for the FAISS index file.
        model_name: HuggingFace model name for the embedder.
        batch_size: Encoding batch size.
    """
    import json
    import sqlite3

    import faiss

    from TrialMine.models.embeddings import TrialEmbedder

    # Step 1: Load model
    embedder = TrialEmbedder(model_name=model_name)

    # Step 2: Get total count, then stream chunks from SQLite
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    logger.info("Found %d trials in %s", n, db_path)

    # Step 3: Embed in chunks streamed from DB to avoid holding all rows in memory
    chunk_size = 2000
    faiss_index = faiss.IndexFlatIP(embedder.dimension)
    all_trial_ids: list[str] = []
    title_lookup: dict[str, str] = {}

    t0 = time.time()
    for offset in range(0, n, chunk_size):
        logger.info("Embedding chunk %d-%d / %d ...", offset, min(offset + chunk_size, n), n)

        rows = conn.execute(
            "SELECT nct_id, title, conditions, brief_summary FROM trials LIMIT ? OFFSET ?",
            (chunk_size, offset),
        ).fetchall()

        chunk_texts = []
        for nct_id, title, conditions_json, summary in rows:
            all_trial_ids.append(nct_id)
            title_lookup[nct_id] = (title or "N/A")[:90]
            parts = []
            if title:
                parts.append(title)
            if conditions_json:
                conds = json.loads(conditions_json)
                if conds:
                    parts.append(" ".join(conds))
            if summary:
                parts.append(summary)
            text = " [SEP] ".join(parts) if parts else ""
            chunk_texts.append(text[:2048])
        del rows

        chunk_embs = embedder.embed_batch(
            chunk_texts, batch_size=batch_size, show_progress=False,
        )
        faiss.normalize_L2(chunk_embs)
        faiss_index.add(chunk_embs)
        del chunk_texts, chunk_embs
    embed_elapsed = time.time() - t0
    conn.close()

    # Step 4: Save FAISS index and mapping
    mapping_path = str(Path(faiss_path).with_suffix(".json"))
    Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(faiss_path))
    with open(mapping_path, "w") as f:
        json.dump(all_trial_ids, f)

    print(f"\n{'='*55}")
    print("  SEMANTIC INDEX COMPLETE")
    print(f"{'='*55}")
    print(f"  Trials embedded : {n:,}")
    print(f"  Embedding time  : {embed_elapsed:.1f}s ({n / embed_elapsed:.1f} trials/sec)")
    print(f"  Embedding dim   : {embedder.dimension}")
    print(f"  Index size      : {Path(faiss_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  FAISS index     : {faiss_path}")
    print(f"  ID mapping      : {mapping_path}")
    print(f"{'='*55}")

    # Test query
    test_query = "immunotherapy for melanoma that has spread"
    print(f"\n  Semantic test query: '{test_query}'\n")
    query_emb = embedder.embed_text(test_query).reshape(1, -1)
    faiss.normalize_L2(query_emb)
    scores, indices = faiss_index.search(query_emb, 5)
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx >= 0:
            nct_id = all_trial_ids[idx]
            print(f"  {i}. [{nct_id}] (cosine={score:.4f})")
            print(f"     {title_lookup.get(nct_id, 'N/A')}")
            print()


def _avg_len(texts: list[str]) -> float:
    """Return the average character length of a list of strings."""
    return sum(len(t) for t in texts) / max(len(texts), 1)


def main() -> None:
    """Parse arguments and build requested indexes."""
    parser = argparse.ArgumentParser(description="Build search indexes")
    parser.add_argument("--db", default="data/trials.db", help="SQLite database path")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index", default="trials", help="Elasticsearch index name")
    parser.add_argument(
        "--faiss-path",
        default="data/trial_embeddings.faiss",
        help="Output path for FAISS index",
    )
    parser.add_argument(
        "--model",
        default="michiyasunaga/BioLinkBERT-base",
        help="HuggingFace embedding model name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build")
    parser.add_argument("--skip-semantic", action="store_true", help="Skip semantic index build")
    args = parser.parse_args()

    db_path = Path(args.db)

    # BM25 index (needs full Trial objects for structured fields)
    if not args.skip_bm25:
        from TrialMine.data.store import load_trials

        logger.info("Loading trials from %s ...", db_path)
        t0 = time.time()
        trials = load_trials(db_path)
        logger.info("Loaded %d trials in %.1fs", len(trials), time.time() - t0)

        if not trials:
            logger.error("No trials found in database. Run the download pipeline first.")
            return

        try:
            build_bm25_index(trials, args.es_url, args.index)
        except Exception:
            logger.exception("BM25 indexing failed (is Elasticsearch running?)")
        del trials  # free before semantic step

    # Semantic index (queries SQLite directly for minimal memory usage)
    if not args.skip_semantic:
        build_semantic_index(db_path, args.faiss_path, args.model, args.batch_size)


if __name__ == "__main__":
    main()
