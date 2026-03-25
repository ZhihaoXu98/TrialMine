"""Compare BM25, semantic, and hybrid search methods side-by-side.

Runs 20 oncology test queries across all three methods,
prints top-3 results for each, computes overlap, and saves
full results to data/evaluation/method_comparison.csv.

Requirements:
- Elasticsearch running with `trials` index
- FAISS index at data/trial_embeddings.faiss
- BioLinkBERT model available

Usage:
    python scripts/compare_methods.py
"""

import csv
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.models.embeddings import TrialEmbedder
from TrialMine.retrieval.bm25 import ElasticsearchIndex
from TrialMine.retrieval.hybrid import HybridRetriever
from TrialMine.retrieval.semantic import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Test queries ─────────────────────────────────────────────────────────────
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

TOP_N = 3  # Number of results to compare per method

# ── Paths ────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "data/trial_embeddings.faiss"
FAISS_MAPPING_PATH = "data/trial_embeddings.json"
OUTPUT_DIR = Path("data/evaluation")
OUTPUT_CSV = OUTPUT_DIR / "method_comparison.csv"


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis."""
    return text[:max_len] + "..." if len(text) > max_len else text


def run_bm25(es_index: ElasticsearchIndex, query: str, top_k: int) -> list[dict]:
    """Run BM25 search and return results."""
    return es_index.search(query=query, top_k=top_k)


def run_semantic(
    faiss_index: FAISSIndex,
    embedder: TrialEmbedder,
    es_index: ElasticsearchIndex,
    query: str,
    top_k: int,
) -> list[dict]:
    """Run semantic search and enrich with ES metadata."""
    query_embedding = embedder.embed_text(query)
    raw = faiss_index.search(query_embedding=query_embedding, top_k=top_k)
    results = []
    for nct_id, score in raw:
        doc = es_index.get_trial(nct_id) or {}
        results.append(
            {
                "nct_id": nct_id,
                "title": doc.get("title", ""),
                "score": score,
            }
        )
    return results


def main() -> None:
    """Run comparison across all methods and queries."""
    # ── Load components ──────────────────────────────────────────────────────
    print("=" * 80)
    print("TrialMine — Search Method Comparison")
    print("=" * 80)

    print("\nLoading Elasticsearch index...")
    es_index = ElasticsearchIndex()

    print("Loading FAISS index...")
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)

    print("Loading BioLinkBERT embedder...")
    embedder = TrialEmbedder()

    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    print(f"\nRunning {len(QUERIES)} queries across 3 methods...\n")

    # ── Collect results ──────────────────────────────────────────────────────
    csv_rows: list[dict] = []
    overlap_scores: list[float] = []

    for i, query in enumerate(QUERIES, 1):
        print("-" * 80)
        print(f"Query {i}/{len(QUERIES)}: \"{query}\"")
        print("-" * 80)

        # BM25
        t0 = time.perf_counter()
        bm25_results = run_bm25(es_index, query, top_k=TOP_N)
        bm25_ms = (time.perf_counter() - t0) * 1000

        # Semantic
        t0 = time.perf_counter()
        semantic_results = run_semantic(faiss_index, embedder, es_index, query, top_k=TOP_N)
        semantic_ms = (time.perf_counter() - t0) * 1000

        # Hybrid
        t0 = time.perf_counter()
        hybrid_results = hybrid.search(query=query, top_k=TOP_N)
        hybrid_ms = (time.perf_counter() - t0) * 1000

        # Print side-by-side top 3
        print(f"\n  BM25 top {TOP_N} ({bm25_ms:.0f} ms):")
        for j, r in enumerate(bm25_results[:TOP_N], 1):
            print(f"    {j}. [{r['nct_id']}] {truncate(r.get('title', ''))}")

        print(f"\n  Semantic top {TOP_N} ({semantic_ms:.0f} ms):")
        for j, r in enumerate(semantic_results[:TOP_N], 1):
            print(f"    {j}. [{r['nct_id']}] {truncate(r.get('title', ''))}")

        print(f"\n  Hybrid top {TOP_N} ({hybrid_ms:.0f} ms):")
        for j, r in enumerate(hybrid_results[:TOP_N], 1):
            src = r.get("source", "")
            print(f"    {j}. [{r['nct_id']}] {truncate(r.get('title', ''))}  ({src})")

        # Overlap: BM25 ∩ Semantic in top N
        bm25_ids = {r["nct_id"] for r in bm25_results[:TOP_N]}
        semantic_ids = {r["nct_id"] for r in semantic_results[:TOP_N]}
        overlap = len(bm25_ids & semantic_ids)
        overlap_scores.append(overlap / TOP_N)

        print(f"\n  Overlap BM25∩Semantic: {overlap}/{TOP_N}")
        print()

        # Collect CSV rows for all methods
        for method_name, results, elapsed in [
            ("bm25", bm25_results, bm25_ms),
            ("semantic", semantic_results, semantic_ms),
            ("hybrid", hybrid_results, hybrid_ms),
        ]:
            for rank, r in enumerate(results[:TOP_N], 1):
                csv_rows.append(
                    {
                        "query": query,
                        "method": method_name,
                        "rank": rank,
                        "nct_id": r["nct_id"],
                        "title": r.get("title", ""),
                        "score": r.get("score", 0.0),
                        "source": r.get("source", ""),
                        "bm25_rank": r.get("bm25_rank", ""),
                        "semantic_rank": r.get("semantic_rank", ""),
                        "search_time_ms": round(elapsed, 2),
                    }
                )

    # ── Summary ──────────────────────────────────────────────────────────────
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Queries evaluated: {len(QUERIES)}")
    print(f"Average BM25∩Semantic overlap (top {TOP_N}): {avg_overlap:.2f} ({avg_overlap * TOP_N:.1f}/{TOP_N})")
    print()

    # Per-query overlap table
    print(f"{'Query':<55} {'Overlap':>10}")
    print("-" * 67)
    for query, score in zip(QUERIES, overlap_scores):
        overlap_count = int(score * TOP_N)
        print(f"{truncate(query, 53):<55} {overlap_count}/{TOP_N}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "method",
                "rank",
                "nct_id",
                "title",
                "score",
                "source",
                "bm25_rank",
                "semantic_rank",
                "search_time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults saved to {OUTPUT_CSV} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
