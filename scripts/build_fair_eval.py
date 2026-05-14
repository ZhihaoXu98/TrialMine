"""Build a held-out test set for fair ablation evaluation.

Generates 50 NEW queries (disjoint from the 20 training queries), pools
top-20 results from BM25, semantic, and hybrid, labels the union with
Claude Haiku, then runs evaluation on ALL 5 pipeline stages.

This fixes three biases in the original evaluation:
1. Train-test overlap: LightGBM was trained on the 20 training queries
2. Label pooling bias: Labels only existed for hybrid results
3. Sample size: 20 queries was too few

Usage:
    python scripts/build_fair_eval.py                    # full run
    python scripts/build_fair_eval.py --label-only       # just label, skip eval
    python scripts/build_fair_eval.py --eval-only        # just eval (labels exist)
    python scripts/build_fair_eval.py --resume           # resume labeling
    python scripts/build_fair_eval.py --limit 5          # preview 5 queries

Requirements:
    - Elasticsearch running with `trials` index
    - FAISS index: data/faiss_finetuned.index + .json
    - Cross-encoder: models/cross-encoder/fine-tuned-v2/
    - LightGBM ranker: models/ranker/v1/model.lgb
    - ANTHROPIC_API_KEY in .env
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── 50 held-out test queries (disjoint from the 20 training queries) ──────────
# Mix of clinical terminology, patient language, misspellings, rare cancers,
# specific treatments, and demographic constraints.
TEST_QUERIES = [
    # Different cancer types not in training set
    "thyroid cancer radioactive iodine treatment",
    "cervical cancer HPV positive immunotherapy",
    "esophageal cancer chemoradiation",
    "stomach cancer HER2 positive trastuzumab",
    "testicular cancer cisplatin resistant",
    "endometrial cancer pembrolizumab",
    "cholangiocarcinoma bile duct cancer targeted therapy",
    "adrenocortical carcinoma advanced treatment",
    "thymoma thymic carcinoma clinical trials",
    "retinoblastoma treatment for children",
    # Patient language style
    "my mom has stage 4 colon cancer what trials are available",
    "lung cancer that spread to brain looking for treatment options",
    "just diagnosed with lymphoma what are the newest treatments",
    "kidney cancer surgery not an option what else is there",
    "my child was diagnosed with bone cancer osteosarcoma",
    "pancreatic cancer not responding to chemo anymore",
    "ovarian cancer came back after remission",
    "leukemia in adults over 60 treatment options",
    "skin cancer keeps coming back after surgery",
    "brain tumor that is not operable",
    # Specific treatments / biomarkers
    "ALK positive lung cancer crizotinib resistance",
    "BRAF V600E melanoma dabrafenib trametinib",
    "microsatellite stable colorectal cancer immunotherapy",
    "PI3K inhibitor breast cancer clinical trial",
    "FGFR2 fusion cholangiocarcinoma pemigatinib",
    "CDK4/6 inhibitor combination hormone receptor positive",
    "bispecific antibody lymphoma treatment",
    "antibody drug conjugate solid tumors",
    "tumor treating fields glioblastoma",
    "PARP inhibitor prostate cancer BRCA mutation",
    # Phase and status specific
    "phase 1 dose escalation solid tumors first in human",
    "phase 3 adjuvant breast cancer trial recruiting",
    "randomized controlled trial lung cancer first line",
    "expanded access compassionate use pancreatic",
    "phase 2 basket trial multiple tumor types",
    # Rare cancers
    "Merkel cell carcinoma immunotherapy",
    "gastrointestinal stromal tumor imatinib resistant",
    "pheochromocytoma paraganglioma treatment",
    "medulloblastoma pediatric recurrent",
    "Ewing sarcoma metastatic treatment",
    # With misspellings / abbreviations
    "non hodgkins lympoma treatment options",
    "DLBCL diffuse large b cell lymphoma R-CHOP",
    "HCC hepatocelular carcinoma sorafenib",
    "AML acute myeloid lukemia elderly",
    "SCLC small cell lung cancer relapsed",
    # Demographic + condition combinations
    "breast cancer in men clinical trial",
    "lung cancer never smoker young adult",
    "colorectal cancer under 40 early onset",
    "prostate cancer African American high risk",
    "pediatric brain tumor diffuse intrinsic pontine glioma",
]

OUTPUT_DIR = Path("data/evaluation")
TEST_LABELS_FILE = OUTPUT_DIR / "test_labels_v2.jsonl"

EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"
CE_MODEL = "models/cross-encoder/fine-tuned-v2"
RANKER_MODEL = "models/ranker/v3-regularized/model.lgb"

POOL_TOP_K = 20  # top-K from each method to pool

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


def pool_from_all_methods(
    query: str,
    es_index,
    faiss_index,
    embedder,
    top_k: int = POOL_TOP_K,
) -> list[dict]:
    """Pool top-K results from BM25, semantic, and hybrid.

    Args:
        query: Search query string.
        es_index: Elasticsearch index.
        faiss_index: FAISS index.
        embedder: Trial embedder.
        top_k: Number of results from each method.

    Returns:
        Deduplicated list of trial dicts with 'retrieved_by' metadata.
    """
    from TrialMine.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

    # BM25
    bm25_results = es_index.search(query=query, top_k=top_k)
    bm25_ids = {r["nct_id"] for r in bm25_results}

    # Semantic
    query_emb = embedder.embed_text(query)
    semantic_raw = faiss_index.search(query_embedding=query_emb, top_k=top_k)
    semantic_ids = {nct_id for nct_id, _ in semantic_raw}

    # Hybrid (top_k from RRF fusion of 200 from each)
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)
    hybrid_results = hybrid.search(query=query, top_k=top_k)
    hybrid_ids = {r["nct_id"] for r in hybrid_results}

    # Pool: union of all three methods
    all_ids = bm25_ids | semantic_ids | hybrid_ids
    seen: dict[str, dict] = {}

    # Prefer hybrid result metadata (most complete)
    for r in hybrid_results:
        seen[r["nct_id"]] = r

    for r in bm25_results:
        if r["nct_id"] not in seen:
            seen[r["nct_id"]] = r

    # Semantic results need metadata enrichment
    for nct_id, score in semantic_raw:
        if nct_id not in seen:
            doc = es_index.get_trial(nct_id) or {}
            seen[nct_id] = {
                "nct_id": nct_id,
                "title": doc.get("title", ""),
                "conditions": doc.get("conditions", ""),
                "phase": doc.get("phase"),
                "status": doc.get("status"),
                "enrollment": doc.get("enrollment"),
                "score": score,
            }

    # Tag sources
    pooled = []
    for nct_id, trial in seen.items():
        sources = []
        if nct_id in bm25_ids:
            sources.append("bm25")
        if nct_id in semantic_ids:
            sources.append("semantic")
        if nct_id in hybrid_ids:
            sources.append("hybrid")
        trial["retrieved_by"] = "+".join(sources)
        pooled.append(trial)

    return pooled


def label_pair(client, query: str, trial: dict, eligibility: str) -> tuple[int, str]:
    """Call Claude Haiku to rate a single (query, trial) pair.

    Args:
        client: Anthropic API client.
        query: Patient search query.
        trial: Dict with trial metadata.
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
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return int(result["score"]), result.get("reason", "")
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Parse error for %s: %s", trial.get("nct_id"), exc)
        return -1, f"parse_error: {str(exc)[:100]}"
    except Exception as exc:
        logger.error("API error for %s: %s", trial.get("nct_id"), exc)
        return -1, f"api_error: {str(exc)[:100]}"


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


def load_test_labels(labels_file: Path) -> dict[int, dict]:
    """Load test labels grouped by query_id.

    Args:
        labels_file: Path to test_labels.jsonl.

    Returns:
        Dict mapping query_id to query text, relevance scores, and relevant IDs.
    """
    queries: dict[int, dict] = {}
    with open(labels_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            if qid not in queries:
                queries[qid] = {
                    "query": rec["query"],
                    "relevance": {},
                    "relevant_ids": set(),
                }
            score = rec["relevance"]
            if score < 0:
                continue
            queries[qid]["relevance"][rec["nct_id"]] = score
            if score >= 2:
                queries[qid]["relevant_ids"].add(rec["nct_id"])
    return queries


def run_fair_evaluation(labels: dict[int, dict]) -> None:
    """Run the fair ablation evaluation on held-out test queries.

    Args:
        labels: Test labels from load_test_labels().
    """
    from TrialMine.evaluation.metrics import mrr, ndcg_at_k
    from TrialMine.models.cross_encoder import CrossEncoderReranker
    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.models.ranker import RankingBlender, compute_features, FEATURE_NAMES
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.hybrid import HybridRetriever
    from TrialMine.retrieval.semantic import FAISSIndex

    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    reranker = None
    if Path(CE_MODEL).exists():
        reranker = CrossEncoderReranker(model_name=CE_MODEL)

    blender = None
    if Path(RANKER_MODEL).exists():
        blender = RankingBlender(model_path=RANKER_MODEL)

    n_queries = len(labels)
    eval_k = 10

    # ── Evaluate each method ──────────────────────────────────────────────
    methods: dict[str, dict] = {}

    # 1. BM25 only
    logger.info("Evaluating BM25 only...")
    ndcg5, ndcg10, mrr_list, lats = [], [], [], []
    for qid in sorted(labels.keys()):
        q = labels[qid]
        t0 = time.perf_counter()
        results = es_index.search(query=q["query"], top_k=eval_k)
        lats.append((time.perf_counter() - t0) * 1000)
        ids = [r["nct_id"] for r in results]
        ndcg5.append(ndcg_at_k(ids, q["relevance"], k=5))
        ndcg10.append(ndcg_at_k(ids, q["relevance"], k=10))
        mrr_list.append(mrr(ids, q["relevant_ids"]))
    methods["BM25 only"] = {"ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr_list, "latencies": lats}

    # 2. Semantic only
    logger.info("Evaluating Semantic only...")
    ndcg5, ndcg10, mrr_list, lats = [], [], [], []
    for qid in sorted(labels.keys()):
        q = labels[qid]
        t0 = time.perf_counter()
        qemb = embedder.embed_text(q["query"])
        results = faiss_index.search(query_embedding=qemb, top_k=eval_k)
        lats.append((time.perf_counter() - t0) * 1000)
        ids = [nct_id for nct_id, _ in results]
        ndcg5.append(ndcg_at_k(ids, q["relevance"], k=5))
        ndcg10.append(ndcg_at_k(ids, q["relevance"], k=10))
        mrr_list.append(mrr(ids, q["relevant_ids"]))
    methods["Semantic only"] = {"ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr_list, "latencies": lats}

    # 3. Hybrid
    logger.info("Evaluating Hybrid...")
    ndcg5, ndcg10, mrr_list, lats = [], [], [], []
    for qid in sorted(labels.keys()):
        q = labels[qid]
        t0 = time.perf_counter()
        results = hybrid.search(query=q["query"], top_k=eval_k)
        lats.append((time.perf_counter() - t0) * 1000)
        ids = [r["nct_id"] for r in results]
        ndcg5.append(ndcg_at_k(ids, q["relevance"], k=5))
        ndcg10.append(ndcg_at_k(ids, q["relevance"], k=10))
        mrr_list.append(mrr(ids, q["relevant_ids"]))
    methods["Hybrid (BM25 + Semantic)"] = {"ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr_list, "latencies": lats}

    # 4. + Cross-Encoder
    if reranker:
        logger.info("Evaluating + Cross-Encoder...")
        ndcg5, ndcg10, mrr_list, lats = [], [], [], []
        for qid in sorted(labels.keys()):
            q = labels[qid]
            t0 = time.perf_counter()
            candidates = hybrid.search(query=q["query"], top_k=50)
            for c in candidates:
                if not c.get("brief_summary"):
                    doc = es_index.get_trial(c["nct_id"])
                    if doc:
                        c["brief_summary"] = doc.get("brief_summary", "")
                parts = [c.get("title", ""), c.get("conditions", ""), c.get("brief_summary", "")]
                c["trial_text"] = (" [SEP] ".join(p for p in parts if p))[:2048]
            ranked = reranker.rerank(q["query"], candidates, top_k=eval_k)
            lats.append((time.perf_counter() - t0) * 1000)
            ids = [r["nct_id"] for r in ranked]
            ndcg5.append(ndcg_at_k(ids, q["relevance"], k=5))
            ndcg10.append(ndcg_at_k(ids, q["relevance"], k=10))
            mrr_list.append(mrr(ids, q["relevant_ids"]))
        methods["+ Cross-Encoder Re-ranking"] = {"ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr_list, "latencies": lats}

    # 5. + Metadata Blender
    if reranker and blender:
        logger.info("Evaluating + Metadata Blender...")
        ndcg5, ndcg10, mrr_list, lats = [], [], [], []
        for qid in sorted(labels.keys()):
            q = labels[qid]
            t0 = time.perf_counter()
            candidates = hybrid.search(query=q["query"], top_k=50)

            # Get individual scores
            bm25_res = es_index.search(query=q["query"], top_k=500)
            bm25_map = {r["nct_id"]: r["score"] for r in bm25_res}
            qemb = embedder.embed_text(q["query"])
            sem_res = faiss_index.search(query_embedding=qemb, top_k=500)
            sem_map = {nct_id: s for nct_id, s in sem_res}

            for c in candidates:
                nct_id = c["nct_id"]
                c["bm25_score"] = bm25_map.get(nct_id, 0.0)
                c["semantic_score"] = sem_map.get(nct_id, 0.0)
                doc = es_index.get_trial(nct_id)
                if doc:
                    c["brief_summary"] = doc.get("brief_summary", "")
                    c["eligibility_criteria"] = doc.get("eligibility_criteria", "")
                parts = [c.get("title", ""), c.get("conditions", ""), c.get("brief_summary", "")]
                c["trial_text"] = (" [SEP] ".join(p for p in parts if p))[:2048]

            # CE scoring
            texts = [c["trial_text"] for c in candidates]
            raw_scores = reranker.score(q["query"], texts)
            for c, raw in zip(candidates, raw_scores):
                c["cross_encoder_score"] = 1 / (1 + math.exp(-raw))

            ranked = blender.rerank(q["query"], candidates, top_k=eval_k)
            lats.append((time.perf_counter() - t0) * 1000)
            ids = [r["nct_id"] for r in ranked]
            ndcg5.append(ndcg_at_k(ids, q["relevance"], k=5))
            ndcg10.append(ndcg_at_k(ids, q["relevance"], k=10))
            mrr_list.append(mrr(ids, q["relevant_ids"]))
        methods["+ Metadata Blender"] = {"ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr_list, "latencies": lats}

    # ── Print THE TABLE ───────────────────────────────────────────────────
    def bootstrap_ci(values, n=1000):
        arr = np.array(values)
        rng = np.random.RandomState(42)
        means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
        means = np.array(means)
        low, high = np.percentile(means, [2.5, 97.5])
        half = (high - low) / 2
        return arr.mean(), half

    print(f"\n\n{'='*85}")
    print(f"  FAIR ABLATION TABLE ({n_queries} held-out test queries)")
    print(f"{'='*85}\n")
    print(f"| {'Method':<33} | {'NDCG@5':^13} | {'NDCG@10':^13} | {'MRR':^13} | {'Latency':^10} |")
    print(f"|{'-'*35}|{'-'*15}|{'-'*15}|{'-'*15}|{'-'*12}|")

    for method, res in methods.items():
        n5_mean, n5_ci = bootstrap_ci(res["ndcg5"])
        n10_mean, n10_ci = bootstrap_ci(res["ndcg10"])
        m_mean, m_ci = bootstrap_ci(res["mrr"])
        lat = np.median(res["latencies"])
        print(
            f"| {method:<33} | {n5_mean:.3f}\u00b1{n5_ci:.2f}  "
            f"| {n10_mean:.3f}\u00b1{n10_ci:.02f}  "
            f"| {m_mean:.3f}\u00b1{m_ci:.02f}  "
            f"| {lat:>6.0f}ms   |"
        )

    print(f"\n{'='*85}")
    print(f"  Bootstrap 95% CI, {n_queries} queries, labels pooled from BM25+Semantic+Hybrid")
    print(f"  LightGBM trained on 150 queries — this is a true held-out test")
    print(f"{'='*85}")

    # ── Per-query breakdown ───────────────────────────────────────────────
    query_ids = sorted(labels.keys())
    print(f"\n  Per-query NDCG@10 (first 20):")
    method_names = list(methods.keys())
    short_names = [m.replace("Hybrid (BM25 + Semantic)", "Hybrid").replace("+ Cross-Encoder Re-ranking", "+ CE").replace("+ Metadata Blender", "+ LGB") for m in method_names]
    header = f"  {'Q#':<4}" + "".join(f"{s:>8}" for s in short_names) + f"  {'Query':<42}"
    print(header)
    print(f"  {'-'*len(header)}")

    for i, qid in enumerate(query_ids[:20]):
        parts = [f"{qid:<4}"]
        for m in method_names:
            parts.append(f"{methods[m]['ndcg10'][i]:>8.3f}")
        parts.append(f"  {labels[qid]['query'][:40]:<42}")
        print(f"  {''.join(parts)}")

    if len(query_ids) > 20:
        print(f"  ... ({len(query_ids) - 20} more queries)")

    # ── Save fair report ──────────────────────────────────────────────────
    report_path = Path("docs/fair-evaluation-report.md")
    report_lines = [
        "# TrialMine Fair Ablation Evaluation Report\n",
        f"\n## Fair Ablation Table ({n_queries} held-out test queries)\n",
    ]
    report_lines.append(f"\n| {'Method':<33} | {'NDCG@5':^13} | {'NDCG@10':^13} | {'MRR':^13} | {'Latency':^10} |")
    report_lines.append(f"|{'-'*35}|{'-'*15}|{'-'*15}|{'-'*15}|{'-'*12}|")
    for method, res in methods.items():
        n5_mean, n5_ci = bootstrap_ci(res["ndcg5"])
        n10_mean, n10_ci = bootstrap_ci(res["ndcg10"])
        m_mean, m_ci = bootstrap_ci(res["mrr"])
        lat = np.median(res["latencies"])
        report_lines.append(
            f"| {method:<33} | {n5_mean:.3f}\u00b1{n5_ci:.02f}  "
            f"| {n10_mean:.3f}\u00b1{n10_ci:.02f}  "
            f"| {m_mean:.3f}\u00b1{m_ci:.02f}  "
            f"| {lat:>6.0f}ms   |"
        )
    report_lines.append(f"\n*Bootstrap 95% CI from 1000 samples over {n_queries} held-out queries.*")
    report_lines.append(f"\n*LightGBM trained on 150 queries — true held-out test.*")
    report_lines.append("\n\n## Key Differences from Original Evaluation\n")
    report_lines.append("1. **Held-out queries**: 50 new queries never seen during LightGBM training (150 training queries)")
    report_lines.append("2. **Pooled labeling**: Labels from BM25 + Semantic + Hybrid union (no method bias)")
    report_lines.append("3. **Expanded training**: LightGBM trained on 150 queries (~6,200 labeled pairs) vs original 20")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    logger.info("Fair report saved to %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build fair held-out evaluation dataset and run ablation",
    )
    parser.add_argument("--label-only", action="store_true", help="Only label, skip eval")
    parser.add_argument("--eval-only", action="store_true", help="Only eval (labels must exist)")
    parser.add_argument("--resume", action="store_true", help="Resume labeling from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries (0=all)")
    parser.add_argument("--db", default="data/trials.db", help="SQLite database path")
    return parser.parse_args()


def main() -> None:
    """Build held-out test set and run fair evaluation."""
    import sqlite3

    args = parse_args()
    queries = TEST_QUERIES
    if args.limit > 0:
        queries = queries[:args.limit]

    if not args.eval_only:
        # ── Setup retrieval ───────────────────────────────────────────────
        from TrialMine.models.embeddings import TrialEmbedder
        from TrialMine.retrieval.bm25 import ElasticsearchIndex
        from TrialMine.retrieval.semantic import FAISSIndex

        if not Path(FAISS_INDEX).exists():
            logger.error("FAISS index not found: %s", FAISS_INDEX)
            sys.exit(1)

        logger.info("Loading retrieval components...")
        es_index = ElasticsearchIndex()
        embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
        faiss_index = FAISSIndex()
        faiss_index.load(FAISS_INDEX, FAISS_MAPPING)

        # Load eligibility from SQLite
        logger.info("Loading eligibility criteria from SQLite...")
        conn = sqlite3.connect(args.db)
        elig_rows = conn.execute("SELECT nct_id, eligibility_criteria FROM trials").fetchall()
        conn.close()
        elig_lookup = {nct_id: (elig or "") for nct_id, elig in elig_rows}
        logger.info("Loaded eligibility for %d trials", len(elig_lookup))

        # ── Pool results from all methods ─────────────────────────────────
        logger.info("Pooling results from BM25 + Semantic + Hybrid for %d queries...", len(queries))
        query_results: list[list[dict]] = []

        for i, query in enumerate(queries):
            pooled = pool_from_all_methods(query, es_index, faiss_index, embedder)
            query_results.append(pooled)
            logger.info(
                "Query %d: %d unique trials pooled — %s",
                i, len(pooled), query[:50],
            )

        total_pairs = sum(len(r) for r in query_results)
        logger.info("Total unique (query, trial) pairs to label: %d", total_pairs)

        # ── Free model memory ─────────────────────────────────────────────
        del embedder, faiss_index
        import gc
        gc.collect()

        # ── Resume support ────────────────────────────────────────────────
        existing = set()
        if args.resume:
            existing = load_existing_labels(TEST_LABELS_FILE)
            logger.info("Resuming: %d pairs already labeled", len(existing))

        # ── Label with Claude Haiku ───────────────────────────────────────
        import anthropic
        client = anthropic.Anthropic()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        mode = "a" if args.resume else "w"
        labeled_count = 0
        skipped = 0
        score_dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}

        # Use query IDs starting at 100 to avoid overlap with training (0-19)
        with open(TEST_LABELS_FILE, mode) as f:
            for qi, (query, results) in enumerate(zip(queries, query_results)):
                query_id = 100 + qi

                for trial in results:
                    nct_id = trial["nct_id"]
                    if (query_id, nct_id) in existing:
                        skipped += 1
                        continue

                    eligibility = elig_lookup.get(nct_id, "")
                    score, reason = label_pair(client, query, trial, eligibility)

                    record = {
                        "query_id": query_id,
                        "query": query,
                        "nct_id": nct_id,
                        "trial_title": trial.get("title", ""),
                        "relevance": score,
                        "reason": reason,
                        "retrieved_by": trial.get("retrieved_by", "unknown"),
                        "labeler": "claude-haiku",
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    score_dist[score] = score_dist.get(score, 0) + 1
                    labeled_count += 1

                    if labeled_count % 50 == 0:
                        print(f"  Labeled {labeled_count} pairs... (query {qi+1}/{len(queries)})")

                    time.sleep(0.1)

        # Summary
        print(f"\n{'='*55}")
        print(f"  LABELING COMPLETE")
        print(f"{'='*55}")
        print(f"  New labels:  {labeled_count}")
        print(f"  Skipped:     {skipped}")
        print(f"  Output:      {TEST_LABELS_FILE}")
        print(f"\n  Score distribution:")
        for score in [0, 1, 2, 3]:
            count = score_dist.get(score, 0)
            pct = count / max(labeled_count, 1) * 100
            print(f"    {score}: {count:>4} ({pct:5.1f}%)")
        errors = score_dist.get(-1, 0)
        if errors:
            print(f"   -1: {errors:>4} (errors)")

    if args.label_only:
        return

    # ── Run fair evaluation ───────────────────────────────────────────────
    if not TEST_LABELS_FILE.exists():
        logger.error("Test labels not found: %s", TEST_LABELS_FILE)
        sys.exit(1)

    labels = load_test_labels(TEST_LABELS_FILE)
    total = sum(len(q["relevance"]) for q in labels.values())
    logger.info("Loaded %d test labels across %d queries", total, len(labels))

    run_fair_evaluation(labels)


if __name__ == "__main__":
    main()
