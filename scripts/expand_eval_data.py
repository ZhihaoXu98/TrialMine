"""Expand evaluation data with 80 new training + 50 new test queries.

Pools top-20 results from BM25, semantic, and hybrid for each query,
labels the union with Claude Haiku, and saves to separate files for
training and held-out testing.

All query IDs are disjoint from existing datasets:
- Original training: IDs 0-19 (labeled_queries.jsonl)
- Original test: IDs 100-149 (test_labels.jsonl)
- NEW training: IDs 200-279 (train_labels_extra.jsonl)
- NEW test: IDs 300-349 (test_labels_v2.jsonl)

Usage:
    python scripts/expand_eval_data.py                  # full run
    python scripts/expand_eval_data.py --train-only     # just training queries
    python scripts/expand_eval_data.py --test-only      # just test queries
    python scripts/expand_eval_data.py --resume         # resume labeling
    python scripts/expand_eval_data.py --limit 5        # preview N queries per set

Requirements:
    - Elasticsearch running with `trials` index
    - FAISS index: data/faiss_finetuned.index + .json
    - ANTHROPIC_API_KEY in .env
"""

import argparse
import json
import logging
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


# ── 80 NEW training queries (IDs 200-279) ───────────────────────────────────
# Disjoint from the 20 original (IDs 0-19) and 50 fair-eval (IDs 100-149).
TRAIN_QUERIES = [
    # Solid tumors — specific subtypes not yet covered
    "squamous cell carcinoma head and neck cetuximab",
    "anaplastic thyroid cancer lenvatinib",
    "clear cell renal carcinoma sunitinib",
    "mucinous ovarian cancer treatment options",
    "adenoid cystic carcinoma salivary gland",
    "nasopharyngeal carcinoma concurrent chemo",
    "ampullary carcinoma clinical trial",
    "small bowel adenocarcinoma treatment",
    "penile cancer squamous cell clinical trial",
    "vulvar cancer recurrent treatment options",
    # Hematologic malignancies
    "follicular lymphoma rituximab maintenance",
    "mantle cell lymphoma ibrutinib relapsed",
    "Waldenström macroglobulinemia treatment",
    "hairy cell leukemia vemurafenib",
    "myelodysplastic syndrome azacitidine",
    "polycythemia vera ruxolitinib",
    "primary myelofibrosis JAK inhibitor",
    "T cell lymphoma romidepsin",
    "Burkitt lymphoma intensive chemotherapy",
    "plasma cell neoplasm clinical trial",
    # Biomarker / molecular driven
    "NTRK fusion solid tumor larotrectinib",
    "RET fusion lung cancer selpercatinib",
    "KRAS G12C mutation lung cancer sotorasib",
    "HER2 low breast cancer trastuzumab deruxtecan",
    "TMB high tumor mutational burden immunotherapy",
    "MSI-H microsatellite instability high pembrolizumab",
    "MET amplification non small cell lung cancer",
    "IDH1 mutation glioma vorasidenib",
    "BRCA2 mutation pancreatic cancer olaparib",
    "PD-L1 high expression first line immunotherapy",
    # Treatment modalities
    "CAR-T cell therapy relapsed DLBCL axicabtagene",
    "bispecific T cell engager multiple myeloma",
    "adoptive cell therapy TIL melanoma",
    "mRNA vaccine personalized neoantigen",
    "oncolytic virus therapy solid tumors",
    "radioligand therapy lutetium PSMA prostate",
    "photodynamic therapy superficial bladder",
    "stereotactic radiosurgery brain metastases",
    "proton beam therapy pediatric cancer",
    "intraperitoneal chemotherapy ovarian cancer",
    # Patient language — colloquial / emotional
    "my father has colon cancer spread to liver what are his options",
    "breast cancer came back in the other breast now what",
    "diagnosed with rare blood cancer need help finding treatment",
    "grandma has lung cancer never smoked any trials",
    "teenager with bone cancer is amputation the only option",
    "cancer in the bile duct what new treatments exist",
    "husband prostate cancer hormone therapy stopped working",
    "lymph node cancer that keeps relapsing",
    "eye cancer in toddler what clinical trials",
    "stage 3 stomach cancer after surgery what next",
    # Trial design / phase specific
    "window of opportunity trial breast cancer presurgical",
    "neoadjuvant immunotherapy trial solid tumors",
    "maintenance therapy after first line chemotherapy",
    "dose finding phase 1 advanced solid tumors",
    "randomized phase 3 adjuvant melanoma",
    "single arm phase 2 rare tumor",
    "platform trial adaptive design oncology",
    "combination immunotherapy checkpoint inhibitor",
    "biomarker driven basket trial NCI-MATCH",
    "real world evidence observational oncology study",
    # Specific clinical scenarios
    "brain metastases from breast cancer leptomeningeal",
    "malignant pleural effusion treatment clinical trial",
    "cancer related fatigue intervention study",
    "skeletal related events bone metastases denosumab",
    "cancer cachexia treatment clinical trial",
    "radiation dermatitis prevention clinical trial",
    "chemotherapy induced neuropathy treatment",
    "immune related adverse events management trial",
    "post transplant lymphoproliferative disorder",
    "secondary cancer after radiation therapy",
    # Demographics and special populations
    "pregnant woman diagnosed with cancer",
    "cancer treatment for patient with kidney failure",
    "elderly patient frail cancer treatment reduced dose",
    "adolescent young adult AYA cancer survivorship",
    "Hispanic Latino cancer clinical trial enrollment",
]

# ── 50 NEW test queries (IDs 300-349) ────────────────────────────────────────
# Disjoint from ALL training queries (IDs 0-19, 100-149, 200-279).
TEST_QUERIES = [
    # Solid tumors — not in any training set
    "uterine leiomyosarcoma advanced treatment",
    "gallbladder cancer unresectable treatment",
    "appendiceal cancer peritoneal metastases HIPEC",
    "gestational trophoblastic neoplasia resistant",
    "anal canal squamous cell carcinoma chemoradiation",
    "laryngeal cancer organ preservation",
    "pleural mesothelioma pemetrexed cisplatin",
    "adrenal gland tumor incidentaloma clinical trial",
    "orbital rhabdomyosarcoma pediatric",
    "desmoid tumor aggressive fibromatosis treatment",
    # Hematologic — not in training
    "acute promyelocytic leukemia ATRA arsenic",
    "splenic marginal zone lymphoma treatment",
    "angioimmunoblastic T cell lymphoma",
    "blastic plasmacytoid dendritic cell neoplasm",
    "systemic mastocytosis midostaurin avapritinib",
    # Biomarker queries not in training
    "ERBB2 amplification gastric cancer",
    "NF1 mutation malignant peripheral nerve sheath tumor",
    "TP53 mutated AML venetoclax",
    "FGFR3 mutation urothelial carcinoma erdafitinib",
    "ALK rearrangement anaplastic large cell lymphoma",
    # Patient language — different style from training
    "mom told she has six months what experimental treatments",
    "second opinion for rare sarcoma in the abdomen",
    "is there anything new for stage 4 melanoma that spread everywhere",
    "three year old with leukemia relapsed after chemo",
    "grandpa liver cancer too old for surgery any trials",
    "wife ovarian cancer platinum resistant need options",
    "son diagnosed with Hodgkin lymphoma best treatment 2026",
    "uncle throat cancer from HPV immunotherapy options",
    "friend pancreas cancer inoperable clinical trials near me",
    "cancer in the lining of the chest what can be done",
    # Treatment-focused not in training
    "ctDNA minimal residual disease guided therapy",
    "tumor infiltrating lymphocyte therapy beyond melanoma",
    "CRISPR gene editing cancer therapy clinical trial",
    "cancer vaccine after surgery to prevent recurrence",
    "hyperthermia combined with chemotherapy",
    # Specific scenarios not in training
    "cancer in pregnancy first trimester treatment options",
    "Wilms tumor bilateral pediatric nephron sparing",
    "Li-Fraumeni syndrome cancer screening protocol",
    "von Hippel-Lindau disease renal cell carcinoma",
    "BRCA1 carrier prophylactic ovarian cancer prevention trial",
    # Comorbidity / complex patients
    "cancer treatment HIV positive patient",
    "autoimmune disease lupus and cancer immunotherapy safe",
    "heart transplant patient diagnosed with lymphoma",
    "dialysis patient with kidney cancer treatment",
    "Down syndrome child with leukemia clinical trial",
    # Underserved / health equity
    "Native American cancer clinical trial access",
    "rural area cancer treatment telemedicine trial",
    "low income cancer patient financial assistance trial",
    "non English speaking cancer clinical trial",
    "veteran military cancer exposure agent orange",
]

OUTPUT_DIR = Path("data/evaluation")
TRAIN_OUTPUT = OUTPUT_DIR / "train_labels_extra.jsonl"
TEST_OUTPUT = OUTPUT_DIR / "test_labels_v2.jsonl"

EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"

POOL_TOP_K = 20

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
    from TrialMine.retrieval.hybrid import HybridRetriever

    bm25_results = es_index.search(query=query, top_k=top_k)
    bm25_ids = {r["nct_id"] for r in bm25_results}

    query_emb = embedder.embed_text(query)
    semantic_raw = faiss_index.search(query_embedding=query_emb, top_k=top_k)
    semantic_ids = {nct_id for nct_id, _ in semantic_raw}

    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)
    hybrid_results = hybrid.search(query=query, top_k=top_k)
    hybrid_ids = {r["nct_id"] for r in hybrid_results}

    seen: dict[str, dict] = {}
    for r in hybrid_results:
        seen[r["nct_id"]] = r
    for r in bm25_results:
        if r["nct_id"] not in seen:
            seen[r["nct_id"]] = r
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


def label_query_set(
    queries: list[str],
    id_offset: int,
    output_file: Path,
    es_index,
    faiss_index,
    embedder,
    elig_lookup: dict[str, str],
    resume: bool = False,
) -> int:
    """Pool results, label with Claude Haiku, and save to JSONL.

    Args:
        queries: List of query strings.
        id_offset: Starting query ID (e.g., 200 for training, 300 for test).
        output_file: Path to save JSONL labels.
        es_index: Elasticsearch index.
        faiss_index: FAISS index.
        embedder: Trial embedder.
        elig_lookup: Dict mapping nct_id to eligibility text.
        resume: Whether to skip already-labeled pairs.

    Returns:
        Number of new labels created.
    """
    import anthropic

    # Pool results
    logger.info("Pooling results for %d queries (IDs %d-%d)...",
                len(queries), id_offset, id_offset + len(queries) - 1)
    query_results: list[list[dict]] = []
    for i, query in enumerate(queries):
        pooled = pool_from_all_methods(query, es_index, faiss_index, embedder)
        query_results.append(pooled)
        logger.info("Query %d: %d unique trials pooled — %s",
                     id_offset + i, len(pooled), query[:50])

    total_pairs = sum(len(r) for r in query_results)
    logger.info("Total pairs to label: %d", total_pairs)

    # Resume support
    existing = set()
    if resume:
        existing = load_existing_labels(output_file)
        logger.info("Resuming: %d pairs already labeled", len(existing))

    # Label with Claude Haiku
    client = anthropic.Anthropic()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if resume else "w"
    labeled_count = 0
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}

    with open(output_file, mode) as f:
        for qi, (query, results) in enumerate(zip(queries, query_results)):
            query_id = id_offset + qi

            for trial in results:
                nct_id = trial["nct_id"]
                if (query_id, nct_id) in existing:
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
                    print(f"  Labeled {labeled_count} pairs... "
                          f"(query {qi+1}/{len(queries)})")

                time.sleep(0.1)

    # Summary
    print(f"\n  Labels saved to {output_file}")
    print(f"  New labels: {labeled_count}")
    print(f"  Score distribution:")
    for score in [0, 1, 2, 3]:
        count = score_dist.get(score, 0)
        pct = count / max(labeled_count, 1) * 100
        print(f"    {score}: {count:>4} ({pct:5.1f}%)")
    errors = score_dist.get(-1, 0)
    if errors:
        print(f"   -1: {errors:>4} (errors)")

    return labeled_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Expand evaluation data with new training and test queries",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only label training queries (IDs 200-279)",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Only label test queries (IDs 300-349)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume labeling from checkpoint",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of queries per set (0=all)",
    )
    parser.add_argument(
        "--db", default="data/trials.db",
        help="SQLite database path",
    )
    return parser.parse_args()


def main() -> None:
    """Pool results, label with Claude Haiku, save training + test labels."""
    import sqlite3

    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.semantic import FAISSIndex

    args = parse_args()

    do_train = not args.test_only
    do_test = not args.train_only

    train_queries = TRAIN_QUERIES
    test_queries = TEST_QUERIES
    if args.limit > 0:
        train_queries = train_queries[:args.limit]
        test_queries = test_queries[:args.limit]

    # Check prerequisites
    if not Path(FAISS_INDEX).exists():
        logger.error("FAISS index not found: %s", FAISS_INDEX)
        sys.exit(1)

    # Load retrieval components
    logger.info("Loading retrieval components...")
    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)

    # Load eligibility from SQLite
    logger.info("Loading eligibility criteria from SQLite...")
    conn = sqlite3.connect(args.db)
    elig_rows = conn.execute(
        "SELECT nct_id, eligibility_criteria FROM trials"
    ).fetchall()
    conn.close()
    elig_lookup = {nct_id: (elig or "") for nct_id, elig in elig_rows}
    logger.info("Loaded eligibility for %d trials", len(elig_lookup))

    total_labeled = 0

    # Label training queries
    if do_train:
        print(f"\n{'='*60}")
        print(f"  LABELING TRAINING QUERIES ({len(train_queries)} queries, IDs 200+)")
        print(f"{'='*60}")
        n = label_query_set(
            queries=train_queries,
            id_offset=200,
            output_file=TRAIN_OUTPUT,
            es_index=es_index,
            faiss_index=faiss_index,
            embedder=embedder,
            elig_lookup=elig_lookup,
            resume=args.resume,
        )
        total_labeled += n

    # Label test queries
    if do_test:
        print(f"\n{'='*60}")
        print(f"  LABELING TEST QUERIES ({len(test_queries)} queries, IDs 300+)")
        print(f"{'='*60}")
        n = label_query_set(
            queries=test_queries,
            id_offset=300,
            output_file=TEST_OUTPUT,
            es_index=es_index,
            faiss_index=faiss_index,
            embedder=embedder,
            elig_lookup=elig_lookup,
            resume=args.resume,
        )
        total_labeled += n

    print(f"\n{'='*60}")
    print(f"  DONE — {total_labeled} total new labels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
