"""Build the full 65-query labeled evaluation dataset (interview-proof).

For each of 65 queries (30 existing + 30 NEW across 7 categories + 5 rare_explicit
from Phase A11), runs the full production pipeline (BM25 + Semantic + RRF +
Cross-Encoder + LightGBM) and labels the top-K results with Claude Haiku on a 0-3
relevance scale.

Why this dataset matters:
    - Diverse: 7 categories (common, rare, pediatric, complex multi-fact,
      geographic, vague/lay, treatment-specific) — stresses different parts
      of the pipeline.
    - Pipeline-faithful: labels the EXACT output users would see, not an
      intermediate hybrid stage.
    - Resumable: writes one (query, trial) record per line, skips any
      (query_id, nct_id) already in the output.

Output: data/evaluation/full_labeled_dataset.jsonl

Usage:
    python scripts/build_full_eval_dataset.py                # full run (65 queries)
    python scripts/build_full_eval_dataset.py --limit 5      # preview first 5 queries
    python scripts/build_full_eval_dataset.py --resume       # skip already-labeled
    python scripts/build_full_eval_dataset.py --new-only     # just the 30 new queries
    python scripts/build_full_eval_dataset.py --rare-explicit-only  # just the 5 IDs 500-504
    python scripts/build_full_eval_dataset.py --top-k 10     # label top-10 instead of top-20

Requirements:
    - Elasticsearch running with `trials` index (`docker start es`)
    - FAISS index: data/faiss_finetuned.index + .json
    - Cross-encoder: models/cross-encoder/fine-tuned/
    - LightGBM ranker: models/ranker/v2/model.lgb
    - ANTHROPIC_API_KEY in .env
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


# ── 30 EXISTING queries (IDs 0-19 from labeled_queries.jsonl + 300-309 from test_labels_v2.jsonl) ──
# Picked to span training-era queries (0-19) and held-out test queries (300-309)
# so the unified set covers both seen and unseen styles.
EXISTING_QUERIES: list[tuple[int, str]] = [
    # IDs 0-19 — original 20 from labeled_queries.jsonl
    (0, "breast cancer hormone receptor positive phase 3"),
    (1, "immunotherapy for non-small cell lung cancer"),
    (2, "CAR-T cell therapy for pediatric leukemia"),
    (3, "clinical trial for glioblastoma that has come back"),
    (4, "melanomt with checkpoint inhibitors"),
    (5, "prostate cancer trials for men over 65"),
    (6, "pancreatic cancer that has spread to the liver"),
    (7, "triple negative breast cancer neoadjuvant"),
    (8, "colorectal cancer with microsatellite instability"),
    (9, "ovarian cancer PARP inhibitor maintenance"),
    (10, "stage 4 kidney cancer immunotherapy combination"),
    (11, "newly diagnosed multiple myeloma treatment"),
    (12, "HPV related head and neck cancer"),
    (13, "sarcoma clinical trials for young adults"),
    (14, "targeted therapy for EGFR mutated lung cancer"),
    (15, "liver cancer hepatocellular carcinoma systemic therapy"),
    (16, "bladder cancer BCG unresponsive"),
    (17, "chronic lymphocytic leukemia ibrutinib"),
    (18, "neuroblastoma high risk children"),
    (19, "mesothelioma immunotherapy combination"),
    # IDs 300-309 — first 10 held-out test queries from test_labels_v2.jsonl
    (300, "uterine leiomyosarcoma advanced treatment"),
    (301, "gallbladder cancer unresectable treatment"),
    (302, "appendiceal cancer peritoneal metastases HIPEC"),
    (303, "gestational trophoblastic neoplasia resistant"),
    (304, "anal canal squamous cell carcinoma chemoradiation"),
    (305, "laryngeal cancer organ preservation"),
    (306, "pleural mesothelioma pemetrexed cisplatin"),
    (307, "adrenal gland tumor incidentaloma clinical trial"),
    (308, "orbital rhabdomyosarcoma pediatric"),
    (309, "desmoid tumor aggressive fibromatosis treatment"),
]


# ── 30 NEW queries (IDs 400-429) — 7 categories ──────────────────────────────
# IDs 400-429 chosen to be disjoint from all prior eval sets (0-19, 100-149,
# 200-274, 300-349). The categories correspond to interview-style failure modes:
# common (sanity), rare (long-tail), pediatric (under-served population),
# complex (multi-fact patient profile — biggest test of the agent), geographic
# (location filter — currently NOT supported by retrieval, will likely surface
# weakly relevant matches), vague (intent-extraction stress), treatment-specific
# (modality-first phrasing).
NEW_QUERIES: list[tuple[int, str, str]] = [
    # Common (5) — what the system will see most often
    (400, "common", "breast cancer HR+ CDK4/6 inhibitor"),
    (401, "common", "NSCLC brain mets"),
    (402, "common", "colon cancer stage 2 high risk"),
    (403, "common", "prostate CRPC"),
    (404, "common", "ovarian BRCA mutation"),
    # Rare (5) — long-tail tumor types
    (405, "rare", "angiosarcoma scalp"),
    (406, "rare", "cholangiocarcinoma"),
    (407, "rare", "merkel cell carcinoma"),
    (408, "rare", "GIST imatinib resistant"),
    (409, "rare", "adrenocortical carcinoma"),
    # Pediatric (3) — under-represented in trial corpus
    (410, "pediatric", "medulloblastoma 6 year old"),
    (411, "pediatric", "wilms tumor relapsed"),
    (412, "pediatric", "childhood rhabdomyosarcoma"),
    # Complex (5) — multi-fact patient profile, biggest test of agent extraction
    (413, "complex", "58M EGFR exon 19 NSCLC failed osimertinib phase 2-3"),
    (414, "complex", "45F TNBC completed AC-T 3mo ago"),
    (415, "complex", "72M myeloma relapsed after lenalidomide bortezomib"),
    (416, "complex", "62F HER2+ metastatic breast post-trastuzumab progression"),
    (417, "complex", "55M MSI-high colorectal post-pembrolizumab progression"),
    # Geographic (3) — location filter (NOT yet wired into retrieval)
    (418, "geographic", "pancreatic cancer trials Texas"),
    (419, "geographic", "leukemia trials Europe"),
    (420, "geographic", "trials at MD Anderson"),
    # Vague (5) — lay/emotional/under-specified queries
    (421, "vague", "I have cancer what trials"),
    (422, "vague", "mom has bone cancer"),
    (423, "vague", "terminal cancer any options"),
    (424, "vague", "just diagnosed need help"),
    (425, "vague", "experimental treatments"),
    # Treatment-specific (4) — modality-first phrasing
    (426, "treatment", "CAR-T lymphoma"),
    (427, "treatment", "PARP inhibitor ovarian"),
    (428, "treatment", "bispecific antibody myeloma"),
    (429, "treatment", "radioligand therapy prostate"),
]


# ── 5 RARE-EXPLICIT queries (IDs 500-504) — added Phase A11 of fix_bi-encoder.md ──
# Targeted probes for the new taxonomy buckets (medulloblastoma, wilms, gist,
# neuroendocrine, biliary) + mesothelioma/neuroblastoma. These queries make the
# rare-cancer-floor + sarcoma-split interventions VISIBLE in aggregate metrics —
# without them, improvements in rare buckets would be drowned out by the 60 other
# queries. Category is the single string "rare_explicit" for slicing in Phase C3.
# Secondary semantic tags (informational only, not used for slicing):
#   500 → [rare, treatment]   501 → [pediatric, rare]   502 → [pediatric, rare]
#   503 → [rare, treatment]   504 → [rare, treatment]
RARE_EXPLICIT_QUERIES: list[tuple[int, str, str]] = [
    (500, "rare_explicit", "pleural mesothelioma immunotherapy trial after pemetrexed failure"),
    (501, "rare_explicit", "high-risk neuroblastoma trial for my 4-year-old"),
    (502, "rare_explicit", "medulloblastoma group 3 trial after surgery and radiation"),
    (503, "rare_explicit", "GIST trial after imatinib resistance"),
    (504, "rare_explicit", "metastatic midgut neuroendocrine tumor trial somatostatin refractory"),
]


# ── 20 EXPANSION queries (IDs 600-619) — added Phase C4 to address small-n issue ──
# Original eval had n=5 per category for complex and vague, with bootstrap CI half-widths
# of ±0.13 to ±0.15 — wide enough that the decision-gate thresholds fell INSIDE the CIs.
# Adding 10 more queries to each tightens CIs to roughly ±0.07, making the gate decidable.
# Used for BOTH v1 and v2 labeling so the comparison stays apples-to-apples on the
# expanded set.
EXPANSION_QUERIES: list[tuple[int, str, str]] = [
    # Complex (10) — multi-fact patient profile, biggest test of agent extraction
    (600, "complex", "67M BRAF V600E metastatic melanoma post-dabrafenib trametinib progression"),
    (601, "complex", "45F triple negative breast cancer BRCA1 mutation neoadjuvant phase 2 trial"),
    (602, "complex", "71M castration-resistant prostate cancer Gleason 9 enzalutamide failure"),
    (603, "complex", "52F KRAS G12C NSCLC adagrasib resistance phase 2 trial"),
    (604, "complex", "34F HER2+ metastatic breast cancer brain metastases tucatinib failure"),
    (605, "complex", "60M MSI-high colorectal cancer pembrolizumab progression second-line"),
    (606, "complex", "28F AML FLT3-ITD positive post-allo transplant relapse"),
    (607, "complex", "55F ovarian cancer BRCA wild type platinum-resistant PARP failure"),
    (608, "complex", "48M IDH1 mutant glioblastoma post-radiation Stupp protocol recurrence"),
    (609, "complex", "8M relapsed B-ALL CD19+ post-CAR-T phase 2"),
    # Vague (10) — lay/caregiver phrasing, missing condition specifics
    (610, "vague", "my dad has cancer what trials should we look at"),
    (611, "vague", "is there anything experimental for my mother who is dying"),
    (612, "vague", "any new treatments for advanced cancer"),
    (613, "vague", "cancer trial please help"),
    (614, "vague", "i have stage 4 cancer what are my options"),
    (615, "vague", "need help finding a trial for my husband"),
    (616, "vague", "my grandma got diagnosed with cancer what's available"),
    (617, "vague", "anything experimental for terminal cancer"),
    (618, "vague", "trials for advanced or recurrent cancer"),
    (619, "vague", "clinical trial for someone dying of cancer"),
]


OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "full_labeled_dataset.jsonl"

# Path constants now env-var-overridable so we can swap v1 ↔ v2 without code edits.
# Defaults are v2 (the current production paths after Phase C1).
EMBEDDER_MODEL = os.environ.get("TRIALMINE_EMBEDDER", "models/embeddings/fine-tuned-v2")
FAISS_INDEX = os.environ.get("TRIALMINE_FAISS_INDEX", "data/faiss_finetuned_v2.index")
FAISS_MAPPING = os.environ.get("TRIALMINE_FAISS_MAPPING", "data/faiss_finetuned_v2.json")
CE_MODEL = os.environ.get("TRIALMINE_CROSS_ENCODER", "models/cross-encoder/fine-tuned")
RANKER_MODEL = os.environ.get("TRIALMINE_RANKER", "models/ranker/v2/model.lgb")


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


def label_pair(client, query: str, trial: dict, eligibility: str) -> tuple[int, str]:
    """Call Claude Haiku to rate a single (query, trial) pair on 0-3.

    Args:
        client: Anthropic API client.
        query: Patient search query.
        trial: Dict with trial metadata (title, conditions, phase, status).
        eligibility: Eligibility criteria text (truncated to 500 chars).

    Returns:
        Tuple of (relevance_score, reason_string). On parse / API failure
        returns (-1, "<error type>: <detail>") so callers can filter.
    """
    prompt = LABELING_PROMPT.format(
        query=query,
        trial_title=trial.get("title", "N/A"),
        conditions=trial.get("conditions", "N/A"),
        phase=trial.get("phase", "N/A"),
        status=trial.get("status", "N/A"),
        eligibility=eligibility[:500],
    )
    text = ""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return int(result["score"]), result.get("reason", "")
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Parse error for %s: %s — raw: %s", trial.get("nct_id"), exc, text[:120])
        return -1, f"parse_error: {str(exc)[:100]}"
    except Exception as exc:
        logger.error("API error for %s: %s", trial.get("nct_id"), exc)
        return -1, f"api_error: {str(exc)[:100]}"


def load_existing_labels(output_file: Path) -> set[tuple[int, str]]:
    """Load already-labeled (query_id, nct_id) pairs for resume support.

    Args:
        output_file: Path to the JSONL output file.

    Returns:
        Set of (query_id, nct_id) tuples already in the file.
    """
    existing: set[tuple[int, str]] = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing.add((rec["query_id"], rec["nct_id"]))
    return existing


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build full 60-query labeled eval dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip already-labeled pairs")
    parser.add_argument("--new-only", action="store_true", help="Only run the 30 new queries")
    parser.add_argument("--existing-only", action="store_true", help="Only run the 30 existing queries")
    parser.add_argument("--rare-explicit-only", action="store_true",
                        help="Only run the 5 rare_explicit queries (IDs 500-504)")
    parser.add_argument("--expansion-only", action="store_true",
                        help="Only run the 20 expansion queries (IDs 600-619), 10 complex + 10 vague")
    parser.add_argument("--list-queries", action="store_true",
                        help="Print the (id, category, query) list and exit (no Haiku, no models)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K from full pipeline to label")
    parser.add_argument("--db", default="data/trials.db", help="SQLite database path")
    parser.add_argument("--rate-limit-s", type=float, default=0.1, help="Sleep between API calls")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path override (default: data/evaluation/full_labeled_dataset.jsonl)")
    return parser.parse_args()


def build_query_list(args: argparse.Namespace) -> list[tuple[int, str, str]]:
    """Construct the (query_id, category, query) list per the CLI flags.

    Existing queries get category 'existing' since the user didn't tag them.
    Default (no flag) includes all 65 queries (30 existing + 30 new + 5 rare_explicit).
    """
    queries: list[tuple[int, str, str]] = []
    if args.expansion_only:
        queries = list(EXPANSION_QUERIES)
    elif args.rare_explicit_only:
        queries = list(RARE_EXPLICIT_QUERIES)
    else:
        if not args.new_only:
            queries.extend([(qid, "existing", q) for qid, q in EXISTING_QUERIES])
        if not args.existing_only:
            queries.extend(NEW_QUERIES)
            queries.extend(RARE_EXPLICIT_QUERIES)
    if args.limit > 0:
        queries = queries[: args.limit]
    return queries


def main() -> None:
    """Run the full pipeline + Haiku labeling for the 65-query dataset."""
    args = parse_args()
    queries = build_query_list(args)

    if args.list_queries:
        for qid, cat, q in queries:
            print(f"{qid}\t{cat}\t{q}")
        return

    # Heavy imports deferred — only loaded when we actually run the pipeline.
    import anthropic

    from TrialMine.models.cross_encoder import CrossEncoderReranker
    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.models.ranker import RankingBlender
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.hybrid import HybridRetriever
    from TrialMine.retrieval.semantic import FAISSIndex

    logger.info("Building eval set for %d queries (top-%d each)", len(queries), args.top_k)

    # ── Load full pipeline components ─────────────────────────────────────────
    logger.info("Loading retrieval components...")
    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)

    if not Path(CE_MODEL).exists():
        logger.error("Cross-encoder not found: %s", CE_MODEL)
        sys.exit(1)
    reranker = CrossEncoderReranker(model_name=CE_MODEL)

    blender = None
    if Path(RANKER_MODEL).exists():
        blender = RankingBlender(model_path=RANKER_MODEL)
    else:
        logger.warning("LightGBM model not found at %s — using CE-blended fallback", RANKER_MODEL)

    # ── Load eligibility lookup ───────────────────────────────────────────────
    logger.info("Loading eligibility criteria from SQLite...")
    conn = sqlite3.connect(args.db)
    elig_lookup = {nct_id: (elig or "") for nct_id, elig in conn.execute(
        "SELECT nct_id, eligibility_criteria FROM trials"
    ).fetchall()}
    conn.close()
    logger.info("Loaded eligibility for %d trials", len(elig_lookup))

    # ── Resume support ────────────────────────────────────────────────────────
    output_file = args.output if args.output is not None else OUTPUT_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_labels(output_file) if args.resume else set()
    if args.resume:
        logger.info("Resuming: %d (query_id, nct_id) pairs already labeled", len(existing))

    # ── Run full pipeline + label top-K per query ─────────────────────────────
    client = anthropic.Anthropic()
    mode = "a" if (args.resume or output_file.exists()) else "w"
    labeled_count = 0
    skipped = 0
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}

    with open(output_file, mode) as f:
        for i, (qid, category, query) in enumerate(queries):
            t0 = time.perf_counter()
            try:
                results, timings = hybrid.full_pipeline(
                    query=query,
                    reranker=reranker,
                    blender=blender,
                    top_k=args.top_k,
                    rerank_top_k=50,
                )
            except Exception as exc:
                logger.error("Pipeline failed for query %d (%s): %s", qid, query, exc)
                continue
            pipeline_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "Q%d [%s] %d results in %.0f ms — %s",
                qid, category, len(results), pipeline_ms, query[:60],
            )

            for rank, trial in enumerate(results):
                nct_id = trial["nct_id"]
                if (qid, nct_id) in existing:
                    skipped += 1
                    continue

                eligibility = elig_lookup.get(nct_id, "")
                score, reason = label_pair(client, query, trial, eligibility)

                record = {
                    "query_id": qid,
                    "query": query,
                    "category": category,
                    "rank": rank + 1,
                    "nct_id": nct_id,
                    "trial_title": trial.get("title", ""),
                    "trial_conditions": trial.get("conditions", ""),
                    "trial_phase": trial.get("phase"),
                    "trial_status": trial.get("status"),
                    "pipeline_score": trial.get("score"),
                    "cross_encoder_score": trial.get("cross_encoder_score"),
                    "relevance": score,
                    "reason": reason,
                    "labeler": "claude-haiku-4-5",
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                score_dist[score] = score_dist.get(score, 0) + 1
                labeled_count += 1

                if labeled_count % 25 == 0:
                    logger.info("  ... labeled %d pairs (query %d/%d)", labeled_count, i + 1, len(queries))

                time.sleep(args.rate_limit_s)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FULL EVAL DATASET BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Queries processed : {len(queries)}")
    print(f"  New labels        : {labeled_count}")
    print(f"  Skipped (resume)  : {skipped}")
    print(f"  Output file       : {output_file}")
    print(f"\n  Score distribution:")
    for score in [0, 1, 2, 3]:
        count = score_dist.get(score, 0)
        pct = count / max(labeled_count, 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {score}: {count:>4} ({pct:5.1f}%) {bar}")
    errors = score_dist.get(-1, 0)
    if errors:
        print(f"   -1: {errors:>4} (parse/API errors)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
