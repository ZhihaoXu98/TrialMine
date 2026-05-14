"""Head-to-head comparison: TrialMine vs ClinicalTrials.gov search.

For each of 20 diverse test queries, hits both:
    - ClinicalTrials.gov API v2 keyword search (their top 20)
    - TrialMine's full pipeline (BM25 + Semantic + RRF + CE + LightGBM, top 20)

Computes overlap, ours-only, theirs-only. Reuses existing Haiku labels from
data/evaluation/full_labeled_dataset.jsonl when present (so we don't re-label
trials we already judged); falls back to a fresh Haiku call for any
ours-only trial that wasn't in the existing top-20 (rare in practice since
the queries are a subset of the 60-query labeled set).

The "ours-only" filter is the key value-add signal: if a trial we surfaced
that CT.gov didn't is judged 2 or 3 by the LLM judge, that's a genuine win.

Outputs:
    - Console table per query + aggregate
    - docs/ctgov_comparison.md with the strongest ours-only wins per query

Usage:
    python scripts/compare_with_ctgov.py             # 20-query head-to-head
    python scripts/compare_with_ctgov.py --limit 5   # smoke test

Requirements:
    - Elasticsearch up (`docker start trialmine-es`)
    - FAISS index + CE model + LightGBM ranker
    - ANTHROPIC_API_KEY (only used if any ours-only trial is unlabeled)
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 20 queries spanning all categories ───────────────────────────────────────
# Drawn from the labeled 60-query set so existing Haiku labels apply.
COMPARISON_QUERIES: list[tuple[str, str]] = [
    # category, query
    ("common", "breast cancer HR+ CDK4/6 inhibitor"),
    ("common", "NSCLC brain mets"),
    ("common", "ovarian BRCA mutation"),
    ("rare", "angiosarcoma scalp"),
    ("rare", "merkel cell carcinoma"),
    ("rare", "GIST imatinib resistant"),
    ("pediatric", "medulloblastoma 6 year old"),
    ("pediatric", "wilms tumor relapsed"),
    ("complex", "58M EGFR exon 19 NSCLC failed osimertinib phase 2-3"),
    ("complex", "45F TNBC completed AC-T 3mo ago"),
    ("complex", "62F HER2+ metastatic breast post-trastuzumab progression"),
    ("vague", "I have cancer what trials"),
    ("vague", "mom has bone cancer"),
    ("vague", "just diagnosed need help"),
    ("geographic", "pancreatic cancer trials Texas"),
    ("geographic", "trials at MD Anderson"),
    ("treatment", "CAR-T lymphoma"),
    ("treatment", "PARP inhibitor ovarian"),
    ("existing", "triple negative breast cancer neoadjuvant"),
    ("existing", "targeted therapy for EGFR mutated lung cancer"),
]

LABELS_FILE = Path("data/evaluation/full_labeled_dataset.jsonl")
OUTPUT_MD = Path("docs/ctgov_comparison.md")

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"

EMBEDDER_MODEL = "models/embeddings/fine-tuned"
FAISS_INDEX = "data/faiss_finetuned.index"
FAISS_MAPPING = "data/faiss_finetuned.json"
CE_MODEL = "models/cross-encoder/fine-tuned-v2"
RANKER_MODEL = "models/ranker/v3-regularized/model.lgb"

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


def ctgov_search(query: str, page_size: int = 20) -> list[dict]:
    """Hit CT.gov v2 keyword search.

    Returns dicts with full metadata needed for downstream LLM judging:
    nct_id, title, conditions, phase, status, eligibility.
    """
    params = {
        "query.term": query,
        "pageSize": page_size,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,EligibilityCriteria",
    }
    try:
        r = requests.get(CTGOV_API, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("CT.gov API error for '%s': %s", query, exc)
        return []

    results = []
    for study in data.get("studies", []):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        conditions = proto.get("conditionsModule", {}).get("conditions", []) or []
        phases = proto.get("designModule", {}).get("phases", []) or []
        elig = proto.get("eligibilityModule", {}).get("eligibilityCriteria", "") or ""
        results.append({
            "nct_id": ident.get("nctId"),
            "title": ident.get("briefTitle", ""),
            "status": status.get("overallStatus", ""),
            "conditions": "; ".join(conditions),
            "phase": "; ".join(phases) if phases else None,
            "eligibility": elig,
        })
    # Be polite to CT.gov (per CLAUDE.md, 0.5s rate limit)
    time.sleep(0.5)
    return results


def load_existing_labels(path: Path) -> dict[tuple[str, str], dict]:
    """Index existing Haiku labels by (query, nct_id)."""
    labels: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return labels
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            labels[(rec["query"], rec["nct_id"])] = rec
    return labels


def label_with_haiku(client, query: str, trial: dict, eligibility: str) -> tuple[int, str]:
    """Label a single (query, trial) pair on 0-3 with Claude Haiku.

    Used only for trials that don't already have a label in
    full_labeled_dataset.jsonl (i.e., not in our top-20 for this query).
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
    except (json.JSONDecodeError, KeyError, IndexError):
        return -1, f"parse_error: {text[:80]}"
    except Exception as exc:
        return -1, f"api_error: {str(exc)[:80]}"


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="TrialMine vs CT.gov head-to-head")
    parser.add_argument("--limit", type=int, default=0, help="Limit queries (0=all)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K from each side")
    parser.add_argument("--db", default="data/trials.db", help="SQLite path")
    parser.add_argument("--judge-theirs", action="store_true",
                        help="Also Haiku-judge CT.gov-only trials (lets us claim "
                             "'X is better than Y', not just 'X adds value')")
    return parser.parse_args()


def main() -> None:
    """Run the head-to-head comparison and write docs/ctgov_comparison.md."""
    import sqlite3

    import anthropic

    from TrialMine.models.cross_encoder import CrossEncoderReranker
    from TrialMine.models.embeddings import TrialEmbedder
    from TrialMine.models.ranker import RankingBlender
    from TrialMine.retrieval.bm25 import ElasticsearchIndex
    from TrialMine.retrieval.hybrid import HybridRetriever
    from TrialMine.retrieval.semantic import FAISSIndex

    args = parse_args()
    queries = COMPARISON_QUERIES
    if args.limit > 0:
        queries = queries[: args.limit]

    logger.info("Loading TrialMine pipeline...")
    es_index = ElasticsearchIndex()
    embedder = TrialEmbedder(model_name=EMBEDDER_MODEL)
    faiss_index = FAISSIndex()
    faiss_index.load(FAISS_INDEX, FAISS_MAPPING)
    hybrid = HybridRetriever(bm25=es_index, semantic=faiss_index, embedder=embedder)
    reranker = CrossEncoderReranker(model_name=CE_MODEL)
    blender = RankingBlender(model_path=RANKER_MODEL) if Path(RANKER_MODEL).exists() else None

    existing_labels = load_existing_labels(LABELS_FILE)
    logger.info("Loaded %d existing labels", len(existing_labels))

    # Eligibility lookup for the rare unlabeled-trial path
    conn = sqlite3.connect(args.db)
    elig_lookup = {nct_id: (elig or "") for nct_id, elig in conn.execute(
        "SELECT nct_id, eligibility_criteria FROM trials"
    ).fetchall()}
    conn.close()

    client = anthropic.Anthropic()

    per_query_records: list[dict] = []
    fresh_labels = 0

    for i, (category, query) in enumerate(queries):
        logger.info("[%d/%d] %s — %s", i + 1, len(queries), category, query)

        # Our side
        ours, _ = hybrid.full_pipeline(
            query=query, reranker=reranker, blender=blender,
            top_k=args.top_k, rerank_top_k=50,
        )
        our_ids = [r["nct_id"] for r in ours]
        ours_meta = {r["nct_id"]: r for r in ours}

        # Their side
        theirs = ctgov_search(query, page_size=args.top_k)
        their_ids = [t["nct_id"] for t in theirs]

        overlap = set(our_ids) & set(their_ids)
        ours_only = [nct_id for nct_id in our_ids if nct_id not in overlap]
        theirs_only = [nct_id for nct_id in their_ids if nct_id not in overlap]

        # Judge ours-only relevance — reuse existing labels first, label the rest with Haiku
        ours_only_judged: list[dict] = []
        for nct_id in ours_only:
            existing = existing_labels.get((query, nct_id))
            if existing and existing.get("relevance", -1) >= 0:
                ours_only_judged.append({
                    "nct_id": nct_id,
                    "title": existing.get("trial_title", ""),
                    "relevance": existing["relevance"],
                    "reason": existing.get("reason", ""),
                    "labeler": existing.get("labeler", "claude-haiku-4-5"),
                })
            else:
                # Cold path — fresh Haiku call
                trial = ours_meta.get(nct_id, {})
                eligibility = elig_lookup.get(nct_id, "")
                score, reason = label_with_haiku(client, query, trial, eligibility)
                fresh_labels += 1
                ours_only_judged.append({
                    "nct_id": nct_id,
                    "title": trial.get("title", ""),
                    "relevance": score,
                    "reason": reason,
                    "labeler": "claude-haiku-4-5 (fresh)",
                })
                time.sleep(0.1)

        relevant_count = sum(1 for j in ours_only_judged if j["relevance"] >= 2)

        # Optional: judge theirs-only with Haiku using CT.gov-supplied metadata
        theirs_only_judged: list[dict] = []
        their_meta = {t["nct_id"]: t for t in theirs}
        if args.judge_theirs:
            for nct_id in theirs_only:
                trial = their_meta.get(nct_id, {})
                # CT.gov sends eligibility text in-band; no SQLite fallback needed
                eligibility = trial.get("eligibility", "")
                score, reason = label_with_haiku(client, query, trial, eligibility)
                theirs_only_judged.append({
                    "nct_id": nct_id,
                    "title": trial.get("title", ""),
                    "relevance": score,
                    "reason": reason,
                    "labeler": "claude-haiku-4-5 (theirs)",
                })
                fresh_labels += 1
                time.sleep(0.1)
        theirs_relevant = sum(1 for j in theirs_only_judged if j["relevance"] >= 2)

        per_query_records.append({
            "category": category,
            "query": query,
            "our_ids": our_ids,
            "their_ids": their_ids,
            "overlap": sorted(overlap),
            "ours_only_judged": ours_only_judged,
            "theirs_only": theirs_only,
            "theirs_only_judged": theirs_only_judged,
            "n_overlap": len(overlap),
            "n_ours_only": len(ours_only),
            "n_theirs_only": len(theirs_only),
            "n_ours_only_relevant": relevant_count,
            "n_theirs_only_relevant": theirs_relevant,
        })

    # ── Console: per-query + aggregate ───────────────────────────────────────
    print()
    print("=" * 110)
    print(f"  TRIALMINE vs CT.GOV HEAD-TO-HEAD ({len(queries)} queries, top-{args.top_k} each)")
    print("=" * 110)
    headers = f"  {'Cat':<11}{'Query':<46}{'∩':>4}{'us only':>9}{'us-rel':>8}{'them only':>11}"
    if args.judge_theirs:
        headers += f"{'them-rel':>10}"
    print(headers)
    print(f"  {'-'*108}")

    agg_overlap = agg_ours = agg_theirs = agg_ours_rel = agg_theirs_rel = 0
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"overlap": 0, "ours": 0, "theirs": 0, "ours_rel": 0, "theirs_rel": 0, "n": 0}
    )

    for rec in per_query_records:
        line = (f"  {rec['category']:<11}{rec['query'][:43]:<46}"
                f"{rec['n_overlap']:>4}{rec['n_ours_only']:>9}"
                f"{rec['n_ours_only_relevant']:>8}{rec['n_theirs_only']:>11}")
        if args.judge_theirs:
            line += f"{rec.get('n_theirs_only_relevant', 0):>10}"
        print(line)
        agg_overlap += rec['n_overlap']
        agg_ours += rec['n_ours_only']
        agg_theirs += rec['n_theirs_only']
        agg_ours_rel += rec['n_ours_only_relevant']
        agg_theirs_rel += rec.get('n_theirs_only_relevant', 0)
        c = by_cat[rec['category']]
        c['overlap'] += rec['n_overlap']
        c['ours'] += rec['n_ours_only']
        c['theirs'] += rec['n_theirs_only']
        c['ours_rel'] += rec['n_ours_only_relevant']
        c['theirs_rel'] += rec.get('n_theirs_only_relevant', 0)
        c['n'] += 1

    print(f"  {'-'*108}")
    n_q = len(queries)
    total_line = (f"  {'TOTAL':<11}{'':<46}{agg_overlap:>4}{agg_ours:>9}"
                  f"{agg_ours_rel:>8}{agg_theirs:>11}")
    avg_line = (f"  {'AVG/query':<11}{'':<46}{agg_overlap/n_q:>4.1f}{agg_ours/n_q:>9.1f}"
                f"{agg_ours_rel/n_q:>8.1f}{agg_theirs/n_q:>11.1f}")
    if args.judge_theirs:
        total_line += f"{agg_theirs_rel:>10}"
        avg_line += f"{agg_theirs_rel/n_q:>10.1f}"
    print(total_line)
    print(avg_line)
    if agg_ours:
        precision = agg_ours_rel / agg_ours
        print(f"\n  Ours-only precision : {agg_ours_rel} / {agg_ours} = {precision:.0%}")
    if args.judge_theirs and agg_theirs:
        their_precision = agg_theirs_rel / agg_theirs
        print(f"  Theirs-only precision: {agg_theirs_rel} / {agg_theirs} = {their_precision:.0%}")
        if agg_ours and agg_theirs:
            delta = (agg_ours_rel / agg_ours) - (agg_theirs_rel / agg_theirs)
            print(f"  Delta (ours - theirs) : {delta:+.1%} precision pts on the disjoint set")
    print()
    print(f"  By category:")
    by_cat_header = (f"    {'Cat':<12}{'n':>3}{'overlap':>9}{'ours-only':>11}"
                     f"{'ours-rel':>10}{'theirs-only':>13}")
    if args.judge_theirs:
        by_cat_header += f"{'them-rel':>11}"
    print(by_cat_header)
    for cat, c in sorted(by_cat.items()):
        line = (f"    {cat:<12}{c['n']:>3}{c['overlap']/c['n']:>9.1f}"
                f"{c['ours']/c['n']:>11.1f}{c['ours_rel']/c['n']:>10.1f}"
                f"{c['theirs']/c['n']:>13.1f}")
        if args.judge_theirs:
            line += f"{c['theirs_rel']/c['n']:>11.1f}"
        print(line)
    if fresh_labels:
        print(f"\n  Fresh Haiku labels needed: {fresh_labels}")
    print("=" * 95)

    # ── Write docs/ctgov_comparison.md ────────────────────────────────────────
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md_lines: list[str] = [
        "# TrialMine vs ClinicalTrials.gov — Head-to-Head\n\n",
        "Comparison of TrialMine's full pipeline (BM25 + Semantic + RRF + Cross-Encoder + ",
        "LightGBM) against ClinicalTrials.gov's native v2 keyword search API across 20 ",
        f"diverse oncology queries spanning seven categories. Top-{args.top_k} from each side.\n\n",
        "## Headline\n\n",
        f"- **Average overlap per query**: {agg_overlap/n_q:.1f} / {args.top_k}\n",
        f"- **Average ours-only per query**: {agg_ours/n_q:.1f}\n",
        f"- **Of those, judged ≥2 by Haiku**: {agg_ours_rel/n_q:.1f} ",
        f"({(agg_ours_rel/agg_ours if agg_ours else 0):.0%} precision on the disjoint set)\n",
        f"- **Average theirs-only per query**: {agg_theirs/n_q:.1f}\n",
    ]
    if args.judge_theirs:
        their_p = agg_theirs_rel / agg_theirs if agg_theirs else 0
        our_p = agg_ours_rel / agg_ours if agg_ours else 0
        delta = our_p - their_p
        md_lines.append(f"- **Of CT.gov-only, judged ≥2 by Haiku**: {agg_theirs_rel/n_q:.1f} "
                        f"({their_p:.0%} precision on the disjoint set)\n")
        md_lines.append(f"- **Precision delta (ours − theirs)**: **{delta:+.1%}** "
                        f"on the disjoint set\n")
    md_lines.append("\n## Per-query summary\n\n")
    if args.judge_theirs:
        md_lines.append("| Category | Query | ∩ | Ours-only | Ours-rel ≥2 | Theirs-only | Theirs-rel ≥2 |\n")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    else:
        md_lines.append("| Category | Query | ∩ | Ours-only | Ours-rel ≥2 | Theirs-only |\n")
        md_lines.append("|---|---|---:|---:|---:|---:|\n")
    for rec in per_query_records:
        q_short = rec['query'].replace("|", "\\|")[:60]
        line = (f"| {rec['category']} | {q_short} | {rec['n_overlap']} | "
                f"{rec['n_ours_only']} | {rec['n_ours_only_relevant']} | "
                f"{rec['n_theirs_only']} |")
        if args.judge_theirs:
            line += f" {rec.get('n_theirs_only_relevant', 0)} |"
        md_lines.append(line + "\n")

    md_lines.append("\n## By category (averages per query)\n\n")
    if args.judge_theirs:
        md_lines.append("| Category | n | overlap | ours-only | ours-rel | theirs-only | theirs-rel |\n")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    else:
        md_lines.append("| Category | n | overlap | ours-only | ours-rel | theirs-only |\n")
        md_lines.append("|---|---:|---:|---:|---:|---:|\n")
    for cat, c in sorted(by_cat.items()):
        line = (f"| {cat} | {c['n']} | {c['overlap']/c['n']:.1f} | "
                f"{c['ours']/c['n']:.1f} | {c['ours_rel']/c['n']:.1f} | "
                f"{c['theirs']/c['n']:.1f} |")
        if args.judge_theirs:
            line += f" {c['theirs_rel']/c['n']:.1f} |"
        md_lines.append(line + "\n")

    # Best ours-only wins per query (relevance == 3, then 2)
    md_lines.append("\n## Strongest ours-only wins per query\n\n")
    md_lines.append("Trials TrialMine surfaced that CT.gov's keyword search did NOT, ")
    md_lines.append("ranked by Haiku-judged relevance.\n\n")

    for rec in per_query_records:
        wins = sorted(
            [j for j in rec["ours_only_judged"] if j["relevance"] >= 2],
            key=lambda j: -j["relevance"],
        )[:3]
        if not wins:
            continue
        md_lines.append(f"### {rec['category']} — `{rec['query']}`\n\n")
        for w in wins:
            title_short = w['title'].replace("|", "\\|")[:140]
            md_lines.append(f"- **{w['nct_id']}** (rel={w['relevance']}) — {title_short}\n")
            md_lines.append(f"  > {w['reason']}\n")
        md_lines.append("\n")

    md_lines.append("## Caveats\n\n")
    md_lines.append("- **CT.gov returns the entire registry (500K+ trials), not an oncology subset.** ")
    md_lines.append("Some `theirs-only` results may be relevant trials we filtered out at ingest.\n")
    md_lines.append("- **CT.gov's ranking is keyword-based, not ML-ranked**. The comparison is ")
    md_lines.append("therefore TrialMine's ML pipeline vs a strong but unranked baseline, not vs ")
    md_lines.append("a competing ML system.\n")
    md_lines.append("- **Ours-only relevance is judged by Claude Haiku**, the same judge that built ")
    md_lines.append("our training labels. This shares blind spots with the system under test. ")
    md_lines.append("See `docs/evaluation-report.md` for the Sonnet-vs-Haiku kappa analysis.\n")

    with open(OUTPUT_MD, "w") as f:
        f.writelines(md_lines)
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
