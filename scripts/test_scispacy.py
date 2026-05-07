"""Sanity-check SciSpacy entity extraction on sample text + 3 real trials.

Run: python scripts/test_scispacy.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.tokens import Doc

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "trials.db"
MODEL_NAME = "en_core_sci_lg"

SAMPLE_TEXT = (
    "Inclusion Criteria: Age 18 and older. Histologically confirmed non-small cell "
    "lung cancer. ECOG performance status 0-1. Exclusion Criteria: Prior treatment "
    "with anti-PD-1 antibody. Active autoimmune disease."
)


def print_entities(doc: Doc, source: str) -> None:
    print(f"\n--- {source} ---")
    print(f"text length: {len(doc.text)} chars  |  entities found: {len(doc.ents)}")
    if not doc.ents:
        print("  (no entities)")
        return
    for ent in doc.ents:
        print(f"  [{ent.start_char:>5}:{ent.end_char:<5}] {ent.label_:<15} {ent.text!r}")


def load_real_trials(db_path: Path, limit: int = 3) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT nct_id, title, eligibility_criteria
            FROM trials
            WHERE eligibility_criteria IS NOT NULL
              AND length(eligibility_criteria) > 200
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def main() -> None:
    print(f"Loading SciSpacy model: {MODEL_NAME}")
    nlp: Language = spacy.load(MODEL_NAME)
    print(f"Model: {nlp.meta['name']} v{nlp.meta['version']}  |  pipeline: {nlp.pipe_names}")

    # 1) Sample text
    print("\n" + "=" * 80)
    print("PART 1: Sample inclusion/exclusion text")
    print("=" * 80)
    print(f"INPUT: {SAMPLE_TEXT}")
    print_entities(nlp(SAMPLE_TEXT), source="sample")

    # 2) Real trials
    print("\n" + "=" * 80)
    print("PART 2: 3 real trials from data/trials.db")
    print("=" * 80)
    trials = load_real_trials(DB_PATH, limit=3)
    for trial in trials:
        nct = trial["nct_id"]
        title = trial["title"]
        crit = trial["eligibility_criteria"]
        header = f"{nct} | {title[:80]}"
        print(f"\nELIGIBILITY ({len(crit)} chars):")
        print(crit[:500] + ("..." if len(crit) > 500 else ""))
        print_entities(nlp(crit), source=header)


if __name__ == "__main__":
    main()
