# Fix Eligibility Parser — UMLS via SciSpacy EntityLinker (Runbook)

> Step-by-step plan to close the parser-quality gap that's been the
> documented top-bottleneck for the project since Phase 10: the
> eligibility filter (`HARD_FILTER_CRITERIA` in
> `src/TrialMine/features/eligibility_filter.py`) only fires on 13 % of
> queries because `required_prior_treatments` matching is literal
> case-insensitive substring (`_any_overlap` in `tools.py:516`).
> `osimertinib` never substring-matches `EGFR TKI naive` even though
> those refer to the same drug class.
>
> **What this runbook delivers.** SciSpacy `EntityLinker` (UMLS) to
> canonicalize drug surface forms to UMLS CUIs (so `Tagrisso` ↔
> `osimertinib`), plus a small hand-curated `DRUG_TO_CLASS_CUIS` table
> (~20–30 oncology drugs → class CUIs) to bridge drug-name ↔
> drug-class (so `osimertinib` ↔ `EGFR TKI`). This is the **Option B**
> path from the engineering review. We explicitly do NOT promise UMLS
> hierarchy traversal — SciSpacy's KB doesn't expose parent relations
> (the `Entity` NamedTuple has only `concept_id`, `canonical_name`,
> `aliases`, `types`, `definition`; no `parents`), so we substitute
> curation for hierarchy.
>
> **How to use this doc.** Each step has a copy-paste prompt to send
> to Claude Code. After each prompt, run the **Verify** check and
> confirm the **Acceptance criteria** before moving on. If a check
> fails, run the **Rollback** and re-do the step — never proceed on a
> failed check.
>
> Phases are sequenced so:
> - Phase A is local prep (~4–5 hr, no GPU, no API spend).
> - Phase B is the focused re-eval on complex+vague (~$0.50–1 in Haiku).
> - Phase C is the holistic decision gate.
> - Phase D is the ship work (only if SHIP).
> - Phase E is the hold/revert path (if not SHIP).

---

## Overview

| | |
|---|---|
| **Goal** | Let the eligibility hard-filter match drug names to drug classes (osimertinib → EGFR TKI; trastuzumab → HER2-targeted therapy) so `required_prior_treatments` can join `HARD_FILTER_CRITERIA`. Resolve Q413 (failed osimertinib) + Q416 (post-trastuzumab progression) — the canonical complex failures since Week 9. |
| **Approach** | (1) Wire SciSpacy `EntityLinker` (UMLS KB) into `src/TrialMine/features/concepts.py`. (2) Add a small hand-curated `DRUG_TO_CLASS_CUIS` table in a new `drug_classes.py` (~20–30 oncology drugs → class CUIs). (3) Add a parallel `_any_overlap_cui` matcher in `tools.py` (alongside the existing `_any_overlap`, NOT replacing it) so prior-treatment matching can opt into CUI-aware matching behind a `DegradationConfig` toggle. **Query-time linking** (not batch-time), so no need to re-parse the 140K-trial corpus. |
| **Total cost** | ~$0.50–1 in Haiku for the focused re-eval on the 30 complex+vague queries (Phase B). Zero cloud spend. ~1–2 GB local disk for the SciSpacy UMLS KB (auto-downloaded on first `add_pipe` call). |
| **Total time** | ~4–5 hr local prep + ~1 hr eval + ~1 hr decision/ship. |
| **Expected lift** | **+0.02 to +0.04 NDCG@5 on `complex` aggregate**, **contingent** on the hand-curated drug-class table covering the held-out queries' drug mentions. With CUI-alias matching alone (no drug-class table), aggregate impact is much smaller (~0 to +0.01) — only catches Tagrisso ↔ osimertinib-style alias misses. Honest framing: real value is **clinical correctness** on Q413/Q416, NOT a metric headline. |
| **Decision gate** | Phase C — holistic review using the Decision-36 / Decision-40 framing. Four signals: (a) drug-class table coverage of held-out queries, (b) parser precision (CUI over-linking risk), (c) held-out NDCG delta, (d) qualitative correctness on Q413/Q416. No rigid pre-registered thresholds. |

**Files touched across all phases:**

| File | Phase | Reason |
|---|---|---|
| `src/TrialMine/features/concepts.py` | A3 | Add `UmlsConceptLinker` (lazy-loaded), `link_to_cuis`, `match_via_cui` |
| `src/TrialMine/features/drug_classes.py` | A4 | NEW — hand-curated `DRUG_TO_CLASS_CUIS` table |
| `src/TrialMine/agents/tools.py` | A5 | Add parallel `_any_overlap_cui`; gate prior-treatment matchers on toggle |
| `src/TrialMine/config.py` | A5 | New `umls_drug_class_matching_enabled` toggle on `DegradationConfig` |
| `src/TrialMine/features/eligibility_filter.py` | D1 (ship only) | Add `required_prior_treatments` to `HARD_FILTER_CRITERIA` |
| `tests/unit/test_concepts.py` | A6 | UMLS linking + CUI-overlap tests (marked `@pytest.mark.slow`) |
| `tests/unit/test_eligibility.py` | A6 | Q413 + Q416 canonical-case tests (marked `@pytest.mark.slow`) |
| `tests/unit/test_drug_classes.py` | A4 | NEW — table-shape + Q413/Q416 spot-checks |
| `scripts/eval_parser_umls.py` | B1 | NEW — focused re-eval harness on 30 complex+vague queries |
| `data/evaluation/full_labeled_complex_vague_umls.jsonl` | B1 | Re-label artefact (~$0.50–1 in Haiku) |
| `data/evaluation/umls_eval_metrics.json` | B1 | Per-query NDCG/MRR + bootstrap 95 % CIs |
| `docs/evaluation-report.md` § 11.6 update | D5 (ship only) | Resolution note |
| `CLAUDE.md` (Decision 41 + What's working bump + close §11.6 in things_can_be_fixed.md) | D6 (ship only) | Documentation |

---

## Best practices for these prompts

1. **One concern per prompt.** Each step is scoped to a single logical change. Don't combine.
2. **Verification is non-optional.** Every step has an explicit probe. Don't skip it.
3. **Quote acceptance criteria back.** When you run a probe, paste the output and explicitly check it against the **Acceptance criteria** before moving on.
4. **Context resets are fine.** Each prompt references this doc; Claude can re-orient even if your session was cleared.
5. **Never edit two files at once.** Finish + verify a step before starting the next.
6. **No "done" without a diff or a probe output.** If a step has no observable output, you didn't verify it.

---

# Phase A — Local prep (~4–5 hr, no GPU, no API spend)

## Step A0 — Session bootstrap

If you're returning after a context reset, send this first:

```
We are working through docs/fix_parser_umls.md to upgrade the
TrialMine eligibility parser with UMLS via SciSpacy EntityLinker.
Read fix_parser_umls.md and report: (1) which phase steps appear
complete based on file state (whether `concepts.py` imports
`EntityLinker` and exposes `link_to_cuis` / `match_via_cui`,
whether `src/TrialMine/features/drug_classes.py` exists,
whether `_any_overlap_cui` exists in `tools.py`, whether the new
`umls_drug_class_matching_enabled` toggle is on
`DegradationConfig`, whether the new eval script
`scripts/eval_parser_umls.py` exists, whether the focused
re-label JSONL is in `data/evaluation/`), and (2) the next step
to run. Do not modify any files yet.
```

This gives Claude a chance to read the runbook and tell you where you are.

---

## Step A1 — Audit the current parser state (30 min)

**Goal:** before touching code, confirm the exact data flow from query →
parsed treatments → eligibility filter. This step is read-only.

### Prompt to send

```
Per Phase A1 of docs/fix_parser_umls.md, audit the current
parser state. Read these files and report — terse, no code
listings:

  1. src/TrialMine/features/eligibility.py
     - Confirm `required_prior_treatments: list[str]` is on the
       `EligibilityProfile` Pydantic model (~line 51)
     - Confirm where the list gets POPULATED during parsing
       (which regex / SciSpacy step adds to it)

  2. src/TrialMine/features/eligibility_filter.py
     - Confirm `HARD_FILTER_CRITERIA` is the closed-vocab tuple
       `("age", "sex", "excluded_prior_treatments")`
     - Confirm `is_hard_unmet(elig)` reads
       `elig["criteria"][name]["verdict"] == "Unmet"`

  3. src/TrialMine/features/concepts.py
     - Confirm `extract_concepts` + `normalise_concept` are
       module-level stubs that `raise NotImplementedError`
     - Confirm there are ZERO callers of those stubs in src/
       scripts/ tests/ (grep)

  4. src/TrialMine/agents/tools.py
     - Confirm `_any_overlap(needles: list[str], haystack: str | None)
       -> tuple[bool, list[str]]` at line 516
     - Confirm it does case-insensitive substring matching
     - Confirm it is CALLED by FOUR functions:
       _match_required_conditions, _match_excluded_conditions,
       _match_required_treatments, _match_excluded_treatments
     - Confirm _match_required_treatments + _match_excluded_treatments
       are at ~line 583 and ~614

  5. src/TrialMine/config.py
     - Confirm `DegradationConfig` is a `pydantic.BaseModel`
       (NOT BaseSettings — not env-driven)
     - Confirm existing pattern: hard toggles like
       `eligibility_hard_filter_enabled: bool = Field(default=True, ...)`
     - Confirm the process-wide singleton is
       `_DEFAULT_DEGRADATION` + `get_default_degradation()`

  6. scripts/parse_eligibility.py
     - Confirm it batch-parses to the `parsed_eligibility`
       SQLite table; we will NOT re-run it (query-time linking
       design)

Output: one short paragraph per file confirming or correcting
each bullet. No code blocks. If anything diverges from the
description, flag it before proceeding.
```

### Verify

You should walk away with confirmation that:
- The matcher to swap is `_any_overlap` for treatments only
- The toggle home is `DegradationConfig`
- The condition matchers (`_match_required_conditions`,
  `_match_excluded_conditions`) will be left untouched

### Acceptance criteria

- All 6 file-level claims confirmed (or any divergence flagged + the runbook
  updated)
- `extract_concepts` / `normalise_concept` confirmed unused → safe to delete
  or replace in A3

### Rollback

N/A (audit only, no edits).

---

## Step A2 — Verify SciSpacy + install the UMLS KB (30 min, ~1–2 GB disk)

**Goal:** confirm SciSpacy + spaCy + `en_core_sci_lg` are usable, and
download the UMLS KB on first `add_pipe` call. The KB is NOT distributed
via a separate pip install — it auto-downloads from S3 the first time
`add_pipe("scispacy_linker", ...)` is called.

### Prompt to send

```
Phase A2 of docs/fix_parser_umls.md — verify SciSpacy + download
the UMLS KB. Run, in the trialmine conda env:

  /opt/miniconda3/envs/trialmine/bin/python - <<'PY'
  import scispacy, spacy
  from scispacy.linking import EntityLinker     # noqa: F401 (registers
                                                #   the "scispacy_linker"
                                                #   spaCy component)
  from scispacy.candidate_generation import UmlsLinkerPaths
  print(f"scispacy: {scispacy.__version__}")    # expect 0.6.x
  print(f"spacy:    {spacy.__version__}")        # expect 3.7.x
  print(f"UMLS KB locally cached: {UmlsLinkerPaths.is_locally_cached()}")
  nlp = spacy.load("en_core_sci_lg")
  print(f"en_core_sci_lg pipe: {nlp.pipe_names}")
  PY

If the KB is NOT locally cached, the first nlp.add_pipe(
"scispacy_linker", config={"linker_name": "umls"}) call below
downloads 4 files (~1–2 GB total) from
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/
to ~/.scispacy/. Wall-clock: 5–15 min depending on bandwidth.

Trigger the download with a one-time warm-up:

  /opt/miniconda3/envs/trialmine/bin/python - <<'PY'
  import time, spacy
  from scispacy.linking import EntityLinker
  t0 = time.time()
  nlp = spacy.load("en_core_sci_lg")
  print("Loading scispacy_linker (first call downloads ~1-2 GB)...")
  nlp.add_pipe(
      "scispacy_linker",
      config={
          "resolve_abbreviations": True,
          "linker_name": "umls",
          "max_entities_per_mention": 5,
          "threshold": 0.85,           # tighter than scispacy default 0.7
      },
  )
  print(f"Linker pipe loaded in {time.time()-t0:.1f}s")

  # Smoke test against a known oncology drug
  doc = nlp("Patient was previously treated with osimertinib.")
  print(f"Entities found: {len(doc.ents)}")
  linker = nlp.get_pipe("scispacy_linker")
  for ent in doc.ents:
      print(f"  text={ent.text!r}  label={ent.label_}")
      for cui, score in (ent._.kb_ents or []):
          entity = linker.kb.cui_to_entity[cui]
          print(f"    -> cui={cui}  score={score:.3f}  "
                f"name={entity.canonical_name!r}  types={entity.types}")
  PY

Report: scispacy version, spacy version, "was KB cached before
this run?", the entities found, and at least one CUI for
osimertinib (expected CUI: C2700554 -- verify the canonical
name contains "osimertinib" or "Tagrisso").
```

### Verify

- `scispacy >= 0.6`, `spacy >= 3.7`
- `en_core_sci_lg` loads with pipe components ≈ `[tok2vec, tagger,
  attribute_ruler, lemmatizer, parser, ner]` (no linker yet — we add it)
- After `add_pipe("scispacy_linker", ...)`, `~/.scispacy/` exists and is
  ~1–2 GB
- The smoke test yields **at least one drug-type CUI** for `osimertinib`
  with `canonical_name` matching the drug name (case-insensitive)

### Acceptance criteria

- KB cached locally after A2
- Linker returns at least one CUI for `osimertinib` with non-empty
  `types` (semantic types like `T121`)
- Linker load on second run (KB already cached) completes in ≤ 30 s

### Rollback

```
rm -rf ~/.scispacy/   # if download corrupted or you want to redo
```

The actual files: `~/.scispacy/nmslib_index.bin`,
`tfidf_vectorizer.joblib`, `tfidf_vectors_sparse.npz`,
`concept_aliases.json`. Deleting forces re-download next run.

---

## Step A3 — Add `UmlsConceptLinker` + `link_to_cuis` + `match_via_cui` to `concepts.py` (90 min)

**Goal:** add a lazy-loaded UMLS linker that maps text → drug-relevant
CUIs, and a `match_via_cui` helper that checks both **direct CUI overlap**
(alias matching) and **drug-class overlap** via the hand-curated table from
A4. **Do NOT promise hierarchy traversal — SciSpacy's KB doesn't support
it.**

### Prompt to send

```
Phase A3 of docs/fix_parser_umls.md -- add UMLS linking to
src/TrialMine/features/concepts.py. The design must:

1. Add a frozen dataclass `ConceptLink`:
       cui: str
       canonical_name: str
       score: float
       semantic_types: tuple[str, ...]   # e.g., ("T121",)

2. Add a module-level lazy singleton for the spaCy pipeline:
       _LINKER_NLP: spacy.Language | None = None
   with a `_get_linker_nlp()` helper that, on first call:
   - Imports spacy + scispacy.linking (the import REGISTERS the
     "scispacy_linker" component with spaCy; do NOT remove the
     import as unused -- it has a side-effect).
   - Calls spacy.load("en_core_sci_lg")
   - Calls nlp.add_pipe("scispacy_linker", config={
         "resolve_abbreviations": True,
         "linker_name": "umls",
         "max_entities_per_mention": 5,
         "threshold": 0.85,
     })
   - Caches the nlp (the linker pipe is reachable via
     nlp.get_pipe("scispacy_linker") for kb.cui_to_entity
     lookups later)
   - Logs a single info message with wall-clock load time so we
     can see it in API startup logs

3. Add a frozenset of drug-relevant UMLS semantic types we
   accept:
       _DRUG_SEMANTIC_TYPES = frozenset({
           "T121",  # Pharmacologic Substance
           "T200",  # Clinical Drug
           "T109",  # Organic Chemical
           "T123",  # Biologically Active Substance
           "T129",  # Immunologic Factor (mAbs)
       })
   Anything not landing in one of these is dropped (so we don't
   match "kidney" -> some anatomy CUI as a drug).

4. Add @functools.lru_cache(maxsize=2000) decorated:
       def link_to_cuis(text: str) -> tuple[ConceptLink, ...]:
   that lazy-loads via _get_linker_nlp(), runs the pipeline on
   text, iterates doc.ents and for each entity's ent._.kb_ents
   list of (cui, score) pairs:
   - Looks up linker.kb.cui_to_entity[cui] to get the Entity
     NamedTuple
   - Filters: any of entity.types is in _DRUG_SEMANTIC_TYPES
   - Emits ConceptLink(cui=cui,
       canonical_name=entity.canonical_name,
       score=float(score),
       semantic_types=tuple(entity.types or ()))
   - Returns a tuple (immutable so it's hashable / cacheable)
   Empty / whitespace-only text returns () without loading the
   pipeline (so the first call cost is paid by real input).

5. Add match_via_cui(text_a, text_b) -> tuple[bool, list[str]]:
   - Imports DRUG_TO_CLASS_CUIS from
     TrialMine.features.drug_classes inside the function body
     (lazy import) so A3 doesn't break before A4 lands. On
     ImportError, fall back to empty dict and log a warning.
   - Computes cuis_a = {l.cui for l in link_to_cuis(text_a)},
     cuis_b similarly
   - Direct overlap: cuis_a & cuis_b -> return (True, sorted)
   - Drug-class overlap: for each cui in cuis_a, look up
     DRUG_TO_CLASS_CUIS.get(cui, frozenset()) and intersect
     with cuis_b; same direction reversed; A's-classes
     intersect B's-classes
   - Return (True, sorted class-CUIs) if non-empty; else
     (False, [])
   - Telemetry log on match: logger.info("UMLS match %s <-> %s
     via %s", text_a, text_b, matched_cuis)

6. REMOVE the unused extract_concepts + normalise_concept stubs
   from concepts.py -- they raise NotImplementedError, have
   zero callers, and the new linker subsumes their intent.

7. Update the module docstring at top of concepts.py to reflect
   the new design (the existing "TODO: future PR" framing is
   now stale).

Use type hints + structured logging via logging.getLogger(__name__)
+ no print(). Show me the full diff (including what's removed).

DO NOT modify eligibility.py, eligibility_filter.py, or
tools.py yet. This step only changes concepts.py.
```

### Verify

Run the smoke test:

```
/opt/miniconda3/envs/trialmine/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
from TrialMine.features.concepts import link_to_cuis

# Test 1: drug name -> CUI(s)
links = link_to_cuis("osimertinib")
print(f"osimertinib -> {len(links)} link(s)")
for l in links[:3]:
    print(f"  cui={l.cui} score={l.score:.3f} name={l.canonical_name!r} types={l.semantic_types}")

# Test 2: drug class -> CUI(s)
links = link_to_cuis("EGFR tyrosine kinase inhibitor")
print(f"EGFR TKI -> {len(links)} link(s)")
for l in links[:3]:
    print(f"  cui={l.cui} score={l.score:.3f} name={l.canonical_name!r} types={l.semantic_types}")

# Test 3: non-drug should return no drug-type links
links = link_to_cuis("kidney")
drug_types = {"T121","T200","T109","T123","T129"}
drug_links = [l for l in links if any(t in drug_types for t in l.semantic_types)]
print(f"kidney -> {len(links)} total, {len(drug_links)} drug-type (expect 0)")

# Test 4: match_via_cui works with no drug_classes module yet
from TrialMine.features.concepts import match_via_cui
ok, matched = match_via_cui("osimertinib", "Tagrisso")
print(f"osimertinib <-> Tagrisso (alias): {ok} via {matched}")  # expect True
ok, matched = match_via_cui("osimertinib", "EGFR TKI")
print(f"osimertinib <-> EGFR TKI (pre-A4): {ok}")  # expect False -- table missing

# Test 5: pre-discover the CUIs A4 will need. scispacy 0.6.2 ships a 2022
# UMLS KB snapshot (umls_2022_ab_cat0129.jsonl) whose CUI assignments
# differ from the public UMLS browser at uts.nlm.nih.gov. Example:
# osimertinib is C4058811 in scispacy, C2700554 in the browser. Both are
# real UMLS CUIs; only the scispacy one will match in match_via_cui. A4
# MUST use the CUIs this test prints, NOT the browser CUIs. Pipe the
# output to a file: `... | tee /tmp/a3_cuis.txt`
print("\n--- A4 CUI discovery (paste these into drug_classes.py) ---")
_DRUG_TYPES = {"T121", "T200", "T109", "T123", "T129"}
_A4_PROBES = [
    # Q413 / Q416 canonical drugs
    "osimertinib", "trastuzumab",
    # Other EGFR TKIs (extend A4 here)
    "erlotinib", "gefitinib", "afatinib", "dacomitinib",
    # Other HER2-targeted
    "pertuzumab", "ado-trastuzumab emtansine", "fam-trastuzumab deruxtecan",
    # Immune checkpoint inhibitors
    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab",
    # Drug-class phrases (the VALUES in the dict)
    "EGFR tyrosine kinase inhibitor", "tyrosine kinase inhibitor",
    "HER2-targeted therapy", "monoclonal antibody",
    "PD-1 inhibitor", "PD-L1 inhibitor", "immune checkpoint inhibitor",
]
for phrase in _A4_PROBES:
    links = link_to_cuis(phrase)
    drug_links = [l for l in links if any(t in _DRUG_TYPES for t in l.semantic_types)]
    if drug_links:
        top = drug_links[0]
        print(f"  {phrase!r:55s} -> {top.cui}  score={top.score:.3f}  name={top.canonical_name!r}")
    else:
        # No drug-type CUI -- A4 must either skip this entry or document the
        # gap explicitly. The class concept may not exist in the 2022 UMLS KB.
        print(f"  {phrase!r:55s} -> NO DRUG-TYPE CUI (A4 must document the gap)")
PY
```

### Acceptance criteria

- `link_to_cuis("osimertinib")` returns ≥ 1 `ConceptLink` whose
  `canonical_name` is `"osimertinib"` or `"Tagrisso"` (any alias) and whose
  `semantic_types` includes at least one of T121/T200/T109/T123
- `link_to_cuis("EGFR tyrosine kinase inhibitor")` returns ≥ 1 link with
  canonical name mentioning "EGFR" / "tyrosine kinase inhibitor"
- `link_to_cuis("kidney")` returns 0 drug-type links (filter dropped
  anatomy)
- `link_to_cuis` called twice with same arg: second call returns in ≤ 5 ms
  (LRU cache)
- `match_via_cui` returns False on drug-class lookup before A4 (table
  doesn't exist) — A4 will populate
- **Test 5** prints a paste-ready CUI table for A4. Capture it:
  `... -m pytest ... | tee /tmp/a3_cuis.txt` or copy from the smoke-test stdout.
  A4 will lift these CUIs verbatim into `DRUG_TO_CLASS_CUIS`.

> **Note recorded from the A2 run (2026-05-14).** scispacy 0.6.2 ships a
> 2022 UMLS KB snapshot. Verified CUIs: osimertinib = `C4058811` (NOT
> the public UMLS browser's `C2700554`). Other drugs/classes have similar
> divergences. Acceptance criteria above are intentionally written
> against `canonical_name` + `semantic_types`, never against a specific
> CUI string, because the scispacy CUI for any concept is only knowable
> by calling `link_to_cuis()` against this exact KB version.

### Rollback

```
git checkout src/TrialMine/features/concepts.py
```

The deleted stubs (`extract_concepts`, `normalise_concept`) are recoverable
from git history.

---

## Step A4 — Build the hand-curated `DRUG_TO_CLASS_CUIS` table (60 min)

**Goal:** create a new module
`src/TrialMine/features/drug_classes.py` containing a `dict[str,
frozenset[str]]` that maps a drug CUI to its drug-class CUIs. The table
is hand-curated, ~20–30 entries, and explicitly biased to oncology drugs
that appear in the held-out eval set (especially Q413, Q416, and other
complex queries).

**This is the single piece of curation work in the whole runbook.** Done
honestly, it takes ~45–60 min: for each drug, look up its CUI in the
NIH UMLS browser (https://uts.nlm.nih.gov/uts/umls/concept/) and
identify the class-level CUI(s) it belongs to.

### Prompt to send

```
Phase A4 of docs/fix_parser_umls.md -- create the hand-curated
DRUG_TO_CLASS_CUIS table.

**CRITICAL: use scispacy's CUIs, not the public UMLS browser's.**
scispacy 0.6.2 ships a frozen 2022 UMLS KB snapshot
(umls_2022_ab_cat0129.jsonl) whose CUI assignments differ from the
public UMLS browser at uts.nlm.nih.gov for many concepts. Example
verified in Phase A2: osimertinib is C4058811 in scispacy, C2700554
in the browser. Both are real UMLS CUIs but only the scispacy one
matches in match_via_cui. EVERY CUI key and value in the dict must
come from Phase A3 Test 5's output (the paste-ready CUI table that
was printed when A3 ran). If you do not have A3 Test 5's output,
stop and re-run A3 -- do NOT fall back to the public UMLS browser
or you will silently produce a dict that never matches anything.

**Findings from the A3 run (2026-05-14)** — the example dict below
already incorporates these; record them here so future maintainers
understand the design:

1. **Brand-name aliases get separate keys.** SciSpacy's 2022 KB
   assigns DIFFERENT CUIs to brand vs generic names. Verified
   pairs: osimertinib=C4058811 vs Tagrisso=C4058817; trastuzumab=
   C0728747 vs Herceptin=C0338204; pembrolizumab=C3658706 vs
   Keytruda=C3855203; nivolumab=C3657270 vs Opdivo=C3872108;
   erlotinib=C1135135 vs Tarceva=C1135136; gefitinib=C1122962 vs
   Iressa=C0919281. Direct-CUI overlap therefore does NOT alias-
   match brand ↔ generic. Trial eligibility text almost always
   uses generic names, but trials and patient queries occasionally
   use brand names, so we list both as keys with the same class-
   value set.

2. **"HER2-targeted therapy" doesn't resolve.** A3's follow-up
   probe found that the alternate phrasing "HER2 inhibitor"
   resolves to C4759996 ('Substance with human epidermal growth
   factor receptor 2 inhibitor mechanism of action'). Use that
   CUI for trastuzumab + pertuzumab + T-DM1. Do NOT fall back to
   the broader C0003250 ('Monoclonal Antibodies') — it would
   false-match every other mAb in the corpus (ipilimumab,
   pembrolizumab, etc.).

3. **"PD-1 inhibitor" and "PD-L1 inhibitor" don't resolve to a
   useful drug-class CUI.** "PD-1 inhibitor" resolves to the
   generic C1999216 ('Inhibitor'); "PD-L1 inhibitor" resolves to
   C0965245 ('CD274 protein, human' — the gene, not the drug
   class). Collapse all 5 immune checkpoint inhibitors
   (pembro/nivo/atezo/durva/ipi) to the parent class C4684977
   ('Immune Checkpoint Inhibitors'). Less granular but
   clinically sound — "any prior ICI" is the most common trial
   restriction in oncology.

4. **fam-trastuzumab deruxtecan (Enhertu) is not in the 2022 KB.**
   A3 Test 5 printed "NO DRUG-TYPE CUI" for it. Omit + document
   the gap; add when the KB updates or a custom UMLS REST lookup
   is wired.

File to create: src/TrialMine/features/drug_classes.py

Required structure:

  """Hand-curated UMLS drug -> drug-class CUI mappings for the
  oncology drugs that appear in our held-out eval set
  (Q413, Q416, and other complex queries).

  This is intentionally small (~20-30 entries) and hand-built --
  we are NOT attempting to mirror UMLS's full hierarchy. SciSpacy's
  UMLS KB does not expose parent relations (the Entity NamedTuple
  has only concept_id / canonical_name / aliases / types /
  definition; no parents). We substitute curation for hierarchy.

  Each entry is documented with a one-line clinical note plus
  the CUI source verified against the NIH UMLS browser at
  https://uts.nlm.nih.gov/uts/umls/concept/<CUI>.

  Maintenance: add an entry whenever a held-out failing query
  points at a new drug we don't yet cover. Stay below ~50
  entries -- beyond that, the right move is to switch to the
  UMLS REST API for live hierarchy lookup (project already has
  umls_api_key field on Settings, currently empty).
  """

  from __future__ import annotations

  # drug CUI -> frozenset of class-level CUIs the drug belongs to
  # All CUIs below were discovered via Phase A3 Test 5 + the A3 HER2 / PD-1
  # follow-up probes (run 2026-05-14, captured in /tmp/a3_smoke.log).
  # Verified against scispacy 0.6.2's 2022 KB. To re-verify after a KB
  # upgrade, re-run A3's smoke test; CUIs may drift.

  DRUG_TO_CLASS_CUIS: dict[str, frozenset[str]] = {
      # ===================================================================
      # EGFR TKIs  -- targets Q413 (failed osimertinib)
      # Class CUIs:
      #   C5574906 'Epidermal growth factor receptor inhibitor'
      #   C1268567 'Protein-tyrosine kinase inhibitor' (broader)
      # ===================================================================
      "C4058811": frozenset({"C5574906", "C1268567"}),  # osimertinib
      "C4058817": frozenset({"C5574906", "C1268567"}),  # Tagrisso (brand)
      "C1135135": frozenset({"C5574906", "C1268567"}),  # erlotinib
      "C1135136": frozenset({"C5574906", "C1268567"}),  # Tarceva (brand)
      "C1122962": frozenset({"C5574906", "C1268567"}),  # gefitinib
      "C0919281": frozenset({"C5574906", "C1268567"}),  # Iressa (brand)
      "C2987648": frozenset({"C5574906", "C1268567"}),  # afatinib
      "C2987430": frozenset({"C5574906", "C1268567"}),  # dacomitinib

      # ===================================================================
      # HER2-targeted -- targets Q416 (post-trastuzumab progression)
      # Class CUI:
      #   C4759996 'Substance with HER2 inhibitor mechanism of action'
      # (Discovered via "HER2 inhibitor"; "HER2-targeted therapy" does
      # not resolve in the 2022 KB. C0003250 'Monoclonal Antibodies' is
      # intentionally NOT used as a broader fallback -- too broad,
      # would false-match unrelated mAbs.)
      # ===================================================================
      "C0728747": frozenset({"C4759996"}),  # trastuzumab
      "C0338204": frozenset({"C4759996"}),  # Herceptin (brand)
      "C1328025": frozenset({"C4759996"}),  # pertuzumab
      "C2935436": frozenset({"C4759996"}),  # ado-trastuzumab emtansine (T-DM1)
      # GAP: fam-trastuzumab deruxtecan (Enhertu) -- "NO DRUG-TYPE CUI"
      # in 2022 KB. Add when the KB updates.

      # ===================================================================
      # Immune checkpoint inhibitors (PD-1 + PD-L1 + CTLA-4)
      # Class CUI:
      #   C4684977 'Immune Checkpoint Inhibitors'
      # (PD-1- and PD-L1-specific class phrases don't resolve to useful
      # CUIs in the 2022 KB -- see Findings #3 above. Collapsing to the
      # ICI parent class is the right level of granularity for "any
      # prior ICI" trial eligibility.)
      # ===================================================================
      "C3658706": frozenset({"C4684977"}),  # pembrolizumab
      "C3855203": frozenset({"C4684977"}),  # Keytruda (brand)
      "C3657270": frozenset({"C4684977"}),  # nivolumab
      "C3872108": frozenset({"C4684977"}),  # Opdivo (brand)
      "C4055433": frozenset({"C4684977"}),  # atezolizumab
      "C4055109": frozenset({"C4684977"}),  # durvalumab
      "C1367202": frozenset({"C4684977"}),  # ipilimumab

      # ===================================================================
      # EXTEND HERE -- the dict above has 19 entries; the unit test
      # asserts >= 20. Cheapest extension path:
      #   1. Pick 1-2 more drugs that appear in the complex/vague
      #      expansion queries (IDs 600-619 in
      #      data/evaluation/full_labeled_dataset_expansion_v2.jsonl).
      #   2. Add them to A3 Test 5's _A4_PROBES list.
      #   3. Re-run the A3 smoke test.
      #   4. Lift the verified CUIs into here.
      # Candidate families:
      #   - CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib)
      #   - ALK inhibitors  (crizotinib, alectinib, lorlatinib)
      #   - PARP inhibitors (olaparib, niraparib, talazoparib)
      #   - VEGFR TKIs      (sorafenib, sunitinib, lenvatinib)
      # ===================================================================
  }

Hard requirements for this step:

1. The 19 entries in the example dict above are already
   verified against scispacy 0.6.2's 2022 KB (Phase A3 run
   2026-05-14). Lift them verbatim. Do NOT modify those entries
   unless you've re-run A3 Test 5 against a different KB version
   -- the comments encode design decisions ("Findings from the
   A3 run" above) that the bare CUIs don't carry. ADD 1-2 more
   drug families to reach the >= 20 entries the unit test
   requires; follow the EXTEND HERE recipe in the dict's
   trailing comment block. Any newly added CUI must come from
   re-running A3 Test 5 with the new drug/class added to
   _A4_PROBES; do NOT fall back to the public UMLS browser as a
   CUI source -- it would produce entries that silently never
   match. (The browser is still useful for reading the CLINICAL
   relationship -- "is C5574906 actually EGFR TKI?" -- but the
   CUI strings going into the dict must match what scispacy
   returns.)

2. Cross-reference with the held-out failing queries. Run:

      grep -h '"category":"complex"' \
           data/evaluation/full_labeled_dataset.jsonl \
           data/evaluation/full_labeled_dataset_expansion_v2.jsonl \
        | head -30

   Identify the drugs mentioned. Make sure each of THOSE drugs
   has an entry. Q413 (osimertinib) and Q416 (trastuzumab) are
   mandatory.

3. Add a docstring block listing the held-out queries this
   table is curated against (so future maintainers see the
   scope assumption).

4. Add a unit test in tests/unit/test_drug_classes.py that:
   - Imports DRUG_TO_CLASS_CUIS
   - Asserts the table has >= 20 entries
   - Asserts every key matches r"^C\d{7}$" (CUI format)
   - Asserts every value is a frozenset (immutable) of strings
     matching the same CUI format
   - Spot-checks Q413: looks up the scispacy CUI for "osimertinib"
     via link_to_cuis(), asserts that CUI is a KEY in
     DRUG_TO_CLASS_CUIS, and asserts the value frozenset contains
     the scispacy CUI for "EGFR tyrosine kinase inhibitor"
     (NOT a hardcoded string -- the test must call link_to_cuis()
     and resolve at test time so it stays correct if scispacy's
     KB is upgraded later). Mark this test @pytest.mark.slow
     because link_to_cuis() loads the linker.
   - Spot-checks Q416: same shape -- look up scispacy CUI for
     "trastuzumab" + "HER2-targeted therapy" via link_to_cuis(),
     assert the membership. @pytest.mark.slow.
   - For the fast-suite tests (no linker), only the shape/format
     assertions run; the canonical-case lookups gate on
     link_to_cuis(), so CI without the KB skips cleanly via the
     same _KB_CACHED gate A6 uses.

Show me the full file + the test file. Don't promise more
coverage than you've actually built -- if a drug's class CUI
isn't in UMLS, say so and leave a TODO.
```

### Verify

```
/opt/miniconda3/envs/trialmine/bin/python -m pytest tests/unit/test_drug_classes.py -v

# Then verify the table actually solves Q413 + Q416:
/opt/miniconda3/envs/trialmine/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
from TrialMine.features.concepts import match_via_cui

# Q413 -- osimertinib failed (EGFR TKI naive trials should be Unmet).
# Uses the class phrase that A3 Test 5 verified resolves cleanly.
ok, matched = match_via_cui("osimertinib", "EGFR tyrosine kinase inhibitor")
print(f"Q413 osimertinib <-> EGFR TKI: {ok} via {matched}")
assert ok, "Q413 must match after A4"

# Q416 -- trastuzumab failed (HER2-naive trials should be Unmet).
# NOTE: uses "HER2 inhibitor" not "HER2-targeted therapy" -- the latter
# does NOT resolve to a drug-type CUI in scispacy's 2022 KB (see Findings
# #2 in the A4 prompt header). This is a known limitation: real trial
# eligibility text that uses the phrase "HER2-targeted therapy" verbatim
# will not match via UMLS alone -- it needs to either (a) name the drug
# (trastuzumab, pertuzumab, T-DM1), or (b) use the phrase "HER2
# inhibitor" / "anti-HER2 mAb" / similar that DOES resolve.
ok, matched = match_via_cui("trastuzumab", "HER2 inhibitor")
print(f"Q416 trastuzumab <-> HER2 inhibitor: {ok} via {matched}")
assert ok, "Q416 must match after A4"

# Brand-name aliasing -- verifies that Tagrisso ↔ osimertinib resolves
# via the SHARED CLASS path (not direct CUI overlap, because scispacy's
# 2022 KB gives them different CUIs: C4058811 vs C4058817).
ok, matched = match_via_cui("Tagrisso", "osimertinib")
print(f"Brand alias Tagrisso <-> osimertinib: {ok} via {matched}")
assert ok, "Brand-name keys must shared-class-match their generic"

# Negative control: unrelated drugs in DIFFERENT classes should NOT match.
ok, matched = match_via_cui("osimertinib", "trastuzumab")
print(f"Negative control osimertinib <-> trastuzumab: {ok}")
assert not ok, "Unrelated drugs must NOT match"
PY
```

### Acceptance criteria

- `tests/unit/test_drug_classes.py` passes (≥ 20 entries, all CUI format,
  all frozenset, Q413 + Q416 spot-checks via `link_to_cuis()`)
- Q413 + Q416 matches succeed via `match_via_cui` (note: Q416 uses
  the "HER2 inhibitor" class phrase per Findings #2)
- Brand-name aliasing works (Tagrisso ↔ osimertinib matches via
  shared-class path, not direct CUI overlap)
- Negative control fails (osimertinib doesn't match trastuzumab —
  different classes)
- Every entry has a clinical-note comment + scispacy 2022-KB-verified
  CUI (NOT public-UMLS-browser-verified; see "CRITICAL" preamble)

### Rollback

```
rm src/TrialMine/features/drug_classes.py tests/unit/test_drug_classes.py
```

`concepts.py:match_via_cui` will lazy-import-fail back to direct-CUI-only
matching (alias matches still work). The ImportError fallback in A3 is
the safety net.

---

## Step A5 — Wire CUI matching into `tools.py` + add `DegradationConfig` toggle (75 min)

**Goal:** add a parallel `_any_overlap_cui` matcher in `tools.py` that
mirrors `_any_overlap`'s signature, then gate
`_match_required_treatments` + `_match_excluded_treatments` on the new
`DegradationConfig.umls_drug_class_matching_enabled` toggle. **Leave the
condition matchers (`_match_required_conditions` /
`_match_excluded_conditions`) on the existing substring matcher** — UMLS
on free-text conditions is a different problem (over-linking risk,
patient-vs-trial wording mismatch) and out of scope for this runbook.

### Prompt to send

```
Phase A5 of docs/fix_parser_umls.md -- wire UMLS matching into
the eligibility check path. Four edits, in order:

1. src/TrialMine/config.py -- add a new toggle to
   DegradationConfig:

       umls_drug_class_matching_enabled: bool = Field(
           default=False,
           description=(
               "Hard toggle: when True, _match_required_treatments "
               "and _match_excluded_treatments use UMLS CUI matching "
               "(via TrialMine.features.concepts.match_via_cui) "
               "instead of literal substring matching. Resolves "
               "drug-name <-> drug-class equivalences "
               "(osimertinib <-> EGFR TKI, trastuzumab <-> "
               "HER2-targeted therapy) backed by a hand-curated "
               "DRUG_TO_CLASS_CUIS table. Per-query latency cost: "
               "~50-200 ms warm (LRU cache) / ~1-2 s cold. Default "
               "is False until Phase D ships per fix_parser_umls.md."
           ),
       )

2. src/TrialMine/agents/tools.py -- add a parallel matcher
   right below the existing _any_overlap definition (~line 522):

       def _any_overlap_cui(
           needles: list[str],
           haystack: str | None,
       ) -> tuple[bool, list[str]]:
           """Like _any_overlap but uses UMLS CUI matching.

           Each needle is tested for CUI overlap (direct OR via
           the hand-curated DRUG_TO_CLASS_CUIS table) against
           the haystack via TrialMine.features.concepts.match_via_cui.
           Falls back to (False, []) on ImportError so the wider
           matcher path stays robust if scispacy / the linker
           aren't available.
           """
           if not haystack or not needles:
               return False, []
           try:
               from TrialMine.features.concepts import match_via_cui
           except ImportError:
               logger.warning(
                   "scispacy / UMLS linker not available; "
                   "_any_overlap_cui falling back to no-match"
               )
               return False, []
           matches: list[str] = []
           for needle in needles:
               if not needle:
                   continue
               ok, _via = match_via_cui(needle, haystack)
               if ok:
                   matches.append(needle)
           return bool(matches), matches

3. src/TrialMine/agents/tools.py -- in BOTH
   _match_required_treatments AND _match_excluded_treatments
   (~lines 583 and 614), replace the `_any_overlap(...)` call
   with a toggle-gated call. Example for
   _match_required_treatments:

       # before:
       found, matches = _any_overlap(required, prior_treatments)

       # after:
       from TrialMine.config import get_default_degradation
       if get_default_degradation().umls_drug_class_matching_enabled:
           found, matches = _any_overlap_cui(required, prior_treatments)
       else:
           found, matches = _any_overlap(required, prior_treatments)

   DO NOT touch _match_required_conditions or
   _match_excluded_conditions. Their substring matcher is left
   alone -- UMLS on free-text conditions is out of scope.

4. Update tools.py's check_trial_eligibility docstring (~line
   657-688) to note that prior-treatment matching is
   UMLS-aware when the toggle is on.

Use type hints + structured logging + no print(). Show me the
full diff for tools.py and config.py.
```

### Verify

```
# Toggle OFF (default): nothing should change.
OMP_NUM_THREADS=1 /opt/miniconda3/envs/trialmine/bin/python -m pytest \
  tests/unit/test_eligibility.py tests/unit/test_degradation.py -q
# Should still pass.

# Toggle ON: smoke test the two matchers head-to-head.
OMP_NUM_THREADS=1 PYTHONPATH=src \
  /opt/miniconda3/envs/trialmine/bin/python - <<'PY'
from TrialMine.config import get_default_degradation
get_default_degradation().umls_drug_class_matching_enabled = True

from TrialMine.agents.tools import _any_overlap, _any_overlap_cui
print("Substring: osimertinib vs 'no prior EGFR TKI' ->",
      _any_overlap(["osimertinib"], "no prior EGFR TKI"))
print("CUI:       osimertinib vs 'no prior EGFR TKI' ->",
      _any_overlap_cui(["osimertinib"], "no prior EGFR TKI"))
# Expect substring -> (False, []); CUI -> (True, ['osimertinib'])
# NOTE: the CUI -> (True, …) outcome above only holds AFTER Phase A5b
# ships. scispacy's 2022 UMLS KB does NOT index the abbreviation "TKI"
# as a drug-class concept — only the spelled-out form
# "tyrosine kinase inhibitor" resolves. Phase A5b adds a pre-expansion
# step inside link_to_cuis() (TKI -> tyrosine kinase inhibitor, plus
# ICI and PARP-i shorthand) that fixes this and unblocks the CUI path
# for abbreviation-heavy eligibility text.
PY
```

### Acceptance criteria

- All existing tests pass with toggle off (no regressions)
- Substring matcher returns `(False, [])` for osimertinib vs 'EGFR TKI'
  text
- CUI matcher returns `(True, ['osimertinib'])` for the same input
- No `print()` in new code; `logger.warning` only on import failure
- Diff shows ZERO changes to `_match_required_conditions` or
  `_match_excluded_conditions`

### Rollback

```
git checkout src/TrialMine/agents/tools.py src/TrialMine/config.py
```

---

## Step A5b — Pre-expand oncology abbreviations before UMLS linking (~30 min)

**Goal:** scispacy's 2022 UMLS KB indexes spelled-out drug-class
concepts (`"tyrosine kinase inhibitor"`, `"immune checkpoint inhibitor"`,
`"PARP inhibitor"`) but **not** their common shorthand forms
(`"TKI"`, `"ICI"`, `"PARP-i"`). Without pre-expansion, eligibility text
written in shorthand silently fails to link, and
:func:`match_via_cui` returns `False` even when the clinical
equivalence is obvious. Phase A5's verify discovered this: the test
haystack `"no prior EGFR TKI"` returned `(False, [])` from the CUI
matcher because scispacy can't link `"TKI"`.

A5b adds a small hand-curated abbreviation-expansion step inside
:func:`link_to_cuis` so the linker sees the spelled-out form. Three
abbreviations covered (TKI / ICI / PARP-i + plural and `-i`-suffix
variants). Bare protein / gene names (PARP, EGFR, HER2, CDK4/6) are
deliberately NOT expanded — they're not drug classes and would
create false matches.

### Prompt to send

```
Phase A5b of docs/fix_parser_umls.md -- add oncology abbreviation
pre-expansion to TrialMine.features.concepts before UMLS linking.

1. src/TrialMine/features/concepts.py -- add a module-level constant
   _ABBREVIATION_EXPANSIONS (tuple of (compiled regex, replacement
   string) pairs) and a private _expand_abbreviations(text: str) -> str
   helper. Cover:
     - \bTKIs?\b           -> "tyrosine kinase inhibitor"
     - \bICIs?\b           -> "immune checkpoint inhibitor"
     - \bPARP-?[iI]\b      -> "PARP inhibitor"
   Use case-insensitive regex for TKI / ICI. PARP-i is intentionally
   case-sensitive on the trailing [iI] to avoid matching the protein
   name "PARP" alone (which is NOT a drug class).

2. In link_to_cuis, call _expand_abbreviations(text) AFTER the
   empty-string guard, BEFORE passing the text to the spaCy
   pipeline. Update the docstring to mention the expansion.

3. Add tests to tests/unit/test_concepts.py:
   - 4 fast (no KB needed) regex tests covering:
     * TKI / TKIs / "EGFR TKI" / "TKI:" all expand correctly; "TKIs2"
       and "aTKIb" do NOT (word-boundary correctness).
     * ICI / ICIs and PARP-i / PARPi / PARP-I / PARPI all expand.
     * Bare PARP / EGFR / HER2 / CDK4/6 stay UNCHANGED.
     * Already-spelled-out forms idempotent (expand(expand(x)) == expand(x)).
   - 3 slow KB-gated tests (use the same _KB_CACHED filesystem check
     as test_drug_classes.py):
     * link_to_cuis("EGFR TKI") overlaps link_to_cuis("EGFR tyrosine
       kinase inhibitor").
     * link_to_cuis("PARPi") overlaps link_to_cuis("PARP inhibitor").
     * link_to_cuis("PARP") does NOT overlap link_to_cuis("PARP
       inhibitor") -- the bare protein name must not become a drug.

Type hints + structured logging + no print().
```

### Verify

```
# Fast tests (no KB):
OMP_NUM_THREADS=1 /opt/miniconda3/envs/trialmine/bin/python -m pytest \
  tests/unit/test_concepts.py -q -m "not slow"

# Slow tests (with KB):
OMP_NUM_THREADS=1 /opt/miniconda3/envs/trialmine/bin/python -m pytest \
  tests/unit/test_concepts.py -q

# Re-run the A5 verify smoke — the CUI matcher should now return True:
OMP_NUM_THREADS=1 PYTHONPATH=src \
  /opt/miniconda3/envs/trialmine/bin/python - <<'PY'
from TrialMine.config import get_default_degradation
get_default_degradation().umls_drug_class_matching_enabled = True
from TrialMine.agents.tools import _any_overlap_cui
print("CUI: osimertinib vs 'no prior EGFR TKI' ->",
      _any_overlap_cui(["osimertinib"], "no prior EGFR TKI"))
print("CUI: pembrolizumab vs 'no prior ICI' ->",
      _any_overlap_cui(["pembrolizumab"], "no prior ICI"))
# Both should return (True, [<drug>])
PY
```

### Acceptance criteria

- Fast tests pass (≤ 1 s) — 4 new regex tests + the existing 4
  ConceptNormalizer tests still green.
- Slow tests pass (≤ 90 s on cold linker) — 3 new link_to_cuis
  integration tests.
- A5 verify smoke now returns `(True, ['osimertinib'])` for the
  `"no prior EGFR TKI"` haystack.
- ICI shorthand also matches: `pembrolizumab` ↔ `"no prior ICI"` → True.
- No `print()`; word-boundary regex; bare protein names unchanged.

### Known limitation (recorded 2026-05-14)

A5b verification surfaced one downstream scispacy NER quirk worth
recording: even with the abbreviation expansion in place,
`link_to_cuis("no prior PARP inhibitor")` returns no drug-type CUIs
because scispacy's 2022 KB NER doesn't extract `"PARP inhibitor"` as a
single entity in some sentence contexts (the bare phrase
`"PARP inhibitor"` does resolve cleanly to C1882413; preceding it with
`"no prior"` breaks the extraction). TKI and ICI cases don't hit this
quirk. Impact should be narrow because the patient side usually
mentions the drug name (olaparib, niraparib) and the trial side comes
from `parsed_eligibility` which stores pre-extracted entity strings,
not raw sentences. Phase B2 will measure whether this matters in
practice; if it does, the planned intervention is a direct-CUI
fallback in `link_to_cuis` that recognizes `"PARP inhibitor"` (and
similar) as a fixed string regardless of NER outcome.

### Rollback

```
git checkout src/TrialMine/features/concepts.py \
              tests/unit/test_concepts.py
```

---

## Step A6 — Add unit tests for the new matching behavior (60 min)

**Goal:** lock in the new matching behavior with tests that document the
canonical Q413 + Q416 cases. Mark linker tests `slow` so CI without the
KB doesn't hang.

### Prompt to send

```
Phase A6 of docs/fix_parser_umls.md -- add tests for the new
matching behavior. Two test files to extend:

1. tests/unit/test_concepts.py -- EXTEND (don't replace existing
   tests):

   Mark new tests @pytest.mark.slow because they load the UMLS
   KB. Add a module-scoped fixture or rely on link_to_cuis's
   LRU to amortize.

   Skip-gate every linker-dependent test. NOTE: scispacy 0.6.2's
   UmlsLinkerPaths.is_locally_cached() is unreliable -- it returns
   False even when the KB is fully cached at ~/.scispacy/datasets/
   (confirmed in the Phase A2 run on 2026-05-14). Check the
   filesystem directly instead:
     import pathlib
     _SCISPACY_KB_DIR = pathlib.Path.home() / ".scispacy" / "datasets"
     _KB_CACHED = (
         _SCISPACY_KB_DIR.exists() and any(_SCISPACY_KB_DIR.iterdir())
     )
     pytestmark = pytest.mark.skipif(
         not _KB_CACHED,
         reason="scispacy UMLS KB not downloaded; run Phase A2 first"
     )

   Required tests:
     - test_link_known_drug_resolves_to_cui -- osimertinib gets
       at least one drug-type CUI
     - test_link_drug_class_resolves_to_cui -- "EGFR tyrosine
       kinase inhibitor" gets at least one CUI
     - test_link_filters_out_anatomy -- "kidney" gets zero
       drug-type CUIs after the semantic-type filter
     - test_link_to_cuis_returns_tuple -- result is a tuple
       (immutable, hashable)
     - test_link_to_cuis_caches -- same arg twice; second call
       in <5 ms after warm cache
     - test_match_via_cui_alias -- Tagrisso <-> osimertinib
       (direct CUI overlap; no DRUG_TO_CLASS_CUIS needed)
     - test_match_via_cui_drug_to_class -- osimertinib <->
       "EGFR tyrosine kinase inhibitor" (requires
       DRUG_TO_CLASS_CUIS from A4)
     - test_match_via_cui_unrelated_drugs -- osimertinib <->
       trastuzumab returns False
     - test_match_via_cui_empty_input -- empty string <->
       anything returns (False, [])

2. tests/unit/test_eligibility.py -- EXTEND:

   Three @pytest.mark.slow tests gated on the toggle:

     - test_q413_failed_osimertinib_egfr_tki_naive_trial
       (Q413 canonical -- patient prior_treatments has
        "osimertinib", trial required_prior_treatments has
        "EGFR TKI naive" or similar -- verdict should be Unmet)
     - test_q416_post_trastuzumab_her2_naive_trial (Q416)
     - test_toggle_off_uses_literal_match -- same Q413 inputs
       but toggle off, verdict is Unknown or Met (because
       substring fails) -- control test for the toggle

   Use a try/finally that resets the toggle on teardown -- don't
   leak global state between tests.

All new tests must:
- Use existing pytest patterns (look at tests/unit/test_concepts.py
  and tests/unit/test_eligibility.py to mirror style)
- Skip cleanly when UmlsLinkerPaths.is_locally_cached() is False
  (so CI without KB doesn't hang)
- Be deterministic: assert CUI membership in expected sets, NOT
  exact equality (scispacy can reorder the candidates list)
- Total runtime <= 60 s on warm KB cache

Show me each diff.
```

### Verify

```
# Fast suite (no linker load):
OMP_NUM_THREADS=1 /opt/miniconda3/envs/trialmine/bin/python -m pytest \
  tests/unit/test_concepts.py tests/unit/test_eligibility.py \
  tests/unit/test_drug_classes.py -q -m "not slow"

# Full suite (with linker):
OMP_NUM_THREADS=1 /opt/miniconda3/envs/trialmine/bin/python -m pytest \
  tests/unit/test_concepts.py tests/unit/test_eligibility.py \
  tests/unit/test_drug_classes.py -q
```

### Acceptance criteria

- Fast suite (no linker) passes in ≤ 5 s
- Full suite (with linker) passes in ≤ 60 s after warm cache
- Q413 + Q416 canonical tests pass with toggle on
- Toggle-off control passes (substring match fails as expected)
- Tests skip cleanly when KB is not cached locally

### Rollback

```
git checkout tests/unit/test_concepts.py tests/unit/test_eligibility.py
```

---

## ✅ Phase A done — checkpoint before eval

State at this point:
- `concepts.py` has `UmlsConceptLinker` + `link_to_cuis` + `match_via_cui`;
  `extract_concepts` / `normalise_concept` stubs removed
- `drug_classes.py` exists with the hand-curated `DRUG_TO_CLASS_CUIS`
  table (~20+ entries, all CUI-verified)
- `tools.py` has parallel `_any_overlap_cui`; both prior-treatment
  matchers gate on toggle
- `config.py` has the `umls_drug_class_matching_enabled` toggle (default
  OFF)
- `eligibility_filter.py` unchanged — `required_prior_treatments` not yet
  in `HARD_FILTER_CRITERIA` (that's a D1 ship-only edit)
- Tests cover Q413 + Q416 canonical cases and pass

### Optional commit point

```
Show me git status. If there are pending edits, propose a commit
message for the Phase A changes and stage the right files
(features/, config.py, tools.py, tests/, fix_parser_umls.md). Do
NOT commit yet -- confirm with me first.
```

---

# Phase B — Focused re-eval (~1 hr, ~$0.50–1 in Haiku)

**Why a focused re-eval (not the full 65q):** the change only affects
queries that mention prior treatments. Of the held-out 65q + 20q
expansion, only the 30 complex+vague queries involve prior-treatment
matching. Re-labelling those is cheap and the signal is focused.

## Step B1 — Write `scripts/eval_parser_umls.py` (60 min)

**Goal:** focused harness that runs the production pipeline with the
toggle ON vs OFF on the same 30 complex+vague queries, with the same
Haiku judge.

### Prompt to send

```
Phase B1 of docs/fix_parser_umls.md -- write
scripts/eval_parser_umls.py. The script must:

1. Load the 30 complex+vague queries (15 complex + 15 vague):
     - data/evaluation/full_labeled_dataset.jsonl (5 complex +
       5 vague in the 65q)
     - data/evaluation/full_labeled_dataset_expansion_v2.jsonl
       (10 complex + 10 vague in the 20q)
   Filter to category in {"complex","vague"}. Confirm count
   is exactly 30 before proceeding (assert).

2. For each query, run the full pipeline TWICE via a context-
   manager toggle helper:

       from contextlib import contextmanager

       @contextmanager
       def umls_toggle(enabled: bool):
           cfg = get_default_degradation()
           prev = cfg.umls_drug_class_matching_enabled
           cfg.umls_drug_class_matching_enabled = enabled
           try:
               yield
           finally:
               cfg.umls_drug_class_matching_enabled = prev

   (NOT global mutation -- risks leaking into other tests.)

3. Capture from each run:
     - Top-10 NCT IDs from the orchestrator output
     - Per-trial eligibility verdict counts for
       required_prior_treatments (how many trials newly Unmet?)
     - Wall-clock latency

4. Re-label any NEW top-10 trials not in the existing label
   files. Use Claude Haiku with the same prompt as
   scripts/build_full_eval_dataset.py. Resume-able via
   checkpoint file after every 5 queries.

5. Output:
     - data/evaluation/full_labeled_complex_vague_umls.jsonl
       -- new pairs labeled (~50-100 expected)
     - data/evaluation/umls_eval_metrics.json -- per-category
       NDCG@5/10 + MRR + bootstrap 95% CIs for both runs over
       the 30 queries
     - Console summary table:
         Category | OFF NDCG@5 | ON NDCG@5 | Delta | OFF #drops | ON #drops

Reuse bootstrap_ci from scripts/c2_eval_pure_ce.py and the
Haiku labelling prompt from scripts/build_full_eval_dataset.py
-- do NOT reinvent either.

DO NOT run yet.
```

### Verify

```
Show me scripts/eval_parser_umls.py. Confirm:
- Reads BOTH source label files and filters to category in
  {"complex","vague"} -- count assert at 30
- Uses the context-manager pattern for the toggle
- Resume logic in place (checkpoint after every 5 queries)
- Bootstrap CIs computed (not just point estimates)
- No print(); structured logging only
```

### Acceptance criteria

- Script identifies exactly 30 queries — `assert` fires if count is wrong
- Toggle is applied via context manager
- Resume logic in place
- No `print()`; structured logging only

### Rollback

```
rm scripts/eval_parser_umls.py
rm -f data/evaluation/full_labeled_complex_vague_umls.jsonl
rm -f data/evaluation/umls_eval_metrics.json
```

---

## Step B2 — Run the focused re-eval (~30 min wall-clock, ~$0.50–1 in Haiku)

### Prompt to send

```
Phase B2 of docs/fix_parser_umls.md -- run the focused re-eval.

Pre-flight:
- docker start es (Elasticsearch up)
- set -a; source .env; set +a (export ANTHROPIC_API_KEY)
- confirm ~$1 of Haiku budget

Run:
  OMP_NUM_THREADS=1 PYTHONPATH=src \
    /opt/miniconda3/envs/trialmine/bin/python \
    scripts/eval_parser_umls.py 2>&1 | tee /tmp/umls_eval.log

Expected wall-clock: ~30 min (30 queries x 2 runs x ~10 s/run
+ Haiku labelling).

After it finishes, parse the output and report:
  - umls_off NDCG@5 on complex aggregate (with CI)
  - umls_on  NDCG@5 on complex aggregate (with CI)
  - Same for vague
  - Per-query: which queries had new trials marked Unmet
  - Q413 + Q416 specifically -- confirm >= 1 newly-Unmet trial
    each (canonical cases)
  - Latency overhead: ON wall-clock vs OFF wall-clock per query
```

### Verify (signals to look for in /tmp/umls_eval.log)

1. **Label cache hit rate ≥ 70 %.** Lower = new pipeline surfaced very
   different candidates → Haiku budget probably blew out.
2. **Per-query verdict-count delta.** Healthy: 1–5 new Unmet trials per
   complex query; 0–3 on vague.
3. **Q413 + Q416 each have ≥ 1 newly-Unmet trial.** If neither does, the
   linker didn't catch the canonical case — investigate before shipping.
4. **Aggregate NDCG@5 delta on complex in [-0.02, +0.06].** Outside that
   range = unexpected; either over-linking causing false drops, or
   under-coverage in `DRUG_TO_CLASS_CUIS`.
5. **Latency overhead < 1 s/query** with warm cache.

### Acceptance criteria

- Eval completes without uncaught exception
- Q413 + Q416 visibly flag at least one previously-undropped trial each
- Aggregate complex NDCG@5 delta in the expected range
- Per-query trigger rate healthy (not 0/15 = matcher isn't firing; not
  15/15 = over-linking)

### Rollback

If the eval surfaces a problem (over-linking, no triggers, NDCG
regression): re-do A3 with a tighter linker threshold (0.85 → 0.90) or
trim A4's table, then re-run B2.

---

# Phase C — Decision gate (manual review, no code)

Same Decision-36 / Decision-40 framing the rest of the project uses.
Holistic review across signals; no rigid pre-registered thresholds.

## Signals to assemble before deciding

### 1. Drug-class table coverage

- Of the 15 complex queries that mention a specific drug, how many drugs
  are in `DRUG_TO_CLASS_CUIS`?
- If coverage < 70 %, metric impact is capped by the table — extending
  the table is cheaper than another eval cycle.

### 2. Aggregate metric impact (the headline)

- complex NDCG@5: toggle-off vs toggle-on. Bootstrap 95 % CIs.
- vague NDCG@5: same.
- **Expected** (per § 11.6 of `docs/evaluation-report.md`): +0.02–0.04 on
  complex aggregate, ~0 on vague (vague rarely has prior-treatment terms).
- Bigger lift on complex = positive surprise. Smaller = drug-class
  coverage is patchy; extending the table is the next dollar.

### 3. Canonical-failure correctness (qualitative)

- Q413 (failed osimertinib): which trials moved Unmet under UMLS?
  Spot-check 3–5 by hand — are they actually osimertinib-naive-required?
- Q416 (post-trastuzumab progression): same for HER2-naive trials.

### 4. Parser precision / over-linking risk (the downside)

- Trigger rate per query. Healthy: 1–5 trials newly Unmet per complex
  query; 0–3 per vague.
- If much higher (10+), the matcher is over-linking. Spot-check 5 random
  newly-Unmet drops — `trastuzumab` matching `PI3K inhibitor` would be a
  red flag.

### 5. Coverage of the failure mode

- Of the 15 complex queries, how many actually MENTION a specific drug?
  If < ~8, UMLS only helps half of complex — the rest are bottlenecked by
  other parser limitations (caregiver phrasings, multi-constraint queries,
  biomarker matching).

### 6. Cost of revert

- Single toggle flip in `config.py`. Zero data loss. Linker stays
  installed; code stays present.

## How to decide

- **Ship if positive overall:** aggregate complex NDCG@5 +0.02+ (any CI),
  Q413 + Q416 visibly correct, trigger rate healthy, no over-linking.
- **Ship anyway with honest writeup** (Decision-36 framing) if metric is
  tied/small but qualitative correctness is real. NDCG undersells because
  Haiku judge has the same drug-class blind spot we're fixing.
- **Hold** if metric tied AND correctness is mixed. Tighten threshold or
  trim table, re-run B2.
- **Revert** if metric regresses 0.02+ on complex OR over-linking obvious
  on spot-check.

### Prompt to send (after laying out the signals)

```
My decision after Phase C of docs/fix_parser_umls.md is:
[SHIP | SHIP-ANYWAY-with-writeup | HOLD-and-tune | REVERT].

Assembled signals:
- Drug-class table coverage of complex queries with drug
  mentions: <%>
- complex NDCG@5: off=<num> [CI] -> on=<num> [CI] (Delta=<delta>)
- vague NDCG@5: off=<num> [CI] -> on=<num> [CI] (Delta=<delta>)
- Q413 visibly correct? <Y/N>; Q416? <Y/N>
- Average newly-Unmet/complex query: <num>
- Spot-check 5 random new drops -- any wrong? <Y/N + details>
- Latency overhead per query: <ms> warm, <ms> cold

Reasoning: <2-3 sentences on the overall picture>

Proceed to Step D1 (ship) / D-hold (hold) / E-revert (revert).
```

---

# Phase D — Ship (~30 min, only if Phase C is SHIP)

## Step D1 — Add `required_prior_treatments` to `HARD_FILTER_CRITERIA` + flip toggle default

### Prompt to send

```
Phase D1 of docs/fix_parser_umls.md -- ship the parser
upgrade.

1. src/TrialMine/features/eligibility_filter.py:
   - Add "required_prior_treatments" to HARD_FILTER_CRITERIA
     (tuple becomes 4 elements)
   - Update inline docstring noting Phase 13:
     "required_prior_treatments precision was gated on
     drug-class matching; with UMLS via SciSpacy (Phase 13) it
     crosses ~90% precision per Phase C B2 numbers and joins
     the closed-vocab whitelist."

2. src/TrialMine/config.py:
   - Flip default of umls_drug_class_matching_enabled from
     False -> True
   - Update docstring: "ON by default since Phase 13 ship".

3. Restart the API + smoke search:

     pkill -f "uvicorn TrialMine.api" 2>/dev/null; sleep 2
     set -a; source .env; set +a
     PYTHONPATH=src OMP_NUM_THREADS=1 \
       nohup /opt/miniconda3/envs/trialmine/bin/python -m uvicorn \
       TrialMine.api.app:app --host 127.0.0.1 --port 8000 \
       > /tmp/trialmine_api.log 2>&1 &
     sleep 35  # extra time for first UMLS KB load if cold
     grep -E "scispacy_linker|UMLS|Linker" /tmp/trialmine_api.log

     curl -X POST http://localhost:8000/api/v1/search \
       -H "Content-Type: application/json" \
       -d '{"query":"50F EGFR positive lung cancer failed osimertinib",
            "top_k":3,"use_agent":true}' \
       | python3 -m json.tool | head -60

   Expected: top-3 results exclude osimertinib-naive-required
   trials. Log shows UMLS match telemetry ("UMLS match
   osimertinib <-> ... via [...]" from concepts.match_via_cui).

Show me each diff before any commit.
```

### Verify

1. `/health` → 200
2. Smoke search returns ≥ 3 results, none osimertinib-naive-required
3. `/tmp/trialmine_api.log` contains UMLS match telemetry
4. `pytest tests/ -k "concepts or eligibility or drug"` passes

### Acceptance criteria

- API serves the new behavior
- Q413-shaped query drops osimertinib-naive trials visibly
- All tests pass

### Rollback

```
# One-line revert: hand-edit src/TrialMine/config.py to flip
# umls_drug_class_matching_enabled back to default=False.
# Restart API; matcher falls back to literal substring.
```

---

## Step D5 — Update `docs/evaluation-report.md` § 11.6

### Prompt to send

```
Phase D5 of docs/fix_parser_umls.md -- update section 11.6 of
docs/evaluation-report.md.

Locate the section 11.6 "Persistently weak slices" subsection.
Append a "Resolution shipped 2026-05-1X" paragraph that:
- Cites Phase 13 + links to docs/fix_parser_umls.md
- Documents the measured impact from B2 (complex NDCG@5 delta
  + Q413/Q416 qualitative)
- Honest framing: NDCG impact modest because Haiku judge has
  the same drug-class blind spot we're fixing; bigger win is
  clinical correctness
- Does NOT remove the section 11.6 limitation acknowledgement
  -- just appends a resolution note

Show me the diff. ~10-15 line addition.
```

---

## Step D6 — CLAUDE.md updates

### Prompt to send

```
Phase D6 of docs/fix_parser_umls.md -- update CLAUDE.md.

1. "Current State" header -- add a Phase 13 entry near the top
   (mirror the Phase 11/12 entries' structure):
     "Phase: 13 (eligibility parser UMLS upgrade) -- Week 11+.
      Wired SciSpacy EntityLinker (UMLS KB) into ..."

2. "What's working" -- extend the Eligibility parser entry to
   mention UMLS-backed prior-treatment matching + hand-curated
   DRUG_TO_CLASS_CUIS table

3. "Key files/data" -- add:
     - src/TrialMine/features/drug_classes.py
     - data/evaluation/full_labeled_complex_vague_umls.jsonl
     - data/evaluation/umls_eval_metrics.json

4. "Design decisions" -- add Decision 41 (mirror Decision 38 +
   Decision 40 structure):
     - Why: drug-class equivalences were the bottleneck per
       Phase C5a + section 11.6
     - How: SciSpacy CUI canonicalization + hand-curated
       DRUG_TO_CLASS_CUIS table for ~20-30 oncology drugs.
       Explicitly NOT UMLS hierarchy traversal (SciSpacy KB
       doesn't expose parent relations); substituted curation
       for hierarchy.
     - Honest read: aggregate NDCG lift modest (+0.02-0.04);
       main value is clinical correctness on Q413/Q416. Haiku
       judge bias undersells the win (same drug-class blind
       spot).
     - Trade-offs: linker ~1-2 GB on disk; per-query latency
       +50-200 ms warm / +1-2 s cold. Drug-class table is
       hand-maintained -- when it exceeds ~50 entries, switch
       to UMLS REST API (umls_api_key field already in
       Settings).

Show me each diff.
```

---

## Step D7 — Mark `things_can_be_fixed.md` issue resolved

### Prompt to send

```
Phase D7 of docs/fix_parser_umls.md -- update
docs/things_can_be_fixed.md. Locate the issue describing the
parser's required_prior_treatments literal-string-match
limitation (near Decision 18 references). Add an "Update
2026-05-1X" banner above the issue noting Phase 13 shipped
UMLS via SciSpacy EntityLinker + a hand-curated drug-class
table, with links to docs/fix_parser_umls.md and to
docs/evaluation-report.md section 11.6. Show me the diff.
```

---

# Phase E — Hold/Revert (if Phase C decision is HOLD or REVERT)

## E-Hold path (~30 min, no shipping)

```
Phase E-Hold of docs/fix_parser_umls.md.

1. Tighten the linker:
   - Raise threshold from 0.85 -> 0.90 or 0.95 in
     concepts.py:_get_linker_nlp()
   - Optionally trim _DRUG_SEMANTIC_TYPES (drop T109 Organic
     Chemical if main source of false positives)
   - Optionally remove edge-case entries from
     DRUG_TO_CLASS_CUIS in drug_classes.py if specific drugs
     are over-matching

2. Re-run Phase B2 with the tighter config. Reuse existing
   label file (full_labeled_complex_vague_umls.jsonl) -- no
   new Haiku spend if no new trials surface; ~$0.10 if a few do.

3. Re-evaluate per Phase C. Cycle until SHIP or REVERT.

No production change while holding -- toggle stays False.
```

## E-Revert path (~10 min)

```
Phase E-Revert of docs/fix_parser_umls.md.

1. Leave config toggle at False (default).
2. Keep all A-phase code -- linker module, helpers, drug_classes
   table, tests stay. Useful evidence + re-try path.
3. Document the failed attempt in docs/things_can_be_fixed.md:
   - What was tried (Phase A wiring + B2 eval)
   - What didn't work (specific numbers from B2)
   - Next intervention candidates:
     (a) Extend DRUG_TO_CLASS_CUIS -- coverage may be binding
     (b) UMLS REST API for live hierarchy (umls_api_key already
         in Settings, currently empty) -- more complete than
         scispacy KB alone
     (c) RxClass NIH API (linker_name="rxnorm" then class
         lookup) -- alternative to UMLS
     (d) GPT-4 / Sonnet at query time for drug-class
         equivalences -- slower + costlier but no over-linking

Show me each step before executing.
```

---

# Appendix — Failure modes to specifically watch for

| Symptom | Likely cause | Fix |
|---|---|---|
| `EntityLinker` imports OK but `add_pipe("scispacy_linker", ...)` fails | KB download interrupted | `rm -rf ~/.scispacy/` and retry A2 |
| First `link_to_cuis()` call takes > 30 s | First-call KB load — expected once per process | Pre-warm at API startup via existing lazy-load pattern |
| `osimertinib` doesn't resolve to any CUI | Linker threshold too high | Drop from 0.85 → 0.7 in A3 |
| `osimertinib` resolves but doesn't match "EGFR TKI" via `match_via_cui` | A4's table missing the osimertinib entry, OR the EGFR TKI class CUI in the table doesn't match what scispacy returns for "EGFR TKI" | `link_to_cuis("EGFR TKI")` — does the returned CUI match `DRUG_TO_CLASS_CUIS[osimertinib]`? If not, add the actual CUI |
| `osimertinib` ⟷ `trastuzumab` returns True (false positive) | Drug-class table has a transitive class CUI that bridges them (both mapped to generic "monoclonal antibody" or "kinase inhibitor") | Trim DRUG_TO_CLASS_CUIS to drug-FAMILY classes only; drop super-broad bridges |
| Q413 doesn't visibly improve in B2 | Trial corpus uses different wording — "no prior tyrosine kinase inhibitor" vs "no prior EGFR TKI" | Link both forms via `link_to_cuis`; if they don't share a CUI, it's an A4 table gap. If they DO share, debug at the `match_via_cui` level |
| Aggregate complex NDCG regresses in B2 | False-positive drops exceed true-positive drops | Tighten threshold (E-Hold) or trim drug-class table |
| `vague` NDCG moves substantially either direction | Unexpected — UMLS shouldn't affect queries with no drug names. If it does, toggle leaked into a non-treatment code path | Verify A5's edits are confined to `_match_required_treatments` + `_match_excluded_treatments` only |
| API restart hangs on UMLS load | KB load too slow for lifespan | Run KB load in a thread, return 200 from `/health` immediately, mark linker `loading=True` until ready |
| `link_to_cuis` test passes locally but CI fails | KB not cached in CI | A6's filesystem `_KB_CACHED` check (`~/.scispacy/datasets/` exists + non-empty) skip-gates the slow tests — CI skips cleanly. Do **not** use `UmlsLinkerPaths.is_locally_cached()` — it returns False even when the KB is fully cached |
| `match_via_cui("osimertinib", "EGFR TKI")` returns False even though A4's table looks complete | A4 used public UMLS browser CUIs (`C2700554` etc.) instead of scispacy's 2022-KB CUIs (osimertinib is `C4058811` in scispacy) | Re-run A3 Test 5 to discover the scispacy CUIs, rebuild `DRUG_TO_CLASS_CUIS` against those, re-run A4's smoke test |

---

# Appendix — Cost recap

| Item | Cost | Phase |
|---|---|---|
| Disk: UMLS KB | ~1–2 GB (one-time, auto-downloaded by `add_pipe`) | A2 |
| Haiku: focused re-eval on 30 complex+vague queries | ~$0.50–1 | B2 |
| Haiku: optional tighter-threshold re-eval | ~$0.10 | E-Hold (if needed) |
| Cloud: nothing — all CPU local | $0 | — |
| **Total** | **~$0.60–1.10** | |

Plus ~5–7 hr engineer wall-clock.

---

# Appendix — Project state going in

> Snapshot of the pieces of the existing pipeline this work plugs into.
> Read once before starting Phase A.

## C.1 — Eligibility filter (Phase 10b, SHIPPED)

`src/TrialMine/features/eligibility_filter.py` is the production filter
shared between the orchestrator (Step 5b) and the eval script. Today
`HARD_FILTER_CRITERIA = ("age", "sex", "excluded_prior_treatments")` —
only the closed-vocab criteria the existing parser nails with > 95 %
precision. `required_prior_treatments` is excluded today because the
matcher (`_any_overlap`, case-insensitive substring) misses drug-class
equivalences. Decision 38 documents the rationale.

This runbook flips that — once CUI matching is wired and Phase B
validates it, `required_prior_treatments` joins the whitelist in D1.

## C.2 — SciSpacy + the existing parser

`src/TrialMine/features/eligibility.py` already uses SciSpacy's
`en_core_sci_lg` for NER on biomedical entities. The new linker is
loaded into a **separate** `nlp` instance (the `_LINKER_NLP` singleton in
A3) — NOT the same one the parser uses — to avoid coupling the two
load-lifecycles. The two-NLP-objects cost is ~2× memory (~1 GB extra),
acceptable.

Phase 6c left `extract_concepts` and `normalise_concept` stubs in
`concepts.py` for this work (Decision 18 upgrade path). Both are unused
(zero callers); A3 deletes them.

## C.3 — Toggling new behavior

The project's established pattern for opt-in behavior is
`DegradationConfig` in `src/TrialMine/config.py`. Existing toggles
(`eligibility_hard_filter_enabled`, `cross_encoder_enabled`,
`eligibility_check_enabled`) all follow `bool = Field(default=…,
description=…)` with the process-wide singleton `_DEFAULT_DEGRADATION` +
`get_default_degradation()` accessor. Adding
`umls_drug_class_matching_enabled` follows the pattern. Default OFF
until Phase D1; one-line flip to revert.

`DegradationConfig` is a `pydantic.BaseModel`, NOT `BaseSettings` — the
toggle is in-process only, not env-var driven. Phase B's eval script
flips the singleton via a context manager (not env vars).

## C.4 — What this does NOT address (be honest in Phase D writeup)

- **Caregiver-phrasing patient profile extraction.** Vague queries hit 0
  % filter trigger rate because `QueryParserAgent` doesn't extract age/sex
  from "my dad has cancer" queries. UMLS doesn't touch this — it's a
  prompt-engineering fix for the QueryParser.
- **Multi-vector encoding (Issue #4).** Some trial fields the bi-encoder
  doesn't surface (e.g., specific exclusion criteria) — UMLS in the parser
  helps only when the entity *was* extracted by NER. If the entity isn't
  in the parser's output, no amount of linking helps.
- **Recall ceiling.** If hybrid retrieval doesn't surface a relevant
  trial in the top-K candidate pool, no parser or ranker can recover it.
- **The "drop LGB?" question.** Independent of parser work; still waiting
  on a pooled-labels eval (~$115).
- **UMLS hierarchy traversal.** SciSpacy's KB doesn't expose UMLS
  parent/child relations. We substitute hand curation
  (`DRUG_TO_CLASS_CUIS`) for hierarchy. When the table exceeds ~50
  entries, the right next step is the UMLS REST API (project already has
  `umls_api_key` field on `Settings`).

Ship UMLS for what it actually improves (drug-class matching → Q413 +
Q416 correctness + modest NDCG aggregate lift on complex). Don't oversell
as a "complex slice rescue" — the slice is multi-bottlenecked, and this
is one intervention.
