# Week 5 — Eligibility Parsing (Student Edition)

A step-by-step walkthrough of everything we built this session,
written as if I'm sitting next to you explaining each piece. By the end
you should be able to (a) explain the system to someone else, (b) navigate
every file we touched, and (c) answer interview questions about it.

If you've never done biomedical NLP, that's fine — every term is defined
when it first appears.

---

## Part 1 — Why are we doing this?

A clinical trial is a study where doctors test a new treatment on real
patients. To join, a patient has to meet the trial's "eligibility
criteria" — a list of rules like *"must be 18 or older"*, *"must have
non-small cell lung cancer"*, *"must NOT have had chemotherapy in the
past 4 weeks"*.

In our database (`data/trials.db`) every trial has these criteria stored
as one long blob of free-text English. Here's a real example
(NCT06073769):

```
Inclusion Criteria:

* Adult participants 19 years of age or older
* Participants who receive oral azacitidine according to the approved label
* For the first 2 years after marketing authorization, all participants
  who have received or are receiving oral azacitidine will also be registered

Exclusion Criteria:

* Participants who are prescribed oral azacitidine for therapeutic
  indications not approved in Korea
* Participants for whom oral azacitidine is contraindicated by the
  Korean prescribing information from the ministry of food and drug safety
```

A computer can't easily answer questions like *"is this trial accepting
adults?"* or *"does it exclude people who already had chemotherapy?"*
unless we **convert the text into structured fields** (numbers, lists,
flags) that a database can query.

That's the whole job for the week: turn 140,703 paragraphs of prose into
140,703 rows of clean data with columns like `min_age_years`,
`required_conditions`, `excluded_conditions`.

The technical name for this task is **information extraction**, and it's
the same problem people solve when they parse a resume into LinkedIn
fields, or a recipe into ingredient lists, or a doctor's notes into ICD
codes.

---

## Part 2 — Some terms you'll see

Before we go further, a quick glossary. You can skim this and come back.

- **NER (Named Entity Recognition)** — a machine learning model that
  reads text and highlights "important nouns" like diseases, drugs, or
  people. Example: given *"Patient was treated with bevacizumab"*, an
  NER returns `bevacizumab` as an entity.
- **SciSpacy** — a Python library (`pip install scispacy`) that's
  basically a medical version of spaCy. It ships with a model called
  `en_core_sci_lg` that's been trained on biomedical text.
- **Regex (regular expression)** — a pattern for matching text. Example:
  `r"\d+ years"` matches "18 years", "65 years", etc. Cheap, fast, but
  only as smart as the patterns you write.
- **BM25** — a classic search algorithm based on word matching. If your
  query says "lung cancer" and a trial says "lung cancer", BM25 scores
  it high. It does NOT understand that "lung neoplasm" means the same
  thing.
- **Pydantic** — a Python library for typed data classes. You declare
  fields with types, and Pydantic validates inputs and gives you
  autocomplete in your IDE.
- **NDCG@5** — a search-quality metric. "Did the right answers show up
  in your top 5?" Higher is better, max is 1.0. (Don't worry about the
  formula for now.)
- **UMLS** — a giant medical dictionary maintained by the National
  Library of Medicine. It has millions of medical terms with synonyms
  and unique IDs. Free with registration.

---

## Part 3 — What we built, step by step

We built five things this session. I'll walk through each in the order
we did them.

### Step 1 — Verify the tools work

**File:** `scripts/test_scispacy.py`

Before writing any real code, we made sure SciSpacy actually runs on our
machine and on real trial text. This is a "smoke test" — it's not part
of the final system; it's just proof of life.

We:

1. Installed SciSpacy (`pip install scispacy`) and the model
   `en_core_sci_lg` (~530 MB).
2. Ran it on a sample sentence: *"Inclusion Criteria: Age 18 and older.
   Histologically confirmed non-small cell lung cancer. ECOG performance
   status 0-1..."*
3. Ran it on three real trials from `data/trials.db`.

What we learned: SciSpacy successfully finds entities like *"non-small
cell lung cancer"*, *"anti-PD-1 antibody"*, *"chemotherapy"*. But it
**only labels them as `ENTITY`** — it doesn't tell you whether something
is a disease, drug, or symptom. It also picks up a lot of boilerplate
like *"Subjects"*, *"study"*, *"days"*. Both facts shaped the rest of
the design.

**Why this matters as a student lesson:** Always run the real tool on
real data before you start designing your system around it. You'll learn
what's actually hard.

### Step 2 — Design the parser (twice)

**No file yet, but `/Users/tonyxu/.claude/plans/i-need-to-build-staged-pine.md`
records the design.**

I wrote a first plan that was too elaborate — five separate pull
requests, three new modules, a YAML config file, multiple Pydantic
enums. You wrote a counter-plan that was simpler — one file, one class,
typed buckets, a 20-trial demo at the end. Your plan was better for v1
and we used a hybrid.

**Why this matters as a student lesson:** Premature complexity is a
real cost. Always ask "what's the minimum viable version?" first. Add
ceremony only when the simple version actually proves insufficient.

The merged plan turned into the next four steps.

### Step 3 — Build the parser

**File:** `src/TrialMine/features/eligibility.py` (~430 lines)

This is the heart of the system. It takes one trial's eligibility text +
the structured `min_age` / `max_age` / `sex` columns from the database
and returns an `EligibilityProfile` Pydantic model with everything
extracted.

Let me walk through each piece in the order it runs.

#### 3a. The output schema — `EligibilityProfile`

This is the "shape" of the answer. Think of it like a form with blanks
to fill in:

```python
class EligibilityProfile(BaseModel):
    min_age_years: float | None      # e.g. 19.0; None if unknown
    max_age_years: float | None
    sex: "All" | "Male" | "Female" | None
    required_conditions: list[str]               # e.g. ["non-small cell lung cancer"]
    excluded_conditions: list[str]
    required_prior_treatments: list[str]         # e.g. ["chemotherapy"]
    excluded_prior_treatments: list[str]
    raw_inclusion: str                            # the original text we parsed
    raw_exclusion: str
    parse_confidence: float                       # 0.0–1.0, our best guess
    section_source: "headers" | "single_header" | "variant" | "fallback"
```

A few decisions baked into this schema:

- **Float ages, not int.** Pediatric trials say things like *"6 months
  to 18 years"*. With ints, "6 months" rounds to 0 and you lose the
  lower bound. Floats let us store 0.5.
- **Four typed buckets, not one big list.** Saying "the patient is in
  exclusion bucket X" is different from "patient has condition X". We
  separate inclusion vs exclusion, and conditions vs treatments, so the
  matcher (next session's job) can reason cleanly.
- **`raw_inclusion` and `raw_exclusion` are kept.** Useful for
  debugging and for showing the patient the actual trial wording later.
- **`parse_confidence` is heuristic.** We document this — it's NOT a
  real probability. It's a number we made up by averaging three
  sub-scores, useful for filtering but not for math.

#### 3b. Splitting inclusion vs exclusion — `_split_sections()`

Trials use roughly four formats. We try them in priority order:

| Tier | What we look for | Example | Confidence |
|---|---|---|---|
| 1 | Both `Inclusion Criteria:` AND `Exclusion Criteria:` | the most common | 1.0 |
| 2 | Only one of those headers | only `Inclusion Criteria:` | 0.8 |
| 3 | Variant headers | `DISEASE CHARACTERISTICS:`, `Key Inclusion`, `Patients must have` | 0.6 |
| 4 | Nothing recognizable | just bullet points | 0.3 |

When tier 1 matches, we slice the text on the header positions: text
between the headers is the inclusion section, text after `Exclusion
Criteria:` is the exclusion section. When tier 4 matches, we put
everything in inclusion and leave exclusion empty (because we genuinely
don't know).

**Why a tier system?** Because real-world data is messy. A binary
"either you parsed it or you didn't" loses calibration. With tiers,
downstream consumers can decide *"I'll only trust trials with confidence
≥ 0.6"* and proceed safely.

#### 3c. Extracting age — `_extract_age()`

This step uses a simple but powerful idea called **column-first**.

The database already has a `min_age` column populated for 93.6% of
trials (with values like `"18 Years"`, `"6 Months"`). When the column
has a value, we **trust it** and skip the regex entirely. Only when the
column is empty do we fall back to scanning the text.

So the order is:

```
1. min_age column has value?       -> use it,                       confidence 1.0
2. text says "between 21 and 75"?  -> use both,                     confidence 0.9
3. text says "Age >= 18 years"?    -> use min,                      confidence 0.9
4. text says "Adults"?             -> guess min=18,                 confidence 0.6
5. nothing matched?                -> None, None,                   confidence 0.0
```

We hand-wrote 12 regex patterns that cover the most common phrasings,
including weirdo cases like:

- `>= 18 years`
- `≥ 18 years` (Unicode symbol)
- `\>= 18 years` (an actual escape leak in the database)
- `between 21 and 75`
- `18-75 years`
- `younger than 65`
- `Adults (18+)`
- `6 months to 18 years` (we convert months to fractional years)

**Why column-first?** Two reasons. (1) The column is more reliable than
parsing prose — somebody at ClinicalTrials.gov already extracted it
carefully. (2) It's faster — no regex pass needed when we have the
answer.

#### 3d. Extracting sex — `_extract_sex()`

Same column-first pattern. The `sex` column is populated 99.9% of the
time with `"ALL"`, `"MALE"`, or `"FEMALE"`. We just normalize the case
to "All" / "Male" / "Female". Text fallback is rarely needed.

#### 3e. Extracting entities — `_extract_entities()`

This is the most subtle piece. It uses SciSpacy + filtering + bucketing.

**Step 1: Run SciSpacy NER on the section text.**

```python
doc = nlp(inclusion_text)
for ent in doc.ents:
    # 'ent' is a span like "non-small cell lung cancer"
```

SciSpacy returns *every* span it thinks looks medically interesting,
including a lot of junk: `"Subjects"`, `"study"`, `"informed consent"`,
`"days"`, `"baseline"`, etc. So we filter.

**Step 2: Filter the noise.**

Each span must pass four checks:

```python
1. length between 2 and 60 characters
2. lowercase text not in our STOP_SPANS list (about 80 boilerplate words)
3. not all digits or punctuation
4. not all stopwords ("the", "of", "and", etc.)
```

If a span fails any check, drop it. The `STOP_SPANS` list is just a
hand-curated set we built by looking at real noise.

**Step 3: Type the survivors.**

We need to know if a span is a *condition* (disease) or a *treatment*
(drug, therapy, surgery). Here's the catch: `en_core_sci_lg` doesn't
tell us. So we use a regex heuristic:

```python
TREATMENT_RE = (therap|chemo|radiat|surger|transplant|inhibit|antibod|
                vaccin|immunother|monoclonal|cytotox|adjuvant|neoadjuvant|
                -mab\b|-nib\b|-inib\b|-zumab\b)
```

If the span text matches any of those patterns (e.g. *"chemotherapy"*,
*"bevacizumab"*, *"targeted therapy"*), it's a treatment. Otherwise it's
a condition.

Is this perfect? **No.** We measured ~70% precision on real data.
Things like `AST` (a lab value, not a condition) and `OR` (a stray word)
still leak through. We documented the limit. Future work is to use UMLS
to type entities properly.

**Step 4: Place each entity into the right bucket.**

If the section was inclusion, conditions go into `required_conditions`
and treatments into `required_prior_treatments`. If exclusion, they go
into `excluded_conditions` and `excluded_prior_treatments`.

#### 3f. Computing confidence — `parse_confidence`

```python
parse_confidence = (section_confidence + age_confidence + sex_confidence) / 3.0
```

Three sub-confidences, equally weighted, averaged. We document this is
not calibrated — it's just a useful filter signal. A trial with
confidence 0.97 is more trustworthy than one with 0.50, but the absolute
number isn't a probability.

### Step 4 — Write tests

**File:** `tests/features/test_eligibility.py` (37 tests)

Tests are how we know the parser actually works. We wrote two flavors:

**Fast tests (32 of them).** These don't load SciSpacy. They test the
regex pieces — age extraction, section split, sex extraction, helper
functions. They run in 50 milliseconds.

**Slow tests (5 of them).** These load SciSpacy once (3 seconds) and
test the full parser on real trial text from the database. They're
marked `@pytest.mark.slow` so you can skip them when iterating quickly.

Examples of what we tested:

```python
("Age >= 18 years",            (18.0, None))
("18 years and older",          (18.0, None))
("at least 18 years of age",    (18.0, None))
("between 21 and 75 years",     (21.0, 75.0))
("18-75 years",                 (18.0, 75.0))
("younger than 65",             (None, 65.0))
("≥ 18 years",                  (18.0, None))
("Age \\>= 18 years",           (18.0, None))   # the escape-leak case
("18 to 75 years of age",       (18.0, 75.0))
("Adults (18+)",                (18.0, None))
("6 months to 18 years",        (0.5, 18.0))
```

Plus tests for None input, empty string, all four section-split tiers,
column-vs-text precedence for age and sex, the stop-list filter, and
treatment-vs-condition typing on real text.

When we ran them: **37/37 passed first try.** No fixes needed.

**Why this matters as a student lesson:** If you can write the test
first (or while you write the code), you ship faster. Fast feedback
catches bugs before they pile up.

### Step 5 — Build a 20-trial demo

**File:** `scripts/demo_eligibility_parser.py`

A demo script's only job is "let me see this thing work on real data so
I can spot-check it." We pulled 20 random trials (a mix of short, medium,
long) and printed a table:

```
nct_id      | min_age | max_age | sex    | cond | excl_c | treat | excl_t | src     | conf
NCT06073769 |   19    |  None   | All    |   5  |   2    |   1   |   1    | headers | 1.00
NCT05535569 |   19    |  None   | All    |  20  |  30    |   2   |   4    | headers | 1.00
...
```

We also added a `--show-buckets` flag that prints the actual contents of
each bucket, so we could verify *"yes, the conditions bucket really has
disease names, not boilerplate"*.

### Step 6 — Build the batch script

**File:** `scripts/parse_eligibility.py`

A demo on 20 trials is great. But we have 140,703 trials. We need a
production-style batch script that can:

- Load every trial from SQLite
- Parse each one
- Save results to a new SQLite table called `parsed_eligibility`
- Show progress with a bar
- Never crash, even if one row is bad
- Be **resumable** — if it crashes at row 80,000, restart and continue
  from there

Key engineering patterns we used:

- **`CREATE TABLE IF NOT EXISTS`** — safe to re-run, won't error if the
  table already exists.
- **`INSERT OR REPLACE`** — overwrites the row if we re-parse.
- **`--resume` flag** — fetches existing nct_ids from
  `parsed_eligibility` and skips them.
- **`commit_every = 200`** — we batch SQLite commits in groups of 200
  rows. Why? Single inserts are 100x slower; one big commit at the end
  loses everything if the process dies. 200 is the sweet spot.
- **`try / except` around every row** — one bad row never kills the
  job. We log the failure and keep going.
- **`tqdm` progress bar** — psychological comfort during long runs.

We tested it on 1,000 trials first. It ran in 43 seconds at 23
trials/sec, with 0 failures. That's the moment we knew it was safe to
run on the full corpus.

### Step 7 — The concept normalizer

**File:** `src/TrialMine/features/concepts.py` (~150 lines)

This is a separate piece of code, not part of the parser. It solves a
different problem: **patients use different words than trials do.**

A patient might write *"my mom has stomach cancer that came back"*. A
trial says *"histologically confirmed gastric adenocarcinoma"*. BM25
search (which compares words literally) won't connect *stomach* to
*gastric*. So we built a small dictionary that maps lay terms to medical
terms and use it to expand the patient's query before searching.

```python
class ConceptNormalizer:
    synonyms = {
        "stomach cancer":         "gastric neoplasm",
        "liver cancer":           "hepatic neoplasm",     # NOT "hepatocellular carcinoma" — too narrow
        "kidney cancer":          "renal neoplasm",
        "colon cancer":           "colorectal neoplasm",  # FIX: was "coloctal" (typo)
        "cancer that came back":  "recurrent neoplasm",
        "spread to bone":         "bone metastasis",
        "stage 4":                "stage IV",
        "chemo":                  "chemotherapy",
        "immuno":                 "immunotherapy",        # NOT chained to "immune checkpoint therapy"
        # ... 36 more entries, 45 total
    }
```

The class has two methods:

- **`normalize(term)`** — given a single phrase, return the medical form
  if known, otherwise return the original.
- **`expand_query(query)`** — given a full sentence, return both the
  original AND a version with substitutions applied. We send both
  through search and combine the results.

#### The bugs you proposed and we caught

You showed me a draft of the synonym dict that had real problems:

| Bug | What was wrong | What we changed it to |
|---|---|---|
| `"colon cancer": "coloctal neoplasm"` | typo — "coloctal" isn't a word | `"colorectal neoplasm"` |
| `"liver cancer": "hepatocellular carcinoma"` | only one type of liver cancer; loses recall on cholangiocarcinoma trials | `"hepatic neoplasm"` (broader) |
| `"kidney cancer": "renal cell carcinoma"` | same problem; loses recall on Wilms tumor trials | `"renal neoplasm"` (broader) |
| `"immunotherapy": "immune checkpoint therapy"` | chained mapping — collapses immunotherapy to a narrower subtype, losing CAR-T, vaccines, etc. | dropped this line entirely |
| `"tumor": "neoplasm"` | replaces a word trials use literally; hurts BM25 | dropped |
| `"advanced": "advanced stage"` | same problem | dropped |

This is the most important lesson from the session: **the value of
review.** A 30-second glance caught five bugs that would have silently
degraded retrieval quality. Always have a second pair of eyes (or one
honest one).

### Step 8 — Run the full 140K parse

We kicked off `python scripts/parse_eligibility.py --resume` in the
background. It took **98.4 minutes** (about an hour and a half) and
processed all 139,703 remaining trials at 23.7 trials per second with
**0 failures**.

Final stats:

| Metric | Value |
|---|---|
| total parsed | 140,703 |
| with `min_age_years` | 132,729 (94.3%) |
| with `max_age_years` | 53,610 (38.1%) |
| with `required_conditions` | 140,123 (99.6%) |
| with `required_prior_treatments` | 78,586 (55.9%) |
| avg `parse_confidence` | 0.977 |
| avg conditions per trial | 30.7 |
| avg excluded conditions per trial | 33.4 |
| avg treatments per trial | 1.9 |
| section_source = `headers` | 92.3% |

The 1,000-trial demo had predicted these numbers within 1%, which means
the demo was a faithful sample.

### Step 9 — Update the project memory

**File:** `CLAUDE.md`

We updated the project's main reference doc to reflect the new state:

- Bumped phase to 6c (eligibility parsing)
- Added a "What's working" entry for the parser and the concept normalizer
- Added a "Design decisions (Week 5)" section with the three big calls
  we made (Decisions 17, 18, 19 — see Part 4 below)
- Updated "What's next" to mention the matcher

We also saved the design decisions to claude memory at
`/Users/tonyxu/.claude/projects/...` so future sessions remember them.

### Step 10 — Commit and push to GitHub

We made two commits and pushed them to `origin/main`:

```
2ef5c65 feat: eligibility parser — SciSpacy + regex hybrid
        - src/TrialMine/features/eligibility.py
        - tests/features/test_eligibility.py
        - scripts/parse_eligibility.py
        - scripts/demo_eligibility_parser.py
        - scripts/test_scispacy.py

8583803 feat: concept normalizer + Week 5 design decisions
        - src/TrialMine/features/concepts.py
        - CLAUDE.md
```

We deliberately did NOT commit `docs/05_eligibility.md` (this file).

---

## Part 4 — Why we made the choices we did

Three big decisions shaped the whole week. Logged in
`docs/design-decisions.md` and project memory.

### Decision 17 — SciSpacy + regex, NOT custom NER

**Custom NER** means training your own deep-learning model to recognize
medical entities. Better than SciSpacy in theory, but it requires
hand-labelling 2,000+ trial sentences (~80–120 hours of medical
annotator time).

**SciSpacy + regex hybrid** uses an off-the-shelf pretrained model plus
hand-written patterns. Quality is around 70%. Time to ship: a few days.

We chose the hybrid because:

- The downstream consumer (a ranker feature, an explainer) can tolerate
  70% precision.
- We don't have the annotated data.
- Annotator hours are the most expensive resource we don't have.

If we were extracting drug *dosages* for a clinical decision support
system, we'd invest in custom NER without hesitation. Different
precision requirements, different choice.

### Decision 18 — Hand-built synonym dict, NOT UMLS yet

**UMLS** is the canonical medical dictionary. SciSpacy ships with an
`EntityLinker` that maps text spans to UMLS Concept Unique Identifiers
(CUIs). High quality, no manual maintenance.

**Hand-built dict** is what we did: 45 entries, hand-written.

We chose the dict because:

- 45 entries cover the common patient phrasings we care about right now.
- It's debuggable — a doctor could read the whole list in a minute.
- UMLS adds ~1 GB of data and complexity.

Three rules every entry obeys (these caught all of your original bugs):

1. **Always broaden, never narrow.** liver cancer → hepatic neoplasm,
   NOT hepatocellular carcinoma.
2. **No chained mappings.** immuno → immunotherapy, full stop.
3. **Don't replace terms trials use literally.** tumor stays as tumor.

The plan: when the dict approaches 50 entries or maintenance becomes a
burden, switch to UMLS.

### Decision 19 — Three categories (Met / Unmet / Unknown), NOT binary

This is for the *matcher* we haven't built yet. When we eventually
decide *"is this patient eligible for this trial?"* we will emit one of
three states per criterion, not two.

Why? Because binary on partial information is **confidently wrong**:

- Trial says: *"Age 18-65, ECOG 0-1, no prior chemo."*
- Patient told us: *"I'm 60."*
- Naive binary system: *"Eligible!"* — but we don't know the patient's
  ECOG or chemo history.
- Three-category: *"Met (age) / Unknown (ECOG) / Unknown (chemo)"* —
  honest, and prompts the patient for the missing info.

Two flavors of Unknown matter:

- **Parser-Unknown.** Our parser couldn't extract that criterion (low
  parse_confidence). Fix in the parser later.
- **Patient-Unknown.** The patient never told us. Prompt for it in the
  UI.

Trial-level rollup rule: any hard *Unmet* → trial Unmet; all *Met* →
Met; otherwise Unknown.

---

## Part 5 — File map (where each thing lives)

If you forget anything from this doc, just open the files in this
order:

| File | Purpose |
|---|---|
| `src/TrialMine/features/eligibility.py` | The parser. Read top to bottom. ~430 lines. |
| `tests/features/test_eligibility.py` | All 37 tests. Reading them is the second-best way to understand the parser. |
| `scripts/test_scispacy.py` | Sanity check for the SciSpacy install. |
| `scripts/demo_eligibility_parser.py` | 20-trial table demo with `--show-buckets`. |
| `scripts/parse_eligibility.py` | Batch script that produced the 140K rows in `parsed_eligibility`. |
| `src/TrialMine/features/concepts.py` | Concept normalizer (synonym dict + UMLS skeletons). |
| `data/trials.db` | SQLite. Contains both `trials` and `parsed_eligibility` tables. |
| `CLAUDE.md` | Project memory; current state, design decisions, what's next. |

The parsed data itself is in `data/trials.db`, table `parsed_eligibility`.
That table is **gitignored** — your machine has it; the GitHub repo
doesn't.

---

## Part 6 — Interview prep

What an MLE interviewer might actually ask you about this work.

### 6.1 The big design question

> *"Walk me through how you'd build an information-extraction system
> for clinical trial eligibility text."*

A good answer covers, in order:

1. **Clarify scope.** What's the downstream consumer? Latency budget?
   Precision requirement?
2. **Sketch the pipeline.** Section split → structured slots → free-form
   entities → schema. State the column-first principle for any structured
   field that already has a column.
3. **Choose tools per slot.** Closed-vocab → regex; open-vocab →
   pretrained NER (SciSpacy / BioBERT-NER) before custom training.
   Justify with annotator-hours math.
4. **Handle quality.** Stop-list, length filters, dedupe, confidence
   scoring. Be honest if the confidence is heuristic.
5. **Plan for scale.** Batch script, idempotent, resumable, never
   crashes on bad rows.
6. **Future work.** UMLS linking, custom NER once you have labels,
   active learning for the long tail.

### 6.2 Specific NLP / system questions

**"Why not train a custom NER model?"**
Annotator cost (~80–120 hours) vs quality gain (~+15% precision). Not
worth it when 70% is acceptable for the downstream feature.

**"Why is `en_core_sci_lg` enough? Why not BioBERT?"**
en_core_sci_lg runs on CPU at 23 trials/sec; BioBERT requires GPU and
is 5x slower. For an offline batch parse, en_core_sci_lg is on the
Pareto frontier.

**"How do you handle the long tail of weird headers?"**
Tier the regex. Catch the common 80% at high confidence; tier 2 with
variants at medium; tier 3 fallback at low. Downstream consumes
confidence and decides what to trust.

**"Why parse the column instead of regex'ing the text every time?"**
Column-first principle. The structured `min_age` column is more
authoritative than parsing prose. Saves compute and is more accurate.

**"Your synonym dict has bugs. How do you prevent regressions?"**
Three rules: always broaden, no chains, no verbatim collisions. Plus a
fixture-based unit test that pins each entry's expected output. Plus
the upgrade path to UMLS so we don't grow the dict past ~50 entries.

**"How would you make `parse_confidence` actually calibrated?"**
Label ~200 trials with binary correctness. Fit a logistic regression of
correctness on the three sub-confidences. Validate with a reliability
diagram. Then ship the calibrated output instead of the heuristic mean.

### 6.3 Code-review questions (try answering these without peeking)

1. **Why does `parse()` never raise on `None` or empty input?**
   Because we batch-parse 140K rows; a crash on one bad row would kill
   the whole job. The CLAUDE.md says: never let the API return 500.
2. **Why is the SciSpacy model loaded lazily?**
   It's 700 MB. Loading at import time would block any process that
   imports `eligibility.py`, including unit tests that don't need it.
3. **Why is `commit_every = 200` rather than 1 or `len(rows)`?**
   Single commits → 100x slower. One commit at end → lose everything on
   crash at 99%. 200 is the balance.
4. **Why dedupe within a section but not across sections?**
   A condition that appears in both inclusion and exclusion is
   meaningfully present in both buckets — deduping across would lose
   the signal.
5. **Why are `raw_inclusion` and `raw_exclusion` kept on the schema?**
   Debugging (re-read what the parser saw) and explanations (show the
   patient the actual trial wording).

### 6.4 Tradeoff questions

**"Should the parser cache its output?"**
Yes — JSONL or a SQLite table keyed by `(nct_id, parser_version)`. Re-running
SciSpacy on 140K trials is too expensive per-request.

**"How do you handle a new trial?"**
Idempotent re-parse with `--resume` (skip nct_ids already at the current
parser_version). When we ship a new parser, bump `parser_version`; that
triggers re-parse on next run.

**"Should `parse_confidence` be a ranker feature?"**
Maybe. Add to FEATURE_NAMES, retrain LightGBM, measure NDCG@5 delta on
the held-out eval. Don't claim it helps without numbers.

**"Will this scale to 1M trials?"**
At 23 trials/sec single-process: ~12 hours. Acceptable. If we needed
faster, use SciSpacy's `nlp.pipe(batch_size=64, n_process=N)` for
parallelism. Tested briefly; macOS pickling issues need a fallback.

---

## Part 7 — What's next (next session)

1. **Build the eligibility matcher.** The function `compute_eligibility_features(trial,
   patient_profile)` consumes our `EligibilityProfile` and emits
   per-criterion Met / Unmet / Unknown per Decision 19. First we need a
   `PatientProfile` Pydantic schema.
2. **Wire `expand_query` into BM25 retrieval.** Run both original and
   expanded queries through Elasticsearch, RRF-fuse the results.
   **Measure NDCG@5 delta on the held-out eval before declaring this
   helps.** Don't just assume it does.
3. **Calibrate `parse_confidence`** if any downstream feature needs a
   real probability.
4. **Replace synonym dict with UMLS** once the dict approaches 50
   entries or patient queries we can't normalize become recurring bugs.

That's the full week. If you can explain Parts 3 and 4 in your own
words, you've internalized the work.
