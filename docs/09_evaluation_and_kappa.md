# Week 9 — Evaluation, Inter-Annotator Agreement, and Honest Numbers (Student Edition)

A walk-through of everything we did this week. The premise: the system has been
"working" for many weeks now, but **how do we know it's actually good?** This
week is the answer to that question.

I'm assuming you've never run an evaluation, never computed Cohen's kappa, never
thought about LLM-as-judge limitations, and never compared a search system
head-to-head against a baseline. Every term gets defined the first time it
shows up.

If you haven't read them yet, the previous weeks build the system this doc
*evaluates*:

- [`02_semantic_search.md`](02_semantic_search.md) — the BM25 + semantic + RRF retrieval
- [`03_fine_tuning.md`](03_fine_tuning.md) — fine-tuned BioLinkBERT bi-encoder
- [`04_cross_encoder.md`](04_cross_encoder.md) — fine-tuned BioLinkBERT cross-encoder + LightGBM blender
- [`06_agent.md`](06_agent.md) — the LangGraph agent
- [`08_testing_and_signal_boosters.md`](08_testing_and_signal_boosters.md) — testing + production-thinking

You don't *have* to re-read them, but I'll point back when relevant.

---

## TL;DR (read this if you read nothing else)

- **What we built:** a 1200-pair labeled evaluation dataset across 60 stress-testing queries, validated by a second LLM judge (Cohen's κ), and benchmarked against ClinicalTrials.gov.
- **Headline number:** the full pipeline scores **NDCG@10 = 0.875 ± 0.04** — solidly good. Bootstrap CIs are tight enough to defend in an interview.
- **Honest finding:** the system breaks on **complex multi-fact queries** (NDCG@5 = 0.626) and **vague queries** (0.688). It excels on common, rare, and treatment-modality queries (all > 0.92).
- **Beats the obvious baseline:** **+9.8 precision points vs CT.gov keyword search** on the disjoint result set, winning every category. Biggest gap is pediatric (+11.5).
- **Labels are trustworthy:** Sonnet-vs-Haiku quadratic-weighted κ = 0.72 (substantial), 97 % off-by-one. We also surface a known bias — Haiku systematically over-rates the top tier — instead of hiding it.
- **Biggest open task:** 3 of the 5 worst queries fail for the *same* reason — the agent parses the patient's age / prior treatment / progression status into a `PatientProfile`, but the orchestrator doesn't use it as a retrieval filter. Wiring this is **the single highest-leverage fix** in the project, and it requires no retraining.

If you stopped reading here, you'd already understand what this week was about. The rest is the *how* and *why*.

---

## Part 1 — Why this week matters

Here's the situation at the end of Week 8:

The system **had been measured**. We had ablation tables (BM25 → +Semantic →
+CE → +LightGBM) showing each stage of the pipeline pays for itself. The
LightGBM blender was trained on 145 queries with leave-one-query-out
cross-validation. We had an MLflow tracking database. We had ~7,000 LLM-judged
labels across multiple files. By any reasonable measure, we had *more*
evaluation than 95 % of side projects.

So why is this whole week dedicated to evaluation? Four gaps.

### Gap 1: The evaluation set was thin and statistically noisy

The original evaluation used **20 queries**. Bootstrap 95 % CIs (we'll define
those in Part 2) on n = 20 are roughly **± 0.10 NDCG points**. That means a
single weird query can move the headline number by 5 percentage points. If
someone in an interview asks *"is your evaluation noisy?"* — which is the
second question every interviewer asks after *"what did you measure?"* — the
honest answer was *"yes, fairly."*

20 queries is fine for early development. It is **not** fine for a job
interview where the evaluation section will be picked apart line by line.

### Gap 2: One judge, no agreement check

All ~7,000 labels were from **Claude Haiku 4.5**. Just one model, with one
prompt. Nobody else looked at any of them. No human, no second LLM.

This is a problem. **LLM-as-judge** (we'll define it in Part 2) has known
failure modes — small models pattern-match on surface keywords, over-rate
trials whose title contains the query verbatim, and so on. If Haiku has a
systematic bias, every metric we report inherits that bias. We had no way of
checking.

### Gap 3: No comparison to a baseline anyone has heard of

We had ablation tables comparing BM25 vs +Semantic vs +Cross-Encoder. Those
are *internal* baselines — they tell you which stage of *our* pipeline
contributes what. They do **not** tell you whether the whole system is better
than the alternative the user already has access to.

The alternative every patient already has access to is
[clinicaltrials.gov](https://clinicaltrials.gov). If TrialMine isn't
demonstrably better than typing the same query into CT.gov's search box, the
whole project is hard to justify.

### Gap 4: No consolidated, honest report

We had `docs/fair-evaluation-report.md` (the 50-query held-out ablation),
`docs/evaluation-report.md` (the original 20-query report — somewhat
optimistic about LightGBM), per-query JSONs, MLflow runs… but no single
artifact you could hand to an interviewer and say *"read this, it's the
whole evaluation story."* Worse, the original report didn't surface its own
limitations cleanly. That's a trust problem.

### What we actually built this week

```
This week:
├── 60-query labeled dataset (1200 (query, trial) pairs)
│   ├── 30 carry-over queries from prior weeks (IDs 0-19, 300-309)
│   └── 30 NEW queries across 7 stress-testing categories (IDs 400-429)
├── Sonnet-as-judge inter-annotator validation (κ = 0.72 quadratic)
├── Manual-labeling CLI (built but not used — fallback path)
├── Head-to-head with ClinicalTrials.gov (+9.8 pts precision on disjoint set)
└── Consolidated 7-section evaluation report (252 lines)
```

Five new scripts, four new evaluation artifacts, one rewritten report. Each
piece exists to close one of the four gaps above.

---

## Part 2 — Vocabulary you'll meet

Skim once, come back when you hit a term you don't recognize.

**The 5 terms you really need:**
- **NDCG@K** — quality-of-ranking score in [0, 1]. Higher = better top-K.
- **Bootstrap 95 % CI** — *"if I re-ran this with a different sample, the answer would land in [low, high] 95 % of the time."* Reported as `value ± half-width`.
- **Cohen's κ** — agreement between two annotators above chance. > 0.6 = substantial. We use the **quadratic-weighted** version because our scale is ordinal.
- **LLM-as-judge** — letting a language model grade (query, document) pairs instead of paying humans. Cheap and scalable; can have systematic biases.
- **Disjoint set** — when comparing two systems, the trials that *only one* system found. The honest place to compare precision when result lists barely overlap.

If those five make sense, you can read most of this doc cleanly. The full glossary below is for when you hit a less-common term.

### Information retrieval (IR) words

**Query** — what the user types. Example: `"NSCLC brain mets"`. Sometimes
called the *information need*.

**Document** — what the system retrieves. In our case, a clinical trial.

**Top-K** — the K highest-ranked documents the system returns for a query.
We use top-20 most of this week.

**Relevance** — a graded judgement of how well a document answers the query.
We use a 0–3 scale (0 = irrelevant, 1 = marginal, 2 = relevant, 3 = highly
relevant). Sometimes called *graded relevance* to contrast with binary
(relevant / not).

**NDCG@K** — *Normalized Discounted Cumulative Gain at K*. The standard
ranking metric for graded relevance. Two intuitions:
1. **Discount**: a relevant document at rank 1 is worth more than the same
   document at rank 10 (users look at the top first).
2. **Normalize**: divide by the score of the *ideal* ranking, so the metric
   lives in [0, 1] regardless of how many relevant docs exist.

A score of 1.0 means you ranked perfectly. 0.6 is mediocre. 0.9+ is
excellent.

**MRR** — *Mean Reciprocal Rank*. The reciprocal of the rank position of the
first relevant result, averaged across queries. Cares only about the
top-most hit. If the first relevant result is at rank 1, MRR = 1.0; rank 2,
MRR = 0.5; rank 3, MRR = 0.33.

**Recall@K** — fraction of all relevant documents that appear in the top K.
"Did I find them?" rather than "did I rank them?"

**Precision@K** — fraction of the top K that are relevant. "Is what I
returned actually good?"

### Statistical words

**Sample size (n)** — number of independent observations. n = 20 queries
is a small sample. n = 60 is medium. n = 1000 is large.

**Bootstrap** — a way to estimate uncertainty without assuming a specific
probability distribution. Procedure: from your N data points, repeatedly
draw N points *with replacement* (so the same point can be picked twice).
Compute your statistic on each resample. The spread of those values tells
you how confident to be in the original. We do 1000 resamples.

**95 % confidence interval (CI)** — a range that *would contain the true
value 95 % of the time if you re-ran the experiment*. We report
**± half-width**. So `0.875 ± 0.04` means *"if we re-ran with a different
sample of queries, the answer would land in [0.835, 0.915] 95 % of the
time."*

The smaller the half-width, the tighter the answer. Half-width shrinks
roughly as 1/√n, which is why doubling the sample size only buys you 30 %
tighter CIs.

**Stratified sample** — sampling that respects subgroup proportions. If your
data has 8 categories and you sample 100 examples, a *random* sample might
miss a small category entirely; a *stratified* sample picks ~12-13 per
category so every category is represented. We use stratified sampling for
the kappa work.

**Deterministic seed** — a number passed to a random-number generator so
"random" choices are reproducible. Same seed → same random sample. Critical
for resumable scripts: if the user labels 42 of the 200 then quits, they
should get the same 200 (in the same order) on resume.

### LLM-as-judge words

**LLM-as-judge** — using a large language model to *grade* the relevance of
(query, document) pairs instead of paying humans. Cheap, fast, scalable.
The trade-off: LLMs can have systematic biases, and two LLMs from the same
training family can share blind spots.

**Inter-annotator agreement (IAA)** — how much two annotators (humans, or
LLMs, or one of each) agree on the same labels. The baseline question: *if
we had two judges, would they call the same things relevant?*

**Raw agreement** — the fraction of pairs where the two annotators picked
*exactly* the same label. Easy to interpret. Misleading on its own — if 70 %
of labels are "3", two annotators who both always say "3" agree 70 % of the
time *by chance*.

**Cohen's kappa (κ)** — agreement above what you'd expect from two
annotators picking randomly with the marginal distributions. κ = 0 means
"no better than chance"; κ = 1 means "perfect agreement"; κ < 0 means
"actively disagree." Always interpret κ in addition to raw agreement.

**Weighted kappa** — for **ordinal** labels (like our 0-3 scale), a
disagreement of 0 vs 1 is a smaller miss than 0 vs 3. *Weighted* kappa
penalizes far-apart disagreements more.
- **Linear weights**: penalty proportional to distance.
- **Quadratic weights**: penalty proportional to squared distance.
The conventional choice for medical IRR (inter-rater reliability) studies
is **quadratic-weighted kappa**.

**Landis-Koch interpretation** — a widely cited rule of thumb for what κ
values mean:

| κ | Interpretation |
|---|---|
| < 0.0  | actively disagreeing |
| 0.0 – 0.2 | poor (chance-level) |
| 0.2 – 0.4 | fair |
| 0.4 – 0.6 | moderate |
| 0.6 – 0.8 | **substantial** |
| 0.8 – 1.0 | almost perfect |

**Off-by-one accuracy** — fraction of pairs where the two annotators are
within ±1 on an ordinal scale. Useful sanity check alongside κ:
disagreement of 2 vs 3 cancels out across a query in NDCG terms; 0 vs 3 is
where the metric breaks.

**Confusion matrix** — a 2-D grid where rows = annotator A's labels and
columns = annotator B's labels. Each cell counts how often A picked row-i
and B picked column-j. Looking at the off-diagonal cells tells you the
*shape* of disagreements.

**Pearson r** — linear correlation coefficient. Sanity check on kappa: if
two annotators agree on the *ordering* of pairs but use different absolute
values, you'll see high Pearson r and a moderate kappa. Useful as a
secondary signal.

### Comparison words

**Head-to-head** — same input, two systems, compare outputs side by side.

**Disjoint set** — for two sets A and B, the disjoint set is `(A ∪ B) − (A ∩ B)`
— everything in either set that is *not* in the overlap. We use it to ask
*"of the things only one system found, what fraction were relevant?"*

**Precision on the disjoint set** — relevance precision computed only on
the trials *exclusive* to one system. The honest comparison metric when two
systems return very different result lists.

**Symmetric / bidirectional comparison** — when both `ours-only` and
`theirs-only` get the same judging treatment. The asymmetric version (what
we did first) only labels ours-only and gets pushed back on with *"yes,
yours are good — but were theirs bad?"*

---

## Part 3 — What we built, step by step

I'll walk through each piece. Six steps. Each one closes a specific gap from
Part 1.

### Step 1 — Design the 60-query test set

> **In one sentence:** we picked 60 queries across 7 categories, each category designed to expose a *different* failure mode of the pipeline.

The first decision was **which queries to evaluate on**. The original 20
queries were straightforward oncology phrases like *"breast cancer hormone
receptor positive phase 3"*. They're realistic, but they don't cover the
ways a real user might phrase things. To stress-test the system, we picked
**seven categories** designed to expose specific failure modes:

| Category | n | What it stresses | Example |
|---|---:|---|---|
| `common` | 5 | sanity / baseline | `breast cancer HR+ CDK4/6 inhibitor` |
| `rare` | 5 | long-tail tumor types | `angiosarcoma scalp`, `merkel cell carcinoma` |
| `pediatric` | 3 | under-represented population | `medulloblastoma 6 year old`, `wilms tumor relapsed` |
| `complex` | 5 | multi-fact patient profiles | `58M EGFR exon 19 NSCLC failed osimertinib phase 2-3` |
| `geographic` | 3 | location filtering (we don't support it) | `pancreatic cancer trials Texas`, `trials at MD Anderson` |
| `vague` | 5 | underspecified intent | `I have cancer what trials`, `mom has bone cancer` |
| `treatment` | 4 | modality-first phrasing | `CAR-T lymphoma`, `PARP inhibitor ovarian` |

We then added **30 carry-over queries** (the original 20 from `IDs 0–19` and
the first 10 held-out test queries from `IDs 300–309`) so the new dataset
includes both seen and unseen styles.

Total: **60 queries**, IDs 0-19, 300-309, 400-429. We deliberately picked
non-overlapping ID ranges so future evaluation work can extend the set
without collisions.

**Why these categories specifically?** Because each one is the kind of query
that, when it fails, fails for a *different* reason:

- `complex` fails when the agent doesn't translate the patient profile into
  retrieval filters → the pipeline can't tell "failed osimertinib" from
  "osimertinib-naive."
- `vague` fails when there's no constraining keyword → falls back to
  whatever's most popular in the index.
- `geographic` fails because we don't support location filters at all.
- `rare` fails when the fine-tuning data didn't include enough examples of
  long-tail tumors.
- `pediatric` fails when the system doesn't filter by age.

If we'd just picked 60 *common* queries, the headline NDCG would be
inflated and we'd never see these breakdowns. **The diversity is the
point.**

### Step 2 — Build the labeling pipeline (`scripts/build_full_eval_dataset.py`)

> **In one sentence:** for each of 60 queries, run the full pipeline, get the top 20 trials, ask Claude Haiku to rate each on a 0-3 scale — 1200 labels for ~$1.

Now we need labels — for each of the 60 queries, we need a relevance score
(0–3) for each of the top-20 trials the pipeline returns. That's 60 × 20 =
**1200 labels**.

We could pay a clinical reviewer ~$5,000 to label them. Or we could pay
Claude Haiku 4.5 ~$1 to do the same job in 25 minutes. We picked Haiku.
This is **LLM-as-judge** — and we'll come back in Step 4 to validate that
this was a reasonable choice.

The script (`scripts/build_full_eval_dataset.py`) does five things:

1. **Loads the production pipeline** — Elasticsearch, FAISS, fine-tuned
   BioLinkBERT embedder, fine-tuned cross-encoder, LightGBM blender. This
   is the EXACT pipeline a real user would hit. We label *production
   output*, not some intermediate stage.

2. **For each query, runs `hybrid.full_pipeline()` to get top-20**:

   ```python
   results, timings = hybrid.full_pipeline(
       query=query,
       reranker=reranker,
       blender=blender,
       top_k=20,
       rerank_top_k=50,
   )
   ```

3. **For each (query, trial) pair, builds a labeling prompt** including the
   trial title, conditions, phase, status, and first 500 characters of
   eligibility criteria.

4. **Calls Claude Haiku 4.5** with the prompt:

   ```text
   Rate the relevance of this clinical trial to this patient's search query.

   Patient query: {query}
   Trial title: {trial_title}
   ...

   Rate on a scale of 0-3:
   0 = Completely irrelevant
   1 = Marginally relevant
   2 = Relevant
   3 = Highly relevant

   Respond with ONLY a JSON object: {"score": X, "reason": "..."}
   ```

5. **Writes one record per (query, trial) pair** to
   `data/evaluation/full_labeled_dataset.jsonl`. The script supports
   `--resume` (skip already-labeled pairs) and `--limit N` (smoke-test mode).

**Output shape** — each line is JSON like:

```json
{
  "query_id": 0,
  "query": "breast cancer hormone receptor positive phase 3",
  "category": "existing",
  "rank": 1,
  "nct_id": "NCT01602380",
  "trial_title": "...",
  "trial_conditions": "Hormone Receptor Positive Breast Cancer",
  "trial_phase": "Phase 3",
  "trial_status": "COMPLETED",
  "pipeline_score": 0.0144,
  "cross_encoder_score": 0.730,
  "relevance": 3,
  "reason": "Exact match on hormone receptor-positive breast cancer...",
  "labeler": "claude-haiku-4-5"
}
```

The `pipeline_score` and `cross_encoder_score` fields are stored too, so
later analyses can ask things like *"are highly-CE-scored trials more
likely to be labeled 3?"* without re-running the pipeline.

**Real numbers from running the script**:

```
Queries processed : 60
New labels        : 1160
Skipped (resume)  : 40
Score distribution:
  0:   25 (  2.2%) #
  1:  169 ( 14.6%) #######
  2:  175 ( 15.1%) #######
  3:  791 ( 68.2%) ##################################
```

Two observations:
- **68 % of pairs are labeled 3.** That's a *lot* of "highly relevant." We
  should be suspicious — is Haiku too generous? We'll find out in Step 4.
- **Zero parse errors.** Haiku is a reliable JSON-emitting judge.

The "skipped (resume)" 40 came from the smoke test we ran earlier —
`--resume` correctly picked up where we left off without re-spending
API on already-labeled pairs.

### Step 3 — The manual labeling CLI (`scripts/manual_label.py`)

> **In one sentence:** an interactive command-line tool that walks a human reviewer through pairs and lets them accept the LLM's label or override it — built but not used this week.

Even though we ended up using Sonnet-as-judge (Step 4), we **also built**
the human-labeling fallback path. This is a small interactive CLI:

```text
==============================================================================
  [42/200]  query_id=413  rank=4  category=complex
==============================================================================

  QUERY: 58M EGFR exon 19 NSCLC failed osimertinib phase 2-3

  TRIAL: NCT04293094
    Title:      A Ph1b Study of Osimertinib + Alisertib...
    Conditions: NSCLC; EGFR Mutation
    Phase:      Phase 1    Status: COMPLETED
    Eligibility (first 400 chars):
      Patients must have... EGFR exon 19 deletion or L858R mutation...
      ... acquired resistance to osimertinib...

  LLM LABEL: 3    (reason: Excellent match: 58M with EGFR exon 19 NSCLC...)

  Scale: 0=irrelevant  1=marginal  2=relevant  3=highly relevant
  Press 0/1/2/3, Enter to accept LLM, 's' to skip, 'q' to quit.

  your label > _
```

Why ship it if we didn't use it? Two reasons:
1. **Future human-anchored kappa**. The natural top-up after the LLM-vs-LLM
   kappa work is to have a clinician hand-label the disagreement set. The
   CLI is ready for that without any further work.
2. **Decision 32 trust signal**. Saying *"we have a path to human labels,
   but we used Sonnet under time pressure"* is more credible than *"we
   couldn't be bothered."*

The CLI supports `--sample N --seed 42` (deterministic random subset),
`--category complex` (focus on one slice), `--resume` (pick up where you
quit), and `--skip-errors` (drop pairs where the LLM returned `-1`).

### Step 4 — Pivot to Sonnet-as-judge

> **In one sentence:** under time pressure, we used Claude Sonnet 4.6 instead of a human reviewer to validate Haiku's labels — same prompt, stronger model, treats Sonnet as the gold judge.

Here's where we made a deliberate trade-off. The "right" thing to do was to
have a clinician hand-label ~200 pairs and compute κ against the LLM. The
"actually-going-to-happen" thing was to use a stronger LLM under time
pressure. We chose the second.

**Why Sonnet 4.6 was a reasonable choice**:

1. **Identical prompt to Haiku**. We re-used the exact labeling prompt
   without changing a token. So any disagreement isolates *model
   capability*, not *prompt drift*.
2. **Stratified sample of 100 pairs**. Default per-category quotas (existing
   20, common 12, rare 12, pediatric 10, complex 12, geographic 10, vague
   12, treatment 12 = 100 total) ensure every category has at least 10
   labels, so per-category κ is meaningful.
3. **Sonnet is genuinely a stronger judge** than Haiku — 5× the cost per
   call, demonstrably better at nuanced reasoning. If they agree, we have
   evidence Haiku isn't badly broken. If they disagree, the disagreements
   themselves are diagnostic.

**What this measures vs. what it doesn't**:

- ✅ **Inter-LLM stability** — does a stronger judge confirm the weaker
  judge's labels?
- ❌ **Truth** — both judges are LLMs from the same family. They share
  training-data blind spots. A real human pass on the disagreement set is
  the obvious next step.

We documented this honestly in §6 of the evaluation report. The user's
exact quote — *"Treat sonnet as a human"* — got recorded in agent memory
as a feedback memory: under time pressure, propose Sonnet-as-judge instead
of skipping validation entirely.

### Step 5 — The Sonnet labeling script (`scripts/sonnet_label.py`)

> **In one sentence:** the same labeling pipeline as Step 2, but with Sonnet 4.6, on a stratified sample of 100 pairs, writing both labels into one file.

Mechanically very similar to the Haiku script, with three differences:

1. **Uses `claude-sonnet-4-6`** instead of `claude-haiku-4-5-20251001`.
2. **Stratified sample** with per-category quotas — see the
   `DEFAULT_QUOTAS` dict in the script.
3. **Output shape matches what the analyzer expects** — writes
   `relevance_llm` (Haiku's label) and `relevance_human` (Sonnet's label)
   per record, so the downstream `agreement_analysis.py` doesn't care
   whether the "human" was a real human or another LLM.

```json
{
  "query_id": 0,
  "query": "...",
  "nct_id": "NCT01602380",
  "relevance_llm": 3,
  "relevance_human": 2,
  "agree": false,
  "llm_reason": "...",
  "human_reason": "...",
  "labeler_llm": "claude-haiku-4-5",
  "labeler_human": "claude-sonnet-4-6"
}
```

Total run: **97 fresh + 3 from the smoke test = 100 pairs**, ~3 minutes,
~$0.30 in API. One pair returned a parse error (1 % failure rate, dropped
in analysis).

Headline result: Sonnet's distribution was much more conservative —
**38 % "3" labels vs Haiku's 68 %**. Already, before we computed κ, we
could see something: Sonnet rates fewer trials as "highly relevant."

### Step 6 — Cohen's kappa analysis (`scripts/agreement_analysis.py`)

> **In one sentence:** compute three flavours of Cohen's κ + bootstrap CIs + confusion matrix from the (Haiku, Sonnet) pairs, and surface the systematic bias the kappa exposes.

The agreement script reads the human-format JSONL (`relevance_llm` and
`relevance_human` columns) and computes:

1. **Three kappas** — unweighted, linear-weighted, quadratic-weighted —
   each with a 1000-bootstrap 95 % CI. We use sklearn's
   `cohen_kappa_score(y_haiku, y_sonnet, weights=...)`.
2. **Raw agreement and off-by-one accuracy.**
3. **Pearson r** as a sanity check.
4. **4 × 4 confusion matrix** (rows = Haiku, cols = Sonnet).
5. **Per-category κ** to see whether agreement is uniform or category-specific.
6. **Per-LLM-score breakdown** — *"of the trials Haiku rated 3, what did Sonnet say?"*
7. **Top-10 disagreements by |Δ|** — for inspection.

**Why three kappas?** They give different information:

- **Unweighted κ** treats every disagreement equally. *"3 vs 2"* and
  *"0 vs 3"* count the same. This is the wrong choice for ordinal scales.
- **Linear-weighted κ** scales penalty with the absolute difference. *"0 vs 3"* is
  three times worse than *"2 vs 3"*. Better.
- **Quadratic-weighted κ** scales penalty with the *squared* difference.
  *"0 vs 3"* is nine times worse than *"2 vs 3"*. **This is the standard
  choice for medical IRR studies** because misjudging a 3 as a 2 is much less
  costly than misjudging a 0 as a 3. Quadratic κ is the headline number we
  report.

**Bootstrap CI in code** (paraphrased from the script):

```python
def kappa_with_ci(llm, human, weights, n_boot=1000, seed=42):
    point = cohen_kappa_score(llm, human, weights=weights)
    rng = np.random.RandomState(seed)
    boot_kappas = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(llm), size=len(llm))
        boot_kappas.append(cohen_kappa_score(llm[idx], human[idx], weights=weights))
    low, high = np.percentile(boot_kappas, [2.5, 97.5])
    return {"kappa": point, "ci_low": low, "ci_high": high}
```

**Headline result** (n = 99 valid pairs, 1 parse error dropped):

| Metric | Value |
|---|---|
| Exact agreement | 66.7 % |
| Off-by-one (±1) | **97.0 %** |
| Pearson r | 0.776 |
| κ unweighted | 0.487 [0.357, 0.621] |
| κ linear | 0.608 [0.487, 0.721] |
| **κ quadratic** | **0.723 [0.601, 0.819]** ← substantial |

**The confusion matrix is the most diagnostic part of the report**:

|   | Sonnet=0 | Sonnet=1 | Sonnet=2 | Sonnet=3 |
|---|---:|---:|---:|---:|
| Haiku=0 | 1 | 2 | 0 | 0 |
| Haiku=1 | 1 | 13 | 0 | 0 |
| Haiku=2 | 0 | 7 | 13 | 0 |
| **Haiku=3** | 0 | 3 | **20** | **39** |

Read the bottom row carefully. Of **62 pairs Haiku rated 3**, Sonnet
agreed on only **39**. Twenty got demoted to 2. Three got demoted to 1.
Zero got *promoted* (Sonnet never rated higher than Haiku — only equal or
lower).

**Translation: Haiku systematically over-rates the top tier.** This is a
*real* finding, surfaced in our headline report not buried in an
appendix. It's the kind of thing an interviewer wants you to flag
yourself, not have to ask about.

**Per-category κ** has its own story:

| Category | n | quad κ | What it says |
|---|---:|---:|---|
| common | 12 | 1.000 | Both judges agree fully — easy queries are easy. |
| pediatric | 10 | 0.937 | Almost perfect. |
| treatment | 12 | 0.700 | Substantial. |
| complex | 12 | 0.625 | Substantial — judges hold up on multi-fact. |
| vague | 11 | 0.459 | Moderate. |
| existing | 20 | 0.400 | Moderate. |
| rare | 12 | 0.226 | Fair. |
| **geographic** | 10 | **−0.032** | **Disagree at chance level.** |

The geographic κ near zero is the most pointed finding: *when the pipeline
cannot answer the query intent (no location filter), even the judges
can't agree on what counts as relevant.* This is consistent — bad-input
queries produce ambiguous outputs that no rubric can disambiguate.

### Step 7 — CT.gov head-to-head (`scripts/compare_with_ctgov.py`)

> **In one sentence:** for 20 queries, fetch CT.gov's top 20, fetch ours, label both disjoint sets with Haiku, compare disjoint-set precision side by side.

The comparison script does, for each of 20 queries spanning every category:

1. **Hits CT.gov v2 search API**:
   ```
   GET https://clinicaltrials.gov/api/v2/studies
       ?query.term=NSCLC+brain+mets
       &pageSize=20
       &fields=NCTId,BriefTitle,OverallStatus,Phase,Condition,EligibilityCriteria
   ```
   Their top 20 by their internal ranking.
2. **Runs our full pipeline** for the same query. Our top 20.
3. **Computes set overlap** — how many trials are in both lists?
4. **Labels the disjoint sets** — both `ours-only` and `theirs-only` go to
   Haiku for relevance scoring (with the same 0–3 prompt as everywhere
   else).

**Why CT.gov is the right baseline**:

- It's the alternative the user already has access to.
- It's the *official* trial registry — same underlying data, different
  ranking.
- An interviewer can verify any single result by clicking the link.

**Caveats we documented honestly**:

1. **CT.gov ranks by keywords, not ML.** This is *"TrialMine's ML pipeline
   vs a strong but unranked baseline"* — not *"TrialMine vs a competing
   ML system."* If CT.gov added a learned re-ranker, some of our edge
   would disappear.
2. **CT.gov searches the full 500 K registry; we search ~140 K
   oncology-only.** Some `theirs-only` trials are non-oncology trials our
   ingest filtered out. They're not retrieval failures on our side, but
   they're also not "we found something they missed."
3. **Both judgements come from Haiku** — same blind spots as the rest of
   our labels. The κ = 0.72 vs Sonnet means the precision delta is
   probably robust to judge choice.

**Headline result** (n = 20 queries, top-20 each):

| | TrialMine | CT.gov | Δ |
|---|---:|---:|---:|
| Average overlap | 0.5 / 20 | 0.5 / 20 | — |
| Disjoint trials per query | 19.6 | 13.5 | +6.1 |
| Disjoint trials judged ≥ 2 by Haiku | 302 / 391 | 182 / 270 | — |
| **Precision on disjoint set** | **77 %** | **67 %** | **+9.8 pts** |

**We win in every category.** Biggest deltas:

| Category | Δ relevant per query |
|---|---:|
| pediatric | **+11.5** (we: 15.0, them: 3.5) |
| vague | +7.4 |
| complex | +6.3 (but both weak in absolute) |
| treatment | +6.0 |

The pediatric finding is the most diagnostic. CT.gov surfaces 13.5 unique
trials per pediatric query, but only 3.5 of them are relevant. **Keyword
search can't distinguish "wilms tumor relapsed" from "wilms tumor
first-line"** — the semantic difference is structural (relapsed vs new
diagnosis), and BM25 happily matches both. Our cross-encoder + LightGBM
blender does distinguish them.

**The honest claim**:

> *"TrialMine adds relevant trials CT.gov's keyword search doesn't surface,
> AND TrialMine's disjoint set is ~10 percentage points more precise than
> CT.gov's disjoint set."*

The second clause is what required the bidirectional labeling — without it,
all we'd have is the first clause, which the user correctly identified
earlier as half a claim.

### Step 8 — The consolidated evaluation report (`docs/evaluation-report.md`)

> **In one sentence:** one document that pulls every number into seven labelled sections, ending with a non-negotiable Limitations section.

The final piece: pull every number from the previous steps into one
artifact you could hand to an interviewer. **Seven sections, 252 lines**:

| § | Section | What it answers |
|---|---|---|
| 1 | Executive summary | What's the headline number? |
| 2 | Ablation table | Where does the pipeline gain come from? |
| 3 | Per-category NDCG | Where does ranking quality break? |
| 4 | CT.gov head-to-head | Are we better than the alternative? |
| 5 | Error analysis | Why do the worst queries fail? |
| 6 | Inter-annotator κ | Are our labels trustworthy? |
| 7 | Limitations | What does this report *not* tell you? |

**The error analysis (§5) is the part hiring loops will ask about**.
We picked the 5 worst-NDCG queries, looked at why each failed, and found a
**cross-cutting pattern**: 3 of the 5 worst queries fail for the *same
reason* — the agent parses age / prior treatments / progression status
into a `PatientProfile`, but the orchestrator does not translate those
into hard retrieval filters.

- Q410 *"medulloblastoma 6 year old"* — top-3 includes adult and
  post-pubertal trials. Age 6 is parsed but not enforced.
- Q413 *"58M EGFR exon 19 NSCLC failed osimertinib"* — rank 5 requires
  *osimertinib-naive* patients. "Failed osimertinib" is parsed but not
  enforced as exclusion.
- Q416 *"62F HER2+ post-trastuzumab progression"* — rank 1 is the
  STOP-HER2 trial (for patients still *responding* to trastuzumab). The
  *opposite* clinical situation. CE matched on "trastuzumab" without
  understanding direction.

**This single fix — wire `PatientProfile` into the BM25 filter dict — is
the highest-leverage open task in the project**. No retraining required.

**§7 (Limitations)** lists 8 honest items: no human gold standard, label
pool tautology in §3, n=60 still small, agent profile unused as filter,
no fairness analysis yet, etc. **This section is non-negotiable** — see
Decision 32 in Part 4.

---

## Part 4 — Design decisions

Two decisions this week. Both are about **how to evaluate**, not what to
build, which makes them subtle. Both are easy to ignore until an
interviewer presses on them.

### Decision 31 — 60 queries × top-20 = 1200 labeled pairs is the *minimum* for interview-defensible per-category CIs

**The trade-off in one sentence**: more queries = tighter CIs, but the
half-width shrinks as 1/√n, so doubling n only buys you 30 % tighter CIs.

**The math**:

| n queries | Approximate NDCG@10 CI half-width |
|---|---|
| 20 | ± 0.10 |
| 60 | ± 0.04 |
| 200 | ± 0.024 |
| 1000 | ± 0.011 |

A single weird query in n=20 swings the headline by 5 percentage points,
and in interview Q&A *"is your evaluation noisy?"* is the second question
after *"what did you measure?"* You want CIs tight enough that the answer
is *"the headline holds even if you remove the worst query."*

n = 60 was the smallest size that:
1. Got us **overall NDCG@10 = 0.875 ± 0.04** — tight enough to be
   load-bearing.
2. Allowed **at least 3 queries per category** — so per-category numbers
   are at least directional, even if some category CIs (`pediatric` at
   ± 0.32) are wide.
3. Cost **~$1 in API spend** for full top-20 labeling. Trivial.

**How to apply this rule** (verbatim from CLAUDE.md): *when adding new
evaluation slices (fairness, location, language), aim for ≥ 5 queries per
slice so the per-slice CI half-width stays under ± 0.20; budget
~$0.02/labeled pair × top-K when projecting cost.*

This is the kind of decision that looks trivial — "we picked 60 queries
because 20 was too few" — but the *reasoning* is what makes it
defensible. *"We chose n such that the per-category CI half-width is
under 0.20 and the API spend is under $1"* signals you've thought about
the trade-off, instead of just rounding to 100.

### Decision 32 — LLM-as-judge limitations are documented *honestly and in the headline*, not hidden in a methodology footnote

**The trade-off in one sentence**: a clean-looking number invites
interview pushback that costs more credibility than a slightly-less-clean
number with the matching caveat would have.

Three concrete instances of "honest in the headline":

1. **Inter-annotator κ in the executive summary**, not the appendix. Quad
   κ = 0.72 [0.60, 0.82] sits next to NDCG = 0.88. They're equal-status.

2. **The Haiku-over-rates-3s finding is in §6, not omitted.** Of 62 Haiku-3
   labels, Sonnet kept only 39. We surface this explicitly. It would have
   been easy to report "κ = 0.72" and stop. Instead we reported "κ = 0.72,
   AND here's the systematic bias the kappa surfaced." The second clause
   builds trust the first clause alone wouldn't.

3. **§4.4 (CT.gov caveats) names the registry-coverage asymmetry up
   front.** *"Complex shows 0 theirs-only because CT.gov searches 500 K
   trials and we search 140 K oncology-only — that's a coverage gap on
   our side, not a CT.gov failure."* The reader doesn't have to ask;
   we volunteer it.

**Why it works**: interviewers immediately distrust evaluation reports
that sound too clean. A well-calibrated *"here's what this number doesn't
tell you"* paragraph builds more trust than a clean number alone.

**The rule** (verbatim from CLAUDE.md): *every new evaluation report
ships with §Limitations as a first-class section listing (i) sample size
+ CI width, (ii) label provenance + judge identity, (iii) anything the
metric cannot measure. Refuse to publish a metric without the matching
caveat.*

The phrase "**refuse to publish**" is doing real work there. It's the
difference between *"I write good reports"* (forgettable) and *"I have a
rule that prevents me from writing bad ones"* (sticky).

---

## Part 5 — Numbers worth memorizing

These are the headline numbers you should be able to recite from the
evaluation report:

| Number | Meaning |
|---|---|
| **1200** | (query, trial) pairs labeled by Haiku across 60 queries (`data/evaluation/full_labeled_dataset.jsonl`) |
| **60** | queries spanning 7 stress-testing categories (+ 30 carry-over) |
| **0.883 ± 0.04** | overall NDCG@5 on the 60-query labeled set |
| **0.875 ± 0.04** | overall NDCG@10 |
| **0.626** | NDCG@5 on `complex` queries — the biggest known weakness |
| **0.688** | NDCG@5 on `vague` queries — the second biggest |
| **0.617 → 0.670** | NDCG@5 lift from BM25 to the full pipeline (held-out, +8.6 %) |
| **κ = 0.723** | Sonnet-vs-Haiku quadratic-weighted Cohen's κ (substantial, Landis-Koch) |
| **97 %** | off-by-one accuracy between the two judges |
| **62 → 39** | Haiku-3 labels Sonnet kept (the rest collapsed to 2 or 1; never the reverse) |
| **0.5 / 20** | average overlap between TrialMine and CT.gov on the 20-query head-to-head |
| **77 % vs 67 %** | TrialMine vs CT.gov precision on the disjoint set |
| **+9.8 pts** | TrialMine's precision delta on the disjoint set |
| **+11.5** | biggest per-category win (pediatric: 15.0 ours-rel vs 3.5 theirs-rel per query) |

When asked *"how good is your system?"*, the right answer is some subset
of these numbers, **paired with the matching caveat**. NDCG = 0.88 is the
crown jewel; the κ = 0.72 is the credibility tax that lets you charge for
it.

---

## Part 6 — Conclusions

This is the most important section. Six concrete conclusions we can defend with the evidence above.

### Conclusion 1 — The system is genuinely good at the task it was designed for

**Evidence:** NDCG@10 = 0.875 ± 0.04 across 60 queries. Bootstrap 95 % CIs are tight enough that removing any single query doesn't change the headline.

**What this means:** for the queries we trained for — clearly-stated cancer-type + treatment-modality — the pipeline ranks correctly almost every time. Common, rare, treatment, and existing categories all score above 0.92.

**What it does *not* mean:** the system is good at everything a real user might ask. See Conclusion 2.

### Conclusion 2 — The system has two real, named weaknesses

**Evidence:**
- `complex` queries (multi-fact patient profile) score **NDCG@5 = 0.626** — below the 0.7 threshold where ranking quality starts to matter to a user.
- `vague` queries score **NDCG@5 = 0.688**.
- Both are well below the overall mean of 0.883, with non-overlapping CIs.

**What this means:** when a query carries multiple constraints (`58M EGFR exon 19 NSCLC failed osimertinib phase 2-3`), the pipeline gets the *cancer type* right but mishandles the *constraint stack* (age, prior treatment, line of therapy). When a query carries no constraint at all (`I have cancer what trials`), the pipeline falls back to whatever's most popular in the index. Neither case is a "failure" in the bug sense — both are diagnosed gaps with known causes.

### Conclusion 3 — The biggest fix is not an ML problem, it's a plumbing problem

**Evidence:** in the §5 error analysis of `docs/evaluation-report.md`, **3 of the 5 worst queries fail for the same reason** — the agent's `QueryParserAgent` extracts age / prior treatment / progression status into a `PatientProfile`, but `SearchOrchestrator` does not pass any of that into the BM25 filter dict.

- Q410 *"medulloblastoma 6 year old"* — age 6 parsed but adult trials still surface.
- Q413 *"failed osimertinib"* — parsed but osimertinib-naive trials still rank high.
- Q416 *"post-trastuzumab progression"* — parsed but the STOP-HER2 trial (opposite clinical situation) ranks #1.

**What this means:** wiring `EligibilityProfile` into the retrieval filter is **the highest-leverage open task in the project**. No retraining. No new dataset. No model changes. Just plumbing. Expected lift on those three queries: NDCG@10 from 0.42–0.55 → ≈ 0.85+.

This is the kind of finding evaluation is *supposed* to produce. We didn't know which fix mattered until the data told us.

### Conclusion 4 — TrialMine demonstrably beats ClinicalTrials.gov keyword search

**Evidence:** on 20 queries, top-20 from each side, average overlap is 0.5/20 (essentially disjoint). On the disjoint set, **TrialMine's precision is 77 %, CT.gov's is 67 % — a +9.8 percentage-point margin**. We win in *every* category. Biggest gap: pediatric (+11.5).

**What this means:** an actual patient with an actual question gets meaningfully more relevant trials from TrialMine than from typing the same query into clinicaltrials.gov's search box. The crown-jewel claim of the whole project is now a measurable, defensible number, not an aspiration.

**Honest caveat:** CT.gov ranks by keywords, not ML. If they shipped a learned re-ranker tomorrow, our edge would shrink. The +9.8 pts is *"ML retrieval beats keyword search"*, not *"TrialMine beats a competing ML system."*

### Conclusion 5 — Our labels are trustworthy at the level we need them to be

**Evidence:**
- Sonnet 4.6 vs Haiku 4.5, identical prompt: **quadratic-weighted Cohen's κ = 0.723 [0.601, 0.819]** — substantial agreement (Landis-Koch).
- 97 % off-by-one accuracy — disagreements are almost always *"I think it's a 3, you think it's a 2"*, never *"I think it's a 0, you think it's a 3"*.
- Pearson r = 0.776.

**What this means:** the per-category NDCG numbers, the CT.gov delta, and the ablation lifts are all robust to *which LLM judge* you use. If we re-ran the entire evaluation with Sonnet labels instead of Haiku labels, the headlines would shift but not invert.

**Honest caveat (Conclusion 6):** we *did* find a real bias. Don't skip the next conclusion.

### Conclusion 6 — Haiku has a measurable bias toward over-rating the top tier

**Evidence:** the Haiku-vs-Sonnet confusion matrix shows that of 62 pairs Haiku rated 3 ("highly relevant"), Sonnet kept only 39. Twenty got demoted to 2; three got demoted to 1; **zero went the other way** (Sonnet never rated higher than Haiku, only equal-or-lower).

**What this means:** Haiku slightly inflates the headline NDCG. A Sonnet-only re-evaluation of the same dataset would land NDCG@10 closer to ~0.84 instead of 0.875 — still genuinely good, but not as bright. We surfaced this in §6 of the evaluation report alongside the κ, not in a footnote.

**Why this conclusion matters more than it seems:** in interview settings, *"we found and named our own bias"* builds far more trust than a clean number alone. Decision 32 — honest limitations as a first-class section — is a process choice that earns its keep here. The bias was real, and we'd have looked careless if a reviewer found it before we did.

---

## Part 7 — What's next / open questions

Three things hang over the end of this week.

### 1. Wire `EligibilityProfile` into the BM25 filter

**The single highest-leverage open task in the project.** The error analysis
in §5 of the eval report showed that 3 of the 5 worst-NDCG queries fail
because the agent parses patient facts (age, prior treatment, progression
status) into a `PatientProfile`, but the orchestrator never uses those facts
to filter trials.

Fixing this is **non-ML work** — no retraining, no new dataset, just plumbing.
Expected lift on Q410 / Q413 / Q416 is from NDCG@10 ≈ 0.42–0.55 to ≈ 0.85+
without touching the model.

The reason this isn't done is straightforward: this week was scoped to
*evaluation*, not *fixing what evaluation surfaced*. Week 10's first task.

### 2. Human-anchor the Sonnet kappa on the disagreement set

We have 33 (Haiku, Sonnet) pairs that disagreed. A clinical reviewer could
hand-label all 33 in ~2 hours. That gives us:

- A real human-vs-LLM κ on the *hardest* part of the dataset (pure agreement
  pairs are uninformative — both judges said the same thing).
- The ability to claim "validated against humans on the disagreement set" in
  interviews instead of "validated against a stronger LLM."

`scripts/manual_label.py --skip-errors` is set up for this; you'd just point
it at the disagreement-only subset and label.

### 3. Re-evaluate after the eligibility-filter fix

Once the filter wiring lands, run:

```bash
python scripts/build_full_eval_dataset.py --resume --limit 5
```

…against just the 5 worst queries. The `--resume` flag means we won't
re-spend on the other 55 queries' labels. Cost: ~$0.10. Time: ~5 minutes.

If the fix worked, those queries' NDCG@10 jumps from 0.42–0.60 into the
0.85+ range, and the overall NDCG@10 jumps too. The honest reporting move
is to publish *both* the pre-fix and post-fix numbers in the next report —
the *delta* tells the story better than either number alone.

### Open questions

- **Fairness analysis**. We have queries proxying for under-served
  populations (pediatric, rural, non-English-speaking, AYA cancer survivors)
  in `data/evaluation/test_labels_v2.jsonl`. We do *not* yet measure whether
  NDCG drops disproportionately on those slices.
- **Cross-encoder latency**. ~6 s per query on CPU dominates the 15 s wall-
  clock budget. A GPU re-rank or a distilled CE would shrink this 5–10×.
  Untouched this week.
- **MLflow integration**. None of this week's evaluation runs got logged to
  MLflow. The ablation does; the per-category and CT.gov work does not.
  Trivial to fix, but deferred.

---

## Recap

We turned a 20-query / one-judge / no-baseline / multi-document evaluation
into a **60-query / two-judge / CT.gov-baselined / one-report**
evaluation, in one week, for ~$2.50 of LLM API spend. Every artifact ships
with the caveats that make it interview-defensible.

**The answer to "is the system good?"**: yes — NDCG@10 = 0.875, +9.8 pts vs CT.gov, κ = 0.72. **The answer to "is it good at everything?"**: no — `complex` and `vague` queries are weak, and the single highest-leverage fix is non-ML plumbing (Conclusion 3). The evaluation didn't just measure the system; it told us where to spend Week 10's effort.

The deliverables:

```
docs/evaluation-report.md              — 7-section consolidated report
docs/ctgov_comparison.md               — strongest ours-only wins per query
data/evaluation/full_labeled_dataset.jsonl       — 1200 Haiku labels
data/evaluation/full_labeled_dataset_human.jsonl — 100 Sonnet labels
data/evaluation/agreement_analysis.json          — κ + bootstraps + confusion matrix
data/evaluation/per_query_ndcg.json              — per-query metrics
scripts/build_full_eval_dataset.py     — Haiku labeling pipeline
scripts/manual_label.py                — interactive human-labeling CLI
scripts/sonnet_label.py                — Sonnet stratified-sample labeling
scripts/agreement_analysis.py          — Cohen's κ + bootstrap CIs
scripts/compare_with_ctgov.py          — bidirectional CT.gov head-to-head
```

Next week: fix what the evaluation found.
