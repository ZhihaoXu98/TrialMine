# Fix Bi-Encoder — Vibe-Coding Runbook

> Step-by-step plan to address Issue #2 (synthetic coverage + batch size) and
> the carved-out pieces of Issue #3 (taxonomy gaps + rare-cancer eval queries)
> from `docs/things_can_be_fixed.md`.
>
> **How to use this doc.** Each step has a copy-paste prompt to send to
> Claude Code. After each prompt, run the **Verify** check and confirm the
> **Acceptance criteria** before moving on. If a check fails, run the
> **Rollback** and re-do that step — never proceed on a failed check.
>
> Phases are sequenced so:
> - Phase A is all local work (no GPU). Do this first, end-to-end.
> - Phase B is a single cloud session (4 hours, unattended).
> - Phase C is local post-training work (re-label, metrics, decision).

---

## Overview

| | |
|---|---|
| **Goal** | Lift `complex` NDCG@5 ≥ 0.68 and `vague` NDCG@5 ≥ 0.74 (both +0.05 over v1 baseline), without regressing common/rare/treatment categories. |
| **Approach** | 10× synthetic patient queries (with 6 shape categories), bump batch 32 → 128, fix taxonomy gaps (medulloblastoma / GIST / NETs / Wilms / biliary / MDS), split sarcoma into 4 sub-buckets, add 5 rare-cancer eval queries. |
| **Total cost** | ~$26 (Haiku $18 + cloud GPU ~$8) |
| **Total time** | ~1 day local work + ~4 hr unattended cloud session |
| **Decision gate** | Phase C, Step C4 — pre-registered thresholds; either ship v2 or revert and pivot to Issue #4 (multi-vector encoding). |

**Files modified across all phases:**

| File | Phase |
|---|---|
| `configs/training_data.yaml` | A2, A4 |
| `configs/training/embeddings.yaml` | A7 |
| `scripts/generate_training_data.py` | A3 |
| `scripts/build_index.py` | A5 |
| `scripts/finetune_embeddings.py` | A6 (add `--override` flag) |
| `scripts/oom_probe.py` | A6 (new) |
| `scripts/select_lr.py` | A6 (new) |
| `scripts/build_full_eval_dataset.py` | A11 |
| `src/TrialMine/api/app.py` (or FAISS path config) | C1 |

---

## Best practices for these prompts

1. **One concern per prompt.** Each step is scoped to a single logical change. Don't combine.
2. **Verification is non-optional.** Every step has an explicit probe. If you skip it, you're flying blind.
3. **Quote acceptance criteria back.** When you run a probe, paste the output and explicitly check it against the **Acceptance criteria** before moving on.
4. **Context resets are fine.** Each prompt references this doc, so Claude can re-orient even if your session was cleared. Phase headers also re-bootstrap context.
5. **Never edit a file in two places at once.** Always finish + verify a step before starting the next.
6. **Do not let me say "done" without showing a diff or a probe output.** If a step has no observable output, you didn't verify it.

---

# Phase A — Local prep (no GPU, ~1 day)

## Step A0 — Session bootstrap (paste this at the start of every session)

If you're returning after a context reset or a break, send this first:

```
We are working through docs/fix_bi-encoder.md to fix the TrialMine
bi-encoder per Issue #2 + carve-outs of Issue #3 from
docs/things_can_be_fixed.md. Read fix_bi-encoder.md and report:
(1) which phase steps appear complete based on file state
(checkpoints, backups, modified scripts), and (2) the next step to
run. Do not modify any files yet.
```

This gives Claude a chance to read the runbook and tell you where you are.

---

## Step A1 — Backups (10 min)

**Goal:** Snapshot v1 artefacts before any destructive change. Re-finetune is a one-way trip without these.

### Prompt to send

```
Run Phase A1 from docs/fix_bi-encoder.md: create v1 backups before we
touch any training artefacts. After completing, run `ls -la` on each
backup target and report file sizes. Do not delete any originals — just
copy.
```

### Files to back up

| Source | Backup |
|---|---|
| `models/embeddings/fine-tuned/` | `models/embeddings/fine-tuned-v1/` |
| `data/faiss_finetuned.index` | `data/faiss_finetuned_v1.index` |
| `data/faiss_finetuned.json` | `data/faiss_finetuned_v1.json` |
| `data/training/train_pairs.jsonl` | `data/training/train_pairs_v1.jsonl` |
| `data/training/val_pairs.jsonl` | `data/training/val_pairs_v1.jsonl` |
| `data/training/synthetic_queries.jsonl` | `data/training/synthetic_queries_v1.jsonl` (`mv`, not `cp`) |
| `data/training/synthetic_checkpoint.jsonl` | `data/training/synthetic_checkpoint_v1.jsonl` (`mv`, not `cp`) |

The last two are `mv` (not `cp`) so that the next synthetic-generation run starts clean.

### Verify

```
List all backup files with sizes and confirm the v1 model directory
has its weights (config.json + pytorch_model.bin or model.safetensors).
```

### Acceptance criteria

- 6 backup files exist (4 copies + 2 moves)
- `models/embeddings/fine-tuned-v1/` contains the model weights (~430 MB)
- `data/training/synthetic_queries.jsonl` and `synthetic_checkpoint.jsonl` no longer exist in the original location

### Rollback

N/A (nondestructive).

---

## Step A2 — Fix taxonomy in `configs/training_data.yaml` (20 min)

**Goal:** Add 6 missing cancer subtypes (medulloblastoma, GIST, NETs, Wilms, biliary, MDS), split sarcoma into 4 buckets.

### Prompt to send

```
Apply the taxonomy fix to configs/training_data.yaml per Phase A2 of
docs/fix_bi-encoder.md.

CRITICAL: cancer_types is order-sensitive (first-match wins in
classify_cancer_type). Specific subtypes MUST come before catch-alls:
- medulloblastoma BEFORE brain (otherwise "brain" eats it)
- biliary BEFORE liver
- sarcoma subtypes (bone, pediatric, soft_tissue) BEFORE sarcoma_other
- Specific subtypes (medulloblastoma, gist, etc.) at the TOP of the
  cancer_types block

After editing, show me the full final cancer_types block (no
truncation) so I can audit ordering.
```

### Final `cancer_types` block (target)

```yaml
cancer_types:
  # ── Specific subtypes FIRST (order-sensitive, first match wins) ──
  medulloblastoma: ["medulloblastoma"]
  wilms: ["wilms"]
  gist: ["GIST", "gastrointestinal stromal"]
  neuroendocrine: ["neuroendocrine", "carcinoid"]
  biliary: ["biliary", "cholangio"]
  mds: ["myelodysplastic"]

  # ── Sarcoma split — subtypes BEFORE catch-all ──
  sarcoma_bone: ["osteosarcoma", "chondrosarcoma"]
  sarcoma_pediatric: ["ewing", "rhabdomyosarcoma"]
  sarcoma_soft_tissue: ["soft tissue sarcoma", "leiomyosarcoma", "liposarcoma", "synovial sarcoma"]
  sarcoma_other: ["sarcoma"]

  # ── Existing groups (do NOT reorder relative to each other) ──
  breast: ["breast"]
  lung: ["lung", "NSCLC", "SCLC", "non-small cell", "small cell lung"]
  prostate: ["prostate"]
  colorectal: ["colorectal", "colon", "rectal"]
  melanoma: ["melanoma"]
  leukemia: ["leukemia", "leukaemia"]
  lymphoma: ["lymphoma"]
  myeloma: ["myeloma"]
  pancreatic: ["pancrea"]
  brain: ["glioblastoma", "glioma", "brain", "GBM", "meningioma"]
  ovarian: ["ovari"]
  kidney: ["kidney", "renal"]
  bladder: ["bladder", "urothelial"]
  liver: ["hepatocellular", "liver", "HCC"]
  head_neck: ["head and neck", "nasophary", "orophary", "laryn", "oral cavity"]
  gastric: ["gastric", "stomach", "esophag"]
  cervical: ["cervic"]
  endometrial: ["endometri", "uterine"]
  thyroid: ["thyroid"]
  mesothelioma: ["mesothelioma"]
  neuroblastoma: ["neuroblastoma"]
```

### Verify (the probe)

```
Run a verification probe per Phase A2 of docs/fix_bi-encoder.md:
classify every trial in data/trials.db with the new taxonomy and print
the per-group corpus count. Compare against the expected outcomes
below and tell me if any deviate by more than 10%.

Expected:
- 'other' drops from 46,948 (33%) to ~30,000–37,000 (21–26%)
- medulloblastoma: ~200 trials (new bucket)
- gist: ~370 trials (new bucket)
- neuroendocrine: ~850–1100 trials (new bucket)
- wilms: ~100 trials (new bucket)
- biliary: ~250–400 trials (new bucket)
- mds: ~150–200 trials (new bucket)
- sarcoma_other + sarcoma_bone + sarcoma_pediatric + sarcoma_soft_tissue ≈ 1,740 (old sarcoma total)
```

### Acceptance criteria

- All 6 new buckets non-empty
- `sarcoma_*` buckets sum to ≈1,740 (±5%)
- `other` < 38,000

### Rollback

```
Restore configs/training_data.yaml from git:
  git checkout configs/training_data.yaml
```

---

## Step A3 — Rewrite synthetic generation: shape prompts + quota sampling (90 min)

**Goal:** Replace the single Haiku prompt with 6 shape-specific templates, and replace random trial selection with quota-aware sampling that floors rare cancers.

### Prompt to send

```
Edit scripts/generate_training_data.py per Phase A3 of
docs/fix_bi-encoder.md. Two changes:

1. Replace USER_PROMPT_TEMPLATE with a SHAPE_PROMPTS dict (6 shapes:
   failed_treatment, post_progression, biomarker, multi_constraint,
   vague, caregiver). Each prompt is shape-specific. Rotate shapes
   round-robin by row index in generate_synthetic_queries.

2. Add a "shape" field to each output row (in both the checkpoint
   write and the final JSONL).

3. Replace the trial-selection block (around line 338–343) with
   quota-aware sampling:
   - Read per_group_floor and per_group_ceiling from
     config["synthetic"]
   - Fill rare-group floors first (with-replacement sampling if a
     group has fewer trials than its floor — log a WARN when this
     happens and report the unique-trial count at the end)
   - Distribute the remainder over un-floored groups proportionally,
     capped at per_group_ceiling

Use type hints, structured logging (no print()), and follow project
conventions per CLAUDE.md. Show me the diff of every change.

Do NOT touch the YAML in this step — quotas will be added to YAML in
Step A4.
```

### Shape prompts (target content)

The script should contain something like:

```python
SHAPE_PROMPTS = {
    "failed_treatment": (
        "Write a 1-2 sentence patient query where the patient's PREVIOUS "
        "treatment STOPPED WORKING. Name the failed drug/regimen "
        "explicitly if shown in the trial info."
    ),
    "post_progression": (
        "Write a 1-2 sentence patient query where the patient PROGRESSED "
        "AFTER a prior therapy. Reference the prior therapy by name if "
        "shown."
    ),
    "biomarker": (
        "Write a 1-2 sentence patient query that mentions a SPECIFIC "
        "BIOMARKER OR MUTATION (e.g., EGFR, HER2, MSI-high, BRCA, PD-L1). "
        "If the trial does not mention one, infer the most likely one "
        "for this cancer type."
    ),
    "multi_constraint": (
        "Write a 1-2 sentence patient query that combines AT LEAST TWO "
        "of: age, stage, line of therapy, biomarker, geography."
    ),
    "vague": (
        "Write a 1-2 sentence VAGUE patient query in everyday language "
        "with NO medical jargon. Paraphrase the condition (e.g., 'a "
        "problem with my blood' for leukemia)."
    ),
    "caregiver": (
        "Write a 1-2 sentence query from a CAREGIVER (parent, spouse, "
        "adult child) describing the patient's situation in the third "
        "person."
    ),
}

SHAPE_ORDER = list(SHAPE_PROMPTS.keys())  # deterministic rotation
```

### Verify

```
Without making any API calls, dry-run the new generate_synthetic_queries
by setting sample_count=18 in a temp config and asserting:
- 18 rows are selected (or whatever the floored/budgeted total is)
- Shape rotation produces 3 of each shape across 18 rows
- The output row schema includes "shape"

If the dry-run requires the Anthropic SDK, stub the client to return a
fixed string so we don't burn tokens. Show me the script output.
```

### Acceptance criteria

- Diff covers: SHAPE_PROMPTS dict, rotation logic, output row has `shape` field, quota-aware sampling block replaces lines 338–343
- Dry-run produces 6 shapes × 3 rows = 18 rows, each tagged with one shape
- No `print()` calls added — uses `logger.info` etc.

### Rollback

```
git checkout scripts/generate_training_data.py
```

---

## Step A4 — Add per-group quotas to `configs/training_data.yaml` (15 min)

**Goal:** Tell the new quota logic where to floor rare cancers and where to cap common ones.

### Prompt to send

```
Add the synthetic per-group quota config to configs/training_data.yaml
per Phase A4 of docs/fix_bi-encoder.md.

Replace the current `synthetic:` block with the expanded version that
includes per_group_floor and per_group_ceiling. Bump sample_count from
1500 to 10000. Show me the final synthetic block.
```

### Target `synthetic:` block

```yaml
synthetic:
  model: "claude-haiku-4-5-20251001"
  sample_count: 10000
  max_requests_per_second: 5
  checkpoint_interval: 100
  # Per-group floors — fill these BEFORE distributing remainder.
  # Sums to 4,000; remaining ~6,000 distributes across un-floored groups.
  per_group_floor:
    mesothelioma: 500
    neuroblastoma: 500
    medulloblastoma: 400
    wilms: 300
    gist: 400
    neuroendocrine: 400
    thyroid: 300
    sarcoma_bone: 300
    sarcoma_pediatric: 300
    sarcoma_soft_tissue: 300
    sarcoma_other: 200
  per_group_ceiling: 600
```

### Verify

```
Sum the per_group_floor values and confirm: (a) total floored = 4,000,
(b) per_group_ceiling is 600, (c) every key in per_group_floor exists
in the cancer_types section. Report any orphans.
```

### Acceptance criteria

- Floors sum to 4,000
- No orphan keys (each floor key has a matching `cancer_types:` entry)
- `sample_count: 10000`

### Rollback

```
git checkout configs/training_data.yaml
```

---

## Step A5 — Update `scripts/build_index.py` for v2 output path (20 min)

**Goal:** Avoid overwriting the v1 FAISS index on rebuild.

### Prompt to send

```
Update scripts/build_index.py per Phase A5 of docs/fix_bi-encoder.md.

Two changes:
1. Add a --output flag (default: data/faiss_finetuned) that controls
   the output path stem for the .index and .json files.
2. The --model fine-tuned alias should keep its current
   models/embeddings/fine-tuned mapping by default, but accept any
   directory passed via a new --model-path flag (use it if provided;
   otherwise resolve from --model alias).

Use argparse, type hints, structured logging. Show me the diff and
confirm the existing call patterns still work
(--skip-bm25 --model fine-tuned).
```

### Verify

```
Run scripts/build_index.py --help and confirm both --output and
--model-path are present with sensible defaults. Then run it with
--dry-run (or equivalent that doesn't write files) to confirm the
output paths resolve correctly for:
  build_index.py --skip-bm25 --model fine-tuned
    -> writes data/faiss_finetuned.index + .json
  build_index.py --skip-bm25 --model-path models/embeddings/fine-tuned-v2 --output data/faiss_finetuned_v2
    -> writes data/faiss_finetuned_v2.index + .json
```

### Acceptance criteria

- Both old and new invocations resolve to the expected paths
- No file writes during the help/dry-run

### Rollback

```
git checkout scripts/build_index.py
```

---

## Step A6 — Add `oom_probe.py`, `select_lr.py`, and `--override` flag (60 min)

**Goal:** Build the three small pieces needed for an unattended cloud session.

### Prompt to send

```
Implement Phase A6 of docs/fix_bi-encoder.md. Three pieces:

1. scripts/oom_probe.py — a <50 LOC script that:
   - Loads BioLinkBERT-base in fp16
   - Runs 5 forward + backward passes at the batch size from
     configs/training/embeddings.yaml
   - Reports peak GPU memory and exits 0 on success, 1 on OOM
   - CLI: --batch INT (override config)
   - Type hints, structured logging, no print()

2. scripts/select_lr.py — a <50 LOC script that:
   - Reads the MLflow experiment "trialmind-embeddings"
   - Filters runs with run_tag matching "sweep-bs128-lr*"
   - Picks the run with highest final val_retrieval_cosine_ndcg@10
   - Writes the winning lr back to configs/training/embeddings.yaml's
     training.learning_rate (preserving all other fields)
   - Prints the winner + the runner-up's NDCG so we can see the gap
   - Type hints, structured logging

3. scripts/finetune_embeddings.py: add a --override KEY=VALUE flag
   (repeatable). Example usage:
     finetune_embeddings.py --override training.learning_rate=4e-5 \
                            --override training.max_steps=500
   Use dotted-path key resolution to set nested dict keys. Cast values
   to int/float/bool/str by best-effort.

Show me each new/modified file as a diff.
```

### Verify

```
Run each of these and confirm:
1. python scripts/oom_probe.py --help — flag exists, help text clean
2. python scripts/select_lr.py --help — no errors at import time
3. python scripts/finetune_embeddings.py --override training.batch_size=8 --dry-run
   (or smallest no-op invocation that exits before training)
   — confirms --override parses and applies

For oom_probe and select_lr, no actual GPU/MLflow needed in this step;
just confirm they import cleanly and have correct CLI surface.
```

### Acceptance criteria

- All three exit successfully on `--help`
- `--override` round-trips a value into the config dict (log the merged config when --dry-run)

### Rollback

```
rm scripts/oom_probe.py scripts/select_lr.py
git checkout scripts/finetune_embeddings.py
```

---

## Step A7 — Update `configs/training/embeddings.yaml` (10 min)

**Goal:** Switch to batch 128, new output dir, new logging cadence. Learning rate stays as a placeholder until the cloud sweep picks the winner.

### Prompt to send

```
Update configs/training/embeddings.yaml per Phase A7 of
docs/fix_bi-encoder.md. Set:
- model.output_dir: "models/embeddings/fine-tuned-v2"
- training.batch_size: 128
- training.learning_rate: 4.0e-5   # placeholder, overwritten by select_lr.py
- training.logging_steps: 20
- training.eval_steps: 250
- training.save_steps: 250
- mlflow.run_tag: "v2-synth10k-bs128-taxfix"

Leave epochs (3), warmup_ratio (0.1), weight_decay (0.01), fp16
(true), save_total_limit (3), load_best_model_at_end (true),
metric_for_best_model unchanged. Show me the final file.
```

### Acceptance criteria

- All 7 fields match the spec above
- v1 fields (epochs, warmup_ratio, etc.) unchanged
- `output_dir` ends in `-v2` (not the original `fine-tuned`)

### Rollback

```
git checkout configs/training/embeddings.yaml
```

---

## Step A8 — Generate 10K synthetic queries via Haiku (~35 min, ~$17)

**Goal:** Run the new shape-aware, quota-aware synthetic generation. Skips hard-neg mining so we can sanity-check the output first.

### Prompt to send

```
Run Phase A8: execute
  python scripts/generate_training_data.py --skip-hard-negatives

Stream the log to stdout. Monitor for: (a) rate-limit errors from the
Anthropic API (>3 consecutive failures = stop), (b) the unique-trial
count log line from the floor-with-replacement logic (warn me if any
group has < 50% unique trials).

Expected wall-clock: 30–40 min. Expected cost: ~$17. Do NOT proceed
to hard-neg mining (Step A10) until A9 passes.
```

### Verify

```
After the run completes, report:
- Total rows in data/training/synthetic_queries.jsonl
- File size
- Any errors or rate-limit retries in the log
```

### Acceptance criteria

- 10,000 rows ± 100 (some skips for empty trial text are OK)
- File size 10–15 MB
- Zero unhandled exceptions in the log

### Rollback (if generation fails partway)

```
# Truncate to a clean point, then resume:
mv data/training/synthetic_queries.jsonl /tmp/synth_partial.jsonl
mv data/training/synthetic_checkpoint.jsonl /tmp/syn_ckpt_partial.jsonl
python scripts/generate_training_data.py --skip-hard-negatives  # fresh start
```

---

## Step A9 — Sanity-check synthetic output (30 min)

**Goal:** Catch bad generation before paying for hard-neg mining + cloud GPU.

### Prompt to send

```
Run the three sanity checks in Phase A9 of docs/fix_bi-encoder.md
against data/training/synthetic_queries.jsonl. Report results inline.

If any check fails, STOP and tell me — do not proceed to Step A10.
```

### The three checks

```bash
# (a) Shape distribution — expect each shape ~1,667 (10,000 / 6)
python -c "
import json, collections
c = collections.Counter()
for line in open('data/training/synthetic_queries.jsonl'):
    c[json.loads(line)['shape']] += 1
total = sum(c.values())
for k, v in c.most_common():
    print(f'{k:20s} {v:5d}  ({100*v/total:.1f}%)')
"

# (b) Cancer group distribution — verify floored groups hit their quotas
python -c "
import json, collections
c = collections.Counter()
for line in open('data/training/synthetic_queries.jsonl'):
    c[json.loads(line)['cancer_group']] += 1
for k, v in c.most_common():
    print(f'{k:28s} {v:5d}')
"

# (c) Read 30 random queries — eyeball for template clones
python -c "
import json, random
rows = [json.loads(l) for l in open('data/training/synthetic_queries.jsonl')]
random.seed(42)
for r in random.sample(rows, 30):
    print(f'[{r[\"shape\"]:18s}] [{r[\"cancer_group\"]:22s}] {r[\"query\"]}')
"
```

### Acceptance criteria

- **(a)** Each shape between 1,400 and 1,900 rows. If one shape ≪ others, rotation is broken — STOP.
- **(b)** Each floored group at or above its quota:
  - mesothelioma ≥ 500
  - neuroblastoma ≥ 500
  - medulloblastoma ≥ 400
  - gist, neuroendocrine ≥ 400
  - wilms, thyroid, sarcoma_bone, sarcoma_pediatric, sarcoma_soft_tissue ≥ 300
  - sarcoma_other ≥ 200
  If any floor missed, the quota logic broke — STOP.
- **(c)** Out of 30 sampled queries, **at most 5 follow the same template** (e.g., "I have X cancer that came back…"). More than 5 → prompt didn't diversify; STOP and iterate `SHAPE_PROMPTS` wording.

### Rollback (if any check fails)

```
# Move bad output aside; iterate the prompt or quota code; rerun A8.
mv data/training/synthetic_queries.jsonl data/training/synthetic_queries_BAD.jsonl
mv data/training/synthetic_checkpoint.jsonl data/training/synthetic_checkpoint_BAD.jsonl
# Then edit scripts/generate_training_data.py to fix the issue, and rerun A8.
```

---

## Step A10 — Run Source 1 + hard-neg mining (~45 min)

**Goal:** Regenerate metadata pairs (Source 1) with the new taxonomy labels, then mine hard negatives (Source 3) against the combined (synthetic + metadata) pair set.

### Prompt to send

```
Run Phase A10: execute
  python scripts/generate_training_data.py --resume

The --resume loads the now-complete synthetic checkpoint (10K NCT IDs
already done) and skips Haiku entirely. It will regenerate Source 1
(metadata pairs, fast, ~5 min) under the NEW taxonomy, then run
Source 3 hard-neg mining (~40 min). Stream the log.

Final outputs to expect:
- data/training/train_pairs.jsonl  (~760K rows, ~1.1 GB)
- data/training/val_pairs.jsonl    (~190K rows, ~280 MB)
```

### Verify

```
Report:
- wc -l data/training/train_pairs.jsonl
- wc -l data/training/val_pairs.jsonl
- Distribution of cancer_group in train_pairs.jsonl (top 20 groups)
- Distribution of "source" field in train_pairs.jsonl (metadata vs synthetic)
```

### Acceptance criteria

- Train ≈ 700–800K rows
- Val ≈ 170–200K rows
- Cancer-group distribution includes the 6 new buckets (medulloblastoma, gist, etc.)
- `source: synthetic` rows ≥ 25K (10K queries × ~3 hard negatives each)

### Rollback

```
# Restore v1 training data:
cp data/training/train_pairs_v1.jsonl data/training/train_pairs.jsonl
cp data/training/val_pairs_v1.jsonl data/training/val_pairs.jsonl
```

---

## Step A11 — Add 5 rare-cancer eval queries (~30 min)

**Goal:** Make rare-cancer lift measurable. Without these, mesothelioma/neuroblastoma/medulloblastoma improvements are invisible in aggregate NDCG.

### Prompt to send

```
Update scripts/build_full_eval_dataset.py per Phase A11 of
docs/fix_bi-encoder.md.

Add 5 new queries with IDs 500–504 to the eval set, each tagged with
a category. Do NOT run the labelling pass yet — that happens in
Step C2 after retraining.

Queries to add:
  ID 500: "pleural mesothelioma immunotherapy trial after pemetrexed failure"
          categories: ["rare", "treatment"]
  ID 501: "high-risk neuroblastoma trial for my 4-year-old"
          categories: ["pediatric", "rare"]
  ID 502: "medulloblastoma group 3 trial after surgery and radiation"
          categories: ["pediatric", "rare"]
  ID 503: "GIST trial after imatinib resistance"
          categories: ["rare", "treatment"]
  ID 504: "metastatic midgut neuroendocrine tumor trial somatostatin refractory"
          categories: ["rare", "treatment"]

Show me the diff. If the script's query list lives in a separate JSON
file, edit that file instead and tell me where.
```

### Verify

```
Run scripts/build_full_eval_dataset.py --dry-run (or --list-queries if
that exists; otherwise add a print that just enumerates IDs and exits)
and confirm IDs 500-504 appear with the right strings and category
tags.
```

### Acceptance criteria

- IDs 500–504 visible, no duplicates with existing IDs
- Category tags match the spec
- No actual Haiku API calls yet

### Rollback

```
git checkout scripts/build_full_eval_dataset.py
```

---

## ✅ Phase A done — checkpoint before cloud

At this point your local state should be:
- All YAML and script edits committed (recommended: a single PR titled "feat(bi-encoder): v2 training pipeline")
- 10K synthetic queries generated and sanity-checked
- 760K-row train set + 190K-row val set ready
- 5 new eval queries added (but not labelled)
- v1 backups in place

**Recommended:** create a git branch + commit Phase A so the cloud session works from a clean snapshot. Do NOT commit the training data files (they're gitignored anyway).

### Prompt to send (optional)

```
Show me git status. If there are pending edits, propose a commit
message for the Phase A changes and stage the right files (configs +
scripts only — never data/ or models/).
```

---

# ⏸ STOP — Cloud session needed

Phase B requires a GPU. Decide between:

| Option | Cost | Friction | Recommended when |
|---|---|---|---|
| **Lambda Labs A100 80GB** | $1.79/hr × 4 hr ≈ $7 | Medium (account + SSH) | You want reliability |
| **RunPod A100 80GB** | $1.89/hr × 4 hr ≈ $8 | Low (web UI) | You want Jupyter |
| **Colab Pro+ A100** | $0 if subscribed | Low | You already have Pro+ AND the existing `notebooks/finetune_biolinkbert.ipynb` |

**My recommendation:** Lambda. Colab's A100 availability has been unreliable, and a 4-hr run with a 12-hr session cap leaves no headroom if something fails.

Before starting Phase B, decide which provider and tell me — Phase B steps differ slightly per provider.

---

# Phase B — Cloud session (~4 hr, unattended)

> **Re-read this whole section before starting**, then send the prompts one
> at a time. Phase B is sequential — don't parallelize.

## Step B0 — Choose provider and write `cloud_run.sh` (30 min)

### Prompt to send

```
Write Phase B0 of docs/fix_bi-encoder.md: create a cloud_run.sh in the
repo root that orchestrates the full cloud session. Target provider:
[FILL IN: lambda | runpod | colab].

Script must:
1. pip install -r requirements.txt
2. Run scripts/oom_probe.py --batch 128; exit if it fails
3. Run an LR sweep: for lr in 2e-5 4e-5 8e-5: run
   finetune_embeddings.py with --override training.learning_rate=$lr,
   --override training.max_steps=500, --override mlflow.run_tag=sweep-bs128-lr${lr},
   --override model.output_dir=/tmp/sweep_${lr}
4. Run scripts/select_lr.py — this writes the winning lr back to
   configs/training/embeddings.yaml
5. Run scripts/finetune_embeddings.py (full 3-epoch run, ~3 hr)
6. Run OMP_NUM_THREADS=1 scripts/build_index.py --skip-bm25 \
     --model-path models/embeddings/fine-tuned-v2 \
     --output data/faiss_finetuned_v2

Use set -euo pipefail. Log to cloud_run.log. Show me the final script.
```

For Colab, this becomes a notebook cell pasted into a new section of `notebooks/finetune_biolinkbert.ipynb`. Adapt accordingly.

### Verify

```
Show me the final cloud_run.sh contents. Confirm:
- set -euo pipefail at top
- 6 steps in order
- Logs to cloud_run.log
- No hardcoded API keys (uses env vars)
```

### Acceptance criteria

- Script is self-contained and re-runnable
- No paths that exist only on local Mac
- Env vars expected: `ANTHROPIC_API_KEY` (for re-label later, not needed during train), `MLFLOW_TRACKING_URI` (if remote tracking)

---

## Step B1 — Provision GPU + sync data (20 min)

This is a **manual** step — you, not Claude, do this. Claude doesn't have credentials.

### What to do

**For Lambda or RunPod:**
1. Provision an A100 80GB instance
2. From your local Mac, sync the repo + training data:
   ```bash
   # From /Users/tonyxu/Desktop/TrialMine
   rsync -avz --exclude='data/trials.db' --exclude='models/' \
         --exclude='data/faiss_*' --exclude='.git/' \
         ./ <user>@<host>:~/TrialMine/
   # Then explicitly sync the training files (large but needed)
   rsync -avz data/training/train_pairs.jsonl \
              data/training/val_pairs.jsonl \
              <user>@<host>:~/TrialMine/data/training/
   ```

**For Colab:**
1. `!git pull` your branch in the notebook
2. Upload `train_pairs.jsonl` and `val_pairs.jsonl` to Drive, mount, copy into the runtime

### Verify

On the cloud box:
```bash
ls -la ~/TrialMine/data/training/  # should show train_pairs.jsonl ~1.1 GB
ls -la ~/TrialMine/scripts/         # should show oom_probe.py, select_lr.py
nvidia-smi                          # should show A100 80GB
```

### Acceptance criteria

- All scripts + configs present on cloud box
- Training files present and correct size
- GPU detected

---

## Step B2 — Run `cloud_run.sh` (~4 hr unattended)

### What to do (manual)

```bash
cd ~/TrialMine
chmod +x cloud_run.sh
nohup ./cloud_run.sh > cloud_run.log 2>&1 &
disown
```

Then disconnect. Come back in 4 hours.

### Verify (after ~4 hours)

```bash
tail -50 cloud_run.log
ls -la models/embeddings/fine-tuned-v2/
ls -la data/faiss_finetuned_v2.index
```

### Acceptance criteria

- `cloud_run.log` ends with successful FAISS build message
- `models/embeddings/fine-tuned-v2/` contains model weights + metadata.json
- `data/faiss_finetuned_v2.index` exists, ~412 MB
- No OOM errors, no rate-limit errors, no NaN-loss errors

### Rollback (if any step in cloud_run.sh failed)

```
# Read the log; the step that failed is the one to debug.
# DO NOT retry the full run — fix the issue, then rerun just the
# remaining steps manually.
```

---

## Step B3 — Download artefacts back to local Mac (10 min)

### What to do

```bash
# On local Mac
rsync -avz <user>@<host>:~/TrialMine/models/embeddings/fine-tuned-v2/ \
       models/embeddings/fine-tuned-v2/
rsync -avz <user>@<host>:~/TrialMine/data/faiss_finetuned_v2.index \
       <user>@<host>:~/TrialMine/data/faiss_finetuned_v2.json \
       data/
rsync -avz <user>@<host>:~/TrialMine/cloud_run.log .
rsync -avz <user>@<host>:~/TrialMine/configs/training/embeddings.yaml \
       configs/training/embeddings.yaml  # picks up the winning LR
rsync -avz <user>@<host>:~/TrialMine/mlflow.db mlflow_cloud.db  # for inspection
```

### Verify (with Claude's help)

```
Confirm Phase B3 sync per docs/fix_bi-encoder.md. Check:
- models/embeddings/fine-tuned-v2/ has config.json + model weights
- data/faiss_finetuned_v2.index exists and is ~400 MB
- configs/training/embeddings.yaml has been updated with the winning LR
  from the sweep (not still 4.0e-5 placeholder)
- cloud_run.log shows successful completion of all 6 steps

If anything is missing or wrong, tell me what to re-sync.
```

### Acceptance criteria

- All 4 artefacts present locally
- v2 model directory has weights file (`model.safetensors` or `pytorch_model.bin`)
- FAISS index ≥ 400 MB

---

## Step B4 — Shut down cloud instance (manual, 2 min)

**Do not skip this step.** A100s cost $1.79/hr whether you use them or not.

For Lambda/RunPod: terminate the instance from the web UI.
For Colab: disconnect runtime.

---

# Phase C — Post-training (local, ~1 hr)

## Step C1 — Wire v2 FAISS index into the API (15 min)

**Goal:** Make the API load the v2 index, not v1.

### Prompt to send

```
Per Phase C1 of docs/fix_bi-encoder.md: update the API + agent
pipeline so that semantic search uses the v2 FAISS index. Two options
— pick the cleaner one for this codebase:

(A) Add a symlink:
    ln -sf faiss_finetuned_v2.index data/faiss_finetuned.index
    ln -sf faiss_finetuned_v2.json data/faiss_finetuned.json
    (no code change; lowest risk)

(B) Update the FAISS path in src/TrialMine/api/app.py (or wherever
    the FAISS path is loaded — likely the lifespan handler or a
    config) to point at faiss_finetuned_v2.

Also update wherever the embedding model path is loaded to point at
models/embeddings/fine-tuned-v2 (not the v1 directory).

Recommend which option and apply it. Show me the diff. Then start the
API locally and confirm /health returns 200 with no FAISS load errors.
```

### Verify

```
1. curl http://localhost:8000/health  → expect {"status": "ok"} or similar
2. Tail the API startup logs and confirm the FAISS path logged is
   data/faiss_finetuned_v2.index (or whichever you wired)
3. Send a test search: curl -X POST http://localhost:8000/api/v1/search \
   -H "Content-Type: application/json" \
   -d '{"query":"lung cancer trials","top_k":3,"use_agent":false}'
   → expect 3 results back
```

### Acceptance criteria

- API starts cleanly, logs show v2 paths loaded
- Sample search returns results (any non-empty list is fine)

### Rollback

```
# If symlinks broke something:
rm data/faiss_finetuned.index data/faiss_finetuned.json
cp data/faiss_finetuned_v1.index data/faiss_finetuned.index
cp data/faiss_finetuned_v1.json data/faiss_finetuned.json
# Or git checkout the api app.py if option B.
```

---

## Step C2 — Re-label all 65 queries top-20 (~20 min, ~$1.30)

**Goal:** Get clean labels under the v2 model. Do NOT use `--resume` — the v2 model surfaces candidates v1 never saw.

### Prompt to send

```
Run Phase C2 per docs/fix_bi-encoder.md:

1. Move data/evaluation/full_labeled_dataset.jsonl to
   data/evaluation/full_labeled_dataset_v1.jsonl (preserves v1 labels
   for comparison).

2. Run scripts/build_full_eval_dataset.py (NO --resume). This
   re-labels all 65 queries × top-20 = 1,300 (query, trial) pairs
   under the v2 model + v2 index, using Claude Haiku as the judge.

3. Expected: ~20 min wall-clock, ~$1.30 in Haiku spend.

After completion, assert wc -l of the new full_labeled_dataset.jsonl
is exactly 1,300 (65 queries × 20 candidates). If it is anything else
(especially if it is < 1,200 + 100 = 1,300 — could indicate
--resume was silently applied), STOP and tell me.
```

### Verify

```
wc -l data/evaluation/full_labeled_dataset.jsonl
# expect 1300

# Confirm the 5 new query IDs are present:
python -c "
import json
ids = set()
for line in open('data/evaluation/full_labeled_dataset.jsonl'):
    ids.add(json.loads(line)['query_id'])
for nid in [500, 501, 502, 503, 504]:
    print(nid, 'IN' if nid in ids else 'MISSING')
"
```

### Acceptance criteria

- Exactly 1,300 rows
- All 5 new query IDs (500–504) present
- All 60 original IDs present too

### Rollback

```
# Restore v1 labels (don't lose v2 file — keep both):
mv data/evaluation/full_labeled_dataset.jsonl data/evaluation/full_labeled_dataset_v2_PARTIAL.jsonl
mv data/evaluation/full_labeled_dataset_v1.jsonl data/evaluation/full_labeled_dataset.jsonl
```

---

## Step C3 — Compute metrics with bootstrap CIs (15 min)

### Prompt to send

```
Run Phase C3 per docs/fix_bi-encoder.md: compute per-category
NDCG@5/10 and MRR for v2 against the new labels in
data/evaluation/full_labeled_dataset.jsonl.

Categories to break out:
- common, rare, pediatric, complex, geographic, vague, treatment
  (existing 7 from docs/evaluation-report.md §3)
- NEW: rare_explicit (queries 500–504 only)

For each category, compute:
- NDCG@5 with bootstrap 95% CI (1000 resamples)
- NDCG@10 with bootstrap 95% CI
- MRR

Output as a markdown table. Compare side-by-side against the v1
baseline numbers in docs/evaluation-report.md §3. Highlight any
category where v2 < v1 - 0.02 (regression) or v2 > v1 + 0.05
(meaningful lift).

Do NOT make a decision yet — that's Step C4.
```

### Verify

```
Confirm the output table includes:
- 8 categories (7 original + rare_explicit)
- 3 metrics per category (NDCG@5, NDCG@10, MRR)
- CIs on NDCG@5 and NDCG@10
- Side-by-side v1 vs v2 columns
```

### Acceptance criteria

- Table is complete (no missing cells)
- CIs are non-degenerate (not zero-width)

---

## Step C4 — Decision gate (30 min, manual review)

**This is the only step where you make a judgment call.** Pre-registered thresholds:

| Slice | v1 baseline NDCG@5 | v2 threshold | Required? |
|---|---|---|---|
| complex | 0.626 | **≥ 0.68** | YES |
| vague | 0.688 | **≥ 0.74** | YES |
| rare_explicit (500–504) | (none — new) | **≥ 0.55** | YES (sanity check) |
| common | 0.92+ | **≥ 0.90** | YES (no regression) |
| rare | 0.92+ | **≥ 0.90** | YES (no regression) |
| treatment | 0.92+ | **≥ 0.90** | YES (no regression) |
| pediatric | varies | within v1 ± 0.05 | Soft |
| geographic | varies | within v1 ± 0.05 | Soft |

### Decision matrix

| Outcome | Action |
|---|---|
| **All required thresholds met** | Ship v2. Proceed to Step C5. |
| **Complex + vague met, BUT one of common/rare/treatment regressed > 0.02** | Investigate: most likely LR over-scaled at bs=128. Possibly revert and pick a different LR. |
| **Complex or vague missed (< threshold)** | Coverage hypothesis falsified. Revert to v1 (Step C5 rollback). Next dollar to Issue #4 (multi-vector encoding). |
| **rare_explicit < 0.55** | Rare-cancer floor + taxonomy fix underperformed. Investigate per-query failures before deciding ship vs revert. |

### Prompt to send (only after you've decided)

```
My decision after Phase C4 of docs/fix_bi-encoder.md is: [SHIP | REVERT
| INVESTIGATE].

Specifically:
- complex NDCG@5: <number> (threshold 0.68: PASS/FAIL)
- vague NDCG@5: <number> (threshold 0.74: PASS/FAIL)
- rare_explicit NDCG@5: <number> (threshold 0.55: PASS/FAIL)
- common: <number> vs v1 0.92+: PASS/FAIL
- rare: <number> vs v1 0.92+: PASS/FAIL
- treatment: <number> vs v1 0.92+: PASS/FAIL

Proceed to Step C5 with this decision.
```

---

## Step C5a — SHIP path (if decision was SHIP) (~30 min)

### Prompt to send

```
Ship v2 per Phase C5a of docs/fix_bi-encoder.md:

1. Rename models/embeddings/fine-tuned (v1 still under -v1 backup) so
   v2 is the canonical model directory, OR keep -v2 as canonical and
   update build_index.py + API to default to -v2. Recommend the
   second (more explicit).

2. Update docs/evaluation-report.md with the v2 per-category table
   from Step C3. Preserve the v1 numbers in a "Historical" appendix.

3. Update CLAUDE.md "What's working" section: bump the bi-encoder
   description to v2, note synth share is now ~4%, batch 128, new
   taxonomy with 6 added buckets + sarcoma split.

4. Update CLAUDE.md "Design decisions" with new entries:
   - Decision 33: shape-aware synthetic generation (6 categories)
   - Decision 34: per-group floor for rare cancers (with-replacement
     sampling); with-replacement risk mitigated by 6-shape prompt
     diversity per trial.
   - Decision 35: taxonomy expansion (medulloblastoma, GIST, NETs,
     Wilms, biliary, MDS; sarcoma split into 4 buckets).

5. Commit everything on a feature branch and open a PR with a clear
   summary of NDCG deltas.

Show me each diff before committing.
```

---

## Step C5b — REVERT path (if decision was REVERT) (~20 min)

### Prompt to send

```
Revert v2 per Phase C5b of docs/fix_bi-encoder.md:

1. Restore v1 model + index as canonical:
   - rm -rf models/embeddings/fine-tuned-v2  # or keep for archival
   - Ensure models/embeddings/fine-tuned/ points at the v1 weights
     (restore from fine-tuned-v1 backup if needed)
   - rm data/faiss_finetuned.index data/faiss_finetuned.json
   - cp data/faiss_finetuned_v1.index data/faiss_finetuned.index
   - cp data/faiss_finetuned_v1.json data/faiss_finetuned.json

2. Keep these archived in the repo (do NOT delete):
   - data/evaluation/full_labeled_dataset_v2_PARTIAL.jsonl (rename to
     full_labeled_dataset_v2_attempt.jsonl)
   - cloud_run.log
   - mlflow_cloud.db
   - The v2 sweep results in MLflow

3. Write a "lessons learned" entry in docs/things_can_be_fixed.md
   under Issue #2 — what was tried, what didn't lift, what the next
   intervention should be (likely Issue #4: multi-vector encoding).

4. Keep the YAML and script changes from Phase A on a branch — they
   are still useful improvements (taxonomy fix, shape prompts,
   build_index --output flag, helper scripts). Do NOT revert those.
   Only revert configs/training/embeddings.yaml back to batch 32
   if you want to fully undo.

Show me each step before executing.
```

---

# Appendix — Failure modes to specifically watch for

These are the same failure modes from our earlier analysis. If any of these
symptoms appear, **STOP** and investigate before continuing.

| Symptom | Likely cause | Fix |
|---|---|---|
| `sarcoma_other` ≫ 619 in taxonomy probe (A2) | YAML order broken — `sarcoma: ["sarcoma"]` listed before subtype keywords | Re-edit YAML, put subtypes first |
| One shape ≫ 1900 in synthetic check (A9a) | Shape rotation off-by-one or bug | Re-read `generate_synthetic_queries`, fix rotation |
| Rare group < its floor in A9b | Quota logic broken — likely an off-by-one or condition mismatch | Re-edit quota block; rerun A8 |
| > 5 of 30 sampled queries follow same template (A9c) | Haiku ignoring shape instruction | Iterate `SHAPE_PROMPTS` wording; rerun A8 |
| API loads v1 FAISS after C1 | Symlink not picked up, or config still points at v1 | Restart API; check startup log for actual path |
| `wc -l full_labeled_dataset.jsonl` < 1300 (C2) | `--resume` was silently applied | Delete the partial file and rerun without resume |
| common NDCG@5 < 0.88 (C3/C4) | LR over-scaled at bs=128 | REVERT; next attempt try lr=4e-5 or lr=2e-5 |
| All v2 numbers equal v1 numbers exactly | API still loading v1 FAISS — see above | Restart API, verify path in startup log |

---

# Appendix — Cost recap

| Item | Cost | Phase |
|---|---|---|
| Haiku API — 10K synthetic queries | ~$17 | A8 |
| Haiku API — re-label 1,300 pairs | ~$1.30 | C2 |
| Lambda A100 80GB — 4 hr | ~$7.20 | B |
| **Total** | **~$26** | |

Plus ~1 day engineer wall-clock (mostly local; cloud run is unattended).
