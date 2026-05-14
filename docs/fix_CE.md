# Fix Cross-Encoder — Vibe-Coding Runbook

> Step-by-step plan to address Issue #5 from `docs/things_can_be_fixed.md`:
> the cross-encoder was trained on binary triplet labels (`BinaryCrossEntropyLoss`)
> and converged to a disease/intervention matcher rather than a graded ranker.
> The val metric (`CERerankingEvaluator_ndcg@10 = 0.993`) is saturated and not
> predictive of production NDCG (pure-CE replacement at NDCG@5 = 0.657 vs
> hybrid baseline 0.816). The blender currently caps CE weight at 0.3 because
> CE is too noisy to trust alone.
>
> **How to use this doc.** Each step has a copy-paste prompt to send to
> Claude Code. After each prompt, run the **Verify** check and confirm the
> **Acceptance criteria** before moving on. If a check fails, run the
> **Rollback** and re-do that step — never proceed on a failed check.
>
> Phases are sequenced so:
> - Phase A is all local work (no GPU). Do this first, end-to-end.
> - Phase B is a single cloud session (~1.5 hr, unattended).
> - Phase C is local post-training work (eval, blender re-tune, decision).

---

## Overview

| | |
|---|---|
| **Goal** | Lift pure-CE NDCG@5 from 0.657 → ~0.80–0.85; lift blended NDCG@5 from 0.829 → ~0.85–0.87; close the 33-pt val/production gap; ship a CE the blender can lean on with weight > 0.3. |
| **Approach** | Build graded `(query, doc_A, doc_B, margin)` pairs from existing ~7K Haiku labels (query-level split, full 65-query test set held out). Swap `BinaryCrossEntropyLoss` → `MarginMSELoss`. **Warm-start** from the existing binary-trained CE checkpoint. Replace the saturated 1-pos/1-neg val evaluator with a pooled production-shaped evaluator. |
| **Total cost** | ~$3 (Lambda A100 40GB ~1.5 hr × $1.79/hr — buys ~30 min wall-clock back vs A10) + $0 API spend. Optional: ~$10 to regrade ~2K Haiku-noisy pairs with Sonnet. |
| **Total time** | ~3 hr local prep + ~1 hr unattended cloud session + ~1 hr decision gate. |
| **Decision gate** | Phase C, Step C4 — **holistic review** of blended NDCG, pure CE NDCG, per-category breakdown, val/prod gap closure, blender α winner. No rigid pre-registered thresholds; ship/revert based on the overall picture, with "ship anyway with honest writeup" as a valid path (precedent: Decision 36). |

**Files modified across all phases:**

| File | Phase |
|---|---|
| `scripts/build_ce_graded_training.py` | A2 (new) |
| `data/training/ce_graded_train.jsonl` | A2 (new) |
| `data/training/ce_graded_val.jsonl` | A2 (new) |
| `configs/training/cross_encoder.yaml` | A4 |
| `scripts/finetune_cross_encoder.py` | A5 (loss + warm-start + override flag + evaluator wiring) |
| `src/TrialMine/evaluation/ce_pooled_evaluator.py` | A5 (new) |
| `cloud_run_ce.sh` | A6 (new) |
| `src/TrialMine/agents/tools.py` (env-default for `TRIALMINE_CROSS_ENCODER`) | C1 |
| `src/TrialMine/models/cross_encoder.py` (blender weights, if re-tuned) | C3 |
| `docs/evaluation-report.md` | C5a |
| `CLAUDE.md` (Decision 39 + "What's working" CE bump) | C5a |

---

## Best practices for these prompts

1. **One concern per prompt.** Each step is scoped to a single logical change. Don't combine.
2. **Verification is non-optional.** Every step has an explicit probe. If you skip it, you're flying blind.
3. **Quote acceptance criteria back.** When you run a probe, paste the output and explicitly check it against the **Acceptance criteria** before moving on.
4. **Context resets are fine.** Each prompt references this doc; Claude can re-orient even if your session was cleared.
5. **Never edit a file in two places at once.** Always finish + verify a step before starting the next.
6. **No "done" without a diff or a probe output.** If a step has no observable output, you didn't verify it.

---

# Phase A — Local prep (~3 hr, no GPU)

## Step A0 — Session bootstrap (paste this at the start of every session)

If you're returning after a context reset or a break, send this first:

```
We are working through docs/fix_CE.md to retrain the TrialMine
cross-encoder per Issue #5 of docs/things_can_be_fixed.md. Read
fix_CE.md and report: (1) which phase steps appear complete based on
file state (backups, new scripts, training data JSONLs, modified
configs), and (2) the next step to run. Do not modify any files yet.
```

This gives Claude a chance to read the runbook and tell you where you are.

---

## Step A1 — Backups (10 min)

**Goal:** Snapshot the binary-trained CE artefacts before any destructive change. The warm-start uses this checkpoint as init, so corrupting it would force a full re-train of v1 first.

### Prompt to send

```
Run Phase A1 from docs/fix_CE.md: create v1 backups of the current
cross-encoder artefacts before we touch any training files. After
completing, run `ls -la` on each backup target and report file sizes.
Do not delete originals — just copy.
```

### Files to back up

| Source | Backup |
|---|---|
| `models/cross-encoder/fine-tuned/` | `models/cross-encoder/fine-tuned-v1/` |
| `configs/training/cross_encoder.yaml` | `configs/training/cross_encoder_v1.yaml` |

Training data backup is **not needed** — we're not regenerating the source label JSONLs; we're only combining them. The 4 label files (`labeled_queries.jsonl`, `test_labels.jsonl`, `train_labels_extra.jsonl`, `test_labels_v2.jsonl`) and the held-out `full_labeled_dataset.jsonl` are all read-only inputs.

### Verify

```
List backup files with sizes and confirm:
- models/cross-encoder/fine-tuned-v1/ contains model.safetensors (~430 MB)
- configs/training/cross_encoder_v1.yaml has the unmodified v1 hyperparameters
```

### Acceptance criteria

- `models/cross-encoder/fine-tuned-v1/model.safetensors` ≥ 420 MB
- `configs/training/cross_encoder_v1.yaml` exists and is byte-identical to the original

### Rollback

N/A (nondestructive).

---

## Step A2 — Build graded training dataset (60 min)

**Goal:** Convert the existing ~7K Haiku-graded `(query, trial, grade ∈ {0,1,2,3})` rows from 4 source files into ~40–50K graded `(query, doc_A, doc_B, margin)` pairs, with a strict **query-level** split (not pair-level) to prevent lexical leakage. Hold out everything in `full_labeled_dataset.jsonl` (the 65-query benchmark, IDs 0–19 + 300–309 + 400–429 + 500–504) and `full_labeled_dataset_expansion_v2.jsonl` (IDs 600–619).

### Prompt to send

```
Create scripts/build_ce_graded_training.py per Phase A2 of
docs/fix_CE.md. The script must:

1. Read all 4 source label files:
   - data/evaluation/labeled_queries.jsonl    (20 q, IDs 0-19)
   - data/evaluation/test_labels.jsonl        (50 q, IDs 100-149)
   - data/evaluation/train_labels_extra.jsonl (75 q, IDs 200-274)
   - data/evaluation/test_labels_v2.jsonl     (50 q, IDs 300-349)

2. Read the held-out set:
   - data/evaluation/full_labeled_dataset.jsonl           (65 q)
   - data/evaluation/full_labeled_dataset_expansion_v2.jsonl (20 q, IDs 600-619)
   Union the query_ids in these two files → HELD_OUT_QIDS.

3. Build a row-level training pool by:
   - Concatenating all 4 source files
   - Dropping any row whose query_id ∈ HELD_OUT_QIDS
   - Deduplicating on (query_id, nct_id), keeping the first occurrence
     (deterministic by file order)

4. Split queries (NOT pairs) into train/val at 80/20:
   - Sort the unique query_ids in the training pool
   - Use random.Random(42).shuffle to deterministically shuffle
   - First 80% → train queries; remaining 20% → val queries

5. For each split (train, val) and each query in that split:
   - Load the query's graded candidates (list of {nct_id, grade, ...})
   - For every ordered pair (A, B) where grade(A) > grade(B), emit a
     training row using the COLUMN NAMES expected by sentence-transformers
     MarginMSELoss for cross-encoders: `query`, `positive`, `negative`,
     `label` (the teacher margin). Audit fields kept alongside but ignored
     by training:
     {
       "query_id": int,            # audit only
       "query": str,                # required by MarginMSELoss
       "positive": str,             # required — trial text for the higher-graded doc
       "negative": str,             # required — trial text for the lower-graded doc
       "label": float,              # required — grade(A) - grade(B), in {1.0, 2.0, 3.0}
       "doc_pos_nct_id": str,      # audit only
       "doc_neg_nct_id": str,      # audit only
       "grade_pos": int,            # audit only — useful for A7 Sonnet re-grade filter
       "grade_neg": int,            # audit only
     }
   - Skip pairs where grade(A) == grade(B) (no ranking signal)
   - Trial text format: title + " [SEP] " + brief_summary (matches
     prepare_trial_text in src/TrialMine/models/embeddings.py)

6. Write outputs:
   - data/training/ce_graded_train.jsonl
   - data/training/ce_graded_val.jsonl
   - data/training/ce_graded_metadata.json — summary stats:
     {"n_train_queries", "n_val_queries", "n_train_pairs", "n_val_pairs",
      "label_distribution_train", "held_out_qid_count",
      "trial_text_lookup_cache_size", "build_timestamp"}

Use type hints, structured logging via `logging.getLogger`, no print(),
project conventions per CLAUDE.md. Show me the diff.

Trial text comes from `data/trials.db`. Inspect the schema first
(`sqlite3 data/trials.db ".schema trials"`) to confirm column names —
likely `nct_id`, `official_title` or `title`, `brief_summary`. Cache
the SQL lookups in-memory so the script runs in ~1 min not 10 min.
```

### Expected output shape

```jsonl
{"query_id": 105, "query": "EGFR positive lung cancer trials", "positive": "Phase II [SEP] ...", "negative": "Pilot study [SEP] ...", "label": 2.0, "doc_pos_nct_id": "NCT01...", "doc_neg_nct_id": "NCT02...", "grade_pos": 3, "grade_neg": 1}
```

### Verify

```
After running, report:
- wc -l data/training/ce_graded_train.jsonl  (expect 30K-50K)
- wc -l data/training/ce_graded_val.jsonl    (expect 5K-15K)
- cat data/training/ce_graded_metadata.json   (the summary)
- Confirm: n_train_queries + n_val_queries == 165 (or close)
- Confirm: held_out_qid_count is exactly the union of
  full_labeled_dataset.jsonl + full_labeled_dataset_expansion_v2.jsonl
  query_ids (should be 85)
- Spot-check: take 3 random train rows and 3 random val rows; verify
  `positive` and `negative` text fields are non-empty and `label` is in {1.0, 2.0, 3.0}
```

### Acceptance criteria

- Train pairs ∈ [30K, 50K]
- Val pairs ∈ [5K, 15K]
- 165 total unique queries split 80/20 (132 train + 33 val approx)
- No query_id appears in BOTH train/val AND held-out (run a probe)
- `label` distribution non-degenerate: at least 5% of pairs have label > 1.0 (otherwise all pairs are adjacent-grade and the ranking signal collapses)
- Trial text lookup cache hit rate > 95%

### Rollback

```
rm scripts/build_ce_graded_training.py
rm -f data/training/ce_graded_train.jsonl
rm -f data/training/ce_graded_val.jsonl
rm -f data/training/ce_graded_metadata.json
```

---

## Step A3 — Sanity-check graded training data (20 min)

**Goal:** Catch label-quality issues before paying for GPU time.

### Prompt to send

```
Run the four sanity checks in Phase A3 of docs/fix_CE.md against
data/training/ce_graded_train.jsonl. Report results inline. If any
check fails, STOP and tell me — do not proceed to A4.
```

### The four checks

```bash
# (a) Label (margin) distribution — expect ~60% label=1, ~30% label=2, ~10% label=3
python -c "
import json, collections
c = collections.Counter()
for line in open('data/training/ce_graded_train.jsonl'):
    c[json.loads(line)['label']] += 1
total = sum(c.values())
for k in sorted(c):
    print(f'label={k:.1f}  count={c[k]:6d}  ({100*c[k]/total:.1f}%)')
"

# (b) Pairs per query distribution
python -c "
import json, collections
c = collections.Counter()
for line in open('data/training/ce_graded_train.jsonl'):
    c[json.loads(line)['query_id']] += 1
counts = sorted(c.values())
n = len(counts)
print(f'queries: {n}')
print(f'pairs/query — min {counts[0]}  p25 {counts[n//4]}  median {counts[n//2]}  p75 {counts[3*n//4]}  max {counts[-1]}')
"

# (c) No-leakage probe — held-out queries must NEVER appear in train/val
python -c "
import json
held = set()
for fn in ['data/evaluation/full_labeled_dataset.jsonl',
          'data/evaluation/full_labeled_dataset_expansion_v2.jsonl']:
    for line in open(fn):
        held.add(json.loads(line)['query_id'])
for split in ['ce_graded_train.jsonl', 'ce_graded_val.jsonl']:
    leaked = set()
    for line in open(f'data/training/{split}'):
        qid = json.loads(line)['query_id']
        if qid in held:
            leaked.add(qid)
    print(f'{split}: {len(leaked)} leaked qids (must be 0): {sorted(leaked)[:10]}')
"

# (d) Eyeball 10 random pairs to confirm positive really looks more relevant
python -c "
import json, random
rows = [json.loads(l) for l in open('data/training/ce_graded_train.jsonl')]
random.seed(42)
for r in random.sample(rows, 10):
    print(f'Q: {r[\"query\"]}')
    print(f'   POS (label={r[\"label\"]}): {r[\"positive\"][:120]}')
    print(f'   NEG: {r[\"negative\"][:120]}')
    print()
"
```

### Acceptance criteria

- **(a)** label=1 dominates (50–70%); label=2 substantial (20–40%); label=3 non-zero (≥ 3%). If label=3 is < 1% (very few `(grade=3, grade=0)` pairs), the largest-margin training signal is sparse — flag but proceed.
- **(b)** Median pairs/query ≥ 50, max ≤ 800. Outliers > 1000 pairs/query suggest a query with > 30 graded candidates that should be capped (but don't cap unless an outlier dominates total pairs > 10%).
- **(c)** Both splits report **0 leaked qids**. Any non-zero → STOP, debug `build_ce_graded_training.py`.
- **(d)** At least 7 of 10 sampled pairs look directionally correct on eyeball (POS more on-topic than NEG). 3 wrong is acceptable noise floor for Haiku labels at κ=0.72.

### Rollback

Same as A2 (delete the JSONLs, fix the builder, rerun).

---

## Step A4 — Update `configs/training/cross_encoder.yaml` (15 min)

**Goal:** Switch the v2 training config: new loss, new output dir, new metric_for_best_model, smaller batch + LR placeholder (winner picked by Phase B2 sweep).

### Prompt to send

```
Update configs/training/cross_encoder.yaml per Phase A4 of
docs/fix_CE.md. Replace v1 contents with the v2 spec below. Show me
the full final file.
```

### Target `cross_encoder.yaml` (full file)

**Schema preserves the v1 field names** (`model.name`, `training.epochs`, top-level `loss:`, `data.train_file/val_file`) so the existing `finetune_cross_encoder.py` parser keeps working. Only the values + 1 new `evaluator:` section change.

```yaml
model:
  name: "models/cross-encoder/fine-tuned"   # warm-start source; overridden to "michiyasunaga/BioLinkBERT-base" during the cold-start sweep arm
  num_labels: 1
  max_length: 512
  output_dir: "models/cross-encoder/fine-tuned-v2"

training:
  epochs: 3                  # graded data is ~4x smaller than binary; more epochs needed
  batch_size: 32             # bumped from v1's 16 — A100 40GB has the headroom and halves wall-clock
  learning_rate: 2.0e-5      # placeholder; sweep in Phase B may overwrite
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true
  logging_steps: 25
  eval_steps: 200
  save_steps: 200
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "pooled_ndcg@10"   # NEW — was CERerankingEvaluator_ndcg@10 (saturated at 0.993)

loss:
  type: "MarginMSELoss"      # CHANGED — was BinaryCrossEntropyLoss

evaluation:
  max_val_samples: 5000      # preserved from v1; pooled evaluator caps via pool_top_k anyway

evaluator:                   # NEW — production-shaped pooled evaluator, see A5
  type: "pooled"
  pool_top_k: 20             # candidates per val query (val pool ~30/q in source files)

data:
  train_file: "data/training/ce_graded_train.jsonl"   # v2 path; v1 was train_pairs.jsonl
  val_file: "data/training/ce_graded_val.jsonl"       # v2 path; v1 was val_pairs.jsonl

inference:
  rerank_top_k: 50           # preserved from v1; production reranks top 50 candidates

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "trialmind-cross-encoder"
  run_tag: "v2-margin-mse-warmstart"
```

### Verify

```
Diff configs/training/cross_encoder.yaml vs configs/training/cross_encoder_v1.yaml.
Confirm these fields differ from v1:
1. model.name → models/cross-encoder/fine-tuned (was BioLinkBERT-base)
2. model.output_dir → models/cross-encoder/fine-tuned-v2
3. training.epochs → 3 (was 3 — unchanged, but confirm)
4. training.batch_size → 32 (was 16)
5. training.metric_for_best_model → pooled_ndcg@10
6. loss.type → MarginMSELoss (was BinaryCrossEntropyLoss)
7. data.train_file → ce_graded_train.jsonl
8. data.val_file → ce_graded_val.jsonl
9. mlflow.run_tag → v2-margin-mse-warmstart
10. NEW evaluator.type + evaluator.pool_top_k section
```

### Acceptance criteria

- 9 fields differ from v1; 1 new section (`evaluator:`)
- `output_dir` ends in `-v2`
- `loss.type: MarginMSELoss`
- All v1 sections preserved (`inference:`, `evaluation:`) — don't drop fields the script still reads

### Rollback

```
cp configs/training/cross_encoder_v1.yaml configs/training/cross_encoder.yaml
```

---

## Step A5 — Modify `scripts/finetune_cross_encoder.py` + add pooled evaluator (90 min)

**Goal:** Three coordinated changes — (1) swap loss, (2) accept warm-start init, (3) replace the saturated 1-pos/1-neg evaluator with a pooled production-shaped one.

### Prompt to send

```
Implement Phase A5 of docs/fix_CE.md. Four pieces:

1. Create src/TrialMine/evaluation/ce_pooled_evaluator.py — a custom
   CrossEncoderEvaluator subclass that, for each val query, pools
   `pool_top_k` graded candidates (from data/training/ce_graded_val.jsonl
   indexed back to per-query graded candidate lists in
   data/evaluation/*labels*.jsonl), runs CE scoring, computes
   NDCG@5/10 + MRR over the FULL pool (not just top-2), and returns
   {"pooled_ndcg@5": ..., "pooled_ndcg@10": ..., "pooled_mrr": ...}.
   Use sentence_transformers.evaluation.SentenceEvaluator as the base
   class (CE evaluators inherit from it). Type hints, no print().

2. Modify scripts/finetune_cross_encoder.py:
   (a) Replace the `from sentence_transformers.cross_encoder.losses
       import BinaryCrossEntropyLoss` line with conditional import
       based on config["loss"]["type"] (note: top-level `loss:` section,
       matching existing schema):
         - "MarginMSELoss" → from sentence_transformers.cross_encoder.losses import MarginMSELoss
         - "BinaryCrossEntropyLoss" → existing behavior (for legacy/revert)
       BEFORE the run starts, smoke-import the chosen class — if
       MarginMSELoss is not available at that path in the installed
       sentence-transformers version, FAIL FAST with a clear message
       (don't silently fall back).
   (b) Modify the dataset loader: when loss is MarginMSELoss, load
       ce_graded_train.jsonl / ce_graded_val.jsonl as a HuggingFace
       Dataset with columns `query`, `positive`, `negative`, `label`
       (these are the canonical column names MarginMSELoss for
       cross-encoders expects). Audit fields like query_id,
       doc_pos_nct_id, grade_pos can be dropped before training.
       DO NOT split into binary pairs — the loss consumes triplets.
   (c) Pass config["model"]["name"] to CrossEncoder() init — this
       supports both warm-start ("models/cross-encoder/fine-tuned") and
       cold-start ("michiyasunaga/BioLinkBERT-base") without code change.
   (d) Wire the new pooled evaluator into the trainer when
       config["evaluator"]["type"] == "pooled". Pass `pool_top_k` from
       the same section to the evaluator constructor.
   (e) Add --override KEY=VAL flag (repeatable) using dotted-path key
       resolution, same as the bi-encoder retrain. This is how the
       sweep arms override model.name + mlflow.run_tag.
   (f) Preserve the existing `evaluation.max_val_samples` subsampling
       behavior — the pooled evaluator should respect that cap.

3. Verify the trainer still saves the best model by the new metric
   ("pooled_ndcg@10") not the old one.

4. Show me each diff. No actual training run yet.
```

### Verify

```
Run:
  python scripts/finetune_cross_encoder.py --override training.epochs=0 \
                                            --override training.max_steps=1 \
                                            --dry-run     # or smallest no-op exit
Confirm:
- Config loads cleanly with the new loss.type field
- MarginMSELoss is imported from sentence_transformers.cross_encoder.losses
  (NOT from sentence_transformers.losses — that's the bi-encoder variant)
- The pooled evaluator class is instantiated
- CrossEncoder() resolves to models/cross-encoder/fine-tuned (or whatever
  model.name is set to via --override)
- --override training.learning_rate=4e-5 correctly mutates the loaded config
```

### Acceptance criteria

- All diffs reviewed and applied
- Dry-run completes without error and logs the merged config
- Pooled evaluator emits 3 metrics (pooled_ndcg@5, pooled_ndcg@10, pooled_mrr) on a smoke val
- No `print()` calls — uses `logger.info`

### Rollback

```
git checkout scripts/finetune_cross_encoder.py
git checkout configs/training/cross_encoder.yaml
rm -f src/TrialMine/evaluation/ce_pooled_evaluator.py
```

---

## Step A6 — Write `cloud_run_ce.sh` (30 min)

**Goal:** Orchestrate the unattended Lambda session: env setup, warm-vs-cold init sweep, full training with winning init, sentinel file.

### Prompt to send

```
Write Phase A6 of docs/fix_CE.md: create cloud_run_ce.sh in the repo
root that orchestrates the full CE retrain on Lambda (Ubuntu 22.04
Lambda Stack — NVIDIA driver + CUDA + PyTorch preinstalled).

Script must:
1. pip install -r requirements.txt — BUT FIRST, if requirements.txt
   pins torch/torchvision/torchaudio, drop those into a temp filtered
   file so we don't clobber Lambda Stack PyTorch. Verify with
   python -c "import torch; print(torch.cuda.is_available())" → True
   before continuing.

2. Run a quick GPU memory probe at batch_size=32, fp16. Exit if OOM.
   (Reuse scripts/oom_probe.py from the bi-encoder runbook; it accepts
   --batch and uses BioLinkBERT-base which is the same as the CE arch.)

3. Create a 5K-row sample for the sweep arms (faster than running
   the sweep on the full 50K-row train file):
     shuf -n 5000 data/training/ce_graded_train.jsonl > data/training/ce_graded_train_sample.jsonl

4. Run the INIT-CHOICE SWEEP (warm vs cold):
   for init in warm cold; do
     base_model=$([[ $init == warm ]] \
                  && echo "models/cross-encoder/fine-tuned" \
                  || echo "michiyasunaga/BioLinkBERT-base")
     python scripts/finetune_cross_encoder.py \
       --override model.name=$base_model \
       --override training.epochs=1 \
       --override training.max_steps=500 \
       --override model.output_dir=/tmp/sweep_${init} \
       --override mlflow.run_tag=ce-sweep-${init} \
       --override data.train_file=data/training/ce_graded_train_sample.jsonl
   done

5. Pick the winner: write scripts/select_ce_init.py (similar to
   select_lr.py from the bi-encoder runbook) that reads MLflow runs
   tagged ce-sweep-warm and ce-sweep-cold, picks the higher
   pooled_ndcg@10, and writes the winning value back to
   configs/training/cross_encoder.yaml under the `model.name` field.

6. Run the FULL training with the winning init (3 epochs on the full
   ce_graded_train.jsonl):
     python scripts/finetune_cross_encoder.py
   Output: models/cross-encoder/fine-tuned-v2/

7. Sentinel file: touch ~/CE_RUN_DONE after step 6 completes cleanly.
   Do NOT auto-terminate the instance — termination is manual via
   Lambda web UI.

Use set -euo pipefail. Log to cloud_run_ce.log. No hardcoded API keys.
Show me the final script.
```

### Verify

```
Show me cloud_run_ce.sh contents and confirm:
- set -euo pipefail at top
- 7 numbered steps, in order
- Sweep arms use --override training.max_steps=500 (not full 3 epochs)
- Sample creation (step 3) happens BEFORE sweep arms (step 4)
- Logs to cloud_run_ce.log
- No env vars required at runtime
```

### Acceptance criteria

- Script is self-contained and re-runnable
- No macOS-only paths
- ANTHROPIC_API_KEY is NOT referenced (no Haiku calls during training)
- MLFLOW_TRACKING_URI either uses local sqlite:///mlflow.db or is unset

### Rollback

```
rm cloud_run_ce.sh
git checkout scripts/finetune_cross_encoder.py  # if select_ce_init.py edits introduced regressions
rm -f scripts/select_ce_init.py
```

---

## Step A7 — (Optional) Sonnet re-grade of Haiku-noisy pairs (~$10, 30 min)

**Goal:** Reduce label noise on training data. SKIP this step if cost is tight; expected lift is modest (5–10% of the predicted improvement).

### Prompt to send

```
Phase A7 of docs/fix_CE.md is OPTIONAL. Skip if cost is tight.

If running: re-grade the ~2,000 pairs from train/val splits that are
most likely to be label-noisy. Use the Sonnet-as-judge prompt from
scripts/sonnet_label.py. Specifically:
1. From data/training/ce_graded_train.jsonl, find pairs where margin
   == 1.0 AND grade_pos == 1 AND grade_neg == 0 — these are the
   trickiest "almost the same relevance" pairs where Haiku is most
   likely to flip.
2. Cap at 2,000 pairs (random sample, seed=42).
3. Re-grade each (query, doc) — Sonnet outputs 0/1/2/3.
4. Update the source row's grades (and recompute margin) ONLY if
   Sonnet disagrees with Haiku.
5. Write data/training/ce_graded_train_sonnet.jsonl (full train set
   with replacements).
6. Update configs/training/cross_encoder.yaml's data.train_path to
   point at the new file.

Expected cost: ~$10 in Sonnet API. Expected wall-clock: 30 min.
Confirm before starting.
```

### Verify

```
Confirm:
- wc -l data/training/ce_graded_train_sonnet.jsonl matches original
- Distribution of margin in new file (some margins should have shifted)
- Spot-check 5 rows where Sonnet flipped Haiku's grade
```

### Acceptance criteria

- Row count unchanged from original
- At least 10% of re-graded pairs have updated margins (otherwise the re-grade was a waste — STOP and investigate prompt fidelity)

### Rollback

```
git checkout configs/training/cross_encoder.yaml  # revert data.train_path
rm data/training/ce_graded_train_sonnet.jsonl
```

---

## ✅ Phase A done — checkpoint before cloud

At this point your local state should be:
- All YAML + script edits ready (no commit yet)
- Backups of v1 CE + v1 config in place
- ce_graded_train.jsonl (~40K pairs) + ce_graded_val.jsonl (~10K pairs) built and sanity-checked
- cloud_run_ce.sh ready
- Optional Sonnet re-grade complete (or skipped)

### Prompt to send (optional)

```
Show me git status. If there are pending edits, propose a commit
message for the Phase A changes and stage the right files (configs +
scripts only — never data/training/ or models/).
```

---

# ⏸ STOP — Cloud session needed

**Provider: Lambda Cloud.** GPU: **A100 40GB SXM4** (`gpu_1x_a100_sxm4`, ~$1.79/hr) preferred — time is the primary constraint. Budget: ~$2.70 (1.5 hr × $1.79) on A100 40GB. Fallback: A100 80GB SXM4 if 40GB unavailable (~$3 for 1.5 hr at typical pricing).

Why A100 over A10 for this run:
- **Speed.** A100 is ~3× faster than A10 on BERT-base fine-tuning. Full Phase B drops from ~1.5 hr to ~1 hr. Extra $1.50 buys ~30 min wall-clock back.
- CE is BERT-base (110M params, ~430 MB fp16). 40 GB is comfortable headroom at batch=32; allows bumping to batch=64 if A100 80GB available.
- Same provider + image as the bi-encoder retrain — re-using the operational muscle memory.

Availability note: if A100 40GB shows Unavailable, try us-east-1, us-west-1, us-west-2, us-midwest-1, then `gpu_1x_a100_sxm4_80gb` if available, then fall back to `gpu_1x_a6000` ($0.80/hr, 48 GB) or `gpu_1x_a10` ($0.75/hr, 24 GB).

## Billing — same Lambda gotchas as the bi-encoder runbook

Lambda has **two states only: `Running` and `Terminated`**. There is no Stop/Pause. Re-read these four points before provisioning:

1. **Only `Terminate` stops billing.** `shutdown -h` from inside the VM, `exit` from SSH, closing the browser — none stop the meter. Click **Terminate** in the web UI: Cloud → Instances → action menu → Terminate.
2. **No pause-and-resume.** Terminate wipes the disk; you'll re-rsync code + training data next session. That's why Phase B is one unattended block.
3. **Persistent Storage bills separately** (~$0.20/GB/month) and survives termination. Don't attach one for this run.
4. **Set a calendar reminder for `now + 2 hr` titled "TERMINATE LAMBDA INSTANCE"** before disconnecting in B2. CE retrain on A100 finishes in ~1 hr; a 2-hr safety net covers the overhead + sync.

---

# Phase B — Cloud session (~1 hr unattended)

> **Re-read this whole section before starting**, then send the prompts one
> at a time. Phase B is sequential — don't parallelize.

## Step B0 — Provision Lambda instance + sync data (15 min)

This is a **manual** step — you, not Claude, do it.

### What to do

1. https://cloud.lambdalabs.com → **Instances** → **Launch instance**.
   - **Instance type:** `gpu_1x_a100_sxm4` ($1.79/hr, 40 GB VRAM). If unavailable in your default region, try us-east-1, us-west-1, us-west-2, us-midwest-1, then fall back to `gpu_1x_a100_sxm4_80gb`, `gpu_1x_a6000`, or `gpu_1x_a10`.
   - **Region:** whichever has stock.
   - **Filesystem:** do NOT attach a Persistent Storage filesystem.
   - **SSH key:** select one already registered.
   - Wait ~2 min for `Running` and copy the public IP.

2. **Set a calendar reminder for `now + 2 hr` titled "TERMINATE LAMBDA INSTANCE".**

3. From your Mac, rsync repo + training data + warm-start checkpoint (replace `<ip>`):

   ```bash
   # From /Users/tonyxu/Desktop/TrialMine
   # .env explicitly excluded — no Haiku calls during training
   rsync -avz --exclude='.env' --exclude='data/trials.db' \
              --exclude='data/faiss_*' \
              --exclude='.git/' --exclude='mlflow.db' \
              --exclude='__pycache__/' \
              --exclude='models/embeddings/' \
              ./ ubuntu@<ip>:~/TrialMine/

   # Training data (graded pairs)
   rsync -avz data/training/ce_graded_train.jsonl \
              data/training/ce_graded_val.jsonl \
              data/training/ce_graded_metadata.json \
              ubuntu@<ip>:~/TrialMine/data/training/

   # Warm-start checkpoint — REQUIRED for the warm-start arm of the sweep
   rsync -avz models/cross-encoder/fine-tuned/ \
              ubuntu@<ip>:~/TrialMine/models/cross-encoder/fine-tuned/

   # data/trials.db is NOT needed during training (text already baked into the JSONL)
   ```

### Verify (SSH in and check)

```bash
ssh ubuntu@<ip>
ls -la ~/TrialMine/data/training/    # ce_graded_train.jsonl present
ls -la ~/TrialMine/scripts/           # finetune_cross_encoder.py + select_ce_init.py
ls -la ~/TrialMine/models/cross-encoder/fine-tuned/   # model.safetensors ~430MB
nvidia-smi                            # A100-SXM4-40GB or fallback
```

### Acceptance criteria

- All scripts + configs present
- Training files present and correct size (train ~10–30 MB, val ~2–6 MB depending on pair count)
- Warm-start checkpoint present (`model.safetensors` ~430 MB)
- GPU detected

---

## Step B1 — Run `cloud_run_ce.sh` (~1 hr unattended on A100)

### What to do (manual)

```bash
# SSH'd into Lambda
cd ~/TrialMine
chmod +x cloud_run_ce.sh
nohup ./cloud_run_ce.sh > cloud_run_ce.log 2>&1 &
disown
# Confirm it's running before disconnecting:
sleep 5 && tail -20 cloud_run_ce.log && ps -p $(pgrep -f cloud_run_ce.sh)
```

Then disconnect. Come back in ~1 hour (A100) or ~1.5 hr (A10/A6000 fallback). Calendar reminder is the backstop.

To peek mid-run without interrupting:
```bash
ssh ubuntu@<ip> "tail -50 ~/TrialMine/cloud_run_ce.log; ls ~/CE_RUN_DONE 2>/dev/null && echo DONE || echo STILL_RUNNING"
```

### Verify (after ~1 hr on A100)

```bash
ssh ubuntu@<ip> '
  echo "=== 1. Sentinel ==="
  ls ~/CE_RUN_DONE 2>/dev/null && echo DONE || echo STILL_RUNNING
  echo "=== 2. Log tail ==="
  tail -40 ~/TrialMine/cloud_run_ce.log
  echo "=== 3. Sweep winner (which init was selected) ==="
  grep "selected init" ~/TrialMine/cloud_run_ce.log | tail -1
  echo "=== 4. v2 model dir ==="
  ls -lh ~/TrialMine/models/cross-encoder/fine-tuned-v2/ | head
  echo "=== 5. Process still running? ==="
  pgrep -af cloud_run_ce.sh || echo "no cloud_run_ce.sh process"
'
```

| Signal | Good |
|---|---|
| 1. Sentinel | `~/CE_RUN_DONE` exists |
| 2. Log tail | Ends with "Training complete" or similar; no Traceback / Killed / OOM / NaN |
| 3. Sweep winner | One of `warm` or `cold` logged with its pooled_ndcg@10 |
| 4. Model dir | Has `model.safetensors` (~430 MB) + `config.json` + `metadata.json` |
| 5. Process | `pgrep` returns empty |

Failure modes:
- **Sentinel missing + process gone** → run failed mid-way. Read log, DO NOT rsync over partial state.
- **Sentinel missing + process running** → just wait. B2 rsync now would race with active writes.
- **All five hold** → safe to proceed to B2.

### Healthy mid-run signals (use the one-liner every 30 min)

Expected sequence in `cloud_run_ce.log`:
1. `Phase 0`: `torch.cuda.is_available() = True`
2. `Phase 1`: OOM probe passes
3. `Phase 2`: 2 sweep runs each completing ~500 steps. Each ends with a `pooled_ndcg@10` metric.
4. `Phase 3`: `select_ce_init.py` logs "winner: warm, pooled_ndcg@10=0.XYZ"
5. `Phase 4`: Full training begins, ~4,700 total steps over 3 epochs.
6. Loss should trend down, `pooled_ndcg@10` trend up across eval checkpoints (~every 200 steps = ~24 evals).
7. End: best model saved, sentinel touched.

GPU util should be 80–100% during training, dropping to ~0 between sweep arms.

---

## Step B2 — Download artefacts back to local Mac (10 min)

### What to do

```bash
# On local Mac — replace <ip>
rsync -avz ubuntu@<ip>:~/TrialMine/models/cross-encoder/fine-tuned-v2/ \
       models/cross-encoder/fine-tuned-v2/
rsync -avz ubuntu@<ip>:~/TrialMine/cloud_run_ce.log .
rsync -avz ubuntu@<ip>:~/TrialMine/configs/training/cross_encoder.yaml \
       configs/training/cross_encoder.yaml      # picks up winning init
rsync -avz ubuntu@<ip>:~/TrialMine/mlflow.db mlflow_ce_cloud.db
```

### Verify (with Claude's help)

```
Confirm Phase B2 sync per docs/fix_CE.md. Check:
- models/cross-encoder/fine-tuned-v2/ has config.json + model.safetensors
- configs/training/cross_encoder.yaml model.base_model field reflects
  the WINNING init (either models/cross-encoder/fine-tuned for warm
  or michiyasunaga/BioLinkBERT-base for cold)
- cloud_run_ce.log shows successful completion of all 5 phases
- mlflow_ce_cloud.db has runs tagged ce-sweep-warm, ce-sweep-cold,
  and v2-margin-mse-warmstart (or -coldstart)

Report the final pooled_ndcg@10 from the v2 best-checkpoint.
```

### Acceptance criteria

- All 4 artefacts present locally
- v2 model dir has `model.safetensors` (~430 MB)
- Sweep result is in the log AND in the config

---

## Step B3 — Terminate the Lambda instance (manual, 2 min)

**Do not skip.** Same Lambda billing rules as the bi-encoder runbook.

1. https://cloud.lambdalabs.com → **Instances** → row's action menu → **Terminate** → confirm.
2. Verify the row disappears from the Instances list.
3. Wait ~10 min, open **Usage**, confirm no active instance-hours accruing.
4. If you accidentally attached Persistent Storage: **Filesystems** → Delete.
5. Dismiss the calendar reminder.

---

# Phase C — Post-training (~1 hr local)

## Step C1 — Wire v2 CE into production (15 min)

**Goal:** Make the agent pipeline + all evaluation scripts load the v2 cross-encoder.

### Prompt to send

```
Per Phase C1 of docs/fix_CE.md: switch the default cross-encoder model
path from models/cross-encoder/fine-tuned to
models/cross-encoder/fine-tuned-v2.

Two options:

(A) Symlink (lowest risk):
    mv models/cross-encoder/fine-tuned models/cross-encoder/fine-tuned-binary
    ln -sf fine-tuned-v2 models/cross-encoder/fine-tuned
    No code change. All callers continue to find the model at the
    canonical path.

(B) Update the env-default in src/TrialMine/agents/tools.py:54 (the
    TRIALMINE_CROSS_ENCODER default) to fine-tuned-v2, and update the
    hardcoded CE_MODEL paths in:
      - scripts/evaluate_cross_encoder.py
      - scripts/compare_with_ctgov.py:84
      - scripts/build_full_eval_dataset.py:203
      - scripts/build_fair_eval.py:118
      - scripts/evaluate.py:65
      - scripts/train_ranker.py:67
      - scripts/demo_reranker.py
      - scripts/ci_quality_gate.py:13

Recommend Option B (more explicit, easier to audit which scripts
target which model). Apply it. Show me each diff.

After applying, start the API locally and confirm /health returns
200 with no CE load errors. Then send a test search:
  curl -X POST http://localhost:8000/api/v1/search \
       -H "Content-Type: application/json" \
       -d '{"query":"lung cancer trials","top_k":3,"use_agent":true}'
Confirm 3 results with explanation + eligibility fields.
```

### Verify

```
1. /health returns 200
2. Test search returns 3 results
3. Logs show: CE loaded from models/cross-encoder/fine-tuned-v2
4. No regressions in tools.py lazy-loading singletons
```

### Acceptance criteria

- API + agent path use v2 CE
- All scripts updated to v2 CE path (env-overridable preferred over hardcoded)
- A test search returns results

### Rollback

```
# If Option A symlink broke something:
rm models/cross-encoder/fine-tuned
mv models/cross-encoder/fine-tuned-binary models/cross-encoder/fine-tuned

# If Option B hardcoded edits broke something:
git checkout src/TrialMine/agents/tools.py scripts/evaluate_cross_encoder.py \
             scripts/build_full_eval_dataset.py scripts/build_fair_eval.py \
             scripts/evaluate.py scripts/train_ranker.py
```

---

## Step C2 — Re-eval pure CE on 65-query held-out test set (~20 min)

**Goal:** Measure how the v2 CE performs as a pure replacement (no blender) vs the binary-CE baseline (NDCG@5 = 0.657).

### Prompt to send

```
Run Phase C2 per docs/fix_CE.md. Re-evaluate the v2 CE on the
held-out 65-query test set:

1. Use scripts/evaluate_cross_encoder.py against
   data/evaluation/full_labeled_dataset.jsonl. The script already
   supports a --model flag.
2. Also evaluate on the expansion test set
   data/evaluation/full_labeled_dataset_expansion_v2.jsonl
   (run separately and report both).
3. Compute NDCG@5, NDCG@10, MRR with bootstrap 95% CIs (1000 resamples).
4. Compare three rows side-by-side:
   - Hybrid baseline (no CE)
   - v1 binary CE (pure replacement) — historical 0.657 NDCG@5
   - v2 graded CE (pure replacement) — this run
5. Per-category breakdown across the 7+1 categories (common, rare,
   pediatric, complex, geographic, vague, treatment, rare_explicit).

Output: a markdown table similar to §3 of docs/evaluation-report.md.
Do NOT make a ship/revert decision yet — that's Step C4.
```

### Verify

```
Confirm the output table includes:
- 3 systems (hybrid, v1 CE, v2 CE) × 3 metrics × 8 categories
- Bootstrap CIs on NDCG@5 and NDCG@10
- Highlights any category where v2 CE < v1 CE - 0.02 (regression)
  or v2 CE > 0.80 (interesting strength)
```

### Acceptance criteria

- Table complete, no missing cells
- CIs are non-degenerate (non-zero width)

---

## Step C3 — Re-tune blender weights (30 min)

**Goal:** Find the new optimal `α * RRF + (1-α) * CE_sigmoid` blend now that CE is trustworthy. Old blend was 0.7/0.3.

### Prompt to send

```
Run Phase C3 per docs/fix_CE.md. Sweep blender weights α ∈ {0.3, 0.4,
0.5, 0.6, 0.7} on the held-out 65-query test set, using the v2 CE.

1. For each α, run the full pipeline against
   data/evaluation/full_labeled_dataset.jsonl (use the existing
   scripts/build_full_eval_dataset.py infrastructure, but with the
   v2 CE wired in and the RRF/CE blend weight as a CLI override).
   Note: the blend lives in src/TrialMine/models/cross_encoder.py —
   you may need to add an RRF_WEIGHT env var or a parameter to
   CrossEncoderReranker.rerank() if it's currently hardcoded.
2. Compute NDCG@5/10 + MRR + 95% CIs for each α.
3. Pick the α with the highest NDCG@5 (break ties by NDCG@10).
4. Report the table.

Do NOT change the production default yet — that's Step C5a after the
decision gate.
```

### Verify

```
Confirm:
- 5 blend ratios evaluated
- Each row has NDCG@5/10 + MRR + CIs
- The chosen α is logged
```

### Acceptance criteria

- 5 rows in the sweep table
- The new optimal α may differ from 0.7 — that's expected with a better CE
- If the optimal α is 0.7 (no change), the v2 CE did not move blender utility — flag as a partial result

---

## Step C4 — Decision gate: holistic review (45 min, manual)

**This is NOT a pre-registered pass/fail gate.** Per the bi-encoder Phase C precedent (Decision 36 — "ship anyway" with honest writeup), the ship/revert call is made by weighing overall performance across multiple signals, not by checking individual thresholds. The signals below are *diagnostic*, not pass/fail.

### Signals to assemble before deciding

Pull these numbers from C2/C3 output. Lay them out side-by-side; don't reduce to a single score.

**1. Production ranking quality** (the headline)
- Blended NDCG@5 at chosen α: v1 = 0.829, v2 = ?
- Blended NDCG@10 at chosen α: v1 = 0.817, v2 = ?
- Blended MRR at chosen α: v1 = 0.950, v2 = ?
- Bootstrap 95% CIs on each — is v2 in v1's CI? Outside? Worse?

**2. Pure CE quality** (architectural signal)
- Pure CE NDCG@5: v1 = 0.657, v2 = ?
- Pure CE NDCG@10: v1 (compute) = ?, v2 = ?
- A meaningful improvement here (e.g., +0.10 or more) means CE is now trustworthy on its own — even if the blender doesn't move, this lets us simplify the architecture in the next iteration.

**3. Val/production gap closure**
- v1: val NDCG@10 0.993 vs production blended 0.829 → 16-pt gap (or 33-pt vs pure CE)
- v2: pooled_ndcg@10 val (informative range 0.80–0.88) vs production blended → ?
- A closing gap means the val metric is now predictive — independently valuable for future iterations.

**4. Per-category breakdown** (regression check)
- For each of the 8 categories, compare v2 blended vs hybrid baseline (no CE).
- Note any category where v2 blended is more than 0.02 below hybrid — that's a real regression worth investigating.
- Note any category with a meaningful lift (+0.05 or more) — that's where v2 is buying us something specific.

**5. Blender weight winner** (diagnostic)
- Did the C3 sweep pick α ≠ 0.7? Movement in either direction is informative:
  - α < 0.7 (more CE weight) → blender trusts CE more, evidence v2 CE is calibrated.
  - α > 0.7 (less CE weight) → counterintuitive; CE somehow lost utility despite better training. Look for what happened.
  - α = 0.7 (unchanged) → CE feature quality didn't move enough to shift the optimum. Partial result.

**6. Init choice winner** (diagnostic, B1 sweep)
- Warm-start won → useful disease-prior preserved, MarginMSE refined ordering on top.
- Cold-start won → v1's disease-matcher prior really was harmful; v2 from base is the cleaner architecture.

**7. Cost-of-revert** (operational)
- Revert is one toggle (path config in tools.py). No data loss. Future iterations can resume from v2 artefacts at any time.

### How to decide

Lay all seven signals out. Apply judgment:

- **Ship if the overall picture is positive.** Examples: blended moves up by +0.02 or more, pure CE materially improves, no per-category regression worse than -0.02. You can also ship if blended is TIED with v1 but pure CE jumped meaningfully — the architectural win (a CE that's actually a ranker) is valuable even at parity.
- **Ship anyway with honest writeup** if the headline is tied but there are real per-category wins, the val metric is now informative, or the architecture is cleaner — and nothing catastrophic regressed. This is the Decision 36 path; document it transparently in `docs/evaluation-report.md`. Don't move goalposts after the fact; just be honest about what moved and what didn't.
- **Revert** if production ranking actually got worse (blended drops by 0.02 or more) AND there's no compensating architectural win. Or if a critical category (common, rare, treatment) regressed by 0.05 or more — those are the easy-mode categories; failure there means the new CE introduced a real defect.
- **Run one more lever before deciding** if the picture is mixed AND a known cheap intervention exists. Concretely: if pure CE NDCG@5 < 0.75, the Haiku label noise is the likely ceiling — run the optional Phase A7 Sonnet re-grade (~$10 + 30 min) and retry training before reverting.

### Prompt to send (after laying out the signals)

```
My decision after the Phase C4 holistic review of docs/fix_CE.md is:
[SHIP | SHIP-ANYWAY-with-writeup | REVERT | RUN-A7-AND-RETRY].

Assembled signals:
- Blended NDCG@5: v1=0.829 → v2=<number> (Δ=<delta>, in CI: YES/NO)
- Blended NDCG@10: v1=0.817 → v2=<number> (Δ=<delta>)
- Pure CE NDCG@5: v1=0.657 → v2=<number> (Δ=<delta>)
- Val/prod gap: v1=16pt → v2=<number>pt
- Per-category lifts/regressions: <enumerate any > ±0.05>
- Blender α winner: <0.3|0.4|0.5|0.6|0.7>
- Init winner: <warm|cold>

Reasoning: <2-3 sentences on the overall picture>

Proceed to Step C5 with this decision.
```

---

## Step C5a — SHIP path (if decision was SHIP or SHIP-ANYWAY-with-writeup) (~30 min)

### Prompt to send

```
Ship v2 CE per Phase C5a of docs/fix_CE.md. If this is a
SHIP-ANYWAY-with-writeup path, be explicit in the docs about what
moved and what didn't — don't paper over the partial result.

1. Update src/TrialMine/agents/tools.py default TRIALMINE_CROSS_ENCODER
   to "models/cross-encoder/fine-tuned-v2" (already done in C1 if
   Option B was applied — verify).

2. Update src/TrialMine/models/cross_encoder.py:
   - Set the blender default α to the C3 winner
   - Add a docstring noting the v2 retrain motivation + new blend ratio

3. Update docs/evaluation-report.md with a new section §10 "Phase 11
   — Cross-encoder retrain (v2)" that includes:
   - Motivation (the 33-pt val/production gap, blender α=0.3 ceiling)
   - Methodology (graded pairs, MarginMSE, warm-vs-cold sweep, pooled evaluator)
   - All 7 holistic-review signals from C4 laid out in a table
   - Three-way comparison table from Step C2 (hybrid / v1 CE / v2 CE)
   - Blender α sweep table from Step C3
   - Per-category NDCG@5/10 with bootstrap 95% CIs
   - Honest "what this fixed" (val/prod gap, blender ceiling if α moved,
     aggregate ranking quality) and "what this didn't fix" (Q413/Q416
     complex eligibility failures — those need parser/eligibility work,
     per Issue #5's own honest prediction)
   - If this is a SHIP-ANYWAY path: explicit acknowledgment of what
     didn't move and why we shipped anyway, mirroring §8's Decision 36
     framing for the bi-encoder v2.

4. Update CLAUDE.md:
   - Bump the "Cross-encoder re-ranker" entry in "What's working" to v2
     (graded pairs, MarginMSE, warm-started)
   - Mark Issue #5 closed in the "What's next" optional list
   - Add Decision 39: "Cross-encoder trained on graded pairs with
     MarginMSELoss. Why: binary-trained CE converged to a disease/
     intervention matcher, val metric saturated at 0.993 while
     production pure-CE NDCG@5 = 0.657; switching to graded pairs
     closed the val/prod gap from <X> pts to <Y> pts and lifted
     blended NDCG@5 from 0.829 to <new>. How to apply: graded labels
     exist scattered across 4 JSONL files — see
     scripts/build_ce_graded_training.py for the build recipe + the
     80/20 query-level split that holds out full_labeled_dataset.jsonl.
     Init choice (warm-start from v1 binary CE vs cold-start from base
     BioLinkBERT) was settled empirically by the Phase B sweep — winner
     was <warm|cold>. Decision-gate framework: holistic review across
     7 signals (blended NDCG, pure CE NDCG, val/prod gap, per-category,
     blender α, init winner, cost-of-revert), not rigid pre-registered
     thresholds — same framing as Decision 36 for bi-encoder v2."

5. Preserve v1 at models/cross-encoder/fine-tuned-v1/ (don't delete).
   Update its metadata.json with a "superseded_by_v2": true field
   for archival clarity.

6. Run the existing test suite:
   make test  # or pytest tests/
   Confirm all CE-touching tests still pass.

7. Commit on a feature branch with three logical commits:
   (a) code (tools.py, cross_encoder.py)
   (b) eval artefacts (full_labeled_dataset re-eval JSONLs if any)
   (c) docs (evaluation-report.md, CLAUDE.md)
   Open a PR titled "feat(ce): v2 graded MarginMSE retrain (+<delta>
   blended NDCG@5)" with a clear summary of the metric deltas.

Show me each diff before committing.
```

---

## Step C5b — REVERT path (if decision was REVERT) (~20 min)

### Prompt to send

```
Revert v2 CE per Phase C5b of docs/fix_CE.md:

1. Restore v1 CE as canonical:
   - Update src/TrialMine/agents/tools.py default to
     "models/cross-encoder/fine-tuned" (the binary-trained checkpoint)
   - Revert any hardcoded CE_MODEL paths edited in Step C1
   - rm -rf models/cross-encoder/fine-tuned-v2  (or keep for archival)
   - Restore configs/training/cross_encoder.yaml from cross_encoder_v1.yaml

2. Keep these archived in the repo (do NOT delete):
   - data/training/ce_graded_train.jsonl
   - data/training/ce_graded_val.jsonl
   - data/training/ce_graded_metadata.json
   - scripts/build_ce_graded_training.py
   - scripts/select_ce_init.py
   - src/TrialMine/evaluation/ce_pooled_evaluator.py
   - cloud_run_ce.sh
   - cloud_run_ce.log
   - mlflow_ce_cloud.db
   These are still useful future-self artefacts.

3. Write a "lessons learned" entry under Issue #5 in
   docs/things_can_be_fixed.md:
   - What was tried (graded pairs + MarginMSE + warm-start)
   - What didn't lift (specific metrics)
   - What the next intervention should be (Sonnet re-grade A7 is the
     likely top suspect if pure-CE NDCG@5 was in 0.65-0.75 range;
     cold-start retry if 0.75-0.80; multi-vector encoding Issue #4
     if all CE-only fixes plateau)

4. The Phase A YAML + script edits are still valid improvements —
   leave them on the branch even if reverting. Only revert if you
   want a fully clean rollback.

Show me each step before executing.
```

---

# Appendix — Failure modes to specifically watch for

| Symptom | Likely cause | Fix |
|---|---|---|
| Leaked qids in A3 check | bug in build_ce_graded_training.py held-out logic | re-edit script; rerun A2 |
| label=3 < 1% in A3a | grade distribution skewed (e.g., few grade=3 docs in training pool) | normal for some splits — proceed but flag the limitation in writeup |
| `ImportError: cannot import name 'MarginMSELoss' from 'sentence_transformers.cross_encoder.losses'` | Older sentence-transformers version doesn't expose CE MarginMSELoss yet | upgrade sentence-transformers to ≥ 3.1, OR fall back to wrapping the bi-encoder MarginMSELoss (`from sentence_transformers.losses import MarginMSELoss`) — but verify the cross-encoder model API is compatible before training |
| OOM in B1 OOM probe | unlikely on A100 40GB at batch=32, but possible on A10 fallback | drop batch_size override to 16; rerun |
| Sweep winner is `cold` not `warm` | Warm-start did lock in disease-matcher bias | proceed with cold (the sweep made the right call); document in C5a writeup that the v1 prior was harmful |
| `pooled_ndcg@10` not improving past 0.7 in mid-training | learning rate too low for warm-start, or training data too noisy | first try `--override training.learning_rate=4e-5`; if still stuck, run A7 Sonnet re-grade and retry |
| Pure CE NDCG@5 < 0.70 in C2 | Haiku label noise is the ceiling | C4 holistic review may say "run A7 then retry" — that's the cheap escape hatch before reverting |
| Pure CE NDCG@5 high but blended did not improve | Blender weights need full LightGBM retrain on v2 CE features | defer LightGBM retrain to next iteration; in C4 this is one of the "ship anyway" patterns — the architectural win is real even if the blender headline hasn't moved |
| API /health 500 after C1 | tools.py singleton cache holds old path | restart API; check log for actual loaded CE path |
| All v2 numbers equal v1 numbers exactly | API still loading v1 CE | restart API, verify path in startup log |

---

# Appendix — Cost recap

| Item | Cost | Phase |
|---|---|---|
| Lambda A100 40GB SXM4 — 1.5 hr | ~$2.69 | B |
| (Fallback A100 80GB — 1.5 hr, if available) | ~$3.00 | B |
| (Fallback A10 — 1.5 hr, slower) | ~$1.13 | B |
| Sonnet API — optional re-grade 2K pairs | ~$10 | A7 (optional) |
| Haiku API — re-label after blender re-tune | $0 (existing labels suffice) | C2/C3 |
| **Total (A100 happy path)** | **~$2.69** | |
| **Total (with optional A7 re-grade)** | **~$13** | |
| **Total (A10 fallback, +30 min slower)** | **~$1.13** | |

Plus ~4 hr engineer wall-clock (mostly local; cloud run is unattended).

---

# Appendix — Why warm-start is the default (revisit if it fails)

The warm-start hypothesis:
- v1 binary CE saw 200K pairs and learned reliable disease/intervention discrimination (lexical features that work in production — the 0.7/0.3 blender confirms this).
- v2 graded training has ~50K pairs — 4× smaller. Cold-starting from base BioLinkBERT might not see enough disease-discriminating signal to relearn it.
- MarginMSE on top of v1 inherits the disease-discrimination prior, then refines ordering — a smaller learning task on top of useful initialisation.

Counter-hypothesis (the "cold-start could win" case):
- v1's disease-matcher prior IS the bug Issue #5 calls out. Warm-starting could lock in the overconfident binary scoring that the new MarginMSE objective is trying to fix.
- If the cold arm wins the B1 sweep, that's evidence the prior is harmful — take it seriously.

The B1 sweep settles this empirically with ~$0.50 of compute. Don't pre-commit; let the data decide.

---

# Appendix — What this fix does NOT address (be honest in C5a writeup)

Per Issue #5's own honest prediction:

> *"Failure modes that won't be fixed by this alone: queries whose intent lives in eligibility text (Q413 / Q416 from Week 9) — that's Issue #4 (multi-vector encoding). CE on full-trial text still can't see fields the bi-encoder didn't surface."*

The CE only re-ranks candidates the bi-encoder surfaces. If the bi-encoder + BM25 don't bring a relevant trial into the top ~50, the CE can't recover it. Specifically:
- **Q413 (failed osimertinib)**: candidate set already contains osimertinib-failed-eligible trials AND osimertinib-required-naive trials. Graded CE may learn to demote the latter — but only if training data has enough such pairs.
- **Q416 (post-trastuzumab progression)**: same shape.
- These remain best-fixed by the parser quality + eligibility filter work (UMLS upgrade path, the current top open task in CLAUDE.md).

Ship v2 CE for what it actually improves (val/prod gap + blender ceiling + aggregate ranking quality). Don't oversell it as a complex/vague fix.

---

# Appendix B — Project state going into CE retrain

> Snapshot of completed work the CE retrain builds on top of, plus
> the open work the CE retrain will NOT fix but should be planned in
> parallel. Read this once before starting Phase A so you know which
> pieces of the stack are already in motion and which are still
> waiting.

## B.1 — Bi-encoder v2 retrain (Phase 10, Week 10, ~$30, SHIPPED)

**What we did.** Followed `docs/fix_bi-encoder.md` end-to-end across Phases A–C:

- **Phase A (local prep).** Regenerated training data — 10K synthetic queries via 6 SHAPE_PROMPTS (`failed_treatment`, `post_progression`, `biomarker`, `multi_constraint`, `vague`, `caregiver`) rotated round-robin. Expanded cancer taxonomy to **31 keys** with sarcoma split into 4 sub-buckets (`bone`, `pediatric`, `soft_tissue`, `other`) and 6 new first-class types (medulloblastoma, Wilms, GIST, neuroendocrine, biliary, MDS). Added per-group floor of **300 trials** for rare cancers with with-replacement sampling when corpus < floor.
- **Phase B (cloud).** Trained on Lambda Cloud A100-40GB SXM4, 5 hr wall-clock, batch=64, lr=2.8e-5, 3 epochs, fp16, MultipleNegativesRankingLoss scale=20. Best checkpoint at step 30000 (epoch 2.7). Final val NDCG@10 = **0.5012** (v1 was 0.4919 on a different val set).
- **Phase C (post-training).** Re-labeled 65-query benchmark with Claude Haiku against the v2 pipeline. **Expanded eval to n=15** on complex + vague when bootstrap CIs from n=5 were too wide to decide the gate — added queries IDs 600–619.

**What shipped vs what didn't.** Pre-registered C4 gates **FAILED**:
- complex NDCG@5: 0.526 (gate ≥ 0.68) ❌
- vague NDCG@5: 0.673 (gate ≥ 0.74) ❌

But the picture was nuanced:
- Apples-to-apples on 80 shared queries: **v1 = 0.798 vs v2 = 0.795 NDCG@5** — practically tied.
- Real per-category wins: **vague +0.048**, **geographic +0.063**, **rare_explicit 0.747** (new slice, passes ≥0.55 floor).
- No regressions on common/rare/treatment floors.

**Decision: SHIP v2** per Decision 36 ("ship anyway with honest writeup"). Reasoning baked into `docs/evaluation-report.md §8`: overall is tied (no cost to ship), complex/vague failures diagnosed as **pipeline-logic gaps** (eligibility-filter not used as hard retrieval filter), not embedding gaps — so reverting trades one tied headline for the same tied headline. The honest writeup is in the report; transparency over rigid gate-following.

**Production artefacts (what the CE retrain inherits):**

| Artefact | Path | Size |
|---|---|---|
| v2 bi-encoder | `models/embeddings/fine-tuned-v2/` | ~430 MB |
| v2 FAISS index | `data/faiss_finetuned_v2.index` + `.json` | 412 MB |
| v1 preserved | `models/embeddings/fine-tuned-v1/` + `data/faiss_finetuned_v1.{index,json}` | revert path |
| v2 training data | `data/training/train_pairs.jsonl` (714K rows) | 1.4 GB |
| Default resolution | `src/TrialMine/agents/tools.py:53` env-overridable constants `TRIALMINE_EMBEDDER` / `TRIALMINE_FAISS_INDEX` / `TRIALMINE_FAISS_MAPPING` | code |

**Why this matters for CE work.** The CE re-ranks candidates the bi-encoder surfaces. v2 bi-encoder changed which candidates land in the top-50, so the CE training labels (collected against pipeline output) reflect v2's candidate distribution. **Always re-eval the CE against the v2 pipeline** — not against v1's. The held-out `full_labeled_dataset.jsonl` already reflects v2 (re-labeled in Phase C2 of the bi-encoder retrain).

**Decisions logged.** 33 (6-shape SHAPE_PROMPTS), 34 (per-group floor + with-replacement), 35 (31-key taxonomy + sarcoma split), 36 (ship-anyway framing), 37 (expanding the eval to n=15 when CIs are too wide).

---

## B.2 — Eligibility hard-filter (Phase 10b, Week 10, ~$0.60, SHIPPED)

**What we did.** Wired the parsed `EligibilityProfile` into the orchestrator's retrieval path as a hard post-RRF filter:

- **Shared module:** `src/TrialMine/features/eligibility_filter.py` (~95 LOC). Owns the "what counts as a hard mismatch" rule. Imported by both production (orchestrator Step 5b) and the offline eval script (`scripts/build_full_eval_dataset.py --apply-eligibility-filter`) so the live rule and the measurement always agree.
- **Production wiring:** `src/TrialMine/agents/orchestrator.py` Step 5b. After eligibility checks run on ALL ranked results (not just top-K), the filter drops trials whose verdict is hard-Unmet on closed-vocab criteria.
- **Toggle:** `DegradationConfig.eligibility_hard_filter_enabled` (default `True`). Lives in `src/TrialMine/config.py`. One flip to A/B or revert.
- **Safety net:** if the filter would empty the result list entirely, the unfiltered list is returned with `safety_net_triggered=True` logged.

**The critical design choice (Decision 38):** filter only fires on `HARD_FILTER_CRITERIA = ("age", "sex", "excluded_prior_treatments")` — the three closed-vocab criteria with > 95% parser precision. The other criteria (`required_conditions`, `required_prior_treatments`) are noisy:
- `required_conditions` is SciSpacy NER output (~70% precision) — false-matches "Histologically", "signed", "clinical trial" as conditions.
- `required_prior_treatments` uses literal string match — misses drug-class equivalences (`"osimertinib"` ≠ `"systemic therapy"` ≠ `"EGFR TKI"`).

Filtering on the overall verdict (first naive design) dropped 10 of 10 trials for Q413 in the smoke test — too aggressive. The closed-vocab whitelist is the right precision/recall trade.

**Measured impact** (30 queries: 15 complex + 15 vague, ~$0.60 Haiku re-label):

| Slice | NDCG@5 before | NDCG@5 after | Δ |
|---|---|---|---|
| Complex (aggregate, n=15) | 0.526 | 0.519 | −0.007 (within CI noise) |
| Vague (aggregate, n=15) | 0.673 | 0.668 | −0.005 (within CI noise) |
| Q607 (55F ovarian BRCA-WT PARP-failure) | 0.513 | 0.614 | **+0.101** |
| Q609 (8M pediatric B-ALL post-CAR-T) | 0.745 | 0.820 | **+0.075** |
| Q608 (filter triggered, 1 trial dropped) | — | — | +0.007 |
| Q606 (filter triggered, 1 trial dropped) | — | — | +0.000 |

**Filter trigger rate: 4 of 30 queries (13%).** Why so low:
- **Vague queries trigger 0%** — `QueryParserAgent` (Haiku) doesn't extract `age` or `sex` from caregiver phrasings like "my dad has cancer". Without those fields, age/sex filters can't fire.
- **`required_prior_treatments` does literal string match** — misses Q413 (failed osimertinib → drop osimertinib-naive trials) and Q416 (post-trastuzumab progression).

**Production artefacts (what the CE retrain inherits):**

| Artefact | Path |
|---|---|
| Shared filter module | `src/TrialMine/features/eligibility_filter.py` |
| Orchestrator wiring | `src/TrialMine/agents/orchestrator.py` (Step 5b) |
| Toggle | `DegradationConfig.eligibility_hard_filter_enabled` in `src/TrialMine/config.py` |
| Eval re-label artefacts | `data/evaluation/full_labeled_complex_vague_filtered_v2.jsonl` |
| Writeup | `docs/evaluation-report.md §9` |

**Why this matters for CE work.** The CE sees whatever candidates survive the filter. If the filter drops a trial that the CE would have re-ranked downward anyway, no harm. If the filter drops a trial the CE would have surfaced, the user never sees it. **For Phase C2 CE eval, run against the same held-out test set with the filter ON** (it's on by default; just don't disable it) — that matches production behaviour.

**Decision logged.** 38 (closed-vocab criteria only).

---

## B.3 — Things I think we must fix (post-CE)

Ordered by leverage (impact ÷ effort), not severity. The CE retrain is **not on this list** because you're already doing it.

### 1. Parser quality: UMLS via SciSpacy `EntityLinker` (Decision 18 upgrade path) — HIGHEST LEVERAGE

**Why must:** the eligibility filter is starved by parser noise. It triggers on 13% of queries today; the structural ceiling without parser improvements is maybe 25–30%. Adding UMLS-backed drug-class matching would let `required_prior_treatments` join the hard-filter whitelist, which would catch Q413/Q416 — the canonical complex failures Phase 10 couldn't solve with embeddings alone.

**What it unlocks:**
- Extends `HARD_FILTER_CRITERIA` in `eligibility_filter.py` to include `required_prior_treatments`
- Drug-class equivalence: `osimertinib` → `EGFR TKI` → matches trials excluding "EGFR-TKI naive"
- Likely +0.05–0.10 NDCG@5 on complex (the gate Phase 10 missed at 0.526 vs ≥0.68)

**Cost:** ~4–8 hr. SciSpacy `EntityLinker` integration in `src/TrialMine/features/concepts.py`, parser precision re-validation on Q413/Q416, focused re-eval (~$0.60).

**Sequence with CE:** parallel-safe. Touches `src/TrialMine/features/` not `models/cross-encoder/`. Could run during CE cloud-training wait if the deps don't conflict.

### 2. Caregiver-phrasing patient-profile extraction in `QueryParserAgent`

**Why must:** vague queries hit 0% eligibility-filter trigger rate today because Haiku doesn't extract `age` or `sex` from queries like "my dad has cancer" or "trials for my 4-year-old daughter". That's a structural blocker — the filter has nothing to filter on.

**What it unlocks:** vague queries (currently the weakest slice at NDCG@5 = 0.668) get the same filter coverage as complex queries.

**Cost:** ~1–2 hr. Prompt tweak in `src/TrialMine/agents/query_parser.py` along the lines of *"if relationship language implies patient age, infer age=<bucket>; if pronouns imply sex, infer sex=<value>; mark `inferred: true` so the orchestrator can downweight if needed"*. Risk: confidently-wrong inferences — Decision 19's `Unknown over false Met` principle applies. Mitigation: emit a per-field `confidence` annotation.

**Sequence with CE:** parallel-safe. Touches `agents/query_parser.py`.

### 3. Centralise path constants in `src/TrialMine/config.py`

**Why must:** Phase 10's evaluation surfaced 16+ scripts with hardcoded copies of `models/cross-encoder/fine-tuned`, `data/faiss_finetuned*`, embedder paths, etc. Adding env-var support to `build_full_eval_dataset.py` mid-Phase-C was a band-aid. Every model version (v2, v3, …) re-paying this tax.

**What it fixes:** one Settings module (already exists at `src/TrialMine/config.py`, just sparsely populated). All scripts import from it. Migration is incremental.

**Cost:** ~2–3 hr to add the Settings fields + migrate the 16 scripts found by `rg "models/cross-encoder/fine-tuned" scripts/`.

**Sequence with CE:** **do BEFORE Phase C1** of the CE retrain. C1 currently asks the implementer to edit 8 separate hardcoded paths. If we centralise first, C1 becomes a 1-line config change.

### 4. Wire v2 bi-encoder into the legacy `api/app.py` non-agent path

**Why must:** the legacy `use_agent=false` route in `src/TrialMine/api/app.py` still loads `data/trial_embeddings.faiss` + off-the-shelf BioLinkBERT (not v2). Low traffic — only the legacy non-agent route hits it — but it's an inconsistency a careful reviewer will catch and ask about.

**What it fixes:** one PR replacing the lifespan loader's hardcoded paths with the same env-overridable constants `tools.py` uses.

**Cost:** ~30 min.

**Sequence with CE:** can fold into the centralised-config work (item 3) — same touch area.

### 5. Extract eligibility matcher from `tools.py:check_trial_eligibility` into `src/TrialMine/features/eligibility_matcher.py`

**Why must (softer):** Decision 19's "checker module" was supposed to live as its own file. Today the matching logic is inside `tools.py:check_trial_eligibility` and the orchestrator calls it via `.invoke({...})` + `json.loads()`. Awkward indirection that blocks standalone unit tests.

**What it fixes:** removes the LangChain `@tool` wrapper from the hot path, lets the matcher be unit-tested without a fake tool registry.

**Cost:** ~2 hr.

**Sequence with CE:** parallel-safe. Defer if not needed.

### What's NOT on the list (deliberately)

- **Multi-vector encoding (Issue #4):** real, but expensive (days of work). The right next big bet AFTER UMLS parser + CE retrain land — by then we'll have evidence about whether the remaining gap is multi-field-text-coverage (Issue #4) or something else.
- **LightGBM blender retrain on v2 CE features:** wait until CE retrain ships and we know if α moved. May not even be needed if Phase C3 sweep finds a useful new α.
- **Human-anchored kappa (Sonnet vs human reviewer on disagreements):** ~2 hr of clinical reviewer time + scheduling. Deferred until we have a clinical advisor; the current Sonnet-as-judge kappa is interview-defensible per Decision 32.
- **Streamlit UI for new agent response shape:** real product debt but not blocking any backend work.

### How to read this list before starting CE

If you have time-and-attention budget for **only one thing besides CE**, do **item 3 (centralise path constants)** first — it makes Phase C1 of the CE retrain dramatically cleaner, costs 2–3 hr, and pays back on every future model version.

If you have budget for **two things**, add **item 2 (caregiver-phrasing extraction)** — it's the cheapest unblocker for vague queries and a credible "I followed the eligibility-filter measurement to its next bottleneck" story.

Item 1 (UMLS) is the highest-leverage but also the highest-effort. Save it for after CE ships, when you have a clean baseline to measure parser improvements against.
