#!/usr/bin/env bash
# cloud_run_ce.sh — orchestrate Phase B (Lambda Cloud GPU) for the v2 CE retrain.
#
# Assumes: Lambda Cloud Ubuntu 22.04 with Lambda Stack (NVIDIA driver,
# CUDA 12.x, PyTorch preinstalled at the system level, matched to host CUDA).
# Repo synced to ~/TrialMine via rsync; the four graded JSONL inputs already
# under data/training/; the warm-start checkpoint at
# models/cross-encoder/fine-tuned/. .env is NOT required — no Haiku calls
# during training.
#
# Logs everything to cloud_run_ce.log. Writes ~/CE_RUN_DONE on full success.
# Does NOT auto-terminate the instance — termination is manual via the Lambda
# web UI (Phase B3 of docs/fix_CE.md) so logs and partial artefacts survive
# any failure.

set -euo pipefail

cd "$(dirname "$0")"

LOG_FILE="cloud_run_ce.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }

# Pin MLflow's tracking store explicitly so select_ce_init.py (a sibling
# Python process) sees the same database the training run wrote to. The
# training script sets this via os.environ in-process, but that doesn't
# propagate to sibling processes — only an export here covers both.
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

log "===== cloud_run_ce.sh start (pid=$$, host=$(hostname)) ====="
log "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

# ----------------------------------------------------------------------
# Step 1: install deps WITHOUT clobbering Lambda Stack PyTorch.
# Source of truth is pyproject.toml — we extract its dependencies block,
# strip torch family, and pip-install the rest so Lambda Stack's CUDA-
# matched torch build stays in place. Mirrors cloud_run.sh (bi-encoder).
# ----------------------------------------------------------------------
log "===== Step 1: pip install (torch filtered out, from pyproject.toml) ====="

FILTERED_REQ="/tmp/requirements_filtered.txt"

awk '/^dependencies = \[/{f=1;next} /^\]/{f=0} f' pyproject.toml \
    | grep -E '^[[:space:]]*"' \
    | sed -e 's/^[[:space:]]*"//' -e 's/",*$//' \
    | grep -v -E '^(torch|torchvision|torchaudio)([[:space:]]*[<>=!~]|$)' \
    > "$FILTERED_REQ"

log "Filtered dependency list ($(wc -l < "$FILTERED_REQ") packages):"
sed 's/^/  /' "$FILTERED_REQ"

python -m pip install -r "$FILTERED_REQ"

# Force-upgrade system-pinned packages that conflict with newer pip-installed
# deps. Lambda Stack apt-installs these at /usr/lib/python3/dist-packages;
# pip respects them as "already satisfied" so we explicitly --upgrade to
# override via ~/.local/lib/python3.10/site-packages (which wins on import).
# Known conflict: Pillow 9.0.1 missing PIL.Image.Resampling that transformers
# 5.x requires (added in Pillow 9.1). Same fix as the bi-encoder retrain.
python -m pip install --upgrade pillow jinja2 markupsafe

# Install the TrialMine package itself (no deps — pip already resolved them).
# --ignore-requires-python lets us install on Lambda Stack's Python 3.10 even
# though pyproject says >=3.11; the training scripts don't use 3.11-only syntax.
# Non-editable install (`pip install .` not `-e .`) — Lambda Stack's setuptools
# 59.6.0 is too old for PEP 660. This is load-bearing: finetune_cross_encoder.py
# imports TrialMine.evaluation.ce_pooled_evaluator, which needs the package on
# sys.path system-wide.
python -m pip install . --no-deps --ignore-requires-python

log "Verifying CUDA availability..."
cuda_ok=$(python -c "import torch; print(torch.cuda.is_available())")
if [[ "$cuda_ok" != "True" ]]; then
    log "ERROR: torch.cuda.is_available() returned '$cuda_ok' (expected True). Aborting."
    exit 1
fi
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}')"
log "CUDA available — pip install OK."

# ----------------------------------------------------------------------
# Step 2: OOM probe at batch=32, fp16. The CE config uses `max_length`
# (not `max_seq_length`) so we point oom_probe.py at the bi-encoder YAML;
# the underlying model (BioLinkBERT-base) is the same architecture, so the
# memory profile transfers. Exits 1 on OOM.
# ----------------------------------------------------------------------
log "===== Step 2: OOM probe (batch=32, fp16) ====="
python scripts/oom_probe.py --batch 32 --config configs/training/embeddings.yaml

# ----------------------------------------------------------------------
# Step 3: 5K-row sample for the init-choice sweep. shuf gives a random
# (non-deterministic) sample — fine here, because the warm and cold arms
# train on the SAME sample (read once below), so the comparison is fair.
# If you re-run the sweep and want exact reproducibility, replace shuf with
# `head -n 5000` (deterministic but biased toward early-emitted pairs).
# ----------------------------------------------------------------------
log "===== Step 3: build 5K-row sweep sample ====="
shuf -n 5000 data/training/ce_graded_train.jsonl \
    > data/training/ce_graded_train_sample.jsonl
log "  sample rows: $(wc -l < data/training/ce_graded_train_sample.jsonl)"

# ----------------------------------------------------------------------
# Step 4: INIT-CHOICE SWEEP — warm-start (v1 CE) vs cold-start (BioLinkBERT-base).
# 500 steps each on the 5K-row sample. Output dirs are throwaway under /tmp;
# real artefacts come from the full run in Step 6.
# ----------------------------------------------------------------------
log "===== Step 4: init-choice sweep (warm vs cold, max_steps=500 each) ====="
for init in warm cold; do
    if [[ "$init" == "warm" ]]; then
        base_model="models/cross-encoder/fine-tuned"
    else
        base_model="michiyasunaga/BioLinkBERT-base"
    fi

    log "----- sweep init=$init  base_model=$base_model -----"
    python scripts/finetune_cross_encoder.py \
        --override model.name="$base_model" \
        --override training.epochs=1 \
        --override training.max_steps=500 \
        --override model.output_dir="/tmp/sweep_${init}" \
        --override mlflow.run_tag="ce-sweep-${init}" \
        --override data.train_file=data/training/ce_graded_train_sample.jsonl
done

# ----------------------------------------------------------------------
# Step 5: pick the winning init. select_ce_init.py reads the two MLflow
# runs, picks the higher pooled_ndcg@10, and overwrites the `model.name`
# field in configs/training/cross_encoder.yaml in-place. The log line will
# contain the literal phrase `selected init:` so the Phase B1 verify
# one-liner can grep it.
# ----------------------------------------------------------------------
log "===== Step 5: select winning init ====="
python scripts/select_ce_init.py
log "model.name now committed to configs/training/cross_encoder.yaml:"
grep -E '^\s*name:' configs/training/cross_encoder.yaml | head -1

# ----------------------------------------------------------------------
# Step 6: full training — 3 epochs on the full ce_graded_train.jsonl,
# with the winning init now baked into the YAML. Output goes to
# models/cross-encoder/fine-tuned-v2/ per the config's `model.output_dir`.
# ----------------------------------------------------------------------
log "===== Step 6: full training (3 epochs, ~15-30 min on A100 40GB) ====="
python scripts/finetune_cross_encoder.py

# ----------------------------------------------------------------------
# Step 7: success sentinel — visible at a glance over SSH. The Phase B1
# verify one-liner checks for ~/CE_RUN_DONE to confirm clean completion.
# ----------------------------------------------------------------------
touch ~/CE_RUN_DONE
log "===== ALL STEPS PASSED. Sentinel: ~/CE_RUN_DONE ====="
log "Next manual steps (see docs/fix_CE.md Phase B2 / B3):"
log "  1. From LOCAL, rsync back: cloud_run_ce.log,"
log "     models/cross-encoder/fine-tuned-v2/, configs/training/cross_encoder.yaml,"
log "     and mlflow.db → mlflow_ce_cloud.db."
log "  2. THEN terminate the Lambda instance from the web UI."
log "     Do NOT terminate before pulling — billing only stops on terminate."
