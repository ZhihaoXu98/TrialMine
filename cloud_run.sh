#!/usr/bin/env bash
# cloud_run.sh — orchestrate Phase B (Lambda Cloud GPU session).
#
# Assumes: Lambda Cloud Ubuntu 22.04 with Lambda Stack (NVIDIA driver,
# CUDA 12.x, PyTorch preinstalled at the system level, matched to the host
# CUDA build). Repo synced to the instance, .env present in repo root,
# data/training/{train,val}_pairs.jsonl already rsync'd up from local.
#
# Logs everything to cloud_run.log. Writes ~/CLOUD_RUN_DONE on full success.
# Does NOT auto-terminate the instance — termination is a manual click in
# the Lambda web UI (see Phase B4 of docs/fix_bi-encoder.md) so we don't
# lose logs or partial artefacts on failure.

set -euo pipefail

cd "$(dirname "$0")"

LOG_FILE="cloud_run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }

log "===== cloud_run.sh start (pid=$$, host=$(hostname)) ====="

# ----------------------------------------------------------------------
# Step 1: install deps WITHOUT clobbering Lambda Stack PyTorch
# Source of truth is pyproject.toml — we extract its dependencies block,
# strip torch family, and pip-install the rest so Lambda Stack's CUDA-
# matched torch build stays in place.
# ----------------------------------------------------------------------
log "===== Step 1: pip install (filtered torch out, from pyproject.toml) ====="

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
# deps. Lambda Stack apt-installs these at /usr/lib/python3/dist-packages, which
# pip respects as "already satisfied" — so we explicitly --upgrade to override
# them via ~/.local/lib/python3.10/site-packages (which wins in Python's
# import precedence). Known conflict: Pillow 9.0.1 missing PIL.Image.Resampling
# that transformers 5.x requires (added in Pillow 9.1).
python -m pip install --upgrade pillow jinja2 markupsafe

# Install the TrialMine package itself (no deps — pip just resolved them above).
# --ignore-requires-python lets us install on Lambda Stack's Python 3.10 even
# though pyproject says >=3.11; our training scripts don't use 3.11-only syntax.
# NOTE: non-editable install (`pip install .` not `-e .`) — Lambda Stack's
# setuptools 59.6.0 is too old to support PEP 660 editable installs from
# pyproject-only projects. Non-editable copies src/TrialMine into site-packages,
# imports work the same way.
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
# Step 2: OOM probe at batch=64 (40GB-GPU sized; MNRL 2-encoder factor)
# ----------------------------------------------------------------------
log "===== Step 2: OOM probe (batch=64) ====="
python scripts/oom_probe.py --batch 64

# ----------------------------------------------------------------------
# Step 3: LR sweep — 1.4e-5, 2.8e-5, 5.6e-5 (sqrt-rescaled for batch=64)
# ----------------------------------------------------------------------
log "===== Step 3: LR sweep (3 short runs, max_steps=500) ====="
for lr in 1.4e-5 2.8e-5 5.6e-5; do
    log "----- sweep lr=$lr -----"
    python scripts/finetune_embeddings.py \
        --override training.learning_rate="$lr" \
        --override training.max_steps=500 \
        --override mlflow.run_tag="sweep-bs64-lr${lr}" \
        --override model.output_dir="/tmp/sweep_${lr}"
done

# ----------------------------------------------------------------------
# Step 4: pick winning LR; rewrites configs/training/embeddings.yaml in place
# ----------------------------------------------------------------------
log "===== Step 4: select winning LR ====="
python scripts/select_lr.py --tag-prefix sweep-bs64-lr
log "learning_rate now committed to configs/training/embeddings.yaml:"
grep -E '^\s*learning_rate:' configs/training/embeddings.yaml

# ----------------------------------------------------------------------
# Step 5: full 3-epoch training run
# ----------------------------------------------------------------------
log "===== Step 5: full training (3 epochs, ~3 hr on A100) ====="
python scripts/finetune_embeddings.py

# ----------------------------------------------------------------------
# Step 6: build FAISS index from the v2 model (OMP_NUM_THREADS=1 to be safe)
# ----------------------------------------------------------------------
log "===== Step 6: build FAISS index ====="
OMP_NUM_THREADS=1 python scripts/build_index.py --skip-bm25 \
    --model-path models/embeddings/fine-tuned-v2 \
    --output data/faiss_finetuned_v2

# ----------------------------------------------------------------------
# Step 7: success sentinel — visible at a glance over SSH
# ----------------------------------------------------------------------
touch ~/CLOUD_RUN_DONE
log "===== ALL STEPS PASSED. Sentinel: ~/CLOUD_RUN_DONE ====="
log "Next manual steps (see docs/fix_bi-encoder.md Phase B3 / B4):"
log "  1. From LOCAL, pull cloud_run.log, models/embeddings/fine-tuned-v2/,"
log "     and data/faiss_finetuned_v2.{index,json} back via rsync."
log "  2. THEN terminate the Lambda instance from the web UI."
log "     Do NOT terminate before pulling — billing only stops on terminate."
