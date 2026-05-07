# syntax=docker/dockerfile:1.7
# Multi-stage build for the TrialMine FastAPI service.
# Builder installs heavy ML deps (torch CPU, scispacy biomedical model);
# runtime is a slim image with only site-packages + libgomp1.

# =============================================================================
# Stage 1 — builder: install Python dependencies and the project package
# =============================================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build toolchain — required for compiling native deps (faiss, lightgbm).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only torch first, in its own layer — it's the largest single wheel
# (~200 MB) and rarely changes, so caching it speeds up rebuilds dramatically.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Project install — pyproject.toml + src/ resolve all other deps via pip.
# Using a non-editable install puts TrialMine into site-packages so the
# runtime stage can drop the source tree entirely.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install .

# SciSpacy biomedical model used by features/eligibility.py — large
# (~600 MB compressed) and pinned to v0.5.4.
RUN pip install \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# =============================================================================
# Stage 2 — runtime: slim image carrying only what the API needs at request time
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    # Required on macOS hosts (FAISS + LightGBM OpenMP conflict per CLAUDE.md);
    # harmless on Linux. Disables OpenMP parallelism inside the process.
    OMP_NUM_THREADS=1

# libgomp1 is the only system runtime dep — used by faiss / lightgbm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull in the installed Python world from the builder. This brings TrialMine,
# torch, scispacy, transformers, sentence-transformers, langgraph, etc.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Run as a non-root user so a compromised process can't write outside /app.
RUN useradd --create-home --shell /usr/sbin/nologin trialmine
USER trialmine
WORKDIR /app

# data/ and models/ are mounted at runtime by docker-compose (read-only).
# Keep the image free of multi-GB indexes / model weights.

EXPOSE 8000

# Default: production server (no --reload). docker-compose can override.
CMD ["uvicorn", "TrialMine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
