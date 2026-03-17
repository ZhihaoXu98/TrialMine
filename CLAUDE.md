# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
TrialMine: ML-powered clinical trial search engine for oncology.
Patients describe their situation → AI finds relevant trials with explanations.

## Architecture
1. Data: ClinicalTrials.gov API v2 → parse → SQLite + Elasticsearch + FAISS
2. Retrieval: BM25 (Elasticsearch) + Semantic (FAISS with fine-tuned BioLinkBERT)
3. Re-ranking: Cross-encoder (fine-tuned BioLinkBERT) → LightGBM metadata blender
4. Agents: LangGraph — QueryParser → SearchOrchestrator (with tools)
5. Serving: FastAPI backend + Streamlit frontend
6. Monitoring: Prometheus + Grafana
7. Tracking: MLflow experiments

## Tech Stack
Python 3.11+, PyTorch, HuggingFace Transformers, sentence-transformers,
FastAPI, Streamlit, Elasticsearch 8.x, FAISS, LangGraph, langchain-anthropic,
LightGBM, MLflow, Optuna, SciSpacy, Docker, Prometheus + GitHub Actions

## ClinicalTrials.gov API v2
- Base: https://clinicaltrials.gov/api/v2/studies
- Pagination: pageToken (NOT page numbers), pageSize max 1000
- Key paths:
  protocolSection.identificationModule.nctId
  protocolSection.identificationModule.officialTitle
  protocolSection.descriptionModule.briefSummary
  protocolSection.conditionsModule.conditions
  protocolSection.armsInterventionsModule.interventions
  protocolSection.eligibilityModule.eligibilityCriteria
  protocolSection.eligibilityModule.minimumAge / maximumAge / sex
  protocolSection.designModule.phases
  protocolSection.statusModule.overallStatus
  protocolSection.designModule.enrollmentInfo.count
  protocolSection.contactsLocationsModule.locations
  protocolSection.sponsorCollaboratorsModule.leadSponsor.name
- Rate limit: add 0.5s delay between requests

## Coding Standards — IMPORTANT
- Type hints on ALL public functions
- Pydantic models for ALL data structures
- YAML configs for hyperparameters — NEVER hardcode magic numbers
- Structured JSON logging via Python logging module — NEVER use print()
- Error handling: NEVER let the API return 500. Always return structured error responses.
- Every function that calls an external service (API, DB, file) MUST have try/except
- Tests for all data parsing and feature engineering functions
- Docstrings on all public classes and functions

## File Structure
See README.md for overview. Key directories:
- src/trialmine/ — all source code
- scripts/ — training, indexing, evaluation scripts
- configs/ — YAML configs
- data/ — raw + processed data (gitignored except evaluation/)
- models/ — trained models (gitignored)
- docs/ — architecture, design decisions, model cards

## Current State
[UPDATE THIS AFTER EVERY SESSION]
Phase: Not started
Last completed: N/A
Next task: Phase 1 — project skeleton + data pipeline
