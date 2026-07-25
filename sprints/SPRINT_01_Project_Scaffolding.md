# SPRINT 01 — Project Scaffolding & Architecture

## Phase Goal
Set up the foundational repository structure as a production-grade monorepo, pinning dependencies, configuring git ignores, and establishing core development conventions.

## Scope
**In Scope:** 
- Folder structure creation (`/data/`, `/ml/`, `/backend/`, `/frontend/`, etc.).
- Writing `requirements.txt` with pinned versions compatible with Qwen2-VL-2B and Colab.
- Writing `AGENTS.md` containing global development rules.
- Writing `.gitignore` to prevent datasets and model weights from being committed.
- Initializing `README.md`.

**Out of Scope:** 
- Any ML code (data loading, modeling, training).
- React/FastAPI initial code (reserved for Sprint 08).

## File/Folder Structure Created
- `/data/`
- `/ml/data_pipeline/`
- `/ml/training/`
- `/ml/evaluation/`
- `/ml/robustness/`
- `/ml/tampering/`
- `/ml/inference/`
- `/ml/configs/`
- `/backend/`
- `/frontend/`
- `/sprints/`
- `/docs/`
- `AGENTS.md`
- `requirements.txt`
- `README.md`
- `.gitignore`

## Task Checklist
- [x] Create project directories.
- [x] Create `.gitignore` to protect datasets/checkpoints.
- [x] Draft `requirements.txt` with pinned versions (Transformers, PyTorch, PEFT, etc.).
- [x] Draft `AGENTS.md` with strict project conventions.
- [x] Draft `README.md`.

## Dependencies
- None.

## Manual Steps Required From Me
- None yet — pure scaffolding.

## Definition of Done
- Directory structure exists.
- Configuration and scaffolding files (`requirements.txt`, `AGENTS.md`, `README.md`, `.gitignore`) are committed/saved.

---

## Continue?
**Summary:** The foundational monorepo structure has been created. `requirements.txt` correctly pins versions to support Qwen2-VL-2B (e.g. `transformers==4.45.0`, `peft`, `bitsandbytes` for 4-bit quant). `AGENTS.md` details reproducibility, config requirements, and hardware constraints (T4 GPU, flash-attn being optional). `.gitignore` is ready to prevent uploading large data or checkpoints.

**Questions before proceeding:**
1. Does the repository structure and the pinned libraries list meet your expectations?
2. Are you ready to proceed to SPRINT 02 (Helper Phase: Dataset Acquisition & Colab GPU Connection)?
