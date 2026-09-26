# AGENTS.md

Guidance for AI coding agents working in this repository (Amazon ML Challenge 2026 — Business Entity Resolution).

## What this project is

A local, production-quality entity-resolution pipeline that matches Source 1 (deduplicated reference) business records against Source 2 and Source 3, optimizing **macro F0.5** (precision-heavy, computed per Source 1 entity then averaged). Authoritative references: `README.md` (challenge problem statement), `Documentation_template.md` (methodology template), `utils/validate_submission.py` (official validator). Do not contradict them.

## Hard rules (violations = challenge disqualification or broken output)

1. **Never use test labels or test data for training, calibration, or threshold fitting.** Test data is used only for inference-time blocking, embedding, and scoring. Splits for training/calibration are made from training data only, grouped by Source 1 entity (never pair-level random).
2. **No external business-identity data.** No external databases, geocoding APIs, commercial ER APIs, or internet-based identity augmentation. Everything is derived from the provided training/test files.
3. **Never hard-code countries.** Country is an open set of string labels. Train has {US, India}; test adds France (unseen in training). Country-specific thresholds may be learned **only for countries represented in the training split**; unseen countries fall back to the global threshold derived from training-country statistics. Never fit thresholds on France/test records.
4. **Output invariants:** every test S1 entity exactly once; empty `matched_entity_ids` for no-match; only S2-/S3- IDs; no duplicates within lists; every final match must be inside `candidate_pairs.tsv` (the exact candidate set fed to the final model).
5. **TSV everywhere.** Read/write with explicit `sep="\t"`; ID lists are comma-joined inside the second column, no quoting.
6. **Final models must be MIT/Apache-2.0 licensed and ≤ 8B parameters.** The Qwen stage has a hard license+param guard at build time; if it fails, the stage auto-skips.
7. **The official validator must PASS before any submission is considered done:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`. `--check-ids` is a diagnostic only — run it when memory allows; never treat it as a gate.

## Environment

- Windows 11 + Git Bash; use POSIX syntax in shell commands (`mv`/`rm`, forward slashes, heredocs).
- **Python lives in `.venv` only (3.12, created via uv). Never install anything globally.** Run tools as `.venv/Scripts/python.exe` (Git Bash: `.venv/Scripts/python`).
- GPU: RTX 4060 Laptop 8 GB VRAM — detect via `src/ber/env.py`; use fp16 and batch-size autosweep; keep 4-bit for the LLM stage. CPU fallbacks must exist for every GPU stage.
- RAM ~24 GB; never load whole multi-GB files into memory — chunk everything (see `src/ber/io_utils.py`).
- Disk budget: ~30 GB free. Artifacts (parquet, embeddings, FAISS indexes) can grow large; clean up stale intermediates rather than committing them.

## Pipeline conventions

- Stages: `S0 bootstrap → S1 data validation → S2 normalization → S3 blocking → S4 features → S5 LightGBM → S6 reranker → S6b LLM adjudication (guarded) → S7 calibration/decision → S8 outputs → S9 validator → S10 packaging`.
- **Every stage must be resumable**: write checkpoints under `artifacts/` with a `.meta.json` manifest (input hashes, params, versions) and skip-if-exists on rerun. `scripts/pipeline.py --stage <name>` runs one stage; `--stage all` runs everything.
- **Cache expensive things**: BGE-M3 embeddings and FAISS retrieval results are content-hash-cached on disk. Never recompute what a manifest says is fresh.
- **Do not create all-pairs comparisons** anywhere; blocking must union multiple strategies with per-S1 candidate caps, and the measured true-pair recall gate (≥ 0.999 on the train/val split) must pass before features are built.
- **Do not use argmax-only matching**: one S1 entity may have many matches (avg 3.46 in training; 5.6% are singletons, which score 1.0 only if predicted empty and 0.0 on any false match).
- Optimization target everywhere is **entity-level macro F0.5**, not pair accuracy or F1.
- New stages/flags go in `configs/config.yaml`; keep thresholds, paths, batch sizes, and seeds there, not hardcoded in modules.

## Repo map

```
dataset/{train,test}/*.tsv   raw challenge data (do not modify)
src/ber/                     library code (one module per stage)
scripts/                     entry points (pipeline.py, make_splits.py, eda_profile.py, ...)
artifacts/                   stage checkpoints (parquet, embeddings, indexes) — gitignore
models/                      trained model files + thresholds.json — gitignore
output/                      matching_results.tsv + candidate_pairs.tsv (final outputs)
reports/                     data_profile.md, blocking_audit.md, model_card.md, error_analysis.md
```

## Verification checklist for any code change

1. `.venv/Scripts/python -m compileall src scripts` (syntax) or run the touched stage on the tiny smoke subset first.
2. Stage invariants: run the stage's built-in asserts (they must never be silenced).
3. After touching S7/S8: rerun the official validator (must PASS).
4. After touching features/model: re-check `reports/blocking_audit.md` and `reports/model_card.md` numbers moved the right way on entity-level macro F0.5.

## Gotchas

- Plan-mode tooling refuses any `git` invocation that writes; in normal mode, still never push/commit unless asked.
- The original student-resource zip sits in the parent directory; the three authoritative docs were materialized from it into this repo — edit those copies, not the zip.
- Entity IDs look numeric but treat them as opaque strings (zero-padded variants exist).
- Some records have empty `business_address` (legitimate, especially S3/India) — handle empties explicitly rather than dropping rows.
- Devanagari script appears in names/addresses; normalization must be transliteration-aware and UTF-8 strict end-to-end.
