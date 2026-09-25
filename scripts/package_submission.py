#!/usr/bin/env python
"""S10 — build the final submission zip per the challenge package spec.

Structure:
  <team>_submission.zip
    output/matching_results.tsv + candidate_pairs.tsv
    code/business_entity_resolution/{src/**, README.md, requirements.txt}
    Documentation_template.md (filled)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fill_docs(cfg_path: Path, out: Path) -> None:
    """Fill the methodology template with measured numbers from artifacts."""
    template = (ROOT / "Documentation_template.md").read_text(encoding="utf-8")
    art = ROOT / "artifacts"
    rep = {}

    def jload(name):
        p = art / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    dv = jload("data_validation.json")
    eda = jload("eda.json")
    ev = jload("model_eval.json")
    thr = jload("calibration/thresholds.json")
    blk = {s: jload(f"blocking/{s}_stats.json") for s in ("train", "test")}

    gt = dv.get("ground_truth", {})
    s1 = dv.get("sources", {}).get("train_source1", {})
    best = thr.get("best", {})
    pairs_train = blk.get("train", {}).get("pairs_total")
    pairs_test = blk.get("test", {}).get("pairs_total")

    filled = template
    filled = filled.replace("[Your Team Name]", "Sagar1093")
    filled = filled.replace("[List all team members]", "Sagar")
    filled = filled.replace("[Date]", "2026-09-25")
    filled = filled.replace(
        "*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*",
        f"- Train: {s1.get('rows', 0):,} S1 reference entities; {gt.get('total_matches', 0):,} true matches "
        f"(avg {gt.get('total_matches', 0) / max(1, gt.get('rows', 1)):.2f}/entity; "
        f"{gt.get('empty_lists', 0) / max(1, gt.get('rows', 1)):.1%} singletons). Countries: US/India train; +France test.\n"
        f"- S1 is a clean Latin-only reference; Devanagari noise is concentrated in S2/S3 "
        f"({eda.get('per_source_name_noise', {}).get('train_source2.tsv', {}).get('devanagari', 0):,} and "
        f"{eda.get('per_source_name_noise', {}).get('train_source3.tsv', {}).get('devanagari', 0):,} names) plus "
        f"domain/DBA-style names (~4%). Heavy character noise remains post-normalization "
        f"(only ~32% of GT pairs are exact after normalization) -> edit/gram features essential.\n"
        f"- Every S2/S3 id matches at most one S1 entity (unique assignment) -> candidate-competition in S7.\n"
        f"- \\x1a/\\x1f/\\x7f mangled-dash noise ~800 cells; cleaned in normalization.",
    )
    filled = filled.replace(
        "**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]",
        "**Approach Type:** Blocking + LightGBM pair classifier + cross-encoder reranker + "
        "calibrated decision engine (grouped-split validation, macro-F0.5 optimized)",
    )
    filled = filled.replace(
        "**Core Innovation:** [Brief description of your main technical contribution]",
        "**Core Innovation:** Six-strategy country-partitioned blocking with trigram rescue; "
        "F0.5-max calibrated thresholds with per-country values only for training countries "
        "(global fallback for unseen countries like France); candidate competition exploiting "
        "the measured unique-assignment structure.",
    )
    filled = filled.replace(
        "- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]",
        "- **Blocking keys used:** normalized name-bag, digit-stripped bag, metaphone (4/2-token), "
        "postal co-occurrence, char-3gram top-K rescue (inverted index)",
    )
    filled = filled.replace(
        "- **Candidate pairs generated:** [total]",
        f"- **Candidate pairs generated:** train {pairs_train:,} / test {pairs_test:,}"
        if pairs_train else "- **Candidate pairs generated:** (see artifacts/blocking/*_stats.json)",
    )
    filled = filled.replace(
        "**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc]",
        "**Model type:** LightGBM (28 lexical/structural features) + bge-reranker-v2-m3 band scoring; "
        "isotonic calibration; thresholds " + json.dumps(thr.get("thresholds", {})),
    )
    filled = filled.replace(
        "**Threshold selection method:** [e.g., F_0.5 optimization on validation set]",
        "**Threshold selection method:** grid sweep maximizing entity-level macro F0.5 on the "
        "grouped 10% validation split; per-country thresholds learned for US/India only; "
        "global fallback for France/unseen countries",
    )
    if ev:
        filled = filled.replace(
            "- **F_0.5 Score (macro):** [your best validation score]",
            f"- **F_0.5 Score (macro):** {best.get('macro_f05', 0):.5f} (val; threshold {best.get('threshold', 0)}) "
            f"| val pair AP {ev.get('val_pair_ap', 0):.5f}",
        )
    out.write_text(filled, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Sagar1093")
    ap.add_argument("--skip-docs-fill", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "submission"
    out_dir.mkdir(exist_ok=True)
    zip_path = out_dir / f"{args.team}_submission"

    # fill documentation
    if not args.skip_docs_fill:
        _fill_docs(ROOT / "configs" / "config.yaml", ROOT / "Documentation_template.md")

    import shutil

    staging = out_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "code" / "business_entity_resolution").mkdir(parents=True)
    shutil.copytree(ROOT / "src", staging / "code" / "business_entity_resolution" / "src")
    shutil.copy2(ROOT / "requirements.txt", staging / "code" / "business_entity_resolution")
    shutil.copy2(ROOT / "scripts" / "pipeline.py", staging / "code" / "business_entity_resolution" / "run_pipeline.py")
    (staging / "code" / "business_entity_resolution" / "README.md").write_text(
        "# Reproduce\n\n"
        "1. `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`\n"
        "2. Place challenge data under `dataset/{train,test}/`.\n"
        "3. `python run_pipeline.py --stage all` (resumable; checkpoints under `artifacts/`).\n"
        "4. Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.\n",
        encoding="utf-8",
    )
    outp = staging / "output"
    outp.mkdir()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        src = ROOT / "output" / f
        if src.exists():
            shutil.copy2(src, outp / f)
    shutil.copy2(ROOT / "Documentation_template.md", staging / "Documentation_template.md")
    shutil.make_archive(str(zip_path), "zip", staging)
    shutil.rmtree(staging)
    print(f"submission zip: {zip_path}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
