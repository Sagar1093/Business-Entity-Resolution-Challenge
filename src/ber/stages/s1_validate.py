"""S1 stage — authoritative data validation gate (resumable via manifest)."""
from __future__ import annotations

from pathlib import Path

from ber import io_utils, validate_data

TRAIN_FILES = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]
TEST_FILES = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]


def run(cfg: dict, force: bool = False) -> None:
    out_json = Path(cfg["paths"]["artifacts_dir"]) / "data_validation.json"
    train_dir, test_dir = Path(cfg["paths"]["dataset_train_dir"]), Path(cfg["paths"]["dataset_test_dir"])
    inputs = {name: train_dir / name for name in TRAIN_FILES}
    inputs.update({name: test_dir / name for name in TEST_FILES})
    params = {"v": 1}
    if not force and io_utils.manifest_ok(out_json, inputs, params):
        rep = io_utils.json_load(out_json)
        print(f"S1 already validated (gate={rep['gate']}) — skipping. Use --force to rerun.")
        return
    report = validate_data.run(cfg)
    io_utils.save_manifest(out_json, inputs, params, extra={"gate": report["gate"]})
