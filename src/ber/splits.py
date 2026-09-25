"""Grouped 90/10 split by Source 1 entity (leakage-proof).

- Stratified by (country, match-count bucket) for balanced validation.
- Deterministic: seeded RNG per stratum; stable across runs.
- NEVER pair-level random; S2/S3 records follow their S1 entity's split.
- Uses train data only; test is never touched.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils

VAL_FRACTION = 0.10


def run(cfg: dict, force: bool = False) -> dict:
    out_path = Path(cfg["splits"]["split_manifest"])
    meta_path = out_path.with_suffix(".meta.json")
    gt_path = Path(cfg["paths"]["dataset_train_dir"]) / "train_ground_truth.tsv"
    s1_path = Path(cfg["paths"]["dataset_train_dir"]) / "train_source1.tsv"
    params = {"val_fraction": VAL_FRACTION, "seed": cfg["seed"], "v": 1}
    if not force and io_utils.manifest_ok(out_path, {"gt": gt_path, "s1": s1_path}, params):
        summary = io_utils.json_load(meta_path)["summary"]
        print(f"Split manifest fresh — skipping. val={summary['val_entities']:,}")
        return summary

    seed = int(cfg["seed"])
    n_matches: dict[str, int] = {}
    for df in io_utils.read_tsv_chunks(gt_path, cfg["io"]["chunk_rows"]):
        counts = df["matched_entity_ids"].str.count(",") + 1
        counts[df["matched_entity_ids"] == ""] = 0
        n_matches.update(dict(zip(df["source1_entity_id"], counts.astype(np.int32))))

    country: dict[str, str] = {}
    for df in io_utils.read_tsv_chunks(s1_path, cfg["io"]["chunk_rows"], columns=["entity_id", "country"]):
        country.update(dict(zip(df["entity_id"], df["country"])))

    strata: dict[tuple, list[str]] = defaultdict(list)
    for s1, n in n_matches.items():
        strata[(country.get(s1, "<none>"), min(int(n), 5))].append(s1)

    rng = np.random.default_rng(seed)
    rows = []
    for key, ids in sorted(strata.items(), key=lambda kv: str(kv[0])):
        ids_arr = np.array(sorted(ids))  # sorted before shuffle => deterministic
        rng.shuffle(ids_arr)
        n_val = int(round(VAL_FRACTION * len(ids_arr)))
        for i, s1 in enumerate(ids_arr):
            rows.append((s1, "val" if i < n_val else "train_models", key[0], int(key[1])))

    manifest = pd.DataFrame(rows, columns=["s1_entity_id", "split", "country", "n_matches_capped"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_parquet(manifest, out_path)

    summary = {
        "train_entities": int((manifest["split"] == "train_models").sum()),
        "val_entities": int((manifest["split"] == "val").sum()),
        "val_fraction_actual": round(float((manifest["split"] == "val").mean()), 5),
        "strata": len(strata),
    }
    io_utils.save_manifest(out_path, {"gt": gt_path, "s1": s1_path}, params, extra={"summary": summary})
    print("Split summary:", summary)
    return summary
