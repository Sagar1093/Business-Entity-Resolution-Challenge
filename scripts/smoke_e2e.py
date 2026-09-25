#!/usr/bin/env python
"""Synthetic end-to-end smoke test: tiny planted-match dataset through
S2 normalize -> S3 blocking -> S3 audit -> S4 features -> S5 model ->
S7 decision -> S8 outputs, all inside a temp artifacts root.

Validates module interfaces and the index-based plumbing (not model quality).
Run:  .venv/Scripts/python scripts/smoke_e2e.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

FIRST = ["rapid", "blue", "summit", "delta", "omega", "north", "green", "silver"]
SECOND = ["logistics", "ventures", "labs", "holdings", "trading", "works", "digital", "group"]
SUFFIX = ["", " llc", " inc", " pvt ltd", " limited"]
CITIES = {"US": ["springfield", "riverside", "franklin"], "India": ["pune", "delhi", "surat"]}
STREETS = ["main street", "oak road", "station road", "mg road"]


def make_record(rng, eid_prefix, i, country, name, addr):
    return {
        "entity_id": f"{eid_prefix}-{i:07d}",
        "business_name": name,
        "business_address": addr,
        "country": country,
    }


def corrupt_name(rng, name):
    toks = name.split()
    if len(toks) > 1 and rng.random() < 0.4:
        rng.shuffle(toks)
    out = []
    for t in toks:
        r = rng.random()
        if r < 0.15 and len(t) > 3:  # char swap
            k = rng.integers(1, len(t) - 2)
            t = t[:k] + t[k + 1] + t[k] + t[k + 2:]
        elif r < 0.25:  # letter dup
            t = t + t[-1]
        out.append(t)
    return " ".join(out)


def build_synthetic(tmp: Path, n_entities=400, per_match=(1, 4)):
    rng = np.random.default_rng(7)
    train_dir = tmp / "dataset" / "train"
    test_dir = tmp / "dataset" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    def gen_split(d, n_entities, with_truth):
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        nid = 1
        n2 = 5000
        n3 = 9000
        for i in range(1, n_entities + 1):
            country = "US" if rng.random() < 0.6 else "India"
            base = f"{rng.choice(FIRST)} {rng.choice(SECOND)}"
            canonical = base + rng.choice(SUFFIX)
            s1_rows.append(make_record(rng, "S1", i, country, canonical,
                                       f"{rng.integers(10, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES[country])}"))
            k = rng.integers(per_match[0], per_match[1] + 1) if with_truth else rng.integers(0, 4)
            matches = []
            for _ in range(k):
                if rng.random() < 0.5:
                    n2 += 1
                    nm = corrupt_name(rng, base) + rng.choice(SUFFIX)
                    s2_rows.append(make_record(rng, "S2", n2, country, nm,
                                               f"{rng.integers(10, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES[country])}"))
                    matches.append(f"S2-{n2:07d}")
                else:
                    n3 += 1
                    nm = corrupt_name(rng, base)
                    if rng.random() < 0.3:
                        nm = "www." + nm.replace(" ", "") + ".com"
                    s3_rows.append(make_record(rng, "S3", n3, country, nm,
                                               f"{rng.integers(10, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES[country])}"))
                    matches.append(f"S3-{n3:07d}")
            if with_truth:
                gt_rows.append({"source1_entity_id": f"S1-{i:07d}",
                                "matched_entity_ids": ",".join(sorted(matches))})
        pd.DataFrame(s1_rows).to_csv(d / "train_source1.tsv" if with_truth else d / "test_source1.tsv",
                                     sep="\t", index=False)
        pd.DataFrame(s2_rows).to_csv(d / "train_source2.tsv" if with_truth else d / "test_source2.tsv",
                                     sep="\t", index=False)
        pd.DataFrame(s3_rows).to_csv(d / "train_source3.tsv" if with_truth else d / "test_source3.tsv",
                                     sep="\t", index=False)
        if with_truth:
            gt = pd.DataFrame(gt_rows)
            # inject some true singletons (empty lists)
            single = gt.sample(frac=0.08, random_state=1).index
            gt.loc[single, "matched_entity_ids"] = ""
            gt.to_csv(d / "train_ground_truth.tsv", sep="\t", index=False)
        return len(s1_rows), len(s2_rows), len(s3_rows)

    a = gen_split(train_dir, n_entities, True)
    b = gen_split(test_dir, 200, False)
    print(f"synthetic: train s1/s2/s3 = {a}, test s1/s2/s3 = {b}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ber_smoke_e2e_"))
    print("workspace:", tmp)
    build_synthetic(tmp)

    # link authoritative data-independent modules to the tmp root
    import ber.config as ber_config
    cfg = ber_config.load()
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["dataset_train_dir"] = str(tmp / "dataset" / "train")
    cfg["paths"]["dataset_test_dir"] = str(tmp / "dataset" / "test")
    cfg["paths"]["artifacts_dir"] = str(tmp / "artifacts")
    cfg["paths"]["models_dir"] = str(tmp / "models")
    cfg["paths"]["reports_dir"] = str(tmp / "reports")
    cfg["paths"]["output_dir"] = str(tmp / "output")
    Path(cfg["paths"]["models_dir"]).mkdir(parents=True)
    Path(cfg["paths"]["reports_dir"]).mkdir(parents=True)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True)
    # tiny split: 80/20 grouped
    from ber import splits as splits_mod
    splits_mod.VAL_FRACTION = 0.2
    cfg["splits"] = dict(cfg["splits"])
    cfg["splits"]["split_manifest"] = str(tmp / "artifacts" / "splits" / "split_manifest.parquet")

    from ber.stages import registry
    for name in ("normalize", "splits", "blocking", "blocking_g", "features"):
        print(f"\n--- stage {name} ---")
        registry.STAGES[name]["run"](cfg, force=True)

    # audit: with planted noise the gate may legitimately fail; run in-process to inspect
    from ber import blocking_audit
    print("\n--- stage blocking_audit ---")
    try:
        blocking_audit.run(cfg, force=True)
    except SystemExit as e:
        print("audit gate (informational for synthetic data):", e)

    print("\n--- stage model ---")
    registry.STAGES["model"]["run"](cfg, force=True)

    print("\n--- stage decision ---")
    registry.STAGES["decision"]["run"](cfg, force=True)

    print("\n--- stage outputs ---")
    registry.STAGES["outputs"]["run"](cfg, force=True)

    m = pd.read_csv(Path(cfg["paths"]["output_dir"]) / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(Path(cfg["paths"]["output_dir"]) / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"\nmatching rows: {len(m)} non-empty: {(m['matched_entity_ids'] != '').sum()}")
    print(f"candidate rows: {len(c)}")
    assert len(m) == 200 and len(c) == 200, "one row per test S1 entity"
    matched_subset = all(
        set(r.split(",")) <= set(cand.split(","))
        for r, cand in zip(m["matched_entity_ids"], c["candidate_entity_ids"]) if r
    )
    assert matched_subset, "matches must be subset of candidates"
    print("\nSMOKE E2E: ALL STRUCTURAL CHECKS PASS")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
