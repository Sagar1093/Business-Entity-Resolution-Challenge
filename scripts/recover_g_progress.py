"""One-off recovery for the killed S3-G run (v1 shards, same df-pruning params).

Validates each existing {split}_candidates_g/shard_N.parquet against the
expected (country, 200k-row slice) boundaries implied by its deterministic
position, and writes progress.json entries for the shards that pass, so
`--stage blocking_g` resumes instead of recomputing ~2h of work.

A shard passes ONLY if every s1_entity_id in it maps to a positional index
inside its slice's row range [si*200k, min((si+1)*200k, n_country)) — this
proves the shard was produced by the current slice scheme (df-pruning only
changed which candidates were emitted, not which S1 rows were visited).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ber import io_utils  # noqa: E402
from src.ber.blocking import SLICE_ROWS  # noqa: E402

ART = Path("artifacts/normalized")
OUT = Path("artifacts/blocking")


def main() -> None:
    for split in ("train", "test"):
        out_dir = OUT / f"{split}_candidates_g"
        if not out_dir.exists():
            continue
        s1 = pd.read_parquet(ART / f"{split}_s1.parquet", columns=["entity_id", "country"])
        # (country, slice) -> allowed row range, in sorted-country order like the stage
        countries = sorted(pd.read_parquet(ART / f"{split}_s2.parquet",
                                           columns=["country"])["country"].unique().tolist())
        ranges: dict[str, tuple[int, int]] = {}
        gid = 0
        for c in countries:
            n = int((s1["country"] == c).sum())
            n_slices = (n + SLICE_ROWS - 1) // SLICE_ROWS
            for si in range(n_slices):
                ranges[f"{c}|{si}"] = (gid, gid + min(SLICE_ROWS, n - si * SLICE_ROWS))
                gid += SLICE_ROWS
        # positional index of every S1 id within its country-ordered sequence
        pos_of: dict[str, int] = {}
        start = 0
        for c in countries:
            ids = s1.loc[s1["country"] == c, "entity_id"].to_numpy()
            pos_of.update(zip(ids.tolist(), range(start, start + len(ids))))
            start += len(ids)

        completed: dict[str, str] = {}
        total = 0
        shard_paths = sorted(out_dir.glob("shard_*.parquet"),
                             key=lambda p: int(p.stem.split("_")[1]))
        for p in shard_paths:
            k = int(p.stem.split("_")[1])
            keys_ordered = sorted(ranges.keys(), key=lambda kk: (countries.index(kk.split("|")[0]),
                                                                 int(kk.split("|")[1])))
            if k >= len(keys_ordered):
                print(f"  [{split}] {p.name}: shard id {k} out of range — SKIP")
                continue
            key = keys_ordered[k]
            lo, hi = ranges[key]
            ids = pq.read_table(p, columns=["s1_entity_id"]).to_pandas()["s1_entity_id"].to_numpy()
            idx = np.array([pos_of.get(e, -1) for e in ids], dtype=np.int64)
            ok = bool(len(idx) == 0 or ((idx >= lo) & (idx < hi)).all())
            print(f"  [{split}] {p.name} -> {key} rows={len(ids):,} "
                  f"{'OK' if ok else 'MISALIGNED — SKIP'}")
            if ok:
                completed[key] = p.name
                total += len(ids)
            else:
                p.unlink()
        io_utils.json_dump({"completed": completed}, out_dir / "progress.json")
        print(f"  [{split}] recovered {len(completed)} slices ({total:,} pairs) -> progress.json")


if __name__ == "__main__":
    main()
