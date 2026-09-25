"""S8 — Output generation (matching_results.tsv + candidate_pairs.tsv).

Hard invariants (asserted before write):
  - exactly one row per test S1 entity (missing/extra -> assertion error)
  - matched ids ⊆ candidate ids per entity (candidate_pairs IS the S3-final set)
  - only S2-/S3- ids; no duplicates within any list
  - canonical TSV: tab-separated, comma-joined ids, empty string for no match

Memory design: candidates are grouped shard-by-shard into per-entity
comma-joined strings (blocking guarantees an entity's candidates live
contiguously in one shard); predictions are small and decoded fully.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import io_utils
from .blocking import iter_candidate_shards

ID_OK = re.compile(r"^S[23]-\d+$")


def _join_sorted(ids) -> str:
    return ",".join(sorted(set(ids)))


def _assert_invariants(matching: dict[str, str], candidates: dict[str, str],
                       required_s1: list[str]) -> None:
    req = set(required_s1)
    miss = req - set(matching)
    extra = set(matching) - req
    assert not miss, f"matching_results missing {len(miss)} S1 entities, e.g. {sorted(miss)[:5]}"
    assert not extra, f"matching_results has {len(extra)} unknown S1 rows, e.g. {sorted(extra)[:5]}"
    for s1, joined in matching.items():
        ids = joined.split(",") if joined else []
        assert all(ID_OK.match(i) for i in ids), f"bad id in matching for {s1}"
        assert len(ids) == len(set(ids)), f"duplicate ids for {s1}"
        cand_set = set(candidates.get(s1, "").split(",")) - {""}
        bad = set(ids) - cand_set
        assert not bad, f"matched ids not in candidates for {s1}: {sorted(bad)[:5]}"


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    preds_path = art / "features" / "test_predictions.parquet"
    if not preds_path.exists():
        raise SystemExit("S8 requires test_predictions.parquet (run S7 first)")

    required: list[str] = []
    for df in io_utils.read_tsv_chunks(
        Path(cfg["paths"]["dataset_test_dir"]) / "test_source1.tsv", 1_000_000,
        columns=["entity_id"],
    ):
        required.extend(df["entity_id"].tolist())

    # candidates: stream shards -> per-entity joined strings
    candidates: dict[str, str] = {}
    n_cand_pairs = 0
    for shard in iter_candidate_shards("test", cfg):
        n_cand_pairs += len(shard)
        grouped = shard.groupby("s1_entity_id")["cand_id"].apply(_join_sorted)
        for s1, joined in grouped.items():
            prev = candidates.get(s1)
            candidates[s1] = joined if not prev else (prev + "," + joined)
        del shard, grouped
    print(f"  candidates grouped: {n_cand_pairs:,} pairs over {len(candidates):,} S1")

    # predictions: decode ids and group
    preds = pd.read_parquet(preds_path)
    matching = {s1: _join_sorted(g) for s1, g in preds.groupby("s1_entity_id")["cand_id"]}
    del preds

    _assert_invariants(matching, candidates, required)

    df_m = pd.DataFrame({
        "source1_entity_id": required,
        "matched_entity_ids": [matching.get(s1, "") for s1 in required],
    })
    df_c = pd.DataFrame({
        "source1_entity_id": required,
        "candidate_entity_ids": [candidates.get(s1, "") for s1 in required],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    io_utils.df_to_tsv(df_m, out_dir / "matching_results.tsv")
    io_utils.df_to_tsv(df_c, out_dir / "candidate_pairs.tsv")

    n_nonempty = int((df_m["matched_entity_ids"] != "").sum())
    print(f"S8 outputs written: {len(df_m):,} rows "
          f"({n_nonempty:,} matched, {len(df_m) - n_nonempty:,} empty) "
          f"| candidate pairs: {n_cand_pairs:,}")
    return {"rows": len(df_m), "nonempty": n_nonempty, "candidate_pairs": n_cand_pairs}
