"""S8 — Output generation (matching_results.tsv + candidate_pairs.tsv).

Hard invariants (asserted before write):
  - exactly one row per test S1 entity (missing -> error; extra -> error)
  - matched ids ⊆ candidate ids per entity (candidate_pairs IS the S3-final set)
  - only S2-/S3- ids; no duplicates within any list
  - canonical TSV: tab-separated, comma-joined ids, empty string for no match
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import io_utils

ID_OK = re.compile(r"^S[23]-\d+$")


def _group(scored: pd.DataFrame, id_col: str) -> dict[str, list[str]]:
    out = scored.groupby("s1_entity_id")[id_col].apply(lambda s: sorted(set(s))).to_dict()
    return out


def _assert_invariants(matching: dict[str, list[str]], candidates: dict[str, list[str]],
                       required_s1: list[str]) -> None:
    req = set(required_s1)
    miss = req - set(matching)
    extra = set(matching) - req
    assert not miss, f"matching_results missing {len(miss)} S1 entities, e.g. {sorted(miss)[:5]}"
    assert not extra, f"matching_results has {len(extra)} unknown S1 rows, e.g. {sorted(extra)[:5]}"
    for s1, ids in matching.items():
        assert all(ID_OK.match(i) for i in ids), f"bad id in matching for {s1}"
        assert len(ids) == len(set(ids)), f"duplicate ids for {s1}"
        cand = set(candidates.get(s1, ()))
        bad = set(ids) - cand
        assert not bad, f"matched ids not in candidates for {s1}: {sorted(bad)[:5]}"


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    preds_path = art / "features" / "test_predictions.parquet"
    cand_path = art / "blocking" / "test_candidates.parquet"
    if not preds_path.exists():
        raise SystemExit("S8 requires test_predictions.parquet (run S7 first)")

    required = []
    for df in io_utils.read_tsv_chunks(
        Path(cfg["paths"]["dataset_test_dir"]) / "test_source1.tsv", 1_000_000,
        columns=["entity_id"],
    ):
        required.extend(df["entity_id"].tolist())

    preds = pd.read_parquet(preds_path)
    cand = pd.read_parquet(cand_path)
    matching = _group(preds, "cand_id")
    candidates = _group(cand, "cand_id")

    _assert_invariants(matching, candidates, required)

    rows_m = [{"source1_entity_id": s1, "matched_entity_ids": ",".join(matching.get(s1, []))}
              for s1 in required]
    rows_c = [{"source1_entity_id": s1, "candidate_entity_ids": ",".join(candidates.get(s1, []))}
              for s1 in required]
    df_m = pd.DataFrame(rows_m)
    df_c = pd.DataFrame(rows_c)
    out_dir.mkdir(parents=True, exist_ok=True)
    io_utils.df_to_tsv(df_m, out_dir / "matching_results.tsv")
    io_utils.df_to_tsv(df_c, out_dir / "candidate_pairs.tsv")

    n_nonempty = int((df_m["matched_entity_ids"] != "").sum())
    print(f"S8 outputs written: {len(df_m):,} rows "
          f"({n_nonempty:,} matched, {len(df_m) - n_nonempty:,} empty)")
    return {"rows": len(df_m), "nonempty": n_nonempty}
