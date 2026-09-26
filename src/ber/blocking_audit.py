"""S3b — Blocking recall audit (train/val split only; uses GT — never test).

Metrics:
  pair recall   = |GT pairs ∩ candidate pairs| / |GT pairs| (val entities)
  entity recall = fraction of val entities whose GT matches are ALL inside
                  the candidate set (the F0.5-relevant ceiling)
  avg candidates per S1, reduction ratio vs all-pairs baseline.

Gate: pair recall >= 0.999 AND full-capture entity rate >= 0.995 (config).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, metrics


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    out_md = Path(cfg["paths"]["reports_dir"]) / "blocking_audit.md"
    res_path = art / "blocking_audit.json"
    if not force and res_path.exists():
        res = io_utils.json_load(res_path)
        if res.get("gate") == "PASS":
            print("blocking audit fresh & PASS — skipping")
            return res

    manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    truth_all = metrics.gt_dict(cfg, only_val=True, manifest=manifest)
    truth_all = {k: v for k, v in truth_all.items() if k in val_ids}
    n_truth_pairs = sum(len(v) for v in truth_all.values())

    # candidate pairs restricted to val entities (base ∪ G trigram shards,
    # slice-aligned — same pairing rule as iter_candidate_shards)
    hit_pairs = 0
    n_cand = 0
    cand_per_s1: dict[str, int] = {}
    cand_dir = art / "blocking" / "train_candidates"
    g_dir = art / "blocking" / "train_candidates_g"
    base_shards = sorted(cand_dir.glob("shard_*.parquet"))
    g_map = {p.name: p for p in sorted(g_dir.glob("shard_*.parquet"))} if g_dir.exists() else {}
    if g_map and len(g_map) != len(base_shards):
        raise RuntimeError(
            f"G/base shard-count mismatch in audit: {len(g_map)} G vs {len(base_shards)} base — "
            "shards must be slice-aligned.")

    def _val_union(shard: Path) -> pd.DataFrame:
        df = pd.read_parquet(shard)
        df = df[df["s1_entity_id"].isin(val_ids)]
        g = g_map.get(shard.name)
        if g is not None:
            dg = pd.read_parquet(g, columns=["s1_entity_id", "cand_id"])
            dg = dg[dg["s1_entity_id"].isin(val_ids)]
            df = pd.concat([df, dg], ignore_index=True).drop_duplicates(
                subset=["s1_entity_id", "cand_id"])
        return df

    for shard in base_shards:
        df = _val_union(shard)
        n_cand += len(df)
        cnt = df.groupby("s1_entity_id").size()
        for k, v in cnt.items():
            cand_per_s1[k] = cand_per_s1.get(k, 0) + int(v)
        merged = df.merge(
            pd.DataFrame([(s1, m) for s1, ms in truth_all.items() for m in ms],
                         columns=["s1_entity_id", "cand_id"]).drop_duplicates(),
            on=["s1_entity_id", "cand_id"], how="inner",
        )
        hit_pairs += len(merged)
        del df, merged

    pair_recall = hit_pairs / max(1, n_truth_pairs)

    # entity-level capture: collect candidate sets per val entity (true sets from GT)
    full_capture = 0
    recalls = []
    val_cand_sets: dict[str, set[str]] = {}
    for shard in base_shards:
        df = _val_union(shard)
        for s1, g in df.groupby("s1_entity_id")["cand_id"]:
            val_cand_sets.setdefault(s1, set()).update(g.to_numpy())
        del df
    for s1, true_set in truth_all.items():
        cands = val_cand_sets.get(s1, set())
        r = len(true_set & cands) / len(true_set) if true_set else 1.0
        recalls.append(r)
        full_capture += (r >= 1.0)
    entity_full_rate = full_capture / max(1, len(truth_all))

    res = {
        "val_entities": len(truth_all),
        "truth_pairs": n_truth_pairs,
        "candidate_pairs_val": n_cand,
        "hit_pairs": int(hit_pairs),
        "pair_recall": round(pair_recall, 6),
        "entity_full_capture_rate": round(entity_full_rate, 6),
        "median_entity_recall": round(float(np.median(recalls)), 6) if recalls else 0.0,
        "avg_candidates_per_s1": round(n_cand / max(1, len(truth_all)), 2),
        "gate_pair_threshold": cfg["blocking"]["recall_gate_pair"],
        "gate_entity_threshold": cfg["blocking"]["recall_gate_entity"],
        "gate": "PENDING",
    }
    res["gate"] = "PASS" if (pair_recall >= res["gate_pair_threshold"]
                             and entity_full_rate >= res["gate_entity_threshold"]) else "FAIL"
    io_utils.json_dump(res, res_path)
    _write_md(res, out_md)
    print(f"blocking audit: pair_recall={pair_recall:.5f} entity_full={entity_full_rate:.5f} "
          f"avg_cand={res['avg_candidates_per_s1']} -> {res['gate']}")
    if res["gate"] != "PASS":
        raise SystemExit("S3 audit FAILED — recall gate not met; improve blocking before S4.")
    return res


def _write_md(res: dict, out: Path) -> None:
    md = [
        "# Blocking audit (val split, train GT only)", "",
        f"- pair recall: **{res['pair_recall']:.5f}** (gate >= {res['gate_pair_threshold']})",
        f"- entity full-capture rate: **{res['entity_full_capture_rate']:.5f}** "
        f"(gate >= {res['gate_entity_threshold']})",
        f"- median per-entity recall: {res['median_entity_recall']:.5f}",
        f"- truth pairs (val): {res['truth_pairs']:,} | candidate pairs (val): {res['candidate_pairs_val']:,}",
        f"- avg candidates per S1: {res['avg_candidates_per_s1']}",
        "", f"**GATE: {res['gate']}**", "",
    ]
    out.write_text("\n".join(md), encoding="utf-8")
