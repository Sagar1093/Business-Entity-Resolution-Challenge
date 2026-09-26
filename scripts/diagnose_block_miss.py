"""Diagnose WHY blocking misses val GT pairs: source mix, name similarity,
address signals, and per-strategy reachability of missed pairs.

Train GT + val entities only (never test). Read-only diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ber import metrics  # noqa: E402

ART = Path("artifacts/normalized")


def trigrams(t: str) -> set[str]:
    t = f"  {t} "
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 5 else ({t.strip()} if t.strip() else set())


def main() -> None:
    manifest = pd.read_parquet("artifacts/splits/split_manifest.parquet")
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    import yaml
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    gt = metrics.gt_dict(cfg, only_val=True, manifest=manifest)
    gt = {k: v for k, v in gt.items() if k in val_ids}

    # load GT pairs
    pairs = pd.DataFrame([(s1, m) for s1, ms in gt.items() for m in ms],
                         columns=["s1_entity_id", "cand_id"]).drop_duplicates()
    print(f"val GT pairs: {len(pairs):,}")

    # candidate sets (base+G union, val only)
    val_cands: dict[str, set[str]] = {}
    for d in ("train_candidates", "train_candidates_g"):
        for p in sorted((ART.parent / "blocking" / d).glob("shard_*.parquet")):
            df = pd.read_parquet(p, columns=["s1_entity_id", "cand_id"])
            df = df[df["s1_entity_id"].isin(val_ids)]
            for s1, g in df.groupby("s1_entity_id")["cand_id"]:
                val_cands.setdefault(s1, set()).update(g.to_numpy())
            del df
    pairs["hit"] = [m in val_cands.get(s1, set()) for s1, m in
                    zip(pairs["s1_entity_id"], pairs["cand_id"])]
    print(f"hit: {pairs['hit'].sum():,}  missed: {(~pairs['hit']).sum():,}")

    # source of candidate id
    id2src = {}
    for src in ("s2", "s3"):
        ids = pd.read_parquet(ART / f"train_{src}.parquet", columns=["entity_id"])["entity_id"]
        id2src.update(dict.fromkeys(ids.to_numpy(), src))
    pairs["src"] = pairs["cand_id"].map(id2src)
    print("\n-- by source --")
    print(pairs.groupby(["src", "hit"]).size().unstack(fill_value=0))

    # name similarity of missed vs hit
    s1n = pd.read_parquet(ART / "train_s1.parquet", columns=["entity_id", "name_norm"])
    n2s1 = dict(zip(s1n["entity_id"], s1n["name_norm"]))
    cand_names: dict[str, str] = {}
    for src in ("s2", "s3"):
        t = pd.read_parquet(ART / f"train_{src}.parquet", columns=["entity_id", "name_norm"])
        cand_names.update(dict(zip(t["entity_id"], t["name_norm"])))
        del t

    miss = pairs[~pairs["hit"]].sample(n=min(20000, (~pairs["hit"]).sum()), random_state=0)
    sim = []
    for s1, c in zip(miss["s1_entity_id"], miss["cand_id"]):
        a, b = n2s1.get(s1, ""), cand_names.get(c, "")
        ga, gb = trigrams(a), trigrams(b)
        sim.append(len(ga & gb) / max(1, min(len(ga), len(gb))) if ga and gb else 0.0)
    sim = np.asarray(sim)
    print("\n-- missed pairs: coverage-normalized trigram sim --")
    for lo, hi in ((0.0, 0.0001), (0.0001, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        n = int(((sim >= lo) & (sim < hi)).sum())
        print(f"  [{lo:.1f},{hi:.1f}): {n:6d} ({n / len(sim):.1%})")
    print(f"  median sim of missed: {np.median(sim):.3f}")

    # name-empty analysis on missed
    empty_s1 = sum(1 for s1 in miss["s1_entity_id"] if not n2s1.get(s1, ""))
    empty_c = sum(1 for c in miss["cand_id"] if not cand_names.get(c, ""))
    print(f"missed with empty S1 name: {empty_s1}, empty cand name: {empty_c} (of {len(miss)})")

    # entity-level: entities with zero recall
    zero = sum(1 for s1, ts in gt.items() if ts and not (ts & val_cands.get(s1, set())))
    print(f"\nval entities with ZERO candidate recall: {zero:,} of {len(gt):,}")
    ex = [s1 for s1, ts in gt.items() if ts and not (ts & val_cands.get(s1, set()))][:10]
    for s1 in ex:
        print(f"  e.g. {s1!r} name={n2s1.get(s1, '')!r} n_true={len(gt[s1])}")


if __name__ == "__main__":
    main()
