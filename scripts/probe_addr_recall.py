"""Probe: do missed val GT pairs share rare ADDRESS tokens (digits/street words)?

Hypothesis: generic-name entities (huge exact-name blocks) are missed because
blocking only uses addr_postal (often empty). If true partners share rare
address tokens (digit runs, street/landmark words), an address-token blocking
strategy (H) will capture them. Train GT + val entities only (never test).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ber import metrics  # noqa: E402

ART = Path("artifacts/normalized")
DF_MIN = 3
DF_CAP = 5000


def addr_tokens(row_addr_norm: str, row_addr_digits: str) -> set[str]:
    toks = set(row_addr_norm.split())
    toks.update(row_addr_digits.split())
    return {t for t in toks if len(t) >= 2}


def main() -> None:
    import yaml
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    manifest = pd.read_parquet("artifacts/splits/split_manifest.parquet")
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    gt = metrics.gt_dict(cfg, only_val=True, manifest=manifest)
    gt = {k: v for k, v in gt.items() if k in val_ids}
    pairs = pd.DataFrame([(s1, m) for s1, ms in gt.items() for m in ms],
                         columns=["s1_entity_id", "cand_id"]).drop_duplicates()

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
    missed = pairs[~pairs["hit"]].sample(n=20000, random_state=0)
    print(f"missed sample: {len(missed):,}")

    s1 = pd.read_parquet(ART / "train_s1.parquet",
                         columns=["entity_id", "addr_norm", "addr_digits", "country"]).set_index("entity_id")
    cand_meta: dict[str, tuple[str, str, str]] = {}
    for src in ("s2", "s3"):
        t = pd.read_parquet(ART / f"train_{src}.parquet",
                            columns=["entity_id", "addr_norm", "addr_digits", "country"])
        cand_meta.update(zip(t["entity_id"], zip(t["addr_norm"], t["addr_digits"], t["country"])))
        del t

    # build token-df per country from ALL candidates (this is the real index cost)
    df_counts: dict[str, Counter] = {}
    for country in ("India", "US"):
        cnt: Counter = Counter()
        rows = [(a, d) for a, d, c in cand_meta.values() if c == country]
        for a, d in rows:
            for t in addr_tokens(a, d):
                cnt[t] += 1
        df_counts[country] = cnt
        print(f"{country}: {len(cnt):,} distinct addr tokens, "
              f"kept(dfs {DF_MIN}-{DF_CAP}): {sum(1 for v in cnt.values() if DF_MIN <= v <= DF_CAP):,}")

    s1_tokens = {e: addr_tokens(r.addr_norm, r.addr_digits) for e, r in s1.iterrows()}
    cand_tokens = {e: addr_tokens(a, d) for e, (a, d, c) in cand_meta.items()}

    shared_counts = []
    for s1e, ce in zip(missed["s1_entity_id"], missed["cand_id"]):
        country = cand_meta[ce][2]
        cnt = df_counts[country]
        st = s1_tokens.get(s1e, set())
        ct = cand_tokens.get(ce, set())
        shared = {t for t in (st & ct) if DF_MIN <= cnt.get(t, 0) <= DF_CAP}
        shared_counts.append(len(shared))
    shared_counts = np.asarray(shared_counts)
    print("\n-- missed pairs: shared rare addr tokens --")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 100)):
        n = int(((shared_counts >= lo) & (shared_counts < hi)).sum())
        print(f"  shared [{lo},{hi}): {n:6d} ({n / len(shared_counts):.1%})")
    print(f"  >=1 shared: {(shared_counts >= 1).mean():.1%}   >=2: {(shared_counts >= 2).mean():.1%}")


if __name__ == "__main__":
    main()
