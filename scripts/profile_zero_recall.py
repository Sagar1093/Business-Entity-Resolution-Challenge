"""Profile TRUE partners of zero-recall val entities: what signal could
still link them? (name exactness, shared rare addr tokens, addr emptiness,
same-name block sizes). Train GT + val entities only (never test).

Vectorized: global token-df via str.split + Counter; per-entity dicts only
for the ~20k involved rows.
"""
from __future__ import annotations

import sys
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ber import metrics  # noqa: E402

ART = Path("artifacts/normalized")


def main() -> None:
    import yaml
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    manifest = pd.read_parquet("artifacts/splits/split_manifest.parquet")
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    gt = metrics.gt_dict(cfg, only_val=True, manifest=manifest)
    gt = {k: v for k, v in gt.items() if k in val_ids}

    val_cands: dict[str, set[str]] = {}
    for d in ("train_candidates", "train_candidates_g", "train_candidates_h"):
        for p in sorted((ART.parent / "blocking" / d).glob("shard_*.parquet")):
            df = pd.read_parquet(p, columns=["s1_entity_id", "cand_id"])
            df = df[df["s1_entity_id"].isin(val_ids)]
            for s1, g in df.groupby("s1_entity_id")["cand_id"]:
                val_cands.setdefault(s1, set()).update(g.to_numpy())
            del df
    zero = {s1: ts for s1, ts in gt.items() if ts and not (ts & val_cands.get(s1, set()))}
    involved: set[str] = set(zero) | {c for ts in zero.values() for c in ts}
    print(f"zero-recall entities: {len(zero):,}, their true pairs: {sum(len(v) for v in zero.values()):,}")

    need = ["entity_id", "country", "name_norm", "addr_norm", "addr_digits", "addr_postal"]
    s1df = pd.read_parquet(ART / "train_s1.parquet", columns=need).set_index("entity_id")
    parts = []
    for src in ("s2", "s3"):
        t = pd.read_parquet(ART / f"train_{src}.parquet", columns=need)
        parts.append(t)
    allc = pd.concat(parts, ignore_index=True)
    del parts
    allc = allc.set_index("entity_id")

    # global address-token df per country (vectorized split + single Counter)
    df_cnt: dict[str, Counter] = {}
    for country, grp in allc.groupby("country"):
        toks = (grp["addr_norm"].fillna("") + " " + grp["addr_digits"].fillna("")).str.split()
        df_cnt[country] = Counter(chain.from_iterable(toks))
        print(f"  token df built for {country}: {len(df_cnt[country]):,} distinct", flush=True)

    inv = allc.loc[allc.index.intersection(involved)]
    s1_inv = s1df.loc[s1df.index.intersection(set(zero))]

    def toks_of(an: str, ad: str) -> set[str]:
        s = set((str(an) + " " + str(ad)).split())
        return {t for t in s if len(t) >= 2}

    n_pairs = n_exact = n_empty_addr_c = n_empty_addr_s1 = n_rare_tok = n_same_postal = 0
    rare_shared_dist = []
    for s1, ts in zero.items():
        r1 = s1_inv.loc[s1]
        t1 = toks_of(r1["addr_norm"], r1["addr_digits"])
        if not r1["addr_norm"]:
            n_empty_addr_s1 += 1
        for ce in ts:
            rc = inv.loc[ce]
            n_pairs += 1
            if rc["name_norm"] == r1["name_norm"]:
                n_exact += 1
            if not rc["addr_norm"]:
                n_empty_addr_c += 1
            if rc["addr_postal"] and rc["addr_postal"] == r1["addr_postal"]:
                n_same_postal += 1
            cnt = df_cnt[rc["country"]]
            shared = {t for t in (t1 & toks_of(rc["addr_norm"], rc["addr_digits"]))
                      if 3 <= cnt.get(t, 0) <= 5000}
            rare_shared_dist.append(len(shared))
            if shared:
                n_rare_tok += 1
    rare_shared_dist = np.asarray(rare_shared_dist)
    print(f"\ntrue pairs of zero-recall entities: {n_pairs:,}")
    print(f"  exact same name_norm:     {n_exact:,} ({n_exact/n_pairs:.1%})")
    print(f"  S1 addr empty: {n_empty_addr_s1:,}  cand addr empty: {n_empty_addr_c:,} ({n_empty_addr_c/n_pairs:.1%})")
    print(f"  same postal:              {n_same_postal:,} ({n_same_postal/n_pairs:.1%})")
    print(f"  >=1 shared rare addr tok: {n_rare_tok:,} ({n_rare_tok/n_pairs:.1%})")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 100)):
        m = int(((rare_shared_dist >= lo) & (rare_shared_dist < hi)).sum())
        print(f"    shared[{lo},{hi}): {m:6d} ({m/len(rare_shared_dist):.1%})")

    # how big is the exact-name group for the exact-name misses?
    name_counts = Counter()
    for src in ("s2", "s3"):
        nm = pd.read_parquet(ART / f"train_{src}.parquet", columns=["name_norm"])["name_norm"]
        name_counts.update(nm.tolist())
    exact_sizes = []
    for s1, ts in zero.items():
        r1 = s1_inv.loc[s1]
        for ce in ts:
            if inv.loc[ce, "name_norm"] == r1["name_norm"]:
                exact_sizes.append(name_counts.get(r1["name_norm"], 0) + (r1["name_norm"] != ""))
    if exact_sizes:
        exact_sizes = np.asarray(exact_sizes)
        print(f"\nexact-name pairs: same-name S2/S3 group size: median={int(np.median(exact_sizes))}, "
              f"p90={int(np.percentile(exact_sizes, 90))}, max={exact_sizes.max()}, "
              f">100: {(exact_sizes > 100).mean():.1%}")


if __name__ == "__main__":
    main()
