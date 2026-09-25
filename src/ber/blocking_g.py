"""S3-G — Incremental trigram blocking for ALL S1 rows (slice-aligned with base).

Why: the audit showed heavy-noise true pairs (e.g. 'burger solution' vs
'burger solutino') missed when an S1 entity already had candidates from
exact/phonetic keys — the old rescue ran only for zero-candidate rows. This
stage runs char-3gram top-K retrieval for EVERY S1 row with
coverage-normalized scoring: score = |shared grams| / min(|gA|, |gB|)
(ranking by how much of the shorter name is explained).

Shards are slice-aligned with base (same sorted-country order, same
200k-row slices) so S4 can union + dedup per slice in index space.
Output: artifacts/blocking/{split}_candidates_g/shard_*.parquet
        (s1_entity_id, cand_id)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils

SLICE_ROWS = 200_000
TOPK = 100
# Prune postings for super-common grams: they carry no discriminative signal
# (every name contains ' a ', 'in', 'an'...) and blow up the per-row cost.
# df threshold is relative to the per-country slice size.
DF_FRACTION = 0.005
MIN_DF_ABS = 3


def _char_trigrams(text: str) -> list[str]:
    t = f"  {text} "
    return [t[i:i + 3] for i in range(len(t) - 2)] if len(t) >= 5 else ([t.strip()] if t.strip() else [])


def _prep_s23_country(nrm: Path, split: str, country: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for src in ("s2", "s3"):
        path = nrm / f"{split}_{src}.parquet"
        frames.append(pq.read_table(path, columns=["entity_id", "name_norm"],
                                    filters=[("country", "=", country)]).to_pandas())
    return pd.concat(frames, ignore_index=True)


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm = art / "normalized"
    out_all = {}
    for split in ("train", "test"):
        out_dir = art / "blocking" / f"{split}_candidates_g"
        inputs = {f"{split}_{s}": nrm / f"{split}_{s}.parquet" for s in ("s1", "s2", "s3")}
        params = {"v": 1, "topk": TOPK, "slice_rows": SLICE_ROWS}
        marker = out_dir / "shard_0.parquet"
        if not force and marker.exists() and io_utils.manifest_ok(marker, inputs, params):
            st = io_utils.json_load(out_dir.parent / f"{split}_stats_g.json")
            print(f"  [{split}] G candidates fresh — skipping ({st['pairs_total']:,} pairs)")
            out_all[split] = str(out_dir)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("shard_*.parquet"):
            old.unlink()

        countries = sorted(pd.read_parquet(nrm / f"{split}_s2.parquet",
                                           columns=["country"])["country"].unique().tolist())
        stats: dict = {}
        shard_id, total = 0, 0
        for country in countries:
            print(f"  [{split}] {country}: building trigram index...", flush=True)
            s23c = _prep_s23_country(nrm, split, country)
            inv: dict[str, list[int]] = defaultdict(list)
            gram_counts = np.empty(len(s23c), dtype=np.float32)
            norms = s23c["name_norm"].to_numpy()
            for i, nm in enumerate(norms):
                grams = _char_trigrams(nm)
                gram_counts[i] = len(grams)
                for tg in set(grams):  # set: postings dedup per candidate
                    inv[tg].append(i)
            df_cap = max(MIN_DF_ABS, int(DF_FRACTION * len(s23c)))
            postings = {k: np.asarray(v, dtype=np.int32) for k, v in inv.items()
                        if len(v) <= df_cap}
            n_pruned = len(inv) - len(postings)
            print(f"  [{split}] {country}: postings {len(postings):,} grams "
                  f"(pruned {n_pruned:,} with df > {df_cap:,})", flush=True)
            del inv, s23c

            s1c_full = pd.read_parquet(nrm / f"{split}_s1.parquet",
                                       columns=["entity_id", "name_norm", "country"])
            s1c_full = s1c_full[s1c_full["country"] == country].reset_index(drop=True)
            s23_ids_c = None
            import pyarrow.parquet as pq
            s23_ids = pd.concat([
                pq.read_table(nrm / f"{split}_{src}.parquet", columns=["entity_id"],
                              filters=[("country", "=", country)]).to_pandas()
                for src in ("s2", "s3")
            ], ignore_index=True)["entity_id"].to_numpy()

            n_slices = (len(s1c_full) + SLICE_ROWS - 1) // SLICE_ROWS
            for si in range(n_slices):
                lo, hi = si * SLICE_ROWS, min((si + 1) * SLICE_ROWS, len(s1c_full))
                sl = s1c_full.iloc[lo:hi]
                out_rows: list[int] = []
                out_vals: list[int] = []
                for pos, nm in enumerate(sl["name_norm"].to_numpy()):
                    grams = [g for g in set(_char_trigrams(nm)) if g in postings]
                    if not grams:
                        continue
                    cand_arr = np.concatenate([postings[g] for g in grams])
                    uniq, counts = np.unique(cand_arr, return_counts=True)
                    score = counts / np.minimum(np.full(len(uniq), max(1, len(grams))),
                                                np.maximum(gram_counts[uniq], 1.0))
                    k = min(TOPK, len(uniq))
                    top = np.argpartition(-score, k - 1)[:k]
                    out_rows.extend([pos] * k)
                    out_vals.extend(int(uniq[i]) for i in top)
                    stats[f"{split}.{country}.G"] = stats.get(f"{split}.{country}.G", 0) + k
                if out_rows:
                    cand = pd.DataFrame({
                        "s1_entity_id": sl["entity_id"].to_numpy()[np.asarray(out_rows, dtype=np.int64)],
                        "cand_id": s23_ids[np.asarray(out_vals, dtype=np.int64)],
                    })
                    io_utils.write_parquet(cand, out_dir / f"shard_{shard_id}.parquet", 500_000)
                    shard_id += 1
                    total += len(cand)
                    del cand
                print(f"  [{split}] {country} G slice {si + 1}/{n_slices} — cumulative {total:,}", flush=True)
            del postings, s1c_full, s23_ids, gram_counts
            import gc
            gc.collect()

        final = {"pairs_total": int(total), "shards": shard_id, "topk": TOPK}
        io_utils.save_manifest(marker if shard_id else out_dir / ".keep", inputs, params,
                               extra={"stats": final})
        io_utils.json_dump(final, out_dir.parent / f"{split}_stats_g.json")
        print(f"  [{split}] G candidates: {final['pairs_total']:,} pairs across {shard_id} shards")
        out_all[split] = str(out_dir)
    return out_all
