"""S3-H — Address-token blocking for ALL S1 rows (slice-aligned with base).

Why: the audit showed generic-name entities ('balaji investment private
limited') whose exact-name blocks exceed the block cap and whose trigram
top-K is a lottery among thousands of same-name candidates. The diagnostic
probe showed 88% of missed val GT pairs share >=1 RARE address token (digit
runs, street/landmark words) while addr_postal alone is often empty. This
stage indexes addr_norm + addr_digits tokens per country (df-pruned) and
emits top-K candidates scored by rarity-weighted shared tokens:
score = sum over shared tokens of 1/sqrt(df)  (specific matches rank first).

Slice alignment invariant: shards are written for EVERY (country, slice) in
the same sorted-country / 200k-row order as the base stage, so
``iter_candidate_shards`` can pair shard_N with base shard_N. Empty slices
still write an (empty) shard. Resumability is at slice granularity via
progress.json; the split manifest is written only on completion.
Output: artifacts/blocking/{split}_candidates_h/shard_*.parquet
        (s1_entity_id, cand_id)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import io_utils
from .blocking import SLICE_ROWS

TOPK = 200
# Token df window: below DF_MIN a token is a typo-grade artifact; above
# DF_CAP it is too common to be selective (street/city words).
# Defaults; overridden by cfg[blocking][rescue_h] when present.
DF_MIN = 3
DF_CAP = 5000


def _empty_cand() -> pd.DataFrame:
    return pd.DataFrame({"s1_entity_id": pd.Series(dtype=object),
                         "cand_id": pd.Series(dtype=object)})


def _addr_tokens(addr_norm: str, addr_digits: str) -> set[str]:
    toks = set(addr_norm.split())
    toks.update(addr_digits.split())
    return {t for t in toks if len(t) >= 2}


def _prep_s23_country(nrm: Path, split: str, country: str) -> pd.DataFrame:
    frames = []
    for src in ("s2", "s3"):
        path = nrm / f"{split}_{src}.parquet"
        frames.append(pq.read_table(path, columns=["entity_id", "addr_norm", "addr_digits"],
                                    filters=[("country", "=", country)]).to_pandas())
    return pd.concat(frames, ignore_index=True)


def _save_progress(prog_path: Path, completed: dict[str, str]) -> None:
    tmp = prog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"completed": completed}), encoding="utf-8")
    tmp.replace(prog_path)


def _load_progress(out_dir: Path) -> tuple[dict[str, tuple[str, int]], int]:
    """Validate + load completed (country|slice) -> (shard name, rows).

    Corrupt/truncated shards (e.g. from a killed process) are dropped and
    slated for recompute rather than poisoning the union.
    """
    prog_path = out_dir / "progress.json"
    if not prog_path.exists():
        return {}, 0
    completed: dict[str, tuple[str, int]] = {}
    total = 0
    for key, name in io_utils.json_load(prog_path).get("completed", {}).items():
        p = out_dir / name
        try:
            names = pq.ParquetFile(p).schema_arrow.names
            if names != ["s1_entity_id", "cand_id"]:
                raise ValueError(f"bad schema {names}")
            rows = pq.read_metadata(p).num_rows
            completed[key] = (name, rows)
            total += rows
        except Exception as e:
            print(f"  [resume] dropping unusable shard {name} ({e!r}) — will redo {key}", flush=True)
            p.unlink(missing_ok=True)
    return completed, total


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm = art / "normalized"
    rc = cfg.get("blocking", {}).get("rescue_h", {}) or {}
    topk = int(rc.get("topk", TOPK))
    df_min = int(rc.get("df_min", DF_MIN))
    df_cap = int(rc.get("df_cap", DF_CAP))
    out_all = {}
    for split in ("train", "test"):
        out_dir = art / "blocking" / f"{split}_candidates_h"
        inputs = {f"{split}_{s}": nrm / f"{split}_{s}.parquet" for s in ("s1", "s2", "s3")}
        params = {"v": 1, "topk": topk, "slice_rows": SLICE_ROWS,
                  "df_min": df_min, "df_cap": df_cap}
        marker = out_dir / "shard_0.parquet"
        if not force and marker.exists() and io_utils.manifest_ok(marker, inputs, params):
            stats_path = out_dir.parent / f"{split}_stats_h.json"
            st = io_utils.json_load(stats_path) if stats_path.exists() else {"pairs_total": -1}
            print(f"  [{split}] H candidates fresh — skipping ({st['pairs_total']:,} pairs)")
            out_all[split] = str(out_dir)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        prog_path = out_dir / "progress.json"
        if force or not prog_path.exists():
            for old in out_dir.glob("shard_*.parquet"):
                old.unlink()
            for stale in ("shard_0.meta.json", ".keep.meta.json"):
                (out_dir / stale).unlink(missing_ok=True)
            _save_progress(prog_path, {})
            completed: dict[str, tuple[str, int]] = {}
            total = 0
        else:
            completed, total = _load_progress(out_dir)
            if completed:
                print(f"  [{split}] resuming H — {len(completed)} slices already done "
                      f"({total:,} pairs on disk)", flush=True)

        countries = sorted(pd.read_parquet(nrm / f"{split}_s2.parquet",
                                           columns=["country"])["country"].unique().tolist())
        stats: dict = {}
        resumed_slices = len(completed)
        cum_slices = 0
        for country in countries:
            print(f"  [{split}] {country}: building address-token index...", flush=True)
            s23c = _prep_s23_country(nrm, split, country)
            inv: dict[str, list[int]] = defaultdict(list)
            tok_df = defaultdict(int)
            for i, (an, ad) in enumerate(zip(s23c["addr_norm"].to_numpy(),
                                             s23c["addr_digits"].to_numpy())):
                for t in _addr_tokens(an, ad):
                    inv[t].append(i)
            postings: dict[str, np.ndarray] = {}
            for t, rows in inv.items():
                if df_min <= len(rows) <= df_cap:
                    postings[t] = np.asarray(rows, dtype=np.int32)
                    tok_df[t] = len(rows)
            n_pruned = len(inv) - len(postings)
            print(f"  [{split}] {country}: postings {len(postings):,} tokens "
                  f"(pruned {n_pruned:,} with df < {df_min} or > {df_cap:,})", flush=True)
            # rarity weight per token (1/sqrt(df)); df>=DF_MIN guaranteed
            tok_w = {t: 1.0 / np.sqrt(float(tok_df[t])) for t in postings}
            del inv, s23c, tok_df

            s1c_full = pd.read_parquet(nrm / f"{split}_s1.parquet",
                                       columns=["entity_id", "addr_norm", "addr_digits", "country"])
            s1c_full = s1c_full[s1c_full["country"] == country].reset_index(drop=True)
            s23_ids = pd.concat([
                pq.read_table(nrm / f"{split}_{src}.parquet", columns=["entity_id"],
                              filters=[("country", "=", country)]).to_pandas()
                for src in ("s2", "s3")
            ], ignore_index=True)["entity_id"].to_numpy()

            n_slices = (len(s1c_full) + SLICE_ROWS - 1) // SLICE_ROWS
            for si in range(n_slices):
                key = f"{country}|{si}"
                if key in completed:
                    print(f"  [{split}] {country} H slice {si + 1}/{n_slices} — "
                          f"already done ({completed[key][1]:,} pairs)", flush=True)
                    continue
                lo, hi = si * SLICE_ROWS, min((si + 1) * SLICE_ROWS, len(s1c_full))
                sl = s1c_full.iloc[lo:hi]
                out_rows: list[int] = []
                out_vals: list[int] = []
                for pos, (an, ad) in enumerate(zip(sl["addr_norm"].to_numpy(),
                                                   sl["addr_digits"].to_numpy())):
                    toks = [t for t in _addr_tokens(an, ad) if t in postings]
                    if not toks:
                        continue
                    cand_arr = np.concatenate([postings[t] for t in toks])
                    w_arr = np.concatenate([np.full(len(postings[t]), tok_w[t], dtype=np.float32)
                                            for t in toks])
                    uniq, inv = np.unique(cand_arr, return_inverse=True)
                    score = np.bincount(inv, weights=w_arr, minlength=len(uniq)).astype(np.float32)
                    k = min(TOPK, len(uniq))
                    top = np.argpartition(-score, k - 1)[:k]
                    out_rows.extend([pos] * k)
                    out_vals.extend(int(uniq[i]) for i in top)
                    stats[f"{split}.{country}.H"] = stats.get(f"{split}.{country}.H", 0) + k
                if out_rows:
                    cand = pd.DataFrame({
                        "s1_entity_id": sl["entity_id"].to_numpy()[np.asarray(out_rows, dtype=np.int64)],
                        "cand_id": s23_ids[np.asarray(out_vals, dtype=np.int64)],
                    })
                else:
                    cand = _empty_cand()  # keep 1:1 slice alignment with base shards
                shard_name = f"shard_{cum_slices + si}.parquet"
                io_utils.write_parquet(cand, out_dir / shard_name, 500_000)
                total += len(cand)
                completed[key] = (shard_name, len(cand))
                _save_progress(prog_path, {k: v[0] for k, v in completed.items()})
                del cand
                print(f"  [{split}] {country} H slice {si + 1}/{n_slices} — cumulative {total:,}", flush=True)
            cum_slices += n_slices
            del postings, s1c_full, s23_ids, tok_w
            import gc
            gc.collect()

        final = {"pairs_total": int(total), "shards": int(cum_slices),
                 "topk": topk, "resumed_slices": int(resumed_slices)}
        io_utils.save_manifest(marker if cum_slices else out_dir / ".keep", inputs, params,
                               extra={"stats": final})
        io_utils.json_dump(final, out_dir.parent / f"{split}_stats_h.json")
        prog_path.unlink(missing_ok=True)
        print(f"  [{split}] H candidates: {final['pairs_total']:,} pairs across {final['shards']} slices "
              f"({resumed_slices} resumed)", flush=True)
        out_all[split] = str(out_dir)
    return out_all
