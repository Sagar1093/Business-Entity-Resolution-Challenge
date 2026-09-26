"""S3-G — Incremental trigram blocking for ALL S1 rows (slice-aligned with base).

Why: the audit showed heavy-noise true pairs (e.g. 'burger solution' vs
'burger solutino') missed when an S1 entity already had candidates from
exact/phonetic keys — the old rescue ran only for zero-candidate rows. This
stage runs char-3gram top-K retrieval for EVERY S1 row with
coverage-normalized scoring: score = |shared grams| / min(|gA|, |gB|)
(ranking by how much of the shorter name is explained).

Slice alignment invariant (v2): shards are written for EVERY (country, slice)
in the same sorted-country / 200k-row order as the base stage, so
``iter_candidate_shards`` can pair shard_N with base shard_N. Empty slices
still write an (empty) shard to keep the 1:1 mapping. Resumability is at
slice granularity via progress.json (key -> shard name; shard row counts are
re-validated on load); the split manifest is written only on completion.
Output: artifacts/blocking/{split}_candidates_g/shard_*.parquet
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

TOPK = 100
# Prune postings for super-common grams: they carry no discriminative signal
# (every name contains ' a ', 'in', 'an'...) and blow up the per-row cost.
# df threshold is relative to the per-country S2/S3 size.
DF_FRACTION = 0.005
MIN_DF_ABS = 3


def _empty_cand() -> pd.DataFrame:
    return pd.DataFrame({"s1_entity_id": pd.Series(dtype=object),
                         "cand_id": pd.Series(dtype=object)})


def _char_trigrams(text: str) -> list[str]:
    t = f"  {text} "
    return [t[i:i + 3] for i in range(len(t) - 2)] if len(t) >= 5 else ([t.strip()] if t.strip() else [])


def _prep_s23_country(nrm: Path, split: str, country: str) -> pd.DataFrame:
    frames = []
    for src in ("s2", "s3"):
        path = nrm / f"{split}_{src}.parquet"
        frames.append(pq.read_table(path, columns=["entity_id", "name_norm"],
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
    out_all = {}
    for split in ("train", "test"):
        out_dir = art / "blocking" / f"{split}_candidates_g"
        inputs = {f"{split}_{s}": nrm / f"{split}_{s}.parquet" for s in ("s1", "s2", "s3")}
        params = {"v": 2, "topk": TOPK, "slice_rows": SLICE_ROWS,
                  "df_fraction": DF_FRACTION, "min_df_abs": MIN_DF_ABS}
        marker = out_dir / "shard_0.parquet"
        if not force and marker.exists() and io_utils.manifest_ok(marker, inputs, params):
            stats_path = out_dir.parent / f"{split}_stats_g.json"
            st = io_utils.json_load(stats_path) if stats_path.exists() else {"pairs_total": -1}
            print(f"  [{split}] G candidates fresh — skipping ({st['pairs_total']:,} pairs)")
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
                print(f"  [{split}] resuming G — {len(completed)} slices already done "
                      f"({total:,} pairs on disk)", flush=True)

        countries = sorted(pd.read_parquet(nrm / f"{split}_s2.parquet",
                                           columns=["country"])["country"].unique().tolist())
        stats: dict = {}
        resumed_slices = len(completed)
        cum_slices = 0
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
            s23_ids = pd.concat([
                pq.read_table(nrm / f"{split}_{src}.parquet", columns=["entity_id"],
                              filters=[("country", "=", country)]).to_pandas()
                for src in ("s2", "s3")
            ], ignore_index=True)["entity_id"].to_numpy()

            n_slices = (len(s1c_full) + SLICE_ROWS - 1) // SLICE_ROWS
            for si in range(n_slices):
                key = f"{country}|{si}"
                if key in completed:
                    print(f"  [{split}] {country} G slice {si + 1}/{n_slices} — "
                          f"already done ({completed[key][1]:,} pairs)", flush=True)
                    continue
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
                else:
                    cand = _empty_cand()  # keep 1:1 slice alignment with base shards
                shard_name = f"shard_{cum_slices + si}.parquet"
                io_utils.write_parquet(cand, out_dir / shard_name, 500_000)
                total += len(cand)
                completed[key] = (shard_name, len(cand))
                _save_progress(prog_path, {k: v[0] for k, v in completed.items()})
                del cand
                print(f"  [{split}] {country} G slice {si + 1}/{n_slices} — cumulative {total:,}", flush=True)
            cum_slices += n_slices
            del postings, s1c_full, s23_ids, gram_counts
            import gc
            gc.collect()

        final = {"pairs_total": int(total), "shards": int(cum_slices),
                 "topk": TOPK, "resumed_slices": int(resumed_slices)}
        io_utils.save_manifest(marker if cum_slices else out_dir / ".keep", inputs, params,
                               extra={"stats": final})
        io_utils.json_dump(final, out_dir.parent / f"{split}_stats_g.json")
        prog_path.unlink(missing_ok=True)
        print(f"  [{split}] G candidates: {final['pairs_total']:,} pairs across {final['shards']} slices "
              f"({resumed_slices} resumed)", flush=True)
        out_all[split] = str(out_dir)
    return out_all
