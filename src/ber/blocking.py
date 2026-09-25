"""S3 — Multi-strategy blocking -> unioned candidate set (memory-safe, sharded).

All strategies are country-partitioned (join on country string equality — an
open-set pair property; France or any new label works with zero changes).

Strategies (union; block cap 2000; per-S1 final cap 1000):
  A  name-bag exact, B digit-stripped bag, C metaphone(4), E metaphone(2),
  D postal co-occurrence, F char-3gram top-K rescue for uncovered S1 rows.

Memory design (fixed OOM): S1 is processed in 200k-row slices; per slice we
emit candidate pairs for all strategies (int32 positions), dedup, apply the
per-S1 cap, resolve to entity-id string pairs, and append a shard parquet.
Peak RAM ~2-3 GB regardless of total pair volume (measured 165M+ raw pairs
from name-bag alone on train).
Output: artifacts/blocking/{split}_candidates/shard_*.parquet
        (pd.read_parquet / iter_parquet_shards read the directory transparently)
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from jellyfish import metaphone

from . import io_utils

DIGITS_RE = re.compile(r"\d")
S1_FINAL_CAP = 1000
BLOCK_CAP = 2000
TRIGRAM_TOPK = 100
SLICE_ROWS = 200_000


def _collapse(token: str) -> str:
    out, prev = [], ""
    for ch in token:
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def _ph(tokens: list[str]) -> str:
    return " ".join((metaphone(_collapse(t)) if len(t) >= 3 else t) for t in tokens[:4])


def _ph2(tokens: list[str]) -> str:
    return " ".join((metaphone(_collapse(t)) if len(t) >= 3 else t) for t in tokens[:2])


def _char_trigrams(text: str) -> list[str]:
    t = f"  {text} "
    return [t[i:i + 3] for i in range(len(t) - 2)] if len(t) >= 5 else ([t.strip()] if t.strip() else [])


def _add_phonetic_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ph = np.empty(len(df), dtype=object)
    ph2 = np.empty(len(df), dtype=object)
    for i, core in enumerate(df["name_core"].to_numpy()):
        toks = core.split() if core else []
        ph[i] = _ph(toks)
        ph2[i] = _ph2(toks)
    df["ph"], df["ph2"] = ph, ph2
    return df


def _blocks(s23c: pd.DataFrame, col: str) -> dict[str, np.ndarray]:
    sub = s23c[s23c[col] != ""]
    return {k: v for k, v in sub.groupby(col, sort=False).indices.items()}


def _emit_exact(s1c: pd.DataFrame, blocks: dict[str, np.ndarray], col: str,
                stats_counter: dict, tag: str) -> np.ndarray:
    """(s1_local_pos int32, s23_local_pos int32) pairs for one strategy on one slice."""
    keys = s1c[col].to_numpy()
    hit_rows: list[int] = []
    hit_vals: list[np.ndarray] = []
    for pos, k in enumerate(keys):
        if not k:
            continue
        b = blocks.get(k)
        if b is not None and 0 < len(b) <= BLOCK_CAP:
            hit_rows.append(pos)
            hit_vals.append(b)
            stats_counter[tag] = stats_counter.get(tag, 0) + len(b)
    if not hit_rows:
        return np.empty((0, 2), dtype=np.int32)
    rows = np.repeat(np.asarray(hit_rows, dtype=np.int32),
                     np.fromiter((len(v) for v in hit_vals), dtype=np.int32, count=len(hit_vals)))
    vals = np.concatenate(hit_vals).astype(np.int32)
    return np.column_stack([rows, vals])


def _dedup(pairs: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return pairs
    view = np.ascontiguousarray(pairs).view([("a", np.int32), ("b", np.int32)])
    return np.unique(view).view(np.int32).reshape(-1, 2)


def _cap_per_s1(pairs: np.ndarray) -> tuple[np.ndarray, int]:
    if not len(pairs):
        return pairs, 0
    order = np.argsort(pairs[:, 0], kind="stable")
    p = pairs[order]
    n_capped = 0
    keep = np.zeros(len(p), dtype=bool)
    uniq, starts = np.unique(p[:, 0], return_index=True)
    starts = list(starts) + [len(p)]
    for i in range(len(uniq)):
        s, e = starts[i], starts[i + 1]
        if e - s > S1_FINAL_CAP:
            keep[s:s + S1_FINAL_CAP] = True
            n_capped += 1
        else:
            keep[s:e] = True
    return p[keep], n_capped


def build_trigram_index(s23c: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Country-level inverted index: gram -> int32 postings array (built ONCE per country)."""
    inv: dict[str, list[int]] = defaultdict(list)
    for i, nm in enumerate(s23c["name_norm"].to_numpy()):
        for tg in _char_trigrams(nm):
            inv[tg].append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in inv.items()}, s23c.index.to_numpy(dtype=np.int32)


def _trigram_rescue(s1c: pd.DataFrame, index: tuple[dict[str, np.ndarray], np.ndarray],
                    uncovered: np.ndarray, stats_counter: dict) -> np.ndarray:
    """Top-K trigram neighbors for uncovered S1 rows using the prebuilt index.
    Vectorized scoring via postings concatenation + np.unique/np.bincount."""
    postings, _ = index
    idxs = np.where(uncovered)[0]
    if not len(idxs):
        return np.empty((0, 2), dtype=np.int32)
    out_rows: list[int] = []
    out_vals: list[int] = []
    norms = s1c["name_norm"].to_numpy()
    for pos in idxs:
        grams = set(_char_trigrams(norms[pos]))
        grams = [g for g in grams if g in postings]
        if not grams:
            continue
        cand_arr = np.concatenate([postings[g] for g in grams])
        uniq, counts = np.unique(cand_arr, return_counts=True)
        if len(uniq) > TRIGRAM_TOPK:
            top_idx = np.argpartition(-counts, TRIGRAM_TOPK)[:TRIGRAM_TOPK]
            uniq = uniq[top_idx]
        out_rows.extend([int(pos)] * len(uniq))
        out_vals.extend(int(i) for i in uniq)
        stats_counter["F_trigram"] = stats_counter.get("F_trigram", 0) + len(uniq)
    if not out_rows:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([np.asarray(out_rows, dtype=np.int32),
                            np.asarray(out_vals, dtype=np.int32)])


BLOCK_COLS = ["entity_id", "country", "name_norm", "name_core", "name_bag",
              "name_key_phonetic", "addr_postal", "name_tok_n"]


def _prep_s23_country(nrm: Path, split: str, country: str) -> pd.DataFrame:
    """Load ONLY one country's S2/S3 rows with blocking columns (RAM-bounded).
    Reads country-partitioned row groups via pyarrow filtering when available."""
    import pyarrow.parquet as pq

    frames = []
    for src in ("s2", "s3"):
        path = nrm / f"{split}_{src}.parquet"
        names = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in BLOCK_COLS if c in names]
        tbl = pq.read_table(path, columns=cols, filters=[("country", "=", country)])
        frames.append(tbl.to_pandas())
        del tbl
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = _add_phonetic_cols(df)
    df["bag_nod"] = df["name_bag"].str.replace(DIGITS_RE, "", regex=True).str.strip()
    return df


def _s1_country_frame(s1_full: pd.DataFrame, country: str) -> pd.DataFrame:
    df = s1_full[s1_full["country"] == country].reset_index(drop=True)
    return df


def _run_split(cfg: dict, split: str, nrm: Path, out_dir: Path, params: dict, inputs: dict) -> dict:
    print(f"  [{split}] loading S1 + country inventory...", flush=True)
    s1_full = _add_phonetic_cols(pd.read_parquet(nrm / f"{split}_s1.parquet"))
    s1_full["bag_nod"] = s1_full["name_bag"].str.replace(DIGITS_RE, "", regex=True).str.strip()
    countries = sorted(pd.read_parquet(nrm / f"{split}_s2.parquet", columns=["country"])["country"].unique().tolist())
    print(f"  [{split}] countries: {countries}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("shard_*.parquet"):
        old.unlink()

    stats: dict = {}
    shard_id, total_pairs = 0, 0
    n_s1_total = len(s1_full)

    s23_gid_by_country: dict[str, np.ndarray] = {}
    for country in countries:
        print(f"  [{split}] {country}: loading S2/S3 slice...", flush=True)
        s23c_full = _prep_s23_country(nrm, split, country)
        s23_gid_c = s23c_full["entity_id"].to_numpy()
        s1c_full = _s1_country_frame(s1_full, country)
        s1_ids_c = s1c_full["entity_id"].to_numpy()

        # precompute strategy blocks + trigram index ONCE per country
        blocks = {
            "A": _blocks(s23c_full, "name_bag"),
            "B": _blocks(s23c_full, "bag_nod"),
            "C": _blocks(s23c_full, "ph"),
            "E": _blocks(s23c_full, "ph2"),
            "D": _blocks(s23c_full, "addr_postal"),
        }
        tri_index = build_trigram_index(s23c_full) if cfg["blocking"]["strategies"]["tfidf_lsh"] else None
        print(f"  [{split}] {country}: S1={len(s1_ids_c):,} S23={len(s23c_full):,} — processing slices...", flush=True)

        n_slices = (len(s1_ids_c) + SLICE_ROWS - 1) // SLICE_ROWS
        for si in range(n_slices):
            lo, hi = si * SLICE_ROWS, min((si + 1) * SLICE_ROWS, len(s1_ids_c))
            s1c = s1c_full.iloc[lo:hi].reset_index(drop=True)
            parts: list[np.ndarray] = []
            for tag, col in (("A_bag", "name_bag"), ("B_bag_nod", "bag_nod"),
                             ("C_ph", "ph"), ("E_ph2", "ph2"), ("D_postal", "addr_postal")):
                arr = _emit_exact(s1c, blocks[tag[0]], col, stats, f"{split}.{country}.{tag}")
                if len(arr):
                    parts.append(arr)
            if tri_index is not None:
                covered = np.zeros(len(s1c), dtype=bool)
                for arr in parts:
                    covered[arr[:, 0]] = True
                f = _trigram_rescue(s1c, tri_index, ~covered, stats)
                if len(f):
                    parts.append(f)
            allp = _dedup(np.concatenate(parts)) if parts else np.empty((0, 2), dtype=np.int32)
            del parts
            allp, n_capped = _cap_per_s1(allp)
            if len(allp):
                cand = pd.DataFrame({
                    "s1_entity_id": s1_ids_c[allp[:, 0] + lo],
                    "cand_id": s23_gid_c[allp[:, 1]],
                })
                io_utils.write_parquet(cand, out_dir / f"shard_{shard_id}.parquet", 200_000)
                shard_id += 1
                total_pairs += len(cand)
                del cand
            stats[f"{split}.{country}.slices.capped"] = stats.get(f"{split}.{country}.slices.capped", 0) + n_capped
            if si % 5 == 0:
                print(f"  [{split}] {country} slice {si + 1}/{n_slices} — cumulative pairs {total_pairs:,}", flush=True)
            del allp, s1c
        del blocks, s23c_full, tri_index, s23_gid_c, s1c_full, s1_ids_c
        import gc
        gc.collect()

    final = {
        "pairs_total": int(total_pairs),
        "shards": shard_id,
        "s1_total": int(n_s1_total),
        "per_strategy": {k: v for k, v in stats.items() if not k.endswith("capped")},
    }
    io_utils.save_manifest(out_dir / "shard_0.parquet" if shard_id else out_dir / ".keep", inputs, params,
                           extra={"stats": final})
    io_utils.json_dump(final, out_dir.parent / f"{split}_stats.json")
    print(f"  [{split}] candidates: {final['pairs_total']:,} pairs across {shard_id} shards", flush=True)
    del s1_full
    return final


def iter_candidate_shards(split: str, cfg: dict):
    """Yield UNIONED+DEDUPED candidate DataFrames shard-by-shard.

    Merges the base shards with the slice-aligned G (trigram) shards when
    present; dedup via packed int64 keys. Consumed by features/outputs.
    """
    out_dir = Path(cfg["paths"]["artifacts_dir"]) / "blocking" / f"{split}_candidates"
    g_dir = Path(cfg["paths"]["artifacts_dir"]) / "blocking" / f"{split}_candidates_g"
    g_files = {p.name: p for p in sorted(g_dir.glob("shard_*.parquet"))} if g_dir.exists() else {}
    for p in sorted(out_dir.glob("shard_*.parquet")):
        base = pd.read_parquet(p)
        g = g_files.get(p.name)
        if g is not None:
            base = pd.concat([base, pd.read_parquet(g, columns=["s1_entity_id", "cand_id"])],
                             ignore_index=True)
            pk = base["s1_entity_id"].to_numpy().astype(np.int64) << np.int64(32) | \
                pd.factorize(base["cand_id"])[0].astype(np.int64)
            # factorize ids differ per shard -> dedup on string pair instead:
            base = base.drop_duplicates(subset=["s1_entity_id", "cand_id"])
            del pk
        yield base


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm = art / "normalized"
    out_all = {}
    for split in ("train", "test"):
        out_dir = art / "blocking" / f"{split}_candidates"
        inputs = {f"{split}_{s}": nrm / f"{split}_{s}.parquet" for s in ("s1", "s2", "s3")}
        params = {"v": 3, "cap": S1_FINAL_CAP, "block_cap": BLOCK_CAP, "topk": TRIGRAM_TOPK}
        marker = out_dir / "shard_0.parquet"
        if not force and marker.exists() and io_utils.manifest_ok(marker, inputs, params):
            st = io_utils.json_load(out_dir.parent / f"{split}_stats.json")
            print(f"  [{split}] candidates fresh — skipping ({st['pairs_total']:,} pairs)")
            out_all[split] = str(out_dir)
            continue
        out_all[split] = _run_split(cfg, split, nrm, out_dir, params, inputs)
    return out_all
