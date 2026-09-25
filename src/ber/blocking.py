"""S3 — Multi-strategy blocking -> unioned candidate set.

All strategies are country-partitioned (join on country string equality — an
open-set pair property; France or any new label works with zero changes).
No strategy ever branches on a hard-coded country list.

Strategies (union; per-S1 final cap 1000; block cap 2000):
  A  name-bag exact            (normalized/suffix-stripped tokens)
  B  name-bag no-digits        (digit-stripped variant)
  C  phonetic (metaphone over up to 4 core tokens)
  D  postal co-occurrence      (addr_postal, non-empty)
  E  phonetic-2                (metaphone over first 2 core tokens: prefix/multi-word variants)
  F  char-3gram rescue         (top-K per 200k-stratum index, only for S1 rows with 0 candidates so far)

Memory design: per-country processing; exact-key blocks emitted via explode;
pairs deduped per (country chunk) with numpy; per-S1 cap applied at the end.
Inputs: artifacts/normalized/{split}_{s1,s2,s3}.parquet
Output: artifacts/blocking/{split}_candidates.parquet (s1_entity_id, cand_id) + stats
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
STRATUM_ROWS = 200_000


def _collapse(token: str) -> str:
    out, prev = [], ""
    for ch in token:
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def _ph(tokens: list[str]) -> str:
    return " ".join(
        (metaphone(_collapse(t)) if len(t) >= 3 else t) for t in tokens[:4]
    )


def _ph2(tokens: list[str]) -> str:
    return " ".join(
        (metaphone(_collapse(t)) if len(t) >= 3 else t) for t in tokens[:2]
    )


def _char_trigrams(text: str) -> list[str]:
    t = f"  {text} "
    return [t[i:i + 3] for i in range(len(t) - 2)] if len(t) >= 5 else ([t.strip()] if t.strip() else [])


def _add_phonetic_cols(df: pd.DataFrame) -> pd.DataFrame:
    cores = df["name_core"].to_numpy()
    ph = np.empty(len(df), dtype=object)
    ph2 = np.empty(len(df), dtype=object)
    for i, core in enumerate(cores):
        toks = core.split() if core else []
        ph[i] = _ph(toks)
        ph2[i] = _ph2(toks)
    df = df.copy()
    df["ph"] = ph
    df["ph2"] = ph2
    return df


def _blocks_from_frame(df: pd.DataFrame, col: str) -> dict[tuple, np.ndarray]:
    """(col_value) -> ndarray of positional indices, within one country frame.
    Empty/NA keys excluded. Blocks larger than BLOCK_CAP are kept (caller caps emits)."""
    sub = df[df[col] != ""]
    return {k: v for k, v in sub.groupby(col, sort=False).indices.items()}


def _emit_exact_pairs(s1c: pd.DataFrame, s23c: pd.DataFrame, col1: str, col23: str,
                      s1_pos_base: int, s23_base: int) -> np.ndarray:
    """Exact-key join of one country's s1 vs s23 on given columns.
    Returns int64 array of (s1_local_pos + s1_pos_base, s23_local_pos + s23_base) pairs."""
    blocks = _blocks_from_frame(s23c, col23)
    keys = s1c[col1].to_numpy()
    hit_rows, hit_vals = [], []
    for pos, k in enumerate(keys):
        if not k:
            continue
        b = blocks.get(k)
        if b is not None and 0 < len(b) <= BLOCK_CAP:
            hit_rows.append(pos)
            hit_vals.append(b)
    if not hit_rows:
        return np.empty((0, 2), dtype=np.int64)
    rows = np.repeat(np.asarray(hit_rows, dtype=np.int64) + s1_pos_base,
                     [len(v) for v in hit_vals])
    vals = np.concatenate(hit_vals).astype(np.int64) + s23_base
    return np.column_stack([rows, vals])


def _postal_pairs(s1c: pd.DataFrame, s23c: pd.DataFrame,
                  s1_base: int, s23_base: int) -> np.ndarray:
    s1p = s1c["addr_postal"].to_numpy()
    s23_blocks = _blocks_from_frame(s23c, "addr_postal")
    hit_rows, hit_vals = [], []
    for pos, p in enumerate(s1p):
        if not p:
            continue
        b = s23_blocks.get(p)
        if b is not None and 0 < len(b) <= BLOCK_CAP:
            hit_rows.append(pos)
            hit_vals.append(b)
    if not hit_rows:
        return np.empty((0, 2), dtype=np.int64)
    rows = np.repeat(np.asarray(hit_rows, dtype=np.int64) + s1_base,
                     [len(v) for v in hit_vals])
    vals = np.concatenate(hit_vals).astype(np.int64) + s23_base
    return np.column_stack([rows, vals])


def _trigram_rescue(s1c: pd.DataFrame, s23c: pd.DataFrame, uncovered_mask: np.ndarray,
                    s1_base: int, s23_base: int) -> np.ndarray:
    """Top-K trigram neighbors for uncovered S1 rows (own stratum index only)."""
    out_rows, out_vals = [], []
    s23c = s23c.reset_index(drop=True)
    stratum_of = np.arange(len(s23c)) // STRATUM_ROWS
    s1_sub = s1c.loc[uncovered_mask].reset_index(drop=True)
    if not len(s1_sub):
        return np.empty((0, 2), dtype=np.int64)
    s1_stratum = np.arange(len(s1_sub)) // STRATUM_ROWS
    for st in range((len(s23c) + STRATUM_ROWS - 1) // STRATUM_ROWS):
        s23_sl = s23c.iloc[st * STRATUM_ROWS:(st + 1) * STRATUM_ROWS]
        inv: dict[str, list[int]] = defaultdict(list)
        for i, nm in enumerate(s23_sl["name_norm"].to_numpy()):
            for tg in _char_trigrams(nm):
                inv[tg].append(i)
        cids_sl = s23_sl.index.to_numpy()
        s1_sl = s1_sub.iloc[np.where(s1_stratum == st)[0]]
        for pos, nm in enumerate(s1_sl["name_norm"].to_numpy()):
            grams = _char_trigrams(nm)
            if not grams:
                continue
            scores: dict[int, int] = defaultdict(int)
            for tg in set(grams):
                for i in inv.get(tg, ()):
                    scores[i] += 1
            if not scores:
                continue
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:TRIGRAM_TOPK]
            gpos = s1_sl.index[pos]
            out_rows.extend([gpos] * len(top))
            out_vals.extend(int(cids_sl[i]) for i, _ in top)
    if not out_rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([
        np.asarray(out_rows, dtype=np.int64) + s1_base,
        np.asarray(out_vals, dtype=np.int64) + s23_base,
    ])


def _dedup_pairs(pairs: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return pairs
    view = np.ascontiguousarray(pairs).view([("a", np.int64), ("b", np.int64)])
    return np.unique(view).view(np.int64).reshape(-1, 2)


def _cap_per_s1(pairs: np.ndarray, s1_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply per-S1 cap; pairs reference s1 by positional index into s1_ids."""
    if not len(pairs):
        return pairs, 0
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    keep_mask = np.zeros(len(pairs), dtype=bool)
    boundaries = np.searchsorted(pairs[:, 0], np.unique(pairs[:, 0]), side="right")
    start = 0
    n_capped = 0
    for end in boundaries:
        seg = end - start
        if seg > S1_FINAL_CAP:
            keep_mask[start:start + S1_FINAL_CAP] = True
            n_capped += 1
        else:
            keep_mask[start:end] = True
        start = end
    return pairs[keep_mask], n_capped


def _run_split(cfg: dict, split: str, nrm: Path, out_path: Path, params: dict, inputs: dict) -> dict:
    s2 = pd.read_parquet(nrm / f"{split}_s2.parquet")
    s3 = pd.read_parquet(nrm / f"{split}_s3.parquet")
    s1_full = pd.read_parquet(nrm / f"{split}_s1.parquet")
    s2["__src"], s3["__src"] = "s2", "s3"
    s23_full = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    s1_full = _add_phonetic_cols(s1_full)
    s23_full = _add_phonetic_cols(s23_full)
    s1_full["bag_nod"] = s1_full["name_bag"].str.replace(DIGITS_RE, "", regex=True).str.strip()
    s23_full["bag_nod"] = s23_full["name_bag"].str.replace(DIGITS_RE, "", regex=True).str.strip()

    countries = sorted(s23_full["country"].unique().tolist())  # open set: works for France
    print(f"  [{split}] countries: {countries}")
    per_country_parts = []
    stats: dict = {}
    s1_gid = s1_full["entity_id"].to_numpy()
    s23_gid = s23_full["entity_id"].to_numpy()

    for country in countries:
        s1c = s1_full[s1_full["country"] == country].reset_index(drop=True)
        s23c = s23_full[s23_full["country"] == country].reset_index(drop=True)
        parts = []
        a = _emit_exact_pairs(s1c, s23c, "name_bag", "name_bag", 0, 0)
        b = _emit_exact_pairs(s1c, s23c, "bag_nod", "bag_nod", 0, 0)
        c = _emit_exact_pairs(s1c, s23c, "ph", "ph", 0, 0)
        e = _emit_exact_pairs(s1c, s23c, "ph2", "ph2", 0, 0)
        d = _postal_pairs(s1c, s23c, 0, 0)
        for tag, arr in (("A_bag", a), ("B_bag_nod", b), ("C_ph", c), ("E_ph2", e), ("D_postal", d)):
            stats[f"{split}.{country}.{tag}"] = int(len(arr))
            if len(arr):
                parts.append(arr)
        covered = np.zeros(len(s1c), dtype=bool)
        for arr in parts:
            covered[arr[:, 0]] = True
        uncovered = ~covered
        stats[f"{split}.{country}.uncovered_s1"] = int(uncovered.sum())
        if uncovered.any() and cfg["blocking"]["strategies"]["tfidf_lsh"]:
            f = _trigram_rescue(s1c, s23c, uncovered, 0, 0)
            stats[f"{split}.{country}.F_trigram"] = int(len(f))
            if len(f):
                parts.append(f)
        allp = _dedup_pairs(np.concatenate(parts)) if parts else np.empty((0, 2), dtype=np.int64)
        allp, n_capped = _cap_per_s1(allp, s1_gid[s1_full["country"] == country].to_numpy())
        stats[f"{split}.{country}.capped"] = n_capped
        stats[f"{split}.{country}.pairs"] = int(len(allp))
        # pairs are (position within s1c frame, position within s23c frame);
        # resolve to global entity ids HERE while the country slices are at hand.
        s1_ids_c = s1_gid[s1_full["country"].to_numpy() == country]
        s23_ids_c = s23_gid[s23_full["country"].to_numpy() == country]
        if len(allp):
            per_country_parts.append(pd.DataFrame({
                "s1_entity_id": s1_ids_c[allp[:, 0]],
                "cand_id": s23_ids_c[allp[:, 1]],
            }))
        del s1c, s23c, parts, allp
        del s1_ids_c, s23_ids_c

    cand = (pd.concat(per_country_parts, ignore_index=True)
            if per_country_parts else
            pd.DataFrame({"s1_entity_id": pd.Series(dtype=object), "cand_id": pd.Series(dtype=object)}))
    per_country_parts.clear()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_parquet(cand, out_path)
    final = {
        "pairs_total": int(len(cand)),
        "s1_with_candidates": int(cand["s1_entity_id"].nunique()),
        "s1_total": int(len(s1_full)),
        "per_country": {k: v for k, v in stats.items()},
    }
    io_utils.save_manifest(out_path, inputs, params, extra={"stats": final})
    io_utils.json_dump(final, out_path.parent / f"{split}_stats.json")
    print(f"  [{split}] candidates: {final['pairs_total']:,} pairs, "
          f"{final['s1_with_candidates']:,}/{final['s1_total']:,} S1 covered")
    del s1_full, s23_full, cand, pairs
    return final


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm = art / "normalized"
    out_all = {}
    for split in ("train", "test"):
        out_path = art / "blocking" / f"{split}_candidates.parquet"
        inputs = {f"{split}_{s}": nrm / f"{split}_{s}.parquet" for s in ("s1", "s2", "s3")}
        params = {"v": 2, "cap": S1_FINAL_CAP, "block_cap": BLOCK_CAP, "topk": TRIGRAM_TOPK}
        if not force and io_utils.manifest_ok(out_path, inputs, params):
            st = io_utils.json_load(out_path.parent / f"{split}_stats.json")
            print(f"  [{split}] candidates fresh — skipping ({st['pairs_total']:,} pairs)")
            out_all[split] = str(out_path)
            continue
        out_all[split] = _run_split(cfg, split, nrm, out_path, params, inputs)
    return out_all
