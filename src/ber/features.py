"""S4 — Pair feature engineering over candidate pairs (vectorized, sharded).

Features per (s1, cand) pair (all open-set; no country branching):
  name:  token jaccard/containment/dice, exact bag, char-3gram jaccard/dice,
         normalized lev ratio, jaro-winkler, token-set diff counts,
         phonetic equality, initials match, len ratio, tok-count diff
  addr:  token jaccard/containment, char-3gram jaccard, lev, postal eq/both-empty,
         digit-run jaccard, landmark co-flag, parts overlap, len ratio, empty flag
  cross: country equality (learned feature), is_s3 source flag, combined lev

Design: precompute per-record token/gram frozensets once per split (positional
arrays); per 5M-pair chunk gather rows via index->position mapping; compute with
numpy + rapidfuzz.process.cpdist; write rolling shards
artifacts/features/{split}_pairs/shard_*.parquet (pd.read_parquet reads the
directory transparently later).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import distance, process

from . import io_utils

DIGITS_RE = re.compile(r"\d+")

FEATURES = [
    "n_tok_jac", "n_tok_cont", "n_tok_dice", "n_tok_miss", "n_tok_extra",
    "n_c3_jac", "n_c3_dice", "n_lev", "n_jw", "n_exact", "n_phonetic",
    "n_initials", "n_len_ratio", "n_tok_n_diff",
    "a_tok_jac", "a_tok_cont", "a_c3_jac", "a_lev", "a_postal_eq",
    "a_postal_both_empty", "a_digits_jac", "a_landmark_both",
    "a_parts_overlap", "a_len_ratio", "a_empty",
    "x_country_eq", "x_is_s3", "x_combined_lev",
]

_NAME_COLS = ["name_norm", "name_core", "name_key_phonetic", "name_tok_n"]
_ADDR_COLS = ["addr_norm", "addr_postal", "addr_digits", "addr_parts", "addr_landmark"]


def _char3(text: str) -> frozenset[str]:
    t = f"  {text} "
    return frozenset(t[i:i + 3] for i in range(len(t) - 2)) if len(t) >= 5 else (
        frozenset({t.strip()}) if t.strip() else frozenset())


def _fset_words(text: str) -> frozenset[str]:
    return frozenset(text.split()) if text else frozenset()


class SideArrays:
    """Positional lookup structures for one normalized source frame."""

    def __len__(self) -> int:
        return len(self.index)

    def __init__(self, df: pd.DataFrame):
        self.index = pd.Index(df["entity_id"])
        self.country = df["country"].to_numpy()
        self.name_norm = df["name_norm"].to_numpy()
        self.name_core = df["name_core"].to_numpy()
        self.ph = df["name_key_phonetic"].to_numpy()
        self.name_tok_n = df["name_tok_n"].to_numpy(dtype=np.float32)
        self.addr_norm = df["addr_norm"].to_numpy()
        self.postal = df["addr_postal"].to_numpy()
        self.addr_parts = df["addr_parts"].to_numpy()
        self.landmark = df["addr_landmark"].to_numpy(dtype=np.float32)
        self.core_sets = np.array([_fset_words(x) for x in self.name_core], dtype=object)
        self.norm_c3 = np.array([_char3(x) for x in self.name_norm], dtype=object)
        self.addr_sets = np.array([_fset_words(x) for x in self.addr_norm], dtype=object)
        self.addr_c3 = np.array([_char3(x) for x in self.addr_norm], dtype=object)
        self.digit_sets = np.array([frozenset(DIGITS_RE.findall(x)) for x in self.addr_norm], dtype=object)
        self.parts_sets = np.array([
            frozenset(p for p in x.split("|") if p) for x in self.addr_parts
        ], dtype=object)
        self.initials = np.array([
            "".join(t[0] for t in x.split()) if x else "" for x in self.name_core
        ], dtype=object)


def _set_pair_stats(sa: np.ndarray, sb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (jaccard, max-containment, dice) over aligned object arrays of frozensets."""
    n = len(sa)
    inter = np.empty(n, dtype=np.float32)
    la = np.fromiter((len(x) for x in sa), dtype=np.float32, count=n)
    lb = np.fromiter((len(x) for x in sb), dtype=np.float32, count=n)
    for i in range(n):
        inter[i] = len(sa[i] & sb[i])
    union = la + lb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
        cont = np.where(np.maximum(la, lb) > 0, inter / np.maximum(la, lb), 0.0)
        dice = np.where((la + lb) > 0, 2 * inter / (la + lb), 0.0)
    return jac.astype(np.float32), cont.astype(np.float32), dice.astype(np.float32)


def _gram_pair_stats(ga: np.ndarray, gb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(ga)
    inter = np.empty(n, dtype=np.float32)
    la = np.fromiter((len(x) for x in ga), dtype=np.float32, count=n)
    lb = np.fromiter((len(x) for x in gb), dtype=np.float32, count=n)
    for i in range(n):
        inter[i] = len(ga[i] & gb[i])
    union = la + lb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
        dice = np.where((la + lb) > 0, 2 * inter / (la + lb), 0.0)
    return jac.astype(np.float32), dice.astype(np.float32)


def _lev_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return process.cpdist(a, b, scorer=distance.Levenshtein.normalized_similarity).astype(np.float32)


def _jw_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return process.cpdist(a, b, scorer=distance.JaroWinkler.normalized_similarity).astype(np.float32)


def compute_chunk(A: SideArrays, B: SideArrays, p1: np.ndarray, p2: np.ndarray,
                  is_s3: np.ndarray) -> pd.DataFrame:
    """All features for aligned position arrays p1 (into A) and p2 (into B)."""
    n_tok_jac, n_tok_cont, n_tok_dice = _set_pair_stats(A.core_sets[p1], B.core_sets[p2])
    n_c3_jac, n_c3_dice = _gram_pair_stats(A.norm_c3[p1], B.norm_c3[p2])
    a_tok_jac, a_tok_cont, _ = _set_pair_stats(A.addr_sets[p1], B.addr_sets[p2])
    a_c3_jac, _ = _gram_pair_stats(A.addr_c3[p1], B.addr_c3[p2])

    name_a, name_b = A.name_norm[p1], B.name_norm[p2]
    addr_a, addr_b = A.addr_norm[p1], B.addr_norm[p2]
    postal_a, postal_b = A.postal[p1], B.postal[p2]

    a_lev = _lev_sim(addr_a, addr_b)
    both_empty = (addr_a == "") & (addr_b == "")
    a_lev[both_empty] = 0.0

    full_a = np.char.add(np.char.add(name_a.astype(str), " "), addr_a.astype(str))
    full_b = np.char.add(np.char.add(name_b.astype(str), " "), addr_b.astype(str))
    x_lev = _lev_sim(full_a, full_b)
    x_lev[both_empty & (name_a == "") & (name_b == "")] = 0.0

    def len_ratio(x, y):
        lx = np.asarray([len(s) for s in x], dtype=np.float32)
        ly = np.asarray([len(s) for s in y], dtype=np.float32)
        mx = np.maximum(lx, ly)
        return np.where(mx > 0, np.minimum(lx, ly) / mx, 1.0)

    d_a = A.digit_sets[p1]
    d_b = B.digit_sets[p2]
    dig_inter = np.fromiter((len(x & y) for x, y in zip(d_a, d_b)), dtype=np.float32, count=len(d_a))
    dig_union = np.fromiter((len(x | y) for x, y in zip(d_a, d_b)), dtype=np.float32, count=len(d_a))
    parts_a, parts_b = A.parts_sets[p1], B.parts_sets[p2]
    parts_inter = np.fromiter((len(x & y) for x, y in zip(parts_a, parts_b)), dtype=np.float32, count=len(parts_a))
    parts_min = np.fromiter((min(len(x), len(y)) for x, y in zip(parts_a, parts_b)), dtype=np.float32, count=len(parts_a))

    sa, sb = A.core_sets[p1], B.core_sets[p2]
    n_miss = np.fromiter((len(x - y) for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))
    n_extra = np.fromiter((len(y - x) for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))

    f = {
        "n_tok_jac": n_tok_jac,
        "n_tok_cont": n_tok_cont,
        "n_tok_dice": n_tok_dice,
        "n_tok_miss": n_miss,
        "n_tok_extra": n_extra,
        "n_c3_jac": n_c3_jac,
        "n_c3_dice": n_c3_dice,
        "n_lev": _lev_sim(name_a, name_b),
        "n_jw": _jw_sim(name_a, name_b),
        "n_exact": (name_a == name_b).astype(np.float32),
        "n_phonetic": ((A.ph[p1] == B.ph[p2]) & (A.ph[p1] != "")).astype(np.float32),
        "n_initials": ((A.initials[p1] == B.initials[p2]) & (A.initials[p1] != "")
                       & (A.name_tok_n[p1] > 1) & (B.name_tok_n[p2] > 1)).astype(np.float32),
        "n_len_ratio": len_ratio(name_a, name_b),
        "n_tok_n_diff": np.abs(A.name_tok_n[p1] - B.name_tok_n[p2]),
        "a_tok_jac": a_tok_jac,
        "a_tok_cont": a_tok_cont,
        "a_c3_jac": a_c3_jac,
        "a_lev": a_lev,
        "a_postal_eq": ((postal_a == postal_b) & (postal_a != "")).astype(np.float32),
        "a_postal_both_empty": ((postal_a == "") & (postal_b == "")).astype(np.float32),
        "a_digits_jac": np.where(dig_union > 0, dig_inter / dig_union, 0.0).astype(np.float32),
        "a_landmark_both": (A.landmark[p1] * B.landmark[p2]),
        "a_parts_overlap": np.where(parts_min > 0, parts_inter / parts_min, 0.0).astype(np.float32),
        "a_len_ratio": len_ratio(addr_a, addr_b),
        "a_empty": (addr_b == "").astype(np.float32),
        "x_country_eq": (A.country[p1] == B.country[p2]).astype(np.float32),
        "x_is_s3": is_s3.astype(np.float32),
        "x_combined_lev": x_lev,
    }
    out = pd.DataFrame({"s1_entity_id": A.index[p1], "cand_id": B.index[p2]})
    for k in FEATURES:
        out[k] = f[k]
    return out


def _iter_pairs_sharded(cand_path: Path, chunk_rows: int):
    yield from io_utils.iter_parquet_chunks(cand_path, chunk_rows)


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm, blk = art / "normalized", art / "blocking"
    out_all = {}
    for split in ("train", "test"):
        out_dir = art / "features" / f"{split}_pairs"
        meta_path = art / "features" / f"{split}_pairs.meta.json"
        inputs = {
            "cand": blk / f"{split}_candidates.parquet",
            "s1": nrm / f"{split}_s1.parquet",
            "s2": nrm / f"{split}_s2.parquet",
            "s3": nrm / f"{split}_s3.parquet",
        }
        params = {"feat_version": 2, "features": FEATURES}
        if not force and meta_path.exists() and io_utils.manifest_ok(out_dir / "shard_0.parquet", inputs, params):
            print(f"  [{split}] features fresh — skipping")
            out_all[split] = str(out_dir)
            continue
        if meta_path.exists():
            meta_path.unlink()
        if out_dir.exists():
            for old in out_dir.glob("shard_*.parquet"):
                old.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{split}] building lookup arrays...", flush=True)
        s1 = pd.read_parquet(nrm / f"{split}_s1.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        s2 = pd.read_parquet(nrm / f"{split}_s2.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        s3 = pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        A = SideArrays(s1)
        n_s3 = len(s3)
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s2, s3
        B = SideArrays(s23)
        is_s3_all = np.zeros(len(s23), dtype=bool)
        is_s3_all[len(s23) - n_s3:] = True

        CH = 5_000_000
        shard, shard_id, total = [], 0, 0
        for it, chunk in enumerate(_iter_pairs_sharded(blk / f"{split}_candidates.parquet", CH)):
            p1 = A.index.get_indexer(chunk["s1_entity_id"].to_numpy())
            p2 = B.index.get_indexer(chunk["cand_id"].to_numpy())
            if (p1 < 0).any() or (p2 < 0).any():
                bad = int((p1 < 0).sum() + (p2 < 0).sum())
                raise AssertionError(f"candidate id not found in normalized frames: {bad} rows")
            feats = compute_chunk(A, B, p1, p2, is_s3_all[p2])
            shard.append(feats)
            total += len(feats)
            print(f"  [{split}] chunk {it + 1}: cumulative {total:,} pairs", flush=True)
            if len(shard) >= 4:
                io_utils.write_parquet(pd.concat(shard, ignore_index=True), out_dir / f"shard_{shard_id}.parquet")
                shard_id += 1
                shard = []
        if shard:
            io_utils.write_parquet(pd.concat(shard, ignore_index=True), out_dir / f"shard_{shard_id}.parquet")
        io_utils.save_manifest(out_dir / "shard_0.parquet", inputs, params,
                               extra={"n_pairs": total, "shards": shard_id + 1, "features": FEATURES})
        # also a standalone meta for the dir
        io_utils.json_dump({"n_pairs": total, "shards": shard_id + 1, "features": FEATURES,
                            "params_fp": io_utils.params_fingerprint(params)}, meta_path)
        out_all[split] = str(out_dir)
        del A, B, s1, s23, is_s3_all
    return out_all


def iter_feature_shards(split: str, cfg: dict):
    """Yield feature DataFrames shard-by-shard (for S5 training/inference)."""
    out_dir = Path(cfg["paths"]["artifacts_dir"]) / "features" / f"{split}_pairs"
    for p in sorted(out_dir.glob("shard_*.parquet")):
        yield pd.read_parquet(p)
