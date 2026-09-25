"""S4 — Pair features as int8-quantized bytes folded into index-based candidate shards.

Why: 312M train + ~200M test pairs x 28 features x fp32 = ~57 GB of parquet
(impossible with ~26 GB free). Instead each candidate shard is re-emitted with
int32 index columns (s1_idx, cand_idx) and 28 feature columns stored as
uint8-quantized bytes (scale 1/255): 28 B/pair -> ~8.7 GB train, ~5.7 GB test.

Layout per shard parquet (artifacts/features/{split}_pairs/shard_*.parquet):
  s1_idx int32, cand_idx int32, f0..f27 uint8
Decode: np.frombuffer(row bytes).reshape(n, 28).astype(np.float32) / 255.0
(see decode_shard / iter_feature_shards below)

Feature order is fixed by FEATURES; side arrays are built once per split.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import distance, process

from . import io_utils
from .blocking import iter_candidate_shards

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
# unbounded counts mapped to [0,1] with a cap before quantization
CAPS = {"n_tok_miss": 10.0, "n_tok_extra": 10.0, "n_tok_n_diff": 10.0}

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


def _set_pair_stats(sa, sb):
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


def _gram_pair_stats(ga, gb):
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


def _lev_sim(a, b):
    return process.cpdist(a, b, scorer=distance.Levenshtein.normalized_similarity).astype(np.float32)


def _jw_sim(a, b):
    return process.cpdist(a, b, scorer=distance.JaroWinkler.normalized_similarity).astype(np.float32)


def _len_ratio(x, y):
    lx = np.fromiter((len(s) for s in x), dtype=np.float32, count=len(x))
    ly = np.fromiter((len(s) for s in y), dtype=np.float32, count=len(y))
    mx = np.maximum(lx, ly)
    return np.where(mx > 0, np.minimum(lx, ly) / mx, 1.0)


def compute_features(A: SideArrays, B: SideArrays, p1: np.ndarray, p2: np.ndarray,
                     is_s3: np.ndarray) -> dict[str, np.ndarray]:
    """All 28 features for aligned position arrays p1 (A side) / p2 (B side)."""
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

    sa, sb = A.core_sets[p1], B.core_sets[p2]
    n_miss = np.fromiter((len(x - y) for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))
    n_extra = np.fromiter((len(y - x) for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))

    d_a, d_b = A.digit_sets[p1], B.digit_sets[p2]
    dig_inter = np.fromiter((len(x & y) for x, y in zip(d_a, d_b)), dtype=np.float32, count=len(d_a))
    dig_union = np.fromiter((len(x | y) for x, y in zip(d_a, d_b)), dtype=np.float32, count=len(d_a))
    parts_a, parts_b = A.parts_sets[p1], B.parts_sets[p2]
    parts_inter = np.fromiter((len(x & y) for x, y in zip(parts_a, parts_b)), dtype=np.float32, count=len(parts_a))
    parts_min = np.fromiter((min(len(x), len(y)) for x, y in zip(parts_a, parts_b)), dtype=np.float32, count=len(parts_a))

    return {
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
        "n_len_ratio": _len_ratio(name_a, name_b),
        "n_tok_n_diff": np.abs(A.name_tok_n[p1] - B.name_tok_n[p2]),
        "a_tok_jac": a_tok_jac,
        "a_tok_cont": a_tok_cont,
        "a_c3_jac": a_c3_jac,
        "a_lev": a_lev,
        "a_postal_eq": ((postal_a == postal_b) & (postal_a != "")).astype(np.float32),
        "a_postal_both_empty": ((postal_a == "") & (postal_b == "")).astype(np.float32),
        "a_digits_jac": np.where(dig_union > 0, dig_inter / dig_union, 0.0).astype(np.float32),
        "a_landmark_both": (A.landmark[p1] * B.landmark[p2]).astype(np.float32),
        "a_parts_overlap": np.where(parts_min > 0, parts_inter / parts_min, 0.0).astype(np.float32),
        "a_len_ratio": _len_ratio(addr_a, addr_b),
        "a_empty": (addr_b == "").astype(np.float32),
        "x_country_eq": (A.country[p1] == B.country[p2]).astype(np.float32),
        "x_is_s3": is_s3.astype(np.float32),
        "x_combined_lev": x_lev,
    }


def _quantize(feats: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 28) uint8 block in FEATURES order with per-feature caps and 1/255 scale."""
    cols = []
    for name in FEATURES:
        v = feats[name]
        cap = CAPS.get(name)
        if cap is not None:
            v = np.minimum(v, cap) / cap
        cols.append(np.clip(v, 0.0, 1.0))
    return (np.stack(cols, axis=1) * 255.0 + 0.5).astype(np.uint8)


def decode_shard(df: pd.DataFrame) -> pd.DataFrame:
    """Quantized shard -> (s1_entity_id, cand_id, 28 float32 features).

    Requires the id-mapping parquets written next to the shards
    (s1_ids.parquet / cand_ids.parquet per split).
    """
    s1_ids = pd.read_parquet(Path(df.attrs["feat_dir"]) / "s1_ids.parquet")["entity_id"].to_numpy()
    cand_ids = pd.read_parquet(Path(df.attrs["feat_dir"]) / "cand_ids.parquet")["entity_id"].to_numpy()
    out = pd.DataFrame({
        "s1_entity_id": s1_ids[df["s1_idx"].to_numpy()],
        "cand_id": cand_ids[df["cand_idx"].to_numpy()],
    })
    q = df[[f"f{i}" for i in range(len(FEATURES))]].to_numpy(dtype=np.uint8)
    for i, name in enumerate(FEATURES):
        cap = CAPS.get(name)
        v = q[:, i].astype(np.float32) / 255.0
        out[name] = v * cap if cap is not None else v
    return out


def _write_id_maps(out_dir: Path, A: SideArrays, B: SideArrays) -> None:
    pd.DataFrame({"entity_id": A.index.to_numpy()}).to_parquet(out_dir / "s1_ids.parquet", index=False)
    pd.DataFrame({"entity_id": B.index.to_numpy()}).to_parquet(out_dir / "cand_ids.parquet", index=False)


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm, blk = art / "normalized", art / "blocking"
    out_all = {}
    if _free_disk_gb(art) < 15.0:
        raise SystemExit(f"S4 aborted: only {_free_disk_gb(art):.1f} GB free (need >= 15 GB).")

    for split in ("train", "test"):
        out_dir = art / "features" / f"{split}_pairs"
        meta_path = art / "features" / f"{split}_pairs.meta.json"
        inputs = {
            "cand_dir": str(blk / f"{split}_candidates"),
            "s1": nrm / f"{split}_s1.parquet",
            "s2": nrm / f"{split}_s2.parquet",
            "s3": nrm / f"{split}_s3.parquet",
        }
        params = {"feat_version": 3, "features": FEATURES, "quantization": "uint8/255"}
        fresh = meta_path.exists() and not force
        if fresh:
            print(f"  [{split}] features fresh — skipping")
            out_all[split] = str(out_dir)
            continue
        if meta_path.exists():
            meta_path.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("shard_*.parquet"):
            old.unlink()

        print(f"  [{split}] building side arrays...", flush=True)
        s1 = pd.read_parquet(nrm / f"{split}_s1.parquet",
                             columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        n_s3 = sum(1 for _ in pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id"])["entity_id"])
        s2 = pd.read_parquet(nrm / f"{split}_s2.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        s3 = pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        A = SideArrays(s1)
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s2, s3
        B = SideArrays(s23)
        is_s3_all = np.zeros(len(s23), dtype=bool)
        is_s3_all[len(s23) - n_s3:] = True
        _write_id_maps(out_dir, A, B)

        total, shard_id = 0, 0
        buf: list[pd.DataFrame] = []
        buf_rows = 0
        for chunk in iter_candidate_shards(split, cfg):
            p1 = A.index.get_indexer(chunk["s1_entity_id"].to_numpy())
            p2 = B.index.get_indexer(chunk["cand_id"].to_numpy())
            if (p1 < 0).any() or (p2 < 0).any():
                raise AssertionError("candidate id missing from normalized frames")
            feats = compute_features(A, B, p1, p2, is_s3_all[p2])
            q = _quantize(feats)
            block = pd.DataFrame({"s1_idx": p1.astype(np.int32), "cand_idx": p2.astype(np.int32)})
            for i in range(len(FEATURES)):
                block[f"f{i}"] = q[:, i]
            buf.append(block)
            buf_rows += len(block)
            total += len(block)
            del feats, q, block
            if buf_rows >= 20_000_000:
                io_utils.write_parquet(pd.concat(buf, ignore_index=True),
                                       out_dir / f"shard_{shard_id}.parquet", 1_000_000)
                shard_id += 1
                buf, buf_rows = [], 0
            print(f"  [{split}] featurized {total:,} pairs", flush=True)
        if buf:
            io_utils.write_parquet(pd.concat(buf, ignore_index=True),
                                   out_dir / f"shard_{shard_id}.parquet", 1_000_000)
            shard_id += 1
        io_utils.save_manifest(out_dir / "shard_0.parquet" if shard_id else out_dir / ".keep",
                               inputs, params, extra={"n_pairs": total, "shards": shard_id})
        io_utils.json_dump({"n_pairs": total, "shards": shard_id, "features": FEATURES,
                            "params_fp": io_utils.params_fingerprint(params)}, meta_path)
        print(f"  [{split}] done: {total:,} pairs -> {shard_id} quantized shards")
        out_all[split] = str(out_dir)
        del A, B, s1, s23, is_s3_all
    return out_all


def _free_disk_gb(path: Path) -> float:
    import shutil

    return shutil.disk_usage(str(path)).free / 2**30


def iter_feature_shards(split: str, cfg: dict, columns: list[str] | None = None):
    """Yield DECODED feature DataFrames shard-by-shard (S5/S7/S8 consumers).

    Columns: s1_entity_id, cand_id, 28 float32 features (+ any requested subset).
    """
    out_dir = Path(cfg["paths"]["artifacts_dir"]) / "features" / f"{split}_pairs"
    s1_ids = pd.read_parquet(out_dir / "s1_ids.parquet")["entity_id"].to_numpy()
    cand_ids = pd.read_parquet(out_dir / "cand_ids.parquet")["entity_id"].to_numpy()
    want_names = [n for n in (columns or FEATURES) if n in FEATURES]
    idxs = [FEATURES.index(n) for n in want_names]
    for p in sorted(out_dir.glob("shard_*.parquet")):
        df = pd.read_parquet(p)
        out = pd.DataFrame({
            "s1_entity_id": s1_ids[df["s1_idx"].to_numpy()],
            "cand_id": cand_ids[df["cand_idx"].to_numpy()],
        })
        q = df[[f"f{i}" for i in idxs]].to_numpy(dtype=np.uint8).astype(np.float32) / 255.0
        for j, name in enumerate(want_names):
            cap = CAPS.get(name)
            out[name] = q[:, j] * cap if cap is not None else q[:, j]
        yield out
