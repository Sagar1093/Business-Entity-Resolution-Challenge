"""S4 — Pair features, LOW-RAM streaming implementation (v4).

Design for a small laptop (~8-16 GB RAM, deadline-bounded):
  - SideArrays store plain numpy string arrays ONLY. The old per-row Python
    frozenset objects (name sets, char-3gram sets, digit sets) blew past RAM
    at 10M rows; the same similarity features are now computed on the fly
    from strings per BATCH (Python object churn is per-batch, not per-dataset).
  - Candidates stream through iter_candidate_shards. With base absent
    (rescue-only mode) per-S1 groups appear per shard in score-descending
    order, so a per-source cap (n_s2_cap / n_s3_cap) is applied per
    (shard, source) group WITHOUT deduplicating 54M-row shards in memory.
    Cross-shard duplicates are removed later per entity in the model/decision
    stages (unique assignment tolerates this).
  - Feature computation runs in batches (features.batch_pairs); each batch
    produces a (n, 28) float32 matrix, quantized to uint8 (1/255 scale) and
    appended to the current shard buffer.
  - Shards are capped (features.max_pairs_per_shard) and written with zstd.

Layout per split (artifacts/features/{split}_pairs/):
  s1_ids.parquet, cand_ids.parquet, manifest.json, shard_*.parquet
  shard columns: s1_idx int32, cand_idx int32, f0..f27 uint8 (FEATURES order)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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

BATCH_PAIRS = 100_000  # feature batch size (float32 matrix + rapidfuzz per batch)
BATCH_PAIRS_CAP = 600_000  # per (shard, source, slice) upper bound — RAM guard


def _char3(text: str) -> set[str]:
    t = f"  {text} "
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 5 else (
        {t.strip()} if t.strip() else set())


class LightSide:
    """String-only side arrays for one source (name + address)."""

    def __init__(self, entity_id: np.ndarray, country: np.ndarray,
                 name_norm: np.ndarray, name_core: np.ndarray, name_ph: np.ndarray,
                 name_tok_n: np.ndarray, addr_norm: np.ndarray, addr_postal: np.ndarray,
                 addr_digits: np.ndarray, addr_parts: np.ndarray, addr_landmark: np.ndarray):
        self.index = pd.Index(entity_id)
        self.country = country
        self.name_norm = name_norm
        self.name_core = name_core
        self.name_ph = name_ph
        self.name_tok_n = name_tok_n
        self.addr_norm = addr_norm
        self.addr_postal = addr_postal
        self.addr_digits = addr_digits
        self.addr_parts = addr_parts
        self.addr_landmark = addr_landmark

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "LightSide":
        return cls(
            df["entity_id"].to_numpy(),
            df["country"].to_numpy(),
            df["name_norm"].to_numpy(),
            df["name_core"].to_numpy(),
            df["name_key_phonetic"].to_numpy(),
            df["name_tok_n"].to_numpy(dtype=np.float32),
            df["addr_norm"].to_numpy(),
            df["addr_postal"].to_numpy(),
            df["addr_digits"].to_numpy(),
            df["addr_parts"].to_numpy(),
            df["addr_landmark"].to_numpy(dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.index)


def _set_pair_stats_str(sa: np.ndarray, sb: np.ndarray):
    """Word-overlap Jaccard / containment / Dice for aligned string arrays."""
    n = len(sa)
    inter = np.fromiter((len(set(a.split()) & set(b.split()))
                         for a, b in zip(sa, sb)), dtype=np.float32, count=n)
    la = np.fromiter((len(a.split()) for a in sa), dtype=np.float32, count=n)
    lb = np.fromiter((len(b.split()) for b in sb), dtype=np.float32, count=n)
    union = la + lb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
        cont = np.where(np.maximum(la, lb) > 0, inter / np.maximum(la, lb), 0.0)
        dice = np.where((la + lb) > 0, 2 * inter / (la + lb), 0.0)
    return jac.astype(np.float32), cont.astype(np.float32), dice.astype(np.float32)


def _gram_pair_stats_str(ga: np.ndarray, gb: np.ndarray):
    """Char-3gram Jaccard / Dice for aligned string arrays."""
    n = len(ga)
    inter = np.fromiter((len(_char3(a) & _char3(b))
                         for a, b in zip(ga, gb)), dtype=np.float32, count=n)
    la = np.fromiter((len(_char3(a)) for a in ga), dtype=np.float32, count=n)
    lb = np.fromiter((len(_char3(b)) for b in gb), dtype=np.float32, count=n)
    union = la + lb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
        dice = np.where((la + lb) > 0, 2 * inter / (la + lb), 0.0)
    return jac.astype(np.float32), dice.astype(np.float32)


def _len_ratio_arr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lx = np.fromiter((len(s) for s in x), dtype=np.float32, count=len(x))
    ly = np.fromiter((len(s) for s in y), dtype=np.float32, count=len(y))
    mx = np.maximum(lx, ly)
    return np.where(mx > 0, np.minimum(lx, ly) / mx, 1.0).astype(np.float32)


def _lev_sim(a, b):
    return process.cpdist(a, b, scorer=distance.Levenshtein.normalized_similarity).astype(np.float32)


def _jw_sim(a, b):
    return process.cpdist(a, b, scorer=distance.JaroWinkler.normalized_similarity).astype(np.float32)


def compute_features_light(A: LightSide, B: LightSide, p1: np.ndarray, p2: np.ndarray,
                           is_s3: np.ndarray) -> np.ndarray:
    """(n, 28) float32 feature matrix in FEATURES order (batch-sized inputs)."""
    n_tok_jac, n_tok_cont, n_tok_dice = _set_pair_stats_str(A.name_core[p1], B.name_core[p2])
    n_c3_jac, n_c3_dice = _gram_pair_stats_str(A.name_norm[p1], B.name_norm[p2])
    a_tok_jac, a_tok_cont, _ = _set_pair_stats_str(A.addr_norm[p1], B.addr_norm[p2])
    a_c3_jac, _ = _gram_pair_stats_str(A.addr_norm[p1], B.addr_norm[p2])

    name_a, name_b = A.name_norm[p1], B.name_norm[p2]
    addr_a, addr_b = A.addr_norm[p1], B.addr_norm[p2]
    postal_a, postal_b = A.addr_postal[p1], B.addr_postal[p2]

    a_lev = _lev_sim(addr_a, addr_b)
    both_empty = (addr_a == "") & (addr_b == "")
    a_lev[both_empty] = 0.0

    full_a = np.char.add(np.char.add(name_a.astype(str), " "), addr_a.astype(str))
    full_b = np.char.add(np.char.add(name_b.astype(str), " "), addr_b.astype(str))
    x_lev = _lev_sim(full_a, full_b)
    x_lev[both_empty & (name_a == "") & (name_b == "")] = 0.0

    # token miss/extra (symmetric difference sizes), string based
    def _toks(arr):
        return [set(s.split()) for s in arr]

    ta, tb = _toks(A.name_core[p1]), _toks(B.name_core[p2])
    n_miss = np.fromiter((len(a - b) for a, b in zip(ta, tb)), dtype=np.float32, count=len(ta))
    n_extra = np.fromiter((len(b - a) for a, b in zip(ta, tb)), dtype=np.float32, count=len(ta))
    del ta, tb

    da = [set(DIGITS_RE.findall(s)) for s in A.addr_norm[p1]]
    db = [set(DIGITS_RE.findall(s)) for s in B.addr_norm[p2]]
    dig_inter = np.fromiter((len(x & y) for x, y in zip(da, db)), dtype=np.float32, count=len(da))
    dig_union = np.fromiter((len(x | y) for x, y in zip(da, db)), dtype=np.float32, count=len(da))
    del da, db
    pa = [frozenset(p for p in s.split("|") if p) for s in A.addr_parts[p1]]
    pb = [frozenset(p for p in s.split("|") if p) for s in B.addr_parts[p2]]
    parts_inter = np.fromiter((len(x & y) for x, y in zip(pa, pb)), dtype=np.float32, count=len(pa))
    parts_min = np.fromiter((min(len(x), len(y)) for x, y in zip(pa, pb)), dtype=np.float32, count=len(pa))
    del pa, pb

    initials_a = np.fromiter(("".join(t[0] for t in s.split()) if s else ""
                              for s in A.name_core[p1]), dtype=object, count=len(p1))
    initials_b = np.fromiter(("".join(t[0] for t in s.split()) if s else ""
                              for s in B.name_core[p2]), dtype=object, count=len(p2))

    cols = {
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
        "n_phonetic": ((A.name_ph[p1] == B.name_ph[p2]) & (A.name_ph[p1] != "")).astype(np.float32),
        "n_initials": ((initials_a == initials_b) & (initials_a != "")
                       & (A.name_tok_n[p1] > 1) & (B.name_tok_n[p2] > 1)).astype(np.float32),
        "n_len_ratio": _len_ratio_arr(name_a, name_b),
        "n_tok_n_diff": np.abs(A.name_tok_n[p1] - B.name_tok_n[p2]),
        "a_tok_jac": a_tok_jac,
        "a_tok_cont": a_tok_cont,
        "a_c3_jac": a_c3_jac,
        "a_lev": a_lev,
        "a_postal_eq": ((postal_a == postal_b) & (postal_a != "")).astype(np.float32),
        "a_postal_both_empty": ((postal_a == "") & (postal_b == "")).astype(np.float32),
        "a_digits_jac": np.where(dig_union > 0, dig_inter / dig_union, 0.0).astype(np.float32),
        "a_landmark_both": (A.addr_landmark[p1] * B.addr_landmark[p2]).astype(np.float32),
        "a_parts_overlap": np.where(parts_min > 0, parts_inter / parts_min, 0.0).astype(np.float32),
        "a_len_ratio": _len_ratio_arr(addr_a, addr_b),
        "a_empty": (addr_b == "").astype(np.float32),
        "x_country_eq": (A.country[p1] == B.country[p2]).astype(np.float32),
        "x_is_s3": is_s3.astype(np.float32),
        "x_combined_lev": x_lev,
    }
    return np.stack([cols[name] for name in FEATURES], axis=1)


def _quantize_q(q: np.ndarray) -> np.ndarray:
    """(n, 28) float32 in [0,1] -> uint8 (255 scale) with per-feature caps applied."""
    out = np.empty(q.shape, dtype=np.uint8)
    for j, name in enumerate(FEATURES):
        v = q[:, j]
        cap = CAPS.get(name)
        if cap is not None:
            v = np.minimum(v, cap) / cap
        out[:, j] = (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return out


def _write_id_maps(out_dir: Path, s1: LightSide, s23: LightSide) -> None:
    pd.DataFrame({"entity_id": s1.index.to_numpy()}).to_parquet(out_dir / "s1_ids.parquet", index=False)
    pd.DataFrame({"entity_id": s23.index.to_numpy()}).to_parquet(out_dir / "cand_ids.parquet", index=False)


def _free_disk_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(str(path)).free / 1e9


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    nrm = art / "normalized"
    fc = cfg.get("features", {}) or {}
    compression = fc.get("compression", "zstd")
    n_s2_cap = int(fc.get("n_s2_cap", 30))
    n_s3_cap = int(fc.get("n_s3_cap", 30))
    max_shard = int(fc.get("max_pairs_per_shard", 20_000_000))
    out_all = {}
    if _free_disk_gb(art) < 15.0:
        raise SystemExit(f"S4 aborted: only {_free_disk_gb(art):.1f} GB free (need >= 15 GB).")

    for split in ("train", "test"):
        out_dir = art / "features" / f"{split}_pairs"
        meta_path = art / "features" / f"{split}_pairs.meta.json"
        inputs = {f"{split}_{s}": nrm / f"{split}_{s}.parquet" for s in ("s1", "s2", "s3")}
        for tag in ("g", "h"):
            d = art / "blocking" / f"{split}_candidates_{tag}"
            if d.exists():
                inputs[f"cand_{tag}"] = str(d)
        params = {"feat_version": 4, "features": FEATURES, "quantization": "uint8/255",
                  "compression": compression, "n_s2_cap": n_s2_cap, "n_s3_cap": n_s3_cap,
                  "max_pairs_per_shard": max_shard, "batch_pairs": BATCH_PAIRS}
        if not force and meta_path.exists() and io_utils.manifest_ok(out_dir / "shard_0.parquet",
                                                                     inputs, params):
            print(f"  [{split}] features fresh — skipping")
            out_all[split] = str(out_dir)
            continue
        if meta_path.exists():
            meta_path.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("shard_*.parquet"):
            old.unlink()

        print(f"  [{split}] building string side arrays...", flush=True)
        s1 = LightSide.from_frame(pd.read_parquet(nrm / f"{split}_s1.parquet",
                                                  columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS]))
        n_s3 = sum(1 for _ in pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id"])["entity_id"])
        s2 = pd.read_parquet(nrm / f"{split}_s2.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        s3 = pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id", "country", *_NAME_COLS, *_ADDR_COLS])
        n_s2 = len(s2)
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s2, s3
        s23l = LightSide.from_frame(s23)
        del s23
        is_s3_all = np.zeros(len(s23l), dtype=bool)
        is_s3_all[len(s23l) - n_s3:] = True
        _write_id_maps(out_dir, s1, s23l)

        total = 0
        shard_id = 0
        buf: list[np.ndarray] = []
        buf_rows = 0

        def flush() -> None:
            nonlocal shard_id, buf, buf_rows
            if not buf_rows:
                return
            # each buffered block: [s1_idx, cand_idx, f0..f27] as int32 columns
            id_block = np.concatenate([b[:, :2] for b in buf], axis=0)
            f_block = np.concatenate([b[:, 2:] for b in buf], axis=0)
            block = pd.DataFrame({
                "s1_idx": id_block[:, 0].astype(np.int32),
                "cand_idx": id_block[:, 1].astype(np.int32),
            })
            for i in range(len(FEATURES)):
                block[f"f{i}"] = f_block[:, i]
            io_utils.write_parquet(block, out_dir / f"shard_{shard_id}.parquet", 1_000_000,
                                   compression=compression)
            shard_id += 1
            buf, buf_rows = [], 0
            del id_block, f_block, block

        for chunk in iter_candidate_shards(split, cfg):
            p1 = s1.index.get_indexer(chunk["s1_entity_id"].to_numpy())
            p2 = s23l.index.get_indexer(chunk["cand_id"].to_numpy())
            if (p1 < 0).any() or (p2 < 0).any():
                raise AssertionError("candidate id missing from normalized frames")
            src_is_s3 = is_s3_all[p2]
            for src_mask, cap in ((~src_is_s3, n_s2_cap), (src_is_s3, n_s3_cap)):
                src_pos = np.where(src_mask)[0]
                if not len(src_pos):
                    continue
                sl = pd.Series(src_pos).groupby(p1[src_pos] // 200_000)
                for _gi, g in sl:
                    # cap within (slice, source) group, without global dedup
                    if len(g) > cap:
                        g = g.iloc[:cap]
                    if len(g) > BATCH_PAIRS_CAP:
                        g = g.iloc[:BATCH_PAIRS_CAP]
                    idx = g.to_numpy(dtype=np.int64)
                    for lo in range(0, len(idx), BATCH_PAIRS):
                        sel = idx[lo:lo + BATCH_PAIRS]
                        q = compute_features_light(s1, s23l, p1[sel], p2[sel], src_is_s3[sel])
                        pair_block = np.column_stack([
                            p1[sel].astype(np.int32), p2[sel].astype(np.int32),
                            _quantize_q(q).astype(np.int32),
                        ])
                        buf.append(pair_block)
                        buf_rows += len(sel)
                        total += len(sel)
                        del q, pair_block
                        if buf_rows >= max_shard:
                            flush()
            del chunk, p1, p2, src_is_s3
            print(f"  [{split}] featurized {total:,} pairs (shard {shard_id})", flush=True)
        flush()
        io_utils.save_manifest(out_dir / "shard_0.parquet" if shard_id else out_dir / ".keep",
                               inputs, params, extra={"n_pairs": total, "shards": shard_id})
        io_utils.json_dump({"n_pairs": total, "shards": shard_id, "features": FEATURES,
                            "params_fp": io_utils.params_fingerprint(params)}, meta_path)
        print(f"  [{split}] done: {total:,} pairs -> {shard_id} quantized shards")
        out_all[split] = str(out_dir)
        del s1, s23l, is_s3_all
    return out_all


def iter_feature_shards(split: str, cfg: dict):
    """Yield (r1 int32, r2 int32, Q uint8 (n, 28)) per feature shard.

    r1/r2 are s1_idx/cand_idx rows; Q is the uint8 feature matrix (FEATURES
    order). Consumed by the model stage — features are decoded ONCE here,
    never recomputed.
    """
    d = Path(cfg["paths"]["artifacts_dir"]) / "features" / f"{split}_pairs"
    for p in sorted(d.glob("shard_*.parquet"), key=lambda x: int(x.stem.split("_")[1])):
        df = pd.read_parquet(p)
        r1 = df["s1_idx"].to_numpy()
        r2 = df["cand_idx"].to_numpy()
        Q = df[[f"f{i}" for i in range(len(FEATURES))]].to_numpy(dtype=np.uint8)
        del df
        yield r1, r2, Q
        del Q
