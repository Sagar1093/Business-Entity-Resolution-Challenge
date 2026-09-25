"""S6 — bge-reranker-v2-m3 scoring of the difficult band (guarded, resumable).

Difficult band: pairs whose S5 probability lies in [band_low, band_high].
The reranker cross-encodes (S1 name+address, cand name+address) and its score
becomes both an S7 gate signal and (optionally) a replacement probability inside
the band. MIT-licensed model (~568M params) — satisfies challenge constraints.

Guard: model load inside try/except; on any failure the stage degrades to a
no-op and S7 uses raw p_match (rerank field filled with NaN).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils


def _load_reranker(cfg: dict):
    try:
        from FlagEmbedding import FlagReranker

        model = FlagReranker(
            cfg["rerank"]["model"], use_fp16=bool(cfg["rerank"]["fp16"]) and _cuda_ok(),
            query_max_length=int(cfg["rerank"]["max_len"]), max_length=int(cfg["rerank"]["max_len"]),
        )
        return model
    except Exception as exc:  # guard: degrade gracefully
        print(f"  reranker unavailable ({exc}) -> skipping band scoring")
        return None


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _pair_text(row) -> str:
    name = row["name_core"] if isinstance(row["name_core"], str) else ""
    addr = row["addr_norm"] if isinstance(row["addr_norm"], str) else ""
    return f"{name} {addr}".strip()


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    out_meta = {"rerank_version": 1, "status": "skipped"}
    for split, scores_path in (("train", art / "features" / "val_scores.parquet"),
                               ("test", art / "features" / "test_scores.parquet")):
        if not scores_path.exists():
            print(f"  [{split}] scores missing ({scores_path}) — run S5 first; skipping rerank")
            continue
        out_path = art / "features" / f"{split}_rerank.parquet"
        params = {"rerank_version": 1, "band": [cfg["rerank"]["band_low"], cfg["rerank"]["band_high"]]}
        if not force and out_path.exists() and io_utils.manifest_ok(out_path, {"scores": scores_path}, params):
            print(f"  [{split}] rerank scores fresh — skipping")
            continue
        model = _load_reranker(cfg)
        if model is None:
            io_utils.save_manifest(out_path, {"scores": scores_path}, params, extra={"status": "unavailable"})
            continue

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ber import normalize  # for name_core/addr_norm lookup via normalized parquets

        nrm = art / "normalized"
        scored = pd.read_parquet(scores_path)
        band = scored[
            (scored["p_match"] >= cfg["rerank"]["band_low"]) & (scored["p_match"] <= cfg["rerank"]["band_high"])
        ]
        print(f"  [{split}] difficult band pairs: {len(band):,} / {len(scored):,}")
        if not len(band):
            pd.DataFrame({"s1_entity_id": pd.Series(dtype=object), "cand_id": pd.Series(dtype=object),
                          "rerank_score": pd.Series(dtype=np.float32)}).to_parquet(out_path, index=False)
            io_utils.save_manifest(out_path, {"scores": scores_path}, params, extra={"status": "empty_band"})
            continue

        # text lookup: s1 from split_s1; candidates from s2+s3
        s1 = pd.read_parquet(nrm / f"{split}_s1.parquet",
                             columns=["entity_id", "name_core", "addr_norm"]).set_index("entity_id")
        s2 = pd.read_parquet(nrm / f"{split}_s2.parquet", columns=["entity_id", "name_core", "addr_norm"])
        s3 = pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id", "name_core", "addr_norm"])
        s23 = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")
        del s2, s3

        q_texts = [ _pair_text(s1.loc[sid]) for sid in band["s1_entity_id"] ]
        d_texts = [ _pair_text(s23.loc[cid]) for cid in band["cand_id"] ]
        bs = int(cfg["rerank"]["batch_size"])
        scores = model.compute_score(list(zip(q_texts, d_texts)), batch_size=bs, normalize=True)
        out = pd.DataFrame({
            "s1_entity_id": band["s1_entity_id"].to_numpy(),
            "cand_id": band["cand_id"].to_numpy(),
            "rerank_score": np.asarray(scores, dtype=np.float32),
        })
        out.to_parquet(out_path, index=False)
        io_utils.save_manifest(out_path, {"scores": scores_path}, params,
                               extra={"status": "ok", "band_pairs": len(out)})
        del s1, s23, q_texts, d_texts
    return out_meta
