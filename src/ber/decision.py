"""S7 — Calibrated decision engine (index-based, vectorized).

Flow:
  1. Isotonic calibration of val p_match vs GT labels (val entities only).
  2. Candidate competition on val (unique assignment) + vectorized F0.5
     threshold sweep (bincount per entity; TRUE counts come from GT, so the
     recall denominator is exact) -> global threshold.
  3. Per-country thresholds learned ONLY for countries represented in the
     training split (US/India). Any other country string (France, ...) falls
     back to thresholds["GLOBAL"] at test time. Never fitted on test data.
  4. Test: stream scored shards -> p_cal -> country-threshold filter ->
     competition via running-max arrays -> decode ids -> test_predictions.parquet.

Outputs: artifacts/calibration/thresholds.json (+ isotonic.pkl),
         artifacts/features/test_predictions.parquet (s1_entity_id, cand_id).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import io_utils, metrics
from .model import PACK_MUL, _labels_for

F05_B = 0.5
F05_NUM = 1.25  # (1 + beta^2), beta = 0.5


def _f05_mean(s1c: np.ndarray, y: np.ndarray, p: np.ndarray, n_true: np.ndarray,
              th: float) -> float:
    """Macro F0.5 over entities 0..n-1 (n = len(n_true)); s1c already remapped 0..n-1.

    True counts come from GT (exact recall denominator). Entities with no rows
    still contribute: pred_count 0 -> singleton 1.0 / non-singleton 0.0.
    """
    n = len(n_true)
    mask = p >= th
    pred_counts = np.bincount(s1c[mask], minlength=n).astype(np.float64)
    tp = np.bincount(s1c[mask & (y == 1)], minlength=n).astype(np.float64)
    fp = pred_counts - tp
    fn = n_true - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_counts > 0, tp / np.maximum(pred_counts, 1e-9), 0.0)
        rec = np.where(n_true > 0, tp / np.maximum(n_true, 1e-9), 0.0)
    f = np.where((prec + rec) > 0, F05_NUM * prec * rec / (F05_B * prec + rec), 0.0)
    f[n_true == 0] = np.where(pred_counts[n_true == 0] == 0, 1.0, 0.0)
    del fp, fn
    return float(f.mean())


def _sweep(s1c, y, p, n_true, grid):
    rows = [{"threshold": float(th), "macro_f05": _f05_mean(s1c, y, p, n_true, th)}
            for th in grid]
    return rows, max(rows, key=lambda r: r["macro_f05"])


def _competition_indices(p1: np.ndarray, p2: np.ndarray, p: np.ndarray,
                         n_cand: int) -> np.ndarray:
    """Per candidate keep the max-scoring S1 (ties: last row wins). Returns winner_s1 (-1 if none)."""
    best_p = np.full(n_cand, -1.0, dtype=np.float32)
    np.maximum.at(best_p, p2, p)
    winner_s1 = np.full(n_cand, -1, dtype=np.int64)
    m = p >= best_p[p2]
    winner_s1[p2[m]] = p1[m]
    return winner_s1


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    out_dir = art / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_scores_path = art / "features" / "val_scores.parquet"
    if not val_scores_path.exists():
        raise SystemExit("S7 requires val_scores.parquet (run S5 first)")

    feat_dir = art / "features" / "train_pairs"
    s1_ids = pd.read_parquet(feat_dir / "s1_ids.parquet")["entity_id"].to_numpy()
    cand_ids = pd.read_parquet(feat_dir / "cand_ids.parquet")["entity_id"].to_numpy()
    manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    country_of_s1 = dict(zip(manifest["s1_entity_id"], manifest["country"]))
    gt_val = metrics.gt_dict(cfg, only_val=True, manifest=manifest)

    # compact entity space over val entities (index = position in val_universe)
    val_universe = np.where(np.isin(s1_ids, list(val_ids)))[0]
    remap = np.full(len(s1_ids), -1, dtype=np.int64)
    remap[val_universe] = np.arange(len(val_universe))
    n_ents = len(val_universe)

    # exact per-entity true-match counts from GT (vectorized via index mapping)
    n_true_full = np.zeros(n_ents, dtype=np.float64)
    pos_in_universe = pd.Series(remap, index=s1_ids)
    r = pos_in_universe.reindex(pd.Index(list(gt_val.keys()))).to_numpy()
    k = np.array([len(ms) for ms in gt_val.values()], dtype=np.float64)
    valid = (r >= 0) & (r < n_ents)
    n_true_full[r[valid].astype(np.int64)] = k[valid]

    vs = pd.read_parquet(val_scores_path)
    p1 = vs["s1_idx"].to_numpy()
    p2 = vs["cand_idx"].to_numpy()
    p_raw = vs["p_match"].to_numpy()
    del vs

    # labels via packed GT
    gt_rows = [(s1, m) for s1, ms in gt_val.items() for m in ms]
    gt_df = pd.DataFrame(gt_rows, columns=["s1", "c"]).drop_duplicates()
    i1 = gt_df["s1"].map(pos_in_universe)
    i2 = gt_df["c"].map(pd.Series(np.arange(len(cand_ids)), index=cand_ids))
    ok = i1.notna() & i2.notna()
    packed_gt = np.sort(np.unique(
        i1[ok].to_numpy(dtype=np.int64) * PACK_MUL + i2[ok].to_numpy(dtype=np.int64)))
    y = _labels_for(p1, p2, packed_gt)

    # isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, y)
    with open(out_dir / "isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    p_cal = iso.predict(p_raw).astype(np.float32)
    del p_raw

    # competition, then per-row winning mask
    winner_s1 = _competition_indices(p1, p2, p_cal, len(cand_ids))
    s1c = remap[p1]
    win_rows = (winner_s1[p2] == p1) & (s1c >= 0)
    s1c_w = s1c[win_rows]
    y_w = y[win_rows]
    p_w = p_cal[win_rows]
    del winner_s1, p1, p2, p_cal, y, s1c

    lo, hi, step = cfg["decision"]["threshold_grid"]
    grid = np.arange(lo, hi + 1e-9, step)
    sweep, best = _sweep(s1c_w, y_w, p_w, n_true_full, grid)
    print(f"global best: th={best['threshold']:.4f} macro_f05={best['macro_f05']:.5f}", flush=True)

    # per-country thresholds — training-represented countries only
    thresholds = {"GLOBAL": float(best["threshold"])}
    country_by_pos = np.array([country_of_s1.get(s, "<none>") for s in s1_ids[val_universe]])
    for c in sorted(set(country_by_pos)):
        if c == "<none>":
            continue
        ent_sel = np.where(country_by_pos == c)[0]
        row_mask = np.isin(s1c_w, ent_sel)
        if not row_mask.any():
            continue
        # remap entity ids into the country-local 0..k-1 space
        local = np.full(n_ents, -1, dtype=np.int64)
        local[ent_sel] = np.arange(len(ent_sel))
        s1c_c = local[s1c_w[row_mask]]
        sweep_c, best_c = _sweep(s1c_c, y_w[row_mask], p_w[row_mask],
                                 n_true_full[ent_sel], grid)
        thresholds[c] = float(best_c["threshold"])
        print(f"  country {c}: th={best_c['threshold']:.4f} macro_f05={best_c['macro_f05']:.5f}", flush=True)

    io_utils.json_dump({
        "thresholds": thresholds,
        "sweep_preview": sweep[:: max(1, len(sweep) // 40)],
        "best": best,
        "policy": ("per-country thresholds learned only for training-represented "
                   "countries; unseen countries (e.g. France) use GLOBAL; never "
                   "fitted on test data"),
    }, out_dir / "thresholds.json")

    # ---- test predictions ----
    test_scores_dir = art / "features" / "test_scores"
    preds_path = art / "features" / "test_predictions.parquet"
    if test_scores_dir.exists() and any(test_scores_dir.glob("shard_*.parquet")):
        t_s1_ids = pd.read_parquet(art / "features" / "test_pairs" / "s1_ids.parquet")["entity_id"].to_numpy()
        t_cand_ids = pd.read_parquet(art / "features" / "test_pairs" / "cand_ids.parquet")["entity_id"].to_numpy()
        t_country = pd.read_parquet(art / "normalized" / "test_s1.parquet",
                                    columns=["country"])["country"].to_numpy()
        assert len(t_country) == len(t_s1_ids)
        th_arr = np.array([thresholds.get(c, thresholds["GLOBAL"]) for c in t_country],
                          dtype=np.float32)  # unseen countries -> GLOBAL (policy)
        kept_p1, kept_p2, kept_p = [], [], []
        best_p_t = np.full(len(t_cand_ids), -1.0, dtype=np.float32)
        for sp in sorted(test_scores_dir.glob("shard_*.parquet")):
            df = pd.read_parquet(sp)
            p1t = df["s1_idx"].to_numpy()
            p2t = df["cand_idx"].to_numpy()
            pt = iso.predict(df["p_match"].to_numpy()).astype(np.float32)
            keep_m = pt >= th_arr[p1t]
            kept_p1.append(p1t[keep_m])
            kept_p2.append(p2t[keep_m])
            kept_p.append(pt[keep_m])
            np.maximum.at(best_p_t, p2t, pt)
            del df, p1t, p2t, pt
        kept_p1 = np.concatenate(kept_p1)
        kept_p2 = np.concatenate(kept_p2)
        kept_p = np.concatenate(kept_p)
        winner_t = _competition_indices(kept_p1, kept_p2, kept_p, len(t_cand_ids))
        cand_final = np.where(winner_t >= 0)[0]
        preds = pd.DataFrame({
            "s1_entity_id": t_s1_ids[winner_t[cand_final]],
            "cand_id": t_cand_ids[cand_final],
        })
        preds.to_parquet(preds_path, index=False)
        print(f"test predictions: {len(preds):,} pairs "
              f"({preds['s1_entity_id'].nunique():,} S1 entities matched)")
    return {"best": best, "thresholds": thresholds}
