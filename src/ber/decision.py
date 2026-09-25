"""S7 — Calibrated decision engine.

Pipeline: isotonic calibration on val -> global F0.5-max threshold sweep on val
-> per-country thresholds learned ONLY for training-represented countries
(fallback = global for unseen countries) -> rerank-score blending inside the
difficult band -> candidate competition (each cand id claims its best S1) ->
emit val/test prediction dicts + thresholds.json.

Never uses test records for fitting; France (unseen) resolves to the global
threshold by design.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import io_utils, metrics


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return iso


def _sweep_f05(scored: pd.DataFrame, truth: dict[str, set[str]], country: dict[str, str],
               grid) -> tuple[list[dict], dict]:
    results = []
    for th in grid:
        pred = metrics.predictions_at_threshold(scored, th)
        m = metrics.entity_f05(pred, truth, per_country=country)
        results.append({"threshold": float(th), **m})
    best = max(results, key=lambda m: m["macro_f05"])
    return results, best


def _apply_country_thresholds(scored: pd.DataFrame, thresholds: dict[str, float],
                              s1_country: dict[str, str]) -> pd.DataFrame:
    """Keep pairs whose calibrated p clears their S1 entity's country threshold."""
    th = scored["s1_entity_id"].map(s1_country).map(thresholds).fillna(
        scored["p_cal"].mean() * 0 + thresholds["GLOBAL"]
    )
    keep = scored["p_cal"] >= th
    return scored[keep].copy(), int((~keep).sum())


def _competition(scores: pd.DataFrame) -> pd.DataFrame:
    """Each candidate id keeps only its highest-scoring S1 claim (unique assignment)."""
    idx = scores.groupby("cand_id")["p_cal"].idxmax()
    return scores.loc[idx].copy()


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    val_scores_path = art / "features" / "val_scores.parquet"
    test_scores_path = art / "features" / "test_scores.parquet"
    out_dir = art / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not val_scores_path.exists():
        raise SystemExit("S7 requires val_scores.parquet (run S5 first)")
    params = {"decision_version": 1}

    val = pd.read_parquet(val_scores_path)
    truth_all = metrics.gt_dict(cfg)
    manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    s1_country = dict(zip(manifest["s1_entity_id"], manifest["country"]))
    truth_val = {k: v for k, v in truth_all.items() if k in val_ids}

    # labels for isotonic (val only)
    gt_rows = [(s1, m) for s1, ms in truth_val.items() for m in ms]
    gt = pd.DataFrame(gt_rows, columns=["s1_entity_id", "cand_id"]).drop_duplicates()
    lab = val[["s1_entity_id", "cand_id"]].merge(gt, on=["s1_entity_id", "cand_id"], how="left", indicator=True)
    y_val = (lab["_merge"] == "both").to_numpy(dtype=np.int8)
    iso = _calibrate(val["p_match"].to_numpy(), y_val)

    val["p_cal"] = iso.predict(val["p_match"].to_numpy()).astype(np.float32)

    # optional rerank blend inside band
    rerank_path = art / "features" / "train_rerank.parquet"
    if rerank_path.exists():
        rr = pd.read_parquet(rerank_path)
        merged = val.merge(rr, on=["s1_entity_id", "cand_id"], how="left")
        w = float(cfg["decision"].get("rerank_blend_w", 0.5))
        band = merged["rerank_score"].notna()
        merged.loc[band, "p_cal"] = (
            (1 - w) * merged.loc[band, "p_cal"] + w * (merged.loc[band, "rerank_score"] > 0).astype(np.float32)
        ).astype(np.float32)
        val = merged
        print(f"  rerank blend applied to {int(band.sum()):,} val pairs (w={w})")

    lo, hi, step = cfg["decision"]["threshold_grid"]
    grid = np.arange(lo, hi + 1e-9, step)
    # apply the SAME candidate-competition policy used at test time, then sweep
    val = _competition(val)
    sweep, best = _sweep_f05(val, truth_val, s1_country, grid)

    # per-country thresholds for TRAINING countries only (US, India); any country
    # absent from this dict (e.g. France in test) falls back to thresholds["GLOBAL"].
    thresholds = {"GLOBAL": float(best["threshold"])}
    by_country = {}
    country_arr = val["s1_entity_id"].map(s1_country)
    for c in sorted(set(s1_country.values())):
        sub = val[country_arr == c]
        ids_c = {k for k, v in s1_country.items() if v == c}
        truth_c = {k: v for k, v in truth_val.items() if k in ids_c}
        _, best_c = _sweep_f05(sub, truth_c, s1_country, grid)
        by_country[c] = best_c["threshold"]
    thresholds.update(by_country)

    calib_path = out_dir / "isotonic.pkl"
    import pickle  # noqa: E402

    with open(calib_path, "wb") as f:
        pickle.dump(iso, f)
    io_utils.json_dump({"thresholds": thresholds, "val_sweep": sweep[:: max(1, len(sweep) // 40)],
                        "best": best, "by_country_best": by_country},
                       out_dir / "thresholds.json")
    val.to_parquet(art / "features" / "val_calibrated.parquet", index=False)

    # ---- test predictions ----
    if test_scores_path.exists():
        test = pd.read_parquet(test_scores_path)
        test["p_cal"] = iso.predict(test["p_match"].to_numpy()).astype(np.float32)
        rr_t = art / "features" / "test_rerank.parquet"
        if rr_t.exists():
            rr = pd.read_parquet(rr_t)
            test = test.merge(rr, on=["s1_entity_id", "cand_id"], how="left")
            w = float(cfg["decision"].get("rerank_blend_w", 0.5))
            band = test["rerank_score"].notna()
            test.loc[band, "p_cal"] = (
                (1 - w) * test.loc[band, "p_cal"] + w * (test.loc[band, "rerank_score"] > 0).astype(np.float32)
            ).astype(np.float32)
        s1_country_test = _load_test_countries(cfg)
        test_kept, n_dropped = _apply_country_thresholds(test, thresholds, s1_country_test)
        test_kept = _competition(test_kept)
        test_kept.to_parquet(art / "features" / "test_predictions.parquet", index=False)
        print(f"  test: kept {len(test_kept):,} pairs after thresholds+competition "
              f"(dropped {n_dropped:,})")
    return {"best_val": best, "thresholds": thresholds}


def _load_test_countries(cfg: dict) -> dict[str, str]:
    pairs = {}
    for df in io_utils.read_tsv_chunks(
        Path(cfg["paths"]["dataset_test_dir"]) / "test_source1.tsv", 1_000_000,
        columns=["entity_id", "country"],
    ):
        pairs.update(dict(zip(df["entity_id"], df["country"])))
    return pairs


def s1_country_map(val: pd.DataFrame, s1_country: dict) -> pd.Series:
    return val["s1_entity_id"].map(s1_country)
