"""S5 — LightGBM pair classifier.

Label: pair positive iff cand_id is in that S1 entity's GT match list (train only).
Split discipline: features from split=='train_models' entities train the model;
split=='val' entities drive early stopping, calibration and metrics — never training.
Test features are used only for inference. Outputs: models/lgbm_pair.txt,
artifacts/model_eval.json, reports/model_card.md, val/test pair scores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, metrics


def _labels_for_pairs(pairs: pd.DataFrame, truth: dict[str, set[str]]) -> np.ndarray:
    """1.0 iff (s1_entity_id, cand_id) is a GT pair. Vectorized via exploded GT."""
    s1col = pairs["s1_entity_id"].to_numpy()
    ccol = pairs["cand_id"].to_numpy()
    y = np.zeros(len(pairs), dtype=np.int8)
    truth_restricted = {k: v for k, v in truth.items() if k in set(np.unique(s1col))} if len(pairs) > 4_000_000 else truth
    for i in range(len(pairs)):
        t = truth_restricted.get(s1col[i])
        if t and ccol[i] in t:
            y[i] = 1
    return y


def _labels_fast(pairs: pd.DataFrame, truth: dict[str, set[str]]) -> np.ndarray:
    """Explode GT into a set of (s1, cand) tuples for O(1) membership via merge."""
    gt_rows = [(s1, m) for s1, ms in truth.items() for m in ms]
    if not gt_rows:
        return np.zeros(len(pairs), dtype=np.int8)
    gt = pd.DataFrame(gt_rows, columns=["s1_entity_id", "cand_id"]).drop_duplicates()
    merged = pairs[["s1_entity_id", "cand_id"]].merge(gt, on=["s1_entity_id", "cand_id"], how="left", indicator=True)
    return (merged["_merge"] == "both").to_numpy(dtype=np.int8)


def load_split_pairs(cfg: dict, split: str, feat_dir: Path) -> pd.DataFrame:
    """Concatenate shards, restricted to a split's S1 entities (via manifest)."""
    parts = []
    ids = None
    if split in ("train_models", "val"):
        manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
        ids = set(manifest.loc[manifest["split"] == split, "s1_entity_id"])
    for shard in sorted(feat_dir.glob("shard_*.parquet")):
        df = pd.read_parquet(shard)
        if ids is not None:
            df = df[df["s1_entity_id"].isin(ids)]
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    feat_train = art / "features" / "train_pairs"
    model_path = models_dir / "lgbm_pair.txt"
    val_scores_path = art / "features" / "val_scores.parquet"
    test_scores_path = art / "features" / "test_scores.parquet"
    params_key = {"model": cfg["model"], "v": 1}

    if not force and model_path.exists() and val_scores_path.exists():
        print("S5 model + val scores fresh — skipping")
        eval_ = io_utils.json_load(art / "model_eval.json")
        return eval_

    import lightgbm as lgb

    print("loading train_models features...", flush=True)
    tr = load_split_pairs(cfg, "train_models", feat_train)
    print(f"  train_models pairs: {len(tr):,}")
    truth_all = metrics.gt_dict(cfg)
    y_tr = _labels_fast(tr, truth_all)
    print(f"  positives: {int(y_tr.sum()):,} ({y_tr.mean():.4%})")
    feat_cols = [c for c in tr.columns if c not in ("s1_entity_id", "cand_id")]
    X_tr = tr[feat_cols].to_numpy(dtype=np.float32)
    del tr  # free the key+frame copy before val loads (peak-RAM control)

    print("loading val features...", flush=True)
    va = load_split_pairs(cfg, "val", feat_train)
    y_va = _labels_fast(va, truth_all)
    X_va = va[feat_cols].to_numpy(dtype=np.float32)
    print(f"  val pairs: {len(va):,} positives {int(y_va.sum()):,}")
    va_keys = va[["s1_entity_id", "cand_id"]].copy()
    del va

    params = dict(cfg["model"]["params"])
    params.update({
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "seed": int(cfg["seed"]),
        "two_round": True,
    })
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=True)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=feat_cols)
    booster = lgb.train(
        params, dtrain, num_boost_round=int(cfg["model"]["num_boost_round"]),
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False)],
    )
    booster.save_model(str(model_path))
    print(f"  best iteration: {booster.best_iteration}")

    # logistic baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr[::5])
    lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(scaler.transform(X_tr[::5]), y_tr[::5])
    from sklearn.metrics import average_precision_score

    lr_ap = average_precision_score(y_va, lr.predict_proba(scaler.transform(X_va))[:, 1])

    va_scored = va_keys
    va_scored["p_match"] = booster.predict(X_va, num_iteration=booster.best_iteration)
    va_scored.to_parquet(val_scores_path, index=False)
    ap = average_precision_score(y_va, va_scored["p_match"])
    print(f"  val pair AP: lgbm={ap:.5f} lr_baseline={lr_ap:.5f}")

    imp = sorted(zip(feat_cols, booster.feature_importance("gain").tolist()), key=lambda kv: -kv[1])[:15]

    # ---- val entity-level F0.5 quick sweep (operating point preview; final in S7) ----
    val_manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
    val_ids = set(val_manifest.loc[val_manifest["split"] == "val", "s1_entity_id"])
    country = dict(zip(val_manifest["s1_entity_id"], val_manifest["country"]))
    truth_val = {k: v for k, v in truth_all.items() if k in val_ids}
    sweep = []
    for th in (0.3, 0.5, 0.7, 0.8, 0.9):
        pred = metrics.predictions_at_threshold(va_scored, th)
        m = metrics.entity_f05(pred, truth_val, per_country=country)
        sweep.append({"threshold": th, **m})
    best = max(sweep, key=lambda m: m["macro_f05"])

    eval_ = {
        "val_pair_ap": ap, "lr_baseline_ap": lr_ap,
        "best_iteration": booster.best_iteration,
        "feature_importance": imp,
        "val_f05_sweep": sweep, "best_val_operating": best,
    }
    io_utils.json_dump(eval_, art / "model_eval.json")
    _write_model_card(eval_, Path(cfg["paths"]["reports_dir"]) / "model_card.md")

    # ---- test inference ----
    print("scoring test pairs...", flush=True)
    del X_tr, X_va, dtrain, dval  # free train/val matrices before test scoring
    import gc
    gc.collect()
    te = load_split_pairs(cfg, "test", art / "features" / "test_pairs")
    X_te = te[feat_cols].to_numpy(dtype=np.float32)
    te_keys = te[["s1_entity_id", "cand_id"]].copy()
    del te
    te_scored = te_keys
    te_scored["p_match"] = booster.predict(X_te, num_iteration=booster.best_iteration)
    te_scored.to_parquet(test_scores_path, index=False)
    print(f"  test pairs scored: {len(te_scored):,}")
    del X_te
    return eval_


def _write_model_card(ev: dict, out: Path) -> None:
    lines = [
        "# Model card — LightGBM pair classifier", "",
        f"- val pair average precision: **{ev['val_pair_ap']:.5f}** "
        f"(logistic baseline {ev['lr_baseline_ap']:.5f})",
        f"- best iteration: {ev['best_iteration']}",
        "",
        "## Top features (gain)",
        "\n".join(f"- `{k}`: {v:,.0f}" for k, v in ev["feature_importance"]),
        "",
        "## Validation entity-level macro F0.5 sweep (preview; final in S7)",
        "| threshold | macro F0.5 | singleton acc |",
        "|---|---|---|",
    ]
    for s in ev["val_f05_sweep"]:
        lines.append(f"| {s['threshold']} | {s['macro_f05']:.5f} | {s['singleton_accuracy']:.3f} |")
    b = ev["best_val_operating"]
    lines += ["", f"Best preview operating point: threshold {b['threshold']} -> "
              f"macro F0.5 **{b['macro_f05']:.5f}** (by country: {b.get('by_country', {})})", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
