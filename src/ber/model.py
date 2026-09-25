"""S5 — LightGBM pair classifier (index-based streaming; full implementation).

Data flow (train):
  pass A: stream candidate index shards -> labels via packed GT lookup;
          count pos/neg per shard; per-shard seeded choice of
          neg_ratio*n_pos negatives (deterministic, resumable).
  pass B: featurize ONLY selected rows (positives + sampled negatives)
          plus ALL val-entity rows (unbiased calibration set).
  train LightGBM w/ early stopping on val; LR baseline; save model.
Val scoring -> artifacts/features/val_scores.parquet (s1_idx, cand_idx, p_match)
Test scoring -> artifacts/features/test_scores/shard_* (same columns)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, metrics
from .blocking import iter_candidate_shards
from .features import (FEATURES, SideArrays, compute_features, _NAME_COLS, _ADDR_COLS)

PACK_MUL = 10_500_000  # > max cand index (10.3M)


def _pos_of(universe: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Order-independent id -> position mapping (universe need not be sorted)."""
    pos = pd.Series(np.arange(len(universe)), index=universe).reindex(pd.Index(ids))
    out = pos.to_numpy()
    assert not pd.isna(out).any(), "id missing from universe"
    return out.astype(np.int64)


def _pack(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return p1.astype(np.int64) * PACK_MUL + p2.astype(np.int64)


def _labels_for(p1: np.ndarray, p2: np.ndarray, gt_packed_sorted: np.ndarray) -> np.ndarray:
    packed = _pack(p1, p2)
    pos = np.searchsorted(gt_packed_sorted, packed)
    pos[pos >= len(gt_packed_sorted)] = 0
    return (gt_packed_sorted[pos] == packed).astype(np.int8)


def _gt_packed_sorted(cfg: dict, s1_ids: np.ndarray, cand_ids: np.ndarray,
                      restrict_s1: set[str]) -> np.ndarray:
    truth = metrics.gt_dict(cfg)
    truth = {k: v for k, v in truth.items() if k in restrict_s1}
    rows = [(s1, m) for s1, ms in truth.items() for m in ms]
    if not rows:
        return np.empty(0, dtype=np.int64)
    gt = pd.DataFrame(rows, columns=["s1", "c"]).drop_duplicates()
    i1 = gt["s1"].map(pd.Series(np.arange(len(s1_ids)), index=s1_ids))
    i2 = gt["c"].map(pd.Series(np.arange(len(cand_ids)), index=cand_ids))
    ok = i1.notna() & i2.notna()
    packed = _pack(i1[ok].to_numpy(dtype=np.int64), i2[ok].to_numpy(dtype=np.int64))
    return np.sort(np.unique(packed))


def _load_side_arrays(nrm: Path, split: str):
    cols = ["entity_id", "country", *_NAME_COLS, *_ADDR_COLS]
    s1 = pd.read_parquet(nrm / f"{split}_s1.parquet", columns=cols)
    s2 = pd.read_parquet(nrm / f"{split}_s2.parquet", columns=cols)
    s3 = pd.read_parquet(nrm / f"{split}_s3.parquet", columns=cols)
    n_s3 = len(s3)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    A, B = SideArrays(s1), SideArrays(s23)
    is_s3 = np.zeros(len(s23), dtype=bool)
    is_s3[len(s23) - n_s3:] = True
    return A, B, is_s3


def _index_shards_exist(cfg: dict, split: str) -> bool:
    d = Path(cfg["paths"]["artifacts_dir"]) / "features" / f"{split}_pairs"
    return (d / "s1_ids.parquet").exists() and any(d.glob("shard_*.parquet"))


def run(cfg: dict, force: bool = False) -> dict:
    art = Path(cfg["paths"]["artifacts_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    model_path = models_dir / "lgbm_pair.txt"
    val_scores_path = art / "features" / "val_scores.parquet"
    test_scores_dir = art / "features" / "test_scores"
    if not force and model_path.exists() and val_scores_path.exists() and test_scores_dir.exists():
        print("S5 fresh — skipping")
        return io_utils.json_load(art / "model_eval.json")

    import lightgbm as lgb

    feat_dir = art / "features" / "train_pairs"
    s1_ids = pd.read_parquet(feat_dir / "s1_ids.parquet")["entity_id"].to_numpy()
    cand_ids = pd.read_parquet(feat_dir / "cand_ids.parquet")["entity_id"].to_numpy()
    manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
    val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    train_ids = set(manifest.loc[manifest["split"] == "train_models", "s1_entity_id"])
    val_row_mask = np.isin(s1_ids, list(val_ids))

    gt_train = _gt_packed_sorted(cfg, s1_ids, cand_ids, train_ids)
    gt_val = _gt_packed_sorted(cfg, s1_ids, cand_ids, val_ids)
    neg_ratio = int(cfg["model"].get("train_neg_ratio", 3))
    seed = int(cfg["seed"])

    # ---- pass A: labels/counts + per-shard negative selection ----
    shard_files = sorted((art / "blocking" / "train_candidates").glob("shard_*.parquet"))
    plan: list[dict] = []
    tot_pos = tot_neg = 0
    print("pass A: counting positives/negatives per shard...", flush=True)
    for si, sp in enumerate(shard_files):
        cand = pd.read_parquet(sp)
        p1 = _pos_of(s1_ids, cand["s1_entity_id"].to_numpy())
        p2 = _pos_of(cand_ids, cand["cand_id"].to_numpy())
        y = _labels_for(p1, p2, gt_train)
        is_val = val_row_mask[p1]
        y_eff = np.where(is_val, 0, y)  # val rows never count as train positives
        pos_idx = np.where((y == 1) & ~is_val)[0]
        neg_idx = np.where((y == 0) & ~is_val)[0]
        rng = np.random.default_rng(seed + si)
        take = min(len(neg_idx), neg_ratio * len(pos_idx))
        neg_sel = rng.choice(neg_idx, size=take, replace=False) if take else np.empty(0, np.int64)
        val_idx = np.where(is_val)[0]
        plan.append({"shard": si, "pos": pos_idx, "neg": np.sort(neg_sel), "val": val_idx})
        tot_pos += len(pos_idx)
        tot_neg += len(neg_sel)
        print(f"  shard {si + 1}/{len(shard_files)}: pos={len(pos_idx):,} neg_kept={take:,} val={len(val_idx):,}", flush=True)
        del cand, p1, p2, y
    print(f"pass A totals: positives={tot_pos:,} negatives_kept={tot_neg:,}", flush=True)

    # ---- pass B: featurize selected rows ----
    print("pass B: featurizing selected rows...", flush=True)
    A, B, is_s3_all = _load_side_arrays(art / "normalized", "train")
    X_parts, y_parts = [], []
    va_X, va_p1_all, va_p2_all = [], [], []
    for item in plan:
        cand = pd.read_parquet(shard_files[item["shard"]])
        p1 = _pos_of(s1_ids, cand["s1_entity_id"].to_numpy())
        p2 = _pos_of(cand_ids, cand["cand_id"].to_numpy())
        del cand
        sel = np.unique(np.concatenate([item["pos"], item["neg"], item["val"]]))
        f = compute_features(A, B, p1[sel], p2[sel], is_s3_all[p2[sel]])
        X = np.stack([f[name] for name in FEATURES], axis=1)
        del f
        pos_in_sel = np.isin(sel, item["pos"])
        val_in_sel = np.isin(sel, item["val"])
        tr_mask = ~val_in_sel
        X_parts.append(X[tr_mask])
        y_parts.append(pos_in_sel[tr_mask].astype(np.int8))
        va_X.append(X[val_in_sel])
        va_p1_all.append(p1[sel][val_in_sel])
        va_p2_all.append(p2[sel][val_in_sel])
        del X, sel, pos_in_sel, val_in_sel, tr_mask
        print(f"  featurized shard {item['shard'] + 1}/{len(plan)}", flush=True)
    X_tr = np.concatenate(X_parts)
    y_tr = np.concatenate(y_parts)
    X_va = np.concatenate(va_X)
    va_p1 = np.concatenate(va_p1_all)
    va_p2 = np.concatenate(va_p2_all)
    del X_parts, y_parts, va_X, va_p1_all, va_p2_all, A, B, is_s3_all, plan
    print(f"train matrix: {X_tr.shape}, val matrix: {X_va.shape}", flush=True)

    params = dict(cfg["model"]["params"])
    params.update({"objective": "binary", "metric": "average_precision",
                   "verbosity": -1, "seed": seed, "two_round": True})
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURES, free_raw_data=True)
    y_va = _labels_for(va_p1, va_p2, gt_val)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=FEATURES)
    booster = lgb.train(params, dtrain, num_boost_round=int(cfg["model"]["num_boost_round"]),
                        valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False)])
    booster.save_model(str(model_path))
    print(f"best iteration: {booster.best_iteration}")

    # LR baseline on a subsample
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    idx = rng.choice(len(X_tr), size=min(500_000, len(X_tr)), replace=False)
    scaler = StandardScaler().fit(X_tr[idx])
    lr = LogisticRegression(max_iter=300).fit(scaler.transform(X_tr[idx]), y_tr[idx])
    lr_ap = average_precision_score(y_va, lr.predict_proba(scaler.transform(X_va))[:, 1])
    del X_tr, y_tr, scaler, lr

    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    ap = average_precision_score(y_va, p_va)
    print(f"val pair AP: lgbm={ap:.5f} lr={lr_ap:.5f}")
    pd.DataFrame({"s1_idx": va_p1, "cand_idx": va_p2, "p_match": p_va}).to_parquet(
        val_scores_path, index=False)
    del X_va, p_va

    imp = sorted(zip(FEATURES, booster.feature_importance("gain").tolist()), key=lambda kv: -kv[1])[:15]
    eval_ = {"val_pair_ap": float(ap), "lr_baseline_ap": float(lr_ap),
             "best_iteration": int(booster.best_iteration),
             "n_train_rows": int(tot_pos + tot_neg), "n_positives": int(tot_pos),
             "feature_importance": imp}
    io_utils.json_dump(eval_, art / "model_eval.json")
    _write_model_card(eval_, Path(cfg["paths"]["reports_dir"]) / "model_card.md")

    # ---- test scoring (streaming, independently resumable) ----
    if test_scores_dir.exists() and any(test_scores_dir.glob("shard_*.parquet")):
        print("test scores fresh — skipping")
        return eval_
    test_scores_dir.mkdir(parents=True, exist_ok=True)
    t_s1_ids = pd.read_parquet(art / "features" / "test_pairs" / "s1_ids.parquet")["entity_id"].to_numpy()
    t_cand_ids = pd.read_parquet(art / "features" / "test_pairs" / "cand_ids.parquet")["entity_id"].to_numpy()
    tA, tB, t_is_s3 = _load_side_arrays(art / "normalized", "test")
    total = 0
    for si, chunk in enumerate(iter_candidate_shards("test", cfg)):
        p1 = tA.index.get_indexer(chunk["s1_entity_id"].to_numpy())
        p2 = tB.index.get_indexer(chunk["cand_id"].to_numpy())
        assert (p1 >= 0).all() and (p2 >= 0).all()
        f = compute_features(tA, tB, p1, p2, t_is_s3[p2])
        X = np.stack([f[name] for name in FEATURES], axis=1)
        del f
        p = np.empty(len(X), dtype=np.float32)
        step = 5_000_000
        for lo in range(0, len(X), step):
            p[lo:lo + step] = booster.predict(X[lo:lo + step], num_iteration=booster.best_iteration)
        io_utils.write_parquet(pd.DataFrame({"s1_idx": p1, "cand_idx": p2, "p_match": p}),
                               test_scores_dir / f"shard_{si}.parquet", 1_000_000)
        total += len(p)
        print(f"  scored test shard {si + 1}: cumulative {total:,}", flush=True)
        del X, p
    print(f"test scoring done: {total:,} pairs")
    return eval_


def _write_model_card(ev: dict, out: Path) -> None:
    lines = [
        "# Model card — LightGBM pair classifier", "",
        f"- val pair average precision: **{ev['val_pair_ap']:.5f}** (logistic baseline {ev['lr_baseline_ap']:.5f})",
        f"- best iteration: {ev['best_iteration']}",
        f"- training rows: {ev['n_train_rows']:,} (positives {ev['n_positives']:,})",
        "",
        "## Top features (gain)",
        "\n".join(f"- `{k}`: {v:,.0f}" for k, v in ev["feature_importance"]),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
