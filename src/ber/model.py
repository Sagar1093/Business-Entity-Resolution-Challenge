"""S5 — LightGBM pair classifier (feature-shard streaming; low-RAM).

Data flow (train):
  pass A: stream FEATURE shards -> labels via packed GT lookup;
          per-shard seeded negative sampling (neg_ratio per positive);
          val-entity rows capped per entity (val_cap_per_s1).
  pass B: re-stream feature shards, keep selected rows only, assemble
          float32 matrices (~1-2 GB at emergency caps).
  train LightGBM w/ early stopping on val; LR baseline; save model.
Val scoring -> artifacts/features/val_scores.parquet (s1_idx, cand_idx, p_match)
Test scoring -> artifacts/features/test_scores/shard_* (same columns)

Features are NEVER recomputed here — they are read from the S4 shards
(uint8, decoded to float32 per shard).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, metrics
from .features import FEATURES, iter_feature_shards

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
    val_cap = int(cfg["model"].get("val_cap_per_s1", 0)) or None
    seed = int(cfg["seed"])

    # ---- pass A: labels/counts + per-shard selection plan ----
    plan: list[dict] = []
    tot_pos = tot_neg = tot_val = 0
    print("pass A: scanning feature shards for labels...", flush=True)
    for si, (r1, r2, _Q) in enumerate(iter_feature_shards("train", cfg)):
        y = _labels_for(r1, r2, gt_train)
        is_val = val_row_mask[r1]
        pos_idx = np.where((y == 1) & ~is_val)[0]
        neg_idx = np.where((y == 0) & ~is_val)[0]
        rng = np.random.default_rng(seed + si)
        take = min(len(neg_idx), neg_ratio * len(pos_idx))
        neg_sel = rng.choice(neg_idx, size=take, replace=False) if take else np.empty(0, np.int64)
        val_idx = np.where(is_val)[0]
        if val_cap is not None and len(val_idx):
            # per-entity cap within this shard (groups contiguous per entity)
            ent = r1[val_idx]
            order = np.argsort(ent, kind="stable")
            ent_sorted = ent[order]
            starts = np.r_[0, np.flatnonzero(np.diff(ent_sorted)) + 1]
            ranks = np.arange(len(ent_sorted)) - np.repeat(starts, np.diff(np.r_[starts, len(ent_sorted)]))
            keep_sorted = ranks < val_cap
            keep = np.zeros(len(val_idx), dtype=bool)
            keep[order[keep_sorted]] = True
            val_idx = val_idx[keep]
        plan.append({"si": si, "pos": pos_idx, "neg": np.sort(neg_sel), "val": val_idx})
        tot_pos += len(pos_idx)
        tot_neg += len(neg_sel)
        tot_val += len(val_idx)
        print(f"  shard {si + 1}: pos={len(pos_idx):,} neg_kept={take:,} val={len(val_idx):,}", flush=True)
        del r1, r2, y, is_val
    print(f"pass A totals: positives={tot_pos:,} negatives_kept={tot_neg:,} val_rows={tot_val:,}", flush=True)

    # ---- pass B: assemble matrices from the persisted features ----
    print("pass B: assembling matrices...", flush=True)
    X_tr_parts: list[np.ndarray] = []
    y_tr_parts: list[np.ndarray] = []
    X_va_parts: list[np.ndarray] = []
    va_p1_parts: list[np.ndarray] = []
    va_p2_parts: list[np.ndarray] = []
    for si, (r1, r2, Q) in enumerate(iter_feature_shards("train", cfg)):
        item = plan[si]
        sel = np.unique(np.concatenate([item["pos"], item["neg"], item["val"]])) if (
            len(item["pos"]) or len(item["neg"]) or len(item["val"])) else np.empty(0, np.int64)
        if len(sel):
            X = Q[sel].astype(np.float32) / 255.0
            pos_in_sel = np.isin(sel, item["pos"])
            val_in_sel = np.isin(sel, item["val"])
            tr_mask = ~val_in_sel
            X_tr_parts.append(X[tr_mask])
            y_tr_parts.append(pos_in_sel[tr_mask].astype(np.int8))
            X_va_parts.append(X[val_in_sel])
            va_p1_parts.append(r1[sel][val_in_sel])
            va_p2_parts.append(r2[sel][val_in_sel])
            del X
        del Q, r1, r2, sel
    X_tr = np.concatenate(X_tr_parts) if X_tr_parts else np.empty((0, len(FEATURES)), np.float32)
    y_tr = np.concatenate(y_tr_parts) if y_tr_parts else np.empty(0, np.int8)
    X_va = np.concatenate(X_va_parts) if X_va_parts else np.empty((0, len(FEATURES)), np.float32)
    va_p1 = np.concatenate(va_p1_parts) if va_p1_parts else np.empty(0, np.int64)
    va_p2 = np.concatenate(va_p2_parts) if va_p2_parts else np.empty(0, np.int64)
    del X_tr_parts, y_tr_parts, X_va_parts, va_p1_parts, va_p2_parts, plan
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

    # val scores for S7 calibration
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration).astype(np.float32)
    pd.DataFrame({"s1_idx": va_p1, "cand_idx": va_p2, "p_match": p_va}).to_parquet(
        val_scores_path, index=False)

    # LR baseline on a subsample
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=min(300_000, len(X_tr)), replace=False)
    scaler = StandardScaler().fit(X_tr[idx])
    lr = LogisticRegression(max_iter=200).fit(scaler.transform(X_tr[idx]), y_tr[idx])
    lr_ap = average_precision_score(y_va, lr.predict_proba(scaler.transform(X_va))[:, 1])
    ap = average_precision_score(y_va, p_va)
    print(f"val pair AP: lgbm={ap:.5f} lr={lr_ap:.5f}")
    del X_tr, y_tr, X_va, p_va, scaler, lr

    imp = sorted(zip(FEATURES, booster.feature_importance("gain").tolist()), key=lambda kv: -kv[1])[:15]
    eval_ = {"val_pair_ap": float(ap), "lr_baseline_ap": float(lr_ap),
             "best_iteration": int(booster.best_iteration),
             "n_train_rows": int(tot_pos + tot_neg), "n_positives": int(tot_pos),
             "n_val_rows": int(tot_val),
             "feature_importance": imp}
    io_utils.json_dump(eval_, art / "model_eval.json")
    _write_model_card(eval_, Path(cfg["paths"]["reports_dir"]) / "model_card.md")

    # ---- test scoring: read persisted test features, predict, save ----
    if test_scores_dir.exists() and any(test_scores_dir.glob("shard_*.parquet")):
        print("test scores fresh — skipping")
        return eval_
    test_scores_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for si, (r1, r2, Q) in enumerate(iter_feature_shards("test", cfg)):
        X = Q.astype(np.float32) / 255.0
        p = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
        del X, Q
        io_utils.write_parquet(pd.DataFrame({
            "s1_idx": r1.astype(np.int32),
            "cand_idx": r2.astype(np.int32),
            "p_match": p,
        }), test_scores_dir / f"shard_{si}.parquet", 1_000_000)
        total += len(p)
        print(f"  scored test shard {si + 1}: cumulative {total:,}", flush=True)
        del r1, r2, p
    print(f"test scoring done: {total:,} pairs")
    return eval_


def _write_model_card(ev: dict, out: Path) -> None:
    lines = [
        "# Model card — LightGBM pair classifier", "",
        f"- val pair average precision: **{ev['val_pair_ap']:.5f}** (logistic baseline {ev['lr_baseline_ap']:.5f})",
        f"- best iteration: {ev['best_iteration']}",
        f"- training rows: {ev['n_train_rows']:,} (positives {ev['n_positives']:,})",
        f"- val calibration rows: {ev['n_val_rows']:,}",
        "",
        "## Top features (gain)",
        "\n".join(f"- `{k}`: {v:,.0f}" for k, v in ev["feature_importance"]),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
