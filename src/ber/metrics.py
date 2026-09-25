"""Entity-level macro F0.5 + pair metrics.

Metric definition (challenge): per Source-1 entity, F0.5 over predicted vs true
matched-id SETS; macro-average over ALL S1 entities (singletons included:
correct empty prediction = 1.0, any prediction on a true singleton = 0.0).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils


def pair_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, scores))


def entity_f05(pred: dict[str, set[str]], truth: dict[str, set[str]],
               per_country: dict[str, str] | None = None) -> dict:
    """Macro F0.5 over entities present in `truth`.

    pred: s1_id -> predicted matched-id set (may be empty)
    truth: s1_id -> GT matched-id set (may be empty)
    per_country: optional s1_id -> country for country breakdown.
    """
    f05s, by_country = [], defaultdict(list)
    n_singleton = 0
    singleton_correct = 0
    for s1, true_set in truth.items():
        pred_set = pred.get(s1, set())
        if not true_set:
            # Singleton rule (README): correct empty prediction = 1.0, any
            # predicted match = 0.0. Must be checked BEFORE prec/rec (0/0).
            f = 1.0 if not pred_set else 0.0
        else:
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f = (1.25 * prec * rec / (0.25 * prec + rec)) if (prec + rec) else 0.0
        f05s.append(f)
        if per_country:
            by_country[per_country.get(s1, "<none>")].append(f)
        if not true_set:
            n_singleton += 1
            singleton_correct += (not pred_set)
    macro = float(np.mean(f05s)) if f05s else 0.0
    out = {
        "macro_f05": macro,
        "n_entities": len(truth),
        "singleton_rate": n_singleton / len(truth) if truth else 0.0,
        "singleton_accuracy": singleton_correct / n_singleton if n_singleton else 0.0,
    }
    if per_country:
        out["by_country"] = {c: float(np.mean(v)) for c, v in sorted(by_country.items())}
    return out


def gt_dict(cfg: dict, only_val: bool | None = None, manifest: pd.DataFrame | None = None) -> dict[str, set[str]]:
    """Load train ground truth as s1_id -> set(matched ids), optionally val-only."""
    path = Path(cfg["paths"]["dataset_train_dir"]) / "train_ground_truth.tsv"
    out: dict[str, set[str]] = {}
    val_ids = None
    if only_val:
        if manifest is None:
            manifest = pd.read_parquet(cfg["splits"]["split_manifest"])
        val_ids = set(manifest.loc[manifest["split"] == "val", "s1_entity_id"])
    for df in io_utils.read_tsv_chunks(path, 1_000_000):
        for s1, m in zip(df["source1_entity_id"], df["matched_entity_ids"]):
            if val_ids is not None and s1 not in val_ids:
                continue
            out[s1] = set(m.split(",")) if m else set()
    return out


def predictions_at_threshold(pairs: pd.DataFrame, threshold: float,
                             prob_col: str = "p_match") -> dict[str, set[str]]:
    """Group pairs with p_match >= threshold into s1_id -> set(cand ids)."""
    sel = pairs[pairs[prob_col] >= threshold]
    return {s1: set(g) for s1, g in sel.groupby("s1_entity_id")["cand_id"]}
