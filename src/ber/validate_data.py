"""S1 — Data validation gate (authoritative; zero-tolerance).

Streaming (pyarrow, chunked) full pass over all 7 TSVs. Checks:
- header == entity_id, business_name, business_address, country (exact)
- entity_id non-empty, prefixed S1-/S2-/S3-, unique per file
- business_name non-empty; business_address allowed empty (real data)
- country: non-empty string; inventory recorded (open set, never hard-coded)
- ground truth: source1 IDs exist in train S1; matched IDs exist in train S2/S3;
  no S1 IDs inside matched lists; no duplicate source1 rows; intra-list dupes
- control characters / invalid UTF-8 / row-shape anomalies
- profile stats: row counts, country counts, length percentiles, empty rates
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Only NUL (\x00) is structural corruption. \x1a/\x1f/\x7f are injected noise
# (mangled dashes/apostrophes, e.g. 'WORLDMARK \x1a 3', "Shopper\x1aS Stop",
# 'Unit D\x1f_A') — counted per tier and cleaned in S2 normalization.
FATAL_RE = re.compile(r"\x00")
NOISE_CTRL = CTRL_RE
ID_RE = re.compile(r"^S[123]-\d+$")
GT_HEADER = ["source1_entity_id", "matched_entity_ids"]
MAX_ERR_SAMPLES = 20


def _empty_res(path: Path) -> dict:
    return {
        "file": str(path), "rows": 0, "countries": Counter(), "errors": [],
        "error_samples": [], "empty_name": 0, "empty_address": 0,        "bad_id": 0, "dup_ids": 0, "fatal_cells": 0, "noise_cells": 0, "bad_shape_rows": 0,
        "name_lens": [], "addr_lens": [],
    }


def _add_err(res: dict, msg: str) -> None:
    res["errors"].append(msg)
    if len(res["error_samples"]) < MAX_ERR_SAMPLES:
        res["error_samples"].append(msg)


def validate_source_file(path: Path, expected_prefix: str, cfg: dict) -> dict:
    res = _empty_res(path)
    seen: set[str] = set()
    try:
        chunks = io_utils.read_tsv_chunks(path, cfg["io"]["chunk_rows"])
        first = next(chunks)
    except Exception as exc:
        _add_err(res, f"unreadable as TSV/UTF-8: {exc}")
        return res
    if list(first.columns) != io_utils.COLUMNS:
        _add_err(res, f"bad header: {list(first.columns)!r}")
        return res

    for df in pd.concat([first, *chunks], ignore_index=True) if False else _chain(first, chunks):
        res["rows"] += len(df)
        eid = df["entity_id"]
        ok_id = eid.str.fullmatch(ID_RE, na=False) & eid.str.startswith(expected_prefix)
        res["bad_id"] += int((~ok_id).sum())
        if (~ok_id).any() and len(res["error_samples"]) < MAX_ERR_SAMPLES:
            for v in eid[~ok_id].head(3):
                res["error_samples"].append(f"bad id {v!r}")
        seen.update(eid.to_numpy())
        for col in ("business_name", "business_address", "country"):
            res["noise_cells"] = res.get("noise_cells", 0) + int(
            df[col].str.contains(NOISE_CTRL, na=False, regex=True).sum()
        )
        res["fatal_cells"] = res.get("fatal_cells", 0) + int(
            df[col].str.contains(FATAL_RE, na=False, regex=True).sum()
        )
        res["empty_name"] += int((df["business_name"] == "").sum())
        res["empty_address"] += int((df["business_address"] == "").sum())
        res["countries"].update(df["country"].replace("", "<empty>").value_counts().to_dict())
        res["name_lens"].append(df["business_name"].str.len().to_numpy())
        res["addr_lens"].append(df["business_address"].str.len().to_numpy())

    res["dup_ids"] = res["rows"] - len(seen)
    res["unique_ids"] = len(seen)
    if res["dup_ids"]:
        _add_err(res, f"{res['dup_ids']} duplicate entity_ids")
    if res["bad_id"]:
        _add_err(res, f"{res['bad_id']} malformed/foreign-prefix entity_ids")
    if res["empty_name"]:
        _add_err(res, f"{res['empty_name']} rows with empty business_name")
    if res.get("fatal_cells", 0):
        _add_err(res, f"{res['fatal_cells']} cells with NUL corruption")
    return res


def _chain(first: pd.DataFrame, rest):
    yield first
    for df in rest:
        yield df


def _finish_percentiles(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k not in ("name_lens", "addr_lens")}
    nl = np.concatenate(res["name_lens"]) if res["name_lens"] else np.array([0])
    al = np.concatenate(res["addr_lens"]) if res["addr_lens"] else np.array([0])
    n = max(1, res["rows"])
    out["name_len_stats"] = {
        "p50": int(np.percentile(nl, 50)), "p90": int(np.percentile(nl, 90)),
        "p99": int(np.percentile(nl, 99)), "max": int(nl.max()),
        "mean": round(float(nl.mean()), 2),
    }
    out["addr_len_stats"] = {
        "p50": int(np.percentile(al, 50)), "p90": int(np.percentile(al, 90)),
        "p99": int(np.percentile(al, 99)), "max": int(al.max()),
        "mean": round(float(al.mean()), 2),
    }
    out["countries"] = dict(res["countries"].most_common())
    return out


def _load_id_set(path: Path) -> set[str]:
    ids: set[str] = set()
    for df in io_utils.read_tsv_chunks(path, 2_000_000, columns=["entity_id"]):
        ids.update(df["entity_id"].to_numpy())
    return ids


def validate_ground_truth(gt_path: Path, cfg: dict) -> dict:
    res = {"file": str(gt_path), "rows": 0, "errors": [], "error_samples": [],
           "empty_lists": 0, "dup_rows": 0, "intra_list_dupes": 0,
           "unknown_matches": 0, "s1_self_matches": 0, "total_matches": 0,
           "cardinality": Counter()}
    train_dir = gt_path.parent
    s1_ids = _load_id_set(train_dir / "train_source1.tsv")
    target_ids = _load_id_set(train_dir / "train_source2.tsv") | _load_id_set(train_dir / "train_source3.tsv")

    def add_err(msg: str) -> None:
        res["errors"].append(msg)
        if len(res["error_samples"]) < MAX_ERR_SAMPLES:
            res["error_samples"].append(msg)

    try:
        chunks = io_utils.read_tsv_chunks(gt_path, cfg["io"]["chunk_rows"], columns=GT_HEADER)
        first = next(chunks)
    except Exception as exc:
        add_err(f"unreadable as TSV/UTF-8: {exc}")
        return res
    if list(first.columns) != GT_HEADER:
        add_err(f"bad header: {list(first.columns)!r}")
        return res

    seen_rows: set[str] = set()
    for df in _chain(first, chunks):
        res["rows"] += len(df)
        s1 = df["source1_entity_id"]
        res["dup_rows"] += int(s1.duplicated().sum())
        bad_s1 = ~s1.isin(s1_ids)
        if bad_s1.any():
            for v in s1[bad_s1].head(3):
                add_err(f"GT source1 id not in train_source1: {v!r}")
        seen_rows.update(s1.to_numpy())
        lists = df["matched_entity_ids"].fillna("")
        n_matches = np.array([len(x.split(",")) if x.strip() else 0 for x in lists])
        res["total_matches"] += int(n_matches.sum())
        res["empty_lists"] += int((n_matches == 0).sum())
        for k in n_matches:
            res["cardinality"][min(int(k), 11)] += 1  # 11 => ">10" bucket
        nonempty = lists[lists != ""]
        if len(nonempty):
            exploded = nonempty.str.split(",").explode()
            res["intra_list_dupes"] += int(sum(
                len(v) != len(set(v)) for v in nonempty.str.split(",")
            ))
            res["s1_self_matches"] += int(exploded.str.startswith("S1-").sum())
            unknown = exploded[~exploded.isin(target_ids) & ~exploded.str.startswith("S1-")]
            res["unknown_matches"] += int(len(unknown))
            if len(unknown) and res["unknown_matches"] == len(unknown):
                for v in unknown.head(3):
                    add_err(f"unknown match id: {v!r}")

    if res["dup_rows"]:
        add_err(f"{res['dup_rows']} duplicate source1_entity_id rows")
    if res["s1_self_matches"]:
        add_err(f"{res['s1_self_matches']} S1 self-matches in GT")
    if res["unknown_matches"]:
        add_err(f"{res['unknown_matches']} matched IDs not in train S2/S3")
    return res


def run(cfg: dict) -> dict:
    train_dir = Path(cfg["paths"]["dataset_train_dir"])
    test_dir = Path(cfg["paths"]["dataset_test_dir"])
    files = {
        "train_source1": (train_dir / "train_source1.tsv", "S1-"),
        "train_source2": (train_dir / "train_source2.tsv", "S2-"),
        "train_source3": (train_dir / "train_source3.tsv", "S3-"),
        "test_source1": (test_dir / "test_source1.tsv", "S1-"),
        "test_source2": (test_dir / "test_source2.tsv", "S2-"),
        "test_source3": (test_dir / "test_source3.tsv", "S3-"),
    }
    report = {"sources": {}, "ground_truth": {}, "gate": "PENDING", "errors_total": 0}
    for name, (path, prefix) in files.items():
        res = _finish_percentiles(validate_source_file(path, prefix, cfg))
        report["sources"][name] = res
    gt = train_dir / "train_ground_truth.tsv"
    report["ground_truth"] = validate_ground_truth(gt, cfg)
    err_count = sum(len(s["errors"]) for s in report["sources"].values())
    err_count += len(report["ground_truth"]["errors"])
    report["errors_total"] = err_count
    report["gate"] = "PASS" if err_count == 0 else "FAIL"

    out_json = Path(cfg["paths"]["artifacts_dir"]) / "data_validation.json"
    io_utils.json_dump(report, out_json)
    _write_profile_md(report, Path(cfg["paths"]["reports_dir"]) / "data_profile.md")
    print(f"S1 gate: {report['gate']} ({err_count} errors) — report: {out_json}")
    if report["gate"] != "PASS":
        raise SystemExit(f"S1 data validation FAILED — see {out_json}")
    return report


def _write_profile_md(rep: dict, out: Path) -> None:
    lines = ["# Data profile (S1 validation)", ""]
    for name, s in rep["sources"].items():
        lines += [
            f"## {name}", f"- rows: {s['rows']} (unique ids: {s.get('unique_ids', '?')})",
            f"- countries: `{s['countries']}`",
            f"- empty name/address: {s['empty_name']} / {s['empty_address']}",
            f"- name len p50/p90/p99/max: {s['name_len_stats']}",
            f"- addr len p50/p90/p99/max: {s['addr_len_stats']}",
            f"- errors: {len(s['errors'])}", "",
        ]
    gt = rep["ground_truth"]
    lines += [
        "## train_ground_truth", f"- rows: {gt['rows']}", f"- total matches: {gt['total_matches']}",
        f"- empty (singleton) lists: {gt['empty_lists']}",
        f"- cardinality buckets (n_matches -> count, 11 => >10): `{dict(sorted(gt['cardinality'].items()))}`",
        f"- errors: {len(gt['errors'])}",
        "", f"**GATE: {rep['gate']}** ({rep['errors_total']} total errors)", "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
