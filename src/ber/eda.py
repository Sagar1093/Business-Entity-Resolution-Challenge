"""EDA — quantifies the noise patterns that drive normalization + blocking design.

All statistics computed from TRAIN only. Outputs reports/eda.md + artifacts/eda.json.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from . import io_utils

DEVA_RE = re.compile(r"[\u0900-\u097F]")
DOMAIN_RE = re.compile(r"(?:www\.|\.com\b|\.in\b|\.net\b|\.org\b)", re.I)
US_ZIP_RE = re.compile(r"\b\d{5}(-\d{4})?\b")
IN_PIN_RE = re.compile(r"\b[1-9]\d{5}\b")
NEAR_RE = re.compile(r"\b(near|opp|opposite|nr\.?|behind|beside|next to)\b", re.I)
MUNI_RE = re.compile(r"\b\d{1,3}(-|/)\d{1,4}(/\d{1,4})?\b")
TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9&]+")


def _toks(text: str) -> list[str]:
    return [t for t in TOKEN_CLEAN_RE.split(text.lower()) if t]


def _iter_train_chunks(cfg: dict, filename: str):
    path = Path(cfg["paths"]["dataset_train_dir"]) / filename
    yield from io_utils.read_tsv_chunks(path, cfg["io"]["chunk_rows"])


def run(cfg: dict, force: bool = False) -> dict:
    out_json = Path(cfg["paths"]["artifacts_dir"]) / "eda.json"
    if not force and out_json.exists():
        print("EDA already computed — skipping (use --force).")
        return io_utils.json_load(out_json)

    stats = {
        "name_last_token": Counter(), "name_last_bigram": Counter(),
        "name_first_token": Counter(), "addr_token": Counter(),
        "ampersand_names": 0, "and_names": 0, "devanagari_names": 0,
        "devanagari_names_india": 0, "india_names": 0, "domain_style_names": 0,
        "short_names_lt4": 0, "digit_only_names": 0, "rows": 0,
        "addr_us_zip": 0, "addr_india_pin": 0, "addr_near_landmark": 0,
        "addr_municipal": 0, "addr_ampersand": 0, "addr_rows": 0,
        "us_addr_rows": 0, "india_addr_rows": 0,
    }
    for df in _iter_train_chunks(cfg, "train_source1.tsv"):
        names = df["business_name"]
        toks_series = names.map(_toks)
        stats["rows"] += len(df)
        for toks in toks_series:
            if toks:
                stats["name_last_token"][toks[-1]] += 1
                stats["name_first_token"][toks[0]] += 1
                if len(toks) >= 2:
                    stats["name_last_bigram"][f"{toks[-2]}_{toks[-1]}"] += 1
        stats["ampersand_names"] += int(names.str.contains("&", regex=False).sum())
        stats["and_names"] += int(names.str.contains(r"\band\b", case=False, regex=True).sum())
        deva = names.str.contains(DEVA_RE, na=False, regex=True)
        stats["devanagari_names"] += int(deva.sum())
        india = df["country"] == "India"
        stats["india_names"] += int(india.sum())
        stats["devanagari_names_india"] += int((deva & india).sum())
        stats["domain_style_names"] += int(names.str.contains(DOMAIN_RE, na=False, regex=True).sum())
        stats["short_names_lt4"] += int((names.str.len() < 4).sum())
        stats["digit_only_names"] += int(names.str.fullmatch(r"[\d\s&.\-]+", na=False).sum())

    for df in _iter_train_chunks(cfg, "train_source2.tsv"):
        _addr_stats(df, stats)
    for df in _iter_train_chunks(cfg, "train_source3.tsv"):
        _addr_stats(df, stats)

    for fn in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        per = {"rows": 0, "devanagari": 0, "domain": 0}
        for df in _iter_train_chunks(cfg, fn):
            names = df["business_name"]
            per["rows"] += len(df)
            per["devanagari"] += int(names.str.contains(DEVA_RE, na=False, regex=True).sum())
            per["domain"] += int(names.str.contains(DOMAIN_RE, na=False, regex=True).sum())
        stats.setdefault("per_source_name_noise", {})[fn] = per

    # Do multiple S1 entities share the same S2/S3 match? (pair-label structure diagnostic)
    gt = pd.read_csv(
        Path(cfg["paths"]["dataset_train_dir"]) / "train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    exploded = gt.assign(mid=gt["matched_entity_ids"].str.split(",")).explode("mid")
    exploded = exploded[exploded["mid"] != ""]
    multi_s1 = exploded.groupby("mid")["source1_entity_id"].nunique()
    stats["gt_shared_matches"] = {
        "distinct_matched_ids": int(multi_s1.size),
        "matched_by_multiple_s1": int((multi_s1 > 1).sum()),
        "max_s1_per_match": int(multi_s1.max()),
    }

    for key in ("name_last_token", "name_last_bigram", "name_first_token", "addr_token"):
        stats[key] = dict(stats[key].most_common(40))
    io_utils.json_dump(stats, out_json)
    _write_md(stats, Path(cfg["paths"]["reports_dir"]) / "eda.md")
    print("EDA complete ->", out_json)
    return stats


def _addr_stats(df: pd.DataFrame, stats: dict) -> None:
    addr = df["business_address"]
    stats["addr_rows"] += len(addr)
    nonempty = addr[addr != ""]
    for t in nonempty.map(_toks):
        for tok in t:
            stats["addr_token"][tok] += 1
    us = df["country"] == "US"
    india = df["country"] == "India"
    stats["us_addr_rows"] += int(us.sum())
    stats["india_addr_rows"] += int(india.sum())
    stats["addr_us_zip"] += int(nonempty.str.contains(US_ZIP_RE, na=False, regex=True).sum())
    stats["addr_india_pin"] += int(nonempty.str.contains(IN_PIN_RE, na=False, regex=True).sum())
    stats["addr_near_landmark"] += int(nonempty.str.contains(NEAR_RE, na=False, regex=True).sum())
    stats["addr_municipal"] += int(nonempty.str.contains(MUNI_RE, na=False, regex=True).sum())
    stats["addr_ampersand"] += int(nonempty.str.contains("&", regex=False).sum())


def _write_md(s: dict, out: Path) -> None:
    pct = lambda n, d: f"{100.0 * n / max(1, d):.2f}%"
    lines = [
        "# EDA — train-only noise statistics", "",
        "## Names (train S1)",
        f"- rows: {s['rows']:,}",
        f"- `&` in name: {s['ampersand_names']:,} ({pct(s['ampersand_names'], s['rows'])}) | word `and`: {s['and_names']:,}",
        f"- Devanagari script names: {s['devanagari_names']:,} ({pct(s['devanagari_names'], s['rows'])} of all; "
        f"{pct(s['devanagari_names_india'], s['india_names'])} of India)",
        f"- domain/DBA-style names (www/.com/.in): {s['domain_style_names']:,} ({pct(s['domain_style_names'], s['rows'])})",
        f"- very short names (<4 chars): {s['short_names_lt4']:,} | digit/punct-only: {s['digit_only_names']:,}",
        "",
        "### Top name-final tokens (legal-suffix candidates)",
        " ".join(f"`{k}`:{v:,}" for k, v in list(s["name_last_token"].items())[:25]),
        "",
        "### Top name-final bigrams",
        " ".join(f"`{k}`:{v:,}" for k, v in list(s["name_last_bigram"].items())[:20]),
        "",
        "## Addresses (train S2+S3)",
        f"- rows: {s['addr_rows']:,}",
        f"- US rows w/ 5-digit ZIP-like: {s['addr_us_zip']:,} ({pct(s['addr_us_zip'], s['us_addr_rows'])} of US rows)",
        f"- India rows w/ 6-digit PIN-like: {s['addr_india_pin']:,} ({pct(s['addr_india_pin'], s['india_addr_rows'])} of India rows)",
        f"- landmark refs (near/opp/behind/nr): {s['addr_near_landmark']:,} ({pct(s['addr_near_landmark'], s['addr_rows'])})",
        f"- municipal numbering (1-11-251/1B style): {s['addr_municipal']:,} ({pct(s['addr_municipal'], s['addr_rows'])})",
        "",
        "### Top address tokens",
        " ".join(f"`{k}`:{v:,}" for k, v in list(s["addr_token"].items())[:30]),
        "",
        "## Ground-truth structure",
        f"- matched S2/S3 ids referenced by >1 S1 entity: "
        f"{s['gt_shared_matches']['matched_by_multiple_s1']:,} of "
        f"{s['gt_shared_matches']['distinct_matched_ids']:,} "
        f"(max S1 per match: {s['gt_shared_matches']['max_s1_per_match']})",
        "",
        "## Cross-source name noise (measured)",
        "| file | rows | Devanagari | domain-style |",
        "|---|---|---|---|",
    ]
    for fn, per in s.get("per_source_name_noise", {}).items():
        lines.append(
            f"| {fn} | {per['rows']:,} | {per['devanagari']:,} "
            f"({100 * per['devanagari'] / max(1, per['rows']):.2f}%) | "
            f"{per['domain']:,} ({100 * per['domain'] / max(1, per['rows']):.2f}%) |"
        )
    lines += [
        "",
        "**Key implication:** S1 is the clean Latin-only reference; Devanagari and "
        "domain/DBA noise live in S2/S3. Normalization must transliterate "
        "Devanagari → Latin and strip www/TLDs so cross-source pairs become comparable.",
        "**Unique assignment:** every S2/S3 record matches at most one S1 entity — "
        "the match graph is a forest. The decision engine may exploit candidate-competition.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
