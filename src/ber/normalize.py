"""S2 — Normalization (fit-free, deterministic; maps are static + EDA-derived).

Per record we produce (parquet columns):
  name_raw, name_norm, name_core (suffix/domain-stripped), name_bag (sorted tokens),
  name_key_phonetic (collapsed letters), addr_raw, addr_norm, addr_bag (sorted),
  addr_digits, addr_postal, addr_parts ('|'-joined comma parts),
  flags: name_deva, name_domain, addr_landmark, name_tok_n, addr_tok_n

Empty addresses stay empty (legitimate data). All steps are open-set safe: no
country-specific branching — country strings flow through untouched.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from jellyfish import metaphone
from unidecode import unidecode

from . import io_utils

# ---------------- character-level cleanup ----------------
NOISE_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NON_ALNUM_SPACE_RE = re.compile(r"[^\w&]+")           # punctuation -> space (& handled first)
UNDERSCORE_RE = re.compile(r"_+")
WS_RE = re.compile(r"\s+")
DEVA_RE = re.compile(r"[\u0900-\u097F]")
ANY_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
DOMAIN_TOKEN_RE = re.compile(r"^(?:www\.)?(?P<core>.+?)\.(?:com|in|net|org|co|biz|info|us|io)$")
DOMAIN_ANYWHERE_RE = re.compile(r"(?P<core>[a-z0-9\-]{3,})\.(?:com|in|net|org|co|biz|info|us|io)\b", re.I)
DOMAIN_PREFIX_RE = re.compile(r"^www\.\S+", re.I)

# ---------------- legal-suffix vocabulary (train-frequency + standard forms; open-set safe) ----------------
SUFFIX_MULTI = (
    ("private limited",), ("pvt ltd",), ("pvt ltd",), ("pvt. ltd.",),
    ("incorporated company",), ("limited liability",),
)
SUFFIX_TOKENS = frozenset({
    # US/UK forms
    "limited", "ltd", "inc", "incorporated", "corp", "corporation", "llc", "llp",
    "lp", "plc", "co", "company", "pc", "pllc", "pa", "pte", "pty",
    # India forms (post-transliteration)
    "praivet", "privet", "privite",  # common misspellings of 'private' seen in noisy sources
    "pvt", "pvtltd",
    # France/test forms (unseen-country safe: just vocabulary, never branch on country)
    "sarl", "sas", "eurl", "srl", "sa",
})

# ---------------- address abbreviation expansion ----------------
ABBREV_MAP = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "blv": "boulevard", "ln": "lane", "hwy": "highway",
    "hwy": "highway", "fl": "floor", "flr": "floor", "apt": "apartment", "appt": "apartment",
    "ste": "suite", "nr": "near", "opp": "opposite", "n": "north", "s": "south",
    "e": "east", "w": "west",
    "sec": "sector", "sect": "sector", "soc": "society",
    "bldg": "building", "bld": "building", "mkt": "market", "twp": "township",
    "mt": "mount", "ft": "fort", "ct": "court", "pl": "place", "sq": "square",
    "ter": "terrace", "xing": "crossing", "cnyn": "canyon", "vlg": "village",
}

LANDMARK_RE = re.compile(r"\b(near|opposite|behind|beside|besides|next|adjacent|above|below)\b")
DIGIT_RUN_RE = re.compile(r"\d+")
POSTAL_RE = re.compile(r"\b(\d{5,6})\b")


def _clean_base(text: str) -> str:
    """Noise chars -> space, NFKC, transliterate non-ASCII, lowercase."""
    if not text:
        return ""
    text = NOISE_CTRL_RE.sub(" ", text)
    if ANY_NON_ASCII_RE.search(text):
        text = unidecode(text)
    text = unicodedata.normalize("NFKC", text).lower()
    return text


def _tokens(text: str) -> list[str]:
    return [t for t in NON_ALNUM_SPACE_RE.split(text) if t]


def _strip_domains(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if t in ("www", "http", "https", "com", "net", "org"):
            continue
        out.append(t)
    return out


def _strip_domain_string(text: str) -> str:
    """'wilfordhancock.com' -> 'wilfordhancock'; 'www.acme.in' -> 'acme' (pre-tokenization)."""
    text = DOMAIN_PREFIX_RE.sub("", text)
    return DOMAIN_ANYWHERE_RE.sub(lambda m: m.group("core"), text)


def _strip_trailing_suffixes(tokens: list[str], max_strips: int = 3) -> list[str]:
    toks = list(tokens)
    for _ in range(max_strips):
        if len(toks) >= 2 and toks[-1] in SUFFIX_TOKENS:
            toks.pop()
        elif len(toks) >= 3 and toks[-1] in ("limited", "ltd", "pvtltd") and (
            toks[-2] in ("private", "pvt", "praivet", "privet", "privite")
        ):
            toks.pop(); toks.pop()  # strip 'private limited' / 'pvt ltd' pairs
        else:
            break
    return toks


def _collapse_letters(token: str) -> str:
    """Collapse repeated chars: 'maarketting' -> 'markering'? no: 'marketing'. Phonetic-ish key."""
    out = []
    prev = ""
    for ch in token:
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def _phonetic_key(tokens: list[str]) -> str:
    parts = []
    for t in tokens[:4]:
        c = _collapse_letters(t)
        parts.append(metaphone(c) if len(c) >= 3 else c)
    return " ".join(parts)


def normalize_name(raw: str) -> dict:
    base = _clean_base(raw)
    base = _strip_domain_string(base)
    amp = base.replace("&", " and ")
    toks = _tokens(amp)
    toks_dom = _strip_domains(toks)
    core = _strip_trailing_suffixes(toks_dom)
    bag = " ".join(sorted(set(core)))
    return {
        "name_raw": raw,
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_bag": bag,
        "name_key_phonetic": _phonetic_key(core),
        "name_deva": int(bool(DEVA_RE.search(raw))),
        "name_domain": int(any(DOMAIN_TOKEN_RE.match(t) for t in _tokens(raw.lower()))),
        "name_tok_n": len(core),
    }


def _expand_abbrevs(tokens: list[str]) -> list[str]:
    return [ABBREV_MAP.get(t, t) for t in tokens]


def _postal(digits_runs: list[str]) -> str:
    """Prefer a 5-6 digit run (US ZIP / India PIN) closest to the end of the address."""
    cands = [d for d in digits_runs if len(d) in (5, 6)]
    return cands[-1] if cands else ""


def normalize_address(raw: str) -> dict:
    if not raw or not raw.strip():
        return {
            "addr_raw": "", "addr_norm": "", "addr_bag": "", "addr_digits": "",
            "addr_postal": "", "addr_parts": "", "addr_landmark": 0, "addr_tok_n": 0,
        }
    base = _clean_base(raw)
    base = base.replace("&", " and ")
    parts = [p.strip() for p in base.split(",") if p.strip()]
    toks = _expand_abbrevs(_tokens(base))
    digits = DIGIT_RUN_RE.findall(base)
    postal = _postal(digits)
    return {
        "addr_raw": raw,
        "addr_norm": " ".join(toks),
        "addr_bag": " ".join(sorted(set(t for t in toks if not t.isdigit()))),
        "addr_digits": " ".join(digits),
        "addr_postal": postal,
        "addr_parts": "|".join(parts),
        "addr_landmark": int(bool(LANDMARK_RE.search(base))),
        "addr_tok_n": len(toks),
    }


NAME_COLS = ["name_raw", "name_norm", "name_core", "name_bag", "name_key_phonetic",
             "name_deva", "name_domain", "name_tok_n"]
ADDR_COLS = ["addr_raw", "addr_norm", "addr_bag", "addr_digits", "addr_postal",
             "addr_parts", "addr_landmark", "addr_tok_n"]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"entity_id": df["entity_id"], "country": df["country"]})
    name_rows = [normalize_name(n) for n in df["business_name"]]
    for col in NAME_COLS:
        out[col] = [r[col] for r in name_rows]
    addr_rows = [normalize_address(a) for a in df["business_address"]]
    for col in ADDR_COLS:
        out[col] = [r[col] for r in addr_rows]
    return out


# ---------------- stage runner ----------------
FILES = {
    "train_s1": ("train_source1.tsv",), "train_s2": ("train_source2.tsv",),
    "train_s3": ("train_source3.tsv",), "test_s1": ("test_source1.tsv",),
    "test_s2": ("test_source2.tsv",), "test_s3": ("test_source3.tsv",),
}


def _normalize_file(cfg: dict, split: str, filename: str) -> Path:
    out_path = Path(cfg["paths"]["artifacts_dir"]) / "normalized" / f"{split}.parquet"
    params = {"normalizer_version": 3}
    if io_utils.manifest_ok(out_path, {"src": Path(cfg["paths"]["dataset_train_dir"]) / filename
                                       if split.startswith("train") else
                                       Path(cfg["paths"]["dataset_test_dir"]) / filename}, params):
        print(f"  [{split}] fresh — skipping")
        return out_path
    n_total = 0
    parts: list[pd.DataFrame] = []
    src = (Path(cfg["paths"]["dataset_train_dir"]) if split.startswith("train")
           else Path(cfg["paths"]["dataset_test_dir"])) / filename
    for i, chunk in enumerate(io_utils.read_tsv_chunks(src, cfg["io"]["chunk_rows"])):
        parts.append(normalize_dataframe(chunk))
        n_total += len(chunk)
        print(f"  [{split}] chunk {i + 1}: {n_total:,} rows", flush=True)
    norm = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_parquet(norm, out_path, cfg["io"]["parquet_row_group_size"])
    io_utils.save_manifest(out_path, {"src": src}, params, extra={"rows": len(norm)})
    return out_path


def run(cfg: dict, force: bool = False) -> dict:
    results = {}
    for split, (filename,) in FILES.items():
        if force:
            p = Path(cfg["paths"]["artifacts_dir"]) / "normalized" / f"{split}.parquet"
            p.unlink(missing_ok=True)
            p.with_suffix(".meta.json").unlink(missing_ok=True)
        results[split] = str(_normalize_file(cfg, split, filename))
    print("S2 normalization complete ->", results)
    return results
