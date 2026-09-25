"""Streaming I/O + resumability helpers.

Design rules:
- Never load a full multi-GB TSV into memory; everything is chunked.
- Missing/empty fields stay as "" strings (no NaN ambiguity for noisy data).
- Every stage writes a `.meta.json` manifest (input fingerprints + params) and
  skips its work when the manifest shows a fresh, matching checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def count_rows(path: str | Path) -> int:
    """Line count minus header, streamed."""
    n = 0
    with open(path, "rb") as f:
        f.readline()
        for _ in f:
            n += 1
    return n


def file_fingerprint(path: str | Path) -> dict:
    """Cheap content fingerprint: size + head/tail hashes (not a full sha256 of GB files)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))  # head 1 MiB
        if size > (1 << 20):
            f.seek(-min(1 << 20, size - (1 << 20)), os.SEEK_END)
            h.update(f.read(1 << 20))  # tail 1 MiB
    h.update(str(size).encode())
    return {"path": str(p), "size": size, "fp": h.hexdigest()[:32]}


def params_fingerprint(params: Any) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_manifest(out_path: str | Path, inputs: dict, params: dict, extra: dict | None = None) -> None:
    meta = {
        "inputs": {k: file_fingerprint(v) if isinstance(v, (str, Path)) else v for k, v in inputs.items()},
        "params_fp": params_fingerprint(params),
        "params": params,
    }
    if extra:
        meta.update(extra)
    Path(out_path).with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )


def manifest_ok(out_path: str | Path, inputs: dict, params: dict) -> bool:
    """True when the checkpoint exists AND its inputs/params still match (resumability)."""
    meta_path = Path(out_path).with_suffix(".meta.json")
    if not Path(out_path).exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("params_fp") != params_fingerprint(params):
        return False
    for key, val in inputs.items():
        if isinstance(val, (str, Path)):
            if meta.get("inputs", {}).get(key, {}).get("fp") != file_fingerprint(val)["fp"]:
                return False
        elif meta.get("inputs", {}).get(key) != val:
            return False
    return True


def read_tsv_chunks(
    path: str | Path, chunk_rows: int = 500_000, columns: list[str] | None = None
) -> Iterator[pd.DataFrame]:
    """Stream a TSV as string DataFrames in chunks. QUOTE_NONE: quotes are literal data."""
    import pyarrow.csv as pac

    parse = pac.ParseOptions(delimiter="\t", quote_char=False, double_quote=False, escape_char=False)
    read = pac.ReadOptions(block_size=64 << 20)
    conv = pac.ConvertOptions(include_columns=columns, strings_can_be_null=False) if columns else pac.ConvertOptions(strings_can_be_null=False)
    with pac.open_csv(str(path), parse_options=parse, read_options=read, convert_options=conv) as reader:
        buf: list[pd.DataFrame] = []
        nrows = 0
        try:
            while True:
                batch = reader.read_next_batch()
                df = batch.to_pandas(types_mapper=pd.ArrowDtype)
                buf.append(df)
                nrows += len(df)
                if nrows >= chunk_rows:
                    out = pd.concat(buf, ignore_index=True) if len(buf) > 1 else buf[0]
                    yield out.astype(str)
                    buf, nrows = [], 0
        except StopIteration:
            pass
        if buf:
            out = pd.concat(buf, ignore_index=True) if len(buf) > 1 else buf[0]
            yield out.astype(str)


def df_to_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """Canonical TSV write: tab sep, no quoting, UTF-8, no index."""
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")


def write_parquet(df: pd.DataFrame, path: str | Path, row_group_size: int = 500_000) -> None:
    df.to_parquet(path, engine="pyarrow", row_group_size=row_group_size, index=False)


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", columns=columns)


def iter_parquet_chunks(path: str | Path, batch_rows: int = 2_000_000) -> Iterator[pd.DataFrame]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_rows):
        yield batch.to_pandas()


def json_dump(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def json_load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
