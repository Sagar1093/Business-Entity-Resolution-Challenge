"""S6b — Qwen2.5-7B-Instruct adjudication of the highly-ambiguous band (guarded).

HARD GUARD (challenge constraint 5): the exact HF repo must be Apache-2.0 and
<= 8B parameters; both are verified BEFORE download/load. Any guard failure or
load error auto-skips the stage (S7 falls back to reranker+LGBM signals).

Input: S5 val/test scores; band = p_match in [band_low, band_high], capped at
max_pairs_fraction of all pairs. Output: {split}_llm.parquet with llm_yes in
{0.0, 0.5, 1.0} per pair (yes / uncertain / no) joined by S7.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils

GUARD_CACHE = Path("artifacts/cache/qwen_guard.json")
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")

SYSTEM_PROMPT = (
    "You are a precise entity-resolution judge. Two business records are shown. "
    "Decide ONLY from the given text whether both records refer to the same "
    "real-world business. Ignore legal suffixes, punctuation, case, accents and "
    "transliteration differences. Answer with exactly one word: YES, NO, or UNSURE."
)


def _guard_check(repo_id: str, max_params_b: float, required_license: str) -> tuple[bool, str]:
    """Verify license + parameter count from the HF API before any download."""
    if GUARD_CACHE.exists():
        cached = json.loads(GUARD_CACHE.read_text(encoding="utf-8"))
        if cached.get("repo_id") == repo_id:
            return cached["ok"], cached["reason"]
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return False, f"HF API unreachable: {exc}"
    cards = meta.get("cardData", {}) or {}
    licenses = meta.get("tags", [])
    lic_ok = any(required_license in str(x).lower() for x in licenses) or (
        required_license in str(cards.get("license", "")).lower())
    cfg = meta.get("config", {}) or {}
    n_params = cfg.get("num_parameters")
    if n_params is None:
        # fall back to parsing the model name (e.g. '7b'); conservative
        m = PARAM_RE.search(repo_id)
        n_params = float(m.group(1)) if m else None
    size_b = (n_params / 1e9) if isinstance(n_params, (int, float)) and n_params > 1000 else (
        n_params if isinstance(n_params, (int, float)) else None)
    if not lic_ok:
        ok, reason = False, f"license not {required_license}: tags={licenses[-5:]}"
    elif size_b is None:
        ok, reason = False, "could not determine parameter count"
    elif size_b > max_params_b:
        ok, reason = False, f"{size_b}B params > {max_params_b}B limit"
    else:
        ok, reason = True, f"license ok, {size_b}B params <= {max_params_b}B"
    GUARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GUARD_CACHE.write_text(json.dumps({"repo_id": repo_id, "ok": ok, "reason": reason}), encoding="utf-8")
    return ok, reason


def _load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(cfg["llm_adjudicate"]["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["llm_adjudicate"]["model"], quantization_config=quant, device_map="auto",
    )
    model.eval()
    return model, tok


def _ask(model, tok, rec_a: str, rec_b: str) -> float:
    user = (
        f"Record A: {rec_a}\nRecord B: {rec_b}\n\n"
        "Same real-world business? Answer one word: YES, NO, or UNSURE."
    )
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    import torch

    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=int(cfg_max_new()), do_sample=False,
                             pad_token_id=tok.eos_token_id or 0)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    if text.startswith("YES"):
        return 1.0
    if text.startswith("NO"):
        return 0.0
    return 0.5


_CFG = {"max_new_tokens": 8}


def cfg_max_new() -> int:
    return _CFG["max_new_tokens"]


def run(cfg: dict, force: bool = False) -> dict:
    global _CFG
    _CFG["max_new_tokens"] = int(cfg["llm_adjudicate"].get("max_new_tokens", 8))
    art = Path(cfg["paths"]["artifacts_dir"])
    ok, reason = _guard_check(
        cfg["llm_adjudicate"]["model"],
        float(cfg["llm_adjudicate"]["expected_params_max_b"]),
        str(cfg["llm_adjudicate"]["required_license"]),
    )
    print(f"  guard: {reason}")
    if not ok:
        print("  S6b AUTO-SKIPPED (guard failed) — S7 falls back to reranker+LGBM signals")
        return {"status": "skipped_guard", "reason": reason}

    results = {}
    for split, scores_path in (("train", art / "features" / "val_scores.parquet"),
                               ("test", art / "features" / "test_scores.parquet")):
        out_path = art / "features" / f"{split}_llm.parquet"
        params = {"llm_version": 1, "band": [cfg["llm_adjudicate"]["band_low"], cfg["llm_adjudicate"]["band_high"]]}
        if not force and out_path.exists() and io_utils.manifest_ok(out_path, {"scores": scores_path}, params):
            results[split] = "fresh"
            continue
        if not scores_path.exists():
            print(f"  [{split}] scores missing — skipping")
            continue
        scored = pd.read_parquet(scores_path)
        band = scored[
            (scored["p_match"] >= cfg["llm_adjudicate"]["band_low"])
            & (scored["p_match"] <= cfg["llm_adjudicate"]["band_high"])
        ]
        cap = int(float(cfg["llm_adjudicate"]["max_pairs_fraction"]) * len(scored))
        if len(band) > cap:
            band = band.nlargest(cap, "p_match")  # most ambiguous get priority? use mid-band instead
        print(f"  [{split}] LLM band: {len(band):,} pairs (cap {cap:,})")
        if not len(band):
            pd.DataFrame({"s1_entity_id": pd.Series(dtype=object), "cand_id": pd.Series(dtype=object),
                          "llm_yes": pd.Series(dtype=np.float32)}).to_parquet(out_path, index=False)
            results[split] = "empty_band"
            continue
        try:
            model, tok = _load_model_and_tokenizer(cfg)
        except Exception as exc:
            print(f"  model load failed ({exc}) -> S6b skipped")
            return {"status": "skipped_load", "reason": str(exc)[:200]}

        nrm = art / "normalized"
        s1 = pd.read_parquet(nrm / f"{split}_s1.parquet", columns=["entity_id", "name_core", "addr_norm"]).set_index("entity_id")
        s2 = pd.read_parquet(nrm / f"{split}_s2.parquet", columns=["entity_id", "name_core", "addr_norm"])
        s3 = pd.read_parquet(nrm / f"{split}_s3.parquet", columns=["entity_id", "name_core", "addr_norm"])
        s23 = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")
        del s2, s3

        answers = []
        for i, (s1id, cid) in enumerate(zip(band["s1_entity_id"], band["cand_id"])):
            ta = f"{s1.loc[s1id, 'name_core']} | {s1.loc[s1id, 'addr_norm']}"
            tb = f"{s23.loc[cid, 'name_core']} | {s23.loc[cid, 'addr_norm']}"
            answers.append(_ask(model, tok, ta, tb))
            if (i + 1) % 200 == 0:
                print(f"  [{split}] adjudicated {i + 1:,}/{len(band):,}", flush=True)
        out = pd.DataFrame({
            "s1_entity_id": band["s1_entity_id"].to_numpy(),
            "cand_id": band["cand_id"].to_numpy(),
            "llm_yes": np.asarray(answers, dtype=np.float32),
        })
        out.to_parquet(out_path, index=False)
        io_utils.save_manifest(out_path, {"scores": scores_path}, params, extra={"pairs": len(out)})
        results[split] = f"ok:{len(out)}"
        del model, tok, s1, s23
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return {"status": "ok", "results": results}
