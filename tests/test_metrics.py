"""Regression tests for the challenge metric (entity-level macro F0.5).

The singleton rule and the README worked example are normative — if any of
these fail, threshold sweeps and model selection are invalid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.metrics import entity_f05  # noqa: E402


def test_readme_worked_example():
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    m = entity_f05(pred, truth)
    assert abs(m["macro_f05"] - 0.714) < 0.001, m


def test_singleton_correct_empty_is_one():
    m = entity_f05({"S1-1": set()}, {"S1-1": set()})
    assert m["macro_f05"] == 1.0, m


def test_singleton_false_match_is_zero():
    m = entity_f05({"S1-1": {"S2-9"}}, {"S1-1": set()})
    assert m["macro_f05"] == 0.0, m


def test_all_missed_is_zero():
    m = entity_f05({"S1-1": set()}, {"S1-1": {"S2-9"}})
    assert m["macro_f05"] == 0.0, m


def test_perfect_multi_match_is_one():
    m = entity_f05({"S1-1": {"S2-1", "S3-1"}}, {"S1-1": {"S2-1", "S3-1"}})
    assert m["macro_f05"] == 1.0, m


def test_missing_prediction_row_counts_as_empty():
    # entity absent from pred dict == predicted empty (true singleton -> 1.0)
    m = entity_f05({}, {"S1-1": set(), "S1-2": {"S2-1"}})
    assert abs(m["macro_f05"] - 0.5) < 1e-9, m


def test_macro_averaging_over_entities():
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    pred = {"S1-1": set(), "S1-2": {"S2-1"}}  # 1.0 + 1.0
    assert abs(entity_f05(pred, truth)["macro_f05"] - 1.0) < 1e-9
    pred_bad = {"S1-1": {"S2-5"}, "S1-2": {"S2-1"}}  # 0.0 + 1.0
    assert abs(entity_f05(pred_bad, truth)["macro_f05"] - 0.5) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL METRIC TESTS PASS")
