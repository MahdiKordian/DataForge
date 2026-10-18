# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataforge import SchemaError, balance_train, build_pipeline, leakage_report, readiness, split_frame
from dataforge.transformers import (CategoricalEncoder, FeatureScaler, InfToNan, MissingValueImputer,
                                    NotFittedError, OutlierHandler)

CFG = {
    "Clean": dict(trim=True, case="Keep as is", to_num=True, dups=True, const=True, max_missing=60,
                  drop_missing_target=True, num_missing="Median", cat_missing="Most common"),
    "Transform": dict(outliers="Clip at 1st and 99th percentile", out_cols=None, skew=True, skew_thr=1.0,
                      skew_method="Log (log1p)", drop=[], corr=0.95),
    "Encode": dict(low="One-hot", max_onehot=10, drop_first=True, high="Frequency", dates=True, encode_target=True),
    "Scale": dict(method="Standard (z-score)", cols=None),
}


def toy(n=60, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"age": r.integers(18, 70, n).astype(float), "income": r.lognormal(10, 1, n),
                         "city": r.choice(["a", "b", "c"], n), "y": r.choice(["yes", "no"], n)})


# ---- parameters come from the training data only ----------------------
def test_scaler_uses_train_statistics_only():
    train = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6]})
    test = pd.DataFrame({"x": [1000.0, 2000, 3000]})
    s = FeatureScaler(method="Standard (z-score)").fit(train)
    assert s.params_["x"]["center"] == pytest.approx(3.5)          # not affected by test
    assert s.transform(test)["x"].iloc[0] == pytest.approx((1000 - 3.5) / train["x"].std())


def test_imputer_fills_with_train_median():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan, 100.0, 200.0]})
    out = MissingValueImputer(num_missing="Median").fit(train).transform(test)
    assert out["x"].iloc[0] == 2.0


def test_outlier_bounds_from_train():
    train = pd.DataFrame({"x": list(range(100))})
    test = pd.DataFrame({"x": [-500, 50, 900]})
    h = OutlierHandler(outliers="Clip to IQR bounds").fit(train)
    lo, hi = h.bounds_["x"]
    assert h.transform(test)["x"].tolist() == [lo, 50, hi]


# ---- the input is never modified ---------------------------------------
def test_transform_does_not_mutate_input():
    df = toy()
    before = df.copy()
    pipe = build_pipeline(CFG, "y")
    pipe.fit_transform(df.copy())
    pipe.transform(df)
    pd.testing.assert_frame_equal(df, before)


# ---- unseen categories / schema ------------------------------------------
def test_unseen_category_is_handled():
    train = pd.DataFrame({"c": ["a", "b", "a", "b"], "y": [0, 1, 0, 1]})
    new = pd.DataFrame({"c": ["a", "zzz"], "y": [0, 1]})
    enc = CategoricalEncoder(low="One-hot", max_onehot=10, drop_first=False, encode_target=False).fit(train, "y")
    out = enc.transform(new)
    assert out.loc[1, ["c_a", "c_b"]].tolist() == [0, 0]
    assert enc.warnings_


def test_missing_column_gives_readable_error():
    df = toy()
    pipe = build_pipeline(CFG, "y")
    pipe.fit_transform(df)
    with pytest.raises(SchemaError, match="income"):
        pipe.transform(df.drop(columns="income"))


def test_output_columns_match_train():
    df = toy(); tr, va, te, _ = split_frame(df, "y", 25, 0, 1)
    pipe = build_pipeline(CFG, "y")
    a = pipe.fit_transform(tr)
    assert list(pipe.transform(te).columns) == list(a.columns)


def test_not_fitted_raises():
    with pytest.raises(NotFittedError):
        FeatureScaler().transform(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(NotFittedError):
        build_pipeline(CFG, "y").transform(toy())


# ---- edge cases -----------------------------------------------------------
def test_infinite_values_become_missing():
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 4.0]})
    assert InfToNan().fit_transform(df)["x"].isna().sum() == 2


def test_empty_column_and_tiny_data_do_not_crash():
    df = pd.DataFrame({"a": [np.nan] * 5, "b": [1.0, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0]})
    out = build_pipeline(CFG, "y").fit_transform(df)
    assert "a" not in out.columns
    tiny = pd.DataFrame({"b": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    build_pipeline(CFG, "y").fit_transform(tiny)


def test_pipeline_is_reproducible():
    df = toy(); tr, _, te, _ = split_frame(df, "y", 25, 0, 7)
    p1, p2 = build_pipeline(CFG, "y"), build_pipeline(CFG, "y")
    pd.testing.assert_frame_equal(p1.fit_transform(tr), p2.fit_transform(tr))
    pd.testing.assert_frame_equal(p1.transform(te), p2.transform(te))


# ---- splitting + leakage -----------------------------------------------------
def test_split_is_disjoint_reproducible_and_stratified():
    df = toy(200)
    df["row"] = range(len(df))
    a = split_frame(df, "y", 20, 10, 3); b = split_frame(df, "y", 20, 10, 3)
    tr, va, te, _ = a
    assert not (set(tr.row) & set(te.row)) and not (set(tr.row) & set(va.row)) and not (set(va.row) & set(te.row))
    assert len(tr) + len(va) + len(te) == len(df)
    pd.testing.assert_frame_equal(tr, b[0])
    assert abs(tr.y.eq("yes").mean() - te.y.eq("yes").mean()) < 0.08


def test_split_rejects_empty_sets():
    with pytest.raises(ValueError):
        split_frame(toy(4), "y", 5, 0, 1)


def test_balance_only_changes_train():
    tr = pd.DataFrame({"x": range(10), "y": [0] * 8 + [1] * 2})
    assert balance_train(tr, "y", "Oversample minority").y.value_counts().nunique() == 1


def test_leakage_guard_finds_shared_rows_and_target_copy():
    raw = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    test = pd.DataFrame({"x": [3, 9], "y": [0, 1]})
    rep = leakage_report(raw, None, test, "y")
    assert any(r["level"] == "error" for r in rep)
    leaky = pd.DataFrame({"x": np.arange(20.0), "copy": np.arange(20.0) * 2, "y": np.arange(20.0)})
    assert any(r["level"] == "warning" for r in leakage_report(leaky, None, None, "y"))


def test_readiness_is_explainable():
    r = readiness(toy(), "y", split_done=False)
    assert 0 <= r["score"] <= 100
    assert all({"name", "passed", "weight", "detail"} <= set(c) for c in r["checks"])
    assert next(c for c in r["checks"] if c["name"] == "Train/test split")["passed"] is False


# ---- real sample datasets ------------------------------------------------------------
@pytest.mark.parametrize("file,target", [("Telco_Customer_Churn.csv", "Churn"), ("Palmer_Penguins.csv", "species"),
                                         ("Ames_Housing.csv", "SalePrice")])
def test_sample_datasets_end_to_end(file, target):
    path = Path(__file__).resolve().parents[1] / "sample_data" / file
    if not path.exists():
        pytest.skip(f"sample_data/{file} is not in this checkout")
    df = pd.read_csv(path)
    tr, va, te, _ = split_frame(df, target, 20, 10, 42)
    pipe = build_pipeline(CFG, target)
    a = pipe.fit_transform(tr); b = pipe.transform(va); c = pipe.transform(te)
    assert list(a.columns) == list(b.columns) == list(c.columns)
    assert not a.select_dtypes(exclude="number").shape[1]
    assert not np.isinf(a.select_dtypes("number")).any().any()
