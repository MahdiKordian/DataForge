# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Regression tests for problems found in the phase 4 review."""

import numpy as np
import pandas as pd
import pytest

from dataforge import build_pipeline, split_frame
from dataforge import security as SE
from dataforge.scale import optimize_dtypes
from dataforge.transformers import (
    FeatureScaler,
    MissingValueImputer,
    OutlierHandler,
    TextCleaner,
    TypeCaster,
)


def test_duplicates_are_removed_from_training_only():
    train = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    t = TextCleaner(dups=True).fit(train)
    assert len(t.transform(train, training=True)) == 2
    assert len(t.transform(train)) == 3


def test_new_data_keeps_its_row_count_through_the_pipeline():
    r = np.random.default_rng(0)
    df = pd.DataFrame({"x": r.normal(size=80), "y": r.choice(["a", "b"], 80)})
    cfg = {
        "Clean": dict(
            trim=True,
            case="Keep as is",
            to_num=True,
            dups=True,
            const=True,
            max_missing=60,
            drop_missing_target=True,
            num_missing="Median",
            cat_missing="Most common",
        )
    }
    pipe = build_pipeline(cfg, "y")
    pipe.fit_transform(df)
    new = pd.concat([df.head(5), df.head(5)], ignore_index=True)
    assert len(pipe.transform(new)) == 10


def test_empty_column_selection_means_none_not_all():
    df = pd.DataFrame({"x": list(range(100)) + [10_000], "z": list(range(101))})
    assert (
        OutlierHandler(outliers="Clip to IQR bounds", out_cols=[]).fit(df).bounds_ == {}
    )
    assert set(
        OutlierHandler(outliers="Clip to IQR bounds", out_cols=None).fit(df).bounds_
    ) == {"x", "z"}
    assert FeatureScaler(method="Standard (z-score)", cols=[]).fit(df).params_ == {}
    assert set(
        FeatureScaler(method="Standard (z-score)", cols=None).fit(df).params_
    ) == {"x", "z"}


def test_outlier_clip_preserves_int64_when_no_values_are_outliers():
    df = pd.DataFrame({"x": range(1, 11)}, dtype="int64")

    handler = OutlierHandler(outliers="Clip to IQR bounds").fit(df)
    result = handler.transform(df)

    assert result["x"].dtype == "int64"
    pd.testing.assert_series_equal(result["x"], df["x"])


def test_outlier_clip_preserves_large_integer_precision():
    base = 2**53

    handler = OutlierHandler(outliers="Clip to IQR bounds")
    handler.fit(pd.DataFrame({"x": [1, 2, 3]}))

    handler.bounds_ = {"x": (base, base + 10)}

    df = pd.DataFrame(
        {"x": [base + 1, base + 20]},
        dtype="int64",
    )

    result = handler.transform(df)

    assert result["x"].dtype == "int64"
    assert result["x"].tolist() == [base + 1, base + 10]


def test_per_column_median_on_a_text_column_does_not_crash():
    df = pd.DataFrame({"c": ["a", "a", None, "b"], "n": [1.0, 2.0, np.nan, 4.0]})
    imp = MissingValueImputer(per_column={"c": "Median"}).fit(df)
    assert imp.fill_["c"] == "a"
    assert imp.transform(df)["c"].isna().sum() == 0


def test_type_caster_note_counts_only_numeric_columns_treated_as_categories():
    df = pd.DataFrame(
        {
            "zip": [10115, 10117, 10119, 10115],
            "city": ["a", "b", "a", "b"],
            "y": [0, 1, 0, 1],
        }
    )
    tc = TypeCaster(types={"zip": "categorical", "city": "categorical"}).fit(df, "y")
    assert tc.notes_ == ["1 numeric column(s) treated as categories"]


def test_dates_written_as_text_are_not_phone_numbers():
    df = pd.DataFrame(
        {
            "when": ["2024-01-15", "2023-12-31", "2022-07-04"] * 10,
            "tel": ["+49 30 1234567"] * 30,
        }
    )
    found = SE.pii_scan(df).groupby("column")["looks like"].apply(list).to_dict()
    assert "when" not in found and "phone number" in found["tel"]


@pytest.mark.parametrize(
    "ip,public",
    [
        ("93.184.216.34", True),
        ("10.0.0.1", False),
        ("127.0.0.1", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("::1", False),
        ("::ffff:10.0.0.1", False),
        ("fe80::1%eth0", False),
    ],
)
def test_public_address_test(ip, public):
    assert SE._is_public(ip) is public


def test_optimize_dtypes_without_categories_leaves_text_alone():
    df = pd.DataFrame({"t": ["a", "b", "a", "b"] * 5, "n": np.arange(20)})
    out, _ = optimize_dtypes(df, categories=False)
    assert str(out["t"].dtype) == str(df["t"].dtype)
    assert str(out["n"].dtype) != "int64"  # still downcast


def test_split_never_returns_overlapping_rows():
    df = pd.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    tr, va, te, _ = split_frame(df, "y", 20, 10, 3)
    assert (
        not (set(tr.x) & set(te.x))
        and not (set(tr.x) & set(va.x))
        and not (set(va.x) & set(te.x))
    )
    assert len(tr) + len(va) + len(te) == 100


def test_missing_value_drop_rows_only_removes_training_rows():
    df = pd.DataFrame(
        {
            "number": [1.0, np.nan, 3.0],
            "category": ["a", "b", None],
            "target": [0, 1, 0],
        }
    )

    imputer = MissingValueImputer(
        num_missing="Drop rows",
        cat_missing="Drop rows",
    ).fit(df, target="target")

    train = imputer.transform(df, training=True)
    validation = imputer.transform(df, training=False)

    assert len(train) == 1
    assert len(validation) == 3

    assert validation["number"].isna().sum() == 1
    assert validation["category"].isna().sum() == 1


def test_missing_target_rows_are_removed_only_from_training():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, np.nan, 1],
        }
    )

    imputer = MissingValueImputer(
        drop_missing_target=True,
    ).fit(df, target="target")

    train = imputer.transform(df, training=True)
    validation = imputer.transform(df, training=False)

    assert len(train) == 2
    assert len(validation) == 3
