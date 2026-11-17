# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Explainable ML readiness. A checklist with weights, not a prediction of model quality."""
from __future__ import annotations

import pandas as pd

from .splitting import is_classification


def _chk(name, passed, weight, detail, fix=None):
    return {"name": name, "passed": passed, "weight": weight, "detail": detail, "fix": fix}


def readiness(df, target=None, split_done=False, train=None, val=None, test=None, leakage=None) -> dict:
    """Returns {"score": int|None, "checks": [...], "note": str}.
    When a split exists, data checks are measured on the TRAINING set."""
    base = train if (split_done and train is not None) else df
    where = "training set" if base is train and train is not None else "whole dataset"
    n_cells = max(base.size, 1)
    miss = int(base.isna().sum().sum())
    dup = int(base.duplicated().sum())
    const = [c for c in base.columns if base[c].nunique(dropna=True) <= 1]
    cats = [c for c in base.columns if c != target and not (pd.api.types.is_numeric_dtype(base[c])
            or pd.api.types.is_bool_dtype(base[c]) or pd.api.types.is_datetime64_any_dtype(base[c]))]
    num = base.select_dtypes("number")
    out_cells = 0
    for c in num.columns:
        s = num[c].dropna()
        if len(s) >= 4:
            q1, q3 = s.quantile([0.25, 0.75]); i = q3 - q1
            if i > 0:
                out_cells += int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
    out_rate = out_cells / max(num.size, 1) * 100

    checks = [
        _chk("Missing values", miss == 0, 3, f"{miss:,} missing cell(s) ({miss / n_cells * 100:.2f}%) in the {where}.", "Run Clean."),
        _chk("Duplicate rows", dup == 0, 2, f"{dup:,} duplicate row(s) in the {where}.", "Run Clean."),
        _chk("Constant columns", not const, 1, f"{len(const)} constant column(s)." + (f" ({', '.join(map(str, const[:3]))})" if const else ""), "Run Clean."),
        _chk("Outliers", out_rate <= 1, 1, f"{out_rate:.2f}% of numeric values lie outside the IQR bounds (threshold 1%).", "Run Transform."),
        _chk("Text columns encoded", not cats, 3, f"{len(cats)} text column(s) still need encoding.", "Run Encode."),
        _chk("Train/test split", bool(split_done), 3, "Split done." if split_done else "No split yet.", "Run Split first."),
    ]
    if target is not None and target in df.columns:
        checks.append(
            _chk(
                "Target has no missing values",
                not df[target].isna().any(),
                3,
                f"{int(df[target].isna().sum())} missing target value(s).",
                "Run Split again to exclude rows with a missing target.",
            )
        )
        s = df[target].dropna()
        if len(s) and is_classification(s):
            share = float(s.value_counts(normalize=True).min())
            checks.append(_chk("Class balance", share >= 0.2, 1, f"Smallest class is {share * 100:.1f}% of the data (threshold 20%).",
                               "Consider balancing the training set."))
    if leakage is not None:
        errs = [x for x in leakage if x["level"] == "error"]
        warns = [x for x in leakage if x["level"] == "warning"]
        checks.append(_chk("Leakage guard", not errs, 4, "No certain leakage found." if not errs else errs[0]["message"], "Remove the overlapping rows."))
        if warns:
            checks.append(_chk("Suspicious features", False, 1, warns[0]["message"], "Review these columns manually."))
    total = sum(c["weight"] for c in checks)
    score = round(100 * sum(c["weight"] for c in checks if c["passed"]) / total) if total else None
    return {"score": score, "checks": checks,
            "note": "A weighted checklist of data-preparation problems. It does not predict model accuracy."}
