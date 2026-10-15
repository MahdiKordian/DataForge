# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Leakage Guard. Separates certain problems ("error") from suspicious ones ("warning")."""
from __future__ import annotations

import pandas as pd


def _hashes(df: pd.DataFrame) -> set:
    return set(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).tolist())


def row_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Number of rows in b that also appear (identical in every column) in a."""
    common = [c for c in a.columns if c in b.columns]
    if not common or a.empty or b.empty:
        return 0
    ha = _hashes(a[common])
    return int(pd.util.hash_pandas_object(b[common].reset_index(drop=True), index=False).isin(ha).sum())


def suspicious_features(train: pd.DataFrame, target, thr: float = 0.98) -> list[tuple[str, float]]:
    if target not in train.columns or not pd.api.types.is_numeric_dtype(train[target]):
        return []
    out = []
    for c in train.select_dtypes("number").columns:
        if c == target or train[c].nunique() < 2:
            continue
        r = train[c].corr(train[target])
        if pd.notna(r) and abs(r) >= thr:
            out.append((c, float(r)))
    return out


def leakage_report(train, val, test, target, processed_train=None) -> list[dict]:
    """train / val / test are the sets as they were split (before any preparation).
    processed_train is the prepared training set, used to look for features that copy the target."""
    items = []
    for name, other in (("validation", val), ("test", test)):
        if other is not None and train is not None:
            k = row_overlap(train, other)
            if k:
                items.append({"level": "error", "message": f"{k} {name} row(s) are identical to training rows. "
                              "The model would be tested on data it has already seen."})
    probe = processed_train if processed_train is not None else train
    if probe is not None and target:
        for c, r in suspicious_features(probe, target):
            items.append({"level": "warning", "message": f"`{c}` is almost perfectly related to the target "
                          f"(correlation {r:+.3f}). Check that it is not computed from the target or known only after the outcome."})
    return items
