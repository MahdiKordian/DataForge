# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
from __future__ import annotations

import numpy as np
import pandas as pd


def is_classification(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 20


def split_frame(df: pd.DataFrame, target, test_pct=20, val_pct=0, seed=42, stratify=True):
    """Random (optionally stratified) split. Returns (train, val_or_None, test, notes).
    Rows with a missing target are removed first. Raises ValueError on an unusable split."""
    notes = []
    if target is not None and target in df.columns and df[target].isna().any():
        k = int(df[target].isna().sum())
        df = df[df[target].notna()]
        notes.append(f"removed {k} row(s) with a missing target before splitting")
    clf = target is not None and target in df.columns and is_classification(df[target].dropna())
    rng = np.random.default_rng(int(seed))
    codes = pd.factorize(df[target])[0] if (stratify and clf) else np.zeros(len(df), int)
    tr, va, te = [], [], []
    for k in np.unique(codes):
        g = rng.permutation(np.flatnonzero(codes == k))
        nt, nv = int(round(len(g) * test_pct / 100)), int(round(len(g) * val_pct / 100))
        te += list(g[:nt]); va += list(g[nt:nt + nv]); tr += list(g[nt + nv:])
    if not tr or not te:
        raise ValueError("The split left an empty train or test set. Use smaller percentages or more rows.")
    train, test = df.iloc[rng.permutation(tr)], df.iloc[rng.permutation(te)]
    val = df.iloc[rng.permutation(va)] if va else None
    reset = lambda d: None if d is None else d.reset_index(drop=True)
    notes.append(f"train {len(train):,} / " + (f"validation {len(val):,} / " if val is not None else "") + f"test {len(test):,}")
    return reset(train), reset(val), reset(test), notes


def balance_train(train: pd.DataFrame, target, how: str, seed=42):
    """Oversample or undersample the TRAINING set only (never validation or test)."""
    if how == "None" or target not in train.columns:
        return train
    cnt = train[target].value_counts()
    n = cnt.max() if how.startswith("Over") else cnt.min()
    out = pd.concat([g.sample(n, replace=len(g) < n, random_state=int(seed)) for _, g in train.groupby(target)])
    return out.sample(frac=1, random_state=int(seed)).reset_index(drop=True)
