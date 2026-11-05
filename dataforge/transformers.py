# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Transformers with a fit / transform contract.

fit(train)       learns every parameter from the TRAINING data only.
transform(df)    applies those stored parameters to any data and never modifies df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NotFittedError(RuntimeError):
    pass


def is_text(s: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(s)
        or pd.api.types.is_bool_dtype(s)
        or pd.api.types.is_datetime64_any_dtype(s)
    )


def numeric_cols(df: pd.DataFrame) -> list:
    return list(df.select_dtypes("number").columns)


def continuous(df: pd.DataFrame, target) -> list:
    return [c for c in numeric_cols(df) if c != target and df[c].nunique() > 2]


def _sorted_unique(s: pd.Series) -> list:
    return sorted(s.dropna().unique(), key=str)


class Transformer:
    name = "Transformer"

    def __init__(self, **params):
        self.params = params
        self.fitted_ = False
        self.notes_: list[str] = []
        self.warnings_: list[str] = []
        self.target_ = None

    def fit(self, df: pd.DataFrame, target=None):
        self.target_ = target if target in df.columns else None
        self.notes_, self.warnings_ = [], []
        self._fit(df.copy())
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        if not self.fitted_:
            raise NotFittedError(f"{self.name} must be fitted before transform().")
        self.warnings_ = []
        return self._transform(df.copy(), training)

    def fit_transform(self, df: pd.DataFrame, target=None) -> pd.DataFrame:
        return self.fit(df, target).transform(df, training=True)

    def _fit(self, df): ...

    def _transform(self, df, training):
        return df


class InfToNan(Transformer):
    name = "Infinite values"

    def _fit(self, df):
        n = int(np.isinf(df.select_dtypes("number")).sum().sum())
        if n:
            self.notes_.append(f"treated {n} infinite value(s) as missing")

    def _transform(self, df, training):
        for c in df.select_dtypes("category").columns:
            df[c] = df[c].astype(object)
        num = df.select_dtypes("number").columns
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
        return df


class TextCleaner(Transformer):
    """trim / case / numbers stored as text / duplicate rows."""

    name = "Text cleanup"

    def _clean(self, s: pd.Series) -> pd.Series:
        p = self.params
        s = s.astype("string")
        if p.get("trim", True):
            s = s.str.strip()
        case = p.get("case", "Keep as is")
        if case == "lowercase":
            s = s.str.lower()
        elif case == "UPPERCASE":
            s = s.str.upper()
        elif case == "Title Case":
            s = s.str.title()
        s = s.mask(s.fillna("x") == "", pd.NA)
        return s.astype(object).where(s.notna(), np.nan)

    def _fit(self, df):
        self.text_cols_ = [c for c in df.columns if is_text(df[c])]
        self.numeric_cols_ = []
        if self.params.get("to_num", True):
            for c in [
                c
                for c in self.text_cols_
                if c not in set(self.params.get("protect") or [])
            ]:
                s = self._clean(df[c])
                num = pd.to_numeric(s, errors="coerce")
                if s.notna().any() and num.notna().sum() == s.notna().sum():
                    self.numeric_cols_.append(c)
        if self.numeric_cols_:
            self.notes_.append(
                f"{len(self.numeric_cols_)} text column(s) will be read as numbers"
            )

    def _transform(self, df, training):
        for c in self.text_cols_:
            if c in df.columns:
                df[c] = self._clean(df[c])
        for c in self.numeric_cols_:
            if c in df.columns:
                before = df[c].notna().sum()
                df[c] = pd.to_numeric(df[c], errors="coerce")
                lost = int(before - df[c].notna().sum())
                if lost:
                    self.warnings_.append(
                        f"`{c}`: {lost} value(s) were not numeric and became missing"
                    )
        if training and self.params.get("dups", True) and df.duplicated().any():
            k = int(df.duplicated().sum())
            df = df.drop_duplicates()
            self.notes_.append(f"removed {k} duplicate row(s)")
        return df


class ColumnFilter(Transformer):
    """Drops mostly-empty and constant columns, judged on the training data."""

    name = "Column filter"

    def _fit(self, df):
        t, p = self.target_, self.params
        drop = []
        if p.get("max_missing", 60) < 100:
            drop += [
                c
                for c in df.columns
                if c != t and df[c].isna().mean() * 100 > p["max_missing"]
            ]
        if p.get("const", True):
            drop += [
                c
                for c in df.columns
                if c != t and c not in drop and df[c].nunique(dropna=True) <= 1
            ]
        self.drop_ = drop
        if drop:
            self.notes_.append(
                f"dropped {len(drop)} column(s): {', '.join(map(str, drop[:5]))}"
            )

    def _transform(self, df, training):
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


class MissingValueImputer(Transformer):
    """Fill values are computed on the training data and reused everywhere."""

    name = "Missing values"

    def _fit(self, df):
        t, p = self.target_, self.params
        self.fill_, self.drop_rows_cols_ = {}, []
        for c in df.columns:
            if c == t or pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            s = df[c]
            default = (
                p.get("num_missing", "Median")
                if pd.api.types.is_numeric_dtype(s)
                else p.get("cat_missing", "Most common")
            )
            strat = (p.get("per_column") or {}).get(c) or default
            v = None
            if strat == "Drop rows":
                self.drop_rows_cols_.append(c)
            elif strat in ("Median", "Mean"):
                if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(
                    s
                ):
                    strat = "Most common"
                    v = s.mode().iloc[0] if s.notna().any() else None
                else:
                    v = s.median() if strat == "Median" else s.mean()
            elif strat == "Most common":
                v = s.mode().iloc[0] if s.notna().any() else None
            elif strat == "Fill with 0":
                v = 0
            elif strat.startswith("Fill with 'U"):
                v = "Unknown"
            if v is not None and not pd.isna(v):
                self.fill_[c] = v
        n = int(df[list(self.fill_)].isna().sum().sum()) if self.fill_ else 0
        if n:
            self.notes_.append(
                f"filled {n} missing value(s) using training-set statistics"
            )

    def _transform(self, df, training):
        t = self.target_

        if training:
            if t and t in df.columns and self.params.get("drop_missing_target", True):
                df = df[df[t].notna()].copy()

            cols = [c for c in self.drop_rows_cols_ if c in df.columns]

            if cols:
                df = df.dropna(subset=cols)

        for c, v in self.fill_.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)

        return df


class OutlierHandler(Transformer):
    """Bounds come from the training data. 'Remove rows' only removes TRAINING rows."""

    name = "Outliers"

    def _fit(self, df):
        m = self.params.get("outliers", "Leave as is")
        self.bounds_ = {}
        if m == "Leave as is":
            return
        chosen = self.params.get("out_cols")
        cols = [
            c
            for c in (continuous(df, self.target_) if chosen is None else chosen)
            if c in df.columns
        ]
        for c in cols:
            s = df[c].dropna()
            if s.empty:
                continue
            if m.startswith("Clip at 1st"):
                lo, hi = s.quantile([0.01, 0.99])
            else:
                q1, q3 = s.quantile([0.25, 0.75])
                if q3 == q1:
                    continue
                lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            self.bounds_[c] = (float(lo), float(hi))

    def _transform(self, df, training):
        m = self.params.get("outliers", "Leave as is")
        if m == "Leave as is" or not self.bounds_:
            return df
        mask, clipped = pd.Series(False, index=df.index), 0
        for c, (lo, hi) in self.bounds_.items():
            if c not in df.columns:
                continue
            out = (df[c] < lo) | (df[c] > hi)
            if m.startswith("Remove"):
                mask |= out
            else:
                clipped += int(out.sum())

                if out.any():
                    values = pd.to_numeric(df[c], errors="raise")
                    if pd.api.types.is_integer_dtype(values.dtype) and (
                        float(lo).is_integer() and float(hi).is_integer()
                    ):
                        df[c] = values.clip(lower=int(lo), upper=int(hi))
                    else:
                        df[c] = values.astype("float64").clip(
                            lower=lo,
                            upper=hi,
                        )
        if m.startswith("Remove"):
            if training and mask.any():
                self.notes_.append(
                    f"removed {int(mask.sum())} training row(s) with outliers"
                )
                df = df[~mask]
        elif clipped and training:
            self.notes_.append(f"clipped {clipped} outlier value(s)")
        return df


class SkewTransformer(Transformer):
    name = "Skew"

    def _fit(self, df):
        thr = self.params.get("skew_thr", 1.0)
        self.shift_ = {}
        for c in continuous(df, self.target_):
            s = df[c].dropna()
            if len(s) > 2 and s.skew() > thr:
                self.shift_[c] = float(max(0.0, -s.min()))
        if self.shift_:
            self.notes_.append(f"reduced right skew in {len(self.shift_)} column(s): {', '.join(map(str, list(self.shift_)[:5]))}")

    def _transform(self, df, training):
        use_log = self.params.get("skew_method", "Log (log1p)").startswith("Log")
        for c, shift in self.shift_.items():
            if c in df.columns:
                x = (df[c] + shift).clip(lower=0)  # unseen values below the training minimum are clamped
                df[c] = np.log1p(x) if use_log else np.sqrt(x)
        return df


class CorrelationFilter(Transformer):
    name = "Near-duplicate columns"

    def _fit(self, df):
        thr = self.params.get("corr")
        self.drop_ = []
        n = [c for c in numeric_cols(df) if c != self.target_]
        if thr and len(n) > 1:
            cm = df[n].corr().abs()
            up = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
            self.drop_ = [c for c in up.columns if (up[c] > thr).any()]
        if self.drop_:
            self.notes_.append(f"dropped {len(self.drop_)} near-duplicate column(s): {', '.join(map(str, self.drop_[:5]))}")

    def _transform(self, df, training):
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


class ColumnDropper(Transformer):
    name = "Dropped columns"

    def _fit(self, df):
        self.drop_ = [c for c in self.params.get("drop", []) if c in df.columns and c != self.target_]
        if self.drop_:
            self.notes_.append(f"dropped {len(self.drop_)} column(s) you chose")

    def _transform(self, df, training):
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


class CategoricalEncoder(Transformer):
    """Categories and frequencies are learned on the training data.
    Categories never seen in training become all-zero (one-hot), -1 (ordinal) or 0 (frequency)."""
    name = "Encoding"

    def _fit(self, df):
        p, t = self.params, self.target_
        self.bool_cols_ = [c for c in df.columns if c != t and pd.api.types.is_bool_dtype(df[c])]
        self.date_cols_ = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])] if p.get("dates", True) else []
        work = self._dates(df.copy())
        cat = [c for c in work.columns if c != t and is_text(work[c])]
        lim = p.get("max_onehot", 10)
        self.low_ = [c for c in cat if work[c].nunique() <= lim]
        self.high_ = [c for c in cat if c not in self.low_]
        self.cats_ = {c: _sorted_unique(work[c]) for c in self.low_}
        self.ord_ = {c: {v: i for i, v in enumerate(_sorted_unique(work[c]))}
                     for c in (self.low_ if p.get("low") == "Ordinal" else []) + (self.high_ if p.get("high") == "Ordinal" else [])}
        self.freq_ = {c: work[c].value_counts(normalize=True).to_dict() for c in self.high_} if p.get("high") == "Frequency" else {}
        self.target_map_ = None
        if t and p.get("encode_target", True) and is_text(work[t]):
            self.target_map_ = {v: i for i, v in enumerate(_sorted_unique(work[t]))}
            self.notes_.append(f"target `{t}` becomes {len(self.target_map_)} numeric classes")
        if self.low_:
            self.notes_.append(f"{len(self.low_)} column(s) {p.get('low', 'One-hot').lower()} encoded")
        if self.high_:
            self.notes_.append(f"{len(self.high_)} high-cardinality column(s): {str(p.get('high', 'Frequency')).lower()}")

    def _dates(self, df):
        for c in self.date_cols_:
            if c in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                for part in ("year", "month", "day", "weekday"):
                    df[f"{c}_{part}"] = getattr(df[c].dt, "dayofweek" if part == "weekday" else part)
                df = df.drop(columns=c)
        return df

    def _ordinal(self, s, mapping):
        out = s.map(mapping)
        return out.where(s.isna() | out.notna(), -1)

    def _transform(self, df, training):
        p, t = self.params, self.target_
        for c in self.bool_cols_:
            if c in df.columns:
                df[c] = df[c].astype(int)
        df = self._dates(df)
        for c in self.low_:
            if c not in df.columns:
                continue
            seen = set(self.cats_[c])
            unseen = int((~df[c].isin(seen) & df[c].notna()).sum())
            if unseen:
                self.warnings_.append(f"`{c}`: {unseen} value(s) with a category not seen in training")
            if p.get("low", "One-hot") == "One-hot":
                keep = self.cats_[c][1:] if p.get("drop_first", True) else self.cats_[c]
                new = {f"{c}_{v}": (df[c] == v).astype(int) for v in keep}
                df = pd.concat([df.drop(columns=c), pd.DataFrame(new, index=df.index)], axis=1)
            else:
                df[c] = self._ordinal(df[c], self.ord_[c])
        for c in self.high_:
            if c not in df.columns:
                continue
            h = p.get("high", "Frequency")
            if h == "Frequency":
                m = df[c].map(self.freq_[c])
                df[c] = m.where(df[c].isna() | m.notna(), 0.0)
            elif h == "Ordinal":
                df[c] = self._ordinal(df[c], self.ord_[c])
            else:
                df = df.drop(columns=c)
        if self.target_map_ is not None and t in df.columns:
            df[t] = df[t].map(self.target_map_)
        return df


class FeatureScaler(Transformer):
    name = "Scaling"

    def _fit(self, df):
        m = self.params.get("method", "Standard (z-score)")
        self.params_ = {}
        if m.startswith("Do not"):
            return
        chosen = self.params.get("cols")
        cols = [c for c in (continuous(df, self.target_) if chosen is None else chosen) if c in df.columns]
        for c in cols:
            s = df[c].astype(float)
            if m.startswith("Standard"):
                a, b = s.mean(), s.std()
            elif m.startswith("Min"):
                a, b = s.min(), s.max() - s.min()
            else:
                a, b = s.median(), s.quantile(0.75) - s.quantile(0.25)
            if b and not pd.isna(b):
                self.params_[c] = {"center": float(a), "scale": float(b)}
        if self.params_:
            self.notes_.append(f"scaled {len(self.params_)} column(s) with {m.split(' (')[0].lower()} scaling (training statistics)")

    def _transform(self, df, training):
        for c, q in self.params_.items():
            if c in df.columns:
                df[c] = (df[c].astype(float) - q["center"]) / q["scale"]
        return df


def _label(s: pd.Series) -> pd.Series:
    """Category labels as text. Whole numbers lose a trailing .0 (so 1.0 and 1 are the same label)."""
    def f(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return s.map(f)


class TypeCaster(Transformer):
    """Applies the variable types the user confirmed (continuous, categorical, ordinal, id, datetime, ...)."""
    name = "Variable types"

    def _fit(self, df):
        given = self.params.get("types") or {}
        self.types_ = {c: v for c, v in given.items() if c in df.columns and c != self.target_ and v not in ("target",)}
        orders = self.params.get("orders") or {}
        self.ranks_ = {}
        for c, v in self.types_.items():
            if v == "ordinal":
                order = orders.get(c) or [str(x) for x in _sorted_unique(_label(df[c]))]
                self.ranks_[c] = {str(o).strip().lower(): i for i, o in enumerate(order)}
        gone = [c for c, v in self.types_.items() if v in ("id", "text")]
        if gone:
            self.notes_.append(f"left out {len(gone)} identifier/text column(s): {', '.join(map(str, gone[:5]))}")
        k = sum(v in ("categorical", "ordinal") and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                for c, v in self.types_.items())
        if k:
            self.notes_.append(f"{k} numeric column(s) treated as categories")

    def _transform(self, df, training):
        for c, v in self.types_.items():
            if c not in df.columns:
                continue
            if v in ("id", "text"):
                df = df.drop(columns=c)
            elif v in ("continuous", "discrete"):
                if not pd.api.types.is_numeric_dtype(df[c]):
                    before = df[c].notna().sum()
                    df[c] = pd.to_numeric(df[c].astype("string").str.strip().replace("", pd.NA), errors="coerce")
                    lost = int(before - df[c].notna().sum())
                    if lost:
                        self.warnings_.append(f"`{c}`: {lost} value(s) were not numbers and became missing")
            elif v == "categorical":
                df[c] = _label(df[c])
            elif v == "ordinal":
                lab = _label(df[c]).astype("string").str.strip().str.lower()
                rk = lab.map(self.ranks_[c]).astype(float)
                unseen = int((lab.notna() & rk.isna()).sum())
                if unseen:
                    self.warnings_.append(f"`{c}`: {unseen} value(s) are not in the order you gave and became missing")
                df[c] = rk
            elif v == "datetime":
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
                except (TypeError, ValueError):
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
