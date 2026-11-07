# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Variable type inference. Every guess carries a confidence and a reason, and the user can override it."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TYPES = [
    "continuous",
    "discrete",
    "categorical",
    "ordinal",
    "id",
    "datetime",
    "text",
    "target",
]
WHAT_HAPPENS = {
    "continuous": "kept as a number (scaled, outliers handled)",
    "discrete": "kept as a number (counts, small whole numbers)",
    "categorical": "encoded as a category, never scaled",
    "ordinal": "turned into ranks using the order you give",
    "id": "left out of the prepared data",
    "datetime": "split into year, month, day and weekday",
    "text": "left out (free text needs its own methods)",
    "target": "the value to predict; never modified except label encoding",
}
SCALES = [
    ["very low", "low", "medium", "high", "very high"],
    ["low", "medium", "high"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["small", "medium", "large"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["none", "mild", "moderate", "severe"],
    ["bronze", "silver", "gold", "platinum"],
]
_ID_NAME = re.compile(r"(?:^|[_\s-])id$|(?<=[a-z])I[dD]$|^id$|uuid|guid", re.I)
_CODE_NAME = re.compile(
    r"zip|postal|postcode|phone|pincode|\bcode\b|_code$|areacode", re.I
)


def _id_name(c) -> bool:
    c = str(c)
    return bool(_ID_NAME.search(c)) and not re.search(
        r"(?:paid|valid|void|rapid)$", c, re.I
    )


def _is_int_valued(s: pd.Series) -> bool:
    x = s.dropna()
    return len(x) > 0 and bool(np.all(np.equal(np.mod(x.astype(float), 1), 0)))


def _guess(c, s: pd.Series, target) -> dict:
    out = lambda t, conf, why, **kw: {
        "type": t,
        "confidence": conf,
        "reason": why,
        **kw,
    }
    if c == target:
        return out("target", "high", "chosen as the target")
    n, u = int(s.notna().sum()), int(s.nunique())
    ratio = u / max(n, 1)
    if pd.api.types.is_datetime64_any_dtype(s):
        return out("datetime", "high", "stored as a date")
    if pd.api.types.is_bool_dtype(s):
        return out("categorical", "high", "true/false values")
    if pd.api.types.is_numeric_dtype(s):
        if _CODE_NAME.search(str(c)):
            return out(
                "categorical",
                "medium",
                "the name suggests a code (zip, phone...), which is a label and not a quantity",
            )
        if _is_int_valued(s):
            if _id_name(c) and ratio > 0.9:
                return out(
                    "id",
                    "high",
                    f"the name says id and {u:,} of {n:,} values are different",
                )
            if n > 50 and ratio > 0.98 and s.dropna().is_monotonic_increasing:
                return out("id", "medium", "counts up row by row, like a row number")
            if u <= 2:
                return out(
                    "categorical", "medium", "only two values, probably a yes/no flag"
                )
            if u <= 10:
                return out(
                    "discrete",
                    "low",
                    f"only {u} whole-number values: a small count or a category? Please check",
                )
            if u <= 50 and ratio < 0.05:
                return out(
                    "discrete",
                    "medium",
                    f"whole numbers with only {u} different values",
                )
        return out("continuous", "high", "numbers with many different values")
    clean = s.astype("string").str.strip().replace("", pd.NA).dropna()
    if clean.empty:
        return out("categorical", "low", "the column is empty")
    if pd.to_numeric(clean, errors="coerce").notna().all():
        lead0 = clean.str.match(r"^0\d+$").any()
        if lead0 or _CODE_NAME.search(str(c)):
            return out(
                "categorical",
                "high" if lead0 else "medium",
                "digits with leading zeros or a code-like name: a label, not a quantity",
            )
        return out("continuous", "medium", "numbers stored as text")
    samp = clean.sample(min(len(clean), 200), random_state=0)
    if (
        samp.str.contains(r"\d").mean() > 0.9
        and samp.str.contains(r"[-/:]").mean() > 0.9
    ):
        try:
            if (
                pd.to_datetime(samp, errors="coerce", format="mixed").notna().mean()
                >= 0.95
            ):
                return out("datetime", "medium", "text that parses as dates")
        except Exception:
            pass
    low = set(clean.str.lower().unique())
    for sc in SCALES:
        if len(low) >= 2 and low <= set(sc):
            return out(
                "ordinal",
                "medium",
                "values form a natural scale",
                order=[v for v in sc if v in low],
            )
    if _id_name(c) and ratio > 0.9:
        return out("id", "high", "the name says id and almost every value is different")
    avg = clean.str.len().mean()
    if avg > 40 and ratio > 0.5:
        return out(
            "text", "medium", f"long, mostly unique text (about {avg:.0f} characters)"
        )
    if u <= 20 or ratio < 0.05:
        return out("categorical", "high", f"{u} different values")
    if ratio > 0.95:
        return out(
            "id",
            "low",
            "almost every value is different; an identifier or free text? Please check",
        )
    return out("categorical", "medium", f"{u} different values (many categories)")


def infer_types(df: pd.DataFrame, target=None) -> dict:
    return {str(c): _guess(c, df[c], target) for c in df.columns}
