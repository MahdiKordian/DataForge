# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""DataForge Advisor: finds problems, explains them and lays out options with pros, cons and conditions.
It never declares a single 'best' method; the user decides and the decision is recorded."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .splitting import is_classification


def merge(base: dict, patch: dict) -> dict:
    """Deep-merge a settings patch (dicts merge, lists are united, anything else is replaced)."""
    out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v
    return out


def _opt(key, label, patch, pros, cons, when=""):
    return {"key": key, "label": label, "patch": patch, "pros": pros, "cons": cons, "when": when}


def _num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _blank_aware(s: pd.Series) -> pd.Series:
    """Missing mask that also counts empty or whitespace-only text as missing."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s.isna()
    return s.isna() | (s.astype("string").str.strip() == "").fillna(False)


def _numeric_text(s: pd.Series) -> pd.Series | None:
    """If a text column holds only numbers (blanks aside), return it as numbers, else None."""
    if _num(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return None
    clean = s.astype("string").str.strip().replace("", pd.NA)
    nn = clean.dropna()
    if len(nn) and pd.to_numeric(nn, errors="coerce").notna().all():
        return pd.to_numeric(clean, errors="coerce")
    return None


def _missing_options(c, s):
    pc = lambda strat: {"Clean": {"per_column": {c: strat}}}
    if _num(s):
        return [
            _opt("median", "Fill with the median", pc("Median"), "Barely moved by extreme values.",
                 "Shrinks the spread and can distort the shape of the distribution.", "Skewed columns or columns with outliers."),
            _opt("mean", "Fill with the mean", pc("Mean"), "Simple and keeps the overall average.",
                 "Pulled by extreme values.", "Roughly symmetric columns without strong outliers."),
            _opt("zero", "Fill with 0", pc("Fill with 0"), "Keeps every row.",
                 "Only correct if a missing value really means zero.", "A missing value genuinely means none."),
            _opt("drop_rows", "Drop rows with a missing value", pc("Drop rows"), "No invented values.",
                 "Loses data and can bias the sample if missingness is not random.", "Very few rows affected."),
            _opt("leave", "Leave as is", pc("Leave as is"), "Lets a model that handles NaN decide.",
                 "Many models fail on missing values.", "You will use a model that supports missing values.")]
    return [
        _opt("mode", "Fill with the most common value", pc("Most common"), "Simple.", "Over-represents the largest category.", "One category clearly dominates."),
        _opt("unknown", "Fill with 'Unknown'", pc("Fill with 'Unknown'"), "Keeps the fact that it was missing.",
             "Adds a new category.", "Missingness itself may carry meaning."),
        _opt("drop_rows", "Drop rows with a missing value", pc("Drop rows"), "No invented values.", "Loses data.", "Very few rows affected."),
        _opt("leave", "Leave as is", pc("Leave as is"), "Keeps the data untouched.", "Encoding or models may fail.", "")]


def diagnose(df: pd.DataFrame, target=None) -> list[dict]:
    """Each finding: id, severity (3 high .. 1 low), problem, reason, evidence, options, confidence, note."""
    out = []
    n = max(len(df), 1)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        as_num = _numeric_text(s)
        miss = _blank_aware(s)
        p = miss.mean() * 100
        if as_num is not None:
            blanks = int((miss & s.notna()).sum())
            out.append({"id": f"numtext:{c}", "column": c, "problem": "Numbers stored as text", "severity": 2,
                        "reason": f"`{c}` is stored as text but every value is a number" + (f", except {blanks} blank value(s)." if blanks else "."),
                        "evidence": {"blank values": blanks, "dtype": str(s.dtype)},
                        "options": [_opt("convert", "Convert to numbers", {"Clean": {"to_num": True}}, "Lets the column be scaled and analysed as a number.",
                                         "Blank values become missing and need a fill strategy.", "The values are really quantities."),
                                    _opt("keep", "Keep as text", {"Clean": {"to_num": False}}, "Nothing changes.",
                                         "It will be encoded as a category instead.", "The values are labels, not quantities (e.g. codes).")],
                        "confidence": "high", "note": ""})
        if 0 < p:
            ev = {"missing %": round(p, 1), "missing cells": int(miss.sum())}
            reason = f"{p:.1f}% of `{c}` is empty."
            num_view = s if _num(s) else as_num
            if num_view is not None and num_view.notna().sum() > 2:
                sk = float(num_view.dropna().skew())
                ev["skew"] = round(sk, 2)
                reason += f" The values are {'strongly skewed' if abs(sk) > 1 else 'fairly symmetric'}, which matters when choosing between mean and median."
            opts = _missing_options(c, s if as_num is None else as_num)
            if p > 60:
                opts.insert(0, _opt("drop_col", "Drop the column", {"Transform": {"drop": [c]}}, "Removes a column that is mostly empty.",
                                    "Loses whatever information the remaining values carry.", "The column is not essential."))
            out.append({"id": f"missing:{c}", "column": c, "problem": "Missing values", "severity": 3 if p > 30 else 2,
                        "reason": reason, "evidence": ev, "options": opts, "confidence": "high",
                        "note": "Missing-value choice depends on why the values are missing, which the data alone cannot tell." if p > 5 else ""})
    d = int(df.duplicated().sum())
    if d:
        out.append({"id": "dups", "column": None, "problem": "Duplicate rows", "severity": 2,
                    "reason": f"{d} row(s) are exact copies of another row.", "evidence": {"duplicates": d},
                    "options": [_opt("remove", "Remove duplicates", {"Clean": {"dups": True}}, "Avoids counting a row twice.",
                                     "Real repeated records would be lost.", "Rows are copies by mistake."),
                                _opt("keep", "Keep them", {"Clean": {"dups": False}}, "Nothing is lost.",
                                     "Duplicates can leak between train and test.", "Repeats are real, e.g. identical transactions.")],
                    "confidence": "medium", "note": "Identical rows are not always errors."})
    const = [c for c in df.columns if c != target and df[c].nunique(dropna=True) <= 1]
    if const:
        out.append({"id": "const", "column": None, "problem": "Constant columns", "severity": 1,
                    "reason": f"{len(const)} column(s) never change: {', '.join(map(str, const[:5]))}.", "evidence": {"columns": const[:10]},
                    "options": [_opt("drop", "Drop them", {"Clean": {"const": True}}, "They carry no information.", "None in practice.", ""),
                                _opt("keep", "Keep them", {"Clean": {"const": False}}, "Schema stays identical.", "Useless column.", "")],
                    "confidence": "high", "note": ""})
    for c in df.columns:
        s = df[c]
        if c == target:
            continue
        idlike = (not _num(s) or pd.api.types.is_integer_dtype(s)) and s.notna().sum() > 20 and s.nunique() / max(s.notna().sum(), 1) > 0.95
        name_id = str(c).lower().replace("_", "").endswith("id") or str(c).lower() in ("id", "index", "uuid")
        if idlike and (name_id or not _num(s)):
            out.append({"id": f"id:{c}", "column": c, "problem": "Looks like an identifier", "severity": 2,
                        "reason": f"`{c}` has {s.nunique():,} different values in {s.notna().sum():,} rows, so almost every row is unique.",
                        "evidence": {"unique ratio": round(s.nunique() / max(s.notna().sum(), 1), 3)},
                        "options": [_opt("drop", "Drop it", {"Transform": {"drop": [c]}}, "A model cannot learn from an ID and may memorise it.",
                                         "You lose the key for joining results back.", "It is only a label for the row."),
                                    _opt("keep", "Keep it", {}, "Available for later joining.", "Can be scaled or encoded by mistake.", "")],
                    "confidence": "medium" if name_id else "low",
                    "note": "" if name_id else "Low confidence: the name does not say it is an ID. Please check."})
        elif _num(s) and pd.api.types.is_integer_dtype(s) and 2 < s.nunique() <= 8 and s.notna().sum() > 30:
            out.append({"id": f"cat?:{c}", "column": c, "problem": "Number that may be a category", "severity": 1,
                        "reason": f"`{c}` holds only {s.nunique()} whole-number values ({', '.join(map(str, sorted(s.dropna().unique())[:6]))}).",
                        "evidence": {"distinct values": int(s.nunique())}, "options": [], "confidence": "low",
                        "note": "Low confidence. If these numbers are labels (not quantities), treat the column as categorical. DataForge will not change it unless you do."})
    cont = [c for c in df.select_dtypes("number").columns if c != target and df[c].nunique() > 2]
    skewed = [c for c in cont if df[c].notna().sum() > 2 and abs(df[c].dropna().skew()) > 1]
    outl = {}
    for c in cont:
        x = df[c].dropna()
        if len(x) >= 4:
            q1, q3 = x.quantile([0.25, 0.75]); i = q3 - q1
            if i > 0:
                r = ((x < q1 - 1.5 * i) | (x > q3 + 1.5 * i)).mean() * 100
                if r > 5:
                    outl[c] = round(float(r), 1)
    if outl:
        out.append({"id": "outliers", "column": None, "problem": "Outliers", "severity": 2,
                    "reason": f"{len(outl)} column(s) have more than 5% of values outside the usual IQR range: "
                              + ", ".join(f"`{k}` ({v}%)" for k, v in list(outl.items())[:5]) + ".",
                    "evidence": outl,
                    "options": [_opt("clip_pct", "Clip at the 1st and 99th percentile", {"Transform": {"outliers": "Clip at 1st and 99th percentile"}},
                                     "Keeps every row.", "Changes real values.", "Extremes are noise or entry errors."),
                                _opt("clip_iqr", "Clip to IQR bounds", {"Transform": {"outliers": "Clip to IQR bounds"}},
                                     "Keeps every row; adapts to the spread.", "Can clip a lot on skewed data.", ""),
                                _opt("remove", "Remove those training rows", {"Transform": {"outliers": "Remove rows with outliers (IQR)"}},
                                     "No altered values.", "Loses rows; only the training set is affected.", "Outliers are clearly errors."),
                                _opt("leave", "Leave them", {"Transform": {"outliers": "Leave as is"}},
                                     "Real extremes stay.", "Can dominate some models.", "Extreme values are real and important.")],
                    "confidence": "medium", "note": "A large value is not an error by itself, for example one very high income."})
    if skewed:
        out.append({"id": "skew", "column": None, "problem": "Skewed numbers", "severity": 1,
                    "reason": f"{len(skewed)} column(s) are strongly skewed: {', '.join(map(str, skewed[:5]))}.",
                    "evidence": {c: round(float(df[c].dropna().skew()), 2) for c in skewed[:10]},
                    "options": [_opt("log", "Log transform", {"Transform": {"skew": True, "skew_method": "Log (log1p)"}},
                                     "Pulls in long right tails.", "Values become harder to read.", "Linear or distance-based models."),
                                _opt("sqrt", "Square root", {"Transform": {"skew": True, "skew_method": "Square root"}},
                                     "Gentler than log.", "May not remove the skew fully.", ""),
                                _opt("none", "Leave as is", {"Transform": {"skew": False}},
                                     "Original scale.", "Skew can hurt some models.", "Tree-based models do not need this.")],
                    "confidence": "medium", "note": ""})
    if target in df.columns:
        t = df[target].dropna()
        if len(t) and is_classification(t):
            share = t.value_counts(normalize=True)
            if share.min() < 0.2:
                out.append({"id": "imbalance", "column": target, "problem": "Imbalanced target", "severity": 2,
                            "reason": f"The smallest class of `{target}` is {share.min() * 100:.1f}% of the rows.",
                            "evidence": {str(k): round(float(v) * 100, 1) for k, v in share.items()},
                            "options": [_opt("over", "Oversample minority (training only)", {"Split": {"balance": "Oversample minority"}},
                                             "Keeps all information.", "Repeats rows, can overfit.", ""),
                                        _opt("under", "Undersample majority (training only)", {"Split": {"balance": "Undersample majority"}},
                                             "Fast.", "Throws data away.", "Plenty of data."),
                                        _opt("none", "Do nothing now", {"Split": {"balance": "None"}},
                                             "No artificial rows.", "Plain accuracy can be misleading.", "Use class weights or suitable metrics instead.")],
                            "confidence": "medium", "note": "Balancing is never applied to validation or test."})
    return sorted(out, key=lambda f: -f["severity"])


def what_if_missing(train: pd.DataFrame, col: str, target=None) -> pd.DataFrame:
    """Side-by-side effect of each strategy on one column, measured on the TRAINING data."""
    s = train[col]
    nt = _numeric_text(s)
    if nt is not None:
        s = nt
    elif not _num(s):
        s = s.where(~_blank_aware(s))
    if not _num(s):
        rows = []
        for name in ("Most common", "Fill with 'Unknown'", "Drop rows"):
            kept = s.dropna() if name == "Drop rows" else s.fillna(s.mode().iloc[0] if name == "Most common" and s.notna().any() else "Unknown")
            top = kept.value_counts(normalize=True)
            rows.append({"strategy": name, "values changed": 0 if name == "Drop rows" else int(s.isna().sum()),
                         "rows kept": len(kept), "rows lost": len(s) - len(kept),
                         "top category share %": round(float(top.iloc[0]) * 100, 1) if len(top) else np.nan})
        return pd.DataFrame(rows)
    ref, tnum = s.dropna(), (train[target] if target in train.columns and _num(train[target]) else None)
    rows = []
    for name, fill in (("Median", s.median()), ("Mean", s.mean()), ("Fill with 0", 0.0), ("Drop rows", None)):
        if fill is None:
            new, kept = s.dropna(), s.notna()
        else:
            new, kept = s.fillna(fill), pd.Series(True, index=s.index)
        row = {"strategy": name, "values changed": 0 if fill is None else int(s.isna().sum()), "rows kept": int(kept.sum()),
               "rows lost": int((~kept).sum()), "mean": new.mean(), "std": new.std(),
               "std change %": (new.std() / ref.std() - 1) * 100 if ref.std() else np.nan}
        if tnum is not None:
            row["corr with target"] = new.corr(tnum[new.index])
        rows.append(row)
    return pd.DataFrame(rows)
