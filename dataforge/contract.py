# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Data Contract: rules a dataset must satisfy, checked every time new data arrives."""
from __future__ import annotations

import difflib

import pandas as pd

MAX_EXAMPLES = 5


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "text"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    clean = s.astype("string").str.strip().replace("", pd.NA).dropna()
    if len(clean) and pd.to_numeric(clean, errors="coerce").notna().all():
        return "numeric"
    return "text"


def _as_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s
    return pd.to_numeric(s.astype("string").str.strip().replace("", pd.NA), errors="coerce")


def infer_contract(df: pd.DataFrame, target=None, max_allowed: int = 20) -> dict:
    """A starting point built from the data. The user is expected to review and edit it."""
    cols = {}
    for c in df.columns:
        s, k = df[c], _kind(df[c])
        rule = {"kind": k, "nullable": bool(s.isna().any() or (k == "numeric" and not pd.api.types.is_numeric_dtype(s))),
                "unique": bool(len(s) > 1 and s.notna().all() and s.is_unique and k == "text"),
                "min": None, "max": None, "allowed": None}
        if k == "numeric":
            n = _as_numeric(s).dropna()
            if len(n):
                rule["min"], rule["max"] = float(n.min()), float(n.max())
        elif k == "text" and 0 < s.nunique() <= max_allowed:
            rule["allowed"] = sorted(map(str, s.dropna().unique()))
        cols[str(c)] = rule
    return {"format": "dataforge-contract", "version": 1, "target": target, "allow_extra_columns": True, "columns": cols}


def validate(df: pd.DataFrame, contract: dict) -> dict:
    """Returns {"status": passed|warnings|failed, "issues": [...], "bad_rows": [index, ...]}."""
    issues, bad = [], set()

    def add(level, col, rule, msg, rows=None):
        rows = [] if rows is None else list(rows)
        issues.append({"level": level, "column": col, "rule": rule, "message": msg, "count": len(rows),
                       "examples": [int(i) if isinstance(i, int) or hasattr(i, "item") else i for i in rows[:MAX_EXAMPLES]]})
        if level == "error":
            bad.update(rows)

    rules = contract["columns"]
    present = {str(c): c for c in df.columns}
    for name in rules:
        if name not in present:
            near = difflib.get_close_matches(name, [c for c in present if c not in rules], n=1, cutoff=0.5)
            hint = f" A similar column exists: `{near[0]}`." if near else ""
            add("error", name, "exists", f"Column `{name}` is missing.{hint}")
    extra = [c for c in present if c not in rules]
    if extra:
        add("error" if not contract.get("allow_extra_columns", True) else "warning", ", ".join(extra[:5]), "extra",
            f"{len(extra)} column(s) are not in the contract: {', '.join(extra[:5])}.")

    for name, r in rules.items():
        if name not in present:
            continue
        s = df[present[name]]
        if r["kind"] == "numeric":
            n = _as_numeric(s)
            wrong = s.notna() & n.isna() & (s.astype("string").str.strip() != "")
            if wrong.any():
                add("error", name, "type", f"`{name}` should be numeric but {int(wrong.sum())} value(s) are not.", s.index[wrong])
            if r.get("min") is not None:
                out = n < r["min"]
                if out.any():
                    add("error", name, "min", f"`{name}`: {int(out.sum())} value(s) below {r['min']:g}.", s.index[out])
            if r.get("max") is not None:
                out = n > r["max"]
                if out.any():
                    add("error", name, "max", f"`{name}`: {int(out.sum())} value(s) above {r['max']:g}.", s.index[out])
            missing = n.isna() & ~wrong
        elif r["kind"] == "datetime":
            d = pd.to_datetime(s, errors="coerce")
            wrong = s.notna() & d.isna()
            if wrong.any():
                add("error", name, "type", f"`{name}` should be a date but {int(wrong.sum())} value(s) are not.", s.index[wrong])
            missing = d.isna() & ~wrong
        else:
            missing = s.isna()
        if not r.get("nullable", True) and missing.any():
            add("error", name, "nullable", f"`{name}` must not be empty but has {int(missing.sum())} empty value(s).", s.index[missing])
        if r.get("allowed"):
            bad_cat = s.notna() & ~s.astype(str).isin(set(map(str, r["allowed"])))
            if bad_cat.any():
                vals = ", ".join(map(str, s[bad_cat].astype(str).unique()[:4]))
                add("error", name, "allowed", f"`{name}`: {int(bad_cat.sum())} value(s) outside the allowed set (e.g. {vals}).", s.index[bad_cat])
        if r.get("unique"):
            dup = s.notna() & s.duplicated(keep=False)
            if dup.any():
                add("error", name, "unique", f"`{name}` must be unique but {int(dup.sum())} row(s) share a value.", s.index[dup])

    errors = [i for i in issues if i["level"] == "error"]
    status = "failed" if errors else "warnings" if issues else "passed"
    return {"status": status, "issues": issues, "bad_rows": sorted(bad, key=str)[:1000],
            "summary": f"{len(errors)} error(s), {len(issues) - len(errors)} warning(s), {len(bad)} row(s) break a rule."}
