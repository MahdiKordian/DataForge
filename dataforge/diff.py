# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Data Quality Diff Engine: what changed between two versions of a dataset.
Modes are only a matter of which two frames you pass in:
  Before vs After        -> loaded data, prepared data
  Version A vs B         -> two files
  Training vs Production -> training data, new data (drift)"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MAX_NUM, MAX_CORR = 200, 40


def _psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 20 or len(b) < 20:
        return float("nan")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _level(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    return "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "large"


def _is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def compare(a: pd.DataFrame, b: pd.DataFrame, label_a: str = "A", label_b: str = "B") -> dict:
    ca, cb = list(a.columns), list(b.columns)
    shared = [c for c in ca if c in cb]
    out = {"labels": (label_a, label_b)}
    out["summary"] = pd.DataFrame([
        {"metric": "rows", label_a: len(a), label_b: len(b), "change": len(b) - len(a)},
        {"metric": "columns", label_a: len(ca), label_b: len(cb), "change": len(cb) - len(ca)},
        {"metric": "duplicate rows", label_a: int(a.duplicated().sum()), label_b: int(b.duplicated().sum()),
         "change": int(b.duplicated().sum() - a.duplicated().sum())},
        {"metric": "missing cells %", label_a: round(a.isna().mean().mean() * 100, 2) if a.size else 0.0,
         label_b: round(b.isna().mean().mean() * 100, 2) if b.size else 0.0,
         "change": round((b.isna().mean().mean() - a.isna().mean().mean()) * 100, 2) if a.size and b.size else 0.0}])
    out["added"], out["removed"] = [c for c in cb if c not in ca], [c for c in ca if c not in cb]
    out["dtype_changes"] = [{"column": c, label_a: str(a[c].dtype), label_b: str(b[c].dtype)}
                            for c in shared if str(a[c].dtype) != str(b[c].dtype)]
    out["missing"] = pd.DataFrame([{"column": c, f"{label_a} %": a[c].isna().mean() * 100, f"{label_b} %": b[c].isna().mean() * 100,
                                    "change": (b[c].isna().mean() - a[c].isna().mean()) * 100} for c in shared])
    rows, flags = [], []
    for c in [c for c in shared if _is_num(a[c]) and _is_num(b[c])][:MAX_NUM]:
        x, y = a[c].replace([np.inf, -np.inf], np.nan).dropna(), b[c].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty or y.empty:
            continue
        sd = x.std() if x.std() and not np.isnan(x.std()) else np.nan
        psi = _psi(x, y)
        ks = float(stats.ks_2samp(x, y).statistic)
        rows.append({"column": c, f"mean {label_a}": x.mean(), f"mean {label_b}": y.mean(),
                     f"median {label_a}": x.median(), f"median {label_b}": y.median(),
                     "mean shift (in SDs)": (y.mean() - x.mean()) / sd if sd == sd else np.nan,
                     "KS": ks, "PSI": psi, "drift": _level(psi)})
        if _level(psi) == "large":
            flags.append({"level": "warning", "message": f"`{c}` changed a lot (PSI {psi:.2f}, mean {x.mean():.3g} -> {y.mean():.3g})."})
    out["numeric"] = pd.DataFrame(rows)
    cat_rows = []
    for c in [c for c in shared if not _is_num(a[c]) and not pd.api.types.is_datetime64_any_dtype(a[c])]:
        va, vb = a[c].dropna().astype(str), b[c].dropna().astype(str)
        new, gone = sorted(set(vb) - set(va)), sorted(set(va) - set(vb))
        pa, pb = va.value_counts(normalize=True), vb.value_counts(normalize=True)
        tvd = float(0.5 * (pa.reindex(pa.index.union(pb.index), fill_value=0)
                           - pb.reindex(pa.index.union(pb.index), fill_value=0)).abs().sum()) if len(va) and len(vb) else np.nan
        cat_rows.append({"column": c, "new categories": ", ".join(new[:5]) + (" ..." if len(new) > 5 else ""),
                         "n new": len(new), "disappeared": len(gone), "share change (TVD)": tvd})
        if new:
            flags.append({"level": "warning", "message": f"`{c}` has {len(new)} category/categories not in {label_a}: {', '.join(new[:3])}."})
    out["categorical"] = pd.DataFrame(cat_rows)
    nc = [c for c in shared if _is_num(a[c]) and _is_num(b[c])]
    nc = sorted(nc, key=lambda c: -(a[c].var() if a[c].var() == a[c].var() else 0))[:MAX_CORR]
    pairs = []
    if len(nc) > 1:
        ra, rb = a[nc].corr(), b[nc].corr()
        for i, c1 in enumerate(nc):
            for c2 in nc[i + 1:]:
                d = rb.loc[c1, c2] - ra.loc[c1, c2]
                if abs(d) >= 0.2:
                    pairs.append({"pair": f"{c1} / {c2}", label_a: ra.loc[c1, c2], label_b: rb.loc[c1, c2], "change": d})
    out["correlations"] = pd.DataFrame(sorted(pairs, key=lambda r: -abs(r["change"]))[:15])
    if out["added"]:
        flags.append({"level": "info", "message": f"{len(out['added'])} column(s) added: {', '.join(map(str, out['added'][:5]))}."})
    if out["removed"]:
        flags.append({"level": "warning", "message": f"{len(out['removed'])} column(s) removed: {', '.join(map(str, out['removed'][:5]))}."})
    for r in out["dtype_changes"]:
        flags.append({"level": "warning", "message": f"`{r['column']}` changed type: {r[label_a]} -> {r[label_b]}."})
    out["flags"] = flags
    out["note"] = ("A change in the data is not automatically a drop in model quality. "
                   "Judge how large the change is, how important the column is, and what it means for the task.")
    return out
