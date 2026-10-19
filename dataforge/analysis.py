# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Statistical analysis engine (no Streamlit). Every measure states what it measures and where it is weak."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

LIMITS = {
    "pearson": "Linear relationship only. A value near 0 does not mean 'unrelated': a curve can give 0. One outlier can change it a lot.",
    "spearman": "Monotonic relationship (always up or always down), based on ranks. Robust to outliers, blind to U-shapes.",
    "cramers_v": "Strength of the link between two categorical columns, 0 to 1 (bias-corrected). Says nothing about direction.",
    "eta": "How much of a numeric column's variation is explained by a category, 0 to 1. Sensitive to tiny groups.",
    "nmi": "Normalised mutual information from binned data: catches non-linear links, but needs plenty of rows and is only approximate.",
    "vif": "How well a column is predicted by the other numeric columns. Above about 5 to 10 suggests collinearity; it matters mostly for linear models.",
    "normality": "A small p-value says the shape is not normal. With many rows, even harmless deviations give small p-values, so look at skew and the chart too.",
    "missing": "These tests can show that missingness depends on other columns. They cannot prove it is random, and they cannot detect missingness that depends on the missing value itself.",
}
PLACEHOLDERS = {"n/a", "na", "null", "none", "nan", "?", "-", "--", "unknown", "missing", "undefined", "tbd", "#n/a"}
SAMPLE = 20_000


def _num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def na_frame(df: pd.DataFrame) -> pd.DataFrame:
    """True where a value is missing. Empty or whitespace-only text counts as missing, like a real NaN."""
    out = df.isna()
    for c in df.columns:
        if not (_num(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
            out[c] = out[c] | (df[c].astype("string").str.strip() == "").fillna(False)
    return out


def _samp(df, n=SAMPLE):
    return df.sample(n, random_state=0) if len(df) > n else df


def split_cols(df: pd.DataFrame, target=None, types: dict | None = None):
    """(numeric, categorical, datetime) columns, using confirmed variable types when given."""
    types = types or {}
    num, cat, dt = [], [], []
    for c in df.columns:
        if c == target:
            continue
        ty, s = types.get(c), df[c]
        if ty in ("id", "text", "target"):
            continue
        if ty == "datetime" or pd.api.types.is_datetime64_any_dtype(s):
            dt.append(c)
        elif ty in ("categorical", "ordinal"):
            cat.append(c)
        elif _num(s):
            num.append(c)
        elif ty in ("continuous", "discrete"):
            num.append(c)
        elif s.nunique() <= max(30, 0.05 * len(s)):
            cat.append(c)
    return num, cat, dt


def as_numeric(df, cols):
    return df[cols].apply(lambda s: s if _num(s) else pd.to_numeric(s.astype("string").str.strip().replace("", pd.NA), errors="coerce"))


# ------------------------------------------------------------------------------------ univariate
def univariate_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = as_numeric(df, [c])[c].replace([np.inf, -np.inf], np.nan)
        v = x.dropna()
        r = {"column": c, "count": len(v), "missing %": x.isna().mean() * 100, "unique": int(v.nunique())}
        if len(v) == 0:
            rows.append(r)
            continue
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        out = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        r.update({"mean": v.mean(), "median": v.median(), "std": v.std(), "variance": v.var(), "min": v.min(), "q1": q1, "q3": q3,
                  "max": v.max(), "IQR": iqr, "skew": v.skew() if len(v) > 2 else np.nan,
                  "kurtosis": v.kurt() if len(v) > 3 else np.nan, "outliers (IQR)": out,
                  "outliers %": out / len(v) * 100, "zeros %": (v == 0).mean() * 100})
        if len(v) >= 20 and v.nunique() > 3 and v.std() > 0:
            p = float(stats.normaltest(v.sample(min(len(v), 5000), random_state=0))[1])
            r["normality p"], r["shape"] = p, ("no clear departure from normal" if p > 0.05 else "not normal")
        rows.append(r)
    return pd.DataFrame(rows)


def _norm_key(v) -> str:
    return re.sub(r"[\W_]+", "", str(v).lower())


def univariate_categorical(df: pd.DataFrame, cols: list, rare_pct: float = 1.0):
    """Returns (table, variants). variants maps a column to groups of spellings that probably mean the same thing."""
    rows, variants = [], {}
    for c in cols:
        s = df[c].dropna().astype(str)
        n = len(s)
        r = {"column": c, "count": n, "missing %": df[c].isna().mean() * 100, "unique": int(s.nunique())}
        if n:
            vc = s.value_counts()
            p = vc / n
            r.update({"top value": vc.index[0], "top share %": p.iloc[0] * 100,
                      "entropy (0-1)": float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0,
                      "rare categories": int((p * 100 < rare_pct).sum()), "singletons": int((vc == 1).sum())})
            groups = {}
            for v in vc.index[:500]:
                groups.setdefault(_norm_key(v), []).append(v)
            bad = [g for g in groups.values() if len(g) > 1]
            if bad:
                variants[c] = bad[:5]
        rows.append(r)
    return pd.DataFrame(rows), variants


def time_profile(df: pd.DataFrame, col: str, value_cols: list | None = None) -> dict:
    t = pd.to_datetime(df[col], errors="coerce")
    bad = int(t.isna().sum() - df[col].isna().sum())
    t = t.dropna().sort_values()
    out = {"column": col, "unparseable": bad}
    if len(t) < 3:
        out["note"] = "Too few dates to profile."
        return out
    gaps = t.diff().dropna()
    med = gaps.median()
    out.update({"first": t.iloc[0], "last": t.iloc[-1], "rows": len(t), "median gap": med,
                "duplicate timestamps": int(t.duplicated().sum()),
                "irregular gaps": int((gaps > 3 * med).sum()) if med > pd.Timedelta(0) else 0})
    per = "D" if med <= pd.Timedelta(days=1) else "W" if med <= pd.Timedelta(days=8) else "M"
    counts = t.dt.to_period(per).value_counts().sort_index()
    full = pd.period_range(counts.index.min(), counts.index.max(), freq=per)
    out["period"], out["empty periods"], out["periods"] = per, int(len(full) - len(counts)), counts
    trends = []
    for c in (value_cols or [])[:8]:
        v = pd.to_numeric(df.loc[t.index, c], errors="coerce") if c in df else None
        if v is not None and v.notna().sum() > 10 and v.nunique() > 2:
            rho, p = stats.spearmanr(np.arange(len(v))[v.notna().to_numpy()], v.dropna())
            trends.append({"column": c, "trend (Spearman)": rho, "p-value": p})
    out["trends"] = pd.DataFrame(trends)
    return out


# ------------------------------------------------------------------------------- relationships
def contingency(x, y) -> np.ndarray:
    """Counts table of two categorical columns (rows with a missing value in either are ignored). Much faster than pd.crosstab."""
    xc, xu = pd.factorize(pd.Series(x).reset_index(drop=True))
    yc, yu = pd.factorize(pd.Series(y).reset_index(drop=True))
    ok = (xc >= 0) & (yc >= 0)
    if not ok.any():
        return np.zeros((0, 0))
    r, c = len(xu), len(yu)
    return np.bincount(xc[ok] * c + yc[ok], minlength=r * c).reshape(r, c).astype(float)


def chi2_cramers(ct: np.ndarray) -> tuple[float, float]:
    """(Cramer's V with bias correction, chi-square p-value) from a counts table."""
    ct = ct[ct.sum(1) > 0][:, ct.sum(0) > 0] if ct.size else ct
    if ct.ndim != 2 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0, 1.0
    n = ct.sum()
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = float(((ct - exp) ** 2 / exp).sum())
    r, k = ct.shape
    p = float(stats.chi2.sf(chi2, (r - 1) * (k - 1)))
    phi2c = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return (float(np.sqrt(phi2c / d)) if d > 0 else 0.0), p


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    return chi2_cramers(contingency(x, y))[0]


def eta_squared(cat: pd.Series, val: pd.Series) -> float:
    d = pd.DataFrame({"g": cat, "v": val}).dropna()
    if d.g.nunique() < 2 or len(d) < 3:
        return 0.0
    tot = ((d.v - d.v.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    within = d.groupby("g")["v"].apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    return float(max(0.0, 1 - within / tot))


def nmi(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """Normalised mutual information between two columns (numeric ones are cut into equal-count bins)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 30:
        return float("nan")
    q = lambda s: pd.qcut(s, min(bins, s.nunique()), duplicates="drop").cat.codes if _num(s) and s.nunique() > bins else pd.factorize(s)[0]
    a, b = q(d.x), q(d.y)
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    pxy = ct / ct.sum()
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    hx, hy = -float((px[px > 0] * np.log(px[px > 0])).sum()), -float((py[py > 0] * np.log(py[py > 0])).sum())
    m = min(hx, hy)
    return float(min(1.0, max(0.0, mi / m))) if m > 1e-9 else 0.0


def _pick(df, cols, k):
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return sorted(cols, key=lambda c: -df[c].notna().sum())[:k]


def correlations(df: pd.DataFrame, num: list, max_cols: int = 40) -> dict:
    cols = _pick(df, num, max_cols)
    x = as_numeric(_samp(df), cols).replace([np.inf, -np.inf], np.nan)
    return {"pearson": x.corr("pearson"), "spearman": x.rank().corr("pearson"), "columns": cols, "capped": len(num) > len(cols)}   # Spearman = Pearson of the ranks


def mixed_associations(df: pd.DataFrame, num: list, cat: list, max_cols: int = 30) -> pd.DataFrame:
    """One 0-to-1 matrix across numeric and categorical columns: |Spearman|, Cramér's V or eta, whichever fits the pair."""
    d = _samp(df)
    cols = _pick(d, num, max_cols // 2 + max_cols % 2) + _pick(d, [c for c in cat if d[c].nunique() <= 50], max_cols // 2)
    nset = set(num)
    n = as_numeric(d, [c for c in cols if c in nset]) if any(c in nset for c in cols) else pd.DataFrame(index=d.index)
    M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in nset and b in nset:
                v = abs(n[a].corr(n[b], method="spearman"))
            elif a not in nset and b not in nset:
                v = cramers_v(d[a], d[b])
            else:
                c_, v_ = (a, b) if a not in nset else (b, a)
                v = np.sqrt(eta_squared(d[c_], n[v_]))
            M.loc[a, b] = M.loc[b, a] = 0.0 if pd.isna(v) else v
    return M


def top_pairs(M: pd.DataFrame, top: int = 15, min_value: float = 0.0) -> pd.DataFrame:
    cols = list(M.columns)
    rows = [(a, b, float(M.loc[a, b])) for i, a in enumerate(cols) for b in cols[i + 1:] if pd.notna(M.loc[a, b])]
    out = pd.DataFrame(rows, columns=["column A", "column B", "value"])
    out = out[out.value.abs() >= min_value]
    return out.reindex(out.value.abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def nonlinear_candidates(df: pd.DataFrame, num: list, max_cols: int = 20, min_nmi: float = 0.15, max_pearson: float = 0.3) -> pd.DataFrame:
    d = _samp(as_numeric(df, _pick(df, num, max_cols)), 5000).replace([np.inf, -np.inf], np.nan)
    rows = []
    for i, a in enumerate(d.columns):
        for b in d.columns[i + 1:]:
            r, m = d[a].corr(d[b]), nmi(d[a], d[b])
            if pd.notna(m) and m >= min_nmi and (pd.isna(r) or abs(r) < max_pearson):
                rows.append({"column A": a, "column B": b, "Pearson": r, "NMI": m})
    return pd.DataFrame(rows).sort_values("NMI", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["column A", "column B", "Pearson", "NMI"])


def vif_table(df: pd.DataFrame, num: list, max_cols: int = 30) -> pd.DataFrame:
    cols = _pick(df, num, max_cols)
    x = as_numeric(_samp(df), cols).replace([np.inf, -np.inf], np.nan).dropna()
    if len(cols) < 2 or len(x) <= len(cols) + 1:
        return pd.DataFrame(columns=["column", "VIF"])
    c = np.corrcoef(x.to_numpy(float), rowvar=False)
    vif = np.diag(np.linalg.pinv(c))
    if np.linalg.matrix_rank(c) < len(cols):
        vif = np.where(vif > 1e6, np.inf, vif)
    return pd.DataFrame({"column": cols, "VIF": vif}).sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------- missing data
def missing_analysis(df: pd.DataFrame, max_cols: int = 40, max_targets: int = 12) -> dict:
    full = na_frame(df)
    d = _samp(df, 10_000)
    miss = full.loc[d.index]
    d = d.mask(miss)    # blanks become real NaN so every test below sees them as missing
    with_m = [c for c in d.columns if miss[c].any()]
    per = pd.DataFrame({"column": with_m, "missing": [int(full[c].sum()) for c in with_m],
                        "missing %": [full[c].mean() * 100 for c in with_m]}).sort_values("missing %", ascending=False).reset_index(drop=True)
    out = {"per_column": per, "limits": LIMITS["missing"]}
    if with_m:
        cols = with_m[:max_cols]
        vc = miss[cols].value_counts().head(10)
        out["patterns"] = pd.DataFrame([{"pattern": ", ".join(c for c, f in zip(cols, p) if f) or "complete rows",
                                         "rows": int(n), "share %": n / len(d) * 100} for p, n in vc.items()])
    else:
        out["patterns"] = pd.DataFrame(columns=["pattern", "rows", "share %"])
    co = []
    if len(with_m) >= 2:
        ind = miss[with_m[:max_cols]].astype(float)
        cm = ind.corr()
        for i, a in enumerate(cm.columns):
            for b in cm.columns[i + 1:]:
                if pd.notna(cm.loc[a, b]) and abs(cm.loc[a, b]) >= 0.3:
                    co.append({"column A": a, "column B": b, "co-missing (phi)": cm.loc[a, b]})
    out["co_missing"] = pd.DataFrame(co).sort_values("co-missing (phi)", key=abs, ascending=False).reset_index(drop=True) if co else pd.DataFrame(columns=["column A", "column B", "co-missing (phi)"])
    rel = []
    others_n = [c for c in d.columns if _num(d[c])][:max_cols]
    others_c = [c for c in d.columns if not _num(d[c]) and d[c].nunique() <= 30][:max_cols]
    for m in [c for c in per["column"] if 0.02 <= miss[c].mean() <= 0.98][:max_targets]:
        ind = miss[m]
        for o in others_n:
            if o == m:
                continue
            a, b = d.loc[ind, o].dropna(), d.loc[~ind, o].dropna()
            if len(a) > 5 and len(b) > 5 and (a.nunique() > 1 or b.nunique() > 1):
                u = stats.mannwhitneyu(a, b)
                rel.append({"column with gaps": m, "related column": o, "test": "Mann-Whitney", "effect (0-1)": abs(2 * u.statistic / (len(a) * len(b)) - 1), "p-value": u.pvalue})
        for o in others_c:
            if o == m:
                continue
            ct = contingency(ind, d[o].astype(str).where(d[o].notna()))
            if ct.shape[0] == 2 and ct.shape[1] >= 2:
                v, pv = chi2_cramers(ct)
                rel.append({"column with gaps": m, "related column": o, "test": "Chi-square", "effect (0-1)": v, "p-value": pv})
    R = pd.DataFrame(rel)
    if len(R):
        R["suggests dependence"] = (R["p-value"] < 0.01 / max(len(R), 1) * 5) & (R["effect (0-1)"] >= 0.1)
        R = R.sort_values(["suggests dependence", "effect (0-1)"], ascending=False).head(25).reset_index(drop=True)
    out["relations"] = R
    out["n_dependent"] = int(R["suggests dependence"].sum()) if len(R) else 0
    return out


# ----------------------------------------------------------------------------------- data quality
def quality_checks(df: pd.DataFrame, target=None) -> list[dict]:
    out = []
    add = lambda level, col, check, msg, n=0: out.append({"level": level, "column": col, "check": check, "message": msg, "count": int(n)})
    n = len(df)
    if int(df.duplicated().sum()):
        add("warning", None, "duplicate rows", f"{int(df.duplicated().sum())} duplicate row(s).", df.duplicated().sum())
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        add("warning", None, "empty rows", f"{empty_rows} row(s) are completely empty.", empty_rows)
    for c in df.columns:
        s = df[c]
        if s.isna().all():
            add("error", c, "empty column", f"`{c}` has no values.", n)
            continue
        if s.nunique(dropna=True) <= 1:
            add("warning", c, "constant", f"`{c}` has a single value.", n)
        if _num(s):
            k = int(np.isinf(s.astype(float)).sum())
            if k:
                add("error", c, "infinite values", f"`{c}` has {k} infinite value(s).", k)
            if re.search(r"age|price|cost|salary|income|amount|count|quantity|qty|weight|height|distance|duration", str(c), re.I):
                neg = int((s < 0).sum())
                if neg:
                    add("warning", c, "implausible negatives", f"`{c}` has {neg} negative value(s), which is unusual for this kind of column.", neg)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        t = s.dropna().astype(str)
        pad = int((t != t.str.strip()).sum())
        if pad:
            add("warning", c, "extra spaces", f"`{c}` has {pad} value(s) with spaces at the start or end.", pad)
        ph = t.str.strip().str.lower().isin(PLACEHOLDERS)
        if ph.any():
            add("error", c, "disguised missing values", f"`{c}` has {int(ph.sum())} placeholder value(s) such as {', '.join(t[ph].unique()[:3])} that probably mean 'missing'.", ph.sum())
        nm = pd.to_numeric(t.str.strip(), errors="coerce").notna().mean()
        if 0.05 < nm < 0.95:
            add("warning", c, "mixed types", f"`{c}` mixes numbers ({nm * 100:.0f}%) and text.", int(nm * len(t)))
        _, var = univariate_categorical(df[[c]], [c]) if t.nunique() <= 200 else (None, {})
        if c in var:
            add("warning", c, "inconsistent spelling", f"`{c}` has spellings that differ only in case or punctuation, e.g. {' / '.join(map(str, var[c][0][:3]))}.", len(var[c]))
        if t.nunique() > 50 and t.nunique() / max(len(t), 1) < 0.5:
            add("info", c, "many categories", f"`{c}` has {t.nunique()} categories; think about grouping the rare ones.", t.nunique())
        samp = t.sample(min(len(t), 200), random_state=0)
        pats = samp.str.replace(r"\d", "9", regex=True).str.replace(r"[A-Za-z]", "a", regex=True).value_counts(normalize=True)
        if samp.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.8 and len(pats) > 2 and pats.iloc[0] < 0.8:
            add("warning", c, "date formats", f"`{c}` looks like dates written in more than one format.", len(pats))
    return out


# --------------------------------------------------------------------------------- scale warnings
def correlation_note(r: float) -> str:
    a = abs(r)
    return "very weak" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.8 else "very strong"
