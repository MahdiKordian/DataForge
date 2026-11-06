# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Target Intelligence: analysis that fits the kind of problem. It describes statistical association, never causes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis as A
from .diff import _psi

KINDS = ["Classification", "Regression", "Time-series forecasting", "Unsupervised"]
CAUSAL = "This is a statistical association. It does not show that the feature causes the target to change."


def guess_kind(df: pd.DataFrame, target) -> str:
    if target is None or target not in df.columns:
        return "Unsupervised"
    s = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 15:
        return "Classification"
    return "Regression"


def _flag(level, msg):
    return {"level": level, "message": msg}


def _feature_table_class(df, target, num, cat) -> pd.DataFrame:
    y = df[target]
    rows = []
    for c in num:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        if d.y.nunique() < 2 or d.x.nunique() < 2:
            continue
        groups = [g.x.to_numpy() for _, g in d.groupby("y") if len(g) > 1]
        p = stats.kruskal(*groups)[1] if len(groups) >= 2 else np.nan
        rows.append(
            {
                "feature": c,
                "type": "numeric",
                "strength (0-1)": float(np.sqrt(A.eta_squared(d.y, d.x))),
                "measure": "eta (group differences)",
                "p-value": p,
                "NMI": A.nmi(d.x, d.y),
                "rows used": len(d),
            }
        )
    for c in cat:
        d = pd.DataFrame({"x": df[c].astype(str).where(df[c].notna()), "y": y}).dropna()
        if d.x.nunique() < 2 or d.y.nunique() < 2 or d.x.nunique() > 100:
            continue
        v, pv = A.chi2_cramers(A.contingency(d.x, d.y))
        rows.append(
            {
                "feature": c,
                "type": "categorical",
                "strength (0-1)": v,
                "measure": "Cramér's V",
                "p-value": pv,
                "NMI": A.nmi(d.x, d.y),
                "rows used": len(d),
            }
        )
    return _finish(rows, len(df))


def _finish(rows, n_total) -> pd.DataFrame:
    """Marks associations measured on very few rows as unreliable and lists them last."""
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["reliable"] = (out["rows used"] >= 30) & (out["rows used"] >= 0.05 * n_total)
    return out.sort_values(["reliable", "strength (0-1)"], ascending=False).reset_index(
        drop=True
    )


def _feature_table_reg(df, target, num, cat) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce")
    rows = []
    for c in num:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(d) < 10 or d.x.nunique() < 2:
            continue
        pr, sr = d.x.corr(d.y), d.x.corr(d.y, method="spearman")
        rows.append(
            {
                "feature": c,
                "type": "numeric",
                "Pearson": pr,
                "Spearman": sr,
                "NMI": A.nmi(d.x, d.y),
                "strength (0-1)": max(abs(pr), abs(sr)),
                "p-value": stats.spearmanr(d.x, d.y)[1],
                "rows used": len(d),
            }
        )
    for c in cat:
        d = pd.DataFrame({"x": df[c].astype(str).where(df[c].notna()), "y": y}).dropna()
        if d.x.nunique() < 2 or d.x.nunique() > 100:
            continue
        groups = [g.y.to_numpy() for _, g in d.groupby("x") if len(g) > 1]
        rows.append(
            {
                "feature": c,
                "type": "categorical",
                "Pearson": np.nan,
                "Spearman": np.nan,
                "NMI": A.nmi(d.x, d.y),
                "strength (0-1)": float(np.sqrt(A.eta_squared(d.x, d.y))),
                "p-value": stats.kruskal(*groups)[1] if len(groups) >= 2 else np.nan,
                "rows used": len(d),
            }
        )
    return _finish(rows, len(df))


def classification(df, target, types=None, splits=None) -> dict:
    num, cat, _ = A.split_cols(df, target, types)
    y = df[target].dropna()
    vc = y.value_counts()
    dist = pd.DataFrame(
        {
            "class": vc.index.astype(str),
            "rows": vc.values,
            "share %": vc.values / len(y) * 100,
        }
    )
    flags = []
    ratio = vc.iloc[0] / vc.iloc[-1] if len(vc) > 1 else np.inf
    if len(vc) < 2:
        flags.append(
            _flag("error", "The target has a single class, so nothing can be learned.")
        )
    if ratio >= 3:
        flags.append(
            _flag(
                "warning",
                f"Imbalanced: the largest class is {ratio:.1f} times the smallest. Plain accuracy can be misleading; "
                "use stratified splits and metrics such as recall, precision or PR-AUC.",
            )
        )
    small = vc[vc < 30]
    if len(small):
        flags.append(
            _flag(
                "warning",
                f"{len(small)} class(es) have fewer than 30 rows ({', '.join(map(str, small.index[:3]))}); estimates for them will be unstable.",
            )
        )
    feats = _feature_table_class(df, target, num, cat)
    low = [
        c
        for c in num + cat
        if df[c].nunique(dropna=True) <= 1
        or (
            df[c].value_counts(normalize=True).iloc[0] > 0.99
            if df[c].notna().any()
            else True
        )
    ]
    if low:
        flags.append(
            _flag(
                "info",
                f"{len(low)} feature(s) are almost constant (top value above 99%): {', '.join(map(str, low[:5]))}.",
            )
        )
    if len(feats):
        unrel = int((~feats["reliable"]).sum())
        if unrel:
            flags.append(
                _flag(
                    "info",
                    f"{unrel} feature(s) are measured on fewer than 30 rows (or under 5% of the data) and are marked unreliable.",
                )
            )
        for _, r in feats[
            (feats["strength (0-1)"] >= 0.9) & feats["reliable"]
        ].iterrows():
            flags.append(
                _flag(
                    "warning",
                    f"Possible leakage: `{r['feature']}` is almost perfectly linked to the target (strength {r['strength (0-1)']:.2f}). "
                    "Check that it is not derived from the target or only known after the outcome.",
                )
            )
    if splits:
        shares = {
            k: v[target].value_counts(normalize=True)
            for k, v in splits.items()
            if v is not None and target in v
        }
        if len(shares) >= 2:
            tb = pd.DataFrame(shares).fillna(0) * 100
            tb.index = tb.index.astype(str)
            if (tb.max(axis=1) - tb.min(axis=1)).max() > 5:
                flags.append(
                    _flag(
                        "warning",
                        "Class shares differ by more than 5 percentage points between the splits. Use a stratified split.",
                    )
                )
            splits_table = tb.reset_index().rename(columns={"index": "class"})
        else:
            splits_table = None
    else:
        splits_table = None
    return {
        "kind": "Classification",
        "distribution": dist,
        "imbalance_ratio": float(ratio),
        "features": feats,
        "flags": flags,
        "splits": splits_table,
        "note": CAUSAL,
    }


def regression(df, target, types=None) -> dict:
    num, cat, _ = A.split_cols(df, target, types)
    y = (
        pd.to_numeric(df[target], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    out_n = int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    summ = {
        "rows": len(y),
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "skew": y.skew(),
        "kurtosis": y.kurt(),
        "zeros %": (y == 0).mean() * 100,
        "negative %": (y < 0).mean() * 100,
        "outliers (IQR)": out_n,
    }
    flags = []
    if abs(summ["skew"]) > 1:
        tip = (
            " If all values are positive, a log transform of the target is worth testing."
            if y.min() > 0 and summ["skew"] > 0
            else ""
        )
        flags.append(
            _flag(
                "info", f"The target is strongly skewed (skew {summ['skew']:.2f}).{tip}"
            )
        )
    if out_n / max(len(y), 1) > 0.05:
        flags.append(
            _flag(
                "warning",
                f"{out_n / len(y) * 100:.1f}% of target values are outliers by the IQR rule.",
            )
        )
    feats = _feature_table_reg(df, target, num, cat)
    nl = (
        feats[
            (feats["type"] == "numeric")
            & (feats["NMI"] >= 0.15)
            & (feats["Pearson"].abs() < 0.3)
            & feats["reliable"]
        ]
        if len(feats)
        else feats
    )
    if len(nl):
        flags.append(
            _flag(
                "info",
                f"{len(nl)} feature(s) look related to the target but not linearly (high NMI, low Pearson): {', '.join(nl['feature'].head(5))}. "
                "A linear model would miss them.",
            )
        )
    if len(feats):
        unrel = int((~feats["reliable"]).sum())
        if unrel:
            flags.append(
                _flag(
                    "info",
                    f"{unrel} feature(s) are measured on fewer than 30 rows (or under 5% of the data) and are marked unreliable.",
                )
            )
        for _, r in feats[
            (feats["Pearson"].abs() >= 0.98) & feats["reliable"]
        ].iterrows():
            flags.append(
                _flag(
                    "warning",
                    f"Possible leakage: `{r['feature']}` has correlation {r['Pearson']:.3f} with the target.",
                )
            )
    mad = (y - y.median()).abs().median()
    rz = 0.6745 * (y - y.median()) / mad if mad > 0 else pd.Series(0.0, index=y.index)
    extreme = (
        df.loc[rz[rz.abs() > 3.5].index, [target]]
        .assign(robust_z=rz[rz.abs() > 3.5])
        .sort_values("robust_z", key=abs, ascending=False)
        .head(10)
    )
    lev = pd.DataFrame()
    nc = [c for c in num if df[c].nunique() > 2][:30]
    if len(nc) >= 2:
        X = A.as_numeric(df, nc).replace([np.inf, -np.inf], np.nan).dropna()
        X = X.sample(min(len(X), 20_000), random_state=0)
        if len(X) > len(nc) + 2:
            Z = ((X - X.mean()) / X.std().replace(0, 1)).to_numpy(float)
            Z = np.column_stack([np.ones(len(Z)), Z])
            h = np.einsum("ij,ji->i", Z @ np.linalg.pinv(Z.T @ Z), Z.T)
            thr = 3 * Z.shape[1] / len(Z)
            lev = (
                pd.DataFrame({"row": X.index, "leverage": h})
                .query("leverage > @thr")
                .sort_values("leverage", ascending=False)
                .head(10)
            )
    if len(lev):
        flags.append(
            _flag(
                "info",
                f"{len(lev)} row(s) have unusually extreme feature combinations (high leverage) and may pull a fitted model strongly.",
            )
        )
    return {
        "kind": "Regression",
        "summary": pd.Series(summ),
        "features": feats,
        "flags": flags,
        "extreme_targets": extreme,
        "high_leverage": lev,
        "note": CAUSAL,
    }


def timeseries(df, target, time_col, types=None) -> dict:
    if time_col is None or time_col not in df.columns:
        return {
            "kind": "Time-series forecasting",
            "flags": [_flag("error", "Choose the date column.")],
            "note": CAUSAL,
        }
    t = pd.to_datetime(df[time_col], errors="coerce")
    d = (
        pd.DataFrame({"t": t, "y": pd.to_numeric(df[target], errors="coerce")})
        .dropna()
        .sort_values("t")
    )
    flags = []
    if len(d) < 20:
        return {
            "kind": "Time-series forecasting",
            "flags": [_flag("error", "Too few dated rows (need at least 20).")],
            "note": CAUSAL,
        }
    gaps = d.t.diff().dropna()
    med = gaps.median()
    irregular = int((gaps > 3 * med).sum()) if med > pd.Timedelta(0) else 0
    rho, p = stats.spearmanr(np.arange(len(d)), d.y)
    half = len(d) // 2
    a, b = d.y.iloc[:half], d.y.iloc[half:]
    lags = []
    idx = np.arange(len(d))
    yy = d.y.to_numpy() - np.polyval(np.polyfit(idx, d.y.to_numpy(), 1), idx)
    denom = float((yy**2).sum())
    for k in range(1, min(len(yy) // 2, 60) + 1):
        lags.append(
            {
                "lag": k,
                "autocorrelation": float((yy[:-k] * yy[k:]).sum() / denom)
                if denom
                else np.nan,
            }
        )
    acf = pd.DataFrame(lags)
    peaks = acf[
        (acf.autocorrelation > 0.3)
        & (acf.autocorrelation >= acf.autocorrelation.shift(1))
        & (acf.autocorrelation >= acf.autocorrelation.shift(-1).fillna(-1))
    ]
    psi = _psi(a, b)
    if irregular:
        flags.append(
            _flag(
                "warning",
                f"{irregular} gap(s) are more than 3 times the usual spacing ({med}). Many models assume regular steps.",
            )
        )
    if int(d.t.duplicated().sum()):
        flags.append(
            _flag(
                "warning",
                f"{int(d.t.duplicated().sum())} timestamp(s) appear more than once.",
            )
        )
    if p < 0.01 and abs(rho) > 0.2:
        flags.append(
            _flag(
                "info",
                f"There is a {'rising' if rho > 0 else 'falling'} trend (Spearman {rho:+.2f}). Differencing or detrending may help.",
            )
        )
    if abs(b.mean() - a.mean()) > 0.5 * (a.std() or 1):
        flags.append(
            _flag(
                "warning",
                f"The mean of the second half differs from the first by {(b.mean() - a.mean()) / (a.std() or 1):+.2f} standard deviations: the series is probably not stationary.",
            )
        )
    if not np.isnan(psi) and psi > 0.25:
        flags.append(
            _flag(
                "warning",
                f"The target distribution shifted a lot over time (PSI {psi:.2f}). Models trained on the past may not suit the future.",
            )
        )
    flags.append(
        _flag(
            "info",
            "Split by time, not at random: train on the earliest part, then validate and test on later parts. "
            "A random split lets the model see the future.",
        )
    )
    flags.append(
        _flag(
            "info",
            "Lag and rolling features must use only past values. Features computed from the full series leak the future.",
        )
    )
    summary = {
        "rows": len(d),
        "first": d.t.iloc[0],
        "last": d.t.iloc[-1],
        "median step": med,
        "irregular gaps": irregular,
        "trend (Spearman)": rho,
        "trend p-value": p,
        "PSI first vs second half": psi,
    }
    return {
        "kind": "Time-series forecasting",
        "summary": pd.Series(summary, dtype=object),
        "acf": acf,
        "seasonal_lags": [int(x) for x in peaks["lag"].head(5)],
        "flags": flags,
        "series": d.rename(columns={"t": "time", "y": target}),
        "note": CAUSAL,
    }


def unsupervised(df, types=None) -> dict:
    num, cat, _ = A.split_cols(df, None, types)
    flags = []
    X = (
        A.as_numeric(df, [c for c in num if df[c].nunique() > 2][:60])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    res = {
        "kind": "Unsupervised",
        "flags": flags,
        "note": "Without a target there is nothing to validate against; judge results by stability and by whether they make sense.",
    }
    if X.shape[1] >= 2 and len(X) > X.shape[1]:
        sd = X.std()
        if sd.max() / max(sd[sd > 0].min(), 1e-12) > 10:
            flags.append(
                _flag(
                    "warning",
                    "Columns have very different scales. Distance-based methods (k-means, PCA) will be dominated by the largest. Scale first.",
                )
            )
        Z = ((X - X.mean()) / sd.replace(0, 1)).to_numpy(float)
        sv = np.linalg.svd(Z, compute_uv=False) ** 2
        ev = sv / sv.sum()
        res["pca"] = pd.DataFrame(
            {
                "component": range(1, len(ev) + 1),
                "explained %": ev * 100,
                "cumulative %": np.cumsum(ev) * 100,
            }
        ).head(15)
        k90 = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
        flags.append(
            _flag(
                "info", f"{k90} of {X.shape[1]} components explain 90% of the variance."
            )
        )
        hi = A.top_pairs(
            pd.DataFrame(
                np.corrcoef(Z, rowvar=False), index=X.columns, columns=X.columns
            ),
            5,
            0.9,
        )
        if len(hi):
            flags.append(
                _flag(
                    "info",
                    f"{len(hi)} pair(s) of columns are almost duplicates (|r| above 0.9); they count twice in distance-based methods.",
                )
            )
    if len(cat):
        flags.append(
            _flag(
                "info",
                f"{len(cat)} categorical column(s) need encoding before distance-based methods.",
            )
        )
    return res


def analyze_target(df, target, kind, time_col=None, types=None, splits=None) -> dict:
    d = df if len(df) <= 100_000 else df.sample(100_000, random_state=0)
    if kind == "Classification":
        return classification(d, target, types, splits)
    if kind == "Regression":
        return regression(d, target, types)
    if kind.startswith("Time"):
        return timeseries(d, target, time_col, types)
    return unsupervised(d, types)
