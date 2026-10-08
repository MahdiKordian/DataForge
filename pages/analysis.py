import html
from itertools import combinations
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats as sps
except Exception:
    sps = None

alt.data_transformers.disable_max_rows()

PAGES = {"dataset": "pages/dataset.py"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&display=swap');
.block-container{max-width:1180px;padding-top:2.5rem}
.dfd{font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.dfd h1{font-size:44px;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.dfd h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:30px 0 10px;padding:0}
.dfd .mono{font-family:'IBM Plex Mono',monospace;font-size:13px;opacity:.65;margin-top:6px}
</style>
"""


def go(page):
    try:
        st.switch_page(PAGES[page])
    except Exception:
        st.toast(f"Page not found: {PAGES[page]}")


def section(t):
    st.markdown(f'<div class="dfd"><h2>{t}</h2></div>', unsafe_allow_html=True)


def show(chart):
    st.altair_chart(chart, use_container_width=True)


def nums(df):
    return list(df.select_dtypes("number").columns)


def cats(df):
    n = set(nums(df))
    return [c for c in df.columns if c not in n and not pd.api.types.is_datetime64_any_dtype(df[c])]


def sample(d, n=5000):
    return d.sample(n, random_state=0) if len(d) > n else d


def is_class(s):
    return not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 20


def heatmap(m, lo=-1, hi=1, scheme="blueorange"):
    m = m.copy()
    m.columns, m.index = m.columns.astype(str), m.index.astype(str)
    long = m.stack().reset_index()
    long.columns = ["x", "y", "v"]
    base = alt.Chart(long).encode(x=alt.X("x:N", sort=list(m.columns), title=None),
                                  y=alt.Y("y:N", sort=list(m.index), title=None))
    out = base.mark_rect().encode(
        color=alt.Color("v:Q", scale=alt.Scale(domain=[lo, hi], scheme=scheme), legend=None),
        tooltip=["x", "y", alt.Tooltip("v:Q", format=".2f")])
    if len(m) <= 12:
        out = out + base.mark_text(fontSize=11).encode(text=alt.Text("v:Q", format=".2f"))
    return out.properties(height=max(220, 34 * len(m)))


def bars(s, title="", horizontal=True):
    d = s.reset_index()
    d.columns = ["k", "v"]
    d["k"] = d["k"].astype(str)
    ch = alt.Chart(d)
    if horizontal:
        return ch.mark_bar().encode(y=alt.Y("k:N", sort="-x", title=None), x=alt.X("v:Q", title=title),
                                    tooltip=["k", alt.Tooltip("v:Q", format=",.2f")])
    return ch.mark_bar().encode(x=alt.X("k:N", sort=None, title=None), y=alt.Y("v:Q", title=title),
                                tooltip=["k", alt.Tooltip("v:Q", format=",.2f")])


def pairs(m):
    rows = [(a, b, m.loc[a, b]) for a, b in combinations(m.columns, 2) if pd.notna(m.loc[a, b])]
    d = pd.DataFrame(rows, columns=["column A", "column B", "correlation"])
    return d.reindex(d.correlation.abs().sort_values(ascending=False).index)


def cramers(x, y):
    ct = pd.crosstab(x, y)
    n = ct.values.sum()
    r, k = ct.shape
    if n == 0 or min(r, k) < 2:
        return 0.0
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = ((ct.values - exp) ** 2 / exp).sum()
    return float(np.sqrt(chi2 / n / (min(r, k) - 1)))


def eta(g, v):
    """Correlation ratio: how much of numeric v is explained by categorical g (0 to 1)."""
    total = ((v - v.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    grp = v.groupby(g)
    return float(np.sqrt((grp.count() * (grp.mean() - v.mean()) ** 2).sum() / total))


def vif(df, cols):
    X = df[cols].dropna()
    X = X.loc[:, X.std() > 0]
    if X.shape[1] < 2 or len(X) <= X.shape[1] + 1:
        return None
    Z = ((X - X.mean()) / X.std()).to_numpy()
    out = {}
    for j, c in enumerate(X.columns):
        y, A = Z[:, j], np.delete(Z, j, 1)
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        r2 = 1 - ((y - A @ beta) ** 2).sum() / (y ** 2).sum()
        out[c] = np.inf if r2 >= 0.9999 else 1 / (1 - r2)
    return pd.Series(out, name="VIF").sort_values(ascending=False)


def outlier_mask(s, method):
    if method.startswith("IQR"):
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    elif method.startswith("Z"):
        lo, hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
    else:
        med = s.median()
        mad = (s - med).abs().median()
        lo = hi = med
        if mad > 0:
            lo, hi = med - 3.5 * mad / 0.6745, med + 3.5 * mad / 0.6745
    return (s < lo) | (s > hi), lo, hi


def view_overview(df):
    n, c = nums(df), cats(df)
    m = st.columns(6)
    m[0].metric("Rows", f"{len(df):,}")
    m[1].metric("Columns", df.shape[1])
    m[2].metric("Numeric", len(n))
    m[3].metric("Categorical", len(c))
    m[4].metric("Duplicates", int(df.duplicated().sum()))
    m[5].metric("Missing cells", f"{df.isna().sum().sum() / max(df.size, 1) * 100:.1f}%")

    section("Key findings")
    notes = []
    for col, p in (df.isna().mean() * 100).items():
        if p > 30:
            notes.append(f"`{col}` is {p:.0f}% missing. Consider dropping or imputing it.")
    for col in n:
        s = df[col].dropna()
        if len(s) > 2 and abs(s.skew()) > 1:
            notes.append(f"`{col}` is skewed (skew {s.skew():.1f}). A log or power transform may help.")
    if len(n) > 1:
        for _, r in pairs(df[n].corr()).head(3).iterrows():
            if abs(r.correlation) > 0.9:
                notes.append(f"`{r['column A']}` and `{r['column B']}` are almost redundant (r = {r.correlation:.2f}).")
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            notes.append(f"`{col}` has a single value and carries no information.")
    for col in c:
        if df[col].nunique() > 20 and df[col].nunique() / len(df) > 0.5:
            notes.append(f"`{col}` has very high cardinality and is hard to encode.")
    t = st.session_state.target
    if t in df.columns and is_class(df[t].dropna()) and len(df[t].dropna()):
        share = df[t].value_counts(normalize=True)
        if share.min() < 0.2:
            notes.append(f"Target `{t}` is imbalanced (smallest class {share.min() * 100:.0f}%).")
    if notes:
        st.markdown("\n".join(f"- {x}" for x in notes[:12]))
    else:
        st.success("Nothing alarming found.")

    if n:
        section("Numeric summary")
        st.dataframe(df[n].describe().T, use_container_width=True)
    if c:
        section("Categorical summary")
        d = df[c].describe().T
        d["top"] = d["top"].astype(str)
        st.dataframe(d, use_container_width=True)


def view_distribution(df):
    col = st.selectbox("Column", list(df.columns))
    s = df[col].dropna()
    if s.empty:
        st.info("This column is empty.")
        return
    if col in nums(df):
        x = s if len(s) <= 20000 else s.sample(20000, random_state=0)
        d = pd.DataFrame({"x": x})
        show(alt.Chart(d).mark_bar().encode(alt.X("x:Q", bin=alt.Bin(maxbins=30), title=col), alt.Y("count()")))
        show(alt.Chart(d).mark_boxplot(extent=1.5).encode(alt.X("x:Q", title=col)).properties(height=90))
        m = st.columns(5)
        for box, (k, v) in zip(m, [("Mean", s.mean()), ("Median", s.median()), ("Std", s.std()),
                                   ("Skew", s.skew()), ("Kurtosis", s.kurt())]):
            box.metric(k, f"{v:,.3g}")
        if sps is not None and len(s) >= 8 and s.nunique() > 1:
            p = sps.shapiro(s.sample(min(len(s), 5000), random_state=0))[1]
            st.caption(f"Shapiro-Wilk p = {p:.4f}. "
                       + ("Looks roughly normal." if p > 0.05 else "Significantly different from normal."))
    else:
        show(bars(s.value_counts().head(20), "count"))
        st.caption(f"{s.nunique()} unique values. Showing the top 20.")


def view_missing(df):
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("No missing values.")
        return
    show(bars(miss, "missing %"))
    per_row = df.isna().sum(axis=1)
    m = st.columns(3)
    m[0].metric("Rows with any missing", f"{(per_row > 0).mean() * 100:.1f}%")
    m[1].metric("Rows missing over half", f"{(per_row > df.shape[1] / 2).sum():,}")
    m[2].metric("Complete rows", f"{(per_row == 0).sum():,}")
    if len(miss) >= 2:
        section("Do columns go missing together?")
        show(heatmap(df[list(miss.index)].isna().astype(int).corr().fillna(0)))
        st.caption("Values near 1 mean the two columns tend to be empty on the same rows.")


def view_corr(df):
    n = nums(df)
    if len(n) < 2:
        st.info("Correlation needs at least two numeric columns.")
        return
    method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
    d = sample(df[n], 3000) if method == "kendall" else df[n]
    m = d.corr(method=method)
    show(heatmap(m))
    section("Strongest pairs")
    thr = st.slider("Flag pairs above |r|", 0.5, 1.0, 0.9, 0.05)
    p = pairs(m)
    flagged = p[p.correlation.abs() >= thr]
    if flagged.empty:
        st.success(f"No pair above {thr}.")
    else:
        st.warning(f"{len(flagged)} pair(s) above {thr}. Consider dropping one column from each.")
    st.dataframe(p.head(15), hide_index=True, use_container_width=True,
                 column_config={"correlation": st.column_config.NumberColumn(format="%.3f")})
    v = vif(df, n)
    if v is not None:
        section("Multicollinearity (VIF)")
        st.dataframe(v.to_frame(), use_container_width=True)
        st.caption("Above 5 is moderate and above 10 is high. Based on rows without missing values.")


def view_outliers(df):
    n = nums(df)
    if not n:
        st.info("No numeric columns.")
        return
    method = st.radio("Method", ["IQR (1.5 x IQR)", "Z-score (|z| > 3)", "Modified Z-score (MAD)"], horizontal=True)
    rows, any_mask = [], pd.Series(False, index=df.index)
    for c in n:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        mask, lo, hi = outlier_mask(s, method)
        any_mask.loc[s.index] |= mask
        rows.append({"column": c, "outliers": int(mask.sum()), "outliers %": mask.mean() * 100,
                     "lower bound": lo, "upper bound": hi})
    t = pd.DataFrame(rows)
    st.metric("Rows with at least one outlier", f"{any_mask.mean() * 100:.1f}%", f"{int(any_mask.sum()):,} rows", delta_color="off")
    if t.empty:
        return
    show(bars(t.set_index("column")["outliers %"], "outliers %"))
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "outliers %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})


def view_categorical(df):
    c = cats(df)
    if not c:
        st.info("No categorical columns.")
        return
    info = pd.DataFrame([{"column": x, "unique": df[x].nunique(),
                          "top value": str(df[x].mode().iloc[0]) if df[x].notna().any() else "-",
                          "top %": df[x].value_counts(normalize=True).max() * 100 if df[x].notna().any() else 0,
                          "rare categories (<1%)": int((df[x].value_counts(normalize=True) < 0.01).sum())}
                         for x in c])
    st.dataframe(info, hide_index=True, use_container_width=True,
                 column_config={"top %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})
    col = st.selectbox("Column", c)
    show(bars(df[col].value_counts().head(20), "count"))
    ok = [x for x in c if 2 <= df[x].nunique() <= 50][:12]
    if len(ok) >= 2:
        section("Association between categorical columns (Cramér's V)")
        m = pd.DataFrame(1.0, index=ok, columns=ok)
        for a, b in combinations(ok, 2):
            d = df[[a, b]].dropna()
            m.loc[a, b] = m.loc[b, a] = cramers(d[a], d[b])
        show(heatmap(m, 0, 1, "blues"))


def view_target(df):
    t = st.session_state.target
    if t not in df.columns:
        st.info("Choose a target column on the Dashboard or Dataset page to unlock this view.")
        return
    y = df[t].dropna()
    clf = is_class(y)
    N = set(nums(df))
    st.markdown(f"Target **{t}**, treated as **{'classification' if clf else 'regression'}**.")
    if clf:
        vc = y.value_counts()
        show(bars(vc, "count", horizontal=False))
        st.caption(f"{len(vc)} classes. Imbalance ratio {vc.max() / max(vc.min(), 1):.1f} : 1.")
    else:
        show(alt.Chart(pd.DataFrame({"x": y})).mark_bar().encode(
            alt.X("x:Q", bin=alt.Bin(maxbins=30), title=t), alt.Y("count()")))
    section("Which columns relate to the target?")
    rows = []
    for x in df.columns:
        if x == t:
            continue
        d = df[[x, t]].dropna()
        if len(d) < 5:
            continue
        if x in N and t in N:
            v, m = abs(d[x].corr(d[t])), "|Pearson|"
        elif x in N:
            v, m = eta(d[t], d[x]), "Correlation ratio"
        elif t in N:
            v, m = eta(d[x], d[t]), "Correlation ratio"
        else:
            v, m = cramers(d[x], d[t]), "Cramér's V"
        rows.append({"column": x, "strength": 0.0 if pd.isna(v) else v, "measure": m})
    r = pd.DataFrame(rows).sort_values("strength", ascending=False)
    if r.empty:
        return
    show(bars(r.set_index("column")["strength"], "strength (0 to 1)"))
    st.caption("Strength is a quick screen. Different measures are only roughly comparable.")
    section("Look closer")
    f = st.selectbox("Column", list(r["column"]))
    d = sample(df[[f, t]].dropna(), 3000)
    d.columns = ["f", "t"]
    top = d["f"].value_counts().head(15).index
    if clf and f in N:
        d["t"] = d["t"].astype(str)
        show(alt.Chart(d).mark_boxplot().encode(alt.X("t:N", title=t), alt.Y("f:Q", title=f), alt.Color("t:N", legend=None)))
    elif clf:
        ct = pd.crosstab(d["f"][d["f"].isin(top)], d["t"].astype(str), normalize="index").stack().reset_index()
        ct.columns = ["f", "t", "share"]
        show(alt.Chart(ct).mark_bar().encode(alt.X("f:N", title=f), alt.Y("share:Q", stack="normalize"), alt.Color("t:N", title=t)))
    elif f in N:
        pts = alt.Chart(d).mark_circle(opacity=0.4).encode(alt.X("f:Q", title=f), alt.Y("t:Q", title=t))
        show(pts + pts.transform_regression("f", "t").mark_line(color="#e8960c"))
    else:
        d = d[d["f"].isin(top)]
        show(alt.Chart(d).mark_boxplot().encode(alt.X("f:N", title=f), alt.Y("t:Q", title=t)))


def view_pca(df):
    n = [c for c in nums(df) if df[c].std() > 0]
    if len(n) < 3:
        st.info("PCA needs at least three numeric columns that vary.")
        return
    X = df[n].fillna(df[n].median())
    Z = ((X - X.mean()) / X.std()).to_numpy()
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    st.caption("Numeric columns are filled with the median and standardized first. This view does not change your data.")
    show(bars(pd.Series(ev[:10] * 100, index=[f"PC{i + 1}" for i in range(min(10, len(ev)))]), "variance explained %", horizontal=False))
    st.markdown(f"The first two components explain **{ev[:2].sum() * 100:.0f}%** of the variance.")
    d = pd.DataFrame({"x": U[:, 0] * S[0], "y": U[:, 1] * S[1]})
    t = st.session_state.target
    enc = [alt.X("x:Q", title="PC1"), alt.Y("y:Q", title="PC2")]
    if t in df.columns:
        d["g"] = df[t].values
        d = d.dropna()
        enc.append(alt.Color("g:N", title=t) if is_class(d["g"]) else alt.Color("g:Q", title=t))
    show(alt.Chart(sample(d)).mark_circle(opacity=0.55, size=40).encode(*enc))
    st.caption("Loadings: how much each column contributes to PC1 and PC2.")
    load = pd.DataFrame(Vt[:2].T, index=n, columns=["PC1", "PC2"])
    st.dataframe(load.reindex(load.PC1.abs().sort_values(ascending=False).index).head(10), use_container_width=True)


def view_compare(df):
    snaps = st.session_state.snapshots
    if not snaps:
        st.info("No snapshots yet. They are created when a dataset is loaded.")
        return
    i = st.selectbox("Compare current data with", range(len(snaps)), format_func=lambda k: snaps[k]["name"])
    base = snaps[i]["df"]

    def facts(d):
        return {"Rows": len(d), "Columns": d.shape[1], "Missing cells": int(d.isna().sum().sum()),
                "Duplicate rows": int(d.duplicated().sum()), "Numeric columns": len(nums(d)),
                "Categorical columns": len(cats(d))}

    t = pd.DataFrame({"before": facts(base), "after": facts(df)})
    t["change"] = t["after"] - t["before"]
    st.dataframe(t, use_container_width=True)
    added, removed = [c for c in df.columns if c not in base.columns], [c for c in base.columns if c not in df.columns]
    if added:
        st.markdown("**Added columns:** " + ", ".join(f"`{c}`" for c in added))
    if removed:
        st.markdown("**Removed columns:** " + ", ".join(f"`{c}`" for c in removed))
    common = [c for c in nums(df) if c in nums(base)]
    if common:
        section("Numeric columns, before and after")
        st.dataframe(pd.DataFrame({
            "mean before": base[common].mean(), "mean after": df[common].mean(),
            "std before": base[common].std(), "std after": df[common].std(),
            "missing before": base[common].isna().sum(), "missing after": df[common].isna().sum()}),
            use_container_width=True)
    if st.session_state.log:
        section("Steps so far")
        st.dataframe(pd.DataFrame(st.session_state.log)[["time", "step", "detail"]], hide_index=True, use_container_width=True)


VIEWS = {"Overview": view_overview, "Distributions": view_distribution, "Missing": view_missing,
         "Correlations": view_corr, "Outliers": view_outliers, "Categorical": view_categorical,
         "Target": view_target, "PCA": view_pca, "Before vs after": view_compare}

ss = st.session_state
for k, v in {"df": None, "df_name": "untitled", "target": None, "log": [], "snapshots": []}.items():
    ss.setdefault(k, v)
st.markdown(CSS, unsafe_allow_html=True)
df = ss.df

if df is None:
    st.markdown('<div class="dfd"><h1>Analysis</h1></div>', unsafe_allow_html=True)
    st.info("No dataset yet. Add one first and the analysis will appear here.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

st.markdown(
    f'<div class="dfd"><h1>Analysis</h1><div class="mono">{html.escape(str(ss.df_name))} &middot; '
    f'{len(df):,} rows &middot; {df.shape[1]} columns'
    f'{" &middot; target: " + html.escape(str(ss.target)) if ss.target else ""}</div></div>',
    unsafe_allow_html=True)
st.write("")
view = st.radio("Method", list(VIEWS), horizontal=True, label_visibility="collapsed")
VIEWS[view](df)