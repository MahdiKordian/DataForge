# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import altair as alt
import pandas as pd
import streamlit as st

from core import go, workflow_bar
from dataforge_engine import analysis as A, target as TG
from dataforge_engine.types import infer_types

ss = st.session_state
for k_, v_ in {
    "df": None,
    "target": None,
    "log": [],
    "snapshots": [],
    "split_done": False,
    "var_types": None,
}.items():
    ss.setdefault(k_, v_)

workflow_bar("Analyze")
st.title("Insights")
st.caption("Statistics that say what they measure and where they are weak.")

if ss.df is None:
    st.info("No dataset yet. Add one first.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

df = ss.df
if ss.split_done and ss.get("n_train"):
    pick = st.radio(
        "Data to analyse",
        ["Training set (recommended)", "Whole current data"],
        horizontal=True,
        help="Decisions about preparation should be based on the training set only, so nothing is learned from the test set.",
    )
    if pick.startswith("Training"):
        df = ss.df.iloc[: ss.n_train]
raw_like = not ss.get("pipeline_cfg")
types = (
    (ss.var_types or {c: v["type"] for c, v in infer_types(df, ss.target).items()})
    if raw_like
    else None
)
target = ss.target if ss.target in df.columns else None
num, cat, dt = A.split_cols(df, target, types)
st.caption(
    f"{len(df):,} rows · {len(num)} numeric, {len(cat)} categorical, {len(dt)} date column(s)"
    + (" · using your confirmed variable types" if ss.var_types and raw_like else "")
)
if len(num) + len(cat) > 80:
    st.caption(
        "Wide data: relationship tables use the 30 to 40 columns with the most values, on a sample of up to 20,000 rows."
    )


def heat(M, lo=-1, hi=1):
    long = (
        M.rename_axis("a").reset_index().melt(id_vars="a", var_name="b", value_name="v")
    )
    st.altair_chart(
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("b:N", title=None, sort=list(M.columns)),
            y=alt.Y("a:N", title=None, sort=list(M.index)),
            color=alt.Color(
                "v:Q", scale=alt.Scale(scheme="blueorange", domain=[lo, hi]), title=None
            ),
            tooltip=["a", "b", alt.Tooltip("v:Q", format=".2f")],
        )
        .properties(height=min(18 * len(M) + 40, 640)),
        width="stretch",
    )


@st.cache_data(show_spinner="Calculating...", max_entries=6)
def c_uni_num(d, cols):
    return A.univariate_numeric(d, cols)


@st.cache_data(show_spinner="Calculating...", max_entries=6)
def c_uni_cat(d, cols):
    return A.univariate_categorical(d, cols)


@st.cache_data(show_spinner="Calculating...", max_entries=6)
def c_corr(d, cols):
    return A.correlations(d, cols)


@st.cache_data(
    show_spinner="Calculating (this can take a few seconds)...", max_entries=4
)
def c_mixed(d, n, c):
    return A.mixed_associations(d, n, c)


@st.cache_data(show_spinner="Calculating...", max_entries=4)
def c_nl(d, cols):
    return A.nonlinear_candidates(d, cols)


@st.cache_data(show_spinner="Calculating...", max_entries=4)
def c_vif(d, cols):
    return A.vif_table(d, cols)


@st.cache_data(show_spinner="Looking at the gaps...", max_entries=4)
def c_missing(d):
    return A.missing_analysis(d)


@st.cache_data(show_spinner="Checking quality...", max_entries=4)
def c_quality(d, t):
    return A.quality_checks(d, t)


tabs = st.tabs(
    [
        "Univariate",
        "Relationships",
        "Missing data",
        "Data quality",
        "Target intelligence",
    ]
)

with tabs[0]:
    kind = st.radio(
        "Columns", ["Numeric", "Categorical", "Dates"], horizontal=True, key="uni_kind"
    )
    if kind == "Numeric":
        if not num:
            st.info("No numeric columns.")
        else:
            T = c_uni_num(df, num[:100])
            st.dataframe(T, hide_index=True, width="stretch")
            st.caption(A.LIMITS["normality"])
            col = st.selectbox("Look at one column", num, key="uni_num_col")
            x = A.as_numeric(df, [col])[col].dropna()
            if len(x):
                st.altair_chart(
                    alt.Chart(
                        pd.DataFrame({col: x.sample(min(len(x), 5000), random_state=0)})
                    )
                    .mark_bar()
                    .encode(alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40)), y="count()")
                    .properties(height=220),
                    width="stretch",
                )
    elif kind == "Categorical":
        if not cat:
            st.info("No categorical columns.")
        else:
            T, var = c_uni_cat(df, cat[:100])
            st.dataframe(T, hide_index=True, width="stretch")
            for c, groups in var.items():
                st.warning(
                    f"`{c}`: these spellings probably mean the same thing: "
                    + "; ".join(" / ".join(map(str, g[:4])) for g in groups[:3])
                )
            col = st.selectbox("Look at one column", cat, key="uni_cat_col")
            vc = (
                df[col]
                .astype(str)
                .where(df[col].notna())
                .value_counts()
                .head(20)
                .rename_axis("value")
                .reset_index(name="rows")
            )
            st.altair_chart(
                alt.Chart(vc)
                .mark_bar()
                .encode(x="rows:Q", y=alt.Y("value:N", sort="-x"))
                .properties(height=min(24 * len(vc) + 30, 500)),
                width="stretch",
            )
    else:
        cands = dt or [
            c
            for c in df.columns
            if df[c].dtype == object
            and df[c].astype(str).str.contains(r"\d{4}[-/]\d{1,2}", regex=True).mean()
            > 0.5
        ]
        if not cands:
            st.info(
                "No date columns found. Set a column's type to datetime on the Variable types page."
            )
        else:
            col = st.selectbox("Date column", cands, key="uni_dt_col")
            P = A.time_profile(df, col, num[:8])
            if "note" in P:
                st.info(P["note"])
            else:
                m = st.columns(4)
                m[0].metric("First", str(P["first"].date()))
                m[1].metric("Last", str(P["last"].date()))
                m[2].metric("Irregular gaps", P["irregular gaps"])
                m[3].metric("Empty periods", P["empty periods"])
                per = P["periods"].rename_axis("period").reset_index(name="rows")
                per["period"] = per["period"].astype(str)
                st.altair_chart(
                    alt.Chart(per)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "period:N", sort=None, axis=alt.Axis(labelOverlap=True)
                        ),
                        y="rows:Q",
                    )
                    .properties(height=200),
                    width="stretch",
                )
                if P["unparseable"]:
                    st.warning(
                        f"{P['unparseable']} value(s) could not be read as dates."
                    )
                if len(P["trends"]):
                    st.dataframe(P["trends"], hide_index=True, width="stretch")

with tabs[1]:
    view = st.radio(
        "View",
        [
            "Numeric correlations",
            "All columns together",
            "Non-linear candidates",
            "Collinearity (VIF)",
        ],
        horizontal=True,
        key="rel_view",
    )
    if view == "Numeric correlations":
        if len(num) < 2:
            st.info("Need at least two numeric columns.")
        else:
            C = c_corr(df, num)
            if C["capped"]:
                st.caption(
                    f"Showing the {len(C['columns'])} numeric columns with the most values."
                )
            which = st.radio("Method", ["Pearson", "Spearman"], horizontal=True)
            M = C["pearson" if which == "Pearson" else "spearman"]
            heat(M)
            st.dataframe(A.top_pairs(M, 15), hide_index=True, width="stretch")
            st.caption(A.LIMITS["pearson" if which == "Pearson" else "spearman"])
    elif view == "All columns together":
        if len(num) + len(cat) < 2:
            st.info("Need at least two columns.")
        else:
            M = c_mixed(df, num, cat)
            heat(M, 0, 1)
            st.dataframe(A.top_pairs(M, 15), hide_index=True, width="stretch")
            st.caption(
                "Each cell is 0 to 1: |Spearman| for two numeric columns, Cramér's V for two categorical, "
                "and the square root of eta² for one of each. "
                + A.LIMITS["cramers_v"]
                + " "
                + A.LIMITS["eta"]
            )
            hi = A.top_pairs(M, 50, 0.95)
            if len(hi):
                st.warning(
                    f"{len(hi)} pair(s) are almost the same information (0.95 or more). Consider dropping one of each pair."
                )
    elif view == "Non-linear candidates":
        if len(num) < 2:
            st.info("Need at least two numeric columns.")
        else:
            R = c_nl(df, num)
            st.dataframe(R, hide_index=True, width="stretch") if len(R) else st.success(
                "No pair looks related without also being linear."
            )
            st.caption(
                "Pairs with high mutual information but low Pearson correlation. "
                + A.LIMITS["nmi"]
            )
    else:
        V = c_vif(df, num)
        if not len(V):
            st.info("Not enough complete rows or numeric columns.")
        else:
            st.dataframe(V, hide_index=True, width="stretch")
            st.caption(A.LIMITS["vif"])

with tabs[2]:
    R = c_missing(df)
    if not len(R["per_column"]):
        st.success("No missing values.")
    else:
        st.dataframe(
            R["per_column"],
            hide_index=True,
            width="stretch",
            column_config={
                "missing %": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100
                )
            },
        )
        st.markdown("**Rows that are missing the same set of columns**")
        st.dataframe(R["patterns"], hide_index=True, width="stretch")
        if len(R["co_missing"]):
            st.markdown("**Columns that tend to be missing together**")
            st.dataframe(R["co_missing"], hide_index=True, width="stretch")
        if len(R["relations"]):
            st.markdown("**Does being missing depend on other columns?**")
            st.dataframe(R["relations"], hide_index=True, width="stretch")
            if R["n_dependent"]:
                st.warning(
                    f"{R['n_dependent']} relationship(s) are statistically clear. Missingness is then not completely random, "
                    "and simply deleting those rows may bias the data."
                )
            else:
                st.info("No column clearly predicts where the gaps are.")
        st.caption(R["limits"])

with tabs[3]:
    Q = c_quality(df, target)
    if not Q:
        st.success("No quality problems found.")
    else:
        icon = {"error": "🔴", "warning": "🟠", "info": "🔵"}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "": icon[q["level"]],
                        "check": q["check"],
                        "column": q["column"],
                        "problem": q["message"],
                        "rows": q["count"],
                    }
                    for q in Q
                ]
            ),
            hide_index=True,
            width="stretch",
        )

with tabs[4]:
    if not target:
        st.info(
            "Choose a target column on the Dashboard to see target analysis. Without a target you can still run the unsupervised check."
        )
    guess = TG.guess_kind(df, target)
    kind = st.selectbox(
        "What kind of problem is it?",
        TG.KINDS,
        index=TG.KINDS.index(guess),
        help=f"Guessed from the target: {guess}.",
    )
    tcol = None
    if kind.startswith("Time"):
        tcol = st.selectbox("Date column", dt or list(df.columns))
    if kind != "Unsupervised" and not target:
        st.stop()
    if st.button("Analyse", type="primary"):
        splits = (
            {
                "train": ss.raw_splits["train"],
                "validation": ss.raw_splits.get("val"),
                "test": ss.raw_splits["test"],
            }
            if ss.get("raw_splits")
            else None
        )
        res = TG.analyze_target(df, target, kind, tcol, types, splits)
        for f in res["flags"]:
            {"error": st.error, "warning": st.warning}.get(f["level"], st.info)(
                f["message"]
            )
        if kind == "Classification":
            c1, c2 = st.columns([1, 2])
            c1.dataframe(res["distribution"], hide_index=True, width="stretch")
            if len(res["features"]):
                c2.dataframe(
                    res["features"],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "strength (0-1)": st.column_config.ProgressColumn(
                            min_value=0, max_value=1, format="%.2f"
                        )
                    },
                )
            if res["splits"] is not None:
                st.markdown("**Class share in each split (%)**")
                st.dataframe(res["splits"], hide_index=True, width="stretch")
        elif kind == "Regression":
            st.dataframe(res["summary"].rename("value").to_frame(), width="stretch")
            if len(res["features"]):
                st.dataframe(
                    res["features"],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "strength (0-1)": st.column_config.ProgressColumn(
                            min_value=0, max_value=1, format="%.2f"
                        )
                    },
                )
            if len(res["extreme_targets"]):
                st.markdown("**Most extreme target values**")
                st.dataframe(res["extreme_targets"], width="stretch")
            if len(res["high_leverage"]):
                st.markdown("**Rows with unusual feature combinations**")
                st.dataframe(res["high_leverage"], hide_index=True, width="stretch")
        elif kind.startswith("Time"):
            if "summary" in res:
                st.dataframe(
                    res["summary"].astype(str).rename("value").to_frame(),
                    width="stretch",
                )
                s = res["series"]
                st.altair_chart(
                    alt.Chart(s.iloc[:: max(1, len(s) // 2000)])
                    .mark_line()
                    .encode(x="time:T", y=alt.Y(f"{target}:Q"))
                    .properties(height=220),
                    width="stretch",
                )
                if res["seasonal_lags"]:
                    st.info(
                        "Autocorrelation peaks at lag(s) "
                        + ", ".join(map(str, res["seasonal_lags"]))
                        + ": a possible repeating cycle."
                    )
        else:
            if "pca" in res:
                st.dataframe(res["pca"], hide_index=True, width="stretch")
        st.caption(res["note"])
