# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import altair as alt
import streamlit as st

from core import go, workflow_bar
from dataforge_engine import lab as LB

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": [], "split_done": False}.items():
    ss.setdefault(k_, v_)

workflow_bar("Prepare")
st.title("Preparation lab")
st.caption("Not sure which method is better? Try several on the training set and compare them with the same model and the same cross-validation. "
           "The test set is never used here, so it stays clean for the final check.")

if not ss.get("split_done") or not ss.get("raw_splits"):
    st.info("Run Split on the Prepare page first. The lab only touches the training set.")
    if st.button("Go to Prepare", type="primary"):
        go("prepare")
    st.stop()
if not LB.sklearn_available():
    st.error("The lab needs scikit-learn. Install it with: pip install scikit-learn")
    st.stop()
t = ss.split_info.get("target")
if not t:
    st.info("Choose a target column on the Dashboard and run Split again.")
    st.stop()

train = ss.raw_splits["train"]
from dataforge_engine.splitting import is_classification
is_class = is_classification(train[t].dropna())
st.write(f"Target `{t}` · {'classification' if is_class else 'regression'} · {len(train):,} training rows")

dim = st.selectbox("What do you want to compare?", list(LB.DIMENSIONS))
choices = LB.DIMENSIONS[dim][2]
vals = st.multiselect("Options to try (2 to 4)", choices, default=choices[:2], max_selections=4)
models = LB.models("classification" if is_class else "regression")
c1, c2, c3 = st.columns(3)
model = c1.selectbox("Model", list(models))
folds = c2.slider("Folds", 3, 10, 5)
metrics = ["macro F1", "balanced accuracy", "accuracy", "ROC AUC"] if is_class else ["R2", "RMSE", "MAE"]
metric = c3.selectbox("Main metric", metrics)
if len(vals) < 2:
    st.info("Pick at least two options.")
    st.stop()
if st.button("Run the comparison", type="primary"):
    types = ({"types": dict(ss.var_types), "orders": dict(ss.get("var_orders", {}))} if ss.get("var_types") else None)
    variants = LB.variants_for(dim, vals, types=types, is_class=is_class)
    bar = st.progress(0.0)
    res = LB.run_lab(train, t, variants, model, folds, 42, metric, progress=bar.progress)
    st.session_state["_lab"] = res
res = st.session_state.get("_lab")
if res:
    T = res["table"]
    st.dataframe(T, hide_index=True, width="stretch")
    F = res["folds"]
    if res["metric"] in F and F[res["metric"]].notna().any():
        st.altair_chart(alt.Chart(F.dropna(subset=[res["metric"]])).mark_boxplot().encode(
            x=alt.X("variant:N", title=None), y=alt.Y(f"{res['metric']}:Q", scale=alt.Scale(zero=False))).properties(height=260), width="stretch")
    for n in res["notes"]:
        st.caption(n)
    st.caption("When you decide, set that option on the Prepare page. The decision is then yours and the evidence is above.")
