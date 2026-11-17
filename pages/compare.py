# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import pandas as pd
import streamlit as st

from core import go, read_upload, workflow_bar
from dataforge_engine import compare

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": []}.items():
    ss.setdefault(k_, v_)

workflow_bar("Analyze")
st.title("Compare versions")
st.caption("Shows exactly what changed between two versions of a dataset: size, schema, missing values, distributions, categories and relationships.")

if ss.df is None:
    st.info("No dataset yet. Add one first.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()


def upload(label, key):
    f = st.file_uploader(label, type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"], key=key)
    if f is None:
        return None, None
    try:
        return read_upload(f)[0], f.name
    except Exception as e:
        st.error(f"Could not read {f.name}: {e}")
        return None, None


mode = st.radio("What do you want to compare?", ["Before vs After", "Dataset version A vs B", "Training vs Production"], horizontal=True)
A = B = None
la, lb = "A", "B"
if mode == "Before vs After":
    snaps = ss.snapshots
    if len(snaps) < 2:
        st.info("Run at least one Prepare step first.")
    else:
        names = [f"{i}: {s['name']}" for i, s in enumerate(snaps)]
        c1, c2 = st.columns(2)
        i = names.index(c1.selectbox("Before", names, index=0))
        j = names.index(c2.selectbox("After", names, index=len(names) - 1))
        A, B, la, lb = snaps[i]["df"], snaps[j]["df"], "before", "after"
elif mode == "Dataset version A vs B":
    c1, c2 = st.columns(2)
    with c1:
        useA = st.checkbox("Use the current dataset as A", True)
        A, _ = (ss.snapshots[0]["df"] if ss.snapshots else ss.df, None) if useA else upload("Version A", "cmpA")
    with c2:
        B, _ = upload("Version B", "cmpB")
    la, lb = "A", "B"
else:
    A = ss.raw_splits["train"] if ss.get("raw_splits") else (ss.snapshots[0]["df"] if ss.snapshots else ss.df)
    st.caption("Training data: " + ("the training set from your split." if ss.get("raw_splits") else "the loaded dataset (no split yet)."))
    B, _ = upload("Production / new data", "cmpP")
    la, lb = "training", "production"

if A is None or B is None:
    st.stop()

r = compare(A, B, la, lb)
st.subheader("Summary")
st.dataframe(r["summary"], hide_index=True, width="stretch")
for fl in r["flags"]:
    {"warning": st.warning, "info": st.info}.get(fl["level"], st.info)(fl["message"])
if not r["flags"]:
    st.success("No notable differences found.")
tabs = st.tabs(["Missing values", "Numeric columns", "Categories", "Correlations", "Schema"])
with tabs[0]:
    m = r["missing"]
    st.dataframe(m[m["change"].abs() > 0] if len(m) else m, hide_index=True, width="stretch")
    if len(m) and not (m["change"].abs() > 0).any():
        st.caption("Missing-value rates are identical.")
with tabs[1]:
    st.dataframe(r["numeric"], hide_index=True, width="stretch")
    st.caption("PSI below 0.1 is stable, 0.1 to 0.25 moderate, above 0.25 large. KS is the largest gap between the two distributions.")
with tabs[2]:
    st.dataframe(r["categorical"], hide_index=True, width="stretch")
with tabs[3]:
    st.dataframe(r["correlations"], hide_index=True, width="stretch") if len(r["correlations"]) else st.caption("No pair changed by 0.2 or more.")
with tabs[4]:
    st.write("Added columns:", ", ".join(map(str, r["added"])) or "none")
    st.write("Removed columns:", ", ".join(map(str, r["removed"])) or "none")
    st.dataframe(pd.DataFrame(r["dtype_changes"]), hide_index=True, width="stretch") if r["dtype_changes"] else st.write("Type changes: none")
st.caption(r["note"])
