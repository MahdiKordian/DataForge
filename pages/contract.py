# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import json

import numpy as np
import pandas as pd
import streamlit as st

from core import go, read_upload, workflow_bar
from dataforge_engine import infer_contract, validate_contract

KINDS = ["numeric", "text", "datetime"]

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": [], "contract": None, "validation_status": None}.items():
    ss.setdefault(k_, v_)

workflow_bar("Validate")
st.title("Data contract")
st.caption("Write down what a valid file looks like, then check every new file against it before it reaches your pipeline.")

if ss.df is None:
    st.info("No dataset yet. Add one first.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

base = ss.snapshots[0]["df"] if ss.snapshots else ss.df

# ---- 1. the contract -----------------------------------------------------------------------
st.subheader("1. Rules")
a, b = st.columns(2)
if a.button("Create from the loaded data", width="stretch"):
    ss.contract = infer_contract(base, ss.target)
    ss.validation_status = None
up = b.file_uploader("...or load a saved contract", type=["json"], label_visibility="collapsed")
if up is not None and st.session_state.get("_contract_file") != up.name:
    try:
        c = json.loads(up.getvalue())
        assert c.get("format") == "dataforge-contract" and "columns" in c
        ss.contract, ss["_contract_file"] = c, up.name
    except Exception:
        st.error("This is not a DataForge contract file.")

if ss.contract is None:
    st.info("Create a starting contract from the loaded data, then edit it. The ranges come from the data you have now, "
            "so adjust them to what is really valid (for example, ages above 120 are never valid).")
    st.stop()

con = ss.contract
rows = []
for name, r in con["columns"].items():
    rows.append({"column": name, "kind": r["kind"], "nullable": r["nullable"], "unique": r["unique"],
                 "min": r["min"], "max": r["max"], "allowed values (comma separated)": ", ".join(r["allowed"]) if r["allowed"] else ""})
edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["column"], key="contract_editor",
                        column_config={"kind": st.column_config.SelectboxColumn(options=KINDS, required=True)})
new_cols = {}
for _, r in edited.iterrows():
    num = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    allowed = [x.strip() for x in str(r["allowed values (comma separated)"]).split(",") if x.strip()]
    new_cols[r["column"]] = {"kind": r["kind"], "nullable": bool(r["nullable"]), "unique": bool(r["unique"]),
                             "min": num(r["min"]) if r["kind"] == "numeric" else None,
                             "max": num(r["max"]) if r["kind"] == "numeric" else None, "allowed": allowed or None}
con = {**con, "columns": new_cols}
con["allow_extra_columns"] = st.checkbox("Allow columns that are not in the contract", con.get("allow_extra_columns", True))
ss.contract = con
st.download_button("Download contract (.json)", json.dumps(con, indent=2), "contract.json", "application/json")

st.subheader("2. Check a file")
src = st.radio("Data to check", ["The current dataset", "A new file"], horizontal=True)
df = None
if src == "The current dataset":
    df = base
else:
    f = st.file_uploader("New file", type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"], key="contract_new")
    if f is not None:
        try:
            df, _ = read_upload(f)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
if df is not None and st.button("Validate", type="primary"):
    rep = validate_contract(df, con)
    ss.validation_status = rep["status"]
    {"passed": st.success, "warnings": st.warning, "failed": st.error}[rep["status"]](
        {"passed": "Validation passed.", "warnings": "Passed with warnings.", "failed": "Validation failed."}[rep["status"]] + " " + rep["summary"])
    if rep["issues"]:
        st.dataframe(pd.DataFrame([{"level": i["level"], "column": i["column"], "rule": i["rule"], "problem": i["message"],
                                    "rows": i["count"], "example row numbers": ", ".join(map(str, i["examples"]))} for i in rep["issues"]]),
                     hide_index=True, width="stretch")
    if rep["bad_rows"]:
        with st.expander(f"Rows that break a rule (first {min(len(rep['bad_rows']), 50)})"):
            st.dataframe(df.loc[rep["bad_rows"][:50]], width="stretch")
    if rep["status"] == "failed":
        st.caption("Fix the data or relax the rule before running the pipeline on this file.")
