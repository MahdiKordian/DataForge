# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from core import go, read_upload, workflow_bar
from dataforge_engine.security import pii_columns
from dataforge_engine import (SchemaError, SpecError, apply_saved, compare, dumps, loads, make_spec, refit, to_script,
                       validate_contract)

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": [], "pipeline_cfg": {}, "df_name": "dataset"}.items():
    ss.setdefault(k_, v_)

workflow_bar("Export")
st.title("Pipeline")
st.caption("Save what you did as one file, and run it again on tomorrow's data.")


def log_export(detail):
    ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "step": "Export", "detail": detail,
                   "rows": len(ss.df) if ss.df is not None else 0, "cols": ss.df.shape[1] if ss.df is not None else 0})


tab_save, tab_run = st.tabs(["Save the current pipeline", "Run a saved pipeline"])

with tab_save:
    if not ss.get("split_done") or ss.get("pipeline") is None or not ss.pipeline_cfg:
        st.info("Run Split and at least one Prepare step first.")
        if st.button("Go to Prepare", type="primary"):
            go("prepare")
    else:
        spec = make_spec(ss.pipeline, ss.get("effective_cfg", ss.pipeline_cfg), ss.split_info, ss.raw_splits["train"], str(ss.df_name))
        st.markdown("**Steps, in the order they run**")
        st.dataframe(pd.DataFrame([{"step": s["step"], "type": s["transformer"],
                                    "settings": json.dumps(s["params"], default=str)[:160]} for s in ss.pipeline.get_params()]),
                     hide_index=True, width="stretch")
        st.caption(f"Target: {spec['target'] or 'none'} · split seed {spec['split']['seed']} · "
                   f"pandas {spec['environment']['pandas']} · Python {spec['environment']['python']}. "
                   "The file also holds every value learned from the training set (means, categories, bounds).")
        pii = [c for c in pii_columns(ss.raw_splits["train"]) if c in set(spec["pipeline_state"]["input_columns"])]
        if pii:
            st.warning("The saved file stores the category values learned from the data. These columns look like personal data: "
                       + ", ".join(map(str, pii[:6])) + ". Drop them (Variable types: id or text) or keep the file private.")
        c1, c2 = st.columns(2)
        c1.download_button("Download pipeline (.json)", dumps(spec), f"{ss.df_name}_pipeline.json", "application/json",
                           width="stretch", on_click=log_export, args=("Downloaded the pipeline file",))
        c2.download_button("Download Python script (.py)", to_script(spec), f"{ss.df_name}_prepare.py", "text/x-python",
                           width="stretch", on_click=log_export, args=("Downloaded the Python script",))

with tab_run:
    pf = st.file_uploader("Saved pipeline (.json)", type=["json"], key="pipe_file")
    df_new, name_new = None, None
    nf = st.file_uploader("Data to prepare", type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"], key="pipe_data")
    if nf is not None:
        try:
            df_new, name_new = read_upload(nf)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    spec = None
    if pf is not None:
        try:
            spec = loads(pf.getvalue())
        except SpecError as e:
            st.error(str(e))
    if spec and df_new is not None:
        mode = st.radio("How to run it", ["Apply the saved pipeline (use the values learned on the original training data)",
                                          "Re-fit from the saved settings (learn again from this data)"])
        rep = validate_contract(df_new, spec["input_contract"])
        if rep["status"] == "failed":
            st.error("The new data breaks the pipeline's input contract. " + rep["summary"])
            st.dataframe(pd.DataFrame([{"column": i["column"], "problem": i["message"]} for i in rep["issues"] if i["level"] == "error"]),
                         hide_index=True, width="stretch")
            st.caption("Fix the file, then upload it again. The pipeline was not run.")
        else:
            if rep["status"] == "warnings":
                st.warning(rep["summary"])
            if st.button("Run", type="primary"):
                try:
                    if mode.startswith("Apply"):
                        out, warns = apply_saved(spec, df_new)
                        st.success(f"Prepared {len(out):,} rows, {out.shape[1]} columns.")
                        for w in warns:
                            st.warning(w)
                        st.dataframe(out.head(20), width="stretch")
                        st.download_button("Download prepared data (.csv)", out.to_csv(index=False).encode("utf-8"),
                                           f"{name_new}_prepared.csv", "text/csv", on_click=log_export, args=("Ran a saved pipeline on new data",))
                        ref = ss.raw_splits["train"].drop(columns=[spec["target"]], errors="ignore") if ss.get("raw_splits") else None
                        if ref is not None:
                            with st.expander("Has the data drifted from the training set?"):
                                r = compare(ref, df_new.drop(columns=[spec["target"]], errors="ignore"), "training", "new")
                                for fl in r["flags"] or [{"level": "info", "message": "No notable differences."}]:
                                    st.write("- " + fl["message"])
                                st.caption(r["note"])
                    else:
                        res = refit(spec, df_new)
                        st.success(f"Re-fitted: train {len(res['train']):,} / test {len(res['test']):,} rows.")
                        st.dataframe(res["train"].head(20), width="stretch")
                        st.download_button("Download training set (.csv)", res["train"].to_csv(index=False).encode("utf-8"),
                                           f"{name_new}_train_prepared.csv", "text/csv", on_click=log_export, args=("Re-fitted a saved pipeline",))
                except SchemaError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"The pipeline could not run on this data: {e}")
