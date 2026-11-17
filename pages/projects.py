# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import streamlit as st

from core import clear_all, read_upload
from dataforge_engine import projects as P
from dataforge_engine.scale import content_fingerprint, fingerprint
from dataforge_engine.security import UnsafeInput, pii_columns

ss = st.session_state
for k_, v_ in {"df": None, "df_name": "dataset", "target": None, "log": [], "snapshots": [], "decisions": {}, "advisor_params": {}}.items():
    ss.setdefault(k_, v_)

st.title("Projects")
st.caption(f"Save your work and come back to it. Projects live in a folder on the computer that runs DataForge ({P.root()}). "
           "They are not sent anywhere.")


def settings_now() -> dict:
    steps = [{"step": e["step"], "params": e.get("params", {})} for e in ss.log if e.get("step") in ("Split", "Clean", "Transform", "Encode", "Scale") and "params" in e]
    return {"target": ss.target, "var_types": ss.get("var_types"), "var_orders": ss.get("var_orders"), "contract": ss.get("contract"),
            "decisions": ss.get("decisions"), "advisor_params": ss.get("advisor_params"), "split_info": ss.get("split_info"), "steps": steps}


st.subheader("Save the current work")
if ss.df is None:
    st.info("Load a dataset first.")
else:
    base = ss.snapshots[0]["df"] if ss.snapshots else ss.df
    name = st.text_input("Project name", value=str(ss.df_name)[:60])
    store = st.checkbox("Keep a copy of the data in the project", True,
                        help="Without it only settings are saved, and you upload the same file again when you reopen the project.")
    risky = pii_columns(base) if store else []
    if risky:
        st.warning("These columns look like personal data and would be stored on this computer: " + ", ".join(map(str, risky[:6]))
                   + ". Untick the box above to save settings only.")
    notes = st.text_area("Notes (optional)", height=70)
    if st.button("Save project", type="primary"):
        try:
            P.save_project(name, base, settings_now(), store, notes)
            st.success(f"Saved '{name}'.")
        except UnsafeInput as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not save: {e}")

st.subheader("Your projects")
items = P.list_projects()
if not items:
    st.caption("No projects yet.")
for it in items:
    with st.expander(f"{it['name']}  ·  {it['rows']:,} rows · {it['steps']} step(s) · saved {it['saved'].replace('T', ' ')}"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Open", key=f"open_{it['name']}", type="primary"):
            try:
                meta, df = P.load_project(it["name"])
                if df is None:
                    ss["pending_open"] = meta
                    st.rerun()
                else:
                    clear_all()
                    ss.update({"df": df.copy(), "df_name": meta["name"], "snapshots": [{"name": "Loaded", "df": df.copy()}],
                               "log": [{"time": meta["saved"][11:19], "step": "Load", "detail": f"Opened project {meta['name']}", "rows": len(df), "cols": df.shape[1]}]})
                    S = meta["settings"]
                    ss.target, ss.var_types, ss.var_orders = S.get("target"), S.get("var_types"), S.get("var_orders") or {}
                    ss.contract, ss.decisions, ss.advisor_params = S.get("contract"), S.get("decisions") or {}, S.get("advisor_params") or {}
                    ss.split_done, ss.pipeline_cfg, ss.cfg_stack = False, {}, []
                    ss.pending_replay = meta.get("steps") or None
                    st.success("Opened. " + ("Go to Prepare to re-run the saved steps." if ss.pending_replay else ""))
            except Exception as e:
                st.error(f"Could not open: {e}")
        try:
            c2.download_button("Download (.zip)", P.export_zip(it["name"]), f"{it['name']}.zip", "application/zip", key=f"zip_{it['name']}")
        except Exception as e:
            c2.caption(str(e))
        if c3.button("Delete", key=f"del_{it['name']}"):
            P.delete_project(it["name"])
            st.rerun()

meta = ss.get("pending_open")
if meta:
    st.warning(f"'{meta['name']}' was saved without data. Upload the same file to continue.")
    f = st.file_uploader("The dataset", type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"], key="reopen")
    if f is not None:
        try:
            df, _ = read_upload(f)
            expected = meta["dataset"].get("content_fingerprint")
            actual = content_fingerprint(df) if expected else fingerprint(df)
            if actual != (expected or meta["dataset"]["fingerprint"]):
                st.warning(
                    "This file is not identical to the one the project was saved with. "
                    "Settings are applied anyway; check the results."
                )
            clear_all()
            ss.update({"df": df.copy(), "df_name": meta["name"], "snapshots": [{"name": "Loaded", "df": df.copy()}],
                       "log": [{"time": "", "step": "Load", "detail": f"Opened project {meta['name']}", "rows": len(df), "cols": df.shape[1]}]})
            S = meta["settings"]
            ss.target, ss.var_types, ss.var_orders = S.get("target"), S.get("var_types"), S.get("var_orders") or {}
            ss.contract, ss.decisions, ss.advisor_params = S.get("contract"), S.get("decisions") or {}, S.get("advisor_params") or {}
            ss.split_done, ss.pipeline_cfg, ss.cfg_stack, ss.pending_replay, ss.pending_open = False, {}, [], meta.get("steps") or None, None
            st.success("Opened.")
        except (UnsafeInput, Exception) as e:
            st.error(f"Could not read the file: {e}")
