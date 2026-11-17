# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import pandas as pd
import streamlit as st

from core import go, workflow_bar
from dataforge_engine import diagnose, merge, what_if_missing

SEV = {3: "🔴", 2: "🟠", 1: "🟡"}

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": [], "decisions": {}, "advisor_params": {}}.items():
    ss.setdefault(k_, v_)

workflow_bar("Inspect")
st.title("Advisor")
st.caption("Finds problems in the loaded data, explains why each matters, and lists your options with their pros and cons. "
           "It does not pick a winner: you decide, and every decision is recorded.")

if ss.df is None:
    st.info("No dataset yet. Add one first.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

base = ss.snapshots[0]["df"] if ss.snapshots else ss.df
target = ss.target if ss.target in base.columns else None
if not target:
    st.caption("Tip: choose a target column on the Dashboard to get target-related advice (class balance).")


@st.cache_data(show_spinner="Looking for problems...", max_entries=4)
def _diagnose(df, target):
    return diagnose(df, target)


findings = _diagnose(base, target)
if not findings:
    st.success("No problems found in the loaded data.")
    st.stop()

by_label = {}
st.write(f"**{len(findings)} finding(s)** in the loaded data ({len(base):,} rows). Open one to see the options.")
for f in findings:
    title = f"{SEV[f['severity']]} {f['problem']}" + (f" · `{f['column']}`" if f["column"] else "")
    done_key = ss.decisions.get(f["id"])
    with st.expander(title + ("  ·  decided" if done_key else ""), expanded=False):
        st.markdown(f["reason"])
        if f["evidence"]:
            st.caption("Evidence: " + ", ".join(f"{k} = {v}" for k, v in list(f["evidence"].items())[:8]))
        if f["note"]:
            st.info(f["note"])
        st.caption(f"Confidence: {f['confidence']}")
        if not f["options"]:
            continue
        labels = {o["key"]: o["label"] for o in f["options"]}
        keys = ["__none__"] + list(labels)
        cur = done_key if done_key in labels else "__none__"
        choice = st.selectbox("Your decision", keys, index=keys.index(cur), key=f"dec_{f['id']}",
                              format_func=lambda x: "Undecided" if x == "__none__" else labels[x])
        if choice == "__none__":
            ss.decisions.pop(f["id"], None)
        else:
            ss.decisions[f["id"]] = choice
        st.dataframe(pd.DataFrame([{"option": o["label"], "pros": o["pros"], "cons": o["cons"], "use when": o["when"]}
                                   for o in f["options"]]), hide_index=True, width="stretch")
        if f["id"].startswith("missing:"):
            if st.button("Test these options on the data", key=f"wi_{f['id']}"):
                st.dataframe(what_if_missing(base, f["column"], target), hide_index=True, width="stretch")
                st.caption("Measured on the loaded data. 'std change %' shows how much each fill shrinks the spread; "
                           "'corr with target' is only shown for a numeric target.")

params, chosen = {}, []
for f in findings:
    key = ss.decisions.get(f["id"])
    opt = next((o for o in f["options"] if o["key"] == key), None)
    if opt:
        params = merge(params, opt["patch"])
        chosen.append(f"{f['problem']}{' (' + str(f['column']) + ')' if f['column'] else ''}: {opt['label']}")
ss.advisor_params = params

st.divider()
st.subheader("Your decisions")
if chosen:
    for c in chosen:
        st.markdown(f"- {c}")
else:
    st.caption("Nothing decided yet. Prepare will use its recommended settings.")
st.caption("Decisions become the default settings of the Prepare steps. Steps you already ran keep their settings until you run them again.")
c1, c2 = st.columns(2)
if c1.button("Record decisions and open Prepare", type="primary", disabled=not chosen, width="stretch"):
    from datetime import datetime
    ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "step": "Advisor",
                   "detail": f"Recorded {len(chosen)} decision(s): " + "; ".join(chosen)[:300],
                   "rows": len(ss.df), "cols": ss.df.shape[1]})
    go("prepare")
if c2.button("Clear decisions", disabled=not chosen, width="stretch"):
    ss.decisions, ss.advisor_params = {}, {}
    st.rerun()
