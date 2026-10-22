# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import pandas as pd
import streamlit as st
from datetime import datetime

from core import go, workflow_bar
from dataforge.security import pii_columns
from dataforge.types import TYPES, WHAT_HAPPENS, infer_types

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": [], "var_types": None, "var_orders": {}}.items():
    ss.setdefault(k_, v_)

workflow_bar("Inspect")
st.title("Variable types")
st.caption("A column stored as a number is not always a quantity (a zip code, a customer id) and text is not always a category. "
           "DataForge guesses, shows how sure it is, and asks you to confirm. The pipeline then treats each column according to its type.")

if ss.df is None:
    st.info("No dataset yet. Add one first.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

base = ss.snapshots[0]["df"] if ss.snapshots else ss.df
target = ss.target if ss.target in base.columns else None


@st.cache_data(show_spinner="Inspecting columns...", max_entries=3)
def _infer(df, target):
    return infer_types(df, target)


inferred = _infer(base, target)
hidden = set(pii_columns(base))
current = ss.var_types or {}
only_low = st.checkbox("Show only uncertain guesses", False)
rows = []
for c, g in inferred.items():
    if c == target:
        continue
    if only_low and g["confidence"] != "low" and current.get(c, g["type"]) == g["type"]:
        continue
    ex = "[hidden: looks like personal data]" if c in hidden else ", ".join(map(str, base[c].dropna().astype(str).unique()[:3]))
    ty = current.get(c, g["type"])
    rows.append({"column": c, "type": ty, "confidence": g["confidence"] if ty == g["type"] else "set by you",
                 "why": g["reason"], "what happens": WHAT_HAPPENS[ty], "examples": ex})
if target:
    st.caption(f"`{target}` is the target and is not listed.")
if not rows:
    st.success("Nothing uncertain to review.")
    st.stop()

opts = [t for t in TYPES if t != "target"]
edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key=f"types_{only_low}",
                        disabled=["column", "confidence", "why", "what happens", "examples"],
                        column_config={"type": st.column_config.SelectboxColumn(options=opts, required=True)})
chosen = {**{c: g["type"] for c, g in inferred.items() if c != target}, **current}
for _, r in edited.iterrows():
    chosen[r["column"]] = r["type"]

orders = dict(ss.var_orders)
ordinal_cols = [c for c, t in chosen.items() if t == "ordinal"]
if ordinal_cols:
    st.subheader("Order of ordinal columns")
    st.caption("List the values from lowest to highest, separated by commas.")
    for c in ordinal_cols:
        guess = inferred.get(c, {}).get("order") or sorted(map(str, base[c].dropna().unique()))
        cur = orders.get(c) or guess
        txt = st.text_input(f"`{c}`", ", ".join(map(str, cur)), key=f"ord_{c}")
        orders[c] = [x.strip() for x in txt.split(",") if x.strip()]
        known = {str(v).strip().lower() for v in base[c].dropna().unique()}
        missing = known - {x.lower() for x in orders[c]}
        if missing:
            st.warning(f"Values not in the order (they will become missing): {', '.join(sorted(missing)[:5])}")

changed = {c: t for c, t in chosen.items() if t != inferred.get(c, {}).get("type")}
st.write(f"**{len(changed)}** column(s) differ from the automatic guess.")
c1, c2 = st.columns(2)
if c1.button("Use these types", type="primary", width="stretch"):
    ss.var_types, ss.var_orders = chosen, {c: orders[c] for c in ordinal_cols}
    ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "step": "Types",
                   "detail": f"Confirmed variable types ({len(changed)} changed from the guess)",
                   "rows": len(ss.df), "cols": ss.df.shape[1]})
    st.success("Saved. The types apply the next time Prepare runs." + (" Run Split again to apply them to the current split." if ss.get("split_done") else ""))
if c2.button("Back to automatic", disabled=not ss.var_types, width="stretch"):
    ss.var_types, ss.var_orders = None, {}
    st.rerun()
if ss.var_types:
    st.caption("Confirmed types are active.")
