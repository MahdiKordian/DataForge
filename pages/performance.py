# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import altair as alt
import pandas as pd
import streamlit as st

from core import go
from dataforge_engine import benchmark as BM
from dataforge_engine import scale as SC

ss = st.session_state
for k_, v_ in {"df": None, "target": None, "log": [], "snapshots": []}.items():
    ss.setdefault(k_, v_)

st.title("Performance")
st.caption("How big is your data, what DataForge will do about it, and how long things take.")

if ss.df is None:
    st.info("No dataset yet. Add one first.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

base = ss.snapshots[0]["df"] if ss.snapshots else ss.df
pl = SC.plan(base)
m = st.columns(4)
m[0].metric("Size", f"{pl['rows']:,} rows"); m[1].metric("Columns", pl["columns"])
m[2].metric("Memory", f"{pl['memory_mb']:.1f} MB"); m[3].metric("Tier", pl["tier"])
st.info(pl["note"])
eng = SC.engines()
st.caption("Optional fast engines: " + ", ".join(f"{k} {v}" if v else f"{k} (not installed)" for k, v in eng.items()))

st.subheader("Use less memory")
st.caption("Stores numbers in smaller types and, if you choose, repeated text as categories. Values do not change.")
c1, c2 = st.columns(2)
cats = c1.checkbox("Turn repeated text into categories", True)
f32 = c2.checkbox("Use 32-bit decimals (about 7 digits of precision)", False)
if st.button("Check the saving"):
    out, rep = SC.optimize_dtypes(base, categories=cats, float32=f32)
    st.session_state["_opt"] = (out, rep)
if "_opt" in st.session_state:
    out, rep = st.session_state["_opt"]
    st.write(f"{rep['before_mb']} MB → **{rep['after_mb']} MB** ({rep['saved %']}% less, {rep['columns changed']} column(s) changed)")
    if st.button("Apply to the loaded data", type="primary"):
        if ss.get("split_done") or any(e.get("step") in ("Split", "Clean", "Transform", "Encode", "Scale") for e in ss.log):
            st.error("Reset the prepare steps first, then optimise.")
        else:
            ss.df = out.copy()
            ss.snapshots = [{"name": "Loaded", "df": out.copy()}] + ss.snapshots[1:]
            st.session_state.pop("_opt")
            st.success("Done. The pipeline reads categories as plain text, so results are unchanged.")
            st.rerun()

st.subheader("Measure on this data")
st.caption("Runs the main tasks on 25%, 50% and 100% of your rows. Large data can take a while.")
if st.button("Run measurement", type="primary"):
    rows = []
    prog = st.progress(0.0)
    fr = [0.25, 0.5, 1.0]
    for i, f in enumerate(fr):
        d = base.sample(frac=f, random_state=0) if f < 1 else base
        t = ss.target if ss.target in d.columns else None
        d = d.rename(columns={t: "y"}) if t else d
        try:
            for name, fn in BM.tasks(d if t else d.assign(y="a")).items():
                rows.append({"rows": len(d), "task": name, "seconds": BM._time(fn)})
        except Exception as e:
            st.error(f"Measurement stopped: {e}")
            break
        prog.progress((i + 1) / len(fr))
    if rows:
        R = pd.DataFrame(rows)
        st.dataframe(R.pivot_table(index="task", columns="rows", values="seconds").round(2), width="stretch")
        st.altair_chart(alt.Chart(R).mark_line(point=True).encode(x="rows:Q", y=alt.Y("seconds:Q", scale=alt.Scale(type="log")), color="task:N").properties(height=300),
                        width="stretch")
        st.caption("Both axes of a straight line on a log chart mean predictable scaling. A steep line marks the task that will hurt first.")
