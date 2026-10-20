# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import json
import time
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st

from core import workflow_bar
from dataforge import balance_train, build_pipeline, leakage_report, split_frame

PAGES = {"dataset": "pages/dataset.py", "dashboard": "pages/dashboard.py"}
STEPS = ["Split", "Clean", "Transform", "Encode", "Scale"]
PROCESS = STEPS[1:]
LABELS = {"Clean": ["Text cleanup", "Column filter", "Missing values"],
          "Transform": ["Outliers", "Skew", "Near-duplicate columns", "Dropped columns"],
          "Encode": ["Encoding"], "Scale": ["Scaling"]}
INFO = {
    "Split": "Create train, validation and test sets FIRST, so every later step learns only from the training set.",
    "Clean": "Fix text, duplicates, constant or mostly empty columns, and missing values.",
    "Transform": "Handle outliers and skew, and drop columns that add nothing.",
    "Encode": "Turn text categories into numbers a model can use.",
    "Scale": "Put numeric columns on a similar range.",
}
CASES = ["Keep as is", "lowercase", "UPPERCASE", "Title Case"]
NUM_MISS = ["Median", "Mean", "Most common", "Fill with 0", "Drop rows", "Leave as is"]
CAT_MISS = ["Most common", "Fill with 'Unknown'", "Drop rows", "Leave as is"]
OUTLIERS = ["Leave as is", "Clip at 1st and 99th percentile", "Clip to IQR bounds", "Remove rows with outliers (IQR)"]
SCALERS = ["Standard (z-score)", "Min-max (0 to 1)", "Robust (median and IQR)", "Do not scale"]

DEFAULTS = {
    "Clean": dict(trim=True, case="Keep as is", to_num=True, dups=True, const=True, max_missing=60,
                  drop_missing_target=True, num_missing="Median", cat_missing="Most common"),
    "Transform": dict(outliers=OUTLIERS[1], out_cols=None, skew=True, skew_thr=1.0, skew_method="Log (log1p)",
                      drop=[], corr=0.95),
    "Encode": dict(low="One-hot", max_onehot=10, drop_first=True, high="Frequency", dates=True, encode_target=True),
    "Scale": dict(method=SCALERS[0], cols=None),
    "Split": dict(test=20, val=0, seed=42, stratify=True, balance="None"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&display=swap');
.block-container{max-width:1080px;padding-top:2.5rem}
.dfd{font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.dfd h1{font-size:44px;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.dfd h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:28px 0 10px;padding:0}
.dfd p{margin:8px 0 0;opacity:.7;max-width:62ch;font-size:17px}
</style>
"""


def defaults_for(step):
    """Recommended settings, adjusted by the decisions recorded on the Advisor page."""
    return {**DEFAULTS[step], **st.session_state.get("advisor_params", {}).get(step, {})}


def nums(df):
    return list(df.select_dtypes("number").columns)


def is_text(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s))


def is_class(s):
    return not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 20


def continuous(df, t):
    return [c for c in nums(df) if c != t and df[c].nunique() > 2]


def ordinal(s):
    m = {v: i for i, v in enumerate(sorted(s.dropna().astype(str).unique()))}
    return s.astype(str).map(m).where(s.notna()), m


def jsonable(x):
    return json.loads(json.dumps(x, default=str))


def go(page):
    try:
        st.switch_page(PAGES[page])
    except Exception:
        st.toast(f"Page not found: {PAGES[page]}")


def clear_prepared():
    """Forget the split, the pipeline and everything derived from them."""
    ss = st.session_state
    ss.split_done = False
    for k in ("train_df", "val_df", "test_df", "split_info", "raw_splits", "pipeline", "leakage", "prep_warnings", "n_train", "effective_cfg"):
        ss.pop(k, None)
    ss.pipeline_cfg, ss.cfg_stack = {}, []


def split_target():
    ss = st.session_state
    return (ss.get("split_info") or {}).get("target")


def effective_cfg(cfg=None):
    """The steps plus the variable types the user confirmed on the Types page."""
    ss = st.session_state
    cfg = dict(ss.pipeline_cfg if cfg is None else cfg)
    if ss.get("var_types"):
        cfg["Types"] = {"types": dict(ss.var_types), "orders": dict(ss.get("var_orders", {}))}
    return cfg


def stage_df(step):
    """The TRAINING data as the given step will see it (only the earlier steps applied)."""
    ss = st.session_state
    before = {k: v for k, v in ss.pipeline_cfg.items() if PROCESS.index(k) < PROCESS.index(step)}
    return build_pipeline(effective_cfg(before), split_target()).fit_transform(ss.raw_splits["train"])


def rebuild():
    """Fit the whole pipeline on the raw training set, then apply it to validation and test."""
    ss = st.session_state
    raw, info, t = ss.raw_splits, ss.split_info, split_target()
    ss.effective_cfg = effective_cfg()
    pipe = build_pipeline(ss.effective_cfg, t)
    train = pipe.fit_transform(raw["train"])
    warns, val = [], None
    if raw["val"] is not None:
        val = pipe.transform(raw["val"])
        warns += pipe.warnings_
    test = pipe.transform(raw["test"])
    warns += pipe.warnings_
    if train.empty or test.empty:
        raise ValueError("Preparation removed every row of the training or test set.")
    ss.pipeline, ss.prep_warnings = pipe, sorted(set(warns))
    ss.df = pd.concat([x for x in (train, val, test) if x is not None], ignore_index=True)
    ss.n_train = len(train)
    ss.train_df = balance_train(train, t, info["balance"], info["seed"])
    ss.val_df, ss.test_df = val, test
    ss.leakage = leakage_report(raw["train"], raw["val"], raw["test"], t, processed_train=train)
    return pipe


def done_steps():
    return {e.get("step") for e in st.session_state.log} & set(STEPS)


def run_step(step, p):
    """Applies one step, logs it and snapshots it. Returns notes, or None on error."""
    ss = st.session_state
    t0 = time.perf_counter()
    if step == "Split":
        base = ss.snapshots[0]["df"]
        t = ss.target if ss.target in base.columns else None
        try:
            train, val, test, notes = split_frame(base, t, p["test"], p["val"], p["seed"], p["stratify"])
        except ValueError as e:
            st.error(str(e))
            return None
        ss.log = [e for e in ss.log if e.get("step") not in STEPS]   # a new split restarts preparation
        ss.snapshots = ss.snapshots[:1]
        clear_prepared()
        ss.raw_splits = {"train": train, "val": val, "test": test}
        ss.split_info = {"test_pct": p["test"], "val_pct": p["val"], "seed": p["seed"],
                         "balance": p["balance"], "target": t, "stratify": p["stratify"]}
        ss.split_done = True
        rebuild()
        if p["balance"] != "None" and t:
            notes.append(f"the training set will be balanced by {p['balance'].split()[0].lower()} after preparation")
    else:
        if not ss.get("split_done"):
            st.error("Run Split first, so the steps can learn from the training set only.")
            return None
        prev = ss.pipeline_cfg.get(step)
        ss.pipeline_cfg[step] = p
        try:
            pipe = rebuild()
        except Exception as e:
            if prev is None:
                ss.pipeline_cfg.pop(step, None)
            else:
                ss.pipeline_cfg[step] = prev
            st.error(f"{step} could not be applied: {e}")
            return None
        ss.cfg_stack.append((step, prev))
        notes = [n for e in pipe.log_ if e["step"] in LABELS[step] for n in e["notes"]]
    ss.snapshots.append({"name": f"After {step.lower()}", "df": ss.df.copy()})
    now = datetime.now().strftime("%H:%M:%S")
    ss.modified_at = now
    ss.log.append({"run_id": uuid.uuid4().hex[:8], "seconds": round(time.perf_counter() - t0, 2), "time": now, "step": step,
                   "detail": "; ".join(notes).capitalize() if notes else "No changes needed",
                   "rows": len(ss.df), "cols": ss.df.shape[1], "params": jsonable(p)})
    return notes


def preview(step, p):
    """Shows what a step would do, without changing anything. Large data is sampled."""
    ss = st.session_state
    if step == "Split":
        base = ss.snapshots[0]["df"]
        t0 = ss.target if ss.target in base.columns else None
        try:
            tr, va, te, notes = split_frame(base, t0, p["test"], p["val"], p["seed"], p["stratify"])
        except ValueError as e:
            st.error(str(e))
            return
        st.info("Preview: " + "; ".join(notes))
        return
    raw = ss.raw_splits["train"]
    sample = raw.sample(min(len(raw), 20_000), random_state=0) if len(raw) > 20_000 else raw
    try:
        now = build_pipeline(effective_cfg(), split_target()).fit_transform(sample)
        pipe = build_pipeline(effective_cfg({**ss.pipeline_cfg, step: p}), split_target())
        after = pipe.fit_transform(sample)
    except Exception as e:
        st.error(f"This step would fail: {e}")
        return
    miss = lambda d: f"{d.isna().sum().sum():,}"
    st.dataframe(pd.DataFrame({"": ["rows", "columns", "missing cells"],
                               "now": [f"{len(now):,}", str(now.shape[1]), miss(now)],
                               "after this step": [f"{len(after):,}", str(after.shape[1]), miss(after)]}),
                 hide_index=True, width="stretch")
    notes = [n for e in pipe.log_ if e["step"] in LABELS[step] for n in e["notes"]]
    st.markdown("**Would do:** " + ("; ".join(notes) if notes else "nothing"))
    gone, new = [c for c in now.columns if c not in after.columns], [c for c in after.columns if c not in now.columns]
    if gone:
        st.caption("Columns removed: " + ", ".join(map(str, gone[:10])))
    if new:
        st.caption("Columns added: " + ", ".join(map(str, new[:10])))
    st.dataframe(after.head(5), width="stretch")
    st.caption("Nothing was changed" + (" (preview uses a sample of the training set)." if sample is not raw else ".")
               + " Press Run to apply, or Undo afterwards.")


def finish(lines):
    ss = st.session_state
    ss.flash = lines
    ss.prepare_focus = next((s for s in STEPS if s not in done_steps()), None)
    st.rerun()


def apply_step(step, p):
    notes = run_step(step, p)
    if notes is not None:
        finish([f"**{step}:** " + ("; ".join(notes) if notes else "nothing needed changing")])


def apply_all():
    lines = []
    for step in STEPS:
        notes = run_step(step, defaults_for(step))
        if notes is None:
            break
        lines.append(f"**{step}:** " + ("; ".join(notes) if notes else "nothing needed changing"))
    finish(lines)


def undo():
    ss = st.session_state
    idx = [i for i, e in enumerate(ss.log) if e.get("step") in STEPS]
    if not idx:
        return
    entry = ss.log.pop(idx[-1])
    if entry["step"] == "Split" or not ss.get("cfg_stack"):
        ss.df = ss.snapshots[0]["df"].copy()
        ss.snapshots = ss.snapshots[:1]
        clear_prepared()
    else:
        step, prev = ss.cfg_stack.pop()
        if prev is None:
            ss.pipeline_cfg.pop(step, None)
        else:
            ss.pipeline_cfg[step] = prev
        rebuild()
        if len(ss.snapshots) > 1:
            ss.snapshots.pop()
    ss.modified_at = datetime.now().strftime("%H:%M:%S")
    ss.flash = [f"Undid **{entry['step']}**."]
    st.rerun()


def reset():
    ss = st.session_state
    ss.df = ss.snapshots[0]["df"].copy()
    ss.snapshots = ss.snapshots[:1]
    ss.log = [e for e in ss.log if e.get("step") not in STEPS]
    ss.artifacts = {}
    clear_prepared()
    ss.modified_at = datetime.now().strftime("%H:%M:%S")
    ss.log.append({"time": ss.modified_at, "step": "Reset", "detail": "Reset to the original data",
                   "rows": len(ss.df), "cols": ss.df.shape[1]})
    ss.flash = ["Reset to the original data."]
    st.rerun()


def facts(step, df, t):
    n = continuous(df, t)
    if step == "Clean":
        return (f"Right now: {int(df.duplicated().sum())} duplicate rows, "
                f"{int((df.nunique(dropna=True) <= 1).sum())} constant columns, {int(df.isna().sum().sum())} missing cells.")
    if step == "Transform":
        sk = sum(df[c].dropna().skew() > 1 for c in n if df[c].notna().sum() > 2)
        return f"Right now: {len(n)} continuous columns, {sk} clearly right-skewed."
    if step == "Encode":
        return f"Right now: {sum(is_text(df[c]) for c in df.columns if c != t)} text column(s) to encode."
    if step == "Scale":
        return f"Right now: {len(n)} continuous column(s) can be scaled."
    return f"Right now: {len(df):,} rows would be divided between the sets."


def ui(step, df, t, key):
    k = lambda name: f"p_{step}_{name}_{key}"
    d, p = defaults_for(step), {}
    a, b = st.columns(2)
    if step == "Clean":
        p["trim"] = a.checkbox("Trim spaces in text", d["trim"], key=k("trim"))
        p["case"] = a.selectbox("Text case", CASES, index=CASES.index(d["case"]), key=k("case"))
        p["to_num"] = a.checkbox("Convert text that looks like numbers", d["to_num"], key=k("to_num"))
        p["dups"] = a.checkbox("Remove duplicate rows", d["dups"], key=k("dups"))
        p["const"] = a.checkbox("Drop columns with a single value", d["const"], key=k("const"))
        p["max_missing"] = b.slider("Drop columns missing more than (%)", 0, 100, d["max_missing"], key=k("mm"),
                                    help="100 keeps every column.")
        p["num_missing"] = b.selectbox("Missing numbers", NUM_MISS, index=NUM_MISS.index(d["num_missing"]), key=k("nm"))
        p["cat_missing"] = b.selectbox("Missing text", CAT_MISS, index=CAT_MISS.index(d["cat_missing"]), key=k("cm"))
        p["per_column"] = dict(d.get("per_column", {}))
        if p["per_column"]:
            b.caption(f"{len(p['per_column'])} column(s) use the choice you made on the Advisor page.")
        p["drop_missing_target"] = bool(t) and b.checkbox("Remove rows where the target is missing", True, key=k("tm"))
    elif step == "Transform":
        cont = continuous(df, t)
        p["outliers"] = a.selectbox("Outliers", OUTLIERS, index=OUTLIERS.index(d["outliers"]), key=k("out"))
        p["out_cols"] = a.multiselect("Apply to columns", cont, default=cont, key=k("oc"))
        p["skew"] = b.checkbox("Reduce right skew", d["skew"], key=k("skew"))
        p["skew_thr"] = b.slider("Skew above", 0.5, 3.0, d["skew_thr"], 0.1, key=k("st"))
        p["skew_method"] = b.selectbox("Method", ["Log (log1p)", "Square root"], index=["Log (log1p)", "Square root"].index(d["skew_method"]), key=k("sm"))
        use = a.checkbox("Drop near-duplicate columns", True, key=k("cu"))
        p["corr"] = b.slider("When |correlation| is above", 0.7, 0.99, d["corr"], 0.01, key=k("cr")) if use else None
        p["drop"] = st.multiselect("Also drop these columns", [c for c in df.columns if c != t],
                                 default=[c for c in d["drop"] if c in df.columns and c != t], key=k("drop"))
    elif step == "Encode":
        p["low"] = a.selectbox("Few categories (up to the limit)", ["One-hot", "Ordinal"], key=k("low"))
        p["max_onehot"] = a.number_input("Category limit", 2, 100, d["max_onehot"], key=k("lim"))
        p["drop_first"] = a.checkbox("Drop the first one-hot column", d["drop_first"], key=k("df"),
                                     help="Avoids a redundant column per category group.")
        p["high"] = b.selectbox("Many categories", ["Frequency", "Ordinal", "Drop the column"], key=k("hi"))
        p["dates"] = b.checkbox("Split dates into year, month, day, weekday", d["dates"], key=k("dt"))
        p["encode_target"] = bool(t) and b.checkbox("Encode a text target as numbers", True, key=k("et"))
    elif step == "Scale":
        cont = continuous(df, t)
        p["method"] = a.selectbox("Method", SCALERS, key=k("m"))
        p["cols"] = b.multiselect("Columns", cont, default=cont, key=k("c"))
        st.caption("Statistics are learned from the training set only, then reused for validation and test.")
    else:
        clf = bool(t) and is_class(df[t].dropna())
        p["test"] = a.slider("Test set (%)", 5, 40, d["test"], key=k("te"))
        p["val"] = a.slider("Validation set (%)", 0, 30, d["val"], key=k("va"), help="0 means no validation set.")
        p["seed"] = b.number_input("Random seed", 0, 10_000, d["seed"], key=k("sd"))
        p["stratify"] = clf and b.checkbox("Keep class proportions (stratify)", True, key=k("sf"))
        p["balance"] = b.selectbox("Balance the training set", ["None", "Oversample minority", "Undersample majority"],
                                   index=["None", "Oversample minority", "Undersample majority"].index(d["balance"]),
                                   key=k("bal")) if clf else "None"
        if not clf:
            st.caption("Pick a classification target on the Dashboard to enable stratifying and balancing.")
    return p


ss = st.session_state
for k_, v_ in {"df": None, "df_name": "untitled", "target": None, "log": [], "snapshots": [],
               "modified_at": None, "split_done": False, "artifacts": {},
               "pipeline_cfg": {}, "cfg_stack": []}.items():
    ss.setdefault(k_, v_)
st.markdown(CSS, unsafe_allow_html=True)
workflow_bar("Prepare")
st.markdown('<div class="dfd"><h1>Prepare</h1><p>Split first, then run the steps one by one, or run them all with '
            "recommended settings. Every step learns from the training set only, is logged and can be undone.</p></div>",
            unsafe_allow_html=True)

replay = ss.pop("pending_replay", None)
if replay and ss.df is not None:
    with st.spinner("Re-running the saved steps..."):
        for e_ in replay:
            if run_step(e_["step"], e_["params"]) is None:
                st.warning(f"Stopped at {e_['step']}: the saved settings did not fit this data.")
                break
        else:
            ss.flash = [f"Re-ran {len(replay)} saved step(s)."]

df = ss.df
if df is None:
    st.write("")
    st.info("No dataset yet. Add one first, then prepare it here.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

t = ss.target if ss.target in df.columns else None
key = abs(hash((tuple(map(str, df.columns)), len(ss.snapshots), ss.split_done)))
focus = ss.pop("prepare_focus", None)
done = done_steps()
focus = focus or next((s for s in STEPS if s not in done), None)

flash = ss.pop("flash", None)
if flash:
    st.success("\n\n".join(flash))
if ss.split_done and split_target() != t:
    st.warning("The target column changed after the split. Run Split again so the steps use the new target.")
for item in ss.get("leakage", []):
    (st.error if item["level"] == "error" else st.warning)(item["message"])
for w_ in ss.get("prep_warnings", []):
    st.warning(w_)

m = st.columns(4)
m[0].metric("Rows", f"{len(df):,}")
m[1].metric("Columns", df.shape[1])
m[2].metric("Missing cells", f"{df.isna().sum().sum() / max(df.size, 1) * 100:.1f}%")
m[3].metric("Steps done", f"{len(done)} / {len(STEPS)}")

a, b, c = st.columns([2, 1, 1])
if a.button("Run all steps with recommended settings", type="primary", width="stretch"):
    with st.spinner("Preparing your data..."):
        apply_all()
has_steps = any(e.get("step") in STEPS for e in ss.log)
if b.button("Undo last step", disabled=not has_steps, width="stretch"):
    undo()
with c.popover("Reset to original", width="stretch", disabled=not has_steps):
    st.write("This removes every step and restores the data as it was loaded.")
    if st.button("Yes, reset", type="primary"):
        reset()

st.markdown('<div class="dfd"><h2>Steps</h2></div>', unsafe_allow_html=True)
for step in STEPS:
    with st.expander(f"{step}  ·  {'done' if step in done else 'to do'}", expanded=(step == focus)):
        st.caption(INFO[step])
        locked = step != "Split" and not ss.split_done
        if locked:
            st.info("Run Split first.")
            continue
        view = df if step == "Split" else stage_df(step)
        st.markdown(facts(step, view, t))
        params = ui(step, view, t, key)
        if step == "Split" and ss.split_done:
            parts = [("Train", ss.train_df), ("Validation", ss.val_df), ("Test", ss.test_df)]
            st.dataframe(pd.DataFrame([{"set": n_, "rows": len(d_), "share %": len(d_) / len(df) * 100}
                                       for n_, d_ in parts if d_ is not None]),
                         hide_index=True, width="stretch")
            st.caption("Stored as train_df, val_df and test_df in session state.")
        pv, rn, _ = st.columns([1, 1, 3])
        if pv.button(f"Preview {step}", key=f"prev_{step}_{key}"):
            preview(step, params)
        if rn.button(f"Run {step}", key=f"run_{step}_{key}", type="primary" if step == focus else "secondary"):
            apply_step(step, params)

st.write("")
n1, n2, _ = st.columns([1, 1, 2])
if n1.button("See the analysis", width="stretch"):
    go("dashboard")
