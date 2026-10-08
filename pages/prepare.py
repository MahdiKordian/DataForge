import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

PAGES = {"dataset": "pages/dataset.py", "dashboard": "pages/dashboard.py"}
STEPS = ["Clean", "Transform", "Encode", "Scale", "Split"]
INFO = {
    "Clean": "Fix text, duplicates, constant or mostly empty columns, and missing values.",
    "Transform": "Handle outliers and skew, and drop columns that add nothing.",
    "Encode": "Turn text categories into numbers a model can use.",
    "Scale": "Put numeric columns on a similar range.",
    "Split": "Create train, validation and test sets. Do this last.",
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


def f_clean(df, p, t, art):
    notes = []
    text = [c for c in df.columns if is_text(df[c])]
    changed = 0
    for c in text:
        s = df[c].astype("string")
        if p["trim"]:
            s = s.str.strip()
        s = {"lowercase": s.str.lower, "UPPERCASE": s.str.upper, "Title Case": s.str.title}.get(p["case"], lambda: s)()
        s = s.mask(s.fillna("x") == "", pd.NA)
        new = s.astype(object).where(s.notna(), np.nan)
        changed += int(((new != df[c]) & ~(new.isna() & df[c].isna())).sum())
        df[c] = new
    if changed:
        notes.append(f"tidied {changed} text cells")
    if p["to_num"]:
        conv = []
        for c in text:
            num = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().any() and num.notna().sum() == df[c].notna().sum():
                df[c], conv = num, conv + [c]
        if conv:
            notes.append(f"converted {len(conv)} text column(s) to numbers")
    if t and p["drop_missing_target"] and df[t].isna().any():
        k = int(df[t].isna().sum())
        df = df[df[t].notna()]
        notes.append(f"removed {k} rows with a missing target")
    if p["dups"] and df.duplicated().any():
        k = int(df.duplicated().sum())
        df = df.drop_duplicates()
        notes.append(f"removed {k} duplicate rows")
    drop = []
    if p["max_missing"] < 100:
        drop += [c for c in df.columns if c != t and df[c].isna().mean() * 100 > p["max_missing"]]
    if p["const"]:
        drop += [c for c in df.columns if c != t and df[c].nunique(dropna=True) <= 1 and c not in drop]
    if drop:
        df = df.drop(columns=drop)
        notes.append(f"dropped {len(drop)} column(s): {', '.join(map(str, drop[:5]))}")
    filled, rows_cols = 0, []
    for c in df.columns:
        if c == t or not df[c].isna().any() or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        strat = p["num_missing"] if pd.api.types.is_numeric_dtype(df[c]) else p["cat_missing"]
        s, v = df[c], None
        if strat == "Drop rows":
            rows_cols.append(c)
            continue
        if strat == "Median":
            v = s.median()
        elif strat == "Mean":
            v = s.mean()
        elif strat == "Most common":
            v = s.mode().iloc[0] if s.notna().any() else None
        elif strat == "Fill with 0":
            v = 0
        elif strat.startswith("Fill with 'U"):
            v = "Unknown"
        if v is not None and not pd.isna(v):
            filled += int(s.isna().sum())
            df[c] = s.fillna(v)
    if rows_cols:
        k = int(df[rows_cols].isna().any(axis=1).sum())
        df = df.dropna(subset=rows_cols)
        notes.append(f"dropped {k} rows with missing values")
    if filled:
        notes.append(f"filled {filled} missing values")
    return df.reset_index(drop=True), notes


def f_transform(df, p, t, art):
    notes = []
    cols = [c for c in (p["out_cols"] or continuous(df, t)) if c in df.columns]
    m = p["outliers"]
    if m != "Leave as is":
        mask, clipped = pd.Series(False, index=df.index), 0
        for c in cols:
            s = df[c]
            if m.startswith("Clip at 1st"):
                lo, hi = s.quantile([0.01, 0.99])
            else:
                q1, q3 = s.quantile([0.25, 0.75])
                if q3 == q1:
                    continue
                lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            out = (s < lo) | (s > hi)
            if m.startswith("Remove"):
                mask |= out
            else:
                clipped += int(out.sum())
                df[c] = s.clip(lo, hi)
        if m.startswith("Remove") and mask.any():
            notes.append(f"removed {int(mask.sum())} rows with outliers")
            df = df[~mask]
        elif clipped:
            notes.append(f"clipped {clipped} outlier values")
    if p["skew"]:
        fixed = []
        for c in continuous(df, t):
            s = df[c]
            if s.dropna().skew() > p["skew_thr"]:
                x = s + max(0, -s.min())
                df[c] = np.log1p(x) if p["skew_method"].startswith("Log") else np.sqrt(x)
                fixed.append(c)
        if fixed:
            notes.append(f"reduced right skew in {len(fixed)} column(s): {', '.join(map(str, fixed[:5]))}")
    if p["corr"]:
        n = [c for c in nums(df) if c != t]
        if len(n) > 1:
            cm = df[n].corr().abs()
            up = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
            hi = [c for c in up.columns if (up[c] > p["corr"]).any()]
            if hi:
                df = df.drop(columns=hi)
                notes.append(f"dropped {len(hi)} near-duplicate column(s): {', '.join(map(str, hi[:5]))}")
    manual = [c for c in p["drop"] if c in df.columns and c != t]
    if manual:
        df = df.drop(columns=manual)
        notes.append(f"dropped {len(manual)} column(s) you chose")
    return df.reset_index(drop=True), notes


def f_encode(df, p, t, art):
    notes, maps = [], {}
    for c in [c for c in df.columns if c != t and pd.api.types.is_bool_dtype(df[c])]:
        df[c] = df[c].astype(int)
    if p["dates"]:
        for c in [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]:
            for part in ("year", "month", "day", "weekday"):
                df[f"{c}_{part}"] = getattr(df[c].dt, "dayofweek" if part == "weekday" else part)
            df = df.drop(columns=c)
            notes.append(f"split date column `{c}` into year, month, day and weekday")
    cat = [c for c in df.columns if c != t and is_text(df[c])]
    low = [c for c in cat if df[c].nunique() <= p["max_onehot"]]
    if low and p["low"] == "One-hot":
        df = pd.get_dummies(df, columns=low, drop_first=p["drop_first"], dtype=int)
        notes.append(f"one-hot encoded {len(low)} column(s)")
    elif low:
        for c in low:
            df[c], maps[c] = ordinal(df[c])
        notes.append(f"ordinal encoded {len(low)} column(s)")
    for c in [c for c in cat if c not in low]:
        if p["high"] == "Frequency":
            df[c] = df[c].map(df[c].value_counts(normalize=True))
        elif p["high"] == "Ordinal":
            df[c], maps[c] = ordinal(df[c])
        else:
            df = df.drop(columns=c)
    if len(cat) > len(low):
        notes.append(f"{len(cat) - len(low)} high-cardinality column(s): {p['high'].lower()}")
    if t and p["encode_target"] and is_text(df[t]):
        df[t], maps[t] = ordinal(df[t])
        notes.append(f"encoded target `{t}` as {len(maps[t])} numeric classes")
    art["maps"] = maps
    return df, notes


def f_scale(df, p, t, art):
    if p["method"] == SCALERS[3]:
        return df, []
    cols = [c for c in (p["cols"] or continuous(df, t)) if c in df.columns]
    params, done = {}, 0
    for c in cols:
        s = df[c].astype(float)
        if p["method"].startswith("Standard"):
            a, b = s.mean(), s.std()
        elif p["method"].startswith("Min"):
            a, b = s.min(), s.max() - s.min()
        else:
            a, b = s.median(), s.quantile(0.75) - s.quantile(0.25)
        if b and not pd.isna(b):
            df[c], params[c], done = (s - a) / b, {"center": a, "scale": b}, done + 1
    art["params"] = params
    return df, [f"scaled {done} column(s) with {p['method'].split(' (')[0].lower()} scaling"] if done else []


def do_split(df, p, t):
    """Returns an error string, or (train, val, test, notes)."""
    clf = t is not None and is_class(df[t].dropna())
    rng = np.random.default_rng(int(p["seed"]))
    codes = pd.factorize(df[t])[0] if (p["stratify"] and clf) else np.zeros(len(df), int)
    tr, va, te = [], [], []
    for k in np.unique(codes):
        g = rng.permutation(np.flatnonzero(codes == k))
        nt, nv = int(round(len(g) * p["test"] / 100)), int(round(len(g) * p["val"] / 100))
        te += list(g[:nt])
        va += list(g[nt:nt + nv])
        tr += list(g[nt + nv:])
    if not tr or not te:
        return "The split left an empty train or test set. Use smaller percentages or more rows."
    train, test = df.iloc[rng.permutation(tr)], df.iloc[rng.permutation(te)]
    val = df.iloc[rng.permutation(va)] if va else None
    notes = [f"train {len(train):,} / " + (f"validation {len(val):,} / " if val is not None else "") + f"test {len(test):,}"]
    if clf and p["balance"] != "None":
        cnt = train[t].value_counts()
        n = cnt.max() if p["balance"].startswith("Over") else cnt.min()
        train = pd.concat([g.sample(n, replace=len(g) < n, random_state=int(p["seed"])) for _, g in train.groupby(t)])
        train = train.sample(frac=1, random_state=int(p["seed"]))
        notes.append(f"balanced the training set by {p['balance'].split()[0].lower()} ({len(train):,} rows)")
    r = lambda d: None if d is None else d.reset_index(drop=True)
    return r(train), r(val), r(test), notes


FUNCS = {"Clean": f_clean, "Transform": f_transform, "Encode": f_encode, "Scale": f_scale}


def clear_split():
    ss = st.session_state
    ss.split_done = False
    for k in ("train_df", "val_df", "test_df", "split_info"):
        ss.pop(k, None)


def done_steps():
    return {e.get("step") for e in st.session_state.log} & set(STEPS)


def run_step(step, p):
    """Applies one step, logs it and snapshots it. Returns notes, or None on error."""
    ss = st.session_state
    df, t = ss.df, (ss.target if ss.target in ss.df.columns else None)
    art, extra = {}, {}
    if step == "Split":
        res = do_split(df, p, t)
        if isinstance(res, str):
            st.error(res)
            return None
        train, val, test, notes = res
        ss.train_df, ss.val_df, ss.test_df = train, val, test
        ss.split_info = {"test_pct": p["test"], "val_pct": p["val"], "seed": p["seed"], "balance": p["balance"]}
        ss.split_done = True
        new = df
    else:
        if ss.split_done:
            clear_split()
            st.toast("The train/test split was reset because the data changed. Run Split again.")
        new, notes = FUNCS[step](df.copy(), p, t, art)
        ss.df = new.reset_index(drop=True)
        ss.snapshots.append({"name": f"After {step.lower()}", "df": ss.df.copy()})
        ss.setdefault("artifacts", {})[step] = art
    now = datetime.now().strftime("%H:%M:%S")
    ss.modified_at = now
    ss.log.append({"time": now, "step": step, "detail": "; ".join(notes).capitalize() if notes else "No changes needed",
                   "rows": len(ss.df), "cols": ss.df.shape[1], "params": jsonable(p)})
    return notes


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
        notes = run_step(step, dict(DEFAULTS[step]))
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
    if entry["step"] != "Split" and len(ss.snapshots) > 1:
        ss.snapshots.pop()
        ss.df = ss.snapshots[-1]["df"].copy()
    clear_split()
    ss.modified_at = datetime.now().strftime("%H:%M:%S")
    ss.flash = [f"Undid **{entry['step']}**."]
    st.rerun()


def reset():
    ss = st.session_state
    ss.df = ss.snapshots[0]["df"].copy()
    ss.snapshots = ss.snapshots[:1]
    ss.log = [e for e in ss.log if e.get("step") not in STEPS]
    ss.artifacts = {}
    clear_split()
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
    d, p = DEFAULTS[step], {}
    a, b = st.columns(2)
    if step == "Clean":
        p["trim"] = a.checkbox("Trim spaces in text", d["trim"], key=k("trim"))
        p["case"] = a.selectbox("Text case", CASES, key=k("case"))
        p["to_num"] = a.checkbox("Convert text that looks like numbers", d["to_num"], key=k("to_num"))
        p["dups"] = a.checkbox("Remove duplicate rows", d["dups"], key=k("dups"))
        p["const"] = a.checkbox("Drop columns with a single value", d["const"], key=k("const"))
        p["max_missing"] = b.slider("Drop columns missing more than (%)", 0, 100, d["max_missing"], key=k("mm"),
                                    help="100 keeps every column.")
        p["num_missing"] = b.selectbox("Missing numbers", NUM_MISS, key=k("nm"))
        p["cat_missing"] = b.selectbox("Missing text", CAT_MISS, key=k("cm"))
        p["drop_missing_target"] = bool(t) and b.checkbox("Remove rows where the target is missing", True, key=k("tm"))
    elif step == "Transform":
        cont = continuous(df, t)
        p["outliers"] = a.selectbox("Outliers", OUTLIERS, index=1, key=k("out"))
        p["out_cols"] = a.multiselect("Apply to columns", cont, default=cont, key=k("oc"))
        p["skew"] = b.checkbox("Reduce right skew", d["skew"], key=k("skew"))
        p["skew_thr"] = b.slider("Skew above", 0.5, 3.0, d["skew_thr"], 0.1, key=k("st"))
        p["skew_method"] = b.selectbox("Method", ["Log (log1p)", "Square root"], key=k("sm"))
        use = a.checkbox("Drop near-duplicate columns", True, key=k("cu"))
        p["corr"] = b.slider("When |correlation| is above", 0.7, 0.99, d["corr"], 0.01, key=k("cr")) if use else None
        p["drop"] = st.multiselect("Also drop these columns", [c for c in df.columns if c != t], key=k("drop"))
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
        st.caption("Statistics come from the full dataset. For strictly leak-free modelling, fit the scaler on the training set only.")
    else:
        clf = bool(t) and is_class(df[t].dropna())
        p["test"] = a.slider("Test set (%)", 5, 40, d["test"], key=k("te"))
        p["val"] = a.slider("Validation set (%)", 0, 30, d["val"], key=k("va"), help="0 means no validation set.")
        p["seed"] = b.number_input("Random seed", 0, 10_000, d["seed"], key=k("sd"))
        p["stratify"] = clf and b.checkbox("Keep class proportions (stratify)", True, key=k("sf"))
        p["balance"] = b.selectbox("Balance the training set", ["None", "Oversample minority", "Undersample majority"],
                                   key=k("bal")) if clf else "None"
        if not clf:
            st.caption("Pick a classification target on the Dashboard to enable stratifying and balancing.")
    return p


ss = st.session_state
for k_, v_ in {"df": None, "df_name": "untitled", "target": None, "log": [], "snapshots": [],
               "modified_at": None, "split_done": False, "artifacts": {}}.items():
    ss.setdefault(k_, v_)
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="dfd"><h1>Prepare</h1><p>Run the steps one by one, or run them all with '
            "recommended settings. Every step is logged and can be undone.</p></div>", unsafe_allow_html=True)

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

m = st.columns(4)
m[0].metric("Rows", f"{len(df):,}")
m[1].metric("Columns", df.shape[1])
m[2].metric("Missing cells", f"{df.isna().sum().sum() / max(df.size, 1) * 100:.1f}%")
m[3].metric("Steps done", f"{len(done)} / {len(STEPS)}")

a, b, c = st.columns([2, 1, 1])
if a.button("Run all steps with recommended settings", type="primary", use_container_width=True):
    with st.spinner("Preparing your data..."):
        apply_all()
has_steps = any(e.get("step") in STEPS for e in ss.log)
if b.button("Undo last step", disabled=not has_steps, use_container_width=True):
    undo()
with c.popover("Reset to original", use_container_width=True, disabled=not has_steps):
    st.write("This removes every step and restores the data as it was loaded.")
    if st.button("Yes, reset", type="primary"):
        reset()

st.markdown('<div class="dfd"><h2>Steps</h2></div>', unsafe_allow_html=True)
for step in STEPS:
    with st.expander(f"{step}  ·  {'done' if step in done else 'to do'}", expanded=(step == focus)):
        st.caption(INFO[step])
        st.markdown(facts(step, df, t))
        params = ui(step, df, t, key)
        if step == "Split" and ss.split_done:
            parts = [("Train", ss.train_df), ("Validation", ss.val_df), ("Test", ss.test_df)]
            st.dataframe(pd.DataFrame([{"set": n_, "rows": len(d_), "share %": len(d_) / len(df) * 100}
                                       for n_, d_ in parts if d_ is not None]),
                         hide_index=True, use_container_width=True)
            st.caption("Stored as train_df, val_df and test_df in session state.")
        if st.button(f"Run {step}", key=f"run_{step}_{key}", type="primary" if step == focus else "secondary"):
            apply_step(step, params)

st.write("")
n1, n2, _ = st.columns([1, 1, 2])
if n1.button("See the analysis", use_container_width=True):
    go("dashboard")