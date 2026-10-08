import html
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

PAGES = {
    "dashboard": "pages/dashboard.py",
}
NONE = "None (unsupervised)"
SOURCES = ["Upload file", "Paste data", "Sample data", "From URL", "Type it in"]
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1256", "cp1252"]
SEPS = {"Auto-detect": None, "Comma ,": ",", "Semicolon ;": ";", "Tab": "\t", "Pipe |": "|"}
KINDS = {"csv": "csv", "tsv": "csv", "txt": "csv", "xlsx": "excel", "xls": "excel",
         "json": "json", "parquet": "parquet"}
PREVIEW_ROWS = 200

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&display=swap');
.block-container{max-width:1080px;padding-top:2.5rem}
.dfd{--amber:#e8960c;--blue:#3b6cff;
--line:color-mix(in srgb,currentColor 16%,transparent);
--soft:color-mix(in srgb,currentColor 6%,transparent);
font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.dfd h1{font-size:44px;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.dfd h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:34px 0 10px;padding:0}
.dfd .sub{margin-top:8px;font-size:17px;opacity:.7;max-width:56ch}
.dfd .mono{font-family:'IBM Plex Mono',monospace;font-size:13px;opacity:.65}
.dfd-cur{display:flex;align-items:center;gap:14px;background:var(--soft);border-radius:12px;
padding:14px 18px;margin:18px 0 4px;border-left:4px solid var(--blue)}
.dfd-cur b{font-size:18px}
</style>
"""


def init_state():
    for k, v in {"df": None, "df_name": "untitled", "target": None, "log": [],
                 "snapshots": [], "modified_at": None, "split_done": False}.items():
        st.session_state.setdefault(k, v)


def go(page):
    try:
        st.switch_page(PAGES[page])
    except Exception:
        st.toast(f"Page not found: {PAGES[page]}")


def commit(df, name, source, target):
    ss = st.session_state
    now = datetime.now().strftime("%H:%M:%S")
    ss.df, ss.df_name, ss.target, ss.modified_at = df, name, target, now
    ss.snapshots = [{"name": "Loaded", "df": df.copy()}]
    ss.log = [{"time": now, "step": "Load", "detail": f"Loaded {name} ({source})",
               "rows": len(df), "cols": df.shape[1]}]
    ss.split_done = False
    ss.pop("prepare_focus", None)
    ss.just_loaded = True
    st.rerun()


def clear_dataset():
    ss = st.session_state
    ss.df, ss.target, ss.modified_at, ss.split_done = None, None, None, False
    ss.log, ss.snapshots = [], []


def tidy(df):
    """Strip column names and make them unique. Data itself is left untouched."""
    df, cols, seen = df.copy(), [], {}
    for i, c in enumerate(df.columns):
        c = str(c).strip() or f"column_{i + 1}"
        n = seen.get(c, 0)
        seen[c] = n + 1
        cols.append(c if n == 0 else f"{c}_{n}")
    df.columns = cols
    return df


@st.cache_data(show_spinner=False)
def sheet_names(raw):
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


@st.cache_data(show_spinner=False)
def parse_bytes(raw, kind, sep, enc, header, sheet):
    buf, h = io.BytesIO(raw), (0 if header else None)
    if kind == "csv":
        df = pd.read_csv(buf, sep=sep, engine="python" if sep is None else "c", encoding=enc, header=h)
    elif kind == "excel":
        df = pd.read_excel(buf, sheet_name=sheet, header=h)
    elif kind == "json":
        df = pd.read_json(buf)
    else:
        df = pd.read_parquet(buf)
    return tidy(df)


@st.cache_data(show_spinner=False, ttl=600)
def parse_url(url):
    u = url.lower().split("?")[0]
    if u.endswith(".json"):
        df = pd.read_json(url)
    elif u.endswith((".xlsx", ".xls")):
        df = pd.read_excel(url)
    elif u.endswith(".parquet"):
        df = pd.read_parquet(url)
    else:
        df = pd.read_csv(url, sep=None, engine="python")
    return tidy(df)


def stem(name):
    return name.rsplit("/", 1)[-1].split("?")[0].rsplit(".", 1)[0] or "dataset"


def sample_churn():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "age": rng.integers(18, 75, n).astype(float),
        "income": rng.lognormal(10.5, 0.6, n).round(0),
        "city": rng.choice(["Tehran", "tehran", "Shiraz", "Tabriz", "Isfahan"], n),
        "plan": rng.choice(["free", "pro", "team"], n, p=[0.7, 0.2, 0.1]),
        "tenure_months": rng.integers(1, 60, n),
        "support_calls": rng.poisson(2, n),
        "churned": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    df.loc[rng.choice(n, 60, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "income"] = np.nan
    return pd.concat([df, df.head(15)], ignore_index=True)


def sample_houses():
    rng = np.random.default_rng(2)
    n = 350
    area = rng.normal(120, 40, n).clip(30)
    df = pd.DataFrame({
        "area_m2": area.round(1),
        "rooms": rng.integers(1, 6, n).astype(float),
        "year_built": rng.integers(1960, 2024, n).astype(float),
        "district": rng.choice(["North", "South", "East", "West", "Center"], n),
        "has_parking": rng.choice(["yes", "no"], n),
    })
    df["price"] = (area * 2400 + rng.normal(0, 25000, n)).round(0)
    df.loc[rng.choice(n, 6, replace=False), "price"] *= 8
    df.loc[rng.choice(n, 40, replace=False), "year_built"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "rooms"] = np.nan
    return df


def sample_students():
    rng = np.random.default_rng(3)
    n = 250
    hours = rng.uniform(0, 12, n).round(1)
    score = (hours * 6 + rng.normal(25, 10, n)).clip(0, 100).round(0)
    df = pd.DataFrame({
        "hours_studied": hours,
        "attendance_pct": rng.uniform(40, 100, n).round(0),
        "school": rng.choice(["A", "B", "C"], n),
        "score": score,
        "passed": (score >= 50).astype(int),
    })
    df.loc[rng.choice(n, 30, replace=False), "attendance_pct"] = np.nan
    return df


SAMPLES = {
    "Customer churn": ("400 customers, missing ages and incomes, duplicate rows, inconsistent city names.", sample_churn),
    "House prices": ("350 houses with price outliers and missing values. A regression example.", sample_houses),
    "Student scores": ("250 students with a pass/fail target. Small and clean enough to learn on.", sample_students),
}


def csv_options(prefix):
    a, b, c = st.columns(3)
    sep = SEPS[a.selectbox("Separator", list(SEPS), key=f"{prefix}_sep")]
    enc = b.selectbox("Encoding", ENCODINGS, key=f"{prefix}_enc")
    header = c.checkbox("First row is the header", value=True, key=f"{prefix}_hdr")
    return sep, enc, header


def source_upload():
    f = st.file_uploader("Drop a file here or browse", type=list(KINDS))
    if not f:
        return None
    raw, kind = f.getvalue(), KINDS[f.name.rsplit(".", 1)[-1].lower()]
    sep = enc = sheet = None
    header = True
    try:
        with st.expander("Import options"):
            if kind == "csv":
                sep, enc, header = csv_options("up")
                enc = enc or "utf-8"
            elif kind == "excel":
                a, b = st.columns(2)
                sheet = a.selectbox("Sheet", sheet_names(raw))
                header = b.checkbox("First row is the header", value=True, key="xl_hdr")
            else:
                st.caption("No options for this file type.")
        if kind == "csv" and enc is None:
            enc = "utf-8"
        return parse_bytes(raw, kind, sep, enc, header, sheet), stem(f.name), f"file {f.name}"
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return None


def source_paste():
    text = st.text_area("Paste CSV or tab-separated text (copy straight from Excel or Sheets)",
                        height=200, placeholder="name,age,city\nAli,34,Tehran\nSara,29,Shiraz")
    sep, _, header = csv_options("paste")
    if not text.strip():
        return None
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python" if sep is None else "c",
                         header=0 if header else None)
        return tidy(df), "pasted_data", "pasted text"
    except Exception as e:
        st.error(f"Could not read this text: {e}")
        return None


def source_sample():
    key = st.radio("Pick a sample", list(SAMPLES), captions=[v[0] for v in SAMPLES.values()])
    return SAMPLES[key][1](), key.lower().replace(" ", "_"), f"sample {key}"


def source_url():
    url = st.text_input("Link to a CSV, Excel, JSON or Parquet file", placeholder="https://example.com/data.csv")
    if not url.strip():
        return None
    try:
        with st.spinner("Downloading..."):
            return parse_url(url.strip()), stem(url.strip()), "URL"
    except Exception as e:
        st.error(f"Could not load this link: {e}")
        return None


def source_manual():
    cols_text = st.text_input("Column names, separated by commas", value="name, age, city")
    cols = [c.strip() for c in cols_text.split(",") if c.strip()]
    if not cols:
        return None
    seed = pd.DataFrame({c: pd.Series([None] * 5, dtype=object) for c in cols})
    edited = st.data_editor(seed, num_rows="dynamic", use_container_width=True, key=f"manual_{cols_text}")
    df = edited.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    if df.empty:
        st.caption("Fill in at least one row to continue.")
        return None
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == df[c].notna().sum():
            df[c] = num
    return tidy(df), "manual_data", "typed in"


RENDER = {"Upload file": source_upload, "Paste data": source_paste, "Sample data": source_sample,
          "From URL": source_url, "Type it in": source_manual}


def section(title):
    st.markdown(f'<div class="dfd"><h2>{title}</h2></div>', unsafe_allow_html=True)


def fmt_bytes(n):
    return f"{n / 1024:.0f} KB" if n < 1024**2 else f"{n / 1024**2:.1f} MB"


def preview_and_load(df, default_name, source):
    section("Preview")
    m = st.columns(4)
    m[0].metric("Rows", f"{len(df):,}")
    m[1].metric("Columns", df.shape[1])
    m[2].metric("Size", fmt_bytes(int(df.memory_usage(deep=True).sum())))
    m[3].metric("Missing cells", f"{df.isna().sum().sum() / max(df.size, 1) * 100:.1f}%")
    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True, height=320)
    if len(df) > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS} of {len(df):,} rows.")
    with st.expander("Column details"):
        info = pd.DataFrame({"column": df.columns, "type": df.dtypes.astype(str).values,
                             "missing %": df.isna().mean().values * 100, "unique": df.nunique().values})
        st.dataframe(info, hide_index=True, use_container_width=True, column_config={
            "missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f%%")})

    section("Settings")
    sig = f"{source}-{df.shape}-{abs(hash(tuple(df.columns)))}"
    a, b = st.columns(2)
    name = a.text_input("Dataset name", value=default_name, key=f"name_{sig}")
    keep = b.multiselect("Columns to keep", list(df.columns), default=list(df.columns), key=f"keep_{sig}")
    target = st.selectbox("Target column (optional)", [NONE] + keep, key=f"target_{sig}",
                          help="The column your model will predict. Leave empty for unsupervised work.")
    if st.session_state.df is not None:
        st.warning("Loading a new dataset replaces the current one and clears its log.")
    if st.button("Use this dataset", type="primary", disabled=not keep or not name.strip()):
        commit(df[keep].copy(), name.strip(), source, None if target == NONE else target)


init_state()
ss = st.session_state
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="dfd"><h1>Dataset</h1><p class="sub">Add the data you want to prepare. '
            "Nothing is changed here; cleaning and transforming happen in later steps.</p></div>",
            unsafe_allow_html=True)

if ss.pop("just_loaded", False):
    st.success("Dataset loaded. Your pipeline starts from this data.")
    if st.button("Open dashboard", type="primary"):
        go("dashboard")

if ss.df is not None:
    st.markdown(
        f'<div class="dfd"><div class="dfd-cur"><div><div class="mono">current dataset</div>'
        f"<b>{html.escape(str(ss.df_name))}</b></div>"
        f'<div class="mono">{len(ss.df):,} rows &middot; {ss.df.shape[1]} columns'
        f'{" &middot; target: " + html.escape(str(ss.target)) if ss.target else ""}</div></div></div>',
        unsafe_allow_html=True,
    )
    a, b, _ = st.columns([1, 1, 4])
    if a.button("Open dashboard", key="cur_dash", use_container_width=True):
        go("dashboard")
    if b.button("Remove", key="cur_clear", use_container_width=True):
        clear_dataset()
        st.rerun()

section("Add data")
source = st.radio("Source", SOURCES, horizontal=True, label_visibility="collapsed")
result = RENDER[source]()
if result is not None:
    df, default_name, label = result
    if df.empty or df.shape[1] == 0:
        st.error("This data has no rows or no columns.")
    else:
        preview_and_load(df, default_name, label)