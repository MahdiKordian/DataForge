
from __future__ import annotations
import io
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

APP_NAME = "DataForge"
VERSION = "v1.0.0"
NONE = "None (unsupervised)"

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"

PAGES = {
    "landing": "pages/landing.py",
    "dataset": "pages/dataset.py",
    "dashboard": "pages/dashboard.py",
    "prepare": "pages/prepare.py",
    "analysis": "pages/analysis.py",
    "export": "pages/export.py",
    "report": "pages/report.py",
    "about": "pages/about.py",
}

KINDS = {"csv": "csv", "tsv": "table", "txt": "table", "xlsx": "excel", "xls": "excel",
         "json": "json", "parquet": "parquet"}
ENCODINGS = ("utf-8-sig", "cp1256", "latin-1")

SAMPLES = [
    {
        "key": "ames", "icon": "🏠", "title": "Ames Housing", "file": "Ames Housing.csv",
        "task": "Regression", "target": "SalePrice",
        "blurb": "Home sales from Ames, Iowa, described by dozens of features. Predict the sale price.",
        "tags": ["Many columns", "Missing values", "Skewed prices"],
    },
    {
        "key": "telco", "icon": "📱", "title": "Telco Customer Churn", "file": "Telco Customer Churn.csv",
        "task": "Classification", "target": "Churn",
        "blurb": "Customers of a telecom company and their contracts. Predict who is going to leave.",
        "tags": ["Class imbalance", "Number stored as text", "ID column"],
    },
    {
        "key": "penguins", "icon": "🐧", "title": "Palmer Penguins", "file": "Palmer Penguins.csv",
        "task": "Classification", "target": "species",
        "blurb": "Measurements of 344 penguins from three islands. Predict the species (three classes).",
        "tags": ["Three classes", "Missing values", "Numbers and text mixed"],
    },
]


def _defaults() -> dict:
    """A fresh dict every call, so lists and dicts are never shared between resets."""
    return {"df": None, "df_name": "untitled", "target": None, "log": [], "snapshots": [],
            "modified_at": None, "split_done": False, "artifacts": {}}


_TRANSIENT = ("train_df", "val_df", "test_df", "split_info", "prepare_focus", "flash", "just_loaded")


def init_state() -> None:
    for k, v in _defaults().items():
        st.session_state.setdefault(k, v)


def clear_all() -> None:
    """Forget the dataset and everything derived from it (log, snapshots, splits, artifacts)."""
    ss = st.session_state
    for k, v in _defaults().items():
        ss[k] = v
    for k in _TRANSIENT:
        ss.pop(k, None)


def commit(df: pd.DataFrame, name: str, source: str, target: str | None = None) -> None:
    """Make df the current dataset with a clean history (same contract as the Dataset page)."""
    clear_all()
    ss = st.session_state
    now = datetime.now().strftime("%H:%M:%S")
    ss.df, ss.df_name, ss.modified_at = df, name, now
    ss.target = target if target in df.columns else None
    ss.snapshots = [{"name": "Loaded", "df": df.copy()}]
    ss.log = [{"time": now, "step": "Load", "detail": f"Loaded {name} ({source})",
               "rows": len(df), "cols": df.shape[1]}]


def go(page: str, **state) -> None:
    st.session_state.update(state)
    try:
        st.switch_page(PAGES[page])
    except Exception:
        st.toast(f"Page not found: {PAGES[page]}")


def stem(name: str) -> str:
    return Path(str(name).split("?")[0]).stem or "dataset"


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and make them unique. The data itself is left untouched."""
    df, cols, seen = df.copy(), [], {}
    for i, c in enumerate(df.columns):
        c = str(c).strip() or f"column_{i + 1}"
        n = seen.get(c, 0)
        seen[c] = n + 1
        cols.append(c if n == 0 else f"{c}_{n}")
    df.columns = cols
    return df


def _read_csv(raw: bytes, sep: str | None) -> pd.DataFrame:
    err: Exception | None = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc,
                               engine="python" if sep is None else "c")
        except UnicodeDecodeError as e:
            err = e
    raise err


@st.cache_data(show_spinner=False, max_entries=8)
def parse_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
    kind = KINDS.get(ext)
    if kind is None:
        raise ValueError(f"unsupported file type .{ext}")
    if kind == "csv":
        df = _read_csv(raw, ",")
        if df.shape[1] == 1:
            try:
                guess = _read_csv(raw, None)
                if guess.shape[1] > 1:
                    df = guess
            except Exception:
                pass
    elif kind == "table":
        df = _read_csv(raw, None)
    elif kind == "excel":
        df = pd.read_excel(io.BytesIO(raw))
    elif kind == "json":
        df = pd.read_json(io.BytesIO(raw))
    else:
        df = pd.read_parquet(io.BytesIO(raw))
    if df.empty or df.shape[1] == 0:
        raise ValueError("the file has no rows or no columns")
    return tidy(df)


def read_upload(f) -> tuple[pd.DataFrame, str]:
    """f is a Streamlit UploadedFile. Returns (dataframe, dataset name)."""
    return parse_bytes(f.getvalue(), f.name), stem(f.name)


def load_sample(spec: dict) -> tuple[pd.DataFrame, str]:
    path = SAMPLE_DIR / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"sample_data/{spec['file']} was not found")
    return parse_bytes(path.read_bytes(), spec["file"]), stem(spec["file"])


def find_column(df: pd.DataFrame, wanted: str | None) -> str | None:
    """Find a column ignoring case, spaces and underscores ('Sale Price' matches 'SalePrice')."""
    if not wanted:
        return None
    norm = lambda s: re.sub(r"[\s_]+", "", str(s)).lower()
    return next((c for c in df.columns if norm(c) == norm(wanted)), None)