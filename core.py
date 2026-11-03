# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.

from __future__ import annotations
import io
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

APP_NAME = "DataForge"
VERSION = "v2.0.1"
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
    "advisor": "pages/advisor.py",
    "types": "pages/types.py",
    "lab": "pages/lab.py",
    "projects": "pages/projects.py",
    "performance": "pages/performance.py",
    "insights": "pages/insights.py",
    "compare": "pages/compare.py",
    "contract": "pages/contract.py",
    "pipeline": "pages/pipeline.py",
    "about": "pages/about.py",
}

KINDS = {"csv": "csv", "tsv": "table", "txt": "table", "xlsx": "excel", "xls": "excel",
         "json": "json", "parquet": "parquet"}
ENCODINGS = ("utf-8-sig", "cp1256", "latin-1")

SAMPLES = [
    {
        "key": "ames", "icon": "🏠", "title": "Ames Housing", "file": "Ames_Housing.csv",
        "task": "Regression", "target": "SalePrice",
        "blurb": "Home sales from Ames, Iowa, described by dozens of features. Predict the sale price.",
        "tags": ["Many columns", "Missing values", "Skewed prices"],
    },
    {
        "key": "telco", "icon": "📱", "title": "Telco Customer Churn", "file": "Telco_Customer_Churn.csv",
        "task": "Classification", "target": "Churn",
        "blurb": "Customers of a telecom company and their contracts. Predict who is going to leave.",
        "tags": ["Class imbalance", "Number stored as text", "ID column"],
    },
    {
        "key": "penguins", "icon": "🐧", "title": "Palmer Penguins", "file": "Palmer_Penguins.csv",
        "task": "Classification", "target": "species",
        "blurb": "Measurements of 344 penguins from three islands. Predict the species (three classes).",
        "tags": ["Three classes", "Missing values", "Numbers and text mixed"],
    },
]


def _defaults() -> dict:
    """A fresh dict every call, so lists and dicts are never shared between resets."""
    return {"df": None, "df_name": "untitled", "target": None, "log": [], "snapshots": [],
            "modified_at": None, "split_done": False, "artifacts": {}}


def _derived_defaults() -> dict:
    """Everything that belongs to one dataset: choices, decisions and settings. Fresh containers on every call."""
    return {"pipeline_cfg": {}, "cfg_stack": [], "decisions": {}, "advisor_params": {}, "var_types": None, "var_orders": {},
            "contract": None, "validation_status": None}


_TRANSIENT = ("train_df", "val_df", "test_df", "split_info", "prepare_focus", "flash", "just_loaded", "raw_splits", "pipeline",
              "leakage", "prep_warnings", "n_train", "effective_cfg", "visited", "_lab", "_opt", "_contract_file",
              "pending_replay", "pending_open")


def init_state() -> None:
    for k, v in {**_defaults(), **_derived_defaults()}.items():
        st.session_state.setdefault(k, v)


def clear_all() -> None:
    """Forget the dataset and everything derived from it (log, snapshots, splits, artifacts)."""
    ss = st.session_state
    for k, v in {**_defaults(), **_derived_defaults()}.items():
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

def _read_csv(
    raw: bytes,
    sep: str | None,
    encoding: str | None = None,
    header: int | None = 0,
) -> pd.DataFrame:
    encodings = (encoding,) if encoding else ENCODINGS
    err = None

    for enc in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                encoding=enc,
                engine="python" if sep is None else "c",
                header=header,
            )
        except UnicodeDecodeError as e:
            err = e

    if err is not None:
        raise err

    raise ValueError("Could not decode CSV file.")


@st.cache_data(show_spinner=False, max_entries=8)
def parse_data_bytes(
    raw: bytes,
    kind: str,
    sep: str | None = None,
    enc: str | None = None,
    header: bool = True,
    sheet: str | int | None = None,
    auto_detect_single_column: bool = False,
) -> pd.DataFrame:
    """Shared parser for uploaded, URL-based and other raw dataset inputs."""
    header_row = 0 if header else None

    if kind == "csv":
        df = _read_csv(raw, sep, enc, header_row)

        if auto_detect_single_column and sep == "," and df.shape[1] == 1:
            try:
                guessed = _read_csv(raw, None, enc, header_row)
                if guessed.shape[1] > 1:
                    df = guessed
            except Exception:
                pass

    elif kind == "table":
        df = _read_csv(raw, sep, enc, header_row)

    elif kind == "excel":
        df = pd.read_excel(
            io.BytesIO(raw),
            sheet_name=0 if sheet is None else sheet,
            header=header_row,
        )

    elif kind == "json":
        df = pd.read_json(io.BytesIO(raw))

    elif kind == "parquet":
        df = pd.read_parquet(io.BytesIO(raw))

    else:
        raise ValueError(f"Unsupported data kind: {kind}")

    if df.empty or df.shape[1] == 0:
        raise ValueError("The file has no rows or no columns.")

    return tidy(df)


@st.cache_data(show_spinner=False, max_entries=8)
def excel_sheet_names(raw: bytes) -> list[str]:
    """Return sheet names from an Excel workbook."""
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


@st.cache_data(show_spinner=False, max_entries=8)
def parse_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    """Read a dataset using its filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
    kind = KINDS.get(ext)

    if kind is None:
        raise ValueError(f"Unsupported file type .{ext}")

    return parse_data_bytes(
        raw,
        kind,
        sep="," if kind == "csv" else None,
        enc=None,
        header=True,
        sheet=0,
        auto_detect_single_column=(kind == "csv"),
    )

def read_upload(f) -> tuple[pd.DataFrame, str]:
    """f is a Streamlit UploadedFile. Returns (dataframe, dataset name)."""
    from dataforge.security import check_upload
    raw = f.getvalue()
    check_upload(f.name, raw)
    return parse_bytes(raw, f.name), stem(f.name)


def sample_path(spec: dict) -> Path | None:
    """The sample's file, or None. Case, spaces, dashes and underscores in the file name do not matter."""
    exact = SAMPLE_DIR / spec["file"]
    if exact.is_file():
        return exact
    if not SAMPLE_DIR.is_dir():
        return None
    norm = lambda s: re.sub(r"[\s_-]+", "", s).lower()
    return next((p for p in sorted(SAMPLE_DIR.iterdir()) if p.is_file() and norm(p.name) == norm(spec["file"])), None)


def load_sample(spec: dict) -> tuple[pd.DataFrame, str]:
    path = sample_path(spec)
    if path is None:
        raise FileNotFoundError(f"The file '{spec['file']}' is not in the sample_data folder ({SAMPLE_DIR}). "
                                "Copy it there, or upload your own file instead.")
    return parse_bytes(path.read_bytes(), path.name), stem(spec["file"])


def find_column(df: pd.DataFrame, wanted: str | None) -> str | None:
    """Find a column ignoring case, spaces and underscores ('Sale Price' matches 'SalePrice')."""
    if not wanted:
        return None
    norm = lambda s: re.sub(r"[\s_]+", "", str(s)).lower()
    return next((c for c in df.columns if norm(c) == norm(wanted)), None)

WORKFLOW = ["Import", "Inspect", "Analyze", "Prepare", "Validate", "Export"]
_PREP_STEPS = {"Split", "Clean", "Transform", "Encode", "Scale"}


def workflow_bar(current: str) -> None:
    """Shows Import > Inspect > Analyze > Prepare > Validate > Export and what to do next."""
    ss = st.session_state
    seen = ss.setdefault("visited", set())
    seen.add(current)
    steps = {e.get("step") for e in ss.get("log", [])}
    done = {
        "Import": ss.get("df") is not None,
        "Inspect": ss.get("target") is not None or "Inspect" in seen,
        "Analyze": "Analyze" in seen,
        "Prepare": _PREP_STEPS <= steps,
        "Validate": ss.get("validation_status") in ("passed", "warnings"),
        "Export": "Export" in steps,
    }
    nxt = next((s for s in WORKFLOW if not done[s] and s != "Analyze"), None)
    chips = ""
    for s in WORKFLOW:
        mark = "✓ " if done[s] else ""
        style = ("font-weight:700;border:1.5px solid currentColor;" if s == current else
                 "opacity:.9;" if done[s] else "opacity:.5;")
        chips += (f"<span style='display:inline-block;padding:3px 12px;margin:0 6px 6px 0;border-radius:99px;"
                  f"border:1px solid color-mix(in srgb,currentColor 25%,transparent);font-size:13px;{style}'>{mark}{s}</span>")
    hint = f"<div style='font-size:13px;opacity:.65'>Next: <b>{nxt}</b></div>" if nxt else \
           "<div style='font-size:13px;opacity:.65'>Every stage is done.</div>"
    st.markdown(f"<div>{chips}{hint}</div>", unsafe_allow_html=True)
