# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Projects: save and reopen a whole piece of work (settings, decisions, optionally the data) in a local folder."""
from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .scale import fingerprint
from .security import inside, safe_project_name

FORMAT, VERSION = "dataforge-project", 1
SETTINGS_KEYS = ["target", "var_types", "var_orders", "contract", "decisions", "advisor_params", "split_info"]


def root() -> Path:
    r = Path(os.environ.get("DATAFORGE_HOME", Path.home() / ".dataforge")) / "projects"
    r.mkdir(parents=True, exist_ok=True)
    return r


def _dir(name: str, must_exist: bool = False) -> Path:
    p = inside(root(), root() / safe_project_name(name))
    if must_exist and not (p / "project.json").exists():
        raise FileNotFoundError(f"Project '{name}' was not found.")
    return p


def _clean(x):
    """Make settings JSON-safe."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def save_project(name: str, df: pd.DataFrame, state: dict, store_data: bool = True, notes: str = "") -> Path:
    """state holds the settings keys, plus 'steps': [{step, params}] to replay. Written atomically."""
    p = _dir(name)
    tmp = p.with_name(p.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    meta = {"format": FORMAT, "version": VERSION, "name": name, "saved": datetime.now().isoformat(timespec="seconds"), "notes": notes[:2000],
            "dataset": {"rows": len(df), "columns": df.shape[1], "fingerprint": fingerprint(df), "column_names": list(map(str, df.columns)),
                        "dtypes": [str(t) for t in df.dtypes], "stored": bool(store_data)},
            "settings": {k: _clean(state.get(k)) for k in SETTINGS_KEYS}, "steps": _clean(state.get("steps", []))}
    if store_data:
        d = df.copy()
        d.columns = d.columns.map(str)
        d.to_parquet(tmp / "data.parquet", compression="zstd", index=False)
    (tmp / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if p.exists():
        shutil.rmtree(p)
    os.replace(tmp, p)
    return p


def list_projects() -> list[dict]:
    out = []
    for d in sorted(root().iterdir()):
        f = d / "project.json"
        if d.is_dir() and not d.name.endswith(".tmp") and f.exists():
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
                out.append({"name": m["name"], "saved": m["saved"], "rows": m["dataset"]["rows"], "columns": m["dataset"]["columns"],
                            "target": m["settings"].get("target"), "steps": len(m.get("steps", [])), "data stored": m["dataset"]["stored"]})
            except Exception:
                continue
    return sorted(out, key=lambda r: r["saved"], reverse=True)


def load_project(name: str):
    """Returns (meta, DataFrame or None). Raises if the stored data does not match its fingerprint."""
    p = _dir(name, must_exist=True)
    meta = json.loads((p / "project.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError("This is not a project this version of DataForge can open.")
    df = None
    if meta["dataset"]["stored"]:
        df = pd.read_parquet(p / "data.parquet")
        if fingerprint(df) != meta["dataset"]["fingerprint"]:
            raise ValueError("The stored data does not match the project record (the file was changed or damaged).")
    return meta, df


def delete_project(name: str) -> None:
    shutil.rmtree(_dir(name, must_exist=True))


def export_zip(name: str) -> bytes:
    p = _dir(name, must_exist=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in p.iterdir():
            z.write(f, f"{p.name}/{f.name}")
    return buf.getvalue()
