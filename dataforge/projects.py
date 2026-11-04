# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Projects: save and reopen a whole piece of work (settings, decisions, optionally the data) in a local folder."""
from __future__ import annotations

import tempfile
import uuid
import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .scale import content_fingerprint, fingerprint
from .security import inside, safe_project_name

FORMAT, VERSION = "dataforge-project", 1
SETTINGS_KEYS = ["target", "var_types", "var_orders", "contract", "decisions", "advisor_params", "split_info"]


def root() -> Path:
    r = Path(os.environ.get("DATAFORGE_HOME", Path.home() / ".dataforge")) / "projects"
    r.mkdir(parents=True, exist_ok=True)
    return r



def _dir(name: str, must_exist: bool = False) -> Path:
    base = root()
    safe = safe_project_name(name)
    p = inside(base, base / safe)

    backups = sorted(
        (
            item for item in base.glob(f".{safe}.backup-*")
            if item.is_dir() and not item.is_symlink()
        ),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )

    if backups and not (p / "project.json").is_file():
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

        os.replace(backups[0], p)

    if must_exist and not (p / "project.json").is_file():
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


def save_project(
    name: str,
    df: pd.DataFrame,
    state: dict,
    store_data: bool = True,
    notes: str = "",
) -> Path:
    """Save a project using a staging directory and backup recovery."""
    p = _dir(name)

    tmp = Path(
        tempfile.mkdtemp(
            prefix=f".{p.name}.tmp-",
            dir=p.parent,
        )
    )

    backup = None

    try:
        meta = {
            "format": FORMAT,
            "version": VERSION,
            "name": name,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "notes": notes[:2000],
            "dataset": {
                "rows": len(df),
                "columns": df.shape[1],
                "fingerprint": fingerprint(df),
                "content_fingerprint": content_fingerprint(df),
                "column_names": list(map(str, df.columns)),
                "dtypes": [str(t) for t in df.dtypes],
                "stored": bool(store_data),
            },
            "settings": {
                k: _clean(state.get(k))
                for k in SETTINGS_KEYS
            },
            "steps": _clean(state.get("steps", [])),
        }

        if store_data:
            d = df.copy()
            d.columns = d.columns.map(str)

            d.to_parquet(
                tmp / "data.parquet",
                compression="zstd",
                index=False,
            )

        (tmp / "project.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

        if p.exists():
            backup = p.with_name(
                f".{p.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(p, backup)

        os.replace(tmp, p)

    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

        if backup is not None and backup.exists() and not p.exists():
            os.replace(backup, p)

        raise

    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

    return p



def list_projects() -> list[dict]:
    base = root()

    for backup in list(base.glob(".*.backup-*")):
        if not backup.is_dir() or backup.is_symlink():
            continue

        name = backup.name[1:].rsplit(".backup-", 1)[0]

        try:
            _dir(name)
        except ValueError:
            continue

    def project_summary(directory: Path) -> dict:
        meta = json.loads(
            (directory / "project.json").read_text(encoding="utf-8")
        )

        return {
            "name": meta["name"],
            "saved": meta["saved"],
            "rows": meta["dataset"]["rows"],
            "columns": meta["dataset"]["columns"],
            "target": meta["settings"].get("target"),
            "steps": len(meta.get("steps", [])),
            "data stored": meta["dataset"]["stored"],
        }

    out = []

    for directory in sorted(base.iterdir()):
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or directory.name.startswith(".")
            or directory.name.endswith(".tmp")
        ):
            continue

        try:
            summary = project_summary(directory)

        except Exception:
            try:
                load_project(directory.name)
                summary = project_summary(directory)
            except Exception:
                continue

        out.append(summary)

    return sorted(out, key=lambda item: item["saved"], reverse=True)




def _read_project(p: Path):
    """Read and validate a project directory."""
    meta = json.loads((p / "project.json").read_text(encoding="utf-8"))

    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError("This is not a project this version of DataForge can open.")

    df = None

    if meta["dataset"]["stored"]:
        df = pd.read_parquet(p / "data.parquet")

        expected = meta["dataset"].get("content_fingerprint")
        actual = content_fingerprint(df) if expected else fingerprint(df)

        if actual != (expected or meta["dataset"]["fingerprint"]):
            raise ValueError(
                "The stored data does not match the project record "
                "(the file was changed or damaged)."
            )

    return meta, df


def load_project(name: str):
    """Load a project and recover from a valid backup if necessary."""
    p = _dir(name, must_exist=True)

    try:
        result = _read_project(p)

    except Exception as original_error:
        backups = sorted(
            (
                item for item in p.parent.glob(f".{p.name}.backup-*")
                if item.is_dir() and not item.is_symlink()
            ),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )

        for backup in backups:
            try:
                result = _read_project(backup)
            except Exception:
                continue

            failed = p.with_name(
                f".{p.name}.corrupt-{uuid.uuid4().hex}"
            )

            if p.exists():
                os.replace(p, failed)

            try:
                os.replace(backup, p)
            except Exception:
                if failed.exists() and not p.exists():
                    os.replace(failed, p)
                raise

            if failed.exists():
                shutil.rmtree(failed, ignore_errors=True)

            break
        else:
            raise original_error

    for backup in p.parent.glob(f".{p.name}.backup-*"):
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup, ignore_errors=True)

    return result



def delete_project(name: str) -> None:
    shutil.rmtree(_dir(name, must_exist=True))


def export_zip(name: str) -> bytes:
    p = _dir(name, must_exist=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in p.iterdir():
            z.write(f, f"{p.name}/{f.name}")
    return buf.getvalue()
