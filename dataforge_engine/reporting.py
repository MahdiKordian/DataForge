# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Content for the three report levels: executive, analytical, technical."""
from __future__ import annotations

import platform

import numpy as np
import pandas as pd

from . import analysis as A


def key_points(df, target, ready: dict | None, leakage: list | None, validation_status: str | None,
               split_done: bool, types: dict | None = None) -> list[str]:
    """Short, quantified statements for someone who will not read the tables."""
    out = []
    q = A.quality_checks(df, target)
    err, warn = sum(i["level"] == "error" for i in q), sum(i["level"] == "warning" for i in q)
    out.append(f"{len(q)} data-quality issue(s) found ({err} serious, {warn} to review)." if q else "No data-quality issues were found.")
    miss = A.na_frame(df).mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        top = ", ".join(f"{c} ({v * 100:.1f}%)" if v < 0.1 else f"{c} ({v * 100:.0f}%)" for c, v in miss.head(3).items())
        out.append(f"{len(miss)} column(s) have missing values. The largest gaps: {top}.")
    else:
        out.append("No missing values.")
    if types:
        low = [c for c, v in types.items() if v.get("confidence") == "low"]
        if low:
            out.append(f"{len(low)} column type(s) are uncertain and need a human decision: {', '.join(map(str, low[:4]))}.")
    if target and target in df.columns:
        s = df[target].dropna()
        if len(s) and (not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 15):
            sh = s.value_counts(normalize=True)
            out.append(f"Target `{target}`: {len(sh)} classes; the smallest is {sh.min() * 100:.1f}% of the rows.")
    out.append("The data has been split into train and test sets." if split_done else "The data has not been split yet.")
    errs = [x for x in (leakage or []) if x["level"] == "error"]
    warns = [x for x in (leakage or []) if x["level"] == "warning"]
    if split_done:
        out.append("Leakage guard: " + (f"{len(errs)} certain problem(s), {len(warns)} suspicion(s)." if errs or warns else "nothing found."))
    if validation_status:
        out.append({"passed": "The data passes its contract.", "warnings": "The data passes its contract with warnings.",
                    "failed": "The data breaks its contract."}.get(validation_status, ""))
    if ready and ready.get("score") is not None:
        bad = [c["name"] for c in ready["checks"] if not c["passed"]]
        out.append(f"Readiness checklist: {ready['score']} / 100." + (f" Still open: {', '.join(bad[:4])}." if bad else ""))
    return out


def pipeline_table(pipe) -> pd.DataFrame:
    if pipe is None or not getattr(pipe, "log_", None):
        return pd.DataFrame()
    rows = []
    for i, e in enumerate(pipe.log_, 1):
        rows.append({"#": i, "step": e["step"], "rows in": e["rows_in"], "rows out": e["rows_out"], "cols in": e["cols_in"],
                     "cols out": e["cols_out"], "what it did": "; ".join(e["notes"]) or "nothing"})
    return pd.DataFrame(rows)


def params_table(pipe) -> pd.DataFrame:
    if pipe is None:
        return pd.DataFrame()
    import json
    return pd.DataFrame([{"step": s["step"], "type": s["transformer"],
                          "settings": json.dumps(s["params"], default=str)[:300]} for s in pipe.get_params()])


def environment() -> dict:
    import scipy
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
