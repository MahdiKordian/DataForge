# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Benchmarks: speed and memory at growing sizes, and correctness against known answers."""
from __future__ import annotations

import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

from . import analysis as A
from .defaults import default_cfg
from .diff import compare
from .pipeline import build_pipeline
from .splitting import split_frame


def make_dataset(rows: int, cols: int = 20, missing: float = 0.05, outliers: float = 0.01, imbalance: float = 0.15,
                 cat_share: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Synthetic table with a known structure: numeric + categorical columns, gaps, outliers and an imbalanced binary target."""
    r = np.random.default_rng(seed)
    n_cat = max(1, int(cols * cat_share))
    n_num = max(1, cols - n_cat)
    X = r.normal(size=(rows, n_num)) * r.uniform(1, 50, n_num) + r.uniform(-100, 100, n_num)
    out = r.random((rows, n_num)) < outliers
    X[out] *= 8
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_num)])
    for i in range(n_cat):
        k = int(r.choice([3, 5, 12, 40]))
        df[f"cat_{i}"] = r.choice([f"v{j}" for j in range(k)], rows)
    signal = df["num_0"] / df["num_0"].std() + (df["cat_0"] == "v0") * 1.5
    p = 1 / (1 + np.exp(-(signal - np.quantile(signal, 1 - imbalance) + 1)))
    df["y"] = np.where(r.random(rows) < p, "yes", "no")
    mask = r.random(df.shape) < missing
    mask[:, df.columns.get_loc("y")] = False
    return df.mask(mask)


def _time(fn, repeat: int = 1) -> float:
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _peak_mb(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1048576
    finally:
        tracemalloc.stop()


def tasks(df: pd.DataFrame) -> dict:
    num, cat, _ = A.split_cols(df, "y")
    tr, _, te, _ = split_frame(df, "y", 20, 0, 1)
    cfg = default_cfg(True)

    def pipeline():
        p = build_pipeline(cfg, "y")
        p.fit_transform(tr)
        p.transform(te)
    return {
        "split": lambda: split_frame(df, "y", 20, 10, 1),
        "profile numeric": lambda: A.univariate_numeric(df, num),
        "profile categorical": lambda: A.univariate_categorical(df, cat),
        "quality checks": lambda: A.quality_checks(df, "y"),
        "correlations": lambda: A.correlations(df, num),
        "mixed associations": lambda: A.mixed_associations(df, num, cat),
        "missing analysis": lambda: A.missing_analysis(df),
        "pipeline fit+transform": pipeline,
        "compare two versions": lambda: compare(df.iloc[: len(df) // 2], df.iloc[len(df) // 2:]),
    }


def run_speed(sizes=((5_000, 20), (20_000, 20), (100_000, 20), (20_000, 100)), measure_memory: bool = True, only=None, progress=None) -> pd.DataFrame:
    rows = []
    for i, (n, c) in enumerate(sizes):
        df = make_dataset(n, c)
        for name, fn in tasks(df).items():
            if only and name not in only:
                continue
            rows.append({"rows": n, "columns": c, "task": name, "seconds": _time(fn),
                         "peak MB": _peak_mb(fn) if measure_memory else np.nan})
        if progress:
            progress((i + 1) / len(sizes))
    return pd.DataFrame(rows)


def scaling_exponents(speed: pd.DataFrame) -> pd.DataFrame:
    """Log-log slope of time against rows at 20 columns. About 1 means linear; much more than 1 means trouble at scale."""
    d = speed[speed["columns"] == speed["columns"].min()]
    out = []
    for task, g in d.groupby("task"):
        g = g[g["seconds"] > 0].sort_values("rows")
        if len(g) >= 2:
            slope = float(np.polyfit(np.log(g["rows"]), np.log(g["seconds"]), 1)[0])
            out.append({"task": task, "time ~ rows^": round(slope, 2), "seconds at largest": float(g["seconds"].iloc[-1])})
    return pd.DataFrame(out).sort_values("time ~ rows^", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------ correctness
def correctness() -> pd.DataFrame:
    """Checks against answers known in advance. Each row: what was checked, expected, measured, pass."""
    out = []

    def add(check, expected, got, ok):
        out.append({"check": check, "expected": expected, "measured": got, "pass": bool(ok)})
    r = np.random.default_rng(0)
    x = pd.Series(r.normal(5, 2, 200_000))
    u = A.univariate_numeric(pd.DataFrame({"x": x}), ["x"]).iloc[0]
    add("mean of N(5,2)", "5.00 +/- 0.02", f"{u['mean']:.3f}", abs(u["mean"] - 5) < 0.02)
    add("std of N(5,2)", "2.00 +/- 0.02", f"{u['std']:.3f}", abs(u["std"] - 2) < 0.02)
    add("skew of a normal", "0 +/- 0.03", f"{u['skew']:.3f}", abs(u["skew"]) < 0.03)
    a = pd.Series(r.choice(list("abc"), 5000)); b = a.map({"a": "x", "b": "y", "c": "z"})
    add("Cramer's V, identical categories", "close to 1", f"{A.cramers_v(a, b):.3f}", A.cramers_v(a, b) > 0.98)
    ind = pd.Series(r.choice(list("xy"), 5000))
    add("Cramer's V, independent categories", "close to 0", f"{A.cramers_v(a, ind):.3f}", A.cramers_v(a, ind) < 0.05)
    t = pd.Series(r.uniform(-3, 3, 5000)); sq = t ** 2 + r.normal(0, .05, 5000)
    add("Pearson blind to a parabola", "|r| < 0.1", f"{abs(t.corr(sq)):.3f}", abs(t.corr(sq)) < 0.1)
    add("NMI sees the parabola", "NMI > 0.3", f"{A.nmi(t, sq):.3f}", A.nmi(t, sq) > 0.3)
    df = make_dataset(5000, 12, seed=1)
    tr, va, te, _ = split_frame(df, "y", 20, 10, 7)
    ids = [set(d.index) for d in (tr, va, te)]
    add("split sizes add up", "5000", str(len(tr) + len(va) + len(te)), len(tr) + len(va) + len(te) == len(df.dropna(subset=["y"])))
    add("class share kept by the split (stratified)", "difference < 1 pt", f"{abs((tr.y == 'yes').mean() - (te.y == 'yes').mean()) * 100:.2f} pt",
        abs((tr.y == "yes").mean() - (te.y == "yes").mean()) < 0.01)
    cfg = default_cfg(True)
    p = build_pipeline(cfg, "y"); trt = p.fit_transform(tr)
    med = tr["num_1"].median()
    imputer = dict(p.steps)["Missing values"]
    add("imputer uses the TRAINING median", f"{med:.4f}", f"{imputer.fill_['num_1']:.4f}", abs(imputer.fill_["num_1"] - med) < 1e-9)
    before = te.copy(); _ = p.transform(te)
    add("transform leaves its input unchanged", "identical", "identical" if te.equals(before) else "CHANGED", te.equals(before))
    cols_ok = list(p.transform(va).columns) == list(trt.columns) == list(p.transform(te).columns)
    add("train, validation and test have the same columns", "True", str(cols_ok), cols_ok)
    p2 = build_pipeline(cfg, "y")
    same = p2.fit_transform(tr).equals(trt)
    add("same data and seed give identical output", "True", str(same), same)
    mean_after = trt["num_2"].mean()
    add("scaled training column has mean about 0", "|mean| < 0.05", f"{mean_after:.4f}", abs(mean_after) < 0.05)
    add("no NaN left after preparation", "0", str(int(trt.isna().sum().sum())), int(trt.isna().sum().sum()) == 0)
    return pd.DataFrame(out)


def environment() -> dict:
    import os
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform(),
            "cpu count": os.cpu_count()}


# ------------------------------------------------------------------------- comparison with other tools
_TOOL_CODE = r"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
try:
    import resource
    _div = 1024 * 1024 if sys.platform == "darwin" else 1024      # ru_maxrss: bytes on macOS, KB on Linux
    def _rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _div
except ImportError:                                               # Windows
    def _rss(): return float("nan")
import pandas as pd
df = pd.read_parquet(sys.argv[2]); tool = sys.argv[1]
if tool == "dataforge":
    from dataforge import analysis as A
    from dataforge.benchmark import tasks
    fns = tasks(df); keep = ["profile numeric", "profile categorical", "quality checks", "correlations", "mixed associations", "missing analysis"]
    def run(): [fns[k]() for k in keep]
elif tool == "ydata_minimal":
    from ydata_profiling import ProfileReport
    def run(): ProfileReport(df, minimal=True, progress_bar=False).to_html()
elif tool == "ydata_default":
    from ydata_profiling import ProfileReport
    def run(): ProfileReport(df, progress_bar=False).to_html()
elif tool == "sweetviz":
    import sweetviz as sv
    def run(): sv.analyze(df, target_feat="y").show_html(filepath="/tmp/_sv.html", open_browser=False, layout="vertical", scale=None)
base = _rss()
t = time.perf_counter(); run(); s = time.perf_counter() - t
peak = _rss() - base
print(json.dumps({"seconds": s, "peak_mb_above_start": None if peak != peak else peak}))
"""

TOOLS = {"dataforge": "DataForge (profile, quality, relationships, missing data)", "ydata_minimal": "ydata-profiling (minimal=True)",
         "ydata_default": "ydata-profiling (default)", "sweetviz": "Sweetviz (with target)"}


def compare_tools(sizes=((5_000, 20), (20_000, 20)), tools=tuple(TOOLS), timeout: int = 900) -> pd.DataFrame:
    """Each tool runs in a fresh process on the same data, so imports and memory are not shared.
    The tools produce different outputs; this measures 'time until a complete analysis exists', not equal work."""
    import json
    import subprocess
    import sys
    import tempfile
    rows = []
    for n, c in sizes:
        with tempfile.TemporaryDirectory() as d:
            f = f"{d}/data.parquet"
            make_dataset(n, c).to_parquet(f)
            for tool in tools:
                r = {"rows": n, "columns": c, "tool": TOOLS[tool]}
                try:
                    p = subprocess.run([sys.executable, "-c", _TOOL_CODE, tool, f], capture_output=True, text=True, timeout=timeout)
                    if p.returncode == 0:
                        r.update(json.loads(p.stdout.strip().splitlines()[-1]))
                    else:
                        r["note"] = "failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:140]
                except subprocess.TimeoutExpired:
                    r["note"] = f"timed out after {timeout}s"
                rows.append(r)
    return pd.DataFrame(rows)
