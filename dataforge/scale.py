# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Working with bigger data: memory, sampling, chunked profiling and optional fast engines (Polars, DuckDB)."""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np
import pandas as pd

MB = 1024 * 1024


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / MB)


def tier(df: pd.DataFrame) -> str:
    m = memory_mb(df)
    return "small" if m < 100 else "medium" if m < 1000 else "large"


def plan(df: pd.DataFrame) -> dict:
    """What DataForge will do to stay responsive at this size."""
    t, m = tier(df), memory_mb(df)
    p = {"tier": t, "memory_mb": round(m, 1), "rows": len(df), "columns": df.shape[1]}
    if t == "small":
        p.update(sample_rows=None, note="Everything runs on the full data.")
    elif t == "medium":
        p.update(sample_rows=200_000, note="Analyses use a sample of up to 200,000 rows. Preparation runs on the full data. "
                 "Reducing memory (smaller number types, categories) is recommended.")
    else:
        p.update(sample_rows=100_000, note="Analyses use a sample of up to 100,000 rows. Consider Polars or DuckDB for the heavy lifting, "
                 "and work on a sample while designing the pipeline.")
    return p


def optimize_dtypes(df: pd.DataFrame, categories: bool = True, float32: bool = False, max_cat_ratio: float = 0.5):
    """Smaller numeric types and optional categories. Values are never changed, only how they are stored.
    float32 is opt-in because it keeps about 7 digits. Returns (frame, report)."""
    before = memory_mb(df)
    out, changes = df.copy(), []
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s) and not str(s.dtype).startswith(("Int", "UInt")):
            new = pd.to_numeric(s, downcast="unsigned" if s.min() >= 0 else "integer") if s.notna().all() else s
        elif pd.api.types.is_float_dtype(s) and float32 and s.notna().any():
            f32 = s.astype("float32")
            new = f32 if np.allclose(f32.astype("float64"), s, rtol=1e-6, atol=0, equal_nan=True) else s
        elif categories and (pd.api.types.is_object_dtype(s) or str(s.dtype) in ("str", "string")):
            n = s.nunique()
            new = s.astype("category") if categories and n <= 5000 and n / max(len(s), 1) <= max_cat_ratio else s
        else:
            new = s
        if str(new.dtype) != str(s.dtype):
            changes.append(c)
            out[c] = new
    after = memory_mb(out)
    return out, {"before_mb": round(before, 2), "after_mb": round(after, 2), "saved %": round((1 - after / before) * 100, 1) if before else 0.0,
                 "columns changed": len(changes)}


def smart_sample(df: pd.DataFrame, n: int, target=None, seed: int = 0) -> pd.DataFrame:
    """Random sample that keeps the class mix of a categorical target (small classes keep at least 50 rows)."""
    if len(df) <= n:
        return df
    if target is not None and target in df.columns and (not pd.api.types.is_numeric_dtype(df[target]) or df[target].nunique() <= 20):
        parts = []
        for _, g in df.groupby(target, observed=True):
            k = max(min(len(g), 50), int(round(n * len(g) / len(df))))
            parts.append(g.sample(min(k, len(g)), random_state=seed))
        out = pd.concat(parts)
        return out.sample(frac=1, random_state=seed) if len(out) <= n * 1.1 else out.sample(n, random_state=seed)
    return df.sample(n, random_state=seed)


def fingerprint(df: pd.DataFrame) -> str:
    """Cheap identity of a dataset (shape, columns, types and a few rows). Use it as a cache key instead of hashing every cell."""
    h = hashlib.sha1()
    h.update(repr((df.shape, list(map(str, df.columns)), [str(t) for t in df.dtypes])).encode())
    if len(df):
        idx = np.unique(np.linspace(0, len(df) - 1, min(len(df), 500)).astype(int))
        h.update(pd.util.hash_pandas_object(df.iloc[idx].astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------ chunked profiling
class ChunkedProfile:
    """Exact count, missing, mean, variance, min and max for numeric columns, and top values for text columns,
    computed chunk by chunk so the whole file is never in memory."""

    def __init__(self, top_cap: int = 20_000):
        self.n = {}; self.miss = {}; self.mean = {}; self.m2 = {}; self.min = {}; self.max = {}; self.cnt = {}; self.cap = top_cap
        self.rows = 0

    def update(self, chunk: pd.DataFrame) -> None:
        self.rows += len(chunk)
        for c in chunk.columns:
            s = chunk[c]
            self.miss[c] = self.miss.get(c, 0) + int(s.isna().sum())
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                v = s.dropna().astype(float)
                v = v[np.isfinite(v)]
                if v.empty:
                    continue
                n2, m2_, mean2 = len(v), float(((v - v.mean()) ** 2).sum()), float(v.mean())
                n1, mean1, M1 = self.n.get(c, 0), self.mean.get(c, 0.0), self.m2.get(c, 0.0)
                n = n1 + n2
                d = mean2 - mean1
                self.mean[c] = mean1 + d * n2 / n                       # Chan et al. parallel update
                self.m2[c] = M1 + m2_ + d * d * n1 * n2 / n
                self.n[c] = n
                self.min[c], self.max[c] = min(self.min.get(c, math.inf), float(v.min())), max(self.max.get(c, -math.inf), float(v.max()))
            else:
                cnt = self.cnt.setdefault(c, Counter())
                if len(cnt) < self.cap:
                    cnt.update(s.dropna().astype(str).value_counts().to_dict())

    def result(self) -> pd.DataFrame:
        rows = []
        for c in list(self.miss):
            r = {"column": c, "rows": self.rows, "missing": self.miss[c], "missing %": self.miss[c] / max(self.rows, 1) * 100}
            if c in self.n:
                r.update({"mean": self.mean[c], "std": math.sqrt(self.m2[c] / (self.n[c] - 1)) if self.n[c] > 1 else float("nan"),
                          "min": self.min[c], "max": self.max[c]})
            elif c in self.cnt and self.cnt[c]:
                top, k = self.cnt[c].most_common(1)[0]
                r.update({"unique (at least)": len(self.cnt[c]), "top value": top, "top share %": k / max(self.rows - self.miss[c], 1) * 100})
            rows.append(r)
        return pd.DataFrame(rows)


def profile_in_chunks(source, chunksize: int = 200_000, **read_kw) -> pd.DataFrame:
    """source: a path or file-like object with CSV text."""
    prof = ChunkedProfile()
    for chunk in pd.read_csv(source, chunksize=chunksize, **read_kw):
        prof.update(chunk)
    return prof.result()


# -------------------------------------------------------------------------------- optional engines
def engines() -> dict:
    out = {}
    for name in ("polars", "duckdb"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            out[name] = None
    return out


def duckdb_summary(path: str) -> pd.DataFrame:
    """SUMMARIZE a CSV or Parquet file with DuckDB, without loading it into pandas. Local trusted paths only."""
    import duckdb
    con = duckdb.connect()
    try:
        rel = con.read_parquet(path) if str(path).lower().endswith(".parquet") else con.read_csv(path)
        return con.sql("SUMMARIZE rel").df()
    finally:
        con.close()


def polars_head(path: str, n: int = 100_000) -> pd.DataFrame:
    """First n rows of a big CSV through Polars' lazy reader (fast, low memory)."""
    import polars as pl
    return pl.scan_csv(path).head(n).collect().to_pandas()
