# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Preparation Lab: compare several preparation choices with one model and one cross-validation scheme.

The function only receives the TRAINING set. The test set is never an argument, so it cannot be used to pick a method.
Inside every fold the pipeline is fitted on the fold's training part only."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .defaults import default_cfg
from .pipeline import build_pipeline
from .splitting import is_classification

MISSING_CHOICES = ["Median", "Mean", "Fill with 0", "Drop rows"]
OUTLIER_CHOICES = ["Leave as is", "Clip to IQR bounds", "Clip at 1st and 99th percentile"]
SCALE_CHOICES = ["Standard (z-score)", "Min-max (0 to 1)", "Robust (median / IQR)", "Do not scale"]
DIMENSIONS = {"Missing numbers": ("Clean", "num_missing", MISSING_CHOICES),
              "Outliers": ("Transform", "outliers", OUTLIER_CHOICES),
              "Scaling": ("Scale", "method", SCALE_CHOICES)}


def sklearn_available() -> bool:
    try:
        import sklearn
        return True
    except ImportError:
        return False


def models(kind: str) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    if kind == "classification":
        return {"Logistic regression": lambda: LogisticRegression(max_iter=1000),
                "Random forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
                "Gradient boosting (accepts missing values)": lambda: HistGradientBoostingClassifier(random_state=0)}
    return {"Ridge regression": lambda: Ridge(), "Random forest": lambda: RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
            "Gradient boosting (accepts missing values)": lambda: HistGradientBoostingRegressor(random_state=0)}


def variants_for(dimension: str, values: list, base: dict | None = None, types: dict | None = None, is_class: bool = True) -> dict:
    """One cfg per value of the chosen dimension, everything else equal."""
    step, key, _ = DIMENSIONS[dimension]
    out = {}
    for v in values:
        cfg = {k: dict(p) for k, p in (base or default_cfg(is_class)).items()}
        cfg[step][key] = v
        if types:
            cfg["Types"] = types
        out[str(v)] = cfg
    return out


def _score(kind, model, Xv, yv):
    from sklearn import metrics as M
    pred = model.predict(Xv)
    if kind == "classification":
        r = {"accuracy": M.accuracy_score(yv, pred), "balanced accuracy": M.balanced_accuracy_score(yv, pred),
             "macro F1": M.f1_score(yv, pred, average="macro")}
        if yv.nunique() == 2 and hasattr(model, "predict_proba"):
            r["ROC AUC"] = M.roc_auc_score(yv, model.predict_proba(Xv)[:, 1])
        return r
    return {"R2": M.r2_score(yv, pred), "RMSE": float(np.sqrt(M.mean_squared_error(yv, pred))), "MAE": M.mean_absolute_error(yv, pred)}


def run_lab(train_raw: pd.DataFrame, target: str, variants: dict, model_name: str | None = None, folds: int = 5,
            seed: int = 42, metric: str | None = None, max_rows: int = 50_000, progress=None) -> dict:
    """Returns {"table", "folds", "metric", "kind", "model", "notes"}."""
    from sklearn.model_selection import KFold, StratifiedKFold
    if not sklearn_available():
        raise RuntimeError("scikit-learn is needed for the Lab: pip install scikit-learn")
    df = train_raw[train_raw[target].notna()]
    notes = []
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
        notes.append(f"Used a random sample of {max_rows:,} training rows to keep the Lab fast.")
    kind = "classification" if is_classification(df[target]) else "regression"
    M = models(kind)
    model_name = model_name if model_name in M else next(iter(M))
    default_metric = "macro F1" if kind == "classification" else "R2"
    metric = metric or default_metric
    ytxt = df[target].astype(str) if kind == "classification" else None
    splitter = (StratifiedKFold(folds, shuffle=True, random_state=seed) if kind == "classification" else KFold(folds, shuffle=True, random_state=seed))
    splits = list(splitter.split(df, ytxt)) if kind == "classification" else list(splitter.split(df))
    rows = []
    total = len(variants) * len(splits)
    done = 0
    for name, cfg in variants.items():
        for fi, (tr_i, va_i) in enumerate(splits):
            rec = {"variant": name, "fold": fi}
            t0 = time.perf_counter()
            try:
                pipe = build_pipeline(cfg, target)
                tr = pipe.fit_transform(df.iloc[tr_i])
                imputer = next((step for label, step in pipe.steps if label == "Missing values"),None,)
                va_raw = df.iloc[va_i].copy()
                drop_cols = [c for c in getattr(imputer, "drop_rows_cols_", [])if c in va_raw.columns]
                if drop_cols:
                    va_raw = va_raw.dropna(subset=drop_cols)
                va = pipe.transform(va_raw)

                if va.empty or tr.empty:
                    raise ValueError("every row was removed")
                bad = [c for c in tr.columns if c != target and not pd.api.types.is_numeric_dtype(tr[c])]
                if bad:
                    raise ValueError(f"column(s) still not numeric: {', '.join(map(str, bad[:3]))}")
                Xt, yt, Xv, yv = tr.drop(columns=[target]), tr[target], va.drop(columns=[target]), va[target]
                if yt.isna().any() or yv.isna().any():
                    raise ValueError("the target has unseen classes in this fold")
                mdl = M[model_name]().fit(Xt, yt)
                rec.update(_score(kind, mdl, Xv, yv))
                rec.update({"rows kept %": len(va) / len(va_i) * 100, "features": Xt.shape[1]})
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:160]
            rec["seconds"] = time.perf_counter() - t0
            rows.append(rec)
            done += 1
            if progress:
                progress(done / total)
    F = pd.DataFrame(rows)
    table = []
    base_scores = None
    for name in variants:
        f = F[F["variant"] == name]
        ok = f[metric].dropna() if metric in f else pd.Series(dtype=float)
        r = {"variant": name, f"{metric} (mean)": ok.mean() if len(ok) else np.nan, f"{metric} (std)": ok.std() if len(ok) > 1 else np.nan,
             "rows kept %": f["rows kept %"].mean() if "rows kept %" in f else np.nan, "features": f["features"].mean() if "features" in f else np.nan,
             "seconds / fold": f["seconds"].mean(), "failed folds": int(f["error"].notna().sum()) if "error" in f else 0}
        if len(ok) == folds:
            if base_scores is None:
                base_scores, r["difference vs first"], r["verdict"] = ok.to_numpy(), 0.0, "baseline"
            else:
                d = ok.to_numpy() - base_scores
                r["difference vs first"] = float(d.mean())
                sd = d.std(ddof=1)
                r["verdict"] = ("no clear difference" if abs(d.mean()) <= sd else "better than the first" if d.mean() > 0 else "worse than the first")
                if metric in ("RMSE", "MAE"):
                    r["verdict"] = {"better than the first": "worse than the first", "worse than the first": "better than the first"}.get(r["verdict"], r["verdict"])
        table.append(r)
    if F.get("error") is not None and F["error"].notna().any():
        notes.append("Some folds failed: " + "; ".join(sorted(set(F["error"].dropna()))[:3]))
    if kind == "classification" and metric == "accuracy":
        notes.append("Accuracy can look good on imbalanced data; macro F1 or balanced accuracy are safer.")
    notes.append("Folds are few and overlap in training data, so differences smaller than the fold-to-fold spread should be read as 'no clear difference'. "
                 "The test set was not used.")
    return {"table": pd.DataFrame(table), "folds": F, "metric": metric, "kind": kind, "model": model_name, "notes": notes}
