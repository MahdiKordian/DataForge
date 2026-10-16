# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import transformers as _tr
from .transformers import (ColumnDropper, ColumnFilter, CategoricalEncoder, CorrelationFilter,
                           FeatureScaler, InfToNan, MissingValueImputer, NotFittedError,
                           OutlierHandler, SkewTransformer, TextCleaner, TypeCaster)


_SKIP = {"fitted_", "notes_", "warnings_", "params", "target_"}
REGISTRY = {n: getattr(_tr, n) for n in ("InfToNan", "TextCleaner", "ColumnFilter", "MissingValueImputer", "OutlierHandler",
                                         "SkewTransformer", "CorrelationFilter", "ColumnDropper", "CategoricalEncoder",
                                         "FeatureScaler", "TypeCaster")}


def _enc(x):
    """Learned parameters -> plain JSON. Dict keys keep their type (stored as pairs)."""
    if isinstance(x, dict):
        return {"__d__": [[_enc(k), _enc(v)] for k, v in x.items()]}
    if isinstance(x, tuple):
        return {"__t__": [_enc(v) for v in x]}
    if isinstance(x, (list, set)):
        return [_enc(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _dec(x):
    if isinstance(x, dict):
        if "__d__" in x:
            return {_dec(k): _dec(v) for k, v in x["__d__"]}
        if "__t__" in x:
            return tuple(_dec(v) for v in x["__t__"])
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


class SchemaError(ValueError):
    """New data does not match the data the pipeline was fitted on."""


class Pipeline:
    def __init__(self, steps, target=None):
        self.steps = steps            # list of (label, Transformer)
        self.target = target
        self.fitted_ = False
        self.log_: list[dict] = []
        self.warnings_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn everything from df (the TRAINING set) and return the transformed df."""
        self.input_columns_ = list(df.columns)
        self.log_ = []
        for label, step in self.steps:
            rows, cols = len(df), df.shape[1]
            step.fit(df, self.target)
            df = step.transform(df, training=True).reset_index(drop=True)
            self.log_.append({"step": label, "rows_in": rows, "rows_out": len(df), "cols_in": cols,
                              "cols_out": df.shape[1], "notes": list(step.notes_)})
        self.output_columns_ = list(df.columns)
        self.fitted_ = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned parameters to validation, test or new data."""
        if not self.fitted_:
            raise NotFittedError("Pipeline must be fitted on the training data first.")
        self.check_schema(df)
        self.warnings_ = []
        for _, step in self.steps:
            df = step.transform(df).reset_index(drop=True)
            self.warnings_ += step.warnings_
        out = df.reindex(columns=self.output_columns_)
        if self.target is not None and self.target not in df.columns:
            out = out.drop(columns=[self.target], errors="ignore")   # new data may have no target column
        return out

    def check_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.input_columns_ if c not in df.columns and c != self.target]
        if missing:
            raise SchemaError("These columns were in the training data but are missing now: "
                              + ", ".join(map(str, missing[:10])) + ". Rename them or re-run the pipeline.")

    def state(self) -> dict:
        """Everything needed to apply this fitted pipeline later, as JSON-safe data (no pickle)."""
        if not self.fitted_:
            raise NotFittedError("Pipeline must be fitted before its state can be saved.")
        steps = []
        for label, s in self.steps:
            learned = {k: _enc(v) for k, v in vars(s).items() if k.endswith("_") and k not in _SKIP}
            steps.append({"label": label, "class": type(s).__name__, "params": _enc(s.params), "learned": learned})
        return {"target": self.target, "input_columns": _enc(self.input_columns_),
                "output_columns": _enc(self.output_columns_), "steps": steps}

    @classmethod
    def from_state(cls, st: dict) -> "Pipeline":
        steps = []
        for e in st["steps"]:
            if e["class"] not in REGISTRY:
                raise ValueError(f"Unknown step type in the saved pipeline: {e['class']}")
            s = REGISTRY[e["class"]](**_dec(e["params"]))
            for k, v in e["learned"].items():
                if not re.fullmatch(r"[a-z][a-z0-9_]*_", k) or k in _SKIP or k.startswith("__"):
                    raise ValueError(f"Unexpected field in the saved pipeline: {k!r}")
                setattr(s, k, _dec(v))
            s.target_, s.fitted_ = st["target"], True
            steps.append((e["label"], s))
        p = cls(steps, st["target"])
        p.input_columns_, p.output_columns_ = _dec(st["input_columns"]), _dec(st["output_columns"])
        p.fitted_ = True
        return p

    def get_params(self) -> list[dict]:
        return [{"step": label, "transformer": type(s).__name__, "params": s.params} for label, s in self.steps]


def build_pipeline(cfg: dict, target=None) -> Pipeline:
    """cfg maps a UI step name (Clean, Transform, Encode, Scale) to its parameter dict.
    Steps always run in the same safe order, whatever order the user clicked them in."""
    steps = [("Infinite values", InfToNan())]
    protect = []
    if cfg.get("Types", {}).get("types"):
        steps.append(("Variable types", TypeCaster(**cfg["Types"])))
        protect = [c for c, v in cfg["Types"]["types"].items() if v in ("categorical", "ordinal", "id", "text")]
    if "Clean" in cfg:
        p = {**cfg["Clean"], "protect": protect}
        steps += [("Text cleanup", TextCleaner(**p)), ("Column filter", ColumnFilter(**p)),
                  ("Missing values", MissingValueImputer(**p))]
    if "Transform" in cfg:
        p = cfg["Transform"]
        steps.append(("Outliers", OutlierHandler(**p)))
        if p.get("skew"):
            steps.append(("Skew", SkewTransformer(**p)))
        if p.get("corr"):
            steps.append(("Near-duplicate columns", CorrelationFilter(**p)))
        if p.get("drop"):
            steps.append(("Dropped columns", ColumnDropper(**p)))
    if "Encode" in cfg:
        steps.append(("Encoding", CategoricalEncoder(**cfg["Encode"])))
    if "Scale" in cfg and not cfg["Scale"].get("method", "").startswith("Do not"):
        steps.append(("Scaling", FeatureScaler(**cfg["Scale"])))
    return Pipeline(steps, target)
