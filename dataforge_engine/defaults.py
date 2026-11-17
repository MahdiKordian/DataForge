# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""One complete, sensible set of preparation settings. Used by the Lab and the benchmarks."""
from __future__ import annotations


def default_cfg(target_is_class: bool = True) -> dict:
    return {
        "Clean": dict(trim=True, case="Keep as is", to_num=True, dups=True, const=True, max_missing=60,
                      drop_missing_target=True, num_missing="Median", cat_missing="Most common"),
        "Transform": dict(outliers="Leave as is", out_cols=None, skew=False, skew_thr=1.0, skew_method="Log (log1p)", drop=[], corr=None),
        "Encode": dict(low="One-hot", max_onehot=10, drop_first=True, high="Frequency", dates=True, encode_target=target_is_class),
        "Scale": dict(method="Standard (z-score)", cols=None),
    }
