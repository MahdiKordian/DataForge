# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""DataForge core: data-preparation logic with no Streamlit dependency."""
from .pipeline import Pipeline, SchemaError, build_pipeline
from .splitting import balance_train, split_frame
from .readiness import readiness
from .leakage import leakage_report
from .contract import infer_contract, validate as validate_contract
from .diff import compare
from .advisor import diagnose, what_if_missing, merge
from .persist import make_spec, dumps, loads, apply_saved, refit, to_script, SpecError

__all__ = ["Pipeline", "SchemaError", "build_pipeline", "split_frame", "balance_train", "readiness", "leakage_report",
           "infer_contract", "validate_contract", "compare", "diagnose", "what_if_missing", "merge",
           "make_spec", "dumps", "loads", "apply_saved", "refit", "to_script", "SpecError"]
