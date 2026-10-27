# DataForge architecture

DataForge is split into a **core** that knows nothing about Streamlit and a **user interface** made of Streamlit pages.
The core can be tested, scripted and reused without a browser.

```
pages/*.py        user interface (Streamlit): forms, tables, charts, buttons
core.py           shared UI helpers: navigation, file reading, workflow bar
dataforge/        the core (pandas, numpy, scipy; no Streamlit)
tests/            automated tests (pytest)
benchmarks/       speed, memory and correctness measurements
```

## The core, module by module

| Module | Job |
|---|---|
| `transformers.py` | Every preparation step is a class with `fit(train)` (learn from the training set) and `transform(df)` (apply what was learned). `transform` never modifies its input. Row-removing steps (duplicates, outliers set to "remove") act on the training set only, so validation, test and new data keep every row. |
| `pipeline.py` | `Pipeline` runs the steps in a fixed safe order, checks the schema of new data, and can save and restore its learned values as JSON (`state()` / `from_state()`). `build_pipeline(cfg)` turns UI settings into a pipeline. |
| `splitting.py` | Random and stratified split; class balancing for the training set only. |
| `leakage.py` | Leakage Guard: identical rows across sets (certain) and features that copy the target (suspicious). |
| `readiness.py` | Weighted, explainable checklist (not a prediction of model quality). |
| `types.py` | Variable type inference with confidence and reason; the user can override. |
| `advisor.py` | Finds problems and lists options with pros, cons and conditions. Never picks a winner. |
| `analysis.py` | Statistics: univariate, correlations (Pearson, Spearman, Cramér's V, eta, NMI), VIF, missing-data analysis, quality checks. Each measure documents its limits. |
| `target.py` | Target Intelligence for classification, regression, time series, unsupervised. |
| `diff.py` | Data Quality Diff Engine: two versions of a dataset, drift (PSI, KS). |
| `contract.py` | Data Contract: rules per column and validation of new files. |
| `persist.py` | One-file pipeline (`.json`), re-run on new data, generated Python script. |
| `reporting.py` | Content for the executive, analytical and technical report levels. |
| `lab.py` | Compare preparation choices by cross-validation on the training set only. |
| `scale.py` | Memory reduction, sampling, chunked profiling, optional Polars and DuckDB. |
| `security.py` | Upload checks, safe URL download, personal-data scan, safe project names. |
| `projects.py` | Save and reopen work in a local folder. |
| `benchmark.py` | Synthetic data, speed and memory, correctness against known answers. |

## The rule that shapes everything: learn from training data only

```
load → [Types] → split → fit on TRAIN → transform VALIDATION and TEST → check leakage → save pipeline
```

1. The split happens **before** any step that learns a value (median, mean, categories, scaling bounds, outlier bounds).
2. The UI keeps only the *settings* of each step. After every change the whole pipeline is rebuilt from the raw training set, so
   Undo is exact and the final result never depends on the order the user clicked.
3. `Remove rows` for outliers and class balancing touch the training set only.
4. The Lab and the Advisor never receive the test set.

## Data flow in the UI

`session_state` holds: `df` (what the pages show), `snapshots` (one per step), `raw_splits` (train / val / test before preparation),
`pipeline_cfg` (settings per step), `pipeline` (the fitted pipeline), `train_df` / `val_df` / `test_df` (prepared sets),
`var_types`, `contract`, `decisions`, `log`.

## Extending

This section describes how a step is added. A pull request that adds a step follows the same rules; see `CONTRIBUTING.md`.

A new step is a subclass of `Transformer` with `_fit` and `_transform`, registered in `pipeline.REGISTRY` and added to `build_pipeline`.
Add a test that checks: parameters come from the training data, the input is not modified, a saved and reloaded pipeline gives the same output.
