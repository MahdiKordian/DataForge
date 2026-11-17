# dataforge-engine

Leakage-aware data preparation for machine learning. This is the core of [DataForge](https://github.com/MahdiKordian/DataForge): pandas, NumPy, SciPy and scikit-learn only, no Streamlit.

The data is split **before** any preprocessing is fitted. Imputation statistics, category mappings, outlier thresholds and scaling parameters are learned from the training set and reused for validation and test data.

## Install

```bash
pip install dataforge-engine
```

Python 3.10 or newer. Optional, for very large files: `pip install "dataforge-engine[fast]"` (adds polars and duckdb).

## Quick start

```python
import pandas as pd
from dataforge_engine import build_pipeline, split_frame
from dataforge_engine.defaults import default_cfg

df = pd.read_csv("customers.csv")
target = "churn"

# 1. Split before fitting anything.
train_raw, val_raw, test_raw, notes = split_frame(df, target, 20, 10, 42, True)

# 2. Fit on the training set, then transform the other sets.
pipe = build_pipeline(default_cfg(target_is_class=True), target)
train_ready = pipe.fit_transform(train_raw)
test_ready = pipe.transform(test_raw)
```

## What is inside

- **Pipeline** with a fit / transform contract: nothing is learned from validation or test data.
- **Leakage report**: identical rows across sets and features that copy the target.
- **Data contract**: infer the expected schema and validate new files against it.
- **Advisor**: finds problems in a dataset and explains the options.
- **Saved pipelines**: export a fitted pipeline as JSON and as a Python script, then apply it to new data.

Full examples (saving and reloading a pipeline, generating `prepare.py`) are in the [main README](https://github.com/MahdiKordian/DataForge#use-it-from-python).

## Links

- Source and issues: https://github.com/MahdiKordian/DataForge
- Live demo of the app: https://dataforge-tool.streamlit.app/
- Changelog: https://github.com/MahdiKordian/DataForge/blob/main/CHANGELOG.md

## License

MIT, copyright 2026 Mahdi Kordian.