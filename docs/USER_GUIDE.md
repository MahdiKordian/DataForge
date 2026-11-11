# DataForge user guide

## The path through the app

**Import → Inspect → Analyze → Prepare → Validate → Export.** The bar at the top of the pages shows where you are.

| Stage | Pages | What you do |
|---|---|---|
| Import | Dataset | Load a file, link, pasted text or sample. Choose the target column on the Dashboard. |
| Inspect | Variable types, Advisor | Confirm what each column *is* (number, category, id, date...). Read the Advisor's findings and decide. |
| Analyze | Insights, Compare, Analysis | Statistics that explain themselves; compare two versions of the data. |
| Prepare | Prepare, Lab | Split first, then Clean, Transform, Encode, Scale. Preview before you apply. Use the Lab when you are not sure which method is better. |
| Validate | Contract | Write the rules a valid file must follow and check new files against them. |
| Export | Export, Pipeline, Report | Download the data, the saved pipeline, the Python script and the report. |

## Why Split comes first

Every value a step learns (a median, a category list, a scaling range) must come from the training set only.
If it is computed on all data, information from the test set leaks into training and the test score looks better than it really is.
DataForge splits first and then learns on the training set. Undo and re-running keep this true.

## Preview, Apply, Undo

Each step has **Preview** (what would change, nothing is saved), **Run** (apply) and the page-level **Undo last step** and **Reset**.
Large data is previewed on a sample of the training set.

## Advisor and Lab: how to decide

The Advisor examines the loaded dataset to identify potential issues and presents preparation options with their pros, cons and applicable conditions. For missing values, it can show how each option would affect the data. You remain in control of the preparation decisions.

When you want to evaluate the impact of preparation choices on model performance, the **Lab** compares alternatives using the same cross-validation procedure on the training set only. It reports the mean score and fold-to-fold spread for each option. Differences smaller than the fold-to-fold spread are reported as "no clear difference". Validation and test data are not used in this comparison.

## Saving and repeating your work

- **Pipeline page → Save:** one JSON file (settings plus everything learned from the training set) and an optional Python script.
- **Pipeline page → Run a saved pipeline:** upload the JSON and a new file. The file is checked against the pipeline's input contract first. *Apply* reuses the learned values; *Re-fit* learns again from the new data.
- **Projects:** save the whole piece of work (settings, decisions, optionally the data) and reopen it later; the saved steps are re-run and reproduce the same result.

## Reading the numbers

- **Readiness score:** a weighted checklist of preparation problems. It is not a prediction of model accuracy.
- **Association strength:** a statistical link, never proof of cause.
- **Drift (PSI):** below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 large. A drifting feature does not automatically mean a worse model.
- **Unreliable:** an association measured on fewer than 30 rows (or under 5% of the data) is marked unreliable.

## Big data

The **Performance** page shows the dataset size tier and the available measures for keeping analysis responsive, including sampling and memory reduction.

For selected large-file tasks, DataForge also provides:

- `profile_in_chunks` reads CSV files in chunks and calculates row counts, missing-value counts, and numeric statistics, including mean, standard deviation, minimum and maximum. Categorical summaries are capped and may be incomplete.
- `duckdb_summary` uses DuckDB to summarize local CSV or Parquet files without loading them into pandas.
- `polars_head` uses Polars to read a limited CSV preview, with a default limit of 100,000 rows.

These utilities support selected inspection tasks. The main data preparation workflow still loads the dataset into memory and does not provide fully out-of-core pipeline execution.
