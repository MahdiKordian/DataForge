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

The Advisor lists options with pros, cons and when to use them; it does not choose for you. For missing values it can show each option's effect on the data.
When the effect on a model matters, the **Lab** runs several choices through the same cross-validation on the training set and reports the mean and spread per option.
A difference smaller than the fold-to-fold spread is reported as "no clear difference". The test set is not used.

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

The **Performance** page shows the size tier and what DataForge does about it (sampling for analyses, memory reduction). For files that do not fit in memory use
`dataforge.scale.profile_in_chunks` (exact summary, chunk by chunk), `duckdb_summary` or `polars_head` (optional packages).
