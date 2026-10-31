# Changelog

All notable changes to DataForge are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-31

A new major version. DataForge is now built around one rule: **learn from the training data only**. The workflow and the core changed, so this is a breaking release.

### Added

- **Split-first preparation.** The data is split before any step learns a value. Every step is fitted on the training set and then applied to validation and test. The pipeline is rebuilt from the raw training set after every change, so Undo is exact.
- **Variable types.** Inference with a confidence and a reason for each column (number, category, ordinal, ID, date, text). You confirm or override.
- **Advisor.** Finds problems and lists options with pros, cons and conditions, including a what-if view for missing values. It never chooses for you.
- **Insights and Target intelligence.** Statistics that state their own limits, missing-data analysis, data-quality checks, and analysis that follows the kind of target (classification, regression, time series, unsupervised).
- **Compare.** Two versions of a dataset side by side, with drift (PSI and KS test).
- **Lab.** Compares preparation choices with cross-validation on the training set only.
- **Leakage Guard.** Reports identical rows across sets and features that copy the target.
- **Data contract.** Infer, edit and validate column rules for new files.
- **Pipeline file and Python script.** One JSON file holds the settings and everything learned from the training data. Apply it to new data, re-fit it, or export it as a script.
- **Projects.** Save and reopen a whole piece of work in a local folder.
- **Reports at three levels.** Executive summary, Analytical and Technical, as HTML or Markdown.
- **Performance page.** Size tier, sampling, memory reduction, and chunked profiling for large files.
- **Security.** Upload checks (type against content, size limit, zip-bomb protection), safe download from a link, a personal-data scan, and safe project names. See `docs/SECURITY.md`.
- **Tests, benchmarks and documentation.** More than 100 automated tests, a benchmark suite, and the guides in `docs/`.
- **Project files.** `CHANGELOG.md`, `CONTRIBUTING.md`, issue and pull request templates, a GitHub Actions workflow that runs the tests, and `.streamlit/config.toml` (usage statistics off).

### Changed

- **License.** DataForge is now released under the MIT License instead of the Apache License 2.0. Anyone may use, change and share it. Pull requests are welcome; see `CONTRIBUTING.md`.
- **Workflow.** Import, Inspect, Analyze, Prepare, Validate, Export. The About page and the README describe it.
- **Streamlit.** The minimum version is now 1.50, because `use_container_width` is deprecated in favor of `width="stretch"`.
- **Sample files.** The names in `sample_data/` use underscores (`Ames_Housing.csv`, `Telco_Customer_Churn.csv`, `Palmer_Penguins.csv`). A card whose file is missing is disabled and shows what to add.
- **Hosted demo.** The address is now https://dataforge-tool.streamlit.app/.

### Fixed

- Duplicate rows were also removed from validation, test and new data, which changed their row count. They are now removed from the training set only.
- Deselecting every column in the Outliers or Scale options applied the step to all columns. An empty selection now means none.
- A per-column Median or Mean on a text column crashed. It now falls back to the most common value.
- With pandas 3, the note on numeric columns treated as categories also counted text columns.
- Date columns that arrived as text in new data raised an error in the encoder. They are converted first.
- Loading a new dataset, opening a project or loading the demo left the split, pipeline, variable types, decisions and contract of the previous dataset in the session. All of it is now cleared.
- Dates such as `2024-01-15` were reported as phone numbers by the personal-data scan.
- A leftover `.tmp` folder from an interrupted save appeared as a project.
- The benchmark failed on Windows because it used the Unix-only `resource` module.
- Operator precedence in the memory-reduction code, and several unused imports.

### Security

- Downloads from a link now reject the shared address range 100.64.0.0/10 and IPv6 scope identifiers are handled. See `docs/SECURITY.md`.

## [1.0.0]

First public release, under the Apache License 2.0. Loading, a health dashboard, a five-step preparation pipeline (Clean, Transform, Encode, Scale, Split), nine analysis views, undo and reset, a step log, and exports with an HTML or Markdown report.

Copies of version 1.0.0 that were already published stay under the Apache License 2.0. The MIT License applies from version 2.0.0.

[2.0.0]: https://github.com/MahdiKordian/DataForge/releases/tag/v2.0.0
[1.0.0]: https://github.com/MahdiKordian/DataForge/releases/tag/v1.0.0
