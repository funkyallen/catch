# Data Inventory

This directory contains the datasets used by the slim CATCH reproduction
scripts.

## Main 30

The main benchmark datasets are the CSV files listed in
`../reproducibility/datasets_manifest.csv` with `Cohort=main_30`.

## Additional OpenML Cohort

The additional OpenML cohort is stored under `external_validation20/`; the
legacy folder name is retained for reproducibility. The included files are
exactly the 20 CSV tables listed in `manifest.csv`, plus the cohort dataset
list.

The manifest records the OpenML data ID, local path, target column used by the
runner, task-type note, row and feature counts, cohort order, access URL, and
pandas content hash for each dataset. It declares 19 ordinary numeric-target
regression tasks for the primary paper aggregate and one derived credit-amount
regression stress task from OpenML `credit_g`. The audit helper checks that the
bundled files still match those manifest fields before any additional-cohort
rerun.

No generated results, raw source archives, or non-bundled extra datasets are
included.

## OpenML-50 Benchmark

`openml50_benchmark/manifest.csv` defines the optional 50-dataset
CATCH-vs-AutoGluon expansion. The raw CSV files in that folder are generated
locally by `scripts/build_openml50_benchmark.py` and ignored by Git. Use
`scripts/audit_openml50_benchmark.py` before running the 50-dataset experiment.
