# Data Inventory

This directory contains the datasets used by the slim CATCH reproduction
scripts.

## Main 30

The main benchmark datasets are the CSV files listed in
`../reproducibility/datasets_manifest.csv` with `Cohort=main_30`.

## External Validation 20

The External Validation 20 cohort is stored under
`external_validation20/`. The included files are exactly the 20 CSV datasets
listed in `manifest.csv`, plus the cohort dataset list.

The manifest records the OpenML data ID, local path, target column used by the
runner, row and feature counts, cohort order, access URL, and pandas content
hash for each dataset. The audit helper checks that the bundled files still
match those manifest fields before any external-validation rerun.

No generated results, raw source archives, or non-bundled extra datasets are
included.
