# Data Inventory

This directory contains the datasets used by the slim CATCH reproduction
scripts.

## Main 30

The main benchmark datasets are the CSV files listed in
`../reproducibility/datasets_manifest.csv` with `Cohort=main_30`.

## External Validation 20

The External Validation 20 cohort is stored under
`external_validation20_20260624/`. The included files are exactly the 20
CSV datasets listed in `clean20_manifest.csv`, plus the frozen-cohort manifest
and dataset list.

The manifest records a fixed-seed randomized OpenML construction rule: exclude
main-benchmark source-family and prior-validation overlaps, exclude synthetic
benchmark-generator rows, apply the fetch helper's task metadata filters,
assign each eligible dataset a seeded pseudo-random rank key with
`selection_seed=20260624`, skip duplicate pandas content hashes and duplicate
name-family keys, and take the first 20 usable datasets in that queue. No model
scores are used to construct this cohort. Re-running the fetch helper also writes
`selection_table_metadata.csv`, which records the metadata-eligible queue,
download/duplicate status, and final selected order.

No generated results, raw source archives, unselected retrieval outputs, or
non-frozen extra datasets are included.
