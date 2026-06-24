# Data Notice

This package includes only the public tabular-regression datasets needed by the
slim CATCH reproduction scripts:

- 30 main benchmark CSV files listed in `reproducibility/datasets_manifest.csv`.
- 20 External Validation 20 OpenML CSV files listed in
  `data/external_validation20_20260624/clean20_manifest.csv`.

External Validation 20 is a frozen fixed-seed randomized OpenML cohort. The
public manifest records the random-selection seed, selection order, selection
rule, source-family/name-family keys, pandas content hashes, and OpenML data
IDs used by the reproduction scripts. The companion selection table records
unusable downloads and duplicate rows without shipping generated result files.

The main dataset manifest uses
`verify_upstream_before_archival_or_submission` as a conservative license-status
flag until source-specific redistribution terms are finalized.

Generated result tables, figures, logs, trained artifacts, unselected retrieval
outputs, raw source archives, and non-frozen external-validation alternatives
are not included.

Before formal archival release or journal submission, verify the redistribution
terms for every bundled dataset and update the project license.
