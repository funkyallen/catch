# Data Notice

This package includes only the public tabular-regression datasets needed by the
slim CATCH reproduction scripts:

- 30 main benchmark CSV files listed in `reproducibility/datasets_manifest.csv`.
- 20 External Validation 20 OpenML CSV files listed in
  `data/external_validation20/manifest.csv`.

External Validation 20 is a bundled OpenML cohort kept separate from the main
benchmark. The public manifest records the OpenML data IDs, local file paths,
target columns, row/feature counts, cohort order, access URLs, and pandas
content hashes used by the reproduction scripts.

The main dataset manifest uses
`verify_upstream_before_archival_or_submission` as a conservative license-status
flag until source-specific redistribution terms are finalized.

Generated result tables, figures, logs, trained artifacts, raw source archives,
and non-bundled external-validation alternatives are not included.

Before formal archival release or journal submission, verify the redistribution
terms for every bundled dataset and update the project license.
