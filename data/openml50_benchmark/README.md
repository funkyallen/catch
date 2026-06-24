# OpenML-50 Benchmark

This optional benchmark is for the focused CATCH-vs-AutoGluon comparison:
50 OpenML data IDs, 10 seeds, and two methods.

The selection rule is deterministic and result-blind: sort the screened OpenML
metadata rows by OpenML data ID and keep the first 50 unique data IDs. The
manifest records the selected IDs, target columns, local filenames, source
URLs, content hashes, and cohort order.

CSV files in this folder are generated locally and ignored by Git. Build or
refresh them with:

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/run_openml50_autogluon_catch.ps1 -Python python -PlanOnly
```

For the first local build from an already downloaded source pool, pass
`-SourcePool <path-to-source-pool>`. Once `manifest.csv` exists, the builder can
download missing CSV files from OpenML through scikit-learn.
