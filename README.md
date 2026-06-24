# catch

Code and public data for reproducing the CATCH experiments:

**CATCH: A Reliability-Calibrated Neural-Tree Complementary Learning Framework
for Label-Scarce Tabular Regression with Auxiliary Unlabeled Covariates**

Repository: https://github.com/funkyallen/catch

This is a code-and-data-only GitHub package. It does not include generated
result tables, figures, logs, trained model artifacts, or manuscript build
outputs.

## Layout

```text
core/              CATCH implementation, supervised and semi-supervised baselines
experiments/catch/ Main runner and table-analysis script
scripts/           OpenML External Validation 20 fetch and audit helpers
tests/             Lightweight CATCH dispatch and analysis tests
reproducibility/   Environment files, run scripts, seeds, manifests
data/              Main 30 datasets and External Validation 20 datasets
```

## Environment

```powershell
conda env create -f reproducibility/environment.yml
conda activate catch-ieee-access
```

or:

```powershell
python -m pip install -r reproducibility/requirements.txt
```

## Reproduction

Quick smoke check, without running the full experiment grid:

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/run_smoke.ps1 -Python python
```

Use `-RequireTrainingDeps` after installing the full requirements to make the
tiny CatBoost+CATCH training smoke mandatory.

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/run_main.ps1 -Python python
powershell -ExecutionPolicy Bypass -File reproducibility/run_ablation.ps1 -Python python
powershell -ExecutionPolicy Bypass -File reproducibility/run_external_validation.ps1 -Python python
```

Outputs are written under `r/` and `paper/tables/`, both ignored by Git.

The main publication runner covers the paper-facing non-foundation comparison
set: CATCH, AutoGluon, TabM, CatBoost, XGBoost, LightGBM, LapBoost, VIME,
COREG, RankUp, and UCVME. TabPFN-v3 is an optional external reference and is
not bundled with this code-and-data package.

Implementation note: CATCH uses residual-calibrated diagnostic scales for
support weighting. The CatBoost staged-response shape is an optimization-path
diagnostic, not an independent-ensemble or epistemic-uncertainty estimate.

## Data

The package keeps only the datasets required by the public CATCH reproduction
scripts: the main 30 benchmark datasets and the 20-dataset External Validation 20
OpenML cohort. See `DATA_NOTICE.md`, `data/README.md`, and
`reproducibility/datasets_manifest.csv`.

External Validation 20 is recorded as a frozen, fixed-seed randomized OpenML
cohort: the fetch helper applies OpenML task metadata filters, excludes
main-benchmark source-family and prior-validation overlaps, excludes synthetic
benchmark-generator rows, assigns each eligible dataset a seeded pseudo-random
rank key (`selection_seed=20260624`), skips duplicate pandas content hashes and
duplicate name-family keys, and downloads the first 20 usable datasets in that
queue. The helper does not use model scores to choose datasets. Re-running the
fetch helper writes the frozen manifest, dataset list, and metadata selection
table; run `python scripts/audit_external_validation20.py` to verify the
bundled manifest, file set, pandas content hashes, random-selection seed, and
selection-rule field.

Before formal archival release or journal submission, verify redistribution
terms for each bundled dataset and replace `LICENSE_PENDING.md` with the final
project license.
