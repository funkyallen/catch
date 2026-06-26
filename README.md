# catch

Code and public data for reproducing the CATCH experiments:

**CATCH: Transparent Neural-Tree Complementation for Failure-Aware Label-Scarce
Tabular Regression**

Repository: https://github.com/funkyallen/catch

This is a code-and-data-only GitHub package. It does not include generated
result tables, figures, logs, trained model artifacts, or manuscript build
outputs.

## Layout

```text
core/              CATCH implementation, supervised and semi-supervised baselines
experiments/catch/ Main runner and table-analysis script
scripts/           Additional OpenML cohort audit helper
tests/             Lightweight CATCH dispatch and analysis tests
reproducibility/   Environment files, run scripts, seeds, manifests
data/              Main 30 datasets and additional OpenML cohort datasets
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

For editable local installs of the core package metadata:

```powershell
python -m pip install -e .
```

Install the full training stack with:

```powershell
python -m pip install -e ".[full]"
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
powershell -ExecutionPolicy Bypass -File reproducibility/run_oof_audit.ps1 -Python python
powershell -ExecutionPolicy Bypass -File reproducibility/run_openml50_autogluon_catch.ps1 -Python python -PlanOnly
```

Outputs are written under `r/` and `paper/tables/`, both ignored by Git.

An auxiliary out-of-fold audit can be run directly when a reviewer asks for a
held-out calibration/generalization check:

```powershell
python experiments/catch/run_oof_calibration_check.py --datasets OpenML_airfoil_self_noise.csv OpenML_debutanizer.csv Review_Real_Estate_Valuation.csv Review_Appliances_Energy.csv SGEMM_GPU.csv Review_QSAR_Aquatic_Toxicity.csv OpenML_wine_quality.csv Review_Energy_Efficiency.csv --methods CATCH AutoGluon CatBoost CATCH-rho0-complement --seeds 42 123 --folds 3 --tag oof_8dataset_audit
```

The main publication runner covers the paper-facing non-foundation comparison
set: CATCH, AutoGluon, TabM, CatBoost, XGBoost, LightGBM, LapBoost, VIME,
COREG, RankUp, and UCVME. TabPFN-v3 is an optional external reference and is
not bundled with this code-and-data package.

The ablation runner distinguishes two final-readout controls. `CATCH-no-CWLS-fusion`
keeps the eta-normalized complement but replaces the learned constrained readout
with a fixed 0.5 neural/complement blend. `CATCH-rho0-complement` fixes `rho=0`
and therefore tests the complement-only endpoint. The OOF audit runner can
include the same complement-only control.

The optional OpenML-50 runner is a focused CATCH-vs-AutoGluon expansion:
50 OpenML data IDs, 10 seeds, and two methods. The first local build can use a
downloaded source pool via `-SourcePool`; later runs can rebuild missing CSVs
from the committed manifest through scikit-learn's OpenML interface. Generated
OpenML-50 CSV files and result tables are local artifacts and are not committed.

Implementation note: CATCH uses residual-calibrated diagnostic scales for
support weighting. The CatBoost staged-response shape is an optimization-path
diagnostic, not an independent-ensemble or epistemic-uncertainty estimate.

## Data

The package keeps only the datasets required by the public CATCH reproduction
scripts: the main 30 benchmark datasets and the additional OpenML cohort. See
`DATA_NOTICE.md`, `data/README.md`, and
`reproducibility/datasets_manifest.csv`.

The additional OpenML cohort is kept separate from the main 30-dataset
benchmark. Its legacy folder name is `external_validation20/` because the
original reproducibility bundle contains 20 files. The primary paper aggregate
uses the 19 ordinary numeric-target regression tasks, while the derived
credit-amount regression row from OpenML `credit_g` is documented as a separate
stress task. The manifest records the exact OpenML data IDs, local files,
runner target columns, task-type notes, row/feature counts, pandas content
hashes, cohort order, and access URLs. Run
`python scripts/audit_external_validation20.py` to verify the bundled manifest,
file set, task-type notes, target columns, and content hashes before running the
five-seed additional-cohort experiment.

The main dataset manifest marks dataset license status as
`verify_upstream_before_archival_or_submission` until source-specific
redistribution terms are finalized.

## License

The repository software is released under the MIT License. Bundled public
datasets remain subject to their upstream terms; see `DATA_NOTICE.md` and the
dataset manifests for source and provenance metadata.

## Citation

Repository citation metadata are provided in `CITATION.cff`. The manuscript
should be cited when bibliographic details are finalized.
