# CATCH Reproducibility

This folder contains the run scripts, environment files, seeds, and manifests
for the slim public CATCH package.

The repository intentionally contains code and data only. Running the commands
below creates local outputs that are ignored by Git.

## Scope

- Main benchmark: 30 datasets, 10 seeds, CATCH and the 10 paper-facing
  non-foundation baselines.
- Label-ratio and unlabeled-contamination sweeps on the same main datasets.
- CATCH ablation: target calibration, eta scaling, CWLS fusion, unlabeled data,
  disagreement-scale diagnostics, and support-scale diagnostics.
- External Validation 20: final 20 OpenML regression datasets, five seeds.
- Optional OOF audit: outer K-fold held-out evaluation for selected datasets,
  methods, and seeds.

## Protocol Notes

- `build_payload_for_protocol` creates train/labeled, unlabeled, and evaluation
  splits from fixed seeds.
- Feature preprocessing is fit on `D_L^X union D_U`; evaluation covariates and
  labels are not used for training.
- Target scaling, when requested by a method, is fit on `D_L^Y` only.
- CATCH diagnostic-scale fields are residual-calibrated weighting scales, not
  calibrated epistemic-uncertainty estimates.
- `datasets_manifest.csv` records the main 30 target columns and dataset audit
  metadata. Its license-status field is conservative and should be finalized
  before archival release or journal submission.
- `data/external_validation20/manifest.csv` records the External Validation 20
  cohort, selected order, OpenML data IDs, target columns, access URLs, and
  content hashes used by the public runner.
- `seeds.csv` records the public seed cohorts used by the scripts.
- The ordinary non-foundation method set is CATCH, AutoGluon, TabM, CatBoost,
  XGBoost, LightGBM, LapBoost, VIME, COREG, RankUp, and UCVME. TabPFN-v3 is
  treated as an optional external reference because its pretrained model and
  license are managed outside this slim public package.

## Commands

Smoke check:

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/run_smoke.ps1 -Python python
```

External-validation manifest audit only:

```powershell
python scripts/audit_external_validation20.py
```

Full reproduction entry points:

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/run_main.ps1 -Python python
powershell -ExecutionPolicy Bypass -File reproducibility/run_ablation.ps1 -Python python
powershell -ExecutionPolicy Bypass -File reproducibility/run_external_validation.ps1 -Python python
powershell -ExecutionPolicy Bypass -File reproducibility/run_oof_audit.ps1 -Python python
```

Optional OOF audit example:

```powershell
python experiments/catch/run_oof_calibration_check.py --datasets OpenML_airfoil_self_noise.csv Review_Real_Estate_Valuation.csv --methods CATCH AutoGluon CatBoost CATCH-rho0-complement --seeds 42 123 --folds 5
```

See `expected_outputs.md` for generated table names.
