# catch

Code and public data for reproducing the CATCH experiments:

**CATCH: Audit-Oriented Complementary Neural-Tree Fusion for Label-Scarce
Tabular Regression**

Repository: https://github.com/funkyallen/catch

This is a code-and-data-only GitHub package. It does not include generated
result tables, figures, logs, trained model artifacts, or manuscript build
outputs.

## Layout

```text
core/              CATCH implementation, supervised and semi-supervised baselines
experiments/catch/ Main runner and table-analysis script
scripts/           External Validation 20 audit helper
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

The ablation runner also exposes `CATCH-rho0-complement`, a fixed-`rho=0`
eta-complement control for measuring the net contribution of the final scalar
rho readout. The OOF audit runner can include the same control.

Implementation note: CATCH uses residual-calibrated diagnostic scales for
support weighting. The CatBoost staged-response shape is an optimization-path
diagnostic, not an independent-ensemble or epistemic-uncertainty estimate.

## Data

The package keeps only the datasets required by the public CATCH reproduction
scripts: the main 30 benchmark datasets and the 20-dataset External Validation 20
OpenML cohort. See `DATA_NOTICE.md`, `data/README.md`, and
`reproducibility/datasets_manifest.csv`.

External Validation 20 is a bundled OpenML external-validation cohort kept
separate from the main 30-dataset benchmark. The manifest records the exact
OpenML data IDs, local files, target column used by the runner, row/feature
counts, pandas content hashes, cohort order, and access URLs. Run
`python scripts/audit_external_validation20.py` to verify the bundled manifest,
file set, target columns, and content hashes before running the five-seed
external-validation experiment.

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
