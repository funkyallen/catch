param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python experiments/catch/run_oof_calibration_check.py `
  --datasets OpenML_airfoil_self_noise.csv OpenML_debutanizer.csv Review_Real_Estate_Valuation.csv Review_Appliances_Energy.csv `
  --methods CATCH AutoGluon CatBoost CATCH-rho0-complement `
  --seeds 42 123 `
  --folds 3 `
  --tag oof_representative `
  --log-dir r/catch_oof_calibration/oof_representative
