param(
  [string]$Python = "python",
  [switch]$SkipTraining,
  [switch]$RequireTrainingDeps
)

$ErrorActionPreference = "Stop"

$SmokeRoot = "r/smoke"
if (Test-Path $SmokeRoot) {
  Remove-Item -LiteralPath $SmokeRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

Write-Host "[smoke] planning wrapper commands"
& $Python experiments/catch/run_catch_publication_experiments.py `
  --experiments dataset_audit `
  --datasets OpenML_airfoil_self_noise.csv `
  --seeds 42 `
  --tag-prefix smoke_plan `
  --run-root "$SmokeRoot/plan" `
  --log-root "$SmokeRoot/logs" `
  --plan-only

Write-Host "[smoke] reading one bundled dataset and writing audit output"
& $Python experiments/catch/run_catch_suite.py `
  --experiments dataset_audit `
  --datasets OpenML_airfoil_self_noise.csv `
  --seeds 42 `
  --tag smoke_audit `
  --log-dir "$SmokeRoot/audit" `
  --smoke

Write-Host "[smoke] auditing additional OpenML cohort manifest and bundled files"
& $Python scripts/audit_external_validation20.py

$SeedFile = "$SmokeRoot/seed_smoke_fixture.csv"
@'
Experiment,Protocol,Dataset,Method,Seed,Status,R2,RMSE,MAE,Time,catch_vf_rc_eta_hat,catch_vf_rc_rho
main_benchmark,default,d1.csv,CATCH,42,ok,0.82,0.18,0.10,1.0,0.40,0.60
main_benchmark,default,d1.csv,NN-only,42,ok,0.74,0.26,0.16,1.0,,
main_benchmark,default,d1.csv,CatBoost,42,ok,0.80,0.20,0.12,1.0,,
main_benchmark,default,d1.csv,LightGBM,42,ok,0.79,0.21,0.13,1.0,,
main_benchmark,default,d1.csv,XGBoost,42,ok,0.78,0.22,0.14,1.0,,
main_benchmark,default,d1.csv,AutoGluon,42,ok,0.81,0.19,0.11,1.0,,
catch_ablation,default,d1.csv,CATCH,42,ok,0.82,0.18,0.10,1.0,,
catch_ablation,default,d1.csv,CATCH-no-target-calibration,42,ok,0.79,0.21,0.13,1.0,,
catch_ablation,default,d1.csv,CATCH-no-eta-scale,42,ok,0.80,0.20,0.12,1.0,,
catch_ablation,default,d1.csv,CATCH-no-CWLS-fusion,42,ok,0.78,0.22,0.14,1.0,,
catch_ablation,default,d1.csv,CATCH-no-U,42,ok,0.77,0.23,0.15,1.0,,
catch_ablation,default,d1.csv,CATCH-no-disagreement-variance,42,ok,0.795,0.205,0.125,1.0,,
catch_ablation,default,d1.csv,CATCH-no-support-variance,42,ok,0.785,0.215,0.135,1.0,,
external_validation,default,h1.csv,CATCH,42,ok,0.76,0.24,0.18,1.0,,
external_validation,default,h1.csv,CatBoost,42,ok,0.73,0.27,0.20,1.0,,
'@ | Set-Content -LiteralPath $SeedFile -Encoding UTF8

Write-Host "[smoke] running table analysis on a tiny fixture"
& $Python experiments/catch/analyze_catch_publication_experiments.py `
  --seed-files $SeedFile `
  --audit-files "$SmokeRoot/audit/dataset_audit_smoke_audit.csv" `
  --out-dir "$SmokeRoot/tables" `
  --bootstrap-samples 10

if (-not $SkipTraining) {
  Write-Host "[smoke] checking optional training dependencies"
  & $Python -c "import importlib.util, sys; missing=[m for m in ['sklearn','torch','catboost','tabm'] if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing)); sys.exit(2 if missing else 0)"
  $DepCode = $LASTEXITCODE
  if ($DepCode -eq 0) {
    Write-Host "[smoke] running one tiny training job: CatBoost + CATCH, one dataset, one seed"
    & $Python experiments/catch/run_catch_suite.py `
      --experiments main_benchmark `
      --datasets Review_Real_Estate_Valuation.csv `
      --seeds 42 `
      --methods CatBoost CATCH `
      --tag smoke_train `
      --log-dir "$SmokeRoot/train" `
      --tree-rounds 2 `
      --tree-depth 2 `
      --autogluon-time-limit 10 `
      --smoke
  } elseif ($RequireTrainingDeps) {
    throw "Training dependency smoke failed. Install reproducibility/requirements.txt and rerun."
  } else {
    Write-Host "[smoke] training dependency smoke skipped because optional ML packages are missing"
  }
}

Write-Host "[smoke] complete"
