param(
  [string]$Python = "python",
  [string]$SourcePool = "",
  [int]$AutogluonTimeLimit = 60,
  [int]$MaxParallelLanes = 1,
  [switch]$SkipBuild,
  [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if (-not $SkipBuild) {
  if ($SourcePool -ne "") {
    Write-Host "[openml50] building data bundle from source pool"
    & $Python scripts/build_openml50_benchmark.py --source-pool $SourcePool
  } elseif (Test-Path "data/openml50_benchmark/manifest.csv") {
    Write-Host "[openml50] ensuring data bundle from manifest"
    & $Python scripts/build_openml50_benchmark.py --download-missing
  } else {
    throw "OpenML-50 manifest is missing. Pass -SourcePool for the first build."
  }
}

Write-Host "[openml50] auditing manifest and bundled files"
& $Python scripts/audit_openml50_benchmark.py

$RunArgs = @(
  "experiments/catch/run_catch_publication_experiments.py",
  "--experiments", "openml50_benchmark",
  "--methods", "CATCH", "AutoGluon",
  "--seeds", "42", "123", "456", "789", "1011", "2027", "3141", "2718", "1618", "9001",
  "--tag-prefix", "openml50_catch_autogluon",
  "--run-root", "r/openml50_benchmark",
  "--log-root", "r/openml50_benchmark_lane_logs",
  "--autogluon-time-limit", "$AutogluonTimeLimit",
  "--max-parallel-lanes", "$MaxParallelLanes"
)

if ($PlanOnly) {
  $RunArgs += "--plan-only"
}

Write-Host "[openml50] running CATCH vs AutoGluon benchmark"
& $Python @RunArgs

if (-not $PlanOnly) {
  Write-Host "[openml50] analyzing OpenML-50 results"
  & $Python experiments/catch/analyze_catch_publication_experiments.py `
    --run-roots r/openml50_benchmark `
    --out-dir paper/tables/openml50_benchmark
}
