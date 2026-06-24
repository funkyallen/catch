param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python experiments/catch/run_catch_publication_experiments.py `
  --experiments external_validation `
  --tag-prefix external_validation `
  --run-root r/catch_external_validation `
  --log-root r/catch_external_validation_lane_logs `
  --seeds 42 123 456 789 1011 `
  --max-parallel-lanes 4

& $Python experiments/catch/analyze_catch_publication_experiments.py `
  --run-roots r/catch_publication r/catch_publication_ablation r/catch_external_validation `
  --out-dir paper/tables/catch_publication_experiments
