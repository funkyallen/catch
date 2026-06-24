param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python experiments/catch/run_catch_publication_experiments.py `
  --experiments catch_ablation `
  --tag-prefix abl `
  --run-root r/catch_publication_ablation `
  --log-root r/catch_publication_ablation_lane_logs `
  --max-parallel-lanes 4

& $Python experiments/catch/analyze_catch_publication_experiments.py `
  --run-roots r/catch_publication r/catch_publication_ablation r/catch_external_validation `
  --out-dir paper/tables/catch_publication_experiments
