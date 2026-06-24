# Expected Outputs

Generated outputs are not bundled in this GitHub package. They are created
locally under `r/` and `paper/tables/`.

`run_smoke.ps1` writes temporary checks under `r/smoke/`.

Primary tables are written to `paper/tables/catch_publication_experiments/`:

- `seed_combined.csv`
- `dataset_mean.csv`
- `dataset_audit.csv`
- `main_default_dataset_mean.csv`
- `main_summary.csv`
- `main_pairwise.csv`
- `main_benchmark_full_seed_combined.csv`
- `main_benchmark_full_dataset_method_mean.csv`
- `main_benchmark_full_method_summary.csv`
- `main_benchmark_full_pairwise_vs_catch.csv`
- `simple_fusion_summary.csv`
- `simple_fusion_pairwise.csv`
- `catch_ablation_dataset_mean.csv`
- `catch_ablation_summary.csv`
- `catch_ablation_pairwise.csv`
- `catch_ablation_component_deltas.csv`
- `external_validation_dataset_mean.csv`
- `external_validation_summary.csv`
- `external_validation_pairwise.csv`
- `external_validation_full_seed_combined.csv`
- `external_validation_full_dataset_method_mean.csv`
- `external_validation_full_method_summary.csv`
- `external_validation_full_pairwise_vs_catch.csv`
- `friedman.csv`
- `diagnostics.csv`

The optional OOF audit runner writes seed/fold-level rows under
`r/catch_oof_calibration/`, for example:

- `seed_oof_calibration_oof_calibration.csv`
