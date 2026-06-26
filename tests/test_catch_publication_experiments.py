import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DEPS_AVAILABLE = (
    importlib.util.find_spec("sklearn") is not None and importlib.util.find_spec("torch") is not None
)
TRAINING_DEPS_REASON = "requires scikit-learn and torch for estimator instantiation"


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_suite = load_module("run_catch_suite_for_catch_publication_tests", "experiments/catch/run_catch_suite.py")
analysis = load_module(
    "catch_publication_analysis_for_tests",
    "experiments/catch/analyze_catch_publication_experiments.py",
)
oof_check = load_module(
    "catch_oof_calibration_check_for_tests",
    "experiments/catch/run_oof_calibration_check.py",
)


class CatchPublicationExperimentTest(unittest.TestCase):
    @unittest.skipUnless(TRAINING_DEPS_AVAILABLE, TRAINING_DEPS_REASON)
    def test_paper_facing_aliases_and_simple_fusion_dispatch(self):
        self.assertEqual(run_suite.canonical_method_name("NN-only"), "NN-only")
        self.assertEqual(run_suite.canonical_method_name("Tree-on-Y"), "CatBoost")
        self.assertEqual(run_suite.canonical_method_name("CatBoost-on-Y"), "CatBoost")
        self.assertEqual(run_suite.canonical_method_name("NN+Tree-Avg"), "NN+Tree-Avg")
        self.assertEqual(run_suite.canonical_method_name("NN+Tree-LS"), "NN+Tree-LS")

        avg_estimator, avg_mode, avg_scaled = run_suite.build_catch_method(
            "NN+Tree-Avg",
            seed=42,
            input_dim=4,
            params={"epochs": 1, "iterations": 2, "n_jobs": 1},
        )
        ls_estimator, ls_mode, ls_scaled = run_suite.build_catch_method(
            "NN+Tree-LS",
            seed=42,
            input_dim=4,
            params={"epochs": 1, "iterations": 2, "n_jobs": 1},
        )

        self.assertEqual(avg_estimator.METHOD_NAME, "NN+Tree-Avg")
        self.assertEqual(ls_estimator.METHOD_NAME, "NN+Tree-LS")
        self.assertEqual(avg_mode, "semi")
        self.assertEqual(ls_mode, "semi")
        self.assertFalse(avg_scaled)
        self.assertFalse(ls_scaled)
        self.assertEqual(
            run_suite.method_role("NN+Tree-Avg"),
            "external_simple_neural_tree_fusion_baseline",
        )

    @unittest.skipUnless(TRAINING_DEPS_AVAILABLE, TRAINING_DEPS_REASON)
    def test_catch_ablation_dispatch(self):
        expected = [
            "CATCH",
            "CATCH-no-target-calibration",
            "CATCH-no-eta-scale",
            "CATCH-no-CWLS-fusion",
            "CATCH-rho0-complement",
            "CATCH-no-U",
            "CATCH-no-disagreement-variance",
            "CATCH-no-support-variance",
        ]
        args = type("Args", (), {"methods": None, "profile": "full"})()
        self.assertEqual(run_suite.experiment_methods("catch_ablation", args), expected)
        self.assertEqual(run_suite.canonical_method_name("CATCH-no-disagreement-variance"), "CATCH-no-disagreement-variance")
        self.assertEqual(run_suite.canonical_method_name("CATCH-no-support-variance"), "CATCH-no-support-variance")
        for method in expected[1:]:
            estimator, _mode, _scaled = run_suite.build_catch_method(
                method,
                seed=42,
                input_dim=4,
                params={"epochs": 1, "catboost_params": {"iterations": 2, "depth": 2}, "n_jobs": 1},
            )
            self.assertEqual(estimator.METHOD_NAME, method)
            self.assertTrue(run_suite.method_role(method).startswith("internal_mechanism_check"))

    def test_openml50_dispatch_is_two_method_comparison(self):
        args = type("Args", (), {"methods": None, "profile": "full"})()
        self.assertEqual(run_suite.experiment_methods("openml50_benchmark", args), ["CATCH", "AutoGluon"])

    def test_oof_audit_defaults_are_paper_facing(self):
        self.assertEqual(
            oof_check.DEFAULT_METHODS,
            ["CATCH", "AutoGluon", "CatBoost", "CATCH-rho0-complement"],
        )
        self.assertEqual(oof_check.DEFAULT_SEEDS, [42, 123, 456])

    def test_analysis_tables_from_seed_rows(self):
        rows = []
        methods = {
            "CATCH": [0.82, 0.84],
            "TabM": [0.74, 0.75],
            "CatBoost": [0.80, 0.81],
            "LightGBM": [0.79, 0.80],
            "XGBoost": [0.78, 0.79],
            "AutoGluon": [0.81, 0.82],
            "LapBoost": [0.76, 0.77],
            "VIME": [0.75, 0.76],
            "COREG": [0.70, 0.71],
            "RankUp": [0.73, 0.74],
            "UCVME": [0.72, 0.73],
            "NN-only": [0.74, 0.75],
            "NN+Tree-Avg": [0.805, 0.815],
            "NN+Tree-LS": [0.81, 0.82],
        }
        for dataset_idx, dataset in enumerate(["d1.csv", "d2.csv"]):
            for seed in [42, 123]:
                for method, values in methods.items():
                    value = values[dataset_idx] + (0.001 if seed == 123 else 0.0)
                    row = {
                        "Experiment": "main_benchmark",
                        "Protocol": "default",
                        "Dataset": dataset,
                        "Method": analysis.canonical_method_name(method),
                        "Seed": seed,
                        "Status": "ok",
                        "R2": value,
                        "RMSE": 1.0 - value,
                        "MAE": 0.5 - value / 10.0,
                        "Time": 1.0,
                    }
                    if method == "CATCH":
                        row["catch_vf_rc_eta_hat"] = 0.4 + 0.1 * dataset_idx
                        row["catch_vf_rc_rho"] = 0.6 - 0.1 * dataset_idx
                    rows.append(row)
        ablation_methods = {
            "CATCH": [0.82, 0.84],
            "CATCH-no-target-calibration": [0.79, 0.81],
            "CATCH-no-eta-scale": [0.80, 0.82],
            "CATCH-no-CWLS-fusion": [0.78, 0.80],
            "CATCH-rho0-complement": [0.805, 0.825],
            "CATCH-no-U": [0.77, 0.79],
            "CATCH-no-disagreement-variance": [0.795, 0.815],
            "CATCH-no-support-variance": [0.785, 0.805],
        }
        for dataset_idx, dataset in enumerate(["d1.csv", "d2.csv"]):
            for seed in [42, 123]:
                for method, values in ablation_methods.items():
                    value = values[dataset_idx] + (0.001 if seed == 123 else 0.0)
                    rows.append(
                        {
                            "Experiment": "catch_ablation",
                            "Protocol": "default",
                            "Dataset": dataset,
                            "Method": method,
                            "Seed": seed,
                            "Status": "ok",
                            "R2": value,
                            "RMSE": 1.0 - value,
                            "MAE": 0.5 - value / 10.0,
                            "Time": 1.0,
                        }
                    )
        for dataset_idx, dataset in enumerate(["h1.csv", "h2.csv"]):
            for seed in [42, 123]:
                for method, values in {"CATCH": [0.81, 0.83], "AutoGluon": [0.80, 0.82]}.items():
                    value = values[dataset_idx] + (0.001 if seed == 123 else 0.0)
                    rows.append(
                        {
                            "Experiment": "openml50_benchmark",
                            "Protocol": "default",
                            "Dataset": dataset,
                            "Method": method,
                            "Seed": seed,
                            "Status": "ok",
                            "R2": value,
                            "RMSE": 1.0 - value,
                            "MAE": 0.5 - value / 10.0,
                            "Time": 1.0,
                        }
                    )
        external_datasets = [
            "external_validation20/OpenMLEV20_41022_Short_Track_Speed_Skating.csv",
            "external_validation20/OpenMLEV20_41187_mauna_loa_atmospheric_co2.csv",
            "external_validation20/OpenMLEV20_31_credit_g.csv",
        ]
        for dataset_idx, dataset in enumerate(external_datasets):
            for seed in [42, 123]:
                for method, values in {"CATCH": [0.91, 0.92, 0.55], "AutoGluon": [0.90, 0.91, 0.51]}.items():
                    value = values[dataset_idx] + (0.001 if seed == 123 else 0.0)
                    rows.append(
                        {
                            "Experiment": "external_validation",
                            "Protocol": "default",
                            "Dataset": dataset,
                            "Method": method,
                            "Seed": seed,
                            "Status": "ok",
                            "R2": value,
                            "RMSE": 1.0 - value,
                            "MAE": 0.5 - value / 10.0,
                            "Time": 1.0,
                        }
                    )
        seed_df = pd.DataFrame(rows)

        ds = analysis.dataset_means(seed_df)
        main_ds = analysis.main_default_dataset_means(ds)
        ablation_ds = analysis.catch_ablation_dataset_means(ds)
        external_ordinary_ds = analysis.external_validation_ordinary_dataset_means(ds)
        external_stress_ds = analysis.external_validation_stress_dataset_means(ds)
        openml50_ds = analysis.openml50_dataset_means(ds)
        main_summary = analysis.summarize_methods(main_ds, analysis.PUBLIC_MAIN_METHODS)
        ablation_summary = analysis.summarize_methods(ablation_ds, analysis.CATCH_ABLATION_METHODS)
        external_ordinary_summary = analysis.summarize_methods_with_ranks(
            external_ordinary_ds,
            analysis.EXTERNAL_VALIDATION_METHODS,
        )
        external_stress_summary = analysis.summarize_methods_with_ranks(
            external_stress_ds,
            analysis.EXTERNAL_VALIDATION_METHODS,
        )
        external_ordinary_pairwise = analysis.pairwise_vs_reference_compact(external_ordinary_ds)
        openml50_summary = analysis.summarize_methods(openml50_ds, analysis.OPENML50_METHODS)
        simple_summary = analysis.summarize_methods(
            analysis.clone_tree_on_y(main_ds),
            analysis.SIMPLE_FUSION_METHODS,
        )
        pairwise = analysis.pairwise_vs_reference(main_ds, bootstrap_samples=100, seed=1)
        ablation_pairwise = analysis.pairwise_vs_reference(ablation_ds, bootstrap_samples=100, seed=1)
        ablation_components = analysis.catch_ablation_component_deltas(ablation_pairwise)
        diagnostics = analysis.diagnostic_summary(seed_df)

        self.assertIn("CATCH", set(main_summary["Method"]))
        self.assertIn("TabM", set(main_summary["Method"]))
        self.assertIn("LapBoost", set(main_summary["Method"]))
        self.assertIn("UCVME", set(main_summary["Method"]))
        self.assertIn("CATCH-no-U", set(ablation_summary["Method"]))
        self.assertIn("CATCH-no-support-variance", set(ablation_summary["Method"]))
        self.assertEqual(external_ordinary_ds["Dataset"].nunique(), 2)
        self.assertEqual(external_stress_ds["Dataset"].nunique(), 1)
        self.assertIn("Top3_Count", set(external_ordinary_summary.columns))
        self.assertEqual(int(external_stress_summary.loc[external_stress_summary["Method"].eq("CATCH"), "Best_Count"].iloc[0]), 1)
        self.assertIn("Holm_Wilcoxon_p", set(external_ordinary_pairwise.columns))
        self.assertIn("AutoGluon", set(openml50_summary["Method"]))
        self.assertIn("Tree-on-Y", set(simple_summary["Method"]))
        self.assertIn("NN+Tree-Avg", set(simple_summary["Method"]))
        self.assertTrue((pairwise["Comparison"].str.startswith("CATCH vs ")).all())
        self.assertIn("Wilcoxon_p_Holm", set(pairwise.columns))
        self.assertGreater(float(pairwise.loc[pairwise["Baseline"].eq("CatBoost"), "Mean_Delta_R2"].iloc[0]), 0.0)
        self.assertGreater(
            float(ablation_pairwise.loc[ablation_pairwise["Baseline"].eq("CATCH-no-U"), "Mean_Delta_R2"].iloc[0]),
            0.0,
        )
        self.assertIn("eta_hat", set(diagnostics["Quantity"]))
        self.assertIn("rho_hat", set(diagnostics["Quantity"]))
        self.assertIn("training-side unlabeled covariate support", set(ablation_components["Component"]))


if __name__ == "__main__":
    unittest.main()
