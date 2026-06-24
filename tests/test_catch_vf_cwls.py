import unittest
import importlib.util

import numpy as np

if importlib.util.find_spec("sklearn") is None or importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("CATCH estimator tests require scikit-learn and torch.")

from core.catch_vf_cwls import (
    CATCHNoCWLSFusionRegressor,
    CATCHNoDisagreementVarianceRegressor,
    CATCHNoEtaScaleRegressor,
    CATCHNoSupportVarianceRegressor,
    CATCHNoTargetCalibrationRegressor,
    CATCHNoUnlabeledRegressor,
    CATCHRegressor,
    CATCHVFCWLSRegressor,
)
from experiments.catch.run_catch_suite import build_catch_method, canonical_method_name, method_role


class CATCHVFCWLSTest(unittest.TestCase):
    def test_public_catch_classes_are_paper_facing(self):
        expected = {
            CATCHRegressor: ("CATCH", "rc_eta_lite"),
            CATCHNoTargetCalibrationRegressor: ("CATCH-no-target-calibration", "catch_no_target_calibration"),
            CATCHNoEtaScaleRegressor: ("CATCH-no-eta-scale", "catch_no_eta_scale"),
            CATCHNoCWLSFusionRegressor: ("CATCH-no-CWLS-fusion", "catch_no_cwls_fusion"),
            CATCHNoUnlabeledRegressor: ("CATCH-no-U", "catch_no_u"),
            CATCHNoDisagreementVarianceRegressor: ("CATCH-no-disagreement-variance", "no_disagreement"),
            CATCHNoSupportVarianceRegressor: ("CATCH-no-support-variance", "no_variance"),
        }
        for cls, (method, mode) in expected.items():
            with self.subTest(method=method):
                self.assertEqual(cls.METHOD_NAME, method)
                self.assertEqual(cls.ABLATION_MODE, mode)

    def test_runner_dispatches_only_current_catch_methods(self):
        expected = {
            "CATCH": "proposed_catch_complementary_adaptive_target_calibration",
            "CATCH-VF-CWLS": "proposed_catch_variational_field_complementary_barycentric_cwls",
            "CATCH-no-target-calibration": "internal_mechanism_check_catch_no_target_calibration",
            "CATCH-no-eta-scale": "internal_mechanism_check_catch_no_eta_scale",
            "CATCH-no-CWLS-fusion": "internal_mechanism_check_catch_no_cwls_fusion",
            "CATCH-no-U": "internal_mechanism_check_catch_no_u",
            "CATCH-no-disagreement-variance": "internal_mechanism_check_catch_no_disagreement_variance",
            "CATCH-no-support-variance": "internal_mechanism_check_catch_no_support_variance",
        }
        for method, role in expected.items():
            with self.subTest(method=method):
                estimator, mode, use_scaled_y = build_catch_method(
                    method,
                    seed=42,
                    input_dim=4,
                    params={"epochs": 1, "catboost_params": {"iterations": 2}, "n_jobs": 1},
                )
                self.assertEqual(canonical_method_name(method), method)
                self.assertEqual(estimator.METHOD_NAME, method)
                self.assertEqual(mode, "semi")
                self.assertFalse(use_scaled_y)
                self.assertEqual(method_role(method), role)

    def test_unknown_catch_entries_are_not_public_api(self):
        with self.assertRaises(ValueError):
            build_catch_method(
                "CATCH-unpublished-exploratory-mode",
                seed=42,
                input_dim=4,
                params={"epochs": 1, "catboost_params": {"iterations": 2}, "n_jobs": 1},
            )

    def test_unpublished_internal_modes_are_rejected_at_fit_time(self):
        model = CATCHVFCWLSRegressor(ablation_mode="unpublished_exploratory_mode")
        X = np.zeros((3, 2), dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            model.fit(X, y)


if __name__ == "__main__":
    unittest.main()
