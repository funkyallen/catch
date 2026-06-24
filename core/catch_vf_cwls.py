"""CATCH-VF-CWLS public estimator.

The "variance" fields in this module are residual-calibrated diagnostic
scales used for weighting and audit outputs. CatBoost staged predictions are
sequential boosting checkpoints, not independent ensemble samples, so their
spread is treated only as a staged-response shape diagnostic.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.catch_base import _CATCHBaseRegressor, _as_2d_float, _safe_vector, _weighted_mse
from core.utils import set_seed


class CATCHVFCWLSRegressor(_CATCHBaseRegressor):
    """Audit-oriented CATCH with one neural anchor and one vector tree."""

    METHOD_NAME = "CATCH-VF-CWLS"
    ABLATION_MODE = "full"

    def __init__(
        self,
        random_state=42,
        epochs=100,
        batch_size=512,
        learning_rate=2e-3,
        weight_decay=1e-4,
        catboost_params=None,
        library_defaults=True,
        device="auto",
        n_jobs=None,
        variance_floor=1e-6,
        tree_shape_stages=64,
        ablation_mode=None,
        verbose=False,
    ):
        super().__init__(
            random_state=random_state,
            beta=1.0,
            em_steps=0,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            catboost_params=catboost_params,
            library_defaults=library_defaults,
            device=device,
            n_jobs=n_jobs,
            variance_floor=variance_floor,
            df_fraction_neural=0.0,
            df_fraction_tree=0.0,
            verbose=verbose,
        )
        self.tree_shape_stages = tree_shape_stages
        self.ablation_mode = str(ablation_mode or self.ABLATION_MODE)

    def _vector_tree_params(self, offset=0):
        params = self._catboost_params(offset=offset)
        params.setdefault("loss_function", "MultiRMSE")
        return params

    def _fit_vector_tree(self, X, y_2d, sample_weight=None, offset=0):
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise RuntimeError("CATCH-VF-CWLS requires the optional `catboost` package.") from exc
        model = CatBoostRegressor(**self._vector_tree_params(offset=offset))
        y_arr = np.asarray(y_2d, dtype=np.float32)
        if y_arr.ndim != 2 or y_arr.shape[1] != 2:
            raise ValueError("Vector tree targets must have shape (n_samples, 2)")
        if sample_weight is None:
            model.fit(X, y_arr)
        else:
            model.fit(X, y_arr, sample_weight=np.maximum(_safe_vector(sample_weight), 0.0).astype(np.float32))
        return model

    @staticmethod
    def _vector_prediction(model, X):
        pred = np.asarray(model.predict(_as_2d_float(X)), dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.shape[1] < 2:
            raise RuntimeError("Vector tree prediction did not return two coordinates")
        return np.nan_to_num(pred[:, :2], nan=0.0, posinf=0.0, neginf=0.0)

    def _vector_tree_shape(self, model, X, coord=0, max_stages=None):
        """Return a staged-response shape diagnostic for one tree coordinate.

        This is intentionally conservative: staged CatBoost predictions trace
        a sequential optimization path, not independent posterior draws. The
        returned shape is later calibrated by labeled residuals and used as a
        nonnegative weighting scale, not as calibrated epistemic uncertainty.
        """
        X_arr = _as_2d_float(X)
        max_stages = int(self.tree_shape_stages if max_stages is None else max_stages)
        try:
            tree_count = int(getattr(model, "tree_count_", 0) or 0)
            if tree_count <= 1:
                raise ValueError("not enough CatBoost trees for staged shape")
            count = max(2, min(max_stages, tree_count))
            endpoints = np.unique(np.linspace(1, tree_count, count, dtype=int))
            stages = []
            for end in endpoints:
                pred = np.asarray(model.predict(X_arr, ntree_start=0, ntree_end=int(end)), dtype=np.float64)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                stages.append(_safe_vector(pred[:, int(coord)]))
            if len(stages) >= 2:
                return np.var(np.vstack(stages), axis=0, ddof=1)
        except Exception:
            pass
        return np.ones(len(X_arr), dtype=np.float64)

    def _fit_shape_scale(self, raw_values, attr_name):
        raw = np.maximum(_safe_vector(raw_values) + float(self.variance_floor), float(self.variance_floor))
        scale = float(np.mean(raw)) if len(raw) else 1.0
        if not np.isfinite(scale) or scale <= float(self.variance_floor):
            scale = 1.0
        setattr(self, attr_name, scale)
        return (raw / scale).astype(np.float64)

    def _apply_shape_scale(self, raw_values, attr_name):
        raw = np.maximum(_safe_vector(raw_values) + float(self.variance_floor), float(self.variance_floor))
        scale = float(getattr(self, attr_name, 1.0))
        if not np.isfinite(scale) or scale <= float(self.variance_floor):
            scale = 1.0
        return (raw / scale).astype(np.float64)

    def _calibrate_variance_scale(self, residual, relative_shape):
        """Calibrate a diagnostic shape by labeled residual energy.

        The historical method name is kept for compatibility with saved
        metrics, but the returned value is a scale factor for weighting rather
        than a standalone predictive-variance estimate.
        """
        res2 = _safe_vector(residual) ** 2
        shape = np.maximum(_safe_vector(relative_shape), float(self.variance_floor))
        scale = float(np.mean(res2 / shape)) if len(res2) else 1.0
        return max(scale, float(self.variance_floor))

    def _calibrate_en_variance_scale(self, residual):
        res2 = _safe_vector(residual) ** 2
        scale = float(np.mean(res2)) if len(res2) else 1.0
        return max(scale, float(self.variance_floor))

    def _eb_labeled_complement_target(self, y, g, v_g):
        y = _safe_vector(y).astype(np.float64)
        g = _safe_vector(g).astype(np.float64)
        v_g = np.maximum(_safe_vector(v_g).astype(np.float64), float(self.variance_floor))
        residual = y - g
        tau2 = float(np.var(residual, ddof=1)) if len(residual) > 1 else float(np.var(residual))
        tau2 = max(tau2, float(self.variance_floor))
        eta = tau2 / np.maximum(tau2 + v_g, float(self.variance_floor))
        eta = np.clip(np.nan_to_num(eta, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        target = y + eta * residual
        denom = float(np.sum(residual**2) + float(self.variance_floor))
        eta_hat = float(np.clip(np.sum(residual * (target - y)) / denom, 0.0, 1.0)) if denom > 0.0 else 0.0
        return target.astype(np.float64), eta.astype(np.float64), eta_hat

    def _closed_form_precision_rho(self, y, g, h, v_g=None, v_h=None):
        """Compute the one-parameter inverse-scale weighted fusion coefficient.

        The private name is retained for compatibility with older experiment
        outputs. The weights are based on residual-calibrated diagnostic
        scales, not on a claim of calibrated statistical precision.
        """
        y = _safe_vector(y).astype(np.float64)
        g = _safe_vector(g).astype(np.float64)
        h = _safe_vector(h).astype(np.float64)
        if len(y) == 0:
            return 0.5, 0.0, 0.0
        d = g - h
        target = y - h
        if v_g is None or v_h is None:
            w = np.ones(len(y), dtype=np.float64)
        else:
            v = np.maximum(_safe_vector(v_g).astype(np.float64), float(self.variance_floor)) + np.maximum(
                _safe_vector(v_h).astype(np.float64), float(self.variance_floor)
            )
            w = 1.0 / np.maximum(v, float(self.variance_floor))
        numerator = float(np.sum(w * d * target))
        denominator = float(np.sum(w * d * d))
        if not np.isfinite(denominator) or denominator <= float(self.variance_floor):
            rho = 0.5
        else:
            rho = float(np.clip(numerator / (denominator + float(self.variance_floor)), 0.0, 1.0))
        return rho, numerator, denominator

    def _robust_unlabeled_moment_scale(self, target_l, target_u):
        l_arr = np.asarray(target_l, dtype=np.float64)
        u_arr = np.asarray(target_u, dtype=np.float64)
        if l_arr.ndim != 2 or u_arr.ndim != 2 or l_arr.shape[1] < 2 or u_arr.shape[1] < 2 or len(l_arr) == 0 or len(u_arr) == 0:
            return 1.0, {
                "lambda_u_base": 0.0,
                "lambda_u_robust": 0.0,
                "scale": 1.0,
                "delta_lu": 0.0,
                "v_l": 0.0,
                "v_u": 0.0,
            }

        def moment_rows(arr):
            a = arr[:, 0]
            c = arr[:, 1]
            return np.column_stack([a * a, a * c, c * c]).astype(np.float64)

        m_l_rows = moment_rows(l_arr)
        m_u_rows = moment_rows(u_arr)
        m_l = np.mean(m_l_rows, axis=0)
        m_u = np.mean(m_u_rows, axis=0)
        v_l = float(np.sum(np.var(m_l_rows, axis=0, ddof=1)) / max(1.0, float(len(m_l_rows)))) if len(m_l_rows) > 1 else 0.0
        v_u = float(np.sum(np.var(m_u_rows, axis=0, ddof=1)) / max(1.0, float(len(m_u_rows)))) if len(m_u_rows) > 1 else 0.0
        delta_lu = float(np.sum((m_l - m_u) ** 2))
        v_l = max(v_l, float(self.variance_floor))
        v_u = max(v_u, float(self.variance_floor))
        lambda_base = float(v_l / (v_l + v_u + float(self.variance_floor)))
        lambda_robust = float(v_l / (v_l + v_u + delta_lu + float(self.variance_floor)))
        scale = float(lambda_robust / max(lambda_base, float(self.variance_floor)))
        scale = float(np.clip(np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
        return scale, {
            "lambda_u_base": lambda_base,
            "lambda_u_robust": lambda_robust,
            "scale": scale,
            "delta_lu": delta_lu,
            "v_l": v_l,
            "v_u": v_u,
        }

    def _fit_coord_scale(self, values, prefix):
        arr = _safe_vector(values).astype(np.float64)
        mean = float(np.mean(arr)) if len(arr) else 0.0
        centered = arr - mean
        var = float(np.mean(centered**2)) if len(arr) else 1.0
        scale = float(np.sqrt(max(var, float(self.variance_floor))))
        setattr(self, f"{prefix}_mean_", mean)
        setattr(self, f"{prefix}_scale_", scale)
        setattr(self, f"{prefix}_var_", max(var, float(self.variance_floor)))
        return mean, scale, max(var, float(self.variance_floor))

    def _coord_normalize(self, values, prefix):
        mean = float(getattr(self, f"{prefix}_mean_", 0.0))
        scale = float(getattr(self, f"{prefix}_scale_", 1.0))
        if not np.isfinite(scale) or scale <= float(self.variance_floor):
            scale = 1.0
        return ((_safe_vector(values) - mean) / (scale + float(self.variance_floor))).astype(np.float64)

    def _coord_denormalize(self, values, prefix):
        mean = float(getattr(self, f"{prefix}_mean_", 0.0))
        scale = float(getattr(self, f"{prefix}_scale_", 1.0))
        if not np.isfinite(scale) or scale <= float(self.variance_floor):
            scale = 1.0
        return (mean + (scale + float(self.variance_floor)) * _safe_vector(values)).astype(np.float64)

    def _coord_shape_denormalize(self, values, prefix):
        scale = float(getattr(self, f"{prefix}_scale_", 1.0))
        if not np.isfinite(scale) or scale <= float(self.variance_floor):
            scale = 1.0
        return (_safe_vector(values) * ((scale + float(self.variance_floor)) ** 2)).astype(np.float64)

    def _maybe_denormalize_final_tree_mean(self, values):
        if bool(getattr(self, "final_tree_target_standardized_", False)):
            return self._coord_denormalize(values, "tree_c")
        return _safe_vector(values).astype(np.float64)

    def _maybe_denormalize_final_tree_shape(self, values):
        if bool(getattr(self, "final_tree_target_standardized_", False)):
            return self._coord_shape_denormalize(values, "tree_c")
        return _safe_vector(values).astype(np.float64)

    def _fit_empirical_bayes_prior(self, target_l, diagonal=False):
        y = np.asarray(target_l, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("Empirical Bayes prior requires two standardized target coordinates")
        if len(y):
            sigma = (y.T @ y) / max(1.0, float(len(y)))
        else:
            sigma = np.eye(2, dtype=np.float64)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        if diagonal:
            sigma = np.diag(np.maximum(np.diag(sigma), float(self.variance_floor)))
        sigma = sigma + float(self.variance_floor) * np.eye(2, dtype=np.float64)
        self.eb_sigma_l_ = sigma.astype(np.float64)
        return self.eb_sigma_l_

    def _eb_posterior_targets(
        self,
        m_u,
        z_u,
        rho_n,
        rho_t,
        v_n_u,
        v_t_u,
        v_m_dis,
        diagonal=False,
        use_disagreement=True,
    ):
        n_u = len(_safe_vector(m_u))
        if n_u == 0:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, np.array([], dtype=np.float64), {
                "omega_mean": 0.0,
                "omega_std": 0.0,
                "omega_min": 0.0,
                "omega_max": 0.0,
                "trace_mean": 0.0,
                "shrink_l2_mean": 0.0,
                "gamma_trace_mean": 0.0,
                "gamma_corr_mean": 0.0,
                "sigma_corr": 0.0,
            }

        sigma = np.asarray(getattr(self, "eb_sigma_l_", np.eye(2)), dtype=np.float64)
        if diagonal:
            sigma = np.diag(np.maximum(np.diag(sigma), float(self.variance_floor)))

        m_u = _safe_vector(m_u).astype(np.float64)
        z_u = _safe_vector(z_u).astype(np.float64)
        rho_n = _safe_vector(rho_n).astype(np.float64)
        rho_t = _safe_vector(rho_t).astype(np.float64)
        v_n_u = np.maximum(_safe_vector(v_n_u).astype(np.float64), float(self.variance_floor))
        v_t_u = np.maximum(_safe_vector(v_t_u).astype(np.float64), float(self.variance_floor))
        v_m_dis = np.maximum(_safe_vector(v_m_dis).astype(np.float64), 0.0) if use_disagreement else np.zeros(n_u)

        r_tilde = np.column_stack([
            self._coord_normalize(m_u, "tree_a"),
            self._coord_normalize(z_u, "tree_c"),
        ]).astype(np.float64)

        s_a = float(getattr(self, "tree_a_scale_", 1.0))
        s_c = float(getattr(self, "tree_c_scale_", 1.0))
        s_a = s_a if np.isfinite(s_a) and s_a > float(self.variance_floor) else 1.0
        s_c = s_c if np.isfinite(s_c) and s_c > float(self.variance_floor) else 1.0
        s_a2 = (s_a + float(self.variance_floor)) ** 2
        s_c2 = (s_c + float(self.variance_floor)) ** 2
        sac = (s_a + float(self.variance_floor)) * (s_c + float(self.variance_floor))

        # Gamma is a local diagnostic covariance matrix for support weighting.
        # The disagreement term is a conservative inflation heuristic; it is
        # not claimed to recover the full covariance of two independent models.
        gamma00 = (rho_n**2) * v_n_u + (rho_t**2) * v_t_u + v_m_dis
        gamma01 = rho_n * (2.0 * rho_n - 1.0) * v_n_u + rho_t * (2.0 * rho_t) * v_t_u + 2.0 * v_m_dis
        gamma11 = ((2.0 * rho_n - 1.0) ** 2) * v_n_u + 4.0 * (rho_t**2) * v_t_u + 4.0 * v_m_dis

        gamma00_t = gamma00 / max(s_a2, float(self.variance_floor))
        gamma11_t = gamma11 / max(s_c2, float(self.variance_floor))
        gamma01_t = gamma01 / max(sac, float(self.variance_floor))
        if diagonal:
            gamma01_t = np.zeros_like(gamma01_t)

        y_plus = np.empty_like(r_tilde)
        omega = np.empty(n_u, dtype=np.float64)
        trace_k = np.empty(n_u, dtype=np.float64)
        gamma_trace = gamma00_t + gamma11_t
        gamma_corr = np.zeros(n_u, dtype=np.float64)
        denom_corr = np.sqrt(np.maximum(gamma00_t * gamma11_t, float(self.variance_floor)))
        gamma_corr = np.divide(gamma01_t, denom_corr, out=np.zeros_like(gamma01_t), where=denom_corr > 0.0)

        eye = np.eye(2, dtype=np.float64)
        for idx in range(n_u):
            gamma = np.array([[gamma00_t[idx], gamma01_t[idx]], [gamma01_t[idx], gamma11_t[idx]]], dtype=np.float64)
            gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
            if diagonal:
                gamma = np.diag(np.maximum(np.diag(gamma), float(self.variance_floor)))
            system = sigma + gamma + float(self.variance_floor) * eye
            try:
                k = sigma @ np.linalg.inv(system)
            except np.linalg.LinAlgError:
                k = sigma @ np.linalg.pinv(system)
            k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
            y_plus[idx] = k @ r_tilde[idx]
            trace_k[idx] = float(np.trace(k))
            omega[idx] = 0.5 * trace_k[idx]

        omega = np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
        shrink_l2 = np.linalg.norm(r_tilde - y_plus, axis=1)
        sigma_corr = 0.0
        sigma_denom = float(np.sqrt(max(sigma[0, 0] * sigma[1, 1], float(self.variance_floor))))
        if sigma_denom > 0.0:
            sigma_corr = float(sigma[0, 1] / sigma_denom)
        diagnostics = {
            "omega_mean": float(np.mean(omega)),
            "omega_std": float(np.std(omega)),
            "omega_min": float(np.min(omega)),
            "omega_max": float(np.max(omega)),
            "trace_mean": float(np.mean(trace_k)),
            "shrink_l2_mean": float(np.mean(shrink_l2)),
            "gamma_trace_mean": float(np.mean(gamma_trace)),
            "gamma_corr_mean": float(np.mean(gamma_corr)),
            "sigma_corr": float(sigma_corr),
        }
        return y_plus.astype(np.float64), omega.astype(np.float64), diagnostics

    def _predict_tree_mean(self, X):
        pred = self._vector_prediction(self.tree_model_, X)
        return self._maybe_denormalize_final_tree_mean(pred[:, 1])

    def _final_tree_mean_shape(self, X):
        pred = self._vector_prediction(self.tree_model_, X)
        return (
            self._maybe_denormalize_final_tree_mean(pred[:, 1]),
            self._maybe_denormalize_final_tree_shape(self._vector_tree_shape(self.tree_model_, X, coord=1)),
        )

    def _poe_predict_scaled(self, X):
        mu_n, n_shape = self._predict_tabm_components(X)
        tree_mu, tree_shape = self._final_tree_mean_shape(X)
        r_n = self._apply_shape_scale(n_shape, "r_n_scale_")
        r_c = self._apply_shape_scale(tree_shape, "r_c_scale_")
        v_n = np.maximum(float(getattr(self, "sigma2_n_", 1.0)) * r_n, float(self.variance_floor))
        v_c = np.maximum(float(getattr(self, "sigma2_c_", 1.0)) * r_c, float(self.variance_floor))
        prediction_mode = str(getattr(self, "final_prediction_mode_", "complement_average"))
        # The returned "var" is a posterior-style diagnostic scale used by
        # downstream audit fields. It omits explicit branch covariance and
        # should not be interpreted as calibrated predictive uncertainty.
        if prediction_mode in {"eta_norm_cwls", "eta_norm_fixed_average", "eta_norm_rho_zero"}:
            eta = float(np.clip(getattr(self, "rc_eta_hat_", 1.0), 0.0, 1.0))
            if prediction_mode == "eta_norm_fixed_average":
                rho = 0.5
            elif prediction_mode == "eta_norm_rho_zero":
                rho = 0.0
            else:
                rho = float(np.clip(getattr(self, "rc_rho_", 0.5), 0.0, 1.0))
            h = (eta * mu_n + tree_mu) / (1.0 + eta)
            v_h = ((eta**2) * v_n + v_c) / ((1.0 + eta) ** 2)
            pred = rho * mu_n + (1.0 - rho) * h
            var = (rho**2) * v_n + ((1.0 - rho) ** 2) * v_h
            weights = np.full(len(pred), rho, dtype=np.float64)
        elif prediction_mode == "rho_cwls":
            rho = float(np.clip(getattr(self, "rc_rho_", 0.5), 0.0, 1.0))
            pred = rho * mu_n + (1.0 - rho) * tree_mu
            var = (rho**2) * v_n + ((1.0 - rho) ** 2) * v_c
            weights = np.full(len(pred), rho, dtype=np.float64)
        elif prediction_mode == "residual":
            pred = mu_n + tree_mu
            var = v_n + v_c
            weights = np.full(len(pred), 0.5, dtype=np.float64)
        else:
            pred = 0.5 * mu_n + 0.5 * tree_mu
            var = 0.25 * (v_n + v_c)
            weights = np.full(len(pred), 0.5, dtype=np.float64)
        return pred.astype(np.float64), var.astype(np.float64), weights

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        start_time = time.time()
        set_seed(int(self.random_state))
        mode = str(getattr(self, "ablation_mode", self.ABLATION_MODE))
        supported_modes = {
            "full",
            "rc_eta_lite",
            "catch_no_target_calibration",
            "catch_no_eta_scale",
            "catch_no_cwls_fusion",
            "catch_no_u",
            "catch_rho0_complement",
            "no_disagreement",
            "no_variance",
        }
        if mode not in supported_modes:
            raise ValueError(f"Unsupported CATCH mode: {mode}")
        en_modes = set()
        rc_modes = {
            "rc_eta_lite",
            "catch_no_target_calibration",
            "catch_no_eta_scale",
            "catch_no_cwls_fusion",
            "catch_no_u",
            "catch_rho0_complement",
        }
        catch_ablation_modes = {
            "catch_no_target_calibration",
            "catch_no_eta_scale",
            "catch_no_cwls_fusion",
            "catch_no_u",
            "catch_rho0_complement",
        }
        raw_target_modes = {"catch_no_target_calibration"}
        no_disagreement_modes = {
            "no_disagreement",
            "rc_eta_lite",
        } | catch_ablation_modes
        no_eb_weight_modes = {
            "no_variance",
            "rc_eta_lite",
        } | catch_ablation_modes
        eb_component_modes = set()
        eb_modes = set(rc_modes)
        en_mode = mode in en_modes
        eb_mode = mode in eb_modes
        rc_mode = mode in rc_modes
        standardized_tree_mode = en_mode or eb_mode
        X_labeled = _as_2d_float(X_labeled)
        y_labeled = _safe_vector(y_labeled)
        if len(y_labeled) != len(X_labeled):
            raise ValueError("y_labeled must have the same length as X_labeled")
        X_unlabeled = (
            np.empty((0, X_labeled.shape[1]), dtype=np.float32)
            if X_unlabeled is None
            else _as_2d_float(X_unlabeled)
        )
        if mode == "catch_no_u":
            X_unlabeled = np.empty((0, X_labeled.shape[1]), dtype=np.float32)

        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        # Feature scaling may use training-side unlabeled covariates; target scaling uses labels only.
        X_fit_for_scaler = np.vstack([X_labeled, X_unlabeled]).astype(np.float32) if len(X_unlabeled) else X_labeled
        self.x_scaler_.fit(X_fit_for_scaler)
        X_l_sc = self.x_scaler_.transform(X_labeled).astype(np.float32)
        X_u_sc = self.x_scaler_.transform(X_unlabeled).astype(np.float32) if len(X_unlabeled) else X_unlabeled
        y_l_sc = self.y_scaler_.fit_transform(y_labeled.reshape(-1, 1)).reshape(-1).astype(np.float64)
        self.n_features_in_ = int(X_labeled.shape[1])
        self.final_tree_target_standardized_ = False

        self.neural_model_ = self._fit_tabm(X_l_sc, y_l_sc, offset=0)
        X_all = np.vstack([X_l_sc, X_u_sc]).astype(np.float32) if len(X_u_sc) else X_l_sc
        n_l = len(X_l_sc)
        n_all, n_shape_raw = self._predict_tabm_components(X_all)
        if standardized_tree_mode:
            self._fit_shape_scale(n_shape_raw[:n_l], "r_n_scale_")
            r_n_all = self._apply_shape_scale(n_shape_raw, "r_n_scale_")
        else:
            r_n_all = self._fit_shape_scale(n_shape_raw, "r_n_scale_")
        n_l_pred, n_u_pred = n_all[:n_l], n_all[n_l:]
        r_n_l, r_n_u = r_n_all[:n_l], r_n_all[n_l:]
        self.sigma2_n_ = (
            self._calibrate_en_variance_scale(y_l_sc - n_l_pred)
            if standardized_tree_mode
            else self._calibrate_variance_scale(y_l_sc - n_l_pred, r_n_l)
        )
        v_n_l = np.maximum(self.sigma2_n_ * r_n_l, float(self.variance_floor))
        v_n_u = np.maximum(self.sigma2_n_ * r_n_u, float(self.variance_floor))

        z_l_raw = 2.0 * y_l_sc - n_l_pred
        eta_l = np.ones(len(y_l_sc), dtype=np.float64)
        eta_hat_target = 1.0
        if rc_mode and mode not in raw_target_modes:
            # The eta target keeps the tree complement anchored to supervised residual evidence.
            z_l, eta_l, eta_hat_target = self._eb_labeled_complement_target(y_l_sc, n_l_pred, v_n_l)
        else:
            z_l = z_l_raw
        eb_diag = {
            "omega_mean": 0.0,
            "omega_std": 0.0,
            "omega_min": 0.0,
            "omega_max": 0.0,
            "trace_mean": 0.0,
            "shrink_l2_mean": 0.0,
            "gamma_trace_mean": 0.0,
            "gamma_corr_mean": 0.0,
            "sigma_corr": 0.0,
        }
        if standardized_tree_mode:
            _, _, s_a2 = self._fit_coord_scale(y_l_sc, "tree_a")
            _, _, s_c2 = self._fit_coord_scale(z_l, "tree_c")
            target_l = np.column_stack([
                self._coord_normalize(y_l_sc, "tree_a"),
                self._coord_normalize(z_l, "tree_c"),
            ]).astype(np.float32)
            if eb_mode:
                self._fit_empirical_bayes_prior(
                    target_l,
                    diagonal=False,
                )
        else:
            s_a2 = float(np.var(y_l_sc, ddof=1)) if len(y_l_sc) > 1 else float(np.var(y_l_sc))
            s_a2 = max(s_a2, float(self.variance_floor))
            s_c2 = float(np.var(z_l, ddof=1)) if len(z_l) > 1 else float(np.var(z_l))
            s_c2 = max(s_c2, float(self.variance_floor))
            target_l = np.column_stack([y_l_sc, z_l]).astype(np.float32)
        self.initial_tree_model_ = self._fit_vector_tree(X_l_sc, target_l, offset=0)
        self.tree_model_ = self.initial_tree_model_

        t0_all = self._vector_prediction(self.initial_tree_model_, X_all)
        if standardized_tree_mode:
            a_l = self._coord_denormalize(t0_all[:n_l, 0], "tree_a")
            a_u = self._coord_denormalize(t0_all[n_l:, 0], "tree_a")
        else:
            a_l, a_u = t0_all[:n_l, 0], t0_all[n_l:, 0]
        t0_shape_raw = self._vector_tree_shape(self.initial_tree_model_, X_all, coord=0)
        if standardized_tree_mode:
            t0_shape_raw = self._coord_shape_denormalize(t0_shape_raw, "tree_a")
            self._fit_shape_scale(t0_shape_raw[:n_l], "r_t_scale_")
            r_t_all = self._apply_shape_scale(t0_shape_raw, "r_t_scale_")
        else:
            r_t_all = self._fit_shape_scale(t0_shape_raw, "r_t_scale_")
        r_t_l, r_t_u = r_t_all[:n_l], r_t_all[n_l:]
        self.sigma2_t_ = (
            self._calibrate_en_variance_scale(y_l_sc - a_l)
            if standardized_tree_mode
            else self._calibrate_variance_scale(y_l_sc - a_l, r_t_l)
        )
        v_t_u = np.maximum(self.sigma2_t_ * r_t_u, float(self.variance_floor))
        eb_target_u_tilde = np.empty((0, 2), dtype=np.float64)

        if len(X_u_sc):
            # Unlabeled rows contribute target-free support through branch
            # agreement and residual-calibrated diagnostic scales.
            if mode == "no_variance":
                rho_n = np.full(len(X_u_sc), 0.5, dtype=np.float64)
                rho_t = np.full(len(X_u_sc), 0.5, dtype=np.float64)
            else:
                inv_n = 1.0 / np.maximum(v_n_u, float(self.variance_floor))
                inv_t = 1.0 / np.maximum(v_t_u, float(self.variance_floor))
                denom = np.maximum(inv_n + inv_t, float(self.variance_floor))
                rho_n = inv_n / denom
                rho_t = inv_t / denom
            m_u = rho_n * n_u_pred + rho_t * a_u
            z_u = 2.0 * m_u - n_u_pred
            if mode == "no_variance":
                v_post_m = np.zeros(len(X_u_sc), dtype=np.float64)
                v_post_z = np.zeros(len(X_u_sc), dtype=np.float64)
            else:
                v_post_m = (v_n_u * v_t_u) / np.maximum(v_n_u + v_t_u + float(self.variance_floor), float(self.variance_floor))
                v_post_z = ((2.0 * rho_n - 1.0) ** 2) * v_n_u + 4.0 * (rho_t**2) * v_t_u
            v_m_dis = rho_n * ((n_u_pred - m_u) ** 2) + rho_t * ((a_u - m_u) ** 2)
            if mode == "no_variance":
                v_m = np.zeros(len(X_u_sc), dtype=np.float64)
                v_z = np.zeros(len(X_u_sc), dtype=np.float64)
            elif mode in no_disagreement_modes:
                v_m = v_post_m
                v_z = v_post_z
            else:
                v_m = v_post_m + v_m_dis
                v_z = v_post_z + 4.0 * v_m_dis
            s_z2 = float(np.var(z_l, ddof=1)) if len(z_l) > 1 else float(np.var(z_l))
            s_z2 = max(s_z2, float(self.variance_floor))
            if mode == "no_variance":
                omega_u = np.ones(len(X_u_sc), dtype=np.float64)
                omega0_u = omega_u.copy()
                omega_a0 = omega_u.copy()
                omega_c0 = omega_u.copy()
            elif en_mode:
                omega_a0 = s_a2 / np.maximum(s_a2 + v_m + float(self.variance_floor), float(self.variance_floor))
                omega_c0 = s_c2 / np.maximum(s_c2 + v_z + float(self.variance_floor), float(self.variance_floor))
                omega0_u = 0.5 * (omega_a0 + omega_c0)
                omega0_mean = float(np.mean(omega0_u)) if len(omega0_u) else 0.0
                omega0_mean2 = float(np.mean(omega0_u**2)) if len(omega0_u) else 0.0
                omega_u = (omega0_mean / (omega0_mean2 + float(self.variance_floor))) * omega0_u
                eb_target_u_tilde = np.empty((0, 2), dtype=np.float64)
            elif eb_mode:
                eb_target_u_tilde, omega_u, eb_diag = self._eb_posterior_targets(
                    m_u,
                    z_u,
                    rho_n,
                    rho_t,
                    v_n_u,
                    v_t_u,
                    v_m_dis,
                    diagonal=False,
                    use_disagreement=(mode not in no_disagreement_modes),
                )
                if mode == "catch_no_target_calibration":
                    eb_target_u_tilde = np.column_stack([
                        self._coord_normalize(m_u, "tree_a"),
                        self._coord_normalize(z_u, "tree_c"),
                    ]).astype(np.float64)
                omega0_u = omega_u.copy()
                omega_a0 = omega_u.copy()
                omega_c0 = omega_u.copy()
            else:
                omega_a0 = np.zeros(len(X_u_sc), dtype=np.float64)
                omega_u = s_z2 / np.maximum(s_z2 + v_z + float(self.variance_floor), float(self.variance_floor))
                omega0_u = omega_u.copy()
                omega_c0 = omega_u.copy()
                eb_target_u_tilde = np.empty((0, 2), dtype=np.float64)
            X_tree = np.vstack([X_l_sc, X_u_sc]).astype(np.float32)
            w_u_base = float(len(X_l_sc)) / max(1.0, float(len(X_u_sc)))
            if mode in no_eb_weight_modes:
                w_u = np.full(len(X_u_sc), w_u_base, dtype=np.float64)
            else:
                w_u = w_u_base * np.maximum(omega_u, 0.0)
            sample_weight = np.concatenate([np.ones(len(X_l_sc), dtype=np.float64), w_u])
            omega0_sum = float(np.sum(omega0_u))
            omega0_sq_sum = float(np.sum(omega0_u**2))
            ess_u = float((omega0_sum**2) / (omega0_sq_sum + float(self.variance_floor))) if en_mode else float(len(X_u_sc))
            if eb_mode:
                ess_u = float((omega0_sum**2) / (omega0_sq_sum + float(self.variance_floor)))
        else:
            rho_n = np.array([], dtype=np.float64)
            rho_t = np.array([], dtype=np.float64)
            m_u = np.array([], dtype=np.float64)
            z_u = np.array([], dtype=np.float64)
            v_post_m = np.array([], dtype=np.float64)
            v_m = np.array([], dtype=np.float64)
            v_post_z = np.array([], dtype=np.float64)
            v_m_dis = np.array([], dtype=np.float64)
            v_z = np.array([], dtype=np.float64)
            omega_u = np.array([], dtype=np.float64)
            omega0_u = np.array([], dtype=np.float64)
            omega_a0 = np.array([], dtype=np.float64)
            omega_c0 = np.array([], dtype=np.float64)
            ess_u = 0.0
            s_z2 = float(np.var(z_l, ddof=1)) if len(z_l) > 1 else float(np.var(z_l))
            X_tree = X_l_sc
            sample_weight = np.ones(len(X_l_sc), dtype=np.float64)
            w_u = np.array([], dtype=np.float64)
            eb_target_u_tilde = np.empty((0, 2), dtype=np.float64)

        final_target_mode = "complement"
        self.final_prediction_mode_ = "complement_average"
        self.final_tree_output_dim_ = 2
        self.final_tree_target_standardized_ = bool(standardized_tree_mode)
        vector_tree_fit_count = 2
        c_l_target = z_l
        c_u_target = z_u if len(X_u_sc) else np.array([], dtype=np.float64)
        if mode == "rc_eta_lite":
            final_target_mode = "rc_eta_lite_empirical_bayes_complement"
        elif mode == "catch_no_target_calibration":
            final_target_mode = "catch_ablation_raw_complementary_target"
        elif mode == "catch_no_eta_scale":
            final_target_mode = "catch_ablation_no_eta_scale"
        elif mode == "catch_no_cwls_fusion":
            final_target_mode = "catch_ablation_no_cwls_fusion"
        elif mode == "catch_no_u":
            final_target_mode = "catch_ablation_no_unlabeled"
        elif mode == "catch_rho0_complement":
            final_target_mode = "catch_control_fixed_rho0_eta_complement"
        target_u = (
            eb_target_u_tilde.astype(np.float32)
            if eb_mode and len(X_u_sc)
            else (
                np.column_stack([m_u, c_u_target]).astype(np.float32)
                if len(X_u_sc)
                else np.empty((0, 2), dtype=np.float32)
            )
        )
        y_tree = np.vstack([target_l, target_u]).astype(np.float32) if len(X_u_sc) else target_l
        self.tree_model_ = self._fit_vector_tree(X_tree, y_tree, sample_weight=sample_weight, offset=101)
        final_all = self._vector_prediction(self.tree_model_, X_all)
        tree_all = self._coord_denormalize(final_all[:, 1], "tree_c") if standardized_tree_mode else final_all[:, 1]
        tree_target_l = c_l_target
        tree_shape_all = self._vector_tree_shape(self.tree_model_, X_all, coord=1)
        if standardized_tree_mode:
            tree_shape_all = self._coord_shape_denormalize(tree_shape_all, "tree_c")

        tree_l, tree_u = tree_all[:n_l], tree_all[n_l:]
        tree_shape_l, tree_shape_u = tree_shape_all[:n_l], tree_shape_all[n_l:]
        self.r_c_scale_ = max(
            float(np.mean(np.maximum(tree_shape_l + float(self.variance_floor), float(self.variance_floor)))),
            1.0,
        )
        r_c_all = self._apply_shape_scale(tree_shape_all, "r_c_scale_")
        r_c_l, r_c_u = r_c_all[:n_l], r_c_all[n_l:]
        self.sigma2_c_ = self._calibrate_variance_scale(tree_target_l - tree_l, r_c_l)

        rc_eta_modes = {
            "rc_eta_lite",
            "catch_no_target_calibration",
            "catch_no_cwls_fusion",
            "catch_no_u",
            "catch_rho0_complement",
        }
        rc_fixed_eta_modes = {"catch_no_cwls_fusion"}
        rc_fixed_rho_zero_modes = {"catch_rho0_complement"}
        rc_rho_modes = (rc_eta_modes - rc_fixed_eta_modes - rc_fixed_rho_zero_modes) | {"catch_no_eta_scale"}
        eta_hat_final = float(eta_hat_target if mode in rc_eta_modes else 1.0)
        eta_hat_final = float(np.clip(np.nan_to_num(eta_hat_final, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
        v_tree_l = np.maximum(self.sigma2_c_ * r_c_l, float(self.variance_floor))
        v_tree_u = np.maximum(self.sigma2_c_ * r_c_u, float(self.variance_floor)) if len(X_u_sc) else np.array([], dtype=np.float64)
        rc_rho = 0.5
        rc_rho_num = 0.0
        rc_rho_den = 0.0

        if mode in rc_rho_modes or mode in rc_fixed_eta_modes or mode in rc_fixed_rho_zero_modes:
            # Final readout is a one-parameter constrained blend, not an unrestricted stacker.
            if mode in rc_eta_modes:
                h_l = (eta_hat_final * n_l_pred + tree_l) / (1.0 + eta_hat_final)
                v_h_l = ((eta_hat_final**2) * v_n_l + v_tree_l) / ((1.0 + eta_hat_final) ** 2)
                if len(X_u_sc):
                    h_u = (eta_hat_final * n_u_pred + tree_u) / (1.0 + eta_hat_final)
                    v_h_u = ((eta_hat_final**2) * np.maximum(self.sigma2_n_ * r_n_u, float(self.variance_floor)) + v_tree_u) / (
                        (1.0 + eta_hat_final) ** 2
                    )
                else:
                    h_u = np.array([], dtype=np.float64)
                    v_h_u = np.array([], dtype=np.float64)
                if mode in rc_fixed_eta_modes:
                    self.final_prediction_mode_ = "eta_norm_fixed_average"
                elif mode in rc_fixed_rho_zero_modes:
                    self.final_prediction_mode_ = "eta_norm_rho_zero"
                else:
                    self.final_prediction_mode_ = "eta_norm_cwls"
            else:
                h_l = tree_l
                v_h_l = v_tree_l
                h_u = tree_u if len(X_u_sc) else np.array([], dtype=np.float64)
                v_h_u = v_tree_u if len(X_u_sc) else np.array([], dtype=np.float64)
                self.final_prediction_mode_ = "rho_cwls"
            if mode in rc_fixed_eta_modes:
                rc_rho, rc_rho_num, rc_rho_den = 0.5, 0.0, 0.0
            elif mode in rc_fixed_rho_zero_modes:
                rc_rho, rc_rho_num, rc_rho_den = 0.0, 0.0, 0.0
            else:
                rc_rho, rc_rho_num, rc_rho_den = self._closed_form_precision_rho(
                    y_l_sc,
                    n_l_pred,
                    h_l,
                    v_g=v_n_l,
                    v_h=v_h_l,
                )
            final_pred_l = rc_rho * n_l_pred + (1.0 - rc_rho) * h_l
            final_var_l = (rc_rho**2) * v_n_l + ((1.0 - rc_rho) ** 2) * v_h_l
        else:
            final_pred_l = 0.5 * n_l_pred + 0.5 * tree_l
            final_var_l = 0.25 * (
                v_n_l + v_tree_l
            )
        train_mse = _weighted_mse(y_l_sc, final_pred_l)
        if len(X_u_sc):
            if self.final_prediction_mode_ in {"eta_norm_cwls", "eta_norm_rho_zero"}:
                final_pred_u = rc_rho * n_u_pred + (1.0 - rc_rho) * h_u
                final_var_u = (rc_rho**2) * np.maximum(self.sigma2_n_ * r_n_u, float(self.variance_floor)) + (
                    (1.0 - rc_rho) ** 2
                ) * v_h_u
            elif self.final_prediction_mode_ == "rho_cwls":
                final_pred_u = rc_rho * n_u_pred + (1.0 - rc_rho) * tree_u
                final_var_u = (rc_rho**2) * np.maximum(self.sigma2_n_ * r_n_u, float(self.variance_floor)) + (
                    (1.0 - rc_rho) ** 2
                ) * v_tree_u
            else:
                final_pred_u = 0.5 * n_u_pred + 0.5 * tree_u
                final_var_u = 0.25 * (
                    np.maximum(self.sigma2_n_ * r_n_u, float(self.variance_floor))
                    + v_tree_u
                )
        else:
            final_pred_u = np.array([], dtype=np.float64)
            final_var_u = np.array([], dtype=np.float64)

        self.rc_eta_hat_ = float(eta_hat_final)
        self.rc_eta_l_mean_ = float(np.mean(eta_l)) if len(eta_l) else 0.0
        self.rc_eta_l_std_ = float(np.std(eta_l)) if len(eta_l) else 0.0
        self.rc_rho_ = float(rc_rho)

        metrics = {
            "paper_method_family": self.METHOD_NAME,
            # Public result metadata keeps the method name aligned with the
            # conservative manuscript wording: these are diagnostic scales and
            # a constrained scalar readout, not calibrated epistemic variances.
            "publication_expansion": (
                "Audit-Oriented Complementary Target Constrained Weighted Least Squares"
                if rc_mode
                else "Complementary Anchored Tree Consensus as a Variational Field"
            ),
            "mode": (
                "diagnostic_constrained_cwls"
                if rc_mode
                else "barycentric_variational_field_cwls"
            ),
            "ablation": str(mode),
            "component_ablation": mode.replace("catch_no_", "no-").replace("rc_", "rc-").replace("_", "-")
            if rc_mode
            else "",
            "final_target_mode": str(final_target_mode),
            "final_prediction_mode": str(self.final_prediction_mode_),
            "neural_backend": "TabM",
            "tree_backend": "CatBoostRegressor-MultiRMSE",
            "neural_device": str(self.device_),
            "labeled_count": int(len(X_l_sc)),
            "unlabeled_count": int(len(X_u_sc)),
            "neural_fit_count": 1,
            "vector_tree_fit_count": int(vector_tree_fit_count),
            "tree_output_dim": int(self.final_tree_output_dim_),
            "sigma2_n": float(self.sigma2_n_),
            "sigma2_t": float(self.sigma2_t_),
            "sigma2_c": float(self.sigma2_c_),
            "s_z2": float(s_z2),
            "s_a2": float(s_a2),
            "s_c2": float(s_c2),
            "target_coordinate_normalized": bool(standardized_tree_mode),
            "eb_omega_mean": float(eb_diag["omega_mean"]),
            "eb_omega_std": float(eb_diag["omega_std"]),
            "eb_omega_min": float(eb_diag["omega_min"]),
            "eb_omega_max": float(eb_diag["omega_max"]),
            "eb_trace_mean": float(eb_diag["trace_mean"]),
            "eb_shrink_l2_mean": float(eb_diag["shrink_l2_mean"]),
            "eb_gamma_trace_mean": float(eb_diag["gamma_trace_mean"]),
            "eb_gamma_corr_mean": float(eb_diag["gamma_corr_mean"]),
            "eb_sigma_corr": float(eb_diag["sigma_corr"]),
            "omega0_u_mean": float(np.mean(omega0_u)) if len(omega0_u) else 0.0,
            "omega0_u_std": float(np.std(omega0_u)) if len(omega0_u) else 0.0,
            "omega_a0_u_mean": float(np.mean(omega_a0)) if len(omega_a0) else 0.0,
            "omega_c0_u_mean": float(np.mean(omega_c0)) if len(omega_c0) else 0.0,
            "ess_u": float(ess_u),
            "unlabeled_weight_sum": float(np.sum(w_u)) if len(w_u) else 0.0,
            "unlabeled_weight_ratio": float(np.sum(w_u) / max(1.0, float(len(X_l_sc)))) if len(w_u) else 0.0,
            "omega_u_mean": float(np.mean(omega_u)) if len(omega_u) else 0.0,
            "omega_u_std": float(np.std(omega_u)) if len(omega_u) else 0.0,
            "omega_u_min": float(np.min(omega_u)) if len(omega_u) else 0.0,
            "omega_u_max": float(np.max(omega_u)) if len(omega_u) else 0.0,
            "rho_n_mean_unlabeled": float(np.mean(rho_n)) if len(rho_n) else 0.0,
            "rho_t_mean_unlabeled": float(np.mean(rho_t)) if len(rho_t) else 0.0,
            "v_m_mean": float(np.mean(v_m)) if len(v_m) else 0.0,
            "v_post_m_mean": float(np.mean(v_post_m)) if len(v_post_m) else 0.0,
            "v_z_mean": float(np.mean(v_z)) if len(v_z) else 0.0,
            "v_post_z_mean": float(np.mean(v_post_z)) if len(v_post_z) else 0.0,
            "v_m_dis_mean": float(np.mean(v_m_dis)) if len(v_m_dis) else 0.0,
            "m_u_std": float(np.std(m_u)) if len(m_u) else 0.0,
            "z_u_std": float(np.std(z_u)) if len(z_u) else 0.0,
            "rc_eta_hat": float(eta_hat_final),
            "rc_eta_l_mean": float(np.mean(eta_l)) if len(eta_l) else 0.0,
            "rc_eta_l_std": float(np.std(eta_l)) if len(eta_l) else 0.0,
            "rc_rho": float(rc_rho),
            "rc_rho_numerator": float(rc_rho_num),
            "rc_rho_denominator": float(rc_rho_den),
            "train_mse_scaled": float(train_mse),
            "posterior_var_labeled_mean": float(np.mean(final_var_l)) if len(final_var_l) else 0.0,
            "posterior_var_unlabeled_mean": float(np.mean(final_var_u)) if len(final_var_u) else 0.0,
            "posterior_mean_unlabeled_std": float(np.std(final_pred_u)) if len(final_pred_u) else 0.0,
        }
        for key, value in list(metrics.items()):
            if key not in {"paper_method_family", "publication_expansion", "mode"}:
                metrics[f"catch_vf_{key}"] = value
        self.catch_vf_metrics_ = metrics
        self.catch_core_metrics_ = {
            "status": "ok",
            "mode": (
                "diagnostic_constrained_cwls"
                if rc_mode
                else "barycentric_variational_field_cwls"
            ),
            "final_formula": (
                "y_hat=rho*g+(1-rho)*((eta*g+t)/(1+eta))"
                if self.final_prediction_mode_ == "eta_norm_cwls"
                else (
                    "y_hat=(eta*g+t)/(1+eta)"
                    if self.final_prediction_mode_ == "eta_norm_rho_zero"
                    else (
                    "y_hat=0.5*g+0.5*((eta*g+t)/(1+eta))"
                    if self.final_prediction_mode_ == "eta_norm_fixed_average"
                    else (
                    "y_hat=rho*g+(1-rho)*t"
                    if self.final_prediction_mode_ == "rho_cwls"
                    else "y_hat=0.5*n_theta+0.5*c_phi, T_phi=(a_phi,c_phi)"
                )
                )
                )
            ),
            "risks": {
                "catch_vf_train_mse_scaled": float(train_mse),
                "q_lambda": 0.0,
                "q_objective": float(train_mse),
            },
            "final_weights": {
                "barycentric_neural": float(rc_rho)
                if self.final_prediction_mode_ in {"eta_norm_cwls", "eta_norm_rho_zero", "rho_cwls"}
                else 0.5,
                "barycentric_tree_complement": float(1.0 - rc_rho)
                if self.final_prediction_mode_ in {"eta_norm_cwls", "eta_norm_rho_zero", "rho_cwls"}
                else 0.5,
                "eta_normalized_complement": float(eta_hat_final)
                if self.final_prediction_mode_ in {"eta_norm_cwls", "eta_norm_rho_zero", "eta_norm_fixed_average"}
                else 1.0,
            },
            "diagnostics": dict(metrics),
            "runtime_s": float(time.time() - start_time),
        }
        self.fit_time_ = float(time.time() - start_time)
        return self

    def predict(self, X):
        if not hasattr(self, "neural_model_") or not hasattr(self, "tree_model_"):
            raise RuntimeError("CATCHVFCWLSRegressor is not fitted")
        X_sc = self.x_scaler_.transform(_as_2d_float(X)).astype(np.float32)
        pred_sc, _, _ = self._poe_predict_scaled(X_sc)
        return self.y_scaler_.inverse_transform(pred_sc.reshape(-1, 1)).reshape(-1).astype(np.float32)


class CATCHNoDisagreementVarianceRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-no-disagreement-variance"
    ABLATION_MODE = "no_disagreement"


class CATCHRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH"
    ABLATION_MODE = "rc_eta_lite"


class CATCHNoTargetCalibrationRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-no-target-calibration"
    ABLATION_MODE = "catch_no_target_calibration"


class CATCHNoEtaScaleRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-no-eta-scale"
    ABLATION_MODE = "catch_no_eta_scale"


class CATCHNoCWLSFusionRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-no-CWLS-fusion"
    ABLATION_MODE = "catch_no_cwls_fusion"


class CATCHRhoZeroComplementRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-rho0-complement"
    ABLATION_MODE = "catch_rho0_complement"


class CATCHNoUnlabeledRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-no-U"
    ABLATION_MODE = "catch_no_u"


class CATCHNoSupportVarianceRegressor(CATCHVFCWLSRegressor):
    METHOD_NAME = "CATCH-no-support-variance"
    ABLATION_MODE = "no_variance"
