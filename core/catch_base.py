"""Internal latent-label neural-tree base used by CATCH.

The public CATCH estimators reuse this TabM/CatBoost training substrate for
standardized features, neural prediction, tree prediction, and diagnostic-shape
helpers. It is kept as an implementation base, not as a separate experiment.
"""

from __future__ import annotations

from contextlib import nullcontext
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from core.runtime_profile import (
    RECOMMENDED_DATALOADER_WORKERS,
    RECOMMENDED_INFERENCE_BATCH_SIZE,
    RECOMMENDED_XGB_N_JOBS,
    autocast_context,
    make_grad_scaler,
)
from core.utils import set_seed


def _as_2d_float(X):
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_vector(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_shape(values, floor=1e-6):
    arr = _safe_vector(values)
    arr = np.maximum(arr, float(floor))
    mean = float(np.mean(arr)) if len(arr) else 1.0
    if not np.isfinite(mean) or mean <= float(floor):
        mean = 1.0
    return (arr / mean).astype(np.float64)


def _weighted_mse(y_true, y_pred, weight=None):
    y = _safe_vector(y_true)
    pred = _safe_vector(y_pred)
    err2 = (y - pred) ** 2
    if weight is None:
        return float(np.mean(err2)) if len(err2) else 0.0
    w = np.maximum(_safe_vector(weight), 0.0)
    denom = float(np.sum(w))
    if denom <= 1e-12:
        return float(np.mean(err2)) if len(err2) else 0.0
    return float(np.sum(w * err2) / denom)


class _CATCHBaseRegressor(BaseEstimator, RegressorMixin):
    """Shared TabM/CatBoost base class for CATCH estimators."""

    METHOD_NAME = "CATCH-base"

    def __init__(
        self,
        random_state=42,
        beta=1.0,
        em_steps=2,
        epochs=120,
        batch_size=512,
        learning_rate=2e-3,
        weight_decay=1e-4,
        catboost_params=None,
        library_defaults=True,
        device="auto",
        n_jobs=None,
        variance_floor=1e-6,
        df_fraction_neural=0.05,
        df_fraction_tree=0.05,
        verbose=False,
    ):
        self.random_state = random_state
        self.beta = beta
        self.em_steps = em_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.catboost_params = catboost_params
        self.library_defaults = library_defaults
        self.device = device
        self.n_jobs = n_jobs
        self.variance_floor = variance_floor
        self.df_fraction_neural = df_fraction_neural
        self.df_fraction_tree = df_fraction_tree
        self.verbose = verbose

    def _resolve_device(self):
        requested = str(self.device or "auto").lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _make_tabm_model(self, input_dim):
        try:
            from tabm import TabM
        except Exception as exc:
            raise RuntimeError("CATCH requires the optional `tabm` package for the neural expert.") from exc
        return TabM.make(n_num_features=int(input_dim), d_out=1).to(self.device_)

    @staticmethod
    def _tabm_heads(raw):
        if raw.ndim == 3:
            return raw.squeeze(-1)
        if raw.ndim == 2:
            return raw
        return raw.reshape(raw.shape[0], -1)

    def _fit_tabm(self, X, y, sample_weight=None, offset=0):
        set_seed(int(self.random_state) + int(offset))
        self.device_ = self._resolve_device()
        model = self._make_tabm_model(X.shape[1])
        optimizer = optim.AdamW(model.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))
        scaler = make_grad_scaler()
        X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        y_tensor = torch.as_tensor(_safe_vector(y).astype(np.float32), dtype=torch.float32)
        if sample_weight is None:
            w_np = np.ones(len(X_tensor), dtype=np.float32)
        else:
            w_np = np.maximum(_safe_vector(sample_weight), 0.0).astype(np.float32)
        w_tensor = torch.as_tensor(w_np, dtype=torch.float32)
        n = int(len(X_tensor))
        batch_size = max(1, int(self.batch_size))
        generator = torch.Generator().manual_seed(int(self.random_state) + int(offset) * 1009)
        model.train()
        for _ in range(max(0, int(self.epochs))):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb = X_tensor.index_select(0, idx).to(self.device_, non_blocking=torch.cuda.is_available())
                yb = y_tensor.index_select(0, idx).to(self.device_, non_blocking=torch.cuda.is_available())
                wb = w_tensor.index_select(0, idx).to(self.device_, non_blocking=torch.cuda.is_available())
                optimizer.zero_grad(set_to_none=True)
                amp_context = autocast_context() if self.device_.type == "cuda" else nullcontext()
                with amp_context:
                    heads = self._tabm_heads(model(xb))
                    loss_per_sample = torch.mean((heads - yb.view(-1, 1)) ** 2, dim=1)
                    loss = torch.sum(loss_per_sample * wb) / torch.clamp(torch.sum(wb), min=1e-12)
                if self.device_.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        return model

    def _predict_tabm_components(self, X):
        X_arr = _as_2d_float(X)
        self.neural_model_.eval()
        preds = []
        vars_ = []
        batch_size = max(1, int(RECOMMENDED_INFERENCE_BATCH_SIZE))
        with torch.no_grad():
            for start in range(0, len(X_arr), batch_size):
                xb = torch.as_tensor(X_arr[start : start + batch_size], dtype=torch.float32).to(
                    self.device_, non_blocking=torch.cuda.is_available()
                )
                heads = self._tabm_heads(self.neural_model_(xb)).detach().float().cpu().numpy()
                preds.append(np.mean(heads, axis=1))
                if heads.shape[1] > 1:
                    vars_.append(np.var(heads, axis=1, ddof=1))
                else:
                    vars_.append(np.zeros(heads.shape[0], dtype=np.float64))
        mu = _safe_vector(np.concatenate(preds)) if preds else np.array([], dtype=np.float64)
        var = _safe_vector(np.concatenate(vars_)) if vars_ else np.array([], dtype=np.float64)
        return mu.astype(np.float64), var.astype(np.float64)

    def _catboost_params(self, offset=0):
        params = {
            "random_seed": int(self.random_state) + int(offset),
            "allow_writing_files": False,
            "verbose": bool(self.verbose),
        }
        if self.n_jobs is not None:
            params["thread_count"] = int(self.n_jobs)
        else:
            params["thread_count"] = int(RECOMMENDED_XGB_N_JOBS)
        params.update(dict(self.catboost_params or {}))
        params["random_seed"] = int(self.random_state) + int(offset)
        params["allow_writing_files"] = False
        params["verbose"] = bool(self.verbose)
        return params

    def _fit_catboost(self, X, y, sample_weight=None, offset=0):
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise RuntimeError("CATCH requires the optional `catboost` package for the tree expert.") from exc
        model = CatBoostRegressor(**self._catboost_params(offset=offset))
        if sample_weight is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=np.maximum(_safe_vector(sample_weight), 0.0).astype(np.float32))
        return model

    def _predict_tree_mean(self, X):
        return _safe_vector(self.tree_model_.predict(_as_2d_float(X))).astype(np.float64)

    def _predict_tree_shape(self, X, max_stages=64):
        """Return a staged-response shape diagnostic for the current tree.

        The spread across staged boosting checkpoints is not an independent
        ensemble variance. It is retained only as a nonnegative shape signal
        that CATCH calibrates with labeled residuals before weighting.
        """
        X_arr = _as_2d_float(X)
        try:
            stages = []
            tree_count = int(getattr(self.tree_model_, "tree_count_", 0) or 0)
            stride = max(1, int(np.ceil(max(tree_count, 1) / float(max_stages))))
            for idx, pred in enumerate(self.tree_model_.staged_predict(X_arr)):
                if idx % stride == 0 or idx + 1 == tree_count:
                    stages.append(_safe_vector(pred))
            if len(stages) >= 2:
                mat = np.vstack(stages)
                return np.var(mat, axis=0, ddof=1)
        except Exception:
            pass
        return np.full(len(X_arr), float(getattr(self, "sigma2_t_", 1.0)), dtype=np.float64)

    def _relative_shapes(self, X_l, X_u):
        X_all = np.vstack([X_l, X_u]).astype(np.float32) if len(X_u) else X_l
        _, n_shape_raw = self._predict_tabm_components(X_all)
        t_shape_raw = self._predict_tree_shape(X_all)
        r_n_all = _normalize_shape(n_shape_raw + float(self.variance_floor), self.variance_floor)
        r_t_all = _normalize_shape(t_shape_raw + float(self.variance_floor), self.variance_floor)
        n_l = len(X_l)
        return r_n_all[:n_l], r_n_all[n_l:], r_t_all[:n_l], r_t_all[n_l:]

    def _effective_df(self, n, fraction):
        n = int(max(1, n))
        df = float(np.clip(float(fraction) * float(n), 1.0, max(1.0, float(n - 1))))
        return df

    def _calibrate_scales(self, X_l, y_l):
        pred_n_l, _ = self._predict_tabm_components(X_l)
        pred_t_l = self._predict_tree_mean(X_l)
        r_n_l, _, r_t_l, _ = self._relative_shapes(X_l, np.empty((0, X_l.shape[1]), dtype=np.float32))
        n = int(len(y_l))
        df_n = self._effective_df(n, self.df_fraction_neural)
        df_t = self._effective_df(n, self.df_fraction_tree)
        sig_n = float(np.sum(((y_l - pred_n_l) ** 2) / np.maximum(r_n_l, self.variance_floor)) / max(1.0, n - df_n))
        sig_t = float(np.sum(((y_l - pred_t_l) ** 2) / np.maximum(r_t_l, self.variance_floor)) / max(1.0, n - df_t))
        floor = float(self.variance_floor)
        self.sigma2_n_ = max(sig_n, floor)
        self.sigma2_t_ = max(sig_t, floor)
        return {
            "sigma2_n": float(self.sigma2_n_),
            "sigma2_t": float(self.sigma2_t_),
            "df_n": float(df_n),
            "df_t": float(df_t),
            "train_mse_n": _weighted_mse(y_l, pred_n_l),
            "train_mse_t": _weighted_mse(y_l, pred_t_l),
        }

    def _posterior_on_unlabeled(self, X_u):
        if len(X_u) == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        mu_n, n_shape = self._predict_tabm_components(X_u)
        mu_t = self._predict_tree_mean(X_u)
        t_shape = self._predict_tree_shape(X_u)
        r_n = _normalize_shape(n_shape + float(self.variance_floor), self.variance_floor)
        r_t = _normalize_shape(t_shape + float(self.variance_floor), self.variance_floor)
        pi_n = 1.0 / np.maximum(float(self.sigma2_n_) * r_n, float(self.variance_floor))
        pi_t = 1.0 / np.maximum(float(self.sigma2_t_) * r_t, float(self.variance_floor))
        denom = np.maximum(pi_n + pi_t, float(self.variance_floor))
        m = (pi_n * mu_n + pi_t * mu_t) / denom
        v = 1.0 / denom
        return m.astype(np.float64), v.astype(np.float64), pi_n.astype(np.float64), pi_t.astype(np.float64)

    def _poe_predict_scaled(self, X):
        mu_n, n_shape = self._predict_tabm_components(X)
        mu_t = self._predict_tree_mean(X)
        t_shape = self._predict_tree_shape(X)
        r_n = _normalize_shape(n_shape + float(self.variance_floor), self.variance_floor)
        r_t = _normalize_shape(t_shape + float(self.variance_floor), self.variance_floor)
        pi_n = 1.0 / np.maximum(float(self.sigma2_n_) * r_n, float(self.variance_floor))
        pi_t = 1.0 / np.maximum(float(self.sigma2_t_) * r_t, float(self.variance_floor))
        denom = np.maximum(pi_n + pi_t, float(self.variance_floor))
        pred = (pi_n * mu_n + pi_t * mu_t) / denom
        var = 1.0 / denom
        weight_n = pi_n / denom
        return pred.astype(np.float64), var.astype(np.float64), weight_n.astype(np.float64)

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        start_time = time.time()
        set_seed(int(self.random_state))
        X_labeled = _as_2d_float(X_labeled)
        y_labeled = _safe_vector(y_labeled)
        if len(y_labeled) != len(X_labeled):
            raise ValueError("y_labeled must have the same length as X_labeled")
        X_unlabeled = (
            np.empty((0, X_labeled.shape[1]), dtype=np.float32)
            if X_unlabeled is None
            else _as_2d_float(X_unlabeled)
        )
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        X_fit_for_scaler = np.vstack([X_labeled, X_unlabeled]).astype(np.float32) if len(X_unlabeled) else X_labeled
        self.x_scaler_.fit(X_fit_for_scaler)
        X_l_sc = self.x_scaler_.transform(X_labeled).astype(np.float32)
        X_u_sc = self.x_scaler_.transform(X_unlabeled).astype(np.float32) if len(X_unlabeled) else X_unlabeled
        y_l_sc = self.y_scaler_.fit_transform(y_labeled.reshape(-1, 1)).reshape(-1).astype(np.float64)
        self.n_features_in_ = int(X_labeled.shape[1])

        self.neural_model_ = self._fit_tabm(X_l_sc, y_l_sc, offset=0)
        self.tree_model_ = self._fit_catboost(X_l_sc, y_l_sc, offset=0)
        scale_history = []
        objective_history = []

        for step in range(max(0, int(self.em_steps)) + 1):
            scale_stats = self._calibrate_scales(X_l_sc, y_l_sc)
            scale_history.append(scale_stats)
            if step >= int(self.em_steps) or len(X_u_sc) == 0:
                break
            m_u, v_u, _, _ = self._posterior_on_unlabeled(X_u_sc)
            r_n_l, r_n_u, r_t_l, r_t_u = self._relative_shapes(X_l_sc, X_u_sc)
            X_train = np.vstack([X_l_sc, X_u_sc]).astype(np.float32)
            y_train = np.concatenate([y_l_sc, m_u]).astype(np.float32)
            w_n = np.concatenate([1.0 / np.maximum(r_n_l, self.variance_floor), float(self.beta) / np.maximum(r_n_u, self.variance_floor)])
            w_t = np.concatenate([1.0 / np.maximum(r_t_l, self.variance_floor), float(self.beta) / np.maximum(r_t_u, self.variance_floor)])
            self.neural_model_ = self._fit_tabm(X_train, y_train, sample_weight=w_n, offset=101 + step)
            self.tree_model_ = self._fit_catboost(X_train, y_train, sample_weight=w_t, offset=201 + step)
            disagreement = _weighted_mse(np.zeros_like(m_u), self._predict_tabm_components(X_u_sc)[0] - self._predict_tree_mean(X_u_sc))
            objective_history.append(
                {
                    "step": int(step + 1),
                    "posterior_var_mean": float(np.mean(v_u)) if len(v_u) else 0.0,
                    "posterior_mean_std": float(np.std(m_u)) if len(m_u) else 0.0,
                    "expert_disagreement_mse": float(disagreement),
                    "w_n_unlabeled_mean": float(np.mean(w_n[len(y_l_sc) :])) if len(X_u_sc) else 0.0,
                    "w_t_unlabeled_mean": float(np.mean(w_t[len(y_l_sc) :])) if len(X_u_sc) else 0.0,
                }
            )
            if self.verbose:
                print(
                    f"[CATCH-base] step={step + 1} sigma2_n={self.sigma2_n_:.4g} "
                    f"sigma2_t={self.sigma2_t_:.4g} post_var={np.mean(v_u):.4g}"
                )

        final_pred_l, final_var_l, final_weight_n_l = self._poe_predict_scaled(X_l_sc)
        train_mse = _weighted_mse(y_l_sc, final_pred_l)
        final_m_u, final_v_u, _, _ = self._posterior_on_unlabeled(X_u_sc)
        catch_base_metrics = {
            "paper_method_family": self.METHOD_NAME,
            "publication_expansion": "CATCH internal latent-label neural-tree base",
            "mode": "catch_internal_latent_label_base",
            "beta": float(self.beta),
            "em_steps": int(self.em_steps),
            "neural_backend": "TabM",
            "tree_backend": "CatBoostRegressor",
            "neural_device": str(self.device_),
            "labeled_count": int(len(X_l_sc)),
            "unlabeled_count": int(len(X_u_sc)),
            "sigma2_n": float(self.sigma2_n_),
            "sigma2_t": float(self.sigma2_t_),
            "train_mse_scaled": float(train_mse),
            "posterior_var_labeled_mean": float(np.mean(final_var_l)) if len(final_var_l) else 0.0,
            "posterior_var_unlabeled_mean": float(np.mean(final_v_u)) if len(final_v_u) else 0.0,
            "posterior_mean_unlabeled_std": float(np.std(final_m_u)) if len(final_m_u) else 0.0,
            "precision_weight_n_mean_labeled": float(np.mean(final_weight_n_l)) if len(final_weight_n_l) else 0.0,
            "scale_history": ";".join(
                f"{s['sigma2_n']:.6g}/{s['sigma2_t']:.6g}" for s in scale_history
            ),
            "objective_history": ";".join(
                f"{s['step']}:{s['posterior_var_mean']:.6g}:{s['expert_disagreement_mse']:.6g}"
                for s in objective_history
            ),
        }
        for key, value in list(catch_base_metrics.items()):
            if key not in {"paper_method_family", "publication_expansion", "mode"}:
                catch_base_metrics[f"catch_base_{key}"] = value
        self.catch_base_metrics_ = catch_base_metrics
        self.catch_core_metrics_ = {
            "status": "ok",
            "mode": "catch_internal_latent_label_base",
            # Legacy "precision" names denote inverse diagnostic-scale weights.
            "final_formula": "posterior_mean=(pi_N*f_N+pi_T*f_T)/(pi_N+pi_T), pi_e=1/(sigma_e^2*r_e(x))",
            "risks": {
                "catch_base_train_mse_scaled": float(train_mse),
                "q_lambda": 0.0,
                "q_objective": float(train_mse),
            },
            "final_weights": {
                "precision_weight_n_mean_labeled": float(self.catch_base_metrics_["precision_weight_n_mean_labeled"]),
                "precision_weight_t_mean_labeled": float(1.0 - self.catch_base_metrics_["precision_weight_n_mean_labeled"]),
            },
            "diagnostics": dict(self.catch_base_metrics_),
            "runtime_s": float(time.time() - start_time),
        }
        self.fit_time_ = float(time.time() - start_time)
        return self

    def predict_distribution(self, X):
        X_sc = self.x_scaler_.transform(_as_2d_float(X)).astype(np.float32)
        pred_sc, var_sc, weight_n = self._poe_predict_scaled(X_sc)
        return {
            "mean_scaled": pred_sc,
            "var_scaled": var_sc,
            "precision_weight_n": weight_n,
            "precision_weight_t": 1.0 - weight_n,
        }

    def predict(self, X):
        if not hasattr(self, "neural_model_") or not hasattr(self, "tree_model_"):
            raise RuntimeError("CATCH base regressor is not fitted")
        dist = self.predict_distribution(X)
        return self.y_scaler_.inverse_transform(dist["mean_scaled"].reshape(-1, 1)).reshape(-1).astype(np.float32)
