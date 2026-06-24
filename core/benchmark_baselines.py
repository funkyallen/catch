from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from core.runtime_profile import (
    RECOMMENDED_DATALOADER_WORKERS,
    RECOMMENDED_INFERENCE_BATCH_SIZE,
    RECOMMENDED_XGB_N_JOBS,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class XGBoostRegressorBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int = 42, library_defaults: bool = False, **params):
        self.random_state = random_state
        self.library_defaults = bool(library_defaults)
        self.params = dict(params)
        self.model_ = None

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        del X_unlabeled
        from xgboost import XGBRegressor

        if self.library_defaults:
            params = {"random_state": self.random_state}
        else:
            params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "n_jobs": RECOMMENDED_XGB_N_JOBS,
                "random_state": self.random_state,
            }
        params.update(self.params)
        self.model_ = XGBRegressor(**params)
        self.model_.fit(X_labeled, y_labeled)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1)


class LightGBMRegressorBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int = 42, verbose: bool = False, library_defaults: bool = False, **params):
        self.random_state = random_state
        self.verbose = verbose
        self.library_defaults = bool(library_defaults)
        self.params = dict(params)
        self.model_ = None

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        del X_unlabeled
        from lightgbm import LGBMRegressor

        if self.library_defaults:
            params = {"random_state": self.random_state, "verbosity": -1 if not self.verbose else 1}
        else:
            params = {
                "objective": "regression",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": RECOMMENDED_XGB_N_JOBS,
                "verbosity": -1,
            }
        params.update(self.params)
        self.model_ = LGBMRegressor(**params)
        self.model_.fit(X_labeled, y_labeled)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1)


class CatBoostRegressorBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int = 42, verbose: bool = False, library_defaults: bool = False, **params):
        self.random_state = random_state
        self.verbose = verbose
        self.library_defaults = bool(library_defaults)
        self.params = dict(params)
        self.model_ = None

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        del X_unlabeled
        from catboost import CatBoostRegressor

        if self.library_defaults:
            params = {
                "random_seed": self.random_state,
                "allow_writing_files": False,
                "verbose": bool(self.verbose),
            }
        else:
            params = {
                "loss_function": "RMSE",
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3.0,
                "random_seed": self.random_state,
                "thread_count": RECOMMENDED_XGB_N_JOBS,
                "allow_writing_files": False,
                "verbose": bool(self.verbose),
            }
        params.update(self.params)
        self.model_ = CatBoostRegressor(**params)
        self.model_.fit(X_labeled, y_labeled)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1)


class TabMRegressorBaseline(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_state: int = 42,
        epochs: int = 120,
        batch_size: int = 512,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        verbose: bool = False,
        library_defaults: bool = False,
    ):
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        self.library_defaults = bool(library_defaults)
        self.scaler_y_ = StandardScaler()
        self.model_ = None
        self.device_ = None

    @staticmethod
    def _center_prediction(prediction):
        if prediction.ndim == 3:
            return prediction.squeeze(-1).mean(dim=1)
        if prediction.ndim == 2:
            return prediction.mean(dim=1)
        return prediction.reshape(-1)

    def _make_model(self, feature_count):
        from tabm import TabM

        if self.library_defaults:
            return TabM.make(n_num_features=int(feature_count), d_out=1)
        try:
            return TabM.make(n_num_features=int(feature_count), d_out=1, k=32, n_blocks=3, d_block=256, dropout=0.1)
        except TypeError:
            return TabM.make(n_num_features=int(feature_count), d_out=1, k=32)

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        del X_unlabeled
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split

        seed_everything(self.random_state)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_tensor = torch.as_tensor(np.asarray(X_labeled, dtype=np.float32), dtype=torch.float32)
        scaled_target = self.scaler_y_.fit_transform(np.asarray(y_labeled, dtype=float).reshape(-1, 1)).reshape(-1)
        target_tensor = torch.as_tensor(scaled_target, dtype=torch.float32)
        dataset = TensorDataset(feature_tensor, target_tensor)
        validation_size = max(1, int(round(0.15 * len(dataset)))) if len(dataset) > 8 else 0
        train_size = len(dataset) - validation_size
        if validation_size > 0:
            generator = torch.Generator().manual_seed(self.random_state)
            train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
        else:
            train_dataset, validation_dataset = dataset, None
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=RECOMMENDED_DATALOADER_WORKERS,
        )
        validation_loader = (
            DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=RECOMMENDED_DATALOADER_WORKERS,
            )
            if validation_dataset is not None
            else None
        )
        self.model_ = self._make_model(feature_tensor.shape[1]).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(int(self.epochs)):
            self.model_.train()
            for batch_features, batch_target in train_loader:
                batch_features = batch_features.to(self.device_)
                batch_target = batch_target.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self._center_prediction(self.model_(batch_features)), batch_target)
                loss.backward()
                optimizer.step()
            if validation_loader is None:
                continue
            self.model_.eval()
            losses = []
            with torch.no_grad():
                for batch_features, batch_target in validation_loader:
                    batch_features = batch_features.to(self.device_)
                    batch_target = batch_target.to(self.device_)
                    losses.append(float(loss_fn(self._center_prediction(self.model_(batch_features)), batch_target).detach().cpu()))
            validation_loss = float(np.mean(losses)) if losses else float("inf")
            if validation_loss < best_loss:
                best_loss = validation_loss
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.model_.eval()
        feature_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(feature_tensor),
            batch_size=RECOMMENDED_INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=RECOMMENDED_DATALOADER_WORKERS,
        )
        predictions = []
        with torch.no_grad():
            for (batch_features,) in loader:
                batch_prediction = self._center_prediction(self.model_(batch_features.to(self.device_))).detach().cpu().numpy()
                predictions.append(batch_prediction)
        scaled_prediction = np.concatenate(predictions).reshape(-1)
        return self.scaler_y_.inverse_transform(scaled_prediction.reshape(-1, 1)).reshape(-1)


class _NeuralTreeFusionBaseline(BaseEstimator, RegressorMixin):
    """Fit TabM and CatBoost-on-y, then fuse their predictions."""

    METHOD_NAME = "NN+Tree-Fusion"
    FUSION_MODE = "average"

    _NEURAL_KEYS = {"lr", "batch_size", "epochs", "learning_rate", "weight_decay", "patience", "library_defaults"}
    _TREE_KEYS = {
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "thread_count",
        "loss_function",
        "bootstrap_type",
        "subsample",
        "rsm",
    }

    def __init__(
        self,
        random_state: int = 42,
        verbose: bool = False,
        library_defaults: bool = False,
        neural_params: dict | None = None,
        tree_params: dict | None = None,
        alpha_clip: bool = True,
        **params,
    ):
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.library_defaults = bool(library_defaults)
        self.neural_params = dict(neural_params or {})
        self.tree_params = dict(tree_params or {})
        self.alpha_clip = bool(alpha_clip)
        self.params = dict(params)

    def _split_params(self):
        neural_params = dict(self.neural_params or {})
        tree_params = dict(self.tree_params or {})
        for key, value in dict(self.params or {}).items():
            if key == "catboost_params" and isinstance(value, dict):
                tree_params.update(value)
            elif key == "tree_iterations":
                tree_params["iterations"] = int(value)
            elif key == "tree_learning_rate":
                tree_params["learning_rate"] = float(value)
            elif key == "tree_depth":
                tree_params["depth"] = int(value)
            elif key == "n_jobs":
                tree_params["thread_count"] = int(value)
            elif key in self._NEURAL_KEYS:
                neural_params[key] = value
            elif key in self._TREE_KEYS:
                tree_params[key] = value
        if "lr" in neural_params and "learning_rate" not in neural_params:
            neural_params["learning_rate"] = neural_params.pop("lr")
        return neural_params, tree_params

    def _fusion_alpha(self, y, neural_pred, tree_pred):
        del y, neural_pred, tree_pred
        return 0.5

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        import time

        start = time.time()
        neural_params, tree_params = self._split_params()
        self.neural_model_ = TabMRegressorBaseline(
            random_state=self.random_state,
            verbose=self.verbose,
            **neural_params,
        )
        self.tree_model_ = CatBoostRegressorBaseline(
            random_state=self.random_state,
            verbose=self.verbose,
            library_defaults=self.library_defaults,
            **tree_params,
        )
        self.neural_model_.fit(X_labeled, y_labeled, X_unlabeled)
        self.tree_model_.fit(X_labeled, y_labeled)

        neural_train = np.asarray(self.neural_model_.predict(X_labeled), dtype=float).reshape(-1)
        tree_train = np.asarray(self.tree_model_.predict(X_labeled), dtype=float).reshape(-1)
        y_train = np.asarray(y_labeled, dtype=float).reshape(-1)
        alpha = float(self._fusion_alpha(y_train, neural_train, tree_train))
        if self.alpha_clip:
            alpha = float(np.clip(alpha, 0.0, 1.0))
        self.alpha_ = alpha

        fused_train = alpha * neural_train + (1.0 - alpha) * tree_train
        risk = float(np.mean((y_train - fused_train) ** 2)) if len(y_train) else 0.0
        self.catch_core_metrics_ = {
            "status": "ok",
            "mode": str(self.FUSION_MODE),
            "final_formula": f"{alpha:.6g}*f_NN + {1.0 - alpha:.6g}*f_CatBoost",
            "risks": {
                "R_N": float(np.mean((y_train - neural_train) ** 2)) if len(y_train) else 0.0,
                "R_T": float(np.mean((y_train - tree_train) ** 2)) if len(y_train) else 0.0,
                "R_teacher": risk,
            },
            "diagnostics": {
                "simple_fusion_alpha": alpha,
                "simple_fusion_mode": str(self.FUSION_MODE),
                "simple_fusion_tree": "CatBoost-on-Y",
                "simple_fusion_neural": "TabM",
            },
            "runtime_s": float(time.time() - start),
        }
        self.fusion_calibration_metrics_ = {"alpha": alpha, "mode": str(self.FUSION_MODE)}
        return self

    def predict(self, X):
        if not hasattr(self, "neural_model_") or not hasattr(self, "tree_model_"):
            raise RuntimeError(f"{self.METHOD_NAME} is not fitted")
        neural_pred = np.asarray(self.neural_model_.predict(X), dtype=float).reshape(-1)
        tree_pred = np.asarray(self.tree_model_.predict(X), dtype=float).reshape(-1)
        alpha = float(getattr(self, "alpha_", 0.5))
        return alpha * neural_pred + (1.0 - alpha) * tree_pred


class NeuralTreeAverageRegressorBaseline(_NeuralTreeFusionBaseline):
    METHOD_NAME = "NN+Tree-Avg"
    FUSION_MODE = "fixed_average"


class NeuralTreeLeastSquaresRegressorBaseline(_NeuralTreeFusionBaseline):
    METHOD_NAME = "NN+Tree-LS"
    FUSION_MODE = "closed_form_least_squares"

    def _fusion_alpha(self, y, neural_pred, tree_pred):
        d = np.asarray(neural_pred, dtype=float).reshape(-1) - np.asarray(tree_pred, dtype=float).reshape(-1)
        target = np.asarray(y, dtype=float).reshape(-1) - np.asarray(tree_pred, dtype=float).reshape(-1)
        denom = float(np.dot(d, d))
        if not np.isfinite(denom) or denom <= 1e-12:
            return 0.5
        return float(np.dot(d, target) / denom)


class AutoGluonTabularRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        time_limit: int | None = 180,
        presets: str | None = "medium_quality",
        random_state: int = 42,
        verbosity: int = 0,
        library_defaults: bool = False,
    ):
        self.time_limit = time_limit
        self.presets = presets
        self.random_state = random_state
        self.verbosity = verbosity
        self.library_defaults = bool(library_defaults)
        self.predictor_ = None
        self.path_ = None

    @staticmethod
    def _to_frame(features, target=None):
        import pandas as pd

        feature_array = np.asarray(features, dtype=float)
        frame = pd.DataFrame(feature_array, columns=[f"f{idx}" for idx in range(feature_array.shape[1])])
        if target is not None:
            frame["target"] = np.asarray(target, dtype=float).reshape(-1)
        return frame

    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        del X_unlabeled
        from autogluon.tabular import TabularPredictor

        seed_everything(self.random_state)
        run_dir = Path(tempfile.mkdtemp(prefix="autogluon_catch_"))
        shutil.rmtree(run_dir, ignore_errors=True)
        train_frame = self._to_frame(X_labeled, y_labeled)
        fit_kwargs = {"train_data": train_frame, "verbosity": self.verbosity}
        if not self.library_defaults:
            fit_kwargs.update({"presets": self.presets, "time_limit": self.time_limit})
        self.predictor_ = TabularPredictor(
            label="target",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=str(run_dir),
            verbosity=self.verbosity,
        ).fit(**fit_kwargs)
        self.path_ = str(run_dir)
        return self

    def predict(self, X):
        prediction = self.predictor_.predict(self._to_frame(X))
        values = np.asarray(prediction, dtype=float).reshape(-1)
        if self.path_:
            shutil.rmtree(self.path_, ignore_errors=True)
            self.path_ = None
        return values
