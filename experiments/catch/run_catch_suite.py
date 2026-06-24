"""Minimal CATCH experiment runner.

This file is the public, code-and-data-only entry point for reproducing the
CATCH paper experiments. It intentionally contains only CATCH, the paper-facing
baselines, and the small compatibility aliases needed by the analysis scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MAIN_MANIFEST = ROOT / "reproducibility" / "datasets_manifest.csv"
EXTERNAL_VALIDATION20_MANIFEST = DATA_DIR / "external_validation20" / "manifest.csv"
OPENML50_MANIFEST = DATA_DIR / "openml50_benchmark" / "manifest.csv"

SEEDS_10 = [42, 123, 456, 789, 1011, 2027, 3141, 2718, 1618, 9001]

MAIN_METHODS = [
    "CATCH",
    "AutoGluon",
    "TabM",
    "CatBoost",
    "XGBoost",
    "LightGBM",
    "LapBoost",
    "VIME",
    "COREG",
    "RankUp",
    "UCVME",
]
LABEL_EFFICIENCY_METHODS = ["NN-only", "NN+Tree-Avg", "CatBoost", "AutoGluon", "CATCH"]
UNLABELED_CONTAMINATION_METHODS = ["NN-only", "CatBoost", "AutoGluon", "CATCH"]
EXTERNAL_VALIDATION_METHODS = [
    "CATCH",
    "AutoGluon",
    "TabM",
    "CatBoost",
    "XGBoost",
    "LightGBM",
    "LapBoost",
    "VIME",
    "COREG",
    "RankUp",
    "UCVME",
]
OPENML50_METHODS = ["CATCH", "AutoGluon"]
CATCH_ABLATION_METHODS = [
    "CATCH",
    "CATCH-no-target-calibration",
    "CATCH-no-eta-scale",
    "CATCH-no-CWLS-fusion",
    "CATCH-rho0-complement",
    "CATCH-no-U",
    "CATCH-no-disagreement-variance",
    "CATCH-no-support-variance",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _main_manifest() -> pd.DataFrame:
    return _read_csv(MAIN_MANIFEST)


def _external_validation_manifest() -> pd.DataFrame:
    return _read_csv(EXTERNAL_VALIDATION20_MANIFEST)


def _openml50_manifest() -> pd.DataFrame:
    return _read_csv(OPENML50_MANIFEST)


def _manifest_dataset_list(cohort: str = "main_30") -> list[str]:
    manifest = _main_manifest()
    if manifest.empty:
        return []
    rows = manifest[manifest["Cohort"].astype(str).eq(cohort)]
    return rows["Dataset"].astype(str).tolist()


def _target_map() -> dict[str, str]:
    manifest = _main_manifest()
    target_map: dict[str, str] = {}
    if not manifest.empty:
        target_map.update(dict(zip(manifest["Dataset"].astype(str), manifest["Target"].astype(str))))

    external = _external_validation_manifest()
    if not external.empty and {"relative_path", "openml_target"}.issubset(external.columns):
        for row in external.itertuples(index=False):
            rel = str(row.relative_path).replace("\\", "/")
            target = str(row.openml_target)
            target_map[rel] = target
            target_map[Path(rel).name] = target
    openml50 = _openml50_manifest()
    if not openml50.empty and {"relative_path", "openml_target"}.issubset(openml50.columns):
        for row in openml50.itertuples(index=False):
            rel = str(row.relative_path).replace("\\", "/")
            target = str(getattr(row, "runner_target", row.openml_target))
            target_map[rel] = target
            target_map[Path(rel).name] = target
    return target_map


def _external_validation_dataset_list() -> list[str]:
    manifest = _external_validation_manifest()
    if manifest.empty:
        return []
    return manifest["relative_path"].astype(str).tolist()


def _openml50_dataset_list() -> list[str]:
    manifest = _openml50_manifest()
    if manifest.empty:
        return []
    return manifest["relative_path"].astype(str).tolist()


DATASETS = _manifest_dataset_list("main_30")
EXTERNAL_VALIDATION20_DATASETS = _external_validation_dataset_list()
OPENML50_DATASETS = _openml50_dataset_list()
TARGET_BY_DATASET = _target_map()


METHOD_ALIASES = {
    "MLP": "NN-only",
    "MLP-only": "NN-only",
    "NN-only": "NN-only",
    "Tree-on-Y": "CatBoost",
    "CatBoost-on-Y": "CatBoost",
    "CATCH_VF_CWLS": "CATCH-VF-CWLS",
    "TabPFN-V3": "TabPFN-v3",
    "TabPFNv3": "TabPFN-v3",
    "tabpfn-v3": "TabPFN-v3",
    "tabpfn_v3": "TabPFN-v3",
}


def canonical_method_name(method_name: object) -> str:
    return METHOD_ALIASES.get(str(method_name), str(method_name))


def split_lanes(items: Iterable[str], lane_count: int = 4) -> list[list[str]]:
    lane_count = max(1, int(lane_count))
    lanes = [[] for _ in range(lane_count)]
    for idx, item in enumerate(items):
        lanes[idx % lane_count].append(str(item))
    return lanes


def set_seed(seed: int = 42) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def release_cuda_cache_if_needed() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def default_datasets() -> list[str]:
    return list(DATASETS)


def _resolve_dataset_path(dataset: str | Path) -> Path:
    path = Path(dataset)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return DATA_DIR / path


def _target_for_path(path: Path, frame: pd.DataFrame) -> str:
    rel = path
    try:
        rel = path.relative_to(DATA_DIR)
    except ValueError:
        pass
    lookup_keys = [str(rel).replace("\\", "/"), path.name]
    for key in lookup_keys:
        if key in TARGET_BY_DATASET and TARGET_BY_DATASET[key] in frame.columns:
            return TARGET_BY_DATASET[key]
    if "target" in frame.columns:
        return "target"
    numeric_cols = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    if numeric_cols:
        return numeric_cols[-1]
    return str(frame.columns[-1])


def load_data(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    dataset_path = _resolve_dataset_path(path)
    frame = pd.read_csv(dataset_path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    target = _target_for_path(dataset_path, frame)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not found in {dataset_path}")
    y = pd.to_numeric(frame[target], errors="coerce")
    keep = y.notna()
    X = frame.loc[keep].drop(columns=[target]).reset_index(drop=True)
    y_arr = y.loc[keep].astype(float).to_numpy()
    return X, y_arr


def _encode_with_fit_columns(
    X_labeled: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
    X_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    fit_frame = pd.concat([X_labeled, X_unlabeled], axis=0, ignore_index=True)
    numeric_cols = list(fit_frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_cols = [col for col in fit_frame.columns if col not in numeric_cols]

    medians = fit_frame[numeric_cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True) if numeric_cols else pd.Series(dtype=float)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if numeric_cols:
            numeric = frame.reindex(columns=numeric_cols).apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(medians).fillna(0.0)
            parts.append(numeric.astype(float))
        if categorical_cols:
            cats = frame.reindex(columns=categorical_cols).fillna("__missing__").astype(str)
            parts.append(pd.get_dummies(cats, dummy_na=False, dtype=float))
        if not parts:
            return pd.DataFrame(index=frame.index)
        return pd.concat(parts, axis=1)

    fit_encoded = transform(fit_frame)
    feature_columns = list(fit_encoded.columns)

    def align(frame: pd.DataFrame) -> pd.DataFrame:
        # Evaluation rows are only aligned to train-side columns; they do not fit encoders.
        return transform(frame).reindex(columns=feature_columns, fill_value=0.0)

    X_l = align(X_labeled).to_numpy(dtype=float)
    X_u = align(X_unlabeled).to_numpy(dtype=float)
    X_e = align(X_eval).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_l, X_u]) if len(X_u) else X_l)
    return scaler.transform(X_l), scaler.transform(X_u), scaler.transform(X_e)


def build_payload_for_protocol(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    seed: int,
    labeled_ratio: float = 0.20,
    test_size: float = 0.50,
    unlabeled_contamination: float = 0.0,
) -> dict[str, object]:
    X_frame = X_all.reset_index(drop=True) if isinstance(X_all, pd.DataFrame) else pd.DataFrame(X_all)
    y = np.asarray(y_all, dtype=float).reshape(-1)
    if len(X_frame) != len(y):
        raise ValueError("X and y have different lengths")
    if len(y) < 20:
        raise ValueError("At least 20 rows are required for the default split")

    from sklearn.model_selection import train_test_split

    # Split first; all preprocessing fits happen only on labeled plus training-side unlabeled rows.
    train_idx, eval_idx = train_test_split(
        np.arange(len(y)),
        test_size=float(test_size),
        random_state=int(seed),
        shuffle=True,
    )
    labeled_count = int(round(float(labeled_ratio) * len(train_idx)))
    labeled_count = max(8, min(labeled_count, len(train_idx) - 1))
    labeled_idx, unlabeled_idx = train_test_split(
        train_idx,
        train_size=labeled_count,
        random_state=int(seed) + 17,
        shuffle=True,
    )

    X_labeled = X_frame.iloc[labeled_idx].reset_index(drop=True)
    X_unlabeled = X_frame.iloc[unlabeled_idx].reset_index(drop=True)
    X_eval = X_frame.iloc[eval_idx].reset_index(drop=True)
    X_l_proc, X_u_proc, X_e_proc = _encode_with_fit_columns(X_labeled, X_unlabeled, X_eval)

    contamination = float(unlabeled_contamination)
    if contamination > 0.0 and len(X_u_proc):
        rng = np.random.RandomState(int(seed) + 2026)
        mask = rng.rand(len(X_u_proc)) < min(max(contamination, 0.0), 1.0)
        if bool(mask.any()):
            noise = rng.normal(size=X_u_proc[mask].shape)
            X_u_proc[mask] = noise

    return {
        "X_labeled_proc": X_l_proc.astype(np.float32),
        "X_unlabeled_proc": X_u_proc.astype(np.float32),
        "X_eval_proc": X_e_proc.astype(np.float32),
        "y_labeled": y[labeled_idx].astype(float),
        "y_eval": y[eval_idx].astype(float),
        "input_dim": int(X_l_proc.shape[1]),
        "n_labeled": int(len(labeled_idx)),
        "n_unlabeled": int(len(unlabeled_idx)),
        "n_eval": int(len(eval_idx)),
    }


def apply_tree_round_profile(method_name, params, rounds=None, learning_rate=None, max_depth=6):
    method = canonical_method_name(method_name)
    out = dict(params or {})
    if rounds is None and learning_rate is None and max_depth is None:
        return out
    if method == "XGBoost":
        if rounds is not None:
            out["n_estimators"] = int(rounds)
        if learning_rate is not None:
            out["learning_rate"] = float(learning_rate)
        if max_depth is not None:
            out["max_depth"] = int(max_depth)
    elif method == "LightGBM":
        if rounds is not None:
            out["n_estimators"] = int(rounds)
        if learning_rate is not None:
            out["learning_rate"] = float(learning_rate)
        if max_depth is not None:
            out["max_depth"] = int(max_depth)
    elif method in {"CatBoost", "NN+Tree-Avg", "NN+Tree-LS"}:
        if rounds is not None:
            out["iterations"] = int(rounds)
        if learning_rate is not None:
            out["learning_rate"] = float(learning_rate)
        if max_depth is not None:
            out["depth"] = int(max_depth)
    elif method == "LapBoost":
        if rounds is not None:
            out["max_iter"] = int(rounds)
        if learning_rate is not None:
            out["learning_rate"] = float(learning_rate)
    elif method.startswith("CATCH"):
        cat_params = dict(out.get("catboost_params") or {})
        if rounds is not None:
            cat_params["iterations"] = int(rounds)
        if learning_rate is not None:
            cat_params["learning_rate"] = float(learning_rate)
        if max_depth is not None:
            cat_params["depth"] = int(max_depth)
        out["catboost_params"] = cat_params
    return out


def _filter_params(params: dict, allowed: set[str]) -> dict:
    return {key: value for key, value in dict(params or {}).items() if key in allowed}


def build_catch_method(method_name, seed, input_dim, params=None, verbose=False):
    del input_dim
    method = canonical_method_name(method_name)
    params = dict(params or {})

    if method in {"NN-only", "TabM"}:
        from core.benchmark_baselines import TabMRegressorBaseline

        if "lr" in params and "learning_rate" not in params:
            params["learning_rate"] = params.pop("lr")
        allowed = {"epochs", "batch_size", "learning_rate", "weight_decay", "patience", "library_defaults"}
        return TabMRegressorBaseline(random_state=seed, verbose=verbose, **_filter_params(params, allowed)), "semi", False

    if method == "CatBoost":
        from core.benchmark_baselines import CatBoostRegressorBaseline

        return CatBoostRegressorBaseline(random_state=seed, verbose=verbose, **params), "supervised", False
    if method == "LightGBM":
        from core.benchmark_baselines import LightGBMRegressorBaseline

        return LightGBMRegressorBaseline(random_state=seed, verbose=verbose, **params), "supervised", False
    if method == "XGBoost":
        from core.benchmark_baselines import XGBoostRegressorBaseline

        return XGBoostRegressorBaseline(random_state=seed, **params), "supervised", False
    if method == "AutoGluon":
        from core.benchmark_baselines import AutoGluonTabularRegressor

        allowed = {"time_limit", "presets", "verbosity", "library_defaults"}
        return AutoGluonTabularRegressor(random_state=seed, **_filter_params(params, allowed)), "supervised", False
    if method == "NN+Tree-Avg":
        from core.benchmark_baselines import NeuralTreeAverageRegressorBaseline

        return NeuralTreeAverageRegressorBaseline(random_state=seed, verbose=verbose, **params), "semi", False
    if method == "NN+Tree-LS":
        from core.benchmark_baselines import NeuralTreeLeastSquaresRegressorBaseline

        return NeuralTreeLeastSquaresRegressorBaseline(random_state=seed, verbose=verbose, **params), "semi", False
    if method in {"LapBoost", "VIME", "COREG", "RankUp", "UCVME"}:
        from core.semi_supervised_baselines import (
            COREGRegressorBaseline,
            LapBoostRegressorBaseline,
            RankUpRegressorBaseline,
            UCVMERegressorBaseline,
            VIMERegressorBaseline,
        )

        ssl_builders = {
            "LapBoost": LapBoostRegressorBaseline,
            "VIME": VIMERegressorBaseline,
            "COREG": COREGRegressorBaseline,
            "RankUp": RankUpRegressorBaseline,
            "UCVME": UCVMERegressorBaseline,
        }
        return ssl_builders[method](random_state=seed, **params), "semi", False
    if method == "TabPFN-v3":
        raise ImportError("TabPFN-v3 is optional and is not bundled in this slim CATCH package.")

    from core.catch_vf_cwls import (
        CATCHNoCWLSFusionRegressor,
        CATCHNoDisagreementVarianceRegressor,
        CATCHNoEtaScaleRegressor,
        CATCHNoSupportVarianceRegressor,
        CATCHNoTargetCalibrationRegressor,
        CATCHNoUnlabeledRegressor,
        CATCHRegressor,
        CATCHRhoZeroComplementRegressor,
        CATCHVFCWLSRegressor,
    )

    builders = {
        "CATCH": CATCHRegressor,
        "CATCH-VF-CWLS": CATCHVFCWLSRegressor,
        "CATCH-no-target-calibration": CATCHNoTargetCalibrationRegressor,
        "CATCH-no-eta-scale": CATCHNoEtaScaleRegressor,
        "CATCH-no-CWLS-fusion": CATCHNoCWLSFusionRegressor,
        "CATCH-rho0-complement": CATCHRhoZeroComplementRegressor,
        "CATCH-no-U": CATCHNoUnlabeledRegressor,
        "CATCH-no-disagreement-variance": CATCHNoDisagreementVarianceRegressor,
        "CATCH-no-support-variance": CATCHNoSupportVarianceRegressor,
    }
    if method in builders:
        return builders[method](random_state=seed, verbose=verbose, **params), "semi", False
    raise ValueError(f"Unknown method: {method_name}")


ROLE_BY_METHOD = {
    "NN-only": "external_neural_baseline",
    "TabM": "external_neural_baseline",
    "CatBoost": "external_tree_baseline",
    "LightGBM": "external_tree_baseline",
    "XGBoost": "external_tree_baseline",
    "AutoGluon": "external_automl_baseline",
    "LapBoost": "external_ssl_regression_baseline",
    "VIME": "external_ssl_regression_baseline",
    "COREG": "external_ssl_regression_baseline",
    "RankUp": "external_ssl_regression_baseline",
    "UCVME": "external_ssl_regression_baseline",
    "TabPFN-v3": "external_foundation_model_baseline",
    "NN+Tree-Avg": "external_simple_neural_tree_fusion_baseline",
    "NN+Tree-LS": "external_simple_neural_tree_fusion_baseline",
    "CATCH": "proposed_catch_complementary_adaptive_target_calibration",
    "CATCH-VF-CWLS": "proposed_catch_variational_field_complementary_barycentric_cwls",
    "CATCH-no-target-calibration": "internal_mechanism_check_catch_no_target_calibration",
    "CATCH-no-eta-scale": "internal_mechanism_check_catch_no_eta_scale",
    "CATCH-no-CWLS-fusion": "internal_mechanism_check_catch_no_cwls_fusion",
    "CATCH-rho0-complement": "internal_mechanism_check_catch_fixed_rho0_eta_complement",
    "CATCH-no-U": "internal_mechanism_check_catch_no_u",
    "CATCH-no-disagreement-variance": "internal_mechanism_check_catch_no_disagreement_variance",
    "CATCH-no-support-variance": "internal_mechanism_check_catch_no_support_variance",
}


def method_role(method_name) -> str:
    method = canonical_method_name(method_name)
    return ROLE_BY_METHOD.get(method, "paper_method")


def experiment_methods(experiment, args) -> list[str]:
    override = getattr(args, "methods", None)
    if override:
        return [str(item) for item in override]
    mapping = {
        "main_benchmark": MAIN_METHODS,
        "runtime_pareto": MAIN_METHODS,
        "label_ratio": LABEL_EFFICIENCY_METHODS,
        "unlabeled_contamination": UNLABELED_CONTAMINATION_METHODS,
        "catch_ablation": CATCH_ABLATION_METHODS,
        "external_validation": EXTERNAL_VALIDATION_METHODS,
        "openml50_benchmark": OPENML50_METHODS,
    }
    return list(mapping.get(str(experiment), []))


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize_model_diagnostics(model) -> str:
    for attr in ["catch_core_metrics_", "fusion_calibration_metrics_", "pseudo_response_metrics_"]:
        if hasattr(model, attr):
            value = getattr(model, attr)
            try:
                return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default)
            except Exception:
                return json.dumps({"repr": repr(value)}, ensure_ascii=False)
    return ""


def params_for_method(method_name: str, args: argparse.Namespace) -> dict:
    method = canonical_method_name(method_name)
    params: dict[str, object] = {}
    if getattr(args, "model_param_profile", "project") == "library-defaults":
        params["library_defaults"] = True
    if method == "AutoGluon":
        params.update({"time_limit": int(args.autogluon_time_limit), "presets": "medium_quality"})
    params = apply_tree_round_profile(
        method,
        params,
        rounds=args.tree_rounds,
        learning_rate=args.tree_lr,
        max_depth=args.tree_depth,
    )
    if getattr(args, "smoke", False):
        if method.startswith("CATCH"):
            params["epochs"] = 2
            cat_params = dict(params.get("catboost_params") or {})
            cat_params["iterations"] = min(int(cat_params.get("iterations", 4)), 4)
            params["catboost_params"] = cat_params
        elif method in {"NN-only", "TabM", "NN+Tree-Avg", "NN+Tree-LS", "VIME", "RankUp", "UCVME"}:
            params["epochs"] = 2
            if "iterations" in params:
                params["iterations"] = min(int(params["iterations"]), 4)
        elif method == "LapBoost":
            params["max_iter"] = min(int(params.get("max_iter", 4)), 4)
        elif method == "COREG":
            params["max_iter"] = min(int(params.get("max_iter", 2)), 2)
    return params


def evaluate_method(method_name: str, seed: int, payload: dict[str, object], params: dict, verbose: bool = False) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    method = canonical_method_name(method_name)
    model, mode, use_scaled_y = build_catch_method(method, seed, int(payload["input_dim"]), params=params, verbose=verbose)
    y_labeled = np.asarray(payload["y_labeled"], dtype=float).reshape(-1)
    y_eval = np.asarray(payload["y_eval"], dtype=float).reshape(-1)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_labeled.reshape(-1, 1)).reshape(-1) if use_scaled_y else y_labeled
    start = time.time()
    if mode == "supervised":
        model.fit(payload["X_labeled_proc"], y_train)
    else:
        model.fit(payload["X_labeled_proc"], y_train, payload["X_unlabeled_proc"])
    elapsed = float(time.time() - start)
    pred = np.asarray(model.predict(payload["X_eval_proc"]), dtype=float).reshape(-1)
    if use_scaled_y:
        pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_eval, pred)))
    diagnostics = serialize_model_diagnostics(model)
    release_cuda_cache_if_needed()
    return {
        "Status": "ok",
        "Error": "",
        "R2": float(r2_score(y_eval, pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_eval, pred)),
        "Time": elapsed,
        "CatchCoreMetrics": diagnostics,
    }


def metric_row(experiment, protocol, dataset, method, seed, metrics, params) -> dict:
    canonical = canonical_method_name(method)
    row = {
        "Experiment": str(experiment),
        "Protocol": str(protocol),
        "Dataset": str(dataset),
        "Method": canonical,
        "MethodRole": method_role(canonical),
        "Seed": int(seed),
        "ParamsJSON": json.dumps(params or {}, sort_keys=True, default=_json_default),
    }
    row.update(metrics)
    return row


def failure_row(experiment, protocol, dataset, method, seed, error, params) -> dict:
    return metric_row(
        experiment,
        protocol,
        dataset,
        method,
        seed,
        {
            "Status": "failed",
            "Error": repr(error),
            "R2": math.nan,
            "RMSE": math.nan,
            "MAE": math.nan,
            "Time": math.nan,
            "CatchCoreMetrics": "",
        },
        params,
    )


def protocols_for_experiment(experiment: str, args: argparse.Namespace) -> list[tuple[str, float, float]]:
    if experiment == "label_ratio":
        return [(f"labeled_ratio_{float(v):.2f}", float(v), 0.0) for v in args.labeled_ratios]
    if experiment == "unlabeled_contamination":
        return [(f"unlabeled_contamination_{float(v):.2f}", 0.20, float(v)) for v in args.unlabeled_contamination_levels]
    return [("default", 0.20, 0.0)]


def datasets_for_experiment(experiment: str, args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return [str(item) for item in args.datasets]
    if experiment == "external_validation":
        return list(EXTERNAL_VALIDATION20_DATASETS)
    if experiment == "openml50_benchmark":
        return list(OPENML50_DATASETS)
    return list(DATASETS)


def run_dataset_audit(args: argparse.Namespace, datasets: list[str]) -> Path:
    rows = []
    for dataset in datasets:
        X, y = load_data(DATA_DIR / dataset)
        frame = X.copy()
        rows.append(
            {
                "Dataset": dataset,
                "Rows": int(len(y)),
                "Features": int(frame.shape[1]),
                "NumericalFeatures": int(len(frame.select_dtypes(include=[np.number, "bool"]).columns)),
                "CategoricalFeatures": int(frame.shape[1] - len(frame.select_dtypes(include=[np.number, "bool"]).columns)),
                "MissingRatio": float(frame.isna().mean().mean()) if frame.size else 0.0,
            }
        )
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dataset_audit_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run_seed_experiment(experiment: str, args: argparse.Namespace, datasets: list[str]) -> Path:
    rows: list[dict] = []
    methods = experiment_methods(experiment, args)
    for dataset in datasets:
        X, y = load_data(DATA_DIR / dataset)
        for protocol, labeled_ratio, contamination in protocols_for_experiment(experiment, args):
            for seed in [int(value) for value in args.seeds]:
                try:
                    payload = build_payload_for_protocol(
                        X,
                        y,
                        seed,
                        labeled_ratio=labeled_ratio,
                        unlabeled_contamination=contamination,
                    )
                except Exception as exc:
                    for method in methods:
                        rows.append(failure_row(experiment, protocol, dataset, method, seed, exc, {}))
                    continue
                for method in methods:
                    params = params_for_method(method, args)
                    print(f"[{experiment}] dataset={dataset} protocol={protocol} seed={seed} method={method}", flush=True)
                    try:
                        metrics = evaluate_method(method, seed, payload, params, verbose=args.verbose)
                        print(f"  -> ok R2={metrics['R2']:.6f} RMSE={metrics['RMSE']:.6f} Time={metrics['Time']:.2f}s", flush=True)
                        rows.append(metric_row(experiment, protocol, dataset, method, seed, metrics, params))
                    except Exception as exc:
                        print(f"  -> failed {repr(exc)}", flush=True)
                        rows.append(failure_row(experiment, protocol, dataset, method, seed, exc, params))
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{args.tag}_{experiment}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["main_benchmark", "dataset_audit"],
        choices=[
            "main_benchmark",
            "dataset_audit",
            "label_ratio",
            "unlabeled_contamination",
            "catch_ablation",
            "runtime_pareto",
            "external_validation",
            "openml50_benchmark",
        ],
    )
    parser.add_argument("--profile", default="full")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_10)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--tag", default="catch_run")
    parser.add_argument("--log-dir", default=str(ROOT / "r" / "catch_publication"))
    parser.add_argument("--autogluon-time-limit", type=int, default=60)
    parser.add_argument("--tree-rounds", type=int, default=300)
    parser.add_argument("--tree-lr", type=float, default=0.05)
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--labeled-ratios", nargs="+", type=float, default=[0.20, 0.40, 0.60, 0.80, 1.00])
    parser.add_argument("--unlabeled-contamination-levels", nargs="+", type=float, default=[0.00, 0.10, 0.20, 0.30, 0.50, 0.70])
    parser.add_argument("--model-param-profile", choices=["project", "library-defaults"], default="project")
    parser.add_argument("--include-tabpfn", action="store_true")
    parser.add_argument("--tabpfn-model-path", default=None)
    parser.add_argument("--tabpfn-device", default="cuda")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written: list[Path] = []
    for experiment in args.experiments:
        datasets = datasets_for_experiment(experiment, args)
        if args.smoke:
            datasets = datasets[:1]
            args.seeds = args.seeds[:1]
        if experiment == "dataset_audit":
            written.append(run_dataset_audit(args, datasets))
        else:
            written.append(run_seed_experiment(experiment, args, datasets))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
