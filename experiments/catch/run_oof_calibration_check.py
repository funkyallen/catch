"""Run an out-of-fold generalization check for CATCH calibration claims.

The main benchmark uses seeded train/eval splits. This auxiliary runner adds
an outer K-fold loop so every reported test row is held out from preprocessing,
semi-supervised training, and final CATCH readout fitting. It is intended as a
conservative audit table, not as a replacement for the main benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.catch.run_catch_suite import (  # noqa: E402
    DATASETS,
    DATA_DIR,
    _encode_with_fit_columns,
    evaluate_method,
    failure_row,
    load_data,
    metric_row,
    params_for_method,
)


DEFAULT_METHODS = ["CATCH", "AutoGluon", "CatBoost", "CATCH-rho0-complement"]
DEFAULT_SEEDS = [42, 123, 456]


def build_fold_payload(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int,
    labeled_ratio: float,
) -> dict[str, object]:
    """Build a payload using only outer-train rows for fitting transforms."""
    from sklearn.model_selection import train_test_split

    X_frame = X_all.reset_index(drop=True) if isinstance(X_all, pd.DataFrame) else pd.DataFrame(X_all)
    y = np.asarray(y_all, dtype=float).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=int)
    eval_idx = np.asarray(eval_idx, dtype=int)
    if len(train_idx) < 10 or len(eval_idx) == 0:
        raise ValueError("OOF folds require at least 10 train rows and one eval row")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--tag", default="oof_calibration")
    parser.add_argument("--log-dir", default=str(ROOT / "r" / "catch_oof_calibration"))
    parser.add_argument("--autogluon-time-limit", type=int, default=60)
    parser.add_argument("--tree-rounds", type=int, default=300)
    parser.add_argument("--tree-lr", type=float, default=0.05)
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--model-param-profile", choices=["project", "library-defaults"], default="project")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run() -> Path:
    from sklearn.model_selection import KFold

    args = parse_args()
    datasets = list(DATASETS if args.datasets is None else args.datasets)
    rows: list[dict[str, object]] = []
    protocol = f"{int(args.folds)}fold_labeled_ratio_{float(args.labeled_ratio):.2f}"

    for dataset in datasets:
        X, y = load_data(DATA_DIR / dataset)
        n_splits = max(2, min(int(args.folds), len(y)))
        for seed in args.seeds:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            for fold, (train_idx, eval_idx) in enumerate(splitter.split(np.arange(len(y))), start=1):
                try:
                    payload = build_fold_payload(X, y, train_idx, eval_idx, seed, args.labeled_ratio)
                except Exception as exc:
                    for method in args.methods:
                        row = failure_row("oof_calibration", protocol, dataset, method, seed, exc, {})
                        row.update({"Fold": int(fold), "NTrain": int(len(train_idx)), "NEval": int(len(eval_idx))})
                        rows.append(row)
                    continue

                for method in args.methods:
                    params = params_for_method(method, args)
                    print(
                        f"[oof_calibration] dataset={dataset} seed={seed} fold={fold}/{n_splits} method={method}",
                        flush=True,
                    )
                    try:
                        metrics = evaluate_method(method, seed + fold * 1009, payload, params, verbose=args.verbose)
                        print(
                            f"  -> ok R2={metrics['R2']:.6f} RMSE={metrics['RMSE']:.6f} Time={metrics['Time']:.2f}s",
                            flush=True,
                        )
                        row = metric_row("oof_calibration", protocol, dataset, method, seed, metrics, params)
                    except Exception as exc:
                        print(f"  -> failed {repr(exc)}", flush=True)
                        row = failure_row("oof_calibration", protocol, dataset, method, seed, exc, params)
                    row.update(
                        {
                            "Fold": int(fold),
                            "NTrain": int(len(train_idx)),
                            "NLabeled": int(payload.get("n_labeled", 0)),
                            "NUnlabeled": int(payload.get("n_unlabeled", 0)),
                            "NEval": int(payload.get("n_eval", len(eval_idx))),
                        }
                    )
                    rows.append(row)

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{args.tag}_oof_calibration.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_path)
    return out_path


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"fatal: {exc!r}", file=sys.stderr)
        raise SystemExit(1)
