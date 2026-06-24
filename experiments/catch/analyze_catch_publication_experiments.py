"""Analyze seed-level outputs from the slim CATCH reproduction package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

try:  # Optional in very small environments.
    from scipy.stats import friedmanchisquare, wilcoxon
except Exception:  # pragma: no cover
    friedmanchisquare = None
    wilcoxon = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOTS = [
    ROOT / "r" / "catch_publication",
    ROOT / "r" / "catch_publication_ablation",
    ROOT / "r" / "catch_external_validation",
]
DEFAULT_OUT_DIR = ROOT / "paper" / "tables" / "catch_publication_experiments"

PUBLIC_MAIN_METHODS = [
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
SIMPLE_FUSION_METHODS = ["NN-only", "Tree-on-Y", "NN+Tree-Avg", "NN+Tree-LS", "CATCH"]
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
CATCH_ABLATION_METHODS = [
    "CATCH",
    "CATCH-no-target-calibration",
    "CATCH-no-eta-scale",
    "CATCH-no-CWLS-fusion",
    "CATCH-no-U",
    "CATCH-no-disagreement-variance",
    "CATCH-no-support-variance",
]

METHOD_DISPLAY = {
    "MLP": "NN-only",
    "MLP-only": "NN-only",
    "NN-only": "NN-only",
    "TabM": "TabM",
    "CatBoost-on-Y": "Tree-on-Y",
    "Tree-on-Y": "Tree-on-Y",
    "TabPFN-V3": "TabPFN-v3",
    "TabPFNv3": "TabPFN-v3",
    "tabpfn-v3": "TabPFN-v3",
    "tabpfn_v3": "TabPFN-v3",
}


def canonical_method_name(method: object) -> str:
    return METHOD_DISPLAY.get(str(method), str(method))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="*", default=[str(path) for path in DEFAULT_RUN_ROOTS])
    parser.add_argument("--seed-files", nargs="*", default=None)
    parser.add_argument("--audit-files", nargs="*", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--reference", default="CATCH")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260616)
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def seed_files_under(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(run_root.glob("**/seed_*.csv"), key=lambda path: path.stat().st_mtime)


def audit_files_under(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(run_root.glob("**/dataset_audit_*.csv"), key=lambda path: path.stat().st_mtime)


def _json_flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            _json_flatten(next_prefix, child, out)
    elif isinstance(value, (list, tuple)):
        return
    else:
        out[prefix] = value


def expand_metric_json(df: pd.DataFrame) -> pd.DataFrame:
    if "CatchCoreMetrics" not in df.columns:
        return df
    additions = []
    for raw in df["CatchCoreMetrics"].fillna(""):
        row: dict[str, object] = {}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            diagnostics = parsed.get("diagnostics", parsed)
            risks = parsed.get("risks", {})
            if isinstance(diagnostics, dict):
                _json_flatten("", diagnostics, row)
            if isinstance(risks, dict):
                _json_flatten("risk", risks, row)
            if "final_formula" in parsed:
                row.setdefault("final_formula", parsed.get("final_formula"))
        additions.append(row)
    expanded = pd.DataFrame(additions)
    for col in expanded.columns:
        if col not in df.columns:
            df[col] = expanded[col]
    return df


def read_seed_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    df["Method"] = df["Method"].map(canonical_method_name)
    df["SourceFile"] = _relative(path)
    df["SourceMTime"] = float(path.stat().st_mtime)
    for col in ["R2", "RMSE", "MAE", "Time", "Seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return expand_metric_json(df)


def load_seed_rows(seed_files: Iterable[Path]) -> pd.DataFrame:
    frames = [read_seed_file(path) for path in seed_files if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("No seed CSV files found.")
    seed = pd.concat(frames, ignore_index=True, sort=False)
    seed = seed.sort_values(["SourceMTime", "SourceFile"]).drop_duplicates(
        subset=["Experiment", "Protocol", "Dataset", "Method", "Seed"],
        keep="last",
    )
    return seed.reset_index(drop=True)


def load_audit_rows(audit_files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in audit_files:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            continue
        if not frame.empty:
            frame["SourceFile"] = _relative(path)
            frame["SourceMTime"] = float(path.stat().st_mtime)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    audit = pd.concat(frames, ignore_index=True, sort=False)
    return audit.sort_values(["SourceMTime", "SourceFile"]).drop_duplicates(subset=["Dataset"], keep="last")


def completed_rows(seed: pd.DataFrame) -> pd.DataFrame:
    ok = seed[seed["Status"].astype(str).eq("ok")].copy()
    return ok[np.isfinite(pd.to_numeric(ok["R2"], errors="coerce"))].copy()


def dataset_means(seed: pd.DataFrame) -> pd.DataFrame:
    ok = completed_rows(seed)
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["Experiment", "Protocol", "Dataset", "Method"], as_index=False)
        .agg(
            Mean_R2=("R2", "mean"),
            Median_R2=("R2", "median"),
            Std_R2=("R2", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_MAE=("MAE", "mean"),
            Mean_Time_s=("Time", "mean"),
            N_Seeds=("Seed", "nunique"),
        )
        .reset_index(drop=True)
    )


def main_default_dataset_means(ds: pd.DataFrame) -> pd.DataFrame:
    if ds.empty:
        return ds
    return ds[ds["Experiment"].astype(str).eq("main_benchmark") & ds["Protocol"].astype(str).eq("default")].reset_index(drop=True)


def catch_ablation_dataset_means(ds: pd.DataFrame) -> pd.DataFrame:
    if ds.empty:
        return ds
    return ds[ds["Experiment"].astype(str).eq("catch_ablation") & ds["Protocol"].astype(str).eq("default")].reset_index(drop=True)


def external_validation_dataset_means(ds: pd.DataFrame) -> pd.DataFrame:
    if ds.empty:
        return ds
    return ds[ds["Experiment"].astype(str).eq("external_validation") & ds["Protocol"].astype(str).eq("default")].reset_index(drop=True)


def clone_tree_on_y(ds: pd.DataFrame) -> pd.DataFrame:
    if ds.empty:
        return ds
    cat = ds[ds["Method"].astype(str).eq("CatBoost")].copy()
    if cat.empty:
        return ds
    cat["Method"] = "Tree-on-Y"
    return pd.concat([ds, cat], ignore_index=True, sort=False)


def summarize_methods(ds: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        part = ds[ds["Method"].astype(str).eq(method)]
        if part.empty:
            continue
        rows.append(
            {
                "Method": method,
                "N_Datasets": int(part["Dataset"].nunique()),
                "Mean_R2": float(part["Mean_R2"].mean()),
                "Median_R2": float(part["Mean_R2"].median()),
                "Mean_RMSE": float(part["Mean_RMSE"].mean()),
                "Mean_MAE": float(part["Mean_MAE"].mean()),
                "Mean_Time_s": float(part["Mean_Time_s"].mean()),
                "Mean_N_Seeds": float(part["N_Seeds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * float(p)))
        adjusted[idx] = running
    return adjusted


def pairwise_vs_reference(
    ds: pd.DataFrame,
    reference: str = "CATCH",
    bootstrap_samples: int = 5000,
    seed: int = 20260616,
) -> pd.DataFrame:
    del bootstrap_samples, seed
    if ds.empty:
        return pd.DataFrame()
    pivot = ds.pivot_table(index="Dataset", columns="Method", values="Mean_R2", aggfunc="mean")
    if reference not in pivot.columns:
        return pd.DataFrame()
    rows = []
    p_values = []
    for baseline in [col for col in pivot.columns if col != reference]:
        paired = pivot[[reference, baseline]].dropna()
        if paired.empty:
            continue
        delta = paired[reference] - paired[baseline]
        if wilcoxon is not None and len(delta) > 0 and not np.allclose(delta, 0.0):
            try:
                p_value = float(wilcoxon(delta).pvalue)
            except Exception:
                p_value = math.nan
        else:
            p_value = math.nan
        p_values.append(p_value)
        rows.append(
            {
                "Comparison": f"{reference} vs {baseline}",
                "Reference": reference,
                "Baseline": baseline,
                "N_Datasets": int(len(delta)),
                "Mean_Delta_R2": float(delta.mean()),
                "Median_Delta_R2": float(delta.median()),
                "Win_Count": int((delta > 0).sum()),
                "Loss_Count": int((delta < 0).sum()),
                "Tie_Count": int((delta == 0).sum()),
                "Wilcoxon_p": p_value,
            }
        )
    adjusted = _holm_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["Wilcoxon_p_Holm"] = p_adj
    return pd.DataFrame(rows).sort_values("Mean_Delta_R2", ascending=False).reset_index(drop=True)


def catch_ablation_component_deltas(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    labels = {
        "CATCH-no-target-calibration": (
            "target calibration",
            "isolated marginal effect of target calibration inside CATCH",
        ),
        "CATCH-no-eta-scale": (
            "eta scaling",
            "isolated marginal effect of eta-normalized complement construction",
        ),
        "CATCH-no-CWLS-fusion": (
            "CWLS reliability fusion",
            "isolated marginal effect of constrained reliability fusion",
        ),
        "CATCH-no-U": (
            "training-side unlabeled covariate support",
            "isolated marginal effect of removing unlabeled covariates; not the total semi-supervised framework effect",
        ),
        "CATCH-no-disagreement-variance": (
            "disagreement-scale diagnostics",
            "isolated marginal effect of disagreement-scale diagnostics",
        ),
        "CATCH-no-support-variance": (
            "support-scale diagnostics",
            "isolated marginal effect of support-scale diagnostics",
        ),
    }
    rows = []
    for row in pairwise.itertuples(index=False):
        baseline = str(row.Baseline)
        component, interpretation = labels.get(baseline, (baseline, "internal CATCH component delta"))
        rows.append(
            {
                "Component": component,
                "Baseline": baseline,
                "Mean_Delta_R2": float(row.Mean_Delta_R2),
                "Median_Delta_R2": float(row.Median_Delta_R2),
                "Win_Count": int(row.Win_Count),
                "Loss_Count": int(row.Loss_Count),
                "Wilcoxon_p_Holm": float(row.Wilcoxon_p_Holm) if pd.notna(row.Wilcoxon_p_Holm) else math.nan,
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def friedman_table(ds: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if friedmanchisquare is None or ds.empty:
        return pd.DataFrame()
    pivot = ds[ds["Method"].isin(methods)].pivot_table(index="Dataset", columns="Method", values="Mean_R2", aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return pd.DataFrame()
    stat, p_value = friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
    return pd.DataFrame([{"N_Datasets": int(pivot.shape[0]), "Methods": "|".join(pivot.columns), "Friedman_chi2": float(stat), "p": float(p_value)}])


def diagnostic_summary(seed: pd.DataFrame) -> pd.DataFrame:
    ok = completed_rows(seed)
    rows = []
    column_options = {
        "eta_hat": ["catch_vf_rc_eta_hat", "rc_eta_hat"],
        "rho_hat": ["catch_vf_rc_rho", "rc_rho"],
    }
    for label, cols in column_options.items():
        for col in cols:
            if col in ok.columns:
                values = pd.to_numeric(ok[col], errors="coerce").dropna()
                if not values.empty:
                    rows.append(
                        {
                            "Quantity": label,
                            "N": int(values.shape[0]),
                            "Mean": float(values.mean()),
                            "Median": float(values.median()),
                            "Std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                        }
                    )
                    break
    return pd.DataFrame(rows)


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    run_roots = [Path(item) for item in args.run_roots]
    seed_files = [Path(item) for item in args.seed_files] if args.seed_files else [path for root in run_roots for path in seed_files_under(root)]
    audit_files = [Path(item) for item in args.audit_files] if args.audit_files else [path for root in run_roots for path in audit_files_under(root)]

    seed = load_seed_rows(seed_files)
    audit = load_audit_rows(audit_files)
    ds = dataset_means(seed)
    main_ds = main_default_dataset_means(ds)
    ablation_ds = catch_ablation_dataset_means(ds)
    external_validation_ds = external_validation_dataset_means(ds)
    simple_ds = clone_tree_on_y(main_ds)
    ablation_pairwise = pairwise_vs_reference(
        ablation_ds,
        reference=args.reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    main_full_seed = seed[
        seed["Experiment"].astype(str).eq("main_benchmark")
        & seed["Protocol"].astype(str).eq("default")
        & seed["Method"].astype(str).isin(PUBLIC_MAIN_METHODS)
    ].copy()
    external_validation_full_seed = seed[
        seed["Experiment"].astype(str).eq("external_validation")
        & seed["Protocol"].astype(str).eq("default")
        & seed["Method"].astype(str).isin(EXTERNAL_VALIDATION_METHODS)
    ].copy()

    out_dir = Path(args.out_dir)
    tables = {
        "seed_combined.csv": seed,
        "dataset_mean.csv": ds,
        "dataset_audit.csv": audit,
        "main_default_dataset_mean.csv": main_ds,
        "main_summary.csv": summarize_methods(main_ds, PUBLIC_MAIN_METHODS),
        "main_pairwise.csv": pairwise_vs_reference(main_ds, reference=args.reference, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed),
        "main_benchmark_full_seed_combined.csv": main_full_seed,
        "main_benchmark_full_dataset_method_mean.csv": main_ds,
        "main_benchmark_full_method_summary.csv": summarize_methods(main_ds, PUBLIC_MAIN_METHODS),
        "main_benchmark_full_pairwise_vs_catch.csv": pairwise_vs_reference(main_ds, reference=args.reference, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed),
        "simple_fusion_summary.csv": summarize_methods(simple_ds, SIMPLE_FUSION_METHODS),
        "simple_fusion_pairwise.csv": pairwise_vs_reference(simple_ds, reference=args.reference, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed),
        "catch_ablation_dataset_mean.csv": ablation_ds,
        "catch_ablation_summary.csv": summarize_methods(ablation_ds, CATCH_ABLATION_METHODS),
        "catch_ablation_pairwise.csv": ablation_pairwise,
        "catch_ablation_component_deltas.csv": catch_ablation_component_deltas(ablation_pairwise),
        "external_validation_dataset_mean.csv": external_validation_ds,
        "external_validation_summary.csv": summarize_methods(external_validation_ds, EXTERNAL_VALIDATION_METHODS),
        "external_validation_pairwise.csv": pairwise_vs_reference(external_validation_ds, reference=args.reference, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed),
        "external_validation_full_seed_combined.csv": external_validation_full_seed,
        "external_validation_full_dataset_method_mean.csv": external_validation_ds,
        "external_validation_full_method_summary.csv": summarize_methods(external_validation_ds, EXTERNAL_VALIDATION_METHODS),
        "external_validation_full_pairwise_vs_catch.csv": pairwise_vs_reference(external_validation_ds, reference=args.reference, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed),
        "friedman.csv": friedman_table(main_ds, [method for method in PUBLIC_MAIN_METHODS if method != "TabPFN-v3"]),
        "diagnostics.csv": diagnostic_summary(seed),
    }
    for filename, table in tables.items():
        write_table(out_dir / filename, table)
    print(f"Wrote {len(tables)} tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
