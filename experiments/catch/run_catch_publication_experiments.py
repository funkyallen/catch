"""Run the CATCH publication experiment package.

This wrapper keeps the paper-facing method list and protocols in one place
while delegating all actual training to ``run_catch_suite.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.catch.run_catch_suite import EXTERNAL_VALIDATION20_DATASETS, DATASETS, split_lanes


SEEDS_10 = ["42", "123", "456", "789", "1011", "2027", "3141", "2718", "1618", "9001"]

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

LABEL_EFFICIENCY_METHODS = [
    "NN-only",
    "NN+Tree-Avg",
    "CatBoost",
    "AutoGluon",
    "CATCH",
]

UNLABELED_CONTAMINATION_METHODS = [
    "NN-only",
    "CatBoost",
    "AutoGluon",
    "CATCH",
]

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
    "CATCH-rho0-complement",
    "CATCH-no-U",
    "CATCH-no-disagreement-variance",
    "CATCH-no-support-variance",
]

EXPERIMENT_METHODS = {
    "main_benchmark": MAIN_METHODS,
    "runtime_pareto": MAIN_METHODS,
    "label_ratio": LABEL_EFFICIENCY_METHODS,
    "unlabeled_contamination": UNLABELED_CONTAMINATION_METHODS,
    "catch_ablation": CATCH_ABLATION_METHODS,
    "external_validation": EXTERNAL_VALIDATION_METHODS,
}


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
        ],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method override for all selected experiments; useful for running only newly added ablations.",
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--include-tabpfn", action="store_true")
    parser.add_argument("--tabpfn-model-path", default=None)
    parser.add_argument("--tabpfn-device", default="cuda")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--autogluon-time-limit", type=int, default=60)
    parser.add_argument("--tree-rounds", type=int, default=300)
    parser.add_argument("--tree-lr", type=float, default=0.05)
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--labeled-ratios", nargs="+", default=["0.20", "0.40", "0.60", "0.80", "1.00"])
    parser.add_argument(
        "--unlabeled-contamination-levels",
        nargs="+",
        default=["0.00", "0.10", "0.20", "0.30", "0.50", "0.70"],
    )
    parser.add_argument("--max-parallel-lanes", type=int, default=None)
    parser.add_argument("--tag-prefix", default="catch_publication")
    parser.add_argument("--run-root", default="r/catch_publication")
    parser.add_argument("--log-root", default="r/catch_publication_lane_logs")
    parser.add_argument("--model-param-profile", choices=["project", "library-defaults"], default="project")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def methods_for_experiment(experiment: str, include_tabpfn: bool, override: list[str] | None = None) -> list[str]:
    methods = list(override or EXPERIMENT_METHODS.get(experiment, []))
    if include_tabpfn and experiment in {"main_benchmark", "runtime_pareto"}:
        methods.insert(-1, "TabPFN-v3")
    return methods


def command_for_lane(args: argparse.Namespace, experiment: str, lane: str, datasets: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/catch/run_catch_suite.py",
        "--experiments",
        experiment,
        "--profile",
        "full",
        "--datasets",
        *datasets,
        "--seeds",
        *(args.seeds or SEEDS_10),
        "--tag",
        f"{args.tag_prefix}_{experiment}_{lane}",
        "--log-dir",
        str(ROOT / args.run_root / experiment / lane),
        "--model-param-profile",
        args.model_param_profile,
    ]
    methods = methods_for_experiment(experiment, args.include_tabpfn, override=args.methods)
    if methods:
        cmd.extend(["--methods", *methods])
    cmd.extend(
        [
            "--autogluon-time-limit",
            str(args.autogluon_time_limit),
            "--tree-rounds",
            str(args.tree_rounds),
            "--tree-lr",
            str(args.tree_lr),
            "--tree-depth",
            str(args.tree_depth),
        ]
    )
    if experiment == "label_ratio":
        cmd.extend(["--labeled-ratios", *[str(v) for v in args.labeled_ratios]])
    if experiment == "unlabeled_contamination":
        cmd.extend(["--unlabeled-contamination-levels", *[str(v) for v in args.unlabeled_contamination_levels]])
    if args.include_tabpfn:
        cmd.extend(
            [
                "--tabpfn-device",
                args.tabpfn_device,
                "--tabpfn-n-estimators",
                str(args.tabpfn_n_estimators),
            ]
        )
        if args.tabpfn_model_path:
            cmd.extend(["--tabpfn-model-path", args.tabpfn_model_path])
    if args.smoke:
        cmd.append("--smoke")
    return cmd


def print_command(cmd: list[str]) -> None:
    print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))


def run_commands(args: argparse.Namespace, commands: list[tuple[str, str, list[str]]]) -> int:
    log_root = ROOT / args.log_root
    log_root.mkdir(parents=True, exist_ok=True)
    max_parallel = args.max_parallel_lanes
    if max_parallel is None:
        uses_gpu_tabpfn = args.include_tabpfn and str(args.tabpfn_device).lower() not in {"cpu", "none"}
        max_parallel = 1 if uses_gpu_tabpfn else 4
    max_parallel = max(1, int(max_parallel))

    exit_code = 0
    for start in range(0, len(commands), max_parallel):
        batch = commands[start : start + max_parallel]
        processes: list[tuple[str, str, subprocess.Popen[bytes], object, object]] = []
        for experiment, lane, cmd in batch:
            stdout_path = log_root / f"{args.tag_prefix}_{experiment}_{lane}.out.log"
            stderr_path = log_root / f"{args.tag_prefix}_{experiment}_{lane}.err.log"
            stdout = stdout_path.open("wb")
            stderr = stderr_path.open("wb")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr)
            processes.append((experiment, lane, proc, stdout, stderr))
            print(f"{experiment}/{lane}: launched")

        for experiment, lane, proc, stdout, stderr in processes:
            code = proc.wait()
            stdout.close()
            stderr.close()
            print(f"{experiment}/{lane}: exit_code={code}")
            exit_code = max(exit_code, int(code != 0))
    return exit_code


def main() -> int:
    args = parse_args()
    commands: list[tuple[str, str, list[str]]] = []
    for experiment in args.experiments:
        default_datasets = EXTERNAL_VALIDATION20_DATASETS if experiment == "external_validation" else DATASETS
        selected_datasets = list(default_datasets if args.datasets is None else args.datasets)
        lane_specs = [
            (f"lane{idx}", lane_datasets)
            for idx, lane_datasets in enumerate(split_lanes(selected_datasets), 1)
            if lane_datasets
        ]
        for lane, lane_datasets in lane_specs:
            cmd = command_for_lane(args, experiment, lane, lane_datasets)
            commands.append((experiment, lane, cmd))

    for experiment, lane, cmd in commands:
        print(f"\n# {experiment}/{lane}")
        print_command(cmd)

    if args.plan_only:
        return 0
    return run_commands(args, commands)


if __name__ == "__main__":
    raise SystemExit(main())
