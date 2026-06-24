"""Audit the External Validation 20 data bundle.

The audit is intentionally lightweight: it verifies the released manifest, the
20 bundled CSV files, OpenML identifiers, runner target columns, task-type
notes, and pandas content hashes. It does not evaluate model results.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "external_validation20" / "manifest.csv"
EXPECTED_RULE = "predefined_openml_external_validation20_cohort"
EXPECTED_TASK_TYPES = {
    "openml_numeric_regression_task",
    "derived_credit_amount_regression_stress_task",
}


def short_content_hash(path: Path) -> str:
    frame = pd.read_csv(path)
    digest = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(digest).hexdigest()[:16]


def audit_manifest(manifest_path: Path = MANIFEST) -> list[str]:
    issues: list[str] = []
    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]

    manifest = pd.read_csv(manifest_path)
    required = {
        "relative_path",
        "openml_did",
        "openml_name",
        "openml_target",
        "task_type",
        "target_use_note",
        "rows",
        "features",
        "content_hash",
        "cohort_order",
        "selection_rule",
    }
    missing_cols = required.difference(manifest.columns)
    if missing_cols:
        return [f"missing manifest columns: {sorted(missing_cols)}"]

    if len(manifest) != 20:
        issues.append(f"expected 20 external-validation rows, found {len(manifest)}")
    if manifest["openml_did"].nunique() != len(manifest):
        issues.append("duplicate OpenML data IDs in manifest")
    if manifest["content_hash"].astype(str).nunique() != len(manifest):
        issues.append("duplicate pandas content hashes in manifest")

    orders = manifest["cohort_order"].astype(int).tolist()
    if orders != list(range(1, len(manifest) + 1)):
        issues.append("cohort_order must be consecutive 1..20 in manifest order")

    rules = set(manifest["selection_rule"].astype(str))
    if rules != {EXPECTED_RULE}:
        issues.append(f"unexpected selection_rule values: {sorted(rules)}")

    task_types = set(manifest["task_type"].astype(str))
    if task_types != EXPECTED_TASK_TYPES:
        issues.append(f"unexpected task_type values: {sorted(task_types)}")
    derived = manifest[manifest["task_type"] == "derived_credit_amount_regression_stress_task"]
    if len(derived) != 1 or int(derived.iloc[0]["openml_did"]) != 31:
        issues.append("expected exactly one declared derived credit-g stress task with OpenML data ID 31")
    numeric_count = int((manifest["task_type"] == "openml_numeric_regression_task").sum())
    if numeric_count != 19:
        issues.append(f"expected 19 ordinary numeric-regression tasks, found {numeric_count}")

    listed_names = {Path(str(row.relative_path)).name for row in manifest.itertuples(index=False)}
    bundled_names = {path.name for path in manifest_path.parent.glob("OpenMLEV20_*.csv")}
    extra_names = sorted(bundled_names.difference(listed_names))
    missing_names = sorted(listed_names.difference(bundled_names))
    if extra_names:
        issues.append(f"extra External Validation 20 CSV files not listed in manifest: {extra_names}")
    if missing_names:
        issues.append(f"manifest-listed External Validation 20 CSV files are missing: {missing_names}")

    for row in manifest.itertuples(index=False):
        rel = Path(str(row.relative_path))
        path = DATA_DIR / rel
        if not path.exists():
            issues.append(f"missing data file: {rel}")
            continue
        frame = pd.read_csv(path, nrows=5)
        target = str(row.openml_target)
        if target not in frame.columns:
            issues.append(f"target column {target!r} missing in {rel}")
        if str(row.task_type) == "derived_credit_amount_regression_stress_task":
            if "class" not in frame.columns:
                issues.append(f"derived credit-g stress task is missing original class covariate in {rel}")
            target_dtype = pd.read_csv(path, usecols=[target])[target].dtype
            if not pd.api.types.is_numeric_dtype(target_dtype):
                issues.append(f"derived credit-g stress target must be numeric in {rel}")
        observed = short_content_hash(path)
        expected = str(row.content_hash)
        if observed != expected:
            issues.append(f"hash mismatch for {rel}: manifest={expected}, observed={observed}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    issues = audit_manifest(args.manifest)
    if issues:
        print("External Validation 20 audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    manifest = pd.read_csv(args.manifest)
    print(
        "External Validation 20 audit passed: "
        f"{len(manifest)} files, {manifest['openml_did'].nunique()} unique OpenML data IDs, "
        "19 ordinary numeric-regression tasks, and 1 declared derived credit-g stress task."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
