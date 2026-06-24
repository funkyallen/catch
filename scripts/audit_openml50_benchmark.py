"""Audit the optional OpenML-50 benchmark bundle.

The audit verifies the manifest, local CSV files, target columns, deterministic
selection rule, and pandas content hashes. It does not evaluate model results.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "openml50_benchmark" / "manifest.csv"
EXPECTED_RULE = "screened_openml_metadata_sorted_by_data_id_first50_no_model_results"


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
        "runner_target",
        "rows",
        "features",
        "content_hash",
        "cohort_order",
        "selection_rule",
    }
    missing_cols = required.difference(manifest.columns)
    if missing_cols:
        return [f"missing manifest columns: {sorted(missing_cols)}"]

    if len(manifest) != 50:
        issues.append(f"expected 50 OpenML-50 rows, found {len(manifest)}")
    if manifest["openml_did"].nunique() != len(manifest):
        issues.append("duplicate OpenML data IDs in OpenML-50 manifest")
    orders = manifest["cohort_order"].astype(int).tolist()
    if orders != list(range(1, len(manifest) + 1)):
        issues.append("cohort_order must be consecutive 1..50 in manifest order")
    rules = set(manifest["selection_rule"].astype(str))
    if rules != {EXPECTED_RULE}:
        issues.append(f"unexpected selection_rule values: {sorted(rules)}")

    listed_names = {Path(str(row.relative_path)).name for row in manifest.itertuples(index=False)}
    bundled_names = {path.name for path in manifest_path.parent.glob("OpenML50_*.csv")}
    extra_names = sorted(bundled_names.difference(listed_names))
    missing_names = sorted(listed_names.difference(bundled_names))
    if extra_names:
        issues.append(f"extra OpenML-50 CSV files not listed in manifest: {extra_names}")
    if missing_names:
        issues.append(f"manifest-listed OpenML-50 CSV files are missing: {missing_names}")

    for row in manifest.itertuples(index=False):
        rel = Path(str(row.relative_path))
        path = DATA_DIR / rel
        if not path.exists():
            issues.append(f"missing data file: {rel}")
            continue
        frame = pd.read_csv(path, nrows=5)
        target = str(row.runner_target)
        if target not in frame.columns:
            issues.append(f"target column {target!r} missing in {rel}")
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
        print("OpenML-50 benchmark audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    manifest = pd.read_csv(args.manifest)
    print(
        "OpenML-50 benchmark audit passed: "
        f"{len(manifest)} files and {manifest['openml_did'].nunique()} unique OpenML data IDs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
