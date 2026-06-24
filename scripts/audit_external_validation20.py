"""Audit the frozen External Validation 20 data bundle.

This lightweight check verifies that the public package contains exactly the
manifest-listed external-validation files, that their short pandas content
hashes match the manifest, and that the cohort is recorded with the fixed-seed
metadata-filtered random OpenML selection rule used by the fetch helper.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "external_validation20_20260624" / "clean20_manifest.csv"
SELECTION_TABLE = DATA_DIR / "external_validation20_20260624" / "selection_table_metadata.csv"
EXPECTED_RULE = "metadata_filtered_openml_seed20260624_random_permutation_exclude_main_family_overlap_deduplicate_content_and_family_take20"
EXPECTED_SEED = "20260624"
DISALLOWED_MAIN_FAMILY_RE = re.compile(
    r"(?:california|housing|houses|house_|house_sales|kings_county|miamihousing|"
    r"brazilian_houses|real_estate|estate_price|rent|bike_sharing|concrete|"
    r"air_quality|health_insurance|healthcare|insurance|medical|wine|weather|"
    r"abalone|debutanizer|puma|cpu|kin8|elevators|energy|appliances|"
    r"parkinsons|naval|qsar|aquatic|toxic|sgemm|diamonds|bank8fm|"
    r"ailerons|space_ga|combined_cycle|power_plant)",
    re.I,
)
DISALLOWED_SYNTHETIC_RE = re.compile(
    r"(?:^|_)(?:fri|friedman|synthetic|artificial|random|mabbob|bbob|ela_as)(?:_|$)",
    re.I,
)


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
        "content_hash",
        "selection_rule",
        "selection_seed",
        "selection_order",
        "selection_rank_key",
        "selection_family_key",
    }
    missing_cols = required.difference(manifest.columns)
    if missing_cols:
        issues.append(f"missing manifest columns: {sorted(missing_cols)}")
        return issues

    if len(manifest) != 20:
        issues.append(f"expected 20 external-validation rows, found {len(manifest)}")
    if manifest["openml_did"].nunique() != len(manifest):
        issues.append("duplicate OpenML data IDs in manifest")
    if manifest["content_hash"].astype(str).nunique() != len(manifest):
        issues.append("duplicate pandas content hashes in manifest")

    rules = set(manifest["selection_rule"].astype(str))
    if rules != {EXPECTED_RULE}:
        issues.append(f"unexpected selection_rule values: {sorted(rules)}")

    seeds = set(manifest["selection_seed"].astype(str))
    if seeds != {EXPECTED_SEED}:
        issues.append(f"unexpected selection_seed values: {sorted(seeds)}")

    orders = manifest["selection_order"].astype(int).tolist()
    if orders != list(range(1, len(manifest) + 1)):
        issues.append("selection_order must be consecutive 1..20 in manifest order")

    if manifest["selection_rank_key"].astype(str).nunique() != len(manifest):
        issues.append("selection_rank_key values are not unique in manifest")
    if manifest["selection_family_key"].astype(str).nunique() != len(manifest):
        issues.append("selection_family_key values are not unique in manifest")
    for row in manifest.itertuples(index=False):
        family_text = f"{row.openml_name} {row.selection_family_key}"
        if DISALLOWED_MAIN_FAMILY_RE.search(str(family_text)):
            issues.append(f"main-benchmark source-family overlap in EV20 row: {row.openml_did} {row.openml_name}")
        if DISALLOWED_SYNTHETIC_RE.search(str(family_text)):
            issues.append(f"synthetic/benchmark-generator source in EV20 row: {row.openml_did} {row.openml_name}")

    listed_names = {Path(str(row.relative_path)).name for row in manifest.itertuples(index=False)}
    bundled_names = {path.name for path in manifest_path.parent.glob("OpenMLEV20_*.csv")}
    extra_names = sorted(bundled_names.difference(listed_names))
    if extra_names:
        issues.append(f"extra EV20 CSV files not listed in manifest: {extra_names}")

    for row in manifest.itertuples(index=False):
        rel = Path(str(row.relative_path))
        path = DATA_DIR / rel
        if not path.exists():
            issues.append(f"missing data file: {rel}")
            continue
        observed = short_content_hash(path)
        expected = str(row.content_hash)
        if observed != expected:
            issues.append(f"hash mismatch for {rel}: manifest={expected}, observed={observed}")

    selection_path = manifest_path.parent / SELECTION_TABLE.name
    if not selection_path.exists():
        issues.append(f"missing metadata selection table: {selection_path}")
        return issues

    # The selection table makes skipped rows auditable without shipping generated results.
    selection = pd.read_csv(selection_path)
    selection_required = {
        "queue_order",
        "selected_order",
        "openml_did",
        "selection_seed",
        "selection_rule",
        "selection_status",
        "selection_rank_key",
        "selection_family_key",
    }
    missing_selection_cols = selection_required.difference(selection.columns)
    if missing_selection_cols:
        issues.append(f"missing selection-table columns: {sorted(missing_selection_cols)}")
        return issues
    if set(selection["selection_seed"].astype(str)) != {EXPECTED_SEED}:
        issues.append("selection table seed does not match expected seed")
    if set(selection["selection_rule"].astype(str)) != {EXPECTED_RULE}:
        issues.append("selection table rule does not match expected rule")
    selected = selection[selection["selection_status"].astype(str).eq("selected_downloaded")].copy()
    if len(selected) != 20:
        issues.append(f"expected 20 selected_downloaded rows, found {len(selected)}")
    selected_dids = selected["openml_did"].astype(int).tolist()
    manifest_dids = manifest["openml_did"].astype(int).tolist()
    if set(selected_dids) != set(manifest_dids):
        issues.append("selected rows in metadata table do not match manifest OpenML data IDs")

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
        "fixed-seed metadata-filtered random selection rule."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
