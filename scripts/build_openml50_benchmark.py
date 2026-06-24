"""Build the optional OpenML-50 CATCH-vs-AutoGluon benchmark bundle.

The selection rule is deterministic and does not inspect model results:
sort the screened OpenML metadata rows by OpenML data ID and keep the first
50 unique data IDs. The script can copy from a local source pool or, when a
manifest already exists, download missing files through scikit-learn's OpenML
interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "openml50_benchmark"
MANIFEST = OUT_DIR / "manifest.csv"
DATASET_LIST = OUT_DIR / "dataset_list.txt"
COHORT_NAME = "OpenML-50 CATCH-vs-AutoGluon benchmark"
FREEZE_DATE = "2026-06-24"
SELECTION_RULE = "screened_openml_metadata_sorted_by_data_id_first50_no_model_results"
REQUIRED_SOURCE_COLUMNS = {
    "file",
    "openml_did",
    "openml_name",
    "openml_target",
}
MANIFEST_COLUMNS = [
    "file",
    "relative_path",
    "openml_did",
    "openml_name",
    "openml_target",
    "runner_target",
    "rows",
    "features",
    "numerical_features",
    "categorical_features",
    "target_unique",
    "missing_ratio",
    "content_hash",
    "cohort_name",
    "cohort_freeze_date",
    "cohort_order",
    "source",
    "access_url",
    "selection_rule",
    "inclusion_note",
]


def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text[:80] or "openml_dataset"


def short_content_hash(path: Path) -> str:
    frame = pd.read_csv(path)
    digest = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(digest).hexdigest()[:16]


def describe_frame(path: Path, target: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} missing in {path}")
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include=["number", "bool"])
    y = pd.to_numeric(frame[target], errors="coerce")
    return {
        "rows": int(len(frame)),
        "features": int(features.shape[1]),
        "numerical_features": int(numeric.shape[1]),
        "categorical_features": int(features.shape[1] - numeric.shape[1]),
        "target_unique": int(y.nunique(dropna=True)),
        "missing_ratio": float(frame.isna().mean().mean()) if frame.size else 0.0,
        "content_hash": short_content_hash(path),
    }


def selected_source_rows(source_manifest: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(source_manifest.open(newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"empty source manifest: {source_manifest}")
    missing = REQUIRED_SOURCE_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"source manifest missing columns: {sorted(missing)}")
    seen: set[int] = set()
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: int(item["openml_did"])):
        did = int(row["openml_did"])
        if did in seen:
            continue
        seen.add(did)
        selected.append(row)
        if len(selected) == 50:
            break
    if len(selected) != 50:
        raise ValueError(f"selection rule produced {len(selected)} rows, expected 50")
    return selected


def fetch_openml_frame(openml_did: int, target: str) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(data_id=int(openml_did), as_frame=True, parser="auto")
    frame = bundle.frame.copy()
    if target not in frame.columns and getattr(bundle, "target", None) is not None:
        frame[target] = bundle.target
    if target not in frame.columns:
        raise ValueError(f"OpenML data ID {openml_did} did not expose target {target!r}")
    return frame


def rows_from_existing_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        return []
    return list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))


def build_from_source_pool(source_pool: Path) -> list[dict[str, object]]:
    source_manifest = source_pool / "manifest.csv"
    selected = selected_source_rows(source_manifest)
    manifest_rows: list[dict[str, object]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for order, row in enumerate(selected, 1):
        did = int(row["openml_did"])
        openml_name = str(row["openml_name"])
        openml_target = str(row["openml_target"])
        source_file = source_pool / str(row["file"])
        if not source_file.exists():
            raise FileNotFoundError(f"missing source CSV: {source_file}")
        dest_name = f"OpenML50_{did}_{safe_name(openml_name)}.csv"
        dest = OUT_DIR / dest_name
        shutil.copyfile(source_file, dest)
        header = pd.read_csv(dest, nrows=0).columns
        runner_target = openml_target if openml_target in header else "target"
        desc = describe_frame(dest, runner_target)
        manifest_rows.append(
            {
                "file": dest_name,
                "relative_path": f"openml50_benchmark/{dest_name}",
                "openml_did": did,
                "openml_name": openml_name,
                "openml_target": openml_target,
                "runner_target": runner_target,
                **desc,
                "cohort_name": COHORT_NAME,
                "cohort_freeze_date": FREEZE_DATE,
                "cohort_order": order,
                "source": "OpenML",
                "access_url": f"https://www.openml.org/d/{did}",
                "selection_rule": SELECTION_RULE,
                "inclusion_note": "openml50_autogluon_catch_two_method_benchmark",
            }
        )
    return manifest_rows


def build_from_manifest_download() -> list[dict[str, object]]:
    rows = rows_from_existing_manifest()
    if not rows:
        raise FileNotFoundError("manifest is missing; pass --source-pool to create it first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for row in rows:
        dest = OUT_DIR / str(row["file"])
        openml_target = str(row["openml_target"])
        runner_target = str(row.get("runner_target") or openml_target)
        if not dest.exists():
            frame = fetch_openml_frame(int(row["openml_did"]), openml_target)
            if runner_target != openml_target and openml_target in frame.columns:
                frame = frame.rename(columns={openml_target: runner_target})
            frame.to_csv(dest, index=False, encoding="utf-8")
        desc = describe_frame(dest, runner_target)
        updated = dict(row)
        updated["runner_target"] = runner_target
        updated.update(desc)
        manifest_rows.append(updated)
    return manifest_rows


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})
    DATASET_LIST.write_text(
        "\n".join(str(row["relative_path"]) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, default=None)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()

    if args.source_pool is not None:
        rows = build_from_source_pool(args.source_pool)
    elif args.download_missing:
        rows = build_from_manifest_download()
    else:
        raise SystemExit("Pass --source-pool for first build, or --download-missing when manifest already exists.")
    write_outputs(rows)
    print(f"Wrote {len(rows)} OpenML-50 rows to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
