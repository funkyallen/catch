"""Fetch the OpenML External Validation 20 suite for CATCH experiments.

The cohort construction rule is deliberately result-blind: it uses only OpenML
task metadata, excludes the local 30-dataset benchmark plus prior validation
IDs, assigns each eligible dataset a fixed-seed pseudo-random rank key, and then
downloads the first 20 usable datasets in that frozen queue. No model score or
outcome summary is used to construct the cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "external_validation20_20260624"
MANIFEST_FILENAME = "clean20_manifest.csv"
DATASET_LIST_FILENAME = "dataset_list_clean20.txt"
SELECTION_TABLE_FILENAME = "selection_table_metadata.csv"
SELECTION_RULE = "metadata_filtered_openml_seed20260624_random_permutation_exclude_main_family_overlap_deduplicate_content_and_family_take20"

PRIOR_EXTERNAL_VALIDATION_DIDS = {
    198,
    209,
    308,
    503,
    504,
    529,
    558,
    688,
    1414,
    41265,
    41539,
    41542,
    41700,
    42183,
    42184,
}

MAIN_BENCHMARK_NAME_TOKENS = {
    "ailerons",
    "ames_housing",
    "bank8fm",
    "california_housing",
    "cpu_act",
    "diamonds",
    "elevators",
    "house_sales",
    "sgemm_gpu",
    "pol",
    "abalone",
    "airfoil_self_noise",
    "cpu_small",
    "debutanizer",
    "kin8nm",
    "medical_cost",
    "puma8nh",
    "space_ga",
    "weather_izmir",
    "wine_quality",
    "air_quality",
    "appliances_energy",
    "bike_sharing_hour",
    "combined_cycle_power_plant",
    "concrete",
    "energy_efficiency",
    "naval_propulsion",
    "parkinsons_telemonitoring",
    "qsar_aquatic_toxicity",
    "real_estate_valuation",
}

PRIOR_EXTERNAL_VALIDATION_NAME_TOKENS = {
    "titanic",
    "wine",
    "cpmp_2015_regression",
    "quake",
    "pollen",
    "analcatdata_supreme",
    "wind",
    "bank32nh",
    "puma32h",
    "visualizing_soil",
    "delta_elevators",
    "dataset_sales",
    "kaggle_bike_sharing_demand_challange",
    "cd4",
    "rainfall_bangladesh",
}

EXCLUDED_NAME_TOKENS = MAIN_BENCHMARK_NAME_TOKENS | PRIOR_EXTERNAL_VALIDATION_NAME_TOKENS
# Source-family exclusions keep the external cohort separate from the main benchmark families.
MAIN_FAMILY_OVERLAP_RE = re.compile(
    r"(?:california|housing|houses|house_|house_sales|kings_county|miamihousing|"
    r"brazilian_houses|real_estate|estate_price|rent|bike_sharing|concrete|"
    r"air_quality|health_insurance|healthcare|insurance|medical|wine|weather|"
    r"abalone|debutanizer|puma|cpu|kin8|elevators|energy|appliances|"
    r"parkinsons|naval|qsar|aquatic|toxic|sgemm|diamonds|bank8fm|"
    r"ailerons|space_ga|combined_cycle|power_plant)",
    re.I,
)
SYNTHETIC_NAME_RE = re.compile(
    r"(?:^|_)(?:fri|friedman|synthetic|artificial|random|mabbob|bbob|ela_as)(?:_|$)",
    re.I,
)


@dataclass(frozen=True)
class ValidationItem:
    task_id: int
    did: int
    name: str
    target: str
    rows: int
    features: int
    numeric_features: int
    symbolic_features: int
    missing_values: int
    missing_ratio: float
    size_bin: str
    rank_key: str
    family_key: str


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return value or "dataset"


def norm_name(text: str) -> str:
    value = slug(text).lower()
    value = re.sub(r"^(openmlh_\d+_|openml_|review_)", "", value)
    return value


def name_family_key(text: str) -> str:
    # Collapse small naming variants so one source family cannot occupy multiple slots.
    value = norm_name(text)
    value = re.sub(r"_\d+_rows?$", "", value)
    value = re.sub(r"_\d+d(?=_|$)", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def seeded_random_rank_key(seed: str, did: int, name: str) -> str:
    payload = f"{seed}:{int(did)}:{name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def short_content_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    val = parse_float(value)
    if math.isnan(val):
        return default
    return int(round(val))


def task_inputs(task: dict[str, Any]) -> dict[str, str]:
    return {str(item.get("name", "")): str(item.get("value", "")) for item in task.get("input", [])}


def task_quality(task: dict[str, Any]) -> dict[str, str]:
    return {str(item.get("name", "")): str(item.get("value", "")) for item in task.get("quality", [])}


def fetch_regression_tasks(limit: int, page_size: int = 500) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for offset in range(0, limit, page_size):
        current_limit = min(page_size, limit - offset)
        url = (
            "https://www.openml.org/api/v1/json/task/list/type/2/"
            f"limit/{current_limit}/offset/{offset}"
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                page = response.json().get("tasks", {}).get("task", [])
                if not isinstance(page, list):
                    raise RuntimeError("OpenML task API did not return a task list")
                tasks.extend(page)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            if tasks:
                print(f"stop task pagination at offset={offset}: {last_error}")
                break
            raise RuntimeError(f"OpenML task API failed at offset={offset}: {last_error}")
        if not page:
            break
    return tasks


def size_bin(rows: int) -> str:
    if rows < 3000:
        return "small"
    if rows < 9000:
        return "medium"
    return "large"


def make_validation_items(tasks: list[dict[str, Any]], seed: str) -> list[ValidationItem]:
    seen_dids: set[int] = set()
    validation_items: list[ValidationItem] = []
    for task in tasks:
        inputs = task_inputs(task)
        quality = task_quality(task)
        did = parse_int(task.get("did"))
        task_id = parse_int(task.get("task_id"))
        name = str(task.get("name", "")).strip()
        target = inputs.get("target_feature", "").strip()
        rows = parse_int(quality.get("NumberOfInstances"))
        features = parse_int(quality.get("NumberOfFeatures"))
        numeric = parse_int(quality.get("NumberOfNumericFeatures"))
        symbolic = parse_int(quality.get("NumberOfSymbolicFeatures"))
        missing_values = parse_int(quality.get("NumberOfMissingValues"))
        denom = max(1, rows * max(1, features))
        missing_ratio = missing_values / denom
        token = norm_name(name)
        family_key = name_family_key(name)

        if not did or did in seen_dids:
            continue
        if did in PRIOR_EXTERNAL_VALIDATION_DIDS or token in EXCLUDED_NAME_TOKENS:
            continue
        if MAIN_FAMILY_OVERLAP_RE.search(token):
            continue
        if not target or not name or SYNTHETIC_NAME_RE.search(token):
            continue
        if str(task.get("status", "")).lower() != "active":
            continue
        if rows < 1000 or rows > 30000:
            continue
        if features < 3 or features > 80:
            continue
        if numeric < 2:
            continue
        if symbolic > 20:
            continue
        if missing_ratio > 0.05:
            continue

        seen_dids.add(did)
        validation_items.append(
            ValidationItem(
                task_id=task_id,
                did=did,
                name=name,
                target=target,
                rows=rows,
                features=features,
                numeric_features=numeric,
                symbolic_features=symbolic,
                missing_values=missing_values,
                missing_ratio=missing_ratio,
                size_bin=size_bin(rows),
                rank_key=seeded_random_rank_key(seed, did, name),
                family_key=family_key,
            )
        )
    return validation_items


def ordered_validation_items(validation_items: list[ValidationItem]) -> list[ValidationItem]:
    return sorted(validation_items, key=lambda item: (item.rank_key, item.did, item.name))


def _write_dataset_worker(item: ValidationItem, out_dir: Path, queue: mp.Queue) -> None:
    try:
        queue.put(write_dataset(item, out_dir))
    except Exception as exc:
        queue.put({"__error__": repr(exc), "openml_did": item.did, "openml_name": item.name})


def write_dataset_with_timeout(item: ValidationItem, out_dir: Path, timeout_seconds: int) -> dict[str, Any] | None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_write_dataset_worker, args=(item, out_dir, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        print(f"skip did={item.did} name={item.name}: timed out after {timeout_seconds}s")
        return None
    if queue.empty():
        print(f"skip did={item.did} name={item.name}: worker returned no result")
        return None
    result = queue.get()
    if isinstance(result, dict) and "__error__" in result:
        print(f"skip did={item.did} name={item.name}: worker error: {result['__error__']}")
        return None
    return result


def write_dataset(item: ValidationItem, out_dir: Path) -> dict[str, Any] | None:
    out_name = f"OpenMLEV20_{item.did}_{slug(item.name)}.csv"
    out_path = out_dir / out_name
    try:
        bunch = fetch_openml(
            data_id=item.did,
            target_column=item.target,
            as_frame=True,
            parser="auto",
        )
    except Exception as exc:
        print(f"skip did={item.did} name={item.name}: fetch failed: {exc}")
        return None

    X = bunch.data.copy()
    y = pd.to_numeric(bunch.target, errors="coerce")
    df = X.copy()
    df["target"] = y
    before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    if len(df) < 1000 or df["target"].nunique(dropna=True) < 10:
        print(f"skip did={item.did} name={item.name}: unusable numeric target after fetch")
        return None

    df.to_csv(out_path, index=False, encoding="utf-8")
    stored_df = pd.read_csv(out_path)
    feature_df = df.iloc[:, :-1]
    numeric_features = int(feature_df.select_dtypes(include=["number", "bool"]).shape[1])
    return {
        "file": out_name,
        "relative_path": f"{out_dir.name}/{out_name}",
        "openml_task_id": item.task_id,
        "openml_did": item.did,
        "openml_name": item.name,
        "openml_target": item.target,
        "selection_size_bin": item.size_bin,
        "selection_rank_key": item.rank_key,
        "selection_family_key": item.family_key,
        "selection_rule": SELECTION_RULE,
        "metadata_rows": item.rows,
        "metadata_features": item.features,
        "metadata_numeric_features": item.numeric_features,
        "metadata_symbolic_features": item.symbolic_features,
        "metadata_missing_ratio": item.missing_ratio,
        "rows_before_target_drop": before,
        "rows": int(df.shape[0]),
        "features": int(feature_df.shape[1]),
        "target_unique": int(df["target"].nunique(dropna=True)),
        "content_hash": short_content_hash(stored_df),
        "numeric_features": numeric_features,
        "categorical_features": int(feature_df.shape[1] - numeric_features),
        "missing_ratio": float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1])),
    }


def selection_table_row(
    item: ValidationItem,
    queue_order: int,
    status: str,
    seed: str,
    selected_order: int | None = None,
) -> dict[str, Any]:
    return {
        "queue_order": queue_order,
        "selected_order": "" if selected_order is None else selected_order,
        "openml_task_id": item.task_id,
        "openml_did": item.did,
        "openml_name": item.name,
        "openml_target": item.target,
        "selection_seed": seed,
        "selection_rule": SELECTION_RULE,
        "selection_status": status,
        "selection_rank_key": item.rank_key,
        "selection_family_key": item.family_key,
        "metadata_rows": item.rows,
        "metadata_features": item.features,
        "metadata_numeric_features": item.numeric_features,
        "metadata_symbolic_features": item.symbolic_features,
        "metadata_missing_ratio": item.missing_ratio,
        "selection_size_bin": item.size_bin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--seed", default="20260624")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--per-dataset-timeout", type=int, default=90)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rebuilds are frozen by rule/seed, so stale EV20 artifacts should not survive a rerun.
    for stale in out_dir.glob("OpenMLEV20_*.csv"):
        stale.unlink()
    for stale_name in (MANIFEST_FILENAME, DATASET_LIST_FILENAME, SELECTION_TABLE_FILENAME):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    tasks = fetch_regression_tasks(args.limit)
    validation_items = ordered_validation_items(make_validation_items(tasks, seed=args.seed))
    if len(validation_items) < args.target_count:
        raise RuntimeError(f"Only {len(validation_items)} validation items passed metadata filters")

    rows: list[dict[str, Any]] = []
    attempted_dids: set[int] = set()
    selected_hashes: set[str] = set()
    selected_families: set[str] = set()
    status_by_did: dict[int, str] = {}
    for item in validation_items:
        print(
            f"fetch did={item.did} task={item.task_id} "
            f"bin={item.size_bin} rows={item.rows} features={item.features} "
            f"name={item.name}"
        )
        attempted_dids.add(item.did)
        row = write_dataset_with_timeout(item, out_dir, args.per_dataset_timeout)
        if row is None:
            status_by_did[item.did] = "eligible_but_fetch_failed_or_unusable"
        else:
            # Content and family checks happen after fetch because conversion changes the final CSV.
            content_hash = str(row["content_hash"])
            if content_hash in selected_hashes:
                status_by_did[item.did] = "eligible_but_duplicate_content"
                duplicate_path = DATA_DIR / str(row["relative_path"])
                if duplicate_path.exists():
                    duplicate_path.unlink()
            elif item.family_key in selected_families:
                status_by_did[item.did] = "eligible_but_duplicate_name_family"
                duplicate_path = DATA_DIR / str(row["relative_path"])
                if duplicate_path.exists():
                    duplicate_path.unlink()
            else:
                selected_hashes.add(content_hash)
                selected_families.add(item.family_key)
                row["selection_seed"] = args.seed
                row["selection_order"] = len(rows) + 1
                rows.append(row)
                status_by_did[item.did] = "selected_downloaded"
        if len(rows) >= args.target_count:
            break
        time.sleep(0.5)

    if len(rows) < args.target_count:
        raise RuntimeError(f"Fetched only {len(rows)} usable datasets")

    selected_dids = {int(row["openml_did"]) for row in rows}
    selected_order_by_did = {int(row["openml_did"]): int(row["selection_order"]) for row in rows}
    selection_rows = []
    for order, item in enumerate(validation_items, start=1):
        if item.did in status_by_did:
            status = status_by_did[item.did]
        elif item.did in attempted_dids:
            status = "eligible_but_fetch_failed_or_unusable"
        else:
            status = "eligible_not_attempted_after_target_count"
        selection_rows.append(selection_table_row(item, order, status, args.seed, selected_order_by_did.get(item.did)))

    manifest_path = out_dir / MANIFEST_FILENAME
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    list_path = out_dir / DATASET_LIST_FILENAME
    list_path.write_text("\n".join(str(row["relative_path"]) for row in rows) + "\n", encoding="utf-8")
    selection_table_path = out_dir / SELECTION_TABLE_FILENAME
    with selection_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    print(f"Wrote {len(rows)} datasets to {out_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote dataset list to {list_path}")
    print(f"Wrote metadata selection table to {selection_table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
