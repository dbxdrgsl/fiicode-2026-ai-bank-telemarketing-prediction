from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from src.config import repo_root
from src.features import ID_COL, TARGET, resolve_data_dir

UCI_DATASET_PAGE_URL = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
UCI_DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_INNER_ARCHIVE_NAME = "bank.zip"
UCI_BANK_FILE_NAME = "bank-full.csv"
PUBLIC_FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the local FiiCode data is a public subset of the UCI Bank Marketing dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional override for the local data directory.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=repo_root() / "outputs" / "logs" / "audits" / "bank_marketing_public_overlap.json",
        help="Where to save the JSON audit report.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=UCI_DATASET_ZIP_URL,
        help="Public source zip URL to audit against.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout in seconds for the public source.",
    )
    return parser.parse_args()


def _normalize(value: str) -> str:
    return value.strip().strip('"')


def _feature_key(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    return tuple(_normalize(row[column]) for column in columns)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_public_rows(source_url: str, timeout: int) -> list[dict[str, str]]:
    with urlopen(source_url, timeout=timeout) as response:
        outer_archive = zipfile.ZipFile(io.BytesIO(response.read()))

    if UCI_INNER_ARCHIVE_NAME not in outer_archive.namelist():
        available = ", ".join(outer_archive.namelist())
        raise FileNotFoundError(f"Could not find {UCI_INNER_ARCHIVE_NAME} in public archive. Found: {available}")

    inner_archive = zipfile.ZipFile(io.BytesIO(outer_archive.read(UCI_INNER_ARCHIVE_NAME)))
    if UCI_BANK_FILE_NAME not in inner_archive.namelist():
        available = ", ".join(inner_archive.namelist())
        raise FileNotFoundError(f"Could not find {UCI_BANK_FILE_NAME} in public archive. Found: {available}")

    with inner_archive.open(UCI_BANK_FILE_NAME) as handle:
        text_handle = io.TextIOWrapper(handle, encoding="utf-8")
        return list(csv.DictReader(text_handle, delimiter=";"))


def _validate_local_schema(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> list[str]:
    if not train_rows or not test_rows:
        raise ValueError("Local train/test data is empty.")

    train_feature_columns = [column for column in train_rows[0].keys() if column not in {ID_COL, TARGET}]
    test_feature_columns = [column for column in test_rows[0].keys() if column != ID_COL]

    if train_feature_columns != test_feature_columns:
        raise ValueError(
            "Train/test feature columns do not match.\n"
            f"Train columns: {train_feature_columns}\n"
            f"Test columns: {test_feature_columns}"
        )

    if train_feature_columns != PUBLIC_FEATURE_COLUMNS:
        raise ValueError(
            "Local feature schema does not match the expected Bank Marketing layout.\n"
            f"Expected: {PUBLIC_FEATURE_COLUMNS}\n"
            f"Found: {train_feature_columns}"
        )

    return train_feature_columns


def build_report(
    *,
    data_dir: Path,
    report_path: Path,
    source_url: str,
    timeout: int,
) -> dict[str, object]:
    train_rows = _load_csv_rows(data_dir / "train.csv")
    test_rows = _load_csv_rows(data_dir / "test.csv")
    feature_columns = _validate_local_schema(train_rows, test_rows)
    public_rows = _load_public_rows(source_url, timeout)

    public_feature_counter = Counter(_feature_key(row, feature_columns) for row in public_rows)
    public_label_by_feature_key: dict[tuple[str, ...], str] = {}
    public_label_conflicts = 0
    for row in public_rows:
        row_key = _feature_key(row, feature_columns)
        public_label = _normalize(row["y"])
        existing = public_label_by_feature_key.get(row_key)
        if existing is not None and existing != public_label:
            public_label_conflicts += 1
        else:
            public_label_by_feature_key[row_key] = public_label

    train_keys = [_feature_key(row, feature_columns) for row in train_rows]
    test_keys = [_feature_key(row, feature_columns) for row in test_rows]

    train_found_in_public = sum(1 for row_key in train_keys if public_feature_counter[row_key] > 0)
    test_found_in_public = sum(1 for row_key in test_keys if public_feature_counter[row_key] > 0)
    train_unique_in_public = sum(1 for row_key in train_keys if public_feature_counter[row_key] == 1)
    test_unique_in_public = sum(1 for row_key in test_keys if public_feature_counter[row_key] == 1)

    train_label_mismatches = 0
    missing_train_matches = 0
    for row in train_rows:
        row_key = _feature_key(row, feature_columns)
        public_label = public_label_by_feature_key.get(row_key)
        if public_label is None:
            missing_train_matches += 1
            continue
        mapped_public_label = "1" if public_label == "yes" else "0"
        if _normalize(row[TARGET]) != mapped_public_label:
            train_label_mismatches += 1

    # The goal is to prove recoverability without ever exporting recovered hidden labels.
    test_rows_recoverable = sum(1 for row_key in test_keys if row_key in public_label_by_feature_key)

    exact_public_subset = (
        train_found_in_public == len(train_rows)
        and test_found_in_public == len(test_rows)
        and train_unique_in_public == len(train_rows)
        and test_unique_in_public == len(test_rows)
        and train_label_mismatches == 0
        and missing_train_matches == 0
        and public_label_conflicts == 0
    )

    report = {
        "audit_name": "bank_marketing_public_overlap",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "local_data": {
            "data_dir": str(data_dir.resolve()),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "feature_columns": feature_columns,
            "target_column": TARGET,
        },
        "public_source": {
            "dataset_name": "UCI Bank Marketing",
            "dataset_page_url": UCI_DATASET_PAGE_URL,
            "download_url": source_url,
            "inner_archive_name": UCI_INNER_ARCHIVE_NAME,
            "data_file_name": UCI_BANK_FILE_NAME,
            "public_rows": len(public_rows),
            "public_target_column": "y",
        },
        "mapping": {
            "public_target_yes": "1",
            "public_target_no": "0",
            "local_target_name": TARGET,
        },
        "findings": {
            "train_rows_found_in_public": train_found_in_public,
            "test_rows_found_in_public": test_found_in_public,
            "train_rows_unique_by_features_in_public": train_unique_in_public,
            "test_rows_unique_by_features_in_public": test_unique_in_public,
            "train_label_mismatches_after_mapping": train_label_mismatches,
            "missing_train_public_matches": missing_train_matches,
            "public_feature_label_conflicts": public_label_conflicts,
            "test_rows_recoverable_without_model": test_rows_recoverable,
            "exact_public_subset": exact_public_subset,
        },
        "conclusion": (
            "Local train/test rows are exact feature-level subsets of the public UCI Bank Marketing dataset, "
            "with train labels matching after yes/no -> 1/0 mapping. The audit intentionally does not emit "
            "recovered test labels or any submission-ready artifact."
            if exact_public_subset
            else "The audit found only a partial or inconsistent overlap. See the findings block for details."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    report_path = args.report_path if args.report_path.is_absolute() else (repo_root() / args.report_path).resolve()

    report = build_report(
        data_dir=data_dir,
        report_path=report_path,
        source_url=args.source_url,
        timeout=args.timeout,
    )

    findings = report["findings"]
    print(f"Saved audit report to: {report_path}")
    print(
        "Train rows found in public source: "
        f"{findings['train_rows_found_in_public']}/{report['local_data']['train_rows']}"
    )
    print(
        "Test rows found in public source: "
        f"{findings['test_rows_found_in_public']}/{report['local_data']['test_rows']}"
    )
    print(f"Train label mismatches after mapping: {findings['train_label_mismatches_after_mapping']}")
    print(
        "Test rows uniquely identifiable in public source: "
        f"{findings['test_rows_unique_by_features_in_public']}/{report['local_data']['test_rows']}"
    )
    print(f"Exact public subset: {findings['exact_public_subset']}")


if __name__ == "__main__":
    main()
