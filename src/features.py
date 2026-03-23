from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.config import repo_root

TARGET = "Subscribed"
ID_COL = "id"
DEFAULT_SEARCH_SEEDS = [42]
DEFAULT_FINAL_SEEDS = [42, 2024, 3407]
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

FeatureSet = Literal["focused", "blend", "blend_buckets"]


@dataclass
class PreparedData:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame | None
    x: pd.DataFrame
    y: pd.Series
    x_test: pd.DataFrame | None
    categorical_columns: list[str]


def is_kaggle_runtime() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


def resolve_data_dir(explicit_dir: Path | None = None) -> Path:
    root = repo_root()
    candidates: list[Path] = []

    if explicit_dir is not None:
        candidates.append(explicit_dir)

    candidates.extend(
        [
            root / "data" / "raw",
            root / "data" / "interim",
            root / "data" / "processed",
            root,
            Path.cwd() / "data" / "raw",
            Path.cwd() / "data",
            Path.cwd(),
        ]
    )

    for root_dir in (Path("/kaggle/input"), Path("/kaggle/input/competitions")):
        if not root_dir.exists():
            continue
        candidates.append(root_dir)
        for child in root_dir.iterdir():
            if child.is_dir():
                candidates.append(child)

    seen: set[str] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in ordered_candidates)
    raise FileNotFoundError(
        "Could not find train.csv and test.csv in any candidate directory.\n"
        f"Searched:\n{searched}"
    )


def _string_series(data: pd.DataFrame, column: str, *, lowercase: bool = False) -> pd.Series:
    values = data[column].fillna("missing").astype(str)
    if lowercase:
        values = values.str.lower()
    return values


def _add_focused_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    month = _string_series(data, "month", lowercase=True)
    job = _string_series(data, "job")
    marital = _string_series(data, "marital")
    education = _string_series(data, "education")
    contact = _string_series(data, "contact")
    poutcome = _string_series(data, "poutcome")
    loan = _string_series(data, "loan")
    default = _string_series(data, "default")
    housing = _string_series(data, "housing")

    data["month"] = month
    data["month_num"] = month.map(MONTH_MAP).fillna(0).astype(np.int16)
    data["pdays_clean"] = data["pdays"].replace(-1, 0)
    data["pdays_log1p"] = np.log1p(data["pdays_clean"])
    data["duration_log1p"] = np.log1p(data["duration"].clip(lower=0))

    data["contacts_total"] = data["campaign"] + data["previous"]
    data["duration_per_campaign"] = data["duration"] / (data["campaign"] + 1)
    data["balance_per_age"] = data["balance"] / (data["age"] + 1)
    data["previous_per_campaign"] = data["previous"] / (data["campaign"] + 1)
    data["campaign_x_previous"] = data["campaign"] * data["previous"]
    data["duration_x_campaign"] = data["duration"] * data["campaign"]

    data["has_any_loan"] = ((housing == "yes") | (loan == "yes")).astype(np.int8)
    data["is_default"] = (default == "yes").astype(np.int8)
    data["was_contacted_before"] = (data["pdays"] != -1).astype(np.int8)
    data["is_cellular"] = (contact == "cellular").astype(np.int8)

    data["age_bucket"] = pd.cut(
        data["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)

    data["job_education"] = job + "__" + education
    data["job_marital"] = job + "__" + marital
    data["contact_month"] = contact + "__" + month
    data["poutcome_month"] = poutcome + "__" + month
    data["loan_default"] = loan + "__" + default

    return data


def _add_blend_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    month = _string_series(data, "month", lowercase=True)
    job = _string_series(data, "job")
    marital = _string_series(data, "marital")
    education = _string_series(data, "education")
    contact = _string_series(data, "contact")
    poutcome = _string_series(data, "poutcome")
    loan = _string_series(data, "loan")
    default = _string_series(data, "default")
    housing = _string_series(data, "housing")

    data["month"] = month
    data["month_num"] = month.map(MONTH_MAP).fillna(0).astype(np.int16)
    data["pdays_was_missing"] = (data["pdays"] == -1).astype(np.int8)
    data["pdays_clean"] = data["pdays"].replace(-1, 999)

    data["duration_log1p"] = np.log1p(data["duration"].clip(lower=0))
    data["balance_log1p"] = np.log1p(data["balance"].clip(lower=0))
    data["balance_abs_log1p"] = np.log1p(data["balance"].abs())
    data["campaign_log1p"] = np.log1p(data["campaign"].clip(lower=0))
    data["previous_log1p"] = np.log1p(data["previous"].clip(lower=0))
    data["pdays_log1p"] = np.log1p(data["pdays_clean"])

    data["contacts_total"] = data["campaign"] + data["previous"]
    data["duration_per_campaign"] = data["duration"] / (data["campaign"] + 1)
    data["balance_per_age"] = data["balance"] / (data["age"] + 1)
    data["previous_per_campaign"] = data["previous"] / (data["campaign"] + 1)
    data["campaign_x_previous"] = data["campaign"] * data["previous"]
    data["duration_x_campaign"] = data["duration"] * data["campaign"]

    data["has_any_loan"] = ((housing == "yes") | (loan == "yes")).astype(np.int8)
    data["is_default"] = (default == "yes").astype(np.int8)
    data["was_contacted_before"] = (data["pdays"] != -1).astype(np.int8)
    data["is_cellular"] = (contact == "cellular").astype(np.int8)

    data["age_bucket"] = pd.cut(
        data["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)

    data["job_education"] = job + "__" + education
    data["job_marital"] = job + "__" + marital
    data["contact_month"] = contact + "__" + month
    data["poutcome_month"] = poutcome + "__" + month
    data["loan_default"] = loan + "__" + default

    return data


def _add_blend_bucket_features(data: pd.DataFrame) -> pd.DataFrame:
    data = _add_blend_features(data)

    pdays_source = data["pdays"].replace(-1, 999)

    data["balance_signed_log1p"] = np.sign(data["balance"]) * np.log1p(data["balance"].abs())
    data["balance_negative"] = (data["balance"] < 0).astype(np.int8)
    data["balance_nonpositive"] = (data["balance"] <= 0).astype(np.int8)

    data["campaign_bucket"] = pd.cut(
        data["campaign"],
        bins=[-1, 1, 2, 4, 9, np.inf],
        labels=["1", "2", "3-4", "5-9", "10+"],
    ).astype(str)
    data["previous_bucket"] = pd.cut(
        data["previous"],
        bins=[-1, 0, 1, 3, np.inf],
        labels=["0", "1", "2-3", "4+"],
    ).astype(str)
    data["pdays_bucket"] = pd.cut(
        pdays_source,
        bins=[-1, 7, 30, 90, 365, np.inf],
        labels=["<=1w", "8-30d", "31-90d", "91-365d", "365d+"],
    ).astype(str)
    data.loc[data["pdays"] == -1, "pdays_bucket"] = "never"
    data["duration_bucket"] = pd.cut(
        data["duration"],
        bins=[-1, 60, 120, 240, 480, np.inf],
        labels=["<=1m", "1-2m", "2-4m", "4-8m", "8m+"],
    ).astype(str)
    data["day_bucket"] = pd.cut(
        data["day"],
        bins=[0, 10, 20, 31],
        labels=["early", "mid", "late"],
        include_lowest=True,
    ).astype(str)

    data["contact_day_bucket"] = data["contact"].astype(str) + "__" + data["day_bucket"]
    data["month_day_bucket"] = data["month"].astype(str) + "__" + data["day_bucket"]
    data["history_state"] = np.where(
        data["previous"] > 0,
        data["poutcome"].astype(str) + "__seen",
        "no_previous",
    )

    return data


def prepare_data(
    data_dir: Path,
    *,
    feature_set: FeatureSet,
    drop_columns: Sequence[str] = (),
    load_test: bool = True,
) -> PreparedData:
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv") if load_test else None

    builders = {
        "focused": _add_focused_features,
        "blend": _add_blend_features,
        "blend_buckets": _add_blend_bucket_features,
    }
    add_features = builders[feature_set]

    train_fe = add_features(train_raw)
    test_fe = add_features(test_raw) if test_raw is not None else None

    if drop_columns:
        train_fe = train_fe.drop(columns=list(drop_columns), errors="ignore")
        if test_fe is not None:
            test_fe = test_fe.drop(columns=list(drop_columns), errors="ignore")

    y = train_fe[TARGET].astype(int)
    x = train_fe.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    x_test = test_fe.drop(columns=[ID_COL], errors="ignore").copy() if test_fe is not None else None

    categorical_columns = x.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_columns:
        x[column] = x[column].fillna("missing").astype(str)
        if x_test is not None:
            x_test[column] = x_test[column].fillna("missing").astype(str)

    return PreparedData(
        train_raw=train_raw,
        test_raw=test_raw,
        x=x,
        y=y,
        x_test=x_test,
        categorical_columns=categorical_columns,
    )
