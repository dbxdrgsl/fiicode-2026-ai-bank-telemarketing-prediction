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

FeatureSet = Literal[
    "focused",
    "blend",
    "blend_buckets",
    "blend_buckets_rare_frequency",
    "blend_buckets_finance_interactions",
    "blend_buckets_contact_interactions",
    "blend_buckets_state_crosses",
    "blend_buckets_catfreq",
    "blend_buckets_target_enc",
]

RARE_FREQUENCY_COLUMNS = [
    "job",
    "education",
    "contact",
    "month",
    "poutcome",
    "age_bucket",
    "job_education",
    "job_marital",
    "contact_month",
    "poutcome_month",
    "loan_default",
    "history_state",
    "contact_day_bucket",
    "month_day_bucket",
]


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


def _apply_shared_rare_frequency_features(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame | None,
    *,
    min_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train_fe = train_fe.copy()
    test_fe = test_fe.copy() if test_fe is not None else None

    frames = [frame for frame in (train_fe, test_fe) if frame is not None]
    combined = pd.concat(frames, axis=0, ignore_index=True)

    for column in RARE_FREQUENCY_COLUMNS:
        if column not in train_fe.columns:
            continue

        values = combined[column].fillna("missing").astype(str)
        counts = values.value_counts(dropna=False)
        frequencies = counts / len(values)

        def collapse(frame: pd.DataFrame) -> pd.Series:
            raw = frame[column].fillna("missing").astype(str)
            return raw.where(raw.map(counts).fillna(0) >= min_count, "rare")

        train_collapsed = collapse(train_fe)
        train_fe[column] = train_collapsed
        train_fe[f"{column}_freq"] = train_collapsed.map(frequencies).fillna(0.0).astype(np.float32)
        train_fe[f"{column}_count"] = train_collapsed.map(counts).fillna(0).astype(np.int32)

        if test_fe is not None:
            test_collapsed = collapse(test_fe)
            test_fe[column] = test_collapsed
            test_fe[f"{column}_freq"] = test_collapsed.map(frequencies).fillna(0.0).astype(np.float32)
            test_fe[f"{column}_count"] = test_collapsed.map(counts).fillna(0).astype(np.int32)

    return train_fe, test_fe


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


def _add_blend_bucket_rare_frequency_features(data: pd.DataFrame) -> pd.DataFrame:
    return _add_blend_bucket_features(data)


def _add_blend_bucket_finance_interactions(data: pd.DataFrame) -> pd.DataFrame:
    data = _add_blend_bucket_features(data)

    contacts_log1p = np.log1p(data["contacts_total"].clip(lower=0))

    data["balance_signed_x_duration_log1p"] = data["balance_signed_log1p"] * data["duration_log1p"]
    data["balance_signed_x_contacts_log1p"] = data["balance_signed_log1p"] * contacts_log1p
    data["balance_signed_x_previous_log1p"] = data["balance_signed_log1p"] * data["previous_log1p"]
    data["balance_abs_per_contact"] = data["balance_abs_log1p"] / (data["contacts_total"] + 1)
    data["balance_signed_per_age"] = data["balance_signed_log1p"] / (data["age"] + 1)
    data["age_x_duration_log1p"] = data["age"] * data["duration_log1p"]
    data["balance_negative_x_has_any_loan"] = data["balance_negative"] * data["has_any_loan"]
    data["balance_nonpositive_x_is_default"] = data["balance_nonpositive"] * data["is_default"]

    return data


def _add_blend_bucket_contact_interactions(data: pd.DataFrame) -> pd.DataFrame:
    data = _add_blend_bucket_features(data)

    data["contacts_total_log1p"] = np.log1p(data["contacts_total"].clip(lower=0))
    data["duration_x_previous_log1p"] = data["duration_log1p"] * data["previous_log1p"]
    data["duration_x_pdays_log1p"] = data["duration_log1p"] * data["pdays_log1p"]
    data["campaign_x_pdays_log1p"] = data["campaign_log1p"] * data["pdays_log1p"]
    data["duration_per_contact_total"] = data["duration"] / (data["contacts_total"] + 1)
    data["previous_share_total_contacts"] = data["previous"] / (data["contacts_total"] + 1)
    data["campaign_share_total_contacts"] = data["campaign"] / (data["contacts_total"] + 1)
    data["contacted_before_x_previous_log1p"] = data["was_contacted_before"] * data["previous_log1p"]
    data["duration_log1p_x_was_contacted_before"] = data["duration_log1p"] * data["was_contacted_before"]

    return data


def _add_blend_bucket_state_crosses(data: pd.DataFrame) -> pd.DataFrame:
    data = _add_blend_bucket_features(data)

    data["age_bucket_contact"] = data["age_bucket"].astype(str) + "__" + data["contact"].astype(str)
    data["age_bucket_history_state"] = data["age_bucket"].astype(str) + "__" + data["history_state"].astype(str)
    data["loan_housing"] = data["loan"].astype(str) + "__" + data["housing"].astype(str)
    data["contact_history_state"] = data["contact"].astype(str) + "__" + data["history_state"].astype(str)
    data["month_pdays_bucket"] = data["month"].astype(str) + "__" + data["pdays_bucket"].astype(str)
    data["job_contact"] = data["job"].astype(str) + "__" + data["contact"].astype(str)
    data["education_contact"] = data["education"].astype(str) + "__" + data["contact"].astype(str)
    data["housing_default"] = data["housing"].astype(str) + "__" + data["default"].astype(str)
    data["marital_loan"] = data["marital"].astype(str) + "__" + data["loan"].astype(str)

    return data


TARGET_ENCODING_COLUMNS = [
    "job",
    "education",
    "contact",
    "month",
    "poutcome",
    "marital",
    "housing",
    "loan",
    "default",
    "age_bucket",
    "campaign_bucket",
    "pdays_bucket",
    "duration_bucket",
    "history_state",
    "job_education",
    "contact_month",
    "poutcome_month",
]


CATFREQ_COLUMNS = [
    "job",
    "education",
    "contact",
    "month",
    "poutcome",
    "marital",
    "age_bucket",
    "campaign_bucket",
    "pdays_bucket",
    "duration_bucket",
    "history_state",
    "job_education",
    "contact_month",
    "poutcome_month",
]


def _apply_category_frequency_features(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add frequency and count features for categorical columns (no target leakage)."""
    train_fe = train_fe.copy()
    test_fe = test_fe.copy() if test_fe is not None else None
    
    # Combine train and test for computing global frequencies
    frames = [frame for frame in (train_fe, test_fe) if frame is not None]
    combined = pd.concat(frames, axis=0, ignore_index=True)
    n_combined = len(combined)
    
    for col in CATFREQ_COLUMNS:
        if col not in train_fe.columns:
            continue
        
        col_values = combined[col].fillna("missing").astype(str)
        counts = col_values.value_counts(dropna=False)
        frequencies = counts / n_combined
        
        # Also compute train-specific frequencies for potential drift features
        train_col = train_fe[col].fillna("missing").astype(str)
        
        train_fe[f"{col}_freq"] = train_col.map(frequencies).fillna(0.0).astype(np.float32)
        train_fe[f"{col}_count"] = train_col.map(counts).fillna(0).astype(np.int32)
        train_fe[f"{col}_log_count"] = np.log1p(train_fe[f"{col}_count"]).astype(np.float32)
        
        if test_fe is not None:
            test_col = test_fe[col].fillna("missing").astype(str)
            test_fe[f"{col}_freq"] = test_col.map(frequencies).fillna(0.0).astype(np.float32)
            test_fe[f"{col}_count"] = test_col.map(counts).fillna(0).astype(np.int32)
            test_fe[f"{col}_log_count"] = np.log1p(test_fe[f"{col}_count"]).astype(np.float32)
    
    return train_fe, test_fe


def _add_blend_bucket_catfreq_features(data: pd.DataFrame) -> pd.DataFrame:
    """Base feature builder for category frequency set - applied in prepare_data."""
    return _add_blend_bucket_features(data)


def _add_blend_bucket_target_enc_features(data: pd.DataFrame) -> pd.DataFrame:
    """Base feature builder for target encoding set - applied in prepare_data."""
    return _add_blend_bucket_features(data)


def _apply_target_encoding(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame | None,
    y_train: pd.Series,
    *,
    smoothing: float = 10.0,
    noise_level: float = 0.01,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Apply target encoding with smoothing to prevent overfitting.
    
    Uses: encoded = (count * mean + global_mean * smoothing) / (count + smoothing)
    This shrinks rare categories toward the global mean.
    """
    rng = np.random.default_rng(seed)
    train_fe = train_fe.copy()
    test_fe = test_fe.copy() if test_fe is not None else None
    
    global_mean = y_train.mean()
    
    for col in TARGET_ENCODING_COLUMNS:
        if col not in train_fe.columns:
            continue
        
        train_col = train_fe[col].fillna("missing").astype(str)
        
        # Compute stats per category
        df_stats = pd.DataFrame({"cat": train_col, "target": y_train})
        agg = df_stats.groupby("cat")["target"].agg(["mean", "count"])
        
        # Smoothed encoding: shrink toward global mean
        smoothed = (agg["count"] * agg["mean"] + global_mean * smoothing) / (agg["count"] + smoothing)
        
        # Apply to train with small noise to reduce overfitting
        train_encoded = train_col.map(smoothed).fillna(global_mean).astype(np.float32)
        train_encoded += rng.normal(0, noise_level, len(train_encoded)).astype(np.float32)
        train_fe[f"{col}_target_enc"] = train_encoded
        
        # Apply to test (no noise)
        if test_fe is not None:
            test_col = test_fe[col].fillna("missing").astype(str)
            test_fe[f"{col}_target_enc"] = test_col.map(smoothed).fillna(global_mean).astype(np.float32)
    
    return train_fe, test_fe


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
        "blend_buckets_rare_frequency": _add_blend_bucket_rare_frequency_features,
        "blend_buckets_finance_interactions": _add_blend_bucket_finance_interactions,
        "blend_buckets_contact_interactions": _add_blend_bucket_contact_interactions,
        "blend_buckets_state_crosses": _add_blend_bucket_state_crosses,
        "blend_buckets_catfreq": _add_blend_bucket_catfreq_features,
        "blend_buckets_target_enc": _add_blend_bucket_target_enc_features,
    }
    add_features = builders[feature_set]

    train_fe = add_features(train_raw)
    test_fe = add_features(test_raw) if test_raw is not None else None

    if feature_set == "blend_buckets_rare_frequency":
        train_fe, test_fe = _apply_shared_rare_frequency_features(train_fe, test_fe)
    
    if feature_set == "blend_buckets_catfreq":
        train_fe, test_fe = _apply_category_frequency_features(train_fe, test_fe)
    
    # Target encoding needs y, applied after basic features
    y_for_te = train_fe[TARGET].astype(int)
    if feature_set == "blend_buckets_target_enc":
        train_fe, test_fe = _apply_target_encoding(train_fe, test_fe, y_for_te)

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
