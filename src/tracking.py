from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import ID_COL, TARGET

EXPERIMENT_RUN_HEADERS = [
    "timestamp_utc",
    "experiment",
    "model_family",
    "feature_set",
    "with_class_weight",
    "search_auc",
    "final_auc",
    "public_lb",
    "private_lb",
    "oof_corr_to_best",
    "submission_path",
    "summary_path",
    "notes",
]

LEADERBOARD_JOURNAL_HEADERS = [
    "timestamp_utc",
    "experiment",
    "competition",
    "submission_path",
    "message",
    "status",
    "local_final_auc",
    "public_lb",
    "private_lb",
    "notes",
]


def _coerce_existing_rows(path: Path, headers: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{header: (row.get(header) or "") for header in headers} for row in reader]


def append_csv_row(path: Path, headers: list[str], row: dict[str, object]) -> None:
    rows = _coerce_existing_rows(path, headers)
    rows.append({header: "" if row.get(header) is None else str(row.get(header)) for header in headers})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def update_latest_experiment_scores(
    path: Path,
    *,
    experiment: str,
    public_lb: float | None = None,
    private_lb: float | None = None,
) -> bool:
    rows = _coerce_existing_rows(path, EXPERIMENT_RUN_HEADERS)
    for row in reversed(rows):
        if row["experiment"] != experiment:
            continue
        if public_lb is not None:
            row["public_lb"] = f"{public_lb:.6f}"
        if private_lb is not None:
            row["private_lb"] = f"{private_lb:.6f}"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_RUN_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        return True
    return False


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_oof_frame(oof_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(oof_path)
    prediction_columns = [column for column in frame.columns if column not in {ID_COL, "y_true"}]
    if not prediction_columns:
        raise ValueError(f"Could not find OOF prediction column in {oof_path}")
    frame = frame[[ID_COL, "y_true", prediction_columns[-1]]].copy()
    frame.columns = [ID_COL, "y_true", "prediction"]
    return frame


def load_submission_frame(submission_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(submission_path)
    return frame[[ID_COL, TARGET]].copy()


def compute_corr_to_best(
    experiment_runs_path: Path,
    *,
    current_experiment: str,
    current_oof_path: Path,
) -> float | None:
    if not experiment_runs_path.exists() or not current_oof_path.exists():
        return None

    rows = _coerce_existing_rows(experiment_runs_path, EXPERIMENT_RUN_HEADERS)
    eligible_rows = []
    for row in rows:
        if row["experiment"] == current_experiment or not row["summary_path"] or not row["final_auc"]:
            continue
        try:
            final_auc = float(row["final_auc"])
        except ValueError:
            continue
        eligible_rows.append((final_auc, Path(row["summary_path"])))

    if not eligible_rows:
        return None

    _, best_summary_path = max(eligible_rows, key=lambda item: item[0])
    if not best_summary_path.exists():
        return None

    best_summary = load_summary(best_summary_path)
    best_oof_path = Path(best_summary["paths"]["oof"])
    if not best_oof_path.exists():
        return None

    current_frame = load_oof_frame(current_oof_path)
    best_frame = load_oof_frame(best_oof_path)
    merged = current_frame.merge(best_frame, on=[ID_COL, "y_true"], suffixes=("_current", "_best"))
    if merged.empty:
        return None
    return float(np.corrcoef(merged["prediction_current"], merged["prediction_best"])[0, 1])


def append_experiment_run(
    path: Path,
    *,
    experiment: str,
    model_family: str,
    feature_set: str,
    with_class_weight: bool,
    search_auc: float | None,
    final_auc: float,
    oof_corr_to_best: float | None,
    submission_path: Path,
    summary_path: Path,
    notes: str,
    public_lb: float | None = None,
    private_lb: float | None = None,
) -> None:
    append_csv_row(
        path,
        EXPERIMENT_RUN_HEADERS,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment,
            "model_family": model_family,
            "feature_set": feature_set,
            "with_class_weight": with_class_weight,
            "search_auc": "" if search_auc is None else f"{search_auc:.6f}",
            "final_auc": f"{final_auc:.6f}",
            "public_lb": "" if public_lb is None else f"{public_lb:.6f}",
            "private_lb": "" if private_lb is None else f"{private_lb:.6f}",
            "oof_corr_to_best": "" if oof_corr_to_best is None else f"{oof_corr_to_best:.6f}",
            "submission_path": submission_path,
            "summary_path": summary_path,
            "notes": notes,
        },
    )


def append_leaderboard_journal(
    path: Path,
    *,
    experiment: str,
    competition: str,
    submission_path: Path,
    message: str,
    status: str,
    local_final_auc: float | None,
    public_lb: float | None = None,
    private_lb: float | None = None,
    notes: str = "",
) -> None:
    append_csv_row(
        path,
        LEADERBOARD_JOURNAL_HEADERS,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment,
            "competition": competition,
            "submission_path": submission_path,
            "message": message,
            "status": status,
            "local_final_auc": "" if local_final_auc is None else f"{local_final_auc:.6f}",
            "public_lb": "" if public_lb is None else f"{public_lb:.6f}",
            "private_lb": "" if private_lb is None else f"{private_lb:.6f}",
            "notes": notes,
        },
    )
