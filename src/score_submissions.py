from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import repo_root
from src.features import ID_COL, TARGET, resolve_data_dir
from src.submit import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score submission.csv files against a labeled reference and optionally "
            "reconcile the local score against scored Kaggle history rows."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to a labeled reference CSV containing id and target columns.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        nargs="*",
        default=[],
        help="One or more submission.csv files to score.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        default=False,
        help="Also score every scored row in outputs/logs/leaderboard_journal.csv against the same reference.",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=repo_root() / "outputs" / "logs" / "leaderboard_journal.csv",
        help="Leaderboard journal path used when --history is enabled.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional override for the competition data directory.")
    parser.add_argument(
        "--reference-id-column",
        type=str,
        default=ID_COL,
        help=f"Reference CSV id column name. Default: {ID_COL}.",
    )
    parser.add_argument(
        "--reference-target-column",
        type=str,
        default=TARGET,
        help=f"Reference CSV target column name. Default: {TARGET}.",
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default=None,
        help="Optional positive-class label for non-numeric reference targets, e.g. yes.",
    )
    parser.add_argument(
        "--negative-label",
        type=str,
        default=None,
        help="Optional negative-class label for non-numeric reference targets, e.g. no.",
    )
    parser.add_argument(
        "--score-decimals",
        type=int,
        default=5,
        help="Leaderboard decimals to use when deciding whether a local score matches Kaggle exactly.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip submission schema validation against data/raw/test.csv before scoring.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to save the full scoring report as JSON.",
    )
    args = parser.parse_args()
    if not args.submission and not args.history:
        parser.error("Provide at least one --submission path or enable --history.")
    return args


def resolve_any_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ValueError("Encountered an empty path.")

    wsl_match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        suffix = wsl_match.group(2).replace("/", "\\")
        return Path(f"{drive}:\\{suffix}").resolve()

    path = Path(text)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def normalize_reference_target(
    series: pd.Series,
    *,
    positive_label: str | None,
    negative_label: str | None,
) -> pd.Series:
    if positive_label is None and negative_label is None:
        values = pd.to_numeric(series, errors="coerce")
    else:
        positive = "1" if positive_label is None else str(positive_label).strip()
        negative = "0" if negative_label is None else str(negative_label).strip()
        mapping = {positive: 1.0, negative: 0.0}
        values = series.astype(str).str.strip().map(mapping)

    if values.isna().any():
        raise ValueError("Reference target contains unmapped or non-numeric values.")
    unique_values = set(values.unique().tolist())
    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(f"Reference target must be binary after normalization; found {sorted(unique_values)}")
    return values.astype(float)


def load_reference_frame(
    reference_path: Path,
    *,
    id_column: str,
    target_column: str,
    positive_label: str | None,
    negative_label: str | None,
) -> pd.DataFrame:
    frame = pd.read_csv(reference_path)
    required = [column for column in (id_column, target_column) if column not in frame.columns]
    if required:
        raise ValueError(f"Reference file is missing required columns: {required}")
    reference = frame[[id_column, target_column]].copy()
    reference.columns = [ID_COL, "reference_target"]
    if reference[ID_COL].duplicated().any():
        raise ValueError(f"Reference file contains duplicate {ID_COL} values.")
    reference["reference_target"] = normalize_reference_target(
        reference["reference_target"],
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return reference


def format_score(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def score_submission(
    submission_path: Path,
    *,
    reference: pd.DataFrame,
    data_dir: Path,
    skip_validation: bool,
) -> dict[str, object]:
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    if not skip_validation:
        errors = validate_submission(submission_path, data_dir)
        if errors:
            raise ValueError(f"Submission validation failed for {submission_path}: {'; '.join(errors)}")

    submission = pd.read_csv(submission_path)
    required = [column for column in (ID_COL, TARGET) if column not in submission.columns]
    if required:
        raise ValueError(f"Submission file is missing required columns: {required}")
    submission = submission[[ID_COL, TARGET]].copy()
    if submission[ID_COL].duplicated().any():
        raise ValueError(f"Submission file contains duplicate {ID_COL} values: {submission_path}")

    submission[TARGET] = pd.to_numeric(submission[TARGET], errors="raise")
    merged = reference.merge(submission, on=ID_COL, how="left")
    missing_reference_ids = int(merged[TARGET].isna().sum())
    if missing_reference_ids:
        raise ValueError(
            f"Submission {submission_path} is missing {missing_reference_ids} reference ids and cannot be scored."
        )

    extra_submission_ids = int(submission[~submission[ID_COL].isin(reference[ID_COL])].shape[0])
    local_score = float(roc_auc_score(merged["reference_target"], merged[TARGET]))
    return {
        "submission_path": str(submission_path),
        "local_score": local_score,
        "reference_rows": int(len(reference)),
        "submission_rows": int(len(submission)),
        "extra_submission_ids": extra_submission_ids,
        "missing_reference_ids": missing_reference_ids,
    }


def load_history_rows(journal_path: Path) -> pd.DataFrame:
    if not journal_path.exists():
        raise FileNotFoundError(f"Leaderboard journal not found: {journal_path}")
    history = pd.read_csv(journal_path, dtype=str).fillna("")
    history = history[history["public_lb"].str.strip() != ""].copy()
    history["public_lb"] = history["public_lb"].astype(float)
    return history


def evaluate_history(
    journal_path: Path,
    *,
    reference: pd.DataFrame,
    data_dir: Path,
    skip_validation: bool,
    score_decimals: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    history = load_history_rows(journal_path)
    for row in history.to_dict("records"):
        raw_submission_path = row["submission_path"]
        result: dict[str, object] = {
            "mode": "history",
            "experiment": row["experiment"],
            "message": row["message"],
            "journal_submission_path": raw_submission_path,
            "kaggle_public_lb": float(row["public_lb"]),
            "status": row["status"],
        }
        try:
            resolved_path = resolve_any_path(raw_submission_path)
            result["submission_path"] = str(resolved_path)
            scored = score_submission(
                resolved_path,
                reference=reference,
                data_dir=data_dir,
                skip_validation=skip_validation,
            )
            result.update(scored)
            kaggle_score = float(row["public_lb"])
            local_score = float(scored["local_score"])
            result["abs_diff"] = abs(local_score - kaggle_score)
            result["exact_match"] = format_score(local_score, score_decimals) == format_score(
                kaggle_score, score_decimals
            )
        except Exception as exc:  # noqa: BLE001 - report every failed history row in the audit output.
            result["submission_path"] = raw_submission_path
            result["error"] = str(exc)
            result["exact_match"] = False
        results.append(result)
    return results


def evaluate_explicit_submissions(
    submission_paths: list[Path],
    *,
    reference: pd.DataFrame,
    data_dir: Path,
    skip_validation: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for raw_path in submission_paths:
        submission_path = resolve_any_path(raw_path)
        result = {"mode": "explicit"}
        try:
            result.update(
                score_submission(
                    submission_path,
                    reference=reference,
                    data_dir=data_dir,
                    skip_validation=skip_validation,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep scoring the remaining submissions.
            result["submission_path"] = str(submission_path)
            result["error"] = str(exc)
        results.append(result)
    return results


def print_results(title: str, rows: list[dict[str, object]], *, score_decimals: int) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("-" * len(title))
    display_rows: list[dict[str, object]] = []
    for row in rows:
        display_row: dict[str, object] = {
            "submission_path": row.get("submission_path", ""),
            "local_score": (
                "" if row.get("local_score") is None else format_score(float(row["local_score"]), score_decimals + 2)
            ),
            "kaggle_public_lb": (
                ""
                if row.get("kaggle_public_lb") is None
                else format_score(float(row["kaggle_public_lb"]), score_decimals)
            ),
            "abs_diff": "" if row.get("abs_diff") is None else f"{float(row['abs_diff']):.8f}",
            "exact_match": row.get("exact_match", ""),
            "reference_rows": row.get("reference_rows", ""),
            "extra_submission_ids": row.get("extra_submission_ids", ""),
            "error": row.get("error", ""),
        }
        if row.get("experiment"):
            display_row["experiment"] = row["experiment"]
        if row.get("message"):
            display_row["message"] = row["message"]
        display_rows.append(display_row)
    print(pd.DataFrame(display_rows).to_string(index=False))


def print_history_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    successful = [row for row in rows if row.get("local_score") is not None]
    exact = [row for row in successful if row.get("exact_match")]
    failed = [row for row in rows if row.get("error")]
    print("\nHistory Summary")
    print("---------------")
    print(f"Scored history rows: {len(rows)}")
    print(f"Successful comparisons: {len(successful)}")
    print(f"Exact matches: {len(exact)}")
    print(f"Failed comparisons: {len(failed)}")
    if successful:
        diffs = [float(row["abs_diff"]) for row in successful if row.get("abs_diff") is not None]
        print(f"Mean absolute diff: {sum(diffs) / len(diffs):.8f}")
        print(f"Max absolute diff:  {max(diffs):.8f}")


def main() -> None:
    args = parse_args()
    reference_path = resolve_any_path(args.reference)
    journal_path = resolve_any_path(args.journal)
    report_json_path = None if args.report_json is None else resolve_any_path(args.report_json)
    data_dir = resolve_data_dir(args.data_dir)
    reference = load_reference_frame(
        reference_path,
        id_column=args.reference_id_column,
        target_column=args.reference_target_column,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    explicit_results = evaluate_explicit_submissions(
        args.submission,
        reference=reference,
        data_dir=data_dir,
        skip_validation=args.skip_validation,
    )
    history_results = (
        evaluate_history(
            journal_path,
            reference=reference,
            data_dir=data_dir,
            skip_validation=args.skip_validation,
            score_decimals=args.score_decimals,
        )
        if args.history
        else []
    )

    print(f"Reference path: {reference_path}")
    print(f"Reference rows: {len(reference)}")
    print_results("Explicit Submission Scores", explicit_results, score_decimals=args.score_decimals)
    print_results("History Reconciliation", history_results, score_decimals=args.score_decimals)
    print_history_summary(history_results)

    if report_json_path is not None:
        payload = {
            "reference_path": str(reference_path),
            "reference_rows": int(len(reference)),
            "score_decimals": args.score_decimals,
            "explicit_results": explicit_results,
            "history_results": history_results,
        }
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved report: {report_json_path}")


if __name__ == "__main__":
    main()
