from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_COMPETITION_SLUG, repo_root
from src.features import resolve_data_dir
from src.tracking import append_leaderboard_journal, load_summary, update_latest_experiment_scores


def validate_submission(submission_path: Path, data_dir: Path) -> list[str]:
    errors: list[str] = []
    submission = pd.read_csv(submission_path)
    expected_test = pd.read_csv(data_dir / "test.csv")

    if list(submission.columns) != ["id", "Subscribed"]:
        errors.append("Submission columns must be exactly: id, Subscribed")

    if len(submission) != len(expected_test):
        errors.append(f"Row count mismatch: got {len(submission)}, expected {len(expected_test)}")

    if "id" in submission.columns and not submission["id"].equals(expected_test["id"]):
        errors.append("id column does not match test.csv ordering")

    if "Subscribed" in submission.columns:
        if submission["Subscribed"].isna().any():
            errors.append("Subscribed contains NaN values")
        if (~submission["Subscribed"].map(lambda value: math.isfinite(float(value)))).any():
            errors.append("Subscribed contains non-finite values")
        if ((submission["Subscribed"] < 0) | (submission["Subscribed"] > 1)).any():
            errors.append("Subscribed contains values outside [0, 1]")

    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission and optionally submit it.")
    parser.add_argument("--submission", type=Path, required=True, help="Path to submission CSV.")
    parser.add_argument("--competition", type=str, default=DEFAULT_COMPETITION_SLUG, help="Competition slug.")
    parser.add_argument("--message", type=str, default="FiiCode submission", help="Submission message.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional override for data directory.")
    parser.add_argument("--experiment", type=str, default=None, help="Optional experiment name override.")
    parser.add_argument("--public-lb", type=float, default=None, help="Optional public leaderboard score to record.")
    parser.add_argument("--private-lb", type=float, default=None, help="Optional private leaderboard score to record.")
    parser.add_argument("--notes", type=str, default="", help="Optional note for the leaderboard journal.")
    parser.add_argument(
        "--run-kaggle-cli",
        action="store_true",
        default=False,
        help="Actually run the kaggle competitions submit command after validation.",
    )
    return parser.parse_args()


def infer_experiment_name(submission_path: Path, experiment: str | None) -> str:
    if experiment:
        return experiment
    return submission_path.parent.name


def resolve_summary_path(experiment_name: str) -> Path:
    return repo_root() / "outputs" / "logs" / experiment_name / "best_run_summary.json"


def main():
    args = parse_args()
    submission_path = args.submission if args.submission.is_absolute() else (repo_root() / args.submission).resolve()
    data_dir = resolve_data_dir(args.data_dir)
    errors = validate_submission(submission_path, data_dir)

    if errors:
        print("Submission validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    experiment_name = infer_experiment_name(submission_path, args.experiment)
    summary_path = resolve_summary_path(experiment_name)
    local_final_auc = None
    if summary_path.exists():
        summary = load_summary(summary_path)
        local_final_auc = summary.get("final_auc")

    kaggle_executable = shutil.which("kaggle")
    if kaggle_executable is None:
        candidate = Path(sys.executable).with_name("kaggle")
        kaggle_executable = str(candidate) if candidate.exists() else "kaggle"

    command = [
        kaggle_executable,
        "competitions",
        "submit",
        "-c",
        args.competition,
        "-f",
        str(submission_path),
        "-m",
        args.message,
    ]
    print("Submission validation passed.")
    print("Suggested command:")
    print(" ".join(command))

    status = "validated"
    if args.run_kaggle_cli:
        subprocess.run(command, check=True, cwd=repo_root())
        status = "submitted"
    if args.public_lb is not None or args.private_lb is not None:
        status = "scored"

    append_leaderboard_journal(
        repo_root() / "outputs" / "logs" / "leaderboard_journal.csv",
        experiment=experiment_name,
        competition=args.competition,
        submission_path=submission_path,
        message=args.message,
        status=status,
        local_final_auc=None if local_final_auc is None else float(local_final_auc),
        public_lb=args.public_lb,
        private_lb=args.private_lb,
        notes=args.notes,
    )
    if args.public_lb is not None or args.private_lb is not None:
        update_latest_experiment_scores(
            repo_root() / "outputs" / "logs" / "experiment_runs.csv",
            experiment=experiment_name,
            public_lb=args.public_lb,
            private_lb=args.private_lb,
        )
    print(f"Leaderboard journal updated for experiment: {experiment_name}")


if __name__ == "__main__":
    main()
