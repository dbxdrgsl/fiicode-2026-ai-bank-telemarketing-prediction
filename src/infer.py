from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ensure_output_dirs, load_experiment_config, repo_root, resolve_experiment_paths
from src.features import ID_COL, TARGET, prepare_data, resolve_data_dir
from src.modeling import compute_train_sample_weights, materialize_params, run_model_cv


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild a submission from an experiment summary.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "experiments" / "exp001_baseline.yaml",
        help="Path to experiment YAML.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to best_run_summary.json. Defaults to outputs/logs/<experiment>/best_run_summary.json.",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Optional override for submission output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    paths = resolve_experiment_paths(config)
    ensure_output_dirs(paths)

    summary_path = args.summary or paths.summary_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    data_dir = resolve_data_dir(Path(summary.get("data_dir")) if summary.get("data_dir") else None)
    prepared = prepare_data(
        data_dir,
        feature_set=config.feature_set,
        drop_columns=config.drop_columns,
    )
    sample_weight = compute_train_sample_weights(
        config.train_weight_mode,
        prepared.x,
        prepared.x_test,
        prepared.categorical_columns,
        thread_count=config.thread_count,
        weight_params=config.train_weight_params,
    )

    best_model_params = materialize_params(config.model_family, summary["best_params"], config.thread_count)
    final_cv = run_model_cv(
        model_family=config.model_family,
        x=prepared.x,
        y=prepared.y,
        x_test=prepared.x_test,
        categorical_columns=prepared.categorical_columns,
        seeds=config.final_seeds,
        n_splits=config.final_folds,
        model_params=best_model_params,
        early_stop=config.early_stop,
        use_class_weight=config.with_class_weight,
        sample_weight=sample_weight,
    )

    output_path = args.output_path or paths.submission_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = prepared.test_raw[[ID_COL]].copy()
    submission[TARGET] = final_cv["test"]
    submission.to_csv(output_path, index=False)

    print(f"Submission rebuilt at: {output_path}")


if __name__ == "__main__":
    main()
