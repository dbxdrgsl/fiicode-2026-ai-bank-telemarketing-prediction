from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ensure_output_dirs, load_experiment_config, repo_root, resolve_experiment_paths
from src.features import prepare_data, resolve_data_dir
from src.modeling import default_ablation_params, run_model_cv


def run_ablation_suite(
    model_family: str,
    x,
    y,
    categorical_columns,
    *,
    n_folds: int,
    seeds: list[int],
    thread_count: int,
    use_class_weight: bool,
) -> dict[str, float]:
    base_params = default_ablation_params(model_family, thread_count)
    tasks = [
        ("A (Full, baseline)", x, use_class_weight),
        ("B (without duration_log1p)", x.drop(columns=["duration_log1p"], errors="ignore"), use_class_weight),
        ("C (without duration_per_campaign)", x.drop(columns=["duration_per_campaign"], errors="ignore"), use_class_weight),
        (
            "D (without both)",
            x.drop(columns=["duration_log1p", "duration_per_campaign"], errors="ignore"),
            use_class_weight,
        ),
        ("E (without class weights)", x, False),
    ]

    results: dict[str, float] = {}
    for index, (name, frame, class_weight_enabled) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {name}")
        local_cats = [column for column in categorical_columns if column in frame.columns]
        cv_output = run_model_cv(
            model_family=model_family,
            x=frame,
            y=y,
            categorical_columns=local_cats,
            seeds=seeds,
            n_splits=n_folds,
            model_params=base_params,
            early_stop=250,
            use_class_weight=class_weight_enabled,
            predict_test=False,
        )
        auc = roc_auc_score(y, cv_output["oof"])
        results[name] = auc
        print(f"    AUC: {auc:.6f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run FiiCode CV ablations from an experiment config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "experiments" / "exp001_baseline.yaml",
        help="Path to experiment YAML.",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--n-seeds", type=int, default=3, help="How many experiment seeds to use.")
    parser.add_argument("--thread-count", type=int, default=None, help="Override model thread count.")
    parser.add_argument(
        "--with-class-weight",
        action="store_true",
        default=True,
        help="Use balanced class weighting (default: True).",
    )
    parser.add_argument(
        "--without-class-weight",
        dest="with_class_weight",
        action="store_false",
        help="Disable class weighting.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Cheap ablation mode: force a single seed for faster directional checks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    paths = resolve_experiment_paths(config)
    ensure_output_dirs(paths)

    data_dir = resolve_data_dir(Path(config.data_dir) if config.data_dir else None)
    prepared = prepare_data(
        data_dir,
        feature_set=config.feature_set,
        drop_columns=config.drop_columns,
        load_test=False,
    )
    n_seeds = 1 if args.quick else args.n_seeds
    seeds = config.final_seeds[:n_seeds]
    thread_count = args.thread_count if args.thread_count is not None else config.thread_count

    print("=" * 70)
    print("FiiCode CV Ablation Suite")
    print("=" * 70)
    print(f"Experiment:   {config.name}")
    print(f"Model family: {config.model_family}")
    print(f"Data dir:     {data_dir}")
    print(f"Feature set:  {config.feature_set}")
    print(f"Drop columns: {config.drop_columns or '[]'}")
    print(f"CV folds:     {args.n_folds}")
    print(f"Seeds:        {seeds}")
    print(f"Thread count: {thread_count}")
    print(f"Class weight: {args.with_class_weight}")
    print(f"Quick mode:   {args.quick}")
    print("=" * 70)

    results = run_ablation_suite(
        config.model_family,
        prepared.x,
        prepared.y,
        prepared.categorical_columns,
        n_folds=args.n_folds,
        seeds=seeds,
        thread_count=thread_count,
        use_class_weight=args.with_class_weight,
    )

    for name, auc in sorted(results.items(), key=lambda item: -item[1]):
        print(f"{name:35s} {auc:.6f}")

    with paths.ablation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "model_family", "ablation", "auc"])
        for name, auc in sorted(results.items(), key=lambda item: item[0]):
            writer.writerow([config.name, config.model_family, name, f"{auc:.6f}"])

    best_name, best_auc = max(results.items(), key=lambda item: item[1])
    print(f"\nBest configuration: {best_name} ({best_auc:.6f})")
    print(f"Ablation results saved to: {paths.ablation_path}")


if __name__ == "__main__":
    main()
