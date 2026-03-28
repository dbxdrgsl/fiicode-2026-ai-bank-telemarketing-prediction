from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ensure_output_dirs, load_experiment_config, repo_root, resolve_experiment_paths
from src.features import ID_COL, TARGET, prepare_data, resolve_data_dir
from src.modeling import compute_train_sample_weights, materialize_params, run_model_cv, suggest_params
from src.tracking import append_experiment_run, compute_corr_to_best

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    TPESampler = None


def default_optuna_storage(summary_dir: Path, study_name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in study_name).strip("._") or "study"
    return f"sqlite:///{(summary_dir / f'{safe_name}.sqlite3').resolve().as_posix()}"


def resolve_optuna_storage(storage: str | None, summary_dir: Path, study_name: str) -> str | None:
    if storage == "memory":
        return None
    if storage:
        return storage
    return default_optuna_storage(summary_dir, study_name)


def resolve_optional_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def load_fixed_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "best_params" in payload:
        return payload["best_params"]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict-like params payload in {path}")
    return payload


def objective_factory(prepared, config, sample_weight):
    def objective(trial):
        cv_output = run_model_cv(
            model_family=config.model_family,
            x=prepared.x,
            y=prepared.y,
            categorical_columns=prepared.categorical_columns,
            seeds=config.search_seeds,
            n_splits=config.search_folds,
            model_params=suggest_params(config.model_family, trial, config.thread_count),
            early_stop=config.early_stop,
            use_class_weight=config.with_class_weight,
            predict_test=False,
            sample_weight=sample_weight,
        )
        return roc_auc_score(prepared.y, cv_output["oof"])

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FiiCode experiment from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "experiments" / "exp001_baseline.yaml",
        help="Path to experiment YAML.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override output root directory.")
    parser.add_argument("--storage", type=str, default=None, help="Override Optuna storage URL.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override target number of Optuna trials.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.output_root is not None:
        config = replace(config, output_root=str(args.output_root))
    if args.n_trials is not None:
        config = replace(config, n_trials=args.n_trials)
    if args.storage is not None:
        config = replace(config, storage=args.storage)

    paths = resolve_experiment_paths(config)
    ensure_output_dirs(paths)
    data_dir = resolve_data_dir(args.data_dir or (Path(config.data_dir) if config.data_dir else None))
    storage = resolve_optuna_storage(config.storage, paths.logs_dir, config.study_name)
    fixed_params_path = resolve_optional_path(config.fixed_params_path)

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
    sample_weight_stats = None
    if sample_weight is not None:
        sample_weight_stats = {
            "min": float(sample_weight.min()),
            "mean": float(sample_weight.mean()),
            "max": float(sample_weight.max()),
            "std": float(sample_weight.std()),
        }

    print("=" * 70)
    print("FiiCode Model Training")
    print("=" * 70)
    print(f"Experiment:           {config.name}")
    print(f"Config path:          {args.config}")
    print(f"Model family:         {config.model_family}")
    print(f"Feature set:          {config.feature_set}")
    print(f"Data dir:             {data_dir}")
    print(f"Output root:          {paths.output_root}")
    print(f"Study name:           {config.study_name}")
    print(f"Study storage:        {storage or 'in-memory'}")
    print(f"Requested trials:     {config.n_trials}")
    print(f"Search CV:            {config.search_folds}-fold, seeds={config.search_seeds}")
    print(f"Final CV:             {config.final_folds}-fold, seeds={config.final_seeds}")
    print(f"Thread count:         {config.thread_count}")
    print(f"Class weight:         {config.with_class_weight}")
    print(f"Train weight mode:    {config.train_weight_mode or 'none'}")
    if sample_weight_stats is not None:
        print(
            "Train weight stats:   "
            f"min={sample_weight_stats['min']:.3f}, "
            f"mean={sample_weight_stats['mean']:.3f}, "
            f"max={sample_weight_stats['max']:.3f}, "
            f"std={sample_weight_stats['std']:.3f}"
        )
    print(f"Drop columns:         {config.drop_columns or 'none'}")
    print(f"Fixed params path:    {fixed_params_path or 'none'}")
    print("=" * 70)

    search_auc = None
    best_trial_number = None
    completed_trials = 0
    best_params: dict

    if fixed_params_path is not None:
        best_params = load_fixed_params(fixed_params_path)
        pd.DataFrame([{"mode": "fixed_params", "source": str(fixed_params_path)}]).to_csv(paths.trials_path, index=False)
        print("\nUsing fixed params. Optuna search skipped.")
    else:
        if optuna is None:
            raise ModuleNotFoundError("optuna not installed. Install with: pip install optuna")

        study = optuna.create_study(
            direction="maximize",
            study_name=config.study_name,
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=42),
        )

        completed_trials = len(study.trials)
        remaining_trials = max(config.n_trials - completed_trials, 0)
        if remaining_trials > 0:
            print(f"\nStarting Optuna search with {remaining_trials} new trial(s)...")
            study.optimize(
                objective_factory(prepared, config, sample_weight),
                n_trials=remaining_trials,
                timeout=config.timeout,
                show_progress_bar=False,
            )
        else:
            print(f"\nStudy already has {completed_trials} trial(s); skipping search.")

        if not study.trials:
            raise ValueError("No Optuna trials available. Set n_trials > 0 or provide fixed_params_path.")

        best_trial = study.best_trial
        best_trial_number = best_trial.number
        best_params = best_trial.params
        search_auc = float(best_trial.value)
        study.trials_dataframe().to_csv(paths.trials_path, index=False)

    best_model_params = materialize_params(config.model_family, best_params, config.thread_count)
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
    final_auc = roc_auc_score(prepared.y, final_cv["oof"])

    submission = prepared.test_raw[[ID_COL]].copy()
    submission[TARGET] = final_cv["test"]
    submission.to_csv(paths.submission_path, index=False)

    oof_frame = prepared.train_raw[[ID_COL]].copy()
    oof_frame["y_true"] = prepared.y.values
    oof_frame["oof_pred"] = final_cv["oof"]
    oof_frame.to_csv(paths.oof_path, index=False)

    paths.best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    metadata = {
        "config_path": str(args.config),
        "config": config.to_dict(),
        "data_dir": str(data_dir),
        "study_storage": "fixed-params" if fixed_params_path is not None else (storage or "memory"),
        "existing_trials": completed_trials,
        "requested_trials": config.n_trials,
        "best_trial_number": best_trial_number,
        "search_auc": search_auc,
        "final_auc": float(final_auc),
        "best_params": best_params,
        "fixed_params_path": str(fixed_params_path) if fixed_params_path is not None else None,
        "sample_weight_stats": sample_weight_stats,
        "paths": {
            "submission": str(paths.submission_path),
            "oof": str(paths.oof_path),
            "trials": str(paths.trials_path),
            "summary": str(paths.summary_path),
            "best_params": str(paths.best_params_path),
        },
    }
    paths.summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    oof_corr_to_best = compute_corr_to_best(
        paths.experiment_runs_path,
        current_experiment=config.name,
        current_oof_path=paths.oof_path,
    )
    append_experiment_run(
        paths.experiment_runs_path,
        experiment=config.name,
        model_family=config.model_family,
        feature_set=config.feature_set,
        with_class_weight=config.with_class_weight,
        search_auc=search_auc,
        final_auc=float(final_auc),
        oof_corr_to_best=oof_corr_to_best,
        submission_path=paths.submission_path,
        summary_path=paths.summary_path,
        notes=config.notes,
    )

    print(f"\nBest trial: {best_trial_number if best_trial_number is not None else 'fixed params'}")
    print(f"  Search AUC: {'n/a' if search_auc is None else f'{search_auc:.6f}'}")
    print(f"  Final OOF AUC: {final_auc:.6f}")
    if oof_corr_to_best is not None:
        print(f"  OOF corr to previous best: {oof_corr_to_best:.6f}")
    print(f"\nSaved submission: {paths.submission_path}")
    print(f"Saved OOF:        {paths.oof_path}")
    print(f"Saved trials:     {paths.trials_path}")
    print(f"Saved summary:    {paths.summary_path}")
    print(f"Saved params:     {paths.best_params_path}")


if __name__ == "__main__":
    main()
