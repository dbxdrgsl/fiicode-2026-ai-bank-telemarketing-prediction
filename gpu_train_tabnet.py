#!/usr/bin/env python3
"""
GPU-only TabNet trainer for FiiCode tabular data.

TabNet uses instance-wise feature selection via attention mechanism,
learning which features matter for EACH prediction dynamically.

Outputs follow the repository artifact contract:
- outputs/models/<experiment>/best_params.json
- outputs/oof/<experiment>/oof_predictions.csv
- outputs/submissions/<experiment>/submission.csv
- outputs/logs/<experiment>/best_run_summary.json
- outputs/logs/<experiment>/optuna_trials.csv (epoch/fold/seed metrics)
- outputs/logs/<experiment>/feature_importance_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DEFAULT_COMPETITION_SLUG, repo_root
from src.features import ID_COL, TARGET, prepare_data, resolve_data_dir
from src.tracking import append_experiment_run, compute_corr_to_best

try:
    import yaml
except ImportError:
    yaml = None


def _coerce_int_list(values: Any, *, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list of integers.")
    return [int(value) for value in values]


@dataclass
class TabNetRunConfig:
    name: str
    seeds: list[int]
    n_d: int
    n_a: int
    n_steps: int
    gamma: float
    lambda_sparse: float
    epochs: int
    batch_size: int
    lr: float
    patience: int
    virtual_batch_size: int = 128
    momentum: float = 0.02
    clip_value: float = 1.0
    tabnet_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, experiment_name: str) -> "TabNetRunConfig":
        default_name = f"{experiment_name}_tabnet"
        return cls(
            name=str(payload.get("name") or default_name),
            seeds=_coerce_int_list(payload.get("seeds", [42]), name=f"{default_name}.seeds"),
            n_d=int(payload.get("n_d", 64)),
            n_a=int(payload.get("n_a", 64)),
            n_steps=int(payload.get("n_steps", 5)),
            gamma=float(payload.get("gamma", 1.5)),
            lambda_sparse=float(payload.get("lambda_sparse", 1e-4)),
            epochs=int(payload.get("epochs", 200)),
            batch_size=int(payload.get("batch_size", 1024)),
            lr=float(payload.get("lr", 0.02)),
            patience=int(payload.get("patience", 50)),
            virtual_batch_size=int(payload.get("virtual_batch_size", 128)),
            momentum=float(payload.get("momentum", 0.02)),
            clip_value=float(payload.get("clip_value", 1.0)),
            tabnet_params=dict(payload.get("tabnet_params", {})),
        )


@dataclass
class ExperimentConfig:
    name: str
    competition_slug: str
    feature_set: str
    drop_columns: list[str]
    data_dir: str | None
    output_root: str
    n_folds: int
    model: TabNetRunConfig
    notes: str = ""

    @classmethod
    def from_yaml_file(cls, path: Path) -> "ExperimentConfig":
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config files.")
        with open(path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if "model" not in config:
            raise ValueError("Config must include a 'model' entry for TabNet configuration.")

        model = TabNetRunConfig.from_dict(config["model"], experiment_name=config["name"])

        return cls(
            name=str(config["name"]),
            competition_slug=str(config.get("competition_slug", DEFAULT_COMPETITION_SLUG)),
            feature_set=str(config["feature_set"]),
            drop_columns=list(config.get("drop_columns", [])),
            data_dir=str(config["data_dir"]) if config.get("data_dir") else None,
            output_root=str(config.get("output_root", "outputs")),
            n_folds=int(config.get("n_folds", 5)),
            model=model,
            notes=str(config.get("notes", "")),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, LabelEncoder]]:
    """Encode categorical columns with LabelEncoder for TabNet."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy() if test_df is not None else None
    encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        train_encoded[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        if test_encoded is not None:
            # Handle unseen categories
            test_encoded[col] = test_encoded[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return train_encoded, test_encoded, encoders


def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Normalize numerical features, leave categoricals as-is."""
    train_norm = train_df.copy()
    test_norm = test_df.copy() if test_df is not None else None

    numerical_cols = [c for c in train_df.columns if c not in categorical_columns]

    if numerical_cols:
        scaler = StandardScaler()
        train_norm[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
        if test_norm is not None:
            test_norm[numerical_cols] = scaler.transform(test_df[numerical_cols])
    else:
        scaler = None

    return train_norm, test_norm, scaler


def train_tabnet_cv(
    config: ExperimentConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame | None,
    categorical_columns: list[str],
    output_root: Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], pd.DataFrame, np.ndarray]:
    """Train TabNet with cross-validation and multiple seeds.
    
    Returns:
        oof_preds, test_preds, best_params, trials_df, avg_feature_importance
    """
    model_cfg = config.model
    n_folds = config.n_folds

    oof_preds = np.zeros(len(x_train))
    test_preds_all = []
    trials_records = []
    feature_importances_all = []

    # Encode categoricals
    x_train_encoded, x_test_encoded, encoders = encode_categoricals(
        x_train, x_test, categorical_columns
    )

    # Normalize features
    x_train_norm, x_test_norm, scaler = normalize_features(
        x_train_encoded, x_test_encoded, categorical_columns
    )

    # Get categorical indices (after encoding)
    cat_indices = [x_train_norm.columns.get_loc(c) for c in categorical_columns]
    cat_dims = [int(x_train_encoded[c].max() + 1) for c in categorical_columns]

    print(f"\n{'='*80}")
    print(f"TabNet Configuration:")
    print(f"  n_d={model_cfg.n_d}, n_a={model_cfg.n_a}, n_steps={model_cfg.n_steps}")
    print(f"  gamma={model_cfg.gamma}, lambda_sparse={model_cfg.lambda_sparse}")
    print(f"  batch_size={model_cfg.batch_size}, lr={model_cfg.lr}")
    print(f"  patience={model_cfg.patience}, epochs={model_cfg.epochs}")
    print(f"  categorical_columns={len(categorical_columns)}, cat_dims={cat_dims}")
    print(f"{'='*80}\n")

    for seed_idx, seed in enumerate(model_cfg.seeds, start=1):
        print(f"\n{'='*80}")
        print(f"Seed {seed_idx}/{len(model_cfg.seeds)}: {seed}")
        print(f"{'='*80}\n")

        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        seed_test_preds = []
        seed_oof_preds = np.zeros(len(x_train))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train_norm, y_train), start=1):
            print(f"\n--- Fold {fold_idx}/{n_folds} (Seed {seed}) ---")

            x_tr = x_train_norm.iloc[train_idx].values
            y_tr = y_train.iloc[train_idx].values
            x_val = x_train_norm.iloc[val_idx].values
            y_val = y_train.iloc[val_idx].values

            # Create TabNet model
            clf = TabNetClassifier(
                n_d=model_cfg.n_d,
                n_a=model_cfg.n_a,
                n_steps=model_cfg.n_steps,
                gamma=model_cfg.gamma,
                lambda_sparse=model_cfg.lambda_sparse,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=model_cfg.lr),
                scheduler_fn=None,
                scheduler_params=None,
                mask_type="sparsemax",
                n_shared=2,
                n_independent=2,
                seed=seed,
                verbose=1,
                cat_idxs=cat_indices,
                cat_dims=cat_dims,
                cat_emb_dim=1,
                momentum=model_cfg.momentum,
                clip_value=model_cfg.clip_value,
                **model_cfg.tabnet_params,
            )

            # Train
            clf.fit(
                X_train=x_tr,
                y_train=y_tr,
                eval_set=[(x_val, y_val)],
                eval_name=["val"],
                eval_metric=["auc"],
                max_epochs=model_cfg.epochs,
                patience=model_cfg.patience,
                batch_size=model_cfg.batch_size,
                virtual_batch_size=model_cfg.virtual_batch_size,
                num_workers=0,
                drop_last=False,
            )

            # Predict validation
            val_preds = clf.predict_proba(x_val)[:, 1]
            seed_oof_preds[val_idx] = val_preds

            fold_auc = roc_auc_score(y_val, val_preds)
            print(f"Fold {fold_idx} AUC: {fold_auc:.6f}")

            # Predict test
            if x_test_norm is not None:
                test_pred = clf.predict_proba(x_test_norm.values)[:, 1]
                seed_test_preds.append(test_pred)

            # Extract feature importance
            feature_imp = clf.feature_importances_
            feature_importances_all.append(feature_imp)

            # Record trial
            trials_records.append({
                "seed": seed,
                "fold": fold_idx,
                "auc": fold_auc,
                "best_epoch": clf.best_epoch,
            })

        # Aggregate seed OOF
        oof_preds += seed_oof_preds / len(model_cfg.seeds)

        if seed_test_preds:
            test_preds_all.append(np.mean(seed_test_preds, axis=0))

        seed_auc = roc_auc_score(y_train, seed_oof_preds)
        print(f"\nSeed {seed} OOF AUC: {seed_auc:.6f}")

    # Final OOF score
    final_oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n{'='*80}")
    print(f"Final OOF AUC: {final_oof_auc:.6f}")
    print(f"{'='*80}\n")

    # Aggregate test predictions
    test_preds = np.mean(test_preds_all, axis=0) if test_preds_all else None

    # Feature importance
    avg_feature_importance = np.mean(feature_importances_all, axis=0)
    feature_importance_df = pd.DataFrame({
        "feature": x_train.columns,
        "importance": avg_feature_importance,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 Features by TabNet Importance:")
    print(feature_importance_df.head(20).to_string(index=False))

    # Best params (simplified for TabNet)
    best_params = {
        "n_d": model_cfg.n_d,
        "n_a": model_cfg.n_a,
        "n_steps": model_cfg.n_steps,
        "gamma": model_cfg.gamma,
        "lambda_sparse": model_cfg.lambda_sparse,
        "lr": model_cfg.lr,
        "batch_size": model_cfg.batch_size,
        "patience": model_cfg.patience,
    }

    trials_df = pd.DataFrame(trials_records)

    return oof_preds, test_preds, best_params, trials_df, avg_feature_importance


def compare_feature_importance(
    tabnet_importance: pd.DataFrame,
    catboost_summary_path: Path | None,
    output_path: Path,
) -> None:
    """Compare TabNet and CatBoost feature importance."""
    if catboost_summary_path is None or not catboost_summary_path.exists():
        print("\nCatBoost summary not found, skipping comparison.")
        tabnet_importance.to_csv(output_path, index=False)
        print(f"Saved TabNet feature importance to: {output_path}")
        return

    # Load CatBoost best params (feature importance is typically not saved in summary)
    # For now, we'll just save TabNet importance and note the comparison is manual
    tabnet_importance.to_csv(output_path, index=False)
    print(f"\nSaved TabNet feature importance to: {output_path}")
    print("Note: Manual comparison with CatBoost feature importance recommended.")
    print(f"CatBoost reference: {catboost_summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TabNet GPU trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--catboost-reference",
        type=str,
        default="outputs/logs/exp012_blend_bucket_features_fixed/best_run_summary.json",
        help="Path to CatBoost baseline for feature importance comparison",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ExperimentConfig.from_yaml_file(config_path)
    print(f"\nExperiment: {config.name}")
    print(f"Feature set: {config.feature_set}")
    print(f"Folds: {config.n_folds}")

    # Resolve data directory
    data_dir = resolve_data_dir(Path(config.data_dir) if config.data_dir else None)
    print(f"Data directory: {data_dir}")

    # Prepare data
    prepared = prepare_data(
        feature_set=config.feature_set,
        data_dir=data_dir,
        drop_columns=config.drop_columns,
    )

    x_train = prepared.x
    y_train = prepared.y
    x_test = prepared.x_test
    categorical_columns = prepared.categorical_columns

    print(f"\nTrain shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape if x_test is not None else 'N/A'}")
    print(f"Categorical columns: {len(categorical_columns)}")

    # Setup output paths
    output_root = Path(config.output_root)
    model_dir = output_root / "models" / config.name
    oof_dir = output_root / "oof" / config.name
    submission_dir = output_root / "submissions" / config.name
    log_dir = output_root / "logs" / config.name

    for directory in [model_dir, oof_dir, submission_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Train TabNet
    oof_preds, test_preds, best_params, trials_df, avg_feature_importance = train_tabnet_cv(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        categorical_columns=categorical_columns,
        output_root=output_root,
    )

    # Save outputs
    oof_path = oof_dir / "oof_predictions.csv"
    oof_df = pd.DataFrame({ID_COL: prepared.train_raw[ID_COL], TARGET: oof_preds})
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved OOF predictions to: {oof_path}")

    if test_preds is not None:
        submission_path = submission_dir / "submission.csv"
        submission_df = pd.DataFrame({
            ID_COL: prepared.test_raw[ID_COL],
            "Subscribed": test_preds,
        })
        submission_df.to_csv(submission_path, index=False)
        print(f"Saved submission to: {submission_path}")

    # Save best params
    best_params_path = model_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params to: {best_params_path}")

    # Save trials
    trials_path = log_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Saved trials to: {trials_path}")

    # Save summary
    final_oof_auc = roc_auc_score(y_train, oof_preds)
    summary = {
        "config_path": str(config_path),
        "config": {
            "name": config.name,
            "feature_set": config.feature_set,
            "n_folds": config.n_folds,
            "model": {
                "n_d": config.model.n_d,
                "n_a": config.model.n_a,
                "n_steps": config.model.n_steps,
                "gamma": config.model.gamma,
                "lambda_sparse": config.model.lambda_sparse,
                "seeds": config.model.seeds,
            },
        },
        "final_auc": final_oof_auc,
        "best_params": best_params,
        "paths": {
            "submission": str(submission_path) if test_preds is not None else None,
            "oof": str(oof_path),
            "trials": str(trials_path),
            "summary": str(log_dir / "best_run_summary.json"),
            "best_params": str(best_params_path),
        },
    }

    summary_path = log_dir / "best_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    # Track experiment
    corr_to_best = compute_corr_to_best(oof_path)
    append_experiment_run(
        experiment_name=config.name,
        oof_score=final_oof_auc,
        lb_score=None,
        correlation_to_best=corr_to_best,
    )
    print(f"\nRecorded experiment run in tracking journal")
    print(f"Correlation to best: {corr_to_best:.4f}" if corr_to_best is not None else "N/A")

    # Feature importance comparison
    catboost_ref_path = Path(args.catboost_reference) if args.catboost_reference else None
    feature_imp_path = log_dir / "feature_importance_comparison.csv"

    # Get TabNet feature importance (returned from training)
    feature_importance_df = pd.DataFrame({
        "feature": x_train.columns,
        "importance": avg_feature_importance,
    }).sort_values("importance", ascending=False)

    compare_feature_importance(feature_importance_df, catboost_ref_path, feature_imp_path)

    print(f"\n{'='*80}")
    print(f"Experiment {config.name} completed successfully!")
    print(f"Final OOF AUC: {final_oof_auc:.6f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
