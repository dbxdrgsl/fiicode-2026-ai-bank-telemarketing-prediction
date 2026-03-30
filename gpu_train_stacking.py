#!/usr/bin/env python3
"""
Neural stacking (exp050) over a curated model whitelist.

This runner enforces CUDA usage and emits the full repository artifact contract:
- outputs/models/<experiment>/best_params.json
- outputs/oof/<experiment>/oof_predictions.csv
- outputs/submissions/<experiment>/submission.csv
- outputs/logs/<experiment>/best_run_summary.json
- outputs/logs/<experiment>/optuna_trials.csv  (fold-level training records)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import repo_root
from src.features import ID_COL, TARGET
from src.tracking import append_experiment_run, compute_corr_to_best

DEFAULT_MODEL_WHITELIST = [
    "exp012_blend_bucket_features_fixed",
    "exp046_tabnet",
    "exp047_tabnet_deep",
    "exp048_blend_exp012_exp046",
    "exp039_attention_gpu",
    "exp037_fttransformer_deep_gpu",
    "exp038_tabresnet_target_enc_gpu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU neural stacking trainer")
    parser.add_argument("--name", type=str, default="exp050_neural_stacking")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_WHITELIST)
    parser.add_argument("--min-models", type=int, default=3)
    parser.add_argument("--min-auc", type=float, default=0.90)
    parser.add_argument("--max-auc", type=float, default=0.99)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--full-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


class StackingNN(nn.Module):
    def __init__(self, n_models: int, hidden_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.fc1 = nn.Linear(n_models, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_selected_models(
    model_names: list[str],
    *,
    min_auc: float,
    max_auc: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, float]]:
    out_dir = repo_root() / "outputs"
    oof_frames: dict[str, pd.DataFrame] = {}
    sub_frames: dict[str, pd.DataFrame] = {}
    aucs: dict[str, float] = {}

    print("\nLoading whitelisted OOF/submission files...")
    for model_name in model_names:
        oof_path = out_dir / "oof" / model_name / "oof_predictions.csv"
        sub_path = out_dir / "submissions" / model_name / "submission.csv"
        if not oof_path.exists() or not sub_path.exists():
            print(f"  - skip {model_name}: missing OOF or submission")
            continue

        oof = pd.read_csv(oof_path)
        sub = pd.read_csv(sub_path)
        required_oof = {ID_COL, "y_true", "oof_pred"}
        required_sub = {ID_COL, TARGET}
        if not required_oof.issubset(oof.columns) or not required_sub.issubset(sub.columns):
            print(f"  - skip {model_name}: schema mismatch")
            continue

        auc = roc_auc_score(oof["y_true"].astype(float), oof["oof_pred"].astype(float))
        if not (min_auc < auc < max_auc):
            print(f"  - skip {model_name}: AUC {auc:.6f} outside ({min_auc}, {max_auc})")
            continue

        oof_frames[model_name] = oof[[ID_COL, "y_true", "oof_pred"]].copy()
        sub_frames[model_name] = sub[[ID_COL, TARGET]].copy()
        aucs[model_name] = float(auc)
        print(f"  + {model_name}: OOF AUC={auc:.6f}")

    return oof_frames, sub_frames, aucs


def build_design_matrices(
    model_names: list[str],
    oof_frames: dict[str, pd.DataFrame],
    sub_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = model_names[0]
    oof_merged = oof_frames[first][[ID_COL, "y_true"]].copy()
    sub_merged = sub_frames[first][[ID_COL]].copy()

    for model_name in model_names:
        frame = oof_frames[model_name]
        y_check = oof_merged[[ID_COL, "y_true"]].merge(
            frame[[ID_COL, "y_true"]],
            on=ID_COL,
            how="left",
            suffixes=("_base", "_other"),
        )
        if y_check["y_true_other"].isna().any():
            raise ValueError(f"Missing OOF ids for model {model_name}.")
        if not np.array_equal(
            y_check["y_true_base"].to_numpy(dtype=float),
            y_check["y_true_other"].to_numpy(dtype=float),
        ):
            raise ValueError(f"y_true mismatch detected for model {model_name}.")

        oof_merged = oof_merged.merge(
            frame[[ID_COL, "oof_pred"]].rename(columns={"oof_pred": model_name}),
            on=ID_COL,
            how="left",
        )
        if oof_merged[model_name].isna().any():
            raise ValueError(f"OOF predictions contain missing ids for model {model_name}.")

        sub_merged = sub_merged.merge(
            sub_frames[model_name].rename(columns={TARGET: model_name}),
            on=ID_COL,
            how="left",
            sort=False,
        )
        if sub_merged[model_name].isna().any():
            raise ValueError(f"Submission predictions contain missing ids for model {model_name}.")

    return oof_merged, sub_merged


def train_cv(
    *,
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_models: int,
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, list[float], list[dict[str, float | int]]]:
    oof_preds = np.zeros(len(y), dtype=np.float64)
    fold_scores: list[float] = []
    trials_records: list[dict[str, float | int]] = []

    for seed in args.seeds:
        print(f"\n{'=' * 80}\nSeed {seed}\n{'=' * 80}")
        set_seed(seed)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), start=1):
            X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y[train_idx], dtype=torch.float32)
            X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.float32)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            model = StackingNN(n_models, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.BCEWithLogitsLoss()

            best_val_auc = -np.inf
            best_epoch = -1
            best_val_pred: np.ndarray | None = None
            patience_counter = 0

            for epoch in range(args.max_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    out = model(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_out = model(X_val.to(device))
                    val_pred = torch.sigmoid(val_out).cpu().numpy()
                    val_auc = roc_auc_score(y_val.numpy(), val_pred)

                if val_auc > best_val_auc:
                    best_val_auc = float(val_auc)
                    best_epoch = epoch + 1
                    best_val_pred = val_pred
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    break

            if best_val_pred is None:
                raise RuntimeError("Training failed to produce validation predictions.")

            print(f"Fold {fold}/{args.n_splits} best AUC: {best_val_auc:.6f} @ epoch {best_epoch}")
            oof_preds[val_idx] += best_val_pred / len(args.seeds)
            fold_scores.append(best_val_auc)
            trials_records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "auc": best_val_auc,
                    "best_epoch": best_epoch,
                }
            )

    return oof_preds, fold_scores, trials_records


def train_full_model(
    *,
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_models: int,
    args: argparse.Namespace,
    device: str,
) -> StackingNN:
    X_full = torch.tensor(X_scaled, dtype=torch.float32)
    y_full = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_full, y_full)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = StackingNN(n_models, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining full-data stacker for {args.full_epochs} epochs...")
    for _ in range(args.full_epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_train_stacking.py, but no GPU was detected.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = "cuda"

    oof_frames, sub_frames, aucs = load_selected_models(
        args.models,
        min_auc=args.min_auc,
        max_auc=args.max_auc,
    )
    selected_model_names = [name for name in args.models if name in oof_frames]
    if len(selected_model_names) < args.min_models:
        raise ValueError(
            f"Need at least {args.min_models} valid models, got {len(selected_model_names)} "
            f"({selected_model_names})."
        )

    print(f"\nUsing {len(selected_model_names)} models: {', '.join(selected_model_names)}")
    oof_merged, sub_merged = build_design_matrices(selected_model_names, oof_frames, sub_frames)

    X = oof_merged[selected_model_names].to_numpy(dtype=np.float32)
    y = oof_merged["y_true"].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    oof_preds, fold_scores, trials_records = train_cv(
        X_scaled=X_scaled,
        y=y,
        n_models=len(selected_model_names),
        args=args,
        device=device,
    )
    final_auc = float(roc_auc_score(y, oof_preds))
    print(f"\nFinal stacking OOF AUC: {final_auc:.6f}")
    print(f"Mean fold AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")

    full_model = train_full_model(
        X_scaled=X_scaled,
        y=y,
        n_models=len(selected_model_names),
        args=args,
        device=device,
    )
    full_model.eval()
    with torch.no_grad():
        X_test = sub_merged[selected_model_names].to_numpy(dtype=np.float32)
        X_test_scaled = scaler.transform(X_test)
        test_logits = full_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device))
        test_pred = torch.sigmoid(test_logits).cpu().numpy()

    root = repo_root()
    outputs = root / "outputs"
    model_dir = outputs / "models" / args.name
    oof_dir = outputs / "oof" / args.name
    submission_dir = outputs / "submissions" / args.name
    logs_dir = outputs / "logs" / args.name
    for directory in (model_dir, oof_dir, submission_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    oof_path = oof_dir / "oof_predictions.csv"
    oof_df = pd.DataFrame({ID_COL: oof_merged[ID_COL], "y_true": y, "oof_pred": oof_preds})
    oof_df.to_csv(oof_path, index=False)

    submission_path = submission_dir / "submission.csv"
    submission_df = pd.DataFrame({ID_COL: sub_merged[ID_COL], TARGET: test_pred})
    submission_df.to_csv(submission_path, index=False)

    trials_path = logs_dir / "optuna_trials.csv"
    pd.DataFrame(trials_records).to_csv(trials_path, index=False)

    best_single_model = max(selected_model_names, key=lambda name: aucs[name])
    best_single_auc = float(aucs[best_single_model])
    improvement = final_auc - best_single_auc
    best_params = {
        "method": "neural_stacking",
        "model_names": selected_model_names,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "full_epochs": args.full_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "n_splits": args.n_splits,
        "seeds": args.seeds,
    }
    best_params_path = model_dir / "best_params.json"
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    summary_path = logs_dir / "best_run_summary.json"
    summary = {
        "config_path": None,
        "config": {
            "name": args.name,
            "model_family": "stacking_nn",
            "feature_set": "mixed",
            "with_class_weight": False,
        },
        "data_dir": None,
        "study_storage": "neural-stacking",
        "existing_trials": 0,
        "requested_trials": len(trials_records),
        "best_trial_number": None,
        "search_auc": final_auc,
        "final_auc": final_auc,
        "best_params": best_params,
        "members": [
            {
                "name": name,
                "model_family": "blend_component",
                "final_auc": float(aucs[name]),
                "corr_to_best": None,
            }
            for name in selected_model_names
        ],
        "best_single_model": best_single_model,
        "best_single_auc": best_single_auc,
        "improvement_over_best_single": improvement,
        "promoted": True,
        "paths": {
            "submission": str(submission_path),
            "oof": str(oof_path),
            "summary": str(summary_path),
            "best_params": str(best_params_path),
            "trials": str(trials_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    experiment_runs_path = outputs / "logs" / "experiment_runs.csv"
    corr_to_best = compute_corr_to_best(
        experiment_runs_path,
        current_experiment=args.name,
        current_oof_path=oof_path,
    )
    append_experiment_run(
        path=experiment_runs_path,
        experiment=args.name,
        model_family="stacking_nn",
        feature_set="mixed",
        with_class_weight=False,
        search_auc=final_auc,
        final_auc=final_auc,
        oof_corr_to_best=corr_to_best,
        submission_path=submission_path,
        summary_path=summary_path,
        notes=f"Neural stacking over: {', '.join(selected_model_names)}",
    )

    print("\n✅ Neural stacking complete")
    print(f"Experiment:       {args.name}")
    print(f"Best single:      {best_single_model} ({best_single_auc:.6f})")
    print(f"Stacking OOF AUC: {final_auc:.6f}")
    print(f"Improvement:      {improvement:.6f}")
    print(f"Saved summary:    {summary_path}")


if __name__ == "__main__":
    main()
