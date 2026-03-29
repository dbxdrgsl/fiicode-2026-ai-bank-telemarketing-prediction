#!/usr/bin/env python3
"""
GPU-only GNN trainer for customer similarity graphs.

Supports GraphSAGE and GAT with multi-seed CV, then writes standard repo artifacts:
- outputs/models/<experiment>/best_params.json
- outputs/oof/<experiment>/oof_predictions.csv
- outputs/submissions/<experiment>/submission.csv
- outputs/logs/<experiment>/best_run_summary.json
- outputs/logs/<experiment>/optuna_trials.csv
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

from src.config import DEFAULT_COMPETITION_SLUG, repo_root
from src.features import ID_COL, TARGET, prepare_data, resolve_data_dir
from src.tracking import append_experiment_run, compute_corr_to_best

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class GNNRunConfig:
    name: str
    competition_slug: str
    feature_set: str
    drop_columns: list[str]
    data_dir: str | None
    output_root: str
    n_splits: int
    family: str
    seeds: list[int]
    k_neighbors: int
    hidden_dim: int
    num_layers: int
    epochs: int
    lr: float
    heads: int
    notes: str

    @classmethod
    def from_yaml_file(cls, path: Path) -> "GNNRunConfig":
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config files.")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = payload.get("model") or {}

        name = str(payload.get("name") or payload.get("experiment_name") or path.stem)
        family = str(model.get("family", "graphsage")).strip().lower()
        if family not in {"graphsage", "gat"}:
            raise ValueError(f"Unsupported GNN family: {family}")

        return cls(
            name=name,
            competition_slug=str(payload.get("competition_slug", DEFAULT_COMPETITION_SLUG)),
            feature_set=str(payload.get("feature_set", "blend_buckets")),
            drop_columns=list(payload.get("drop_columns", [])),
            data_dir=str(payload["data_dir"]) if payload.get("data_dir") else None,
            output_root=str(payload.get("output_root", "outputs")),
            n_splits=int(payload.get("n_splits", 5)),
            family=family,
            seeds=[int(seed) for seed in model.get("seeds", [42])],
            k_neighbors=int(model.get("k_neighbors", 15)),
            hidden_dim=int(model.get("hidden_dim", 128)),
            num_layers=int(model.get("num_layers", 2)),
            epochs=int(model.get("epochs", 100)),
            lr=float(model.get("lr", 1e-3)),
            heads=int(model.get("heads", 4)),
            notes=str(payload.get("notes", "")),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_encoded = x_train.copy()
    test_encoded = x_test.copy()

    for column in categorical_columns:
        combined = pd.concat(
            [
                train_encoded[column].fillna("missing").astype(str),
                test_encoded[column].fillna("missing").astype(str),
            ],
            axis=0,
            ignore_index=True,
        )
        categories = pd.Index(combined.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}
        train_encoded[column] = (
            train_encoded[column].fillna("missing").astype(str).map(mapping).astype(np.int32)
        )
        test_encoded[column] = (
            test_encoded[column].fillna("missing").astype(str).map(mapping).astype(np.int32)
        )

    for column in train_encoded.columns:
        train_encoded[column] = pd.to_numeric(train_encoded[column], errors="coerce")
        test_encoded[column] = pd.to_numeric(test_encoded[column], errors="coerce")
        fill_value = float(train_encoded[column].median())
        if not np.isfinite(fill_value):
            fill_value = 0.0
        train_encoded[column] = train_encoded[column].fillna(fill_value)
        test_encoded[column] = test_encoded[column].fillna(fill_value)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train_encoded.values).astype(np.float32)
    x_test_scaled = scaler.transform(test_encoded.values).astype(np.float32)
    return x_train_scaled, x_test_scaled


def build_customer_graph(features: np.ndarray, *, k_neighbors: int) -> torch.Tensor:
    n_samples = features.shape[0]
    n_neighbors = min(max(2, k_neighbors + 1), n_samples)
    print(f"Building k-NN graph with k={n_neighbors - 1} over {n_samples} nodes...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
    nbrs.fit(features)
    _, indices = nbrs.kneighbors(features)

    edges: set[tuple[int, int]] = set()
    for node_idx, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            src = int(node_idx)
            dst = int(neighbor)
            edges.add((src, dst))
            edges.add((dst, src))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    print(f"Graph built: {n_samples} nodes, {edge_index.shape[1]} directed edges")
    return edge_index


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads, concat=True)])
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            )
        self.fc = nn.Linear(hidden_channels * heads, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def make_model(config: GNNRunConfig, in_channels: int) -> nn.Module:
    if config.family == "graphsage":
        return GraphSAGE(in_channels, hidden_channels=config.hidden_dim, num_layers=config.num_layers)
    return GAT(
        in_channels,
        hidden_channels=config.hidden_dim,
        num_layers=config.num_layers,
        heads=config.heads,
    )


def train_fold(
    model: nn.Module,
    data: Data,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    patience: int = 20,
) -> tuple[float, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    best_auc = -np.inf
    best_epoch = 0
    wait = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_idx_t], data.y[train_idx_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_pred = torch.sigmoid(logits[val_idx_t]).detach().cpu().numpy()
        val_auc = roc_auc_score(y_train[val_idx], val_pred)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return float(best_auc), best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU GNN trainer")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional data directory override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (repo_root() / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = GNNRunConfig.from_yaml_file(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_train_gnn.py, but no GPU was detected.")
    device = torch.device("cuda")

    print(f"\n{'=' * 78}")
    print(f"GNN Experiment: {config.name}")
    print(f"Family:         {config.family}")
    print(f"Feature set:    {config.feature_set}")
    print(f"Folds:          {config.n_splits}")
    print(f"Seeds:          {config.seeds}")
    print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 78}\n")

    data_dir = resolve_data_dir(args.data_dir or (Path(config.data_dir) if config.data_dir else None))
    prepared = prepare_data(
        data_dir=data_dir,
        feature_set=config.feature_set,
        drop_columns=config.drop_columns,
        load_test=True,
    )
    if prepared.x_test is None or prepared.test_raw is None:
        raise RuntimeError("Test features are required for submission generation.")

    y_train = prepared.y.astype(int).reset_index(drop=True)
    x_train_scaled, x_test_scaled = encode_features(prepared.x, prepared.x_test, prepared.categorical_columns)
    n_train = x_train_scaled.shape[0]

    edge_index_train = build_customer_graph(x_train_scaled, k_neighbors=config.k_neighbors)
    edge_index_test = build_customer_graph(x_test_scaled, k_neighbors=config.k_neighbors)
    graph_data = Data(
        x=torch.from_numpy(x_train_scaled).to(device),
        edge_index=edge_index_train.to(device),
        y=torch.from_numpy(y_train.to_numpy(dtype=np.float32)).to(device),
    )
    test_graph = Data(
        x=torch.from_numpy(x_test_scaled).to(device),
        edge_index=edge_index_test.to(device),
    )

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = []
    trial_rows: list[dict[str, Any]] = []
    fold_aucs: list[float] = []

    for seed in config.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(prepared.x, y_train), start=1):
            print(f"Fold {fold}/{config.n_splits}")
            model = make_model(config, in_channels=x_train_scaled.shape[1])
            best_auc, best_epoch = train_fold(
                model,
                graph_data,
                y_train.to_numpy(),
                train_idx,
                val_idx,
                epochs=config.epochs,
                lr=config.lr,
                device=device,
            )

            model.eval()
            with torch.no_grad():
                train_probs = torch.sigmoid(model(graph_data.x, graph_data.edge_index)).detach().cpu().numpy()
                test_probs = torch.sigmoid(model(test_graph.x, test_graph.edge_index)).detach().cpu().numpy()

            oof_preds[val_idx] += train_probs[val_idx] / len(config.seeds)
            test_preds.append(test_probs)
            fold_aucs.append(best_auc)

            print(f"  best AUC={best_auc:.6f} @ epoch {best_epoch}")
            trial_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "auc": best_auc,
                    "best_epoch": best_epoch,
                }
            )

    final_auc = float(roc_auc_score(y_train, oof_preds))
    mean_fold_auc = float(np.mean(fold_aucs))
    std_fold_auc = float(np.std(fold_aucs))
    final_test_pred = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32)

    print(f"\n{'=' * 78}")
    print(f"Final OOF AUC: {final_auc:.6f}")
    print(f"Fold mean/std: {mean_fold_auc:.6f} +/- {std_fold_auc:.6f}")
    print(f"{'=' * 78}\n")

    root = repo_root()
    output_root = (root / config.output_root).resolve()
    model_dir = output_root / "models" / config.name
    oof_dir = output_root / "oof" / config.name
    submission_dir = output_root / "submissions" / config.name
    log_dir = output_root / "logs" / config.name
    for directory in (model_dir, oof_dir, submission_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    best_params_path = model_dir / "best_params.json"
    oof_path = oof_dir / "oof_predictions.csv"
    submission_path = submission_dir / "submission.csv"
    trials_path = log_dir / "optuna_trials.csv"
    summary_path = log_dir / "best_run_summary.json"

    best_params = {
        "family": config.family,
        "seeds": config.seeds,
        "k_neighbors": config.k_neighbors,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "epochs": config.epochs,
        "lr": config.lr,
        "heads": config.heads if config.family == "gat" else None,
    }
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            ID_COL: prepared.train_raw[ID_COL],
            "y_true": y_train.to_numpy(),
            "oof_pred": oof_preds,
        }
    ).to_csv(oof_path, index=False)

    pd.DataFrame(
        {
            ID_COL: prepared.test_raw[ID_COL],
            TARGET: final_test_pred,
        }
    ).to_csv(submission_path, index=False)

    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)

    summary_payload = {
        "config_path": str(config_path),
        "config": {
            "name": config.name,
            "competition_slug": config.competition_slug,
            "feature_set": config.feature_set,
            "drop_columns": config.drop_columns,
            "data_dir": config.data_dir,
            "output_root": config.output_root,
            "n_splits": config.n_splits,
            "model": best_params,
            "notes": config.notes,
        },
        "final_auc": final_auc,
        "search_auc": None,
        "best_params": best_params,
        "paths": {
            "submission": str(submission_path),
            "oof": str(oof_path),
            "trials": str(trials_path),
            "summary": str(summary_path),
            "best_params": str(best_params_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    experiment_runs_path = output_root / "logs" / "experiment_runs.csv"
    corr_to_best = compute_corr_to_best(
        experiment_runs_path,
        current_experiment=config.name,
        current_oof_path=oof_path,
    )
    append_experiment_run(
        experiment=config.name,
        model_family=f"gnn_{config.family}",
        feature_set=config.feature_set,
        with_class_weight=False,
        search_auc=None,
        final_auc=final_auc,
        oof_corr_to_best=corr_to_best,
        submission_path=submission_path,
        summary_path=summary_path,
        notes=config.notes or f"GNN {config.family} with k={config.k_neighbors}",
        path=experiment_runs_path,
    )

    print(f"Saved summary:     {summary_path}")
    print(f"Saved submission:  {submission_path}")
    print(f"Saved OOF:         {oof_path}")


if __name__ == "__main__":
    main()
