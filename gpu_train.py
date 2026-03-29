#!/usr/bin/env python3
"""
GPU-only neural trainer for FiiCode tabular data.

This script trains custom (non-pretrained) neural architectures:
- attention
- ft_transformer (from scratch)
- tabresnet

Outputs follow the repository artifact contract per model:
- outputs/models/<experiment>/best_params.json
- outputs/oof/<experiment>/oof_predictions.csv
- outputs/submissions/<experiment>/submission.csv
- outputs/logs/<experiment>/best_run_summary.json
- outputs/logs/<experiment>/optuna_trials.csv  (epoch/fold/seed metrics; not Optuna)
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_COMPETITION_SLUG, repo_root
from src.features import ID_COL, TARGET, prepare_data, resolve_data_dir
from src.tracking import append_experiment_run, compute_corr_to_best

try:
    import yaml
except ImportError:
    yaml = None


def _normalize_family(raw_family: str) -> str:
    normalized = raw_family.strip().lower().replace("-", "_")
    aliases = {
        "attention": "attention",
        "attention_net": "attention",
        "ft_transformer": "ft_transformer",
        "fttransformer": "ft_transformer",
        "resnet": "tabresnet",
        "tabresnet": "tabresnet",
        "tab_resnet": "tabresnet",
        "tabular_resnet": "tabresnet",
    }
    if normalized not in aliases:
        allowed = sorted(set(aliases.values()))
        raise ValueError(f"Unsupported model family: {raw_family!r}. Expected one of: {allowed}")
    return aliases[normalized]


def _coerce_int_list(values: Any, *, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list of integers.")
    return [int(value) for value in values]


@dataclass
class ModelRunConfig:
    name: str
    family: str
    seeds: list[int]
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    max_grad_norm: float = 1.0
    num_workers: int = 0
    amp: bool = True
    model_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, experiment_name: str) -> "ModelRunConfig":
        if "family" not in payload:
            raise ValueError("Each model entry must include a 'family' field.")
        family = _normalize_family(str(payload["family"]))
        default_name = f"{experiment_name}_{family}"
        return cls(
            name=str(payload.get("name") or default_name),
            family=family,
            seeds=_coerce_int_list(payload.get("seeds", [42]), name=f"{default_name}.seeds"),
            epochs=int(payload.get("epochs", 80)),
            batch_size=int(payload.get("batch_size", 1024)),
            lr=float(payload.get("lr", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-2)),
            patience=int(payload.get("patience", 12)),
            max_grad_norm=float(payload.get("max_grad_norm", 1.0)),
            num_workers=int(payload.get("num_workers", 0)),
            amp=bool(payload.get("amp", True)),
            model_params=dict(payload.get("model_params", {})),
        )


@dataclass
class BlendConfig:
    enabled: bool = True
    name: str = "gpu_nn_blend"
    samples: int = 12000
    seed: int = 42
    min_improvement: float = 0.0
    max_models: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, default_name: str) -> "BlendConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            name=str(payload.get("name") or default_name),
            samples=int(payload.get("samples", 12000)),
            seed=int(payload.get("seed", 42)),
            min_improvement=float(payload.get("min_improvement", 0.0)),
            max_models=int(payload["max_models"]) if payload.get("max_models") is not None else None,
        )


@dataclass
class GPURunConfig:
    name: str
    competition_slug: str
    feature_set: str
    drop_columns: list[str]
    data_dir: str | None
    output_root: str
    require_cuda: bool
    deterministic: bool
    cv_folds: int
    cv_seed: int
    notes: str
    models: list[ModelRunConfig]
    blend: BlendConfig | None

    @classmethod
    def from_yaml(cls, path: Path) -> "GPURunConfig":
        if yaml is None:
            raise ModuleNotFoundError("PyYAML not installed. Install with: pip install pyyaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping YAML payload in {path}")

        name = str(payload.get("name") or path.stem)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("Config must include a non-empty 'models' list.")
        models = [ModelRunConfig.from_dict(dict(item), experiment_name=name) for item in raw_models]

        raw_blend = payload.get("blend")
        blend = None
        if raw_blend is not None:
            if not isinstance(raw_blend, dict):
                raise ValueError("'blend' must be a mapping when provided.")
            blend = BlendConfig.from_dict(raw_blend, default_name=f"{name}_blend")

        return cls(
            name=name,
            competition_slug=str(payload.get("competition_slug", DEFAULT_COMPETITION_SLUG)),
            feature_set=str(payload.get("feature_set", "blend_buckets")),
            drop_columns=[str(column) for column in payload.get("drop_columns", [])],
            data_dir=str(payload["data_dir"]) if payload.get("data_dir") is not None else None,
            output_root=str(payload.get("output_root", "outputs")),
            require_cuda=bool(payload.get("require_cuda", True)),
            deterministic=bool(payload.get("deterministic", False)),
            cv_folds=int(payload.get("cv_folds", 5)),
            cv_seed=int(payload.get("cv_seed", 42)),
            notes=str(payload.get("notes", "")),
            models=models,
            blend=blend,
        )


@dataclass
class EncodedTabularData:
    train_ids: np.ndarray
    test_ids: np.ndarray
    y: np.ndarray
    x_num_train: np.ndarray
    x_num_test: np.ndarray
    x_cat_train: np.ndarray
    x_cat_test: np.ndarray
    num_columns: list[str]
    cat_columns: list[str]
    cat_dims: list[int]


@dataclass
class ModelResult:
    run_name: str
    family: str
    oof: np.ndarray
    test: np.ndarray
    final_auc: float
    fold_seed_rows: list[dict[str, Any]]
    best_params_payload: dict[str, Any]
    note: str


def maybe_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def set_global_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(require_cuda: bool) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError(
            "CUDA is required by this run config but no GPU is visible to PyTorch. "
            "Use a CUDA-enabled environment and verify with `python -c \"import torch; print(torch.cuda.is_available())\"`."
        )
    if has_cuda:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({gpu_name})")
        return device
    print("Using device: cpu")
    return torch.device("cpu")


def encode_tabular_inputs(prepared) -> EncodedTabularData:
    if prepared.x_test is None or prepared.test_raw is None:
        raise ValueError("Test features are required for submission prediction, but x_test/test_raw is missing.")

    x_train = prepared.x.copy()
    x_test = prepared.x_test.copy()
    cat_columns = list(prepared.categorical_columns)
    num_columns = [column for column in x_train.columns if column not in cat_columns]

    x_cat_train = np.zeros((len(x_train), len(cat_columns)), dtype=np.int64)
    x_cat_test = np.zeros((len(x_test), len(cat_columns)), dtype=np.int64)
    cat_dims: list[int] = []
    for idx, column in enumerate(cat_columns):
        train_values = x_train[column].fillna("missing").astype(str)
        test_values = x_test[column].fillna("missing").astype(str)
        combined = pd.concat([train_values, test_values], axis=0, ignore_index=True)
        codes, _ = pd.factorize(combined, sort=True)
        n_classes = int(codes.max() + 1)
        x_cat_train[:, idx] = codes[: len(x_train)]
        x_cat_test[:, idx] = codes[len(x_train) :]
        cat_dims.append(max(n_classes, 1))

    if num_columns:
        train_num = x_train[num_columns].astype(np.float32).copy()
        test_num = x_test[num_columns].astype(np.float32).copy()
        medians = train_num.median()
        train_num = train_num.fillna(medians)
        test_num = test_num.fillna(medians)
        scaler = StandardScaler()
        x_num_train = scaler.fit_transform(train_num).astype(np.float32)
        x_num_test = scaler.transform(test_num).astype(np.float32)
    else:
        x_num_train = np.zeros((len(x_train), 0), dtype=np.float32)
        x_num_test = np.zeros((len(x_test), 0), dtype=np.float32)

    return EncodedTabularData(
        train_ids=prepared.train_raw[ID_COL].to_numpy(),
        test_ids=prepared.test_raw[ID_COL].to_numpy(),
        y=prepared.y.astype(np.float32).to_numpy(),
        x_num_train=x_num_train,
        x_num_test=x_num_test,
        x_cat_train=x_cat_train,
        x_cat_test=x_cat_test,
        num_columns=num_columns,
        cat_columns=cat_columns,
        cat_dims=cat_dims,
    )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"Transformer dim={dim} must be divisible by heads={heads}.")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_ff = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_out
        ff_input = self.norm2(tokens)
        return tokens + self.ff(ff_input)


class AttentionNet(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_dims: list[int],
        *,
        emb_dim: int = 24,
        hidden_dim: int = 192,
        n_layers: int = 4,
        heads: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.token_dim = hidden_dim
        self.num_numeric = num_numeric
        self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in cat_dims])
        self.num_token = nn.Linear(num_numeric, hidden_dim) if num_numeric > 0 else None
        self.cat_proj = nn.Linear(emb_dim, hidden_dim) if self.embeddings else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, heads=heads, dropout=dropout, ff_mult=2) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.size(0)
        token_chunks: list[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]

        if self.num_token is not None:
            token_chunks.append(self.num_token(x_num).unsqueeze(1))

        if self.embeddings:
            cat_tokens = torch.stack([emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1)
            token_chunks.append(self.cat_proj(cat_tokens))

        tokens = torch.cat(token_chunks, dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(-1)


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_dims: list[int],
        *,
        token_dim: int = 192,
        n_layers: int = 6,
        heads: int = 8,
        dropout: float = 0.2,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_numeric = num_numeric
        self.embeddings = nn.ModuleList([nn.Embedding(dim, token_dim) for dim in cat_dims])
        self.num_weight = nn.Parameter(torch.empty(num_numeric, token_dim))
        self.num_bias = nn.Parameter(torch.zeros(num_numeric, token_dim))
        if num_numeric > 0:
            nn.init.xavier_uniform_(self.num_weight)

        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(token_dim, heads=heads, dropout=dropout, ff_mult=ff_mult) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.size(0)
        token_chunks: list[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]

        if self.num_numeric > 0:
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            token_chunks.append(num_tokens)

        if self.embeddings:
            cat_tokens = torch.stack([emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1)
            token_chunks.append(cat_tokens)

        tokens = torch.cat(token_chunks, dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class TabResNet(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_dims: list[int],
        *,
        emb_dim: int = 24,
        hidden_dim: int = 320,
        n_blocks: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in cat_dims])
        input_dim = num_numeric + len(cat_dims) * emb_dim
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            cat_flat = torch.cat([emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1)
            x = torch.cat([x_num, cat_flat], dim=1)
        else:
            x = x_num
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def build_model(
    model_cfg: ModelRunConfig,
    *,
    num_numeric: int,
    cat_dims: list[int],
) -> nn.Module:
    family = model_cfg.family
    params = dict(model_cfg.model_params)
    if family == "attention":
        defaults = {"emb_dim": 24, "hidden_dim": 192, "n_layers": 4, "heads": 6, "dropout": 0.2}
        return AttentionNet(num_numeric, cat_dims, **(defaults | params))
    if family == "ft_transformer":
        defaults = {"token_dim": 192, "n_layers": 6, "heads": 8, "dropout": 0.2, "ff_mult": 4}
        return FTTransformer(num_numeric, cat_dims, **(defaults | params))
    if family == "tabresnet":
        defaults = {"emb_dim": 24, "hidden_dim": 320, "n_blocks": 6, "dropout": 0.2}
        return TabResNet(num_numeric, cat_dims, **(defaults | params))
    raise ValueError(f"Unsupported model family: {family}")


def predict_probabilities(
    model: nn.Module,
    *,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    require_cuda: bool,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_num, batch_cat in loader:
            batch_num = batch_num.to(device, non_blocking=True)
            batch_cat = batch_cat.to(device, non_blocking=True)
            if require_cuda and (not batch_num.is_cuda or not batch_cat.is_cuda):
                raise RuntimeError("Expected CUDA tensors during GPU-required inference.")
            with maybe_autocast(device, use_amp):
                logits = model(batch_num, batch_cat)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds, axis=0)


def run_model_cv(
    model_cfg: ModelRunConfig,
    encoded: EncodedTabularData,
    *,
    cv_folds: int,
    cv_seed: int,
    device: torch.device,
    deterministic: bool,
    require_cuda: bool,
) -> ModelResult:
    print("=" * 78)
    print(f"Training {model_cfg.name} ({model_cfg.family})")
    print(
        f"folds={cv_folds}, seeds={model_cfg.seeds}, epochs={model_cfg.epochs}, "
        f"batch_size={model_cfg.batch_size}, lr={model_cfg.lr}"
    )
    print("=" * 78)

    y_binary = encoded.y.astype(int)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    oof = np.zeros(len(encoded.y), dtype=np.float32)
    test_preds = np.zeros(len(encoded.test_ids), dtype=np.float32)
    fold_seed_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(encoded.x_num_train, y_binary), start=1):
        fold_val_preds: list[np.ndarray] = []
        fold_test_preds: list[np.ndarray] = []
        print(f"\nFold {fold_idx}/{cv_folds}:")

        train_dataset = TensorDataset(
            torch.from_numpy(encoded.x_num_train[train_idx]),
            torch.from_numpy(encoded.x_cat_train[train_idx]),
            torch.from_numpy(encoded.y[train_idx]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_cfg.batch_size,
            shuffle=True,
            num_workers=model_cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=model_cfg.num_workers > 0,
        )

        val_num = torch.from_numpy(encoded.x_num_train[valid_idx]).to(device, non_blocking=True)
        val_cat = torch.from_numpy(encoded.x_cat_train[valid_idx]).to(device, non_blocking=True)

        for seed in model_cfg.seeds:
            trial_seed = int(seed) + fold_idx * 1000
            set_global_seed(trial_seed, deterministic=deterministic)

            model = build_model(
                model_cfg,
                num_numeric=encoded.x_num_train.shape[1],
                cat_dims=encoded.cat_dims,
            ).to(device)
            if require_cuda and not next(model.parameters()).is_cuda:
                raise RuntimeError("Model parameters are not on CUDA while require_cuda=true.")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=model_cfg.lr,
                weight_decay=model_cfg.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_cfg.epochs)
            scaler = make_grad_scaler(device, enabled=model_cfg.amp and device.type == "cuda")
            use_amp = bool(model_cfg.amp and device.type == "cuda")

            best_auc = -np.inf
            best_epoch = 0
            epochs_ran = 0
            no_improve = 0
            best_state: dict[str, torch.Tensor] | None = None

            for epoch in range(1, model_cfg.epochs + 1):
                model.train()
                for batch_num, batch_cat, batch_y in train_loader:
                    batch_num = batch_num.to(device, non_blocking=True)
                    batch_cat = batch_cat.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    if require_cuda and (not batch_num.is_cuda or not batch_cat.is_cuda or not batch_y.is_cuda):
                        raise RuntimeError("Expected CUDA tensors during GPU-required training.")

                    optimizer.zero_grad(set_to_none=True)
                    with maybe_autocast(device, use_amp):
                        logits = model(batch_num, batch_cat)
                        loss = F.binary_cross_entropy_with_logits(logits, batch_y)

                    scaler.scale(loss).backward()
                    if model_cfg.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step()
                epochs_ran = epoch

                model.eval()
                with torch.no_grad():
                    with maybe_autocast(device, use_amp):
                        val_logits = model(val_num, val_cat)
                    val_probs = torch.sigmoid(val_logits).float().cpu().numpy()

                try:
                    val_auc = float(roc_auc_score(y_binary[valid_idx], val_probs))
                except ValueError:
                    val_auc = 0.5

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch
                    no_improve = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= model_cfg.patience:
                        break

            if best_state is None:
                raise RuntimeError(f"Training produced no checkpoint for {model_cfg.name}, fold={fold_idx}, seed={seed}")

            model.load_state_dict(best_state)
            val_pred = predict_probabilities(
                model,
                x_num=encoded.x_num_train[valid_idx],
                x_cat=encoded.x_cat_train[valid_idx],
                device=device,
                batch_size=max(model_cfg.batch_size, 4096),
                num_workers=model_cfg.num_workers,
                use_amp=use_amp,
                require_cuda=require_cuda,
            )
            test_pred = predict_probabilities(
                model,
                x_num=encoded.x_num_test,
                x_cat=encoded.x_cat_test,
                device=device,
                batch_size=max(model_cfg.batch_size, 4096),
                num_workers=model_cfg.num_workers,
                use_amp=use_amp,
                require_cuda=require_cuda,
            )

            fold_val_preds.append(val_pred)
            fold_test_preds.append(test_pred)
            fold_seed_rows.append(
                {
                    "fold": fold_idx,
                    "seed": int(seed),
                    "trial_seed": int(trial_seed),
                    "best_auc": float(best_auc),
                    "best_epoch": int(best_epoch),
                    "epochs_ran": int(epochs_ran),
                }
            )
            print(f"  seed={seed}: best_auc={best_auc:.6f}, best_epoch={best_epoch}, epochs_ran={epochs_ran}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fold_oof = np.mean(np.vstack(fold_val_preds), axis=0).astype(np.float32)
        fold_test = np.mean(np.vstack(fold_test_preds), axis=0).astype(np.float32)
        oof[valid_idx] = fold_oof
        test_preds += fold_test / cv_folds
        fold_auc = roc_auc_score(y_binary[valid_idx], fold_oof)
        print(f"  fold_avg_auc={fold_auc:.6f}")

        del val_num, val_cat

    final_auc = float(roc_auc_score(y_binary, oof))
    print(f"\n{model_cfg.name} final OOF AUC: {final_auc:.6f}")

    model_params_payload = {
        "family": model_cfg.family,
        "seeds": model_cfg.seeds,
        "epochs": model_cfg.epochs,
        "batch_size": model_cfg.batch_size,
        "lr": model_cfg.lr,
        "weight_decay": model_cfg.weight_decay,
        "patience": model_cfg.patience,
        "max_grad_norm": model_cfg.max_grad_norm,
        "num_workers": model_cfg.num_workers,
        "amp": model_cfg.amp,
        "model_params": model_cfg.model_params,
    }
    note = f"GPU custom {model_cfg.family} with seed-bagging ({len(model_cfg.seeds)} seeds) on {cv_folds}-fold CV."
    return ModelResult(
        run_name=model_cfg.name,
        family=model_cfg.family,
        oof=oof,
        test=test_preds,
        final_auc=final_auc,
        fold_seed_rows=fold_seed_rows,
        best_params_payload=model_params_payload,
        note=note,
    )


def _resolve_output_root(config_output_root: str) -> Path:
    root = Path(config_output_root)
    if root.is_absolute():
        return root.resolve()
    return (repo_root() / root).resolve()


def _ensure_run_dirs(output_root: Path, run_name: str) -> dict[str, Path]:
    models_dir = output_root / "models" / run_name
    oof_dir = output_root / "oof" / run_name
    submissions_dir = output_root / "submissions" / run_name
    logs_dir = output_root / "logs" / run_name
    for path in (models_dir, oof_dir, submissions_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "best_params": models_dir / "best_params.json",
        "oof": oof_dir / "oof_predictions.csv",
        "submission": submissions_dir / "submission.csv",
        "trials": logs_dir / "optuna_trials.csv",
        "summary": logs_dir / "best_run_summary.json",
    }


def persist_result(
    config: GPURunConfig,
    *,
    config_path: Path,
    output_root: Path,
    data_dir: Path,
    encoded: EncodedTabularData,
    result: ModelResult,
) -> None:
    paths = _ensure_run_dirs(output_root, result.run_name)

    oof_frame = pd.DataFrame(
        {
            ID_COL: encoded.train_ids,
            "y_true": encoded.y.astype(int),
            "oof_pred": result.oof,
        }
    )
    oof_frame.to_csv(paths["oof"], index=False)

    submission = pd.DataFrame(
        {
            ID_COL: encoded.test_ids,
            TARGET: np.clip(result.test, 0.0, 1.0),
        }
    )
    submission.to_csv(paths["submission"], index=False)

    trials_frame = pd.DataFrame(result.fold_seed_rows)
    trials_frame.to_csv(paths["trials"], index=False)

    paths["best_params"].write_text(json.dumps(result.best_params_payload, indent=2), encoding="utf-8")

    seeds_for_summary = result.best_params_payload.get("seeds", [])
    if not isinstance(seeds_for_summary, list):
        seeds_for_summary = []

    summary = {
        "config_path": str(config_path),
        "config": {
            "name": result.run_name,
            "competition_slug": config.competition_slug,
            "model_family": result.family,
            "feature_set": config.feature_set,
            "drop_columns": config.drop_columns,
            "data_dir": config.data_dir,
            "output_root": config.output_root,
            "n_trials": 0,
            "search_folds": config.cv_folds,
            "search_seeds": seeds_for_summary,
            "final_folds": config.cv_folds,
            "final_seeds": seeds_for_summary,
            "early_stop": int(result.best_params_payload.get("patience", 0)),
            "study_name": result.run_name,
            "storage": None,
            "timeout": None,
            "thread_count": -1,
            "with_class_weight": False,
            "fixed_params_path": None,
            "train_weight_mode": None,
            "train_weight_params": {},
            "notes": config.notes or result.note,
        },
        "data_dir": str(data_dir),
        "study_storage": "gpu-cv",
        "existing_trials": 0,
        "requested_trials": len(result.fold_seed_rows),
        "best_trial_number": None,
        "search_auc": None,
        "final_auc": float(result.final_auc),
        "best_params": result.best_params_payload,
        "fixed_params_path": None,
        "sample_weight_stats": None,
        "paths": {
            "submission": str(paths["submission"]),
            "oof": str(paths["oof"]),
            "trials": str(paths["trials"]),
            "summary": str(paths["summary"]),
            "best_params": str(paths["best_params"]),
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    experiment_runs_path = output_root / "logs" / "experiment_runs.csv"
    corr_to_best = compute_corr_to_best(
        experiment_runs_path,
        current_experiment=result.run_name,
        current_oof_path=paths["oof"],
    )
    append_experiment_run(
        experiment_runs_path,
        experiment=result.run_name,
        model_family=result.family,
        feature_set=config.feature_set,
        with_class_weight=False,
        search_auc=None,
        final_auc=float(result.final_auc),
        oof_corr_to_best=corr_to_best,
        submission_path=paths["submission"],
        summary_path=paths["summary"],
        notes=config.notes or result.note,
    )

    print(f"Saved summary:    {paths['summary']}")
    print(f"Saved submission: {paths['submission']}")
    print(f"Saved OOF:        {paths['oof']}")
    print(f"Saved trials:     {paths['trials']}")
    print(f"Saved params:     {paths['best_params']}")


def run_blend_search(
    members: list[ModelResult],
    blend_cfg: BlendConfig,
    *,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame, dict[str, float], list[ModelResult]]:
    selected = sorted(members, key=lambda item: item.final_auc, reverse=True)
    if blend_cfg.max_models is not None:
        selected = selected[: blend_cfg.max_models]

    names = [result.run_name for result in selected]
    oof_matrix = np.column_stack([result.oof for result in selected])
    test_matrix = np.column_stack([result.test for result in selected])
    rng = np.random.default_rng(blend_cfg.seed)

    weight_candidates = [np.eye(len(selected), dtype=np.float64)[idx] for idx in range(len(selected))]
    if blend_cfg.samples > 0:
        random_weights = rng.dirichlet(np.ones(len(selected), dtype=np.float64), size=blend_cfg.samples)
        weight_candidates.extend(random_weights)

    rows: list[dict[str, Any]] = []
    best_auc = -np.inf
    best_weights = weight_candidates[0]
    for trial_idx, weights in enumerate(weight_candidates):
        preds = oof_matrix @ weights
        auc = float(roc_auc_score(y_true, preds))
        row = {"trial": trial_idx, "auc": auc}
        row.update({f"w_{name}": float(weight) for name, weight in zip(names, weights)})
        rows.append(row)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    best_oof = (oof_matrix @ best_weights).astype(np.float32)
    best_test = (test_matrix @ best_weights).astype(np.float32)
    best_weight_map = {name: float(weight) for name, weight in zip(names, best_weights)}
    return best_oof, best_test, best_auc, pd.DataFrame(rows), best_weight_map, selected


def persist_blend_result(
    config: GPURunConfig,
    blend_cfg: BlendConfig,
    *,
    config_path: Path,
    output_root: Path,
    data_dir: Path,
    encoded: EncodedTabularData,
    members: list[ModelResult],
) -> None:
    blend_oof, blend_test, blend_auc, trials_frame, weight_map, selected = run_blend_search(
        members,
        blend_cfg,
        y_true=encoded.y.astype(int),
    )
    best_single = max(selected, key=lambda item: item.final_auc)
    improvement = float(blend_auc - best_single.final_auc)
    promoted = bool(improvement >= blend_cfg.min_improvement)

    result = ModelResult(
        run_name=blend_cfg.name,
        family="blend",
        oof=blend_oof,
        test=blend_test,
        final_auc=float(blend_auc),
        fold_seed_rows=trials_frame.to_dict(orient="records"),
        best_params_payload={
            "weights": weight_map,
            "members": [member.run_name for member in selected],
            "samples": blend_cfg.samples,
            "seed": blend_cfg.seed,
            "best_single_model": best_single.run_name,
            "best_single_auc": float(best_single.final_auc),
            "improvement_over_best_single": improvement,
            "promoted": promoted,
        },
        note=(
            f"OOF blend of {', '.join(member.run_name for member in selected)}; "
            f"improvement over best single={improvement:.6f}, promoted={promoted}."
        ),
    )
    persist_result(
        config,
        config_path=config_path,
        output_root=output_root,
        data_dir=data_dir,
        encoded=encoded,
        result=result,
    )
    print(f"Blend best single: {best_single.run_name} ({best_single.final_auc:.6f})")
    print(f"Blend final AUC:   {blend_auc:.6f}")
    print(f"Blend improvement: {improvement:.6f}")
    print(f"Blend promoted:    {promoted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom GPU neural models for FiiCode from YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "experiments" / "exp025_gpu_nn_trio.yaml",
        help="Path to GPU experiment YAML.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional data directory override.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output root override.")
    parser.add_argument("--skip-blend", action="store_true", help="Skip blend search even if config has blend enabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (repo_root() / args.config).resolve()
    config = GPURunConfig.from_yaml(config_path)

    if args.output_root is not None:
        config.output_root = str(args.output_root)
    data_dir_override = args.data_dir
    data_dir_candidate = data_dir_override or (Path(config.data_dir) if config.data_dir else None)
    data_dir = resolve_data_dir(data_dir_candidate)

    output_root = _resolve_output_root(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FiiCode GPU Neural Training")
    print("=" * 78)
    print(f"Config:            {config_path}")
    print(f"Experiment name:   {config.name}")
    print(f"Feature set:       {config.feature_set}")
    print(f"Data dir:          {data_dir}")
    print(f"Output root:       {output_root}")
    print(f"Require CUDA:      {config.require_cuda}")
    print(f"Deterministic:     {config.deterministic}")
    print(f"CV:                {config.cv_folds}-fold (seed={config.cv_seed})")
    print(f"Models:            {', '.join(model.name for model in config.models)}")
    print("=" * 78)

    device = resolve_device(config.require_cuda)
    set_global_seed(config.cv_seed, deterministic=config.deterministic)

    prepared = prepare_data(
        data_dir,
        feature_set=config.feature_set,
        drop_columns=config.drop_columns,
        load_test=True,
    )
    encoded = encode_tabular_inputs(prepared)
    print(
        f"Prepared features: {len(encoded.num_columns)} numeric + "
        f"{len(encoded.cat_columns)} categorical ({len(encoded.y)} train rows)"
    )

    members: list[ModelResult] = []
    for model_cfg in config.models:
        result = run_model_cv(
            model_cfg,
            encoded,
            cv_folds=config.cv_folds,
            cv_seed=config.cv_seed,
            device=device,
            deterministic=config.deterministic,
            require_cuda=config.require_cuda,
        )
        persist_result(
            config,
            config_path=config_path,
            output_root=output_root,
            data_dir=data_dir,
            encoded=encoded,
            result=result,
        )
        members.append(result)

    if config.blend and config.blend.enabled and not args.skip_blend and len(members) >= 2:
        print("\nRunning OOF blend search across trained GPU models...")
        persist_blend_result(
            config,
            config.blend,
            config_path=config_path,
            output_root=output_root,
            data_dir=data_dir,
            encoded=encoded,
            members=members,
        )

    print("\nRun complete.")
    print("Next step: validate and optionally submit the best candidate with src.submit.")


if __name__ == "__main__":
    main()
