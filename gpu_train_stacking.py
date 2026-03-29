#!/usr/bin/env python3
"""
Neural stacking: Train neural meta-model on OOF predictions.
GPU-accelerated - learns optimal blending automatically.
"""

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

# Find all valid OOF predictions
def load_all_oof_predictions():
    """Load all OOF predictions for stacking."""
    oof_dir = repo_root() / 'outputs' / 'oof'
    models = {}
    
    exclude = ['exp022_target_encoding', 'exp020']  # Exclude overfit
    
    for model_dir in oof_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if any(ex in model_dir.name for ex in exclude):
            continue
            
        oof_file = model_dir / 'oof_predictions.csv'
        if oof_file.exists():
            try:
                df = pd.read_csv(oof_file)
                if 'oof_pred' in df.columns and 'y_true' in df.columns:
                    auc = roc_auc_score(df['y_true'], df['oof_pred'])
                    if 0.90 < auc < 0.99:  # Sanity check
                        models[model_dir.name] = df
                        print(f"  ✓ {model_dir.name}: {auc:.6f}")
            except Exception as e:
                print(f"  ✗ {model_dir.name}: {e}")
    
    return models


class StackingNN(nn.Module):
    """Neural network for stacking predictions."""
    def __init__(self, n_models, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(n_models, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)


def train_stacking():
    """Train neural stacking model."""
    print("=" * 70)
    print("NEURAL STACKING ENSEMBLE")
    print("=" * 70)
    
    # Load all OOF predictions
    print("\nLoading OOF predictions...")
    models = load_all_oof_predictions()
    
    if len(models) < 3:
        print("ERROR: Need at least 3 models for stacking")
        return
    
    print(f"\nUsing {len(models)} models for stacking")
    
    # Merge all predictions
    model_names = sorted(models.keys())
    merged = models[model_names[0]][['id', 'y_true']].copy()
    
    for name in model_names:
        merged[name] = models[name]['oof_pred'].values
    
    print(f"Merged shape: {merged.shape}")
    
    # Prepare features and target
    X = merged[model_names].values.astype(np.float32)
    y = merged['y_true'].values.astype(np.float32)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Cross-validation with multiple seeds
    seeds = [42, 43, 44, 45, 46]
    n_splits = 5
    
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed {seed}")
        print(f"{'='*70}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            print(f"\nFold {fold+1}/{n_splits}...")
            
            X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y[train_idx], dtype=torch.float32)
            X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.float32)
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            
            # Model
            model = StackingNN(len(model_names), hidden_dim=64).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            
            # Train
            best_val_auc = 0
            patience_counter = 0
            patience = 20
            
            for epoch in range(200):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    out = model(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    X_val_dev = X_val.to(device)
                    val_out = model(X_val_dev)
                    val_pred = torch.sigmoid(val_out).cpu().numpy()
                    val_auc = roc_auc_score(y_val.numpy(), val_pred)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_val_pred = val_pred
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            print(f"  Best AUC: {best_val_auc:.6f}")
            oof_preds[val_idx] += best_val_pred / len(seeds)
            fold_scores.append(best_val_auc)
    
    # Final OOF score
    final_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*70}")
    print(f"Final Stacking OOF AUC: {final_auc:.6f}")
    print(f"Mean fold AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    print(f"{'='*70}")
    
    # Save OOF
    out_dir = repo_root() / 'outputs'
    exp_name = 'exp050_neural_stacking'
    
    oof_dir = out_dir / 'oof' / exp_name
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({
        ID_COL: merged['id'],
        'y_true': y,
        'oof_pred': oof_preds
    })
    oof_df.to_csv(oof_dir / 'oof_predictions.csv', index=False)
    
    # Train on full data for test predictions
    print("\nTraining on full data for test predictions...")
    X_full = torch.tensor(X_scaled, dtype=torch.float32)
    y_full = torch.tensor(y, dtype=torch.float32)
    full_dataset = TensorDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)
    
    final_model = StackingNN(len(model_names), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(100):
        final_model.train()
        for batch_X, batch_y in full_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = final_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
    
    # Load test predictions
    print("\nLoading test predictions...")
    test_preds = {}
    for name in model_names:
        sub_path = out_dir / 'submissions' / name / 'submission.csv'
        if sub_path.exists():
            df = pd.read_csv(sub_path)
            test_preds[name] = df['Subscribed'].values
    
    if len(test_preds) == len(model_names):
        test_X = np.array([test_preds[name] for name in model_names]).T.astype(np.float32)
        test_X_scaled = scaler.transform(test_X)
        test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)
        
        final_model.eval()
        with torch.no_grad():
            test_out = final_model(test_X_tensor)
            test_pred = torch.sigmoid(test_out).cpu().numpy()
        
        # Save submission
        sub_dir = out_dir / 'submissions' / exp_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        test_ids = pd.read_csv(out_dir / 'submissions' / model_names[0] / 'submission.csv')['id']
        sub_df = pd.DataFrame({
            'id': test_ids,
            'Subscribed': test_pred
        })
        sub_df.to_csv(sub_dir / 'submission.csv', index=False)
        print(f"Saved submission to: {sub_dir / 'submission.csv'}")
    
    # Save summary
    log_dir = out_dir / 'logs' / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'experiment_name': exp_name,
        'method': 'neural_stacking',
        'n_models': len(model_names),
        'models': model_names,
        'final_auc': float(final_auc),
        'mean_fold_auc': float(np.mean(fold_scores)),
        'std_fold_auc': float(np.std(fold_scores))
    }
    
    with open(log_dir / 'best_run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Neural stacking complete!")
    print(f"   OOF AUC: {final_auc:.6f}")


if __name__ == '__main__':
    train_stacking()
