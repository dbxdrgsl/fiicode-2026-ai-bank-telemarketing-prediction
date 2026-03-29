#!/usr/bin/env python3
"""
GNN training script for customer similarity graph.

Uses PyTorch Geometric to train Graph Neural Networks that capture
customer relationships CatBoost cannot see.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool

try:
    import yaml
except ImportError:
    yaml = None

from src.config import repo_root
from src.features import ID_COL, TARGET, prepare_data, resolve_data_dir
from src.tracking import append_experiment_run


def build_customer_graph(X_df, k=15):
    """Build k-NN customer similarity graph."""
    print(f"Building k-NN graph with k={k}...")
    
    # Select features for similarity
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X_df[numeric_cols].fillna(0).values
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
    nbrs.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    
    # Create edge list (skip self-loops)
    edge_list = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip first (self)
            edge_list.append([i, neighbor])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"Graph: {X_df.shape[0]} nodes, {edge_index.shape[1]} edges")
    
    return edge_index, scaler


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
        self.fc = nn.Linear(hidden_channels * heads, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def train_gnn_fold(model, data, train_idx, val_idx, epochs=100, lr=1e-3, device='cuda'):
    """Train GNN for one fold."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    model = model.to(device)
    data = data.to(device)
    
    best_val_auc = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = torch.sigmoid(out[val_idx]).cpu().numpy()
            val_true = data.y[val_idx].cpu().numpy()
            val_auc = roc_auc_score(val_true, val_pred)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch+1}, best AUC: {best_val_auc:.6f}")
            break
    
    return best_val_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    experiment_name = config['experiment_name']
    feature_set = config.get('feature_set', 'blend_buckets')
    model_type = config['model']['family']  # 'graphsage' or 'gat'
    seeds = config['model'].get('seeds', [42, 43, 44, 45, 46])
    k_neighbors = config['model'].get('k_neighbors', 15)
    hidden_dim = config['model'].get('hidden_dim', 128)
    num_layers = config['model'].get('num_layers', 2)
    epochs = config['model'].get('epochs', 100)
    lr = config['model'].get('lr', 1e-3)
    n_splits = config.get('n_splits', 5)
    
    print(f"\n{'='*60}")
    print(f"GNN Training: {experiment_name}")
    print(f"Model: {model_type}, Seeds: {seeds}, K: {k_neighbors}")
    print(f"{'='*60}\n")
    
    # Load data
    data_dir = resolve_data_dir()
    X_train, y_train, X_test = prepare_data(data_dir, feature_set=feature_set)
    
    # Build graph
    edge_index, scaler = build_customer_graph(X_train, k=k_neighbors)
    
    # Prepare features
    feature_cols = [c for c in X_train.columns if c not in [ID_COL, TARGET]]
    X_train_numeric = X_train[feature_cols].fillna(0).values
    X_test_numeric = X_test[feature_cols].fillna(0).values
    
    X_train_scaled = scaler.transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    
    # Create PyG Data object for training
    train_data = Data(
        x=torch.tensor(X_train_scaled, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_train.values, dtype=torch.float32)
    )
    
    # Multi-seed CV
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    oof_preds = np.zeros(len(y_train))
    test_preds = []
    fold_scores = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"Fold {fold+1}/{n_splits}...")
            
            # Create model
            if model_type == 'graphsage':
                model = GraphSAGE(X_train_scaled.shape[1], hidden_dim, num_layers)
            elif model_type == 'gat':
                model = GAT(X_train_scaled.shape[1], hidden_dim, num_layers)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train
            val_auc = train_gnn_fold(model, train_data, train_idx, val_idx, 
                                     epochs=epochs, lr=lr, device=device)
            fold_scores.append(val_auc)
            
            # OOF predictions
            model.eval()
            with torch.no_grad():
                train_data = train_data.to(device)
                out = model(train_data.x, train_data.edge_index)
                val_pred = torch.sigmoid(out[val_idx]).cpu().numpy()
                oof_preds[val_idx] += val_pred / len(seeds)
            
            # Test predictions
            test_data = Data(
                x=torch.tensor(X_test_scaled, dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long)  # No edges for test
            )
            test_data = test_data.to(device)
            with torch.no_grad():
                test_out = model(test_data.x, test_data.edge_index)
                test_pred = torch.sigmoid(test_out).cpu().numpy()
                test_preds.append(test_pred)
    
    # Compute final scores
    final_oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n{'='*60}")
    print(f"Final OOF AUC: {final_oof_auc:.6f}")
    print(f"Mean fold AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    print(f"{'='*60}\n")
    
    # Average test predictions
    final_test_pred = np.mean(test_preds, axis=0)
    
    # Save outputs
    out_dir = repo_root() / 'outputs'
    
    # OOF predictions
    oof_dir = out_dir / 'oof' / experiment_name
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({
        ID_COL: X_train[ID_COL],
        'y_true': y_train,
        'oof_pred': oof_preds
    })
    oof_df.to_csv(oof_dir / 'oof_predictions.csv', index=False)
    
    # Submission
    sub_dir = out_dir / 'submissions' / experiment_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame({
        ID_COL: X_test[ID_COL],
        'Subscribed': final_test_pred
    })
    sub_df.to_csv(sub_dir / 'submission.csv', index=False)
    
    # Summary
    log_dir = out_dir / 'logs' / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'oof_auc': final_oof_auc,
        'mean_fold_auc': float(np.mean(fold_scores)),
        'std_fold_auc': float(np.std(fold_scores)),
        'seeds': seeds,
        'k_neighbors': k_neighbors,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }
    with open(log_dir / 'best_run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Track in experiment journal
    append_experiment_run(
        experiment_name=experiment_name,
        oof_auc=final_oof_auc,
        config_path=str(args.config),
        submission_path=str(sub_dir / 'submission.csv'),
        note=f"GNN {model_type} k={k_neighbors}"
    )
    
    print(f"Saved outputs to {out_dir}")


if __name__ == '__main__':
    main()
