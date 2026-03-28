#!/usr/bin/env python3
"""
TabNet Neural Network for Tabular Data
Self-trained from scratch (no pretrained weights)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=== TabNet Training ===")

# Load data
train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Feature engineering
def create_features(df, fit_encoders=None, fit_scalers=None):
    df = df.copy()
    
    # Categorical columns
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                'contact', 'month', 'poutcome']
    
    # Encode categoricals
    encoders = fit_encoders or {}
    for col in cat_cols:
        if col in df.columns:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = encoders[col].transform(df[col].astype(str))
    
    # Numeric features (log transforms)
    for col in ['balance', 'duration', 'campaign', 'pdays']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(np.clip(df[col], 0, None))
    
    # Interactions
    if 'duration' in df.columns and 'poutcome' in df.columns:
        df['duration_poutcome'] = df['duration'] * (df['poutcome'] == 2).astype(int)  # success encoded
    
    if 'balance' in df.columns and 'age' in df.columns:
        df['balance_age'] = df['balance'] / (df['age'] + 1)
    
    # Drop ID and target if present
    drop_cols = ['id', 'Subscribed']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df, encoders

# Prepare data
X_train_raw, encoders = create_features(train_df)
X_test_raw, _ = create_features(test_df, fit_encoders=encoders)

y_train = train_df['Subscribed'].values

# Ensure same columns
common_cols = list(set(X_train_raw.columns) & set(X_test_raw.columns))
X_train_raw = X_train_raw[common_cols]
X_test_raw = X_test_raw[common_cols]

print(f"Features: {len(common_cols)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw.values)
X_test_scaled = scaler.transform(X_test_raw.values)

# Simplified TabNet-style model with attention
class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x[:, :x.size(1)//2] * torch.sigmoid(x[:, x.size(1)//2:])

class TabNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_steps=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        
        # Shared feature transformer
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Step-specific transformers
        self.step_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(n_steps)
        ])
        
        # Attention for each step
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.BatchNorm1d(input_dim)
            ) for _ in range(n_steps)
        ])
        
        # Output projection
        self.output_fc = nn.Linear(hidden_dim * n_steps, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)
        
        outputs = []
        for step in range(self.n_steps):
            # Apply mask
            masked = x * prior_scales
            
            # Feature transformation
            h = self.shared_fc(masked)
            h = self.step_fcs[step](h)
            outputs.append(h)
            
            # Update attention
            attn = self.attentions[step](h)
            attn = F.softmax(attn, dim=-1)
            prior_scales = prior_scales * (1 - attn)
        
        # Aggregate
        aggregated = torch.cat(outputs, dim=1)
        return self.output_fc(aggregated)

# Training function
def train_tabnet(X_train, y_train, X_val, y_val, hidden_dim=128, n_steps=3, 
                 epochs=100, batch_size=1024, lr=0.01, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train.reshape(-1, 1))
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val.reshape(-1, 1))
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    model = TabNetBlock(X_train.shape[1], hidden_dim=hidden_dim, n_steps=n_steps)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    best_state = None
    patience = 15
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = torch.sigmoid(model(xb)).cpu().numpy()
                val_preds.extend(pred.flatten())
                val_true.extend(yb.numpy().flatten())
        
        auc = roc_auc_score(val_true, val_preds)
        
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    model.load_state_dict(best_state)
    return model, best_auc

# Cross-validation
print("\n5-Fold CV with 3-seed ensemble...")

n_seeds = 3
n_folds = 5

all_oof = np.zeros(len(X_train_scaled))
all_test_preds = []
fold_aucs = []

for seed in range(n_seeds):
    print(f"\nSeed {seed}:")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 1000 + 42)
    seed_oof = np.zeros(len(X_train_scaled))
    seed_test = np.zeros(len(X_test_scaled))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model, val_auc = train_tabnet(
            X_tr, y_tr, X_val, y_val,
            hidden_dim=128, n_steps=4, epochs=80, 
            batch_size=1024, lr=0.02, seed=seed * 100 + fold
        )
        
        # OOF predictions
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            val_pred = torch.sigmoid(model(torch.FloatTensor(X_val).to(device))).cpu().numpy().flatten()
            test_pred = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device))).cpu().numpy().flatten()
        
        seed_oof[val_idx] = val_pred
        seed_test += test_pred / n_folds
        
        print(f"  Fold {fold}: AUC={val_auc:.5f}")
        fold_aucs.append(val_auc)
    
    seed_auc = roc_auc_score(y_train, seed_oof)
    print(f"  Seed {seed} OOF AUC: {seed_auc:.5f}")
    
    all_oof += seed_oof / n_seeds
    all_test_preds.append(seed_test)

# Final ensemble
final_test = np.mean(all_test_preds, axis=0)
final_auc = roc_auc_score(y_train, all_oof)

print(f"\n=== Final TabNet Ensemble ===")
print(f"OOF AUC: {final_auc:.5f}")
print(f"Mean fold AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

# Save
import os
output_dir = 'outputs/oof/exp_tabnet_ensemble'
os.makedirs(output_dir, exist_ok=True)

# OOF predictions
oof_df = pd.DataFrame({
    'id': train_df['id'],
    'oof_pred': all_oof,
    'target': y_train
})
oof_df.to_csv(f'{output_dir}/oof_predictions.csv', index=False)

# Test predictions
test_pred_df = pd.DataFrame({
    'id': test_df['id'],
    'pred': final_test
})
test_pred_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)

print(f"\nSaved to {output_dir}/")
