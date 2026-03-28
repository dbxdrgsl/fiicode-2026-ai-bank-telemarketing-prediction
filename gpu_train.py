"""
GPU TRAINING SCRIPT - Run this on your GPU machine
Transfer train.csv and test.csv to the same directory as this script
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("GPU NEURAL NETWORK TRAINING")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
y = train_df['Subscribed'].values

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Feature preparation
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'poutcome']
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Encode categoricals
cat_dims = []
all_cat_train = []
all_cat_test = []

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]]).astype(str)
    le.fit(combined)
    all_cat_train.append(le.transform(train_df[col].astype(str)))
    all_cat_test.append(le.transform(test_df[col].astype(str)))
    cat_dims.append(len(le.classes_))

X_cat_train = np.column_stack(all_cat_train)
X_cat_test = np.column_stack(all_cat_test)

# Prepare numeric features
train_num = train_df.copy()
test_num = test_df.copy()

for df in [train_num, test_num]:
    df['duration_log'] = np.log1p(df['duration'])
    df['balance_log'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['age_sq'] = df['age'] ** 2 / 1000
    df['campaign_log'] = np.log1p(df['campaign'])
    df['pdays_contacted'] = (df['pdays'] != -1).astype(float)

num_cols_ext = num_cols + ['duration_log', 'balance_log', 'age_sq', 'campaign_log', 'pdays_contacted']

scaler = StandardScaler()
X_num_train = scaler.fit_transform(train_num[num_cols_ext].values)
X_num_test = scaler.transform(test_num[num_cols_ext].values)

print(f"Features: {X_num_train.shape[1]} numeric + {len(cat_dims)} categorical")

# ============================================================
# MODEL 1: Attention Network (our best NN architecture)
# ============================================================
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class AttentionNet(nn.Module):
    def __init__(self, num_cont, cat_dims, emb_dim=16, hidden_dim=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, emb_dim) for d in cat_dims])
        self.num_proj = nn.Linear(num_cont, hidden_dim)
        self.cat_proj = nn.Linear(len(cat_dims) * emb_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn_blocks = nn.ModuleList([AttentionBlock(hidden_dim, heads=4, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        cat_embs = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        num_feat = self.num_proj(x_num).unsqueeze(1)
        cat_feat = self.cat_proj(cat_embs).unsqueeze(1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, num_feat, cat_feat], dim=1)
        for block in self.attn_blocks:
            x = block(x)
        return self.head(x[:, 0])

# ============================================================
# MODEL 2: ResNet-style Network
# ============================================================
class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        
    def forward(self, x):
        return F.relu(x + self.net(x))

class TabularResNet(nn.Module):
    def __init__(self, num_cont, cat_dims, emb_dim=16, hidden_dim=256, n_blocks=4, dropout=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(d, emb_dim) for d in cat_dims])
        input_dim = num_cont + len(cat_dims) * emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_num, x_cat):
        cat_embs = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        x = torch.cat([x_num, cat_embs], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model_class, model_kwargs, X_num_train, X_cat_train, y, X_num_test, X_cat_test, 
                n_folds=5, n_seeds=10, epochs=100, batch_size=256, lr=1e-3, name="model"):
    print(f"\n{'='*60}")
    print(f"Training {name} with {n_seeds} seeds per fold")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_oof = np.zeros(len(y))
    all_test = np.zeros(len(X_num_test))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_num_train, y)):
        print(f"\nFold {fold}: ", end="", flush=True)
        
        fold_val_preds = []
        fold_test_preds = []
        
        for seed in range(n_seeds):
            torch.manual_seed(42 + fold * 100 + seed)
            np.random.seed(42 + fold * 100 + seed)
            
            model = model_class(**model_kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            train_ds = TensorDataset(
                torch.FloatTensor(X_num_train[tr_idx]),
                torch.LongTensor(X_cat_train[tr_idx]),
                torch.FloatTensor(y[tr_idx])
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            
            best_auc = 0
            best_state = None
            patience = 15
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                for batch_num, batch_cat, batch_y in train_loader:
                    batch_num = batch_num.to(device)
                    batch_cat = batch_cat.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    out = model(batch_num, batch_cat).squeeze()
                    loss = F.binary_cross_entropy_with_logits(out, batch_y)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    val_num = torch.FloatTensor(X_num_train[val_idx]).to(device)
                    val_cat = torch.LongTensor(X_cat_train[val_idx]).to(device)
                    val_pred = torch.sigmoid(model(val_num, val_cat)).squeeze().cpu().numpy()
                    val_auc = roc_auc_score(y[val_idx], val_pred)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
            
            # Load best and predict
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            
            with torch.no_grad():
                val_num = torch.FloatTensor(X_num_train[val_idx]).to(device)
                val_cat = torch.LongTensor(X_cat_train[val_idx]).to(device)
                val_pred = torch.sigmoid(model(val_num, val_cat)).squeeze().cpu().numpy()
                
                test_num = torch.FloatTensor(X_num_test).to(device)
                test_cat = torch.LongTensor(X_cat_test).to(device)
                test_pred = torch.sigmoid(model(test_num, test_cat)).squeeze().cpu().numpy()
            
            fold_val_preds.append(val_pred)
            fold_test_preds.append(test_pred)
            print(f"s{seed}={best_auc:.4f} ", end="", flush=True)
        
        # Average seeds
        all_oof[val_idx] = np.mean(fold_val_preds, axis=0)
        all_test += np.mean(fold_test_preds, axis=0) / n_folds
        fold_auc = roc_auc_score(y[val_idx], all_oof[val_idx])
        print(f"| avg={fold_auc:.4f}")
    
    total_auc = roc_auc_score(y, all_oof)
    print(f"\n{name} Final OOF AUC: {total_auc:.5f}")
    
    return all_oof, all_test, total_auc

# ============================================================
# TRAIN ALL MODELS
# ============================================================
results = {}

# 1. Attention Network (10 seeds)
attn_oof, attn_test, attn_auc = train_model(
    AttentionNet, 
    {'num_cont': X_num_train.shape[1], 'cat_dims': cat_dims, 'emb_dim': 16, 'hidden_dim': 128, 'n_layers': 3, 'dropout': 0.2},
    X_num_train, X_cat_train, y, X_num_test, X_cat_test,
    n_folds=5, n_seeds=10, epochs=100, name="AttentionNet_10seed"
)
results['attention'] = (attn_oof, attn_test, attn_auc)

# 2. Deeper Attention (more layers)
deep_attn_oof, deep_attn_test, deep_attn_auc = train_model(
    AttentionNet,
    {'num_cont': X_num_train.shape[1], 'cat_dims': cat_dims, 'emb_dim': 24, 'hidden_dim': 192, 'n_layers': 5, 'dropout': 0.25},
    X_num_train, X_cat_train, y, X_num_test, X_cat_test,
    n_folds=5, n_seeds=10, epochs=100, name="DeepAttention_10seed"
)
results['deep_attention'] = (deep_attn_oof, deep_attn_test, deep_attn_auc)

# 3. ResNet (5 seeds - faster)
resnet_oof, resnet_test, resnet_auc = train_model(
    TabularResNet,
    {'num_cont': X_num_train.shape[1], 'cat_dims': cat_dims, 'emb_dim': 16, 'hidden_dim': 256, 'n_blocks': 4, 'dropout': 0.3},
    X_num_train, X_cat_train, y, X_num_test, X_cat_test,
    n_folds=5, n_seeds=5, epochs=80, name="ResNet_5seed"
)
results['resnet'] = (resnet_oof, resnet_test, resnet_auc)

# ============================================================
# SAVE RESULTS
# ============================================================
os.makedirs('gpu_outputs', exist_ok=True)

for name, (oof, test, auc) in results.items():
    pd.DataFrame({'oof_pred': oof}).to_csv(f'gpu_outputs/{name}_oof.csv', index=False)
    pd.DataFrame({'id': test_df['id'], 'Subscribed': test}).to_csv(f'gpu_outputs/{name}_test.csv', index=False)
    print(f"Saved {name}: AUC={auc:.5f}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for name, (_, _, auc) in results.items():
    print(f"{name}: {auc:.5f}")

# Check correlations
print("\nCorrelations:")
for n1, (oof1, _, _) in results.items():
    for n2, (oof2, _, _) in results.items():
        if n1 < n2:
            corr = np.corrcoef(oof1, oof2)[0, 1]
            print(f"  {n1} vs {n2}: {corr:.4f}")

print("\n" + "="*70)
print("DONE! Transfer gpu_outputs/*.csv back to your main machine")
print("="*70)
