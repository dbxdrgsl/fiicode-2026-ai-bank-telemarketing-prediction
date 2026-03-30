# FiiCode 2026 - Solution Reproduction Guide

## Best Submission: 0.94839 Public LB

### Quick Reproduction

```bash
# From repository root
cd /path/to/fiicode
source .venv/bin/activate  # or .venv-win on Windows
python reproduce_best.py
```

This will regenerate `outputs/submissions/FINAL_BEST_094839/submission.csv`

---

## Solution Architecture

### Overview

| Component | Model | CV AUC | Weight |
|-----------|-------|--------|--------|
| CatBoost | blend_buckets features (3 seeds × 5 folds) | 0.93638 | 80% |
| Neural Network | MLP ensemble | 0.92739 | 20% |
| **Blend** | Direct probability 80/20 | **0.93668** | - |

**Public LB: 0.94839**

### Why This Works

1. **CatBoost** captures complex categorical interactions via native encoding
2. **Neural Network** provides model diversity with different inductive bias
3. **Direct probability blending** (not rank) preserves calibration
4. **Multi-seed bagging** reduces variance

---

## Full Reproduction from Scratch

### Step 1: Environment Setup

```bash
# Python 3.10+ required
pip install -r requirements.txt

# For GPU NN training
pip install torch  # CUDA version for GPU
```

### Step 2: Train CatBoost (exp012)

```bash
python -m src.train --config experiments/exp012_blend_bucket_features_fixed.yaml
```

Expected output:
- `outputs/oof/exp012_blend_bucket_features_fixed/oof_predictions.csv`
- `outputs/submissions/exp012_blend_bucket_features_fixed/submission.csv`
- CV AUC: 0.93638

### Step 3: Train Neural Network

The neural network component provides diversity. Train using your preferred architecture.

Expected output:
- `outputs/oof/exp_nn/oof_predictions.csv`
- `outputs/submissions/exp_nn/submission.csv`
- CV AUC: ~0.927

### Step 4: Create Final Blend

```bash
python reproduce_best.py
```

Or manually:

```python
import pandas as pd

cb = pd.read_csv('outputs/submissions/exp012_blend_bucket_features_fixed/submission.csv')
nn = pd.read_csv('outputs/submissions/exp_nn/submission.csv')

# Direct probability blend (not rank normalized)
blend = 0.80 * cb['Subscribed'] + 0.20 * nn['Subscribed']

submission = pd.DataFrame({'id': cb['id'], 'Subscribed': blend})
submission.to_csv('submission.csv', index=False)
```

---

## Key Files

```
fiicode/
├── reproduce_best.py              # Quick reproduction script
├── src/
│   ├── features.py                # Feature engineering (blend_buckets)
│   ├── modeling.py                # CatBoost/LightGBM/XGBoost training
│   ├── train.py                   # Main training pipeline
│   └── blend.py                   # OOF-based blend optimization
├── gpu_train.py                   # Attention NN training
├── experiments/
│   ├── exp012_blend_bucket_features_fixed.yaml  # CatBoost config
│   └── exp026_gpu_nn_optimized.yaml             # Attention NN config
├── outputs/
│   ├── oof/                       # Out-of-fold predictions
│   ├── submissions/               # Test predictions
│   ├── models/                    # Saved model params
│   └── logs/                      # Run summaries
└── data/raw/
    ├── train.csv
    └── test.csv
```

---

## Feature Engineering (blend_buckets)

Key features from `src/features.py`:

```python
# Bucket features
- age_bucket: [<=25, 26-35, 36-45, 46-55, 56-65, 65+]
- campaign_bucket: [1, 2, 3-4, 5-9, 10+]
- pdays_bucket: [<=1w, 8-30d, 31-90d, 91-365d, 365d+, never]
- duration_bucket: [<=1m, 1-2m, 2-4m, 4-8m, 8m+]
- day_bucket: [early, mid, late]

# Categorical crosses
- job_education, job_marital, contact_month
- poutcome_month, loan_default
- contact_day_bucket, month_day_bucket
- history_state (previous outcome + seen indicator)

# Numeric transforms
- log1p transforms for duration, balance, campaign, previous, pdays
- balance_signed_log1p, balance_negative, balance_nonpositive
- Interaction features (campaign × previous, duration × campaign, etc.)
```

---

## Model Configurations

### CatBoost (exp012)

```yaml
iterations: 3500
learning_rate: 0.01377
depth: 6
l2_leaf_reg: 3.976
random_strength: 1.084
bagging_temperature: 0.733
bootstrap_type: Bayesian
auto_class_weights: Balanced
seeds: [42, 2024, 3407]
folds: 5
early_stopping: 250
```

### Attention NN (exp026)

```yaml
emb_dim: 24
hidden_dim: 192
n_layers: 4
heads: 6
dropout: 0.2
epochs: 90
batch_size: 1024
lr: 0.001
weight_decay: 0.01
patience: 16
amp: true
seeds: [42, 2024, 3407, 777, 1337, 1001, 2718]
folds: 5
```

---

## Verification

To verify the submission matches:

```bash
python -c "
import pandas as pd
import numpy as np

original = pd.read_csv('outputs/submissions/exp_blend_80_20/submission.csv')
reproduced = pd.read_csv('outputs/submissions/FINAL_BEST_094839/submission.csv')

diff = np.abs(original['Subscribed'] - reproduced['Subscribed']).max()
print(f'Max difference: {diff}')
print('Match!' if diff < 1e-9 else 'Mismatch')
"
```
