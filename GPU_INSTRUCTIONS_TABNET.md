# TabNet GPU Training Instructions

## Overview

TabNet is a neural network architecture that uses **instance-wise feature selection** via attention mechanism. Unlike CatBoost which learns fixed global feature importance, TabNet dynamically decides which features matter for **each individual prediction**.

This creates an orthogonal modeling approach ideal for blending with tree-based models.

## Key TabNet Characteristics

1. **Sparse Attention Masks**: Most features are zeroed out (masked) for each sample
2. **Sequential Decision Steps**: Features are selected across multiple reasoning steps (n_steps)
3. **Interpretability**: Built-in feature importance and attention masks
4. **Dynamic Feature Selection**: Different features used for different predictions

## Required Packages

Ensure `pytorch-tabnet` is installed in your GPU environment:

```bash
pip install pytorch-tabnet
```

Dependencies:
- PyTorch (with CUDA support)
- scikit-learn
- pandas
- numpy

## Experiments

### exp046_tabnet (Base Configuration)
- **n_d/n_a**: 64 (embedding dimensions)
- **n_steps**: 5 (decision steps)
- **seeds**: 5 seeds for robustness
- **batch_size**: 1024
- **lr**: 0.02
- **patience**: 50 epochs early stopping
- **Expected runtime**: ~45-60 minutes on V100/A100

**Command**:
```bash
python gpu_train_tabnet.py --config experiments/exp046_tabnet.yaml
```

### exp047_tabnet_deep (Deep Configuration)
- **n_d/n_a**: 128 (increased capacity)
- **n_steps**: 7 (more reasoning steps)
- **seeds**: 5 seeds
- **lr**: 0.015 (reduced for stability)
- **Expected runtime**: ~1.5-2 hours on V100/A100

**Command**:
```bash
python gpu_train_tabnet.py --config experiments/exp047_tabnet_deep.yaml
```

## Output Files

Each experiment produces:

1. **OOF Predictions**: `outputs/oof/<exp_name>/oof_predictions.csv`
2. **Submission**: `outputs/submissions/<exp_name>/submission.csv`
3. **Best Params**: `outputs/models/<exp_name>/best_params.json`
4. **Summary**: `outputs/logs/<exp_name>/best_run_summary.json`
5. **Trials**: `outputs/logs/<exp_name>/optuna_trials.csv` (fold/seed metrics)
6. **Feature Importance**: `outputs/logs/<exp_name>/feature_importance_comparison.csv`

## Feature Importance Interpretation

TabNet's feature importance is calculated from attention masks across all decision steps:

- **High importance**: Feature is frequently selected in early decision steps
- **Low importance**: Feature is rarely used or selected in later steps
- **Sparse selection**: Most features have near-zero importance for individual predictions

### Comparison with CatBoost

Expected differences:

1. **CatBoost**: Global importance based on split gains across all trees
2. **TabNet**: Instance-wise importance averaged across samples
3. **Divergence**: Features that work differently for subgroups may rank differently

To compare:
```bash
# View TabNet importance
cat outputs/logs/exp046_tabnet/feature_importance_comparison.csv

# Compare with CatBoost (manual)
# CatBoost importance typically stored in model artifacts or can be extracted from training logs
```

## GPU Memory Requirements

- **exp046**: ~4-6 GB VRAM
- **exp047**: ~8-10 GB VRAM

If OOM occurs:
- Reduce `batch_size` to 512 or 256
- Reduce `n_d` and `n_a` to 32 or 48
- Reduce `n_steps` to 3 or 4

## Monitoring Training

TabNet prints per-fold validation AUC:
```
--- Fold 1/5 (Seed 42) ---
epoch 0  | loss: 0.5234 | val_auc: 0.8756 |  0:00:12s
epoch 1  | loss: 0.4987 | val_auc: 0.8923 |  0:00:24s
...
Fold 1 AUC: 0.9123
```

## Expected Performance

Based on similar tabular datasets:

- **OOF CV AUC**: 0.930-0.937 (competitive with exp012 CatBoost: 0.936)
- **LB Score**: Should be within 0.005 of CV (TabNet generalizes well)
- **Training stability**: Multiple seeds reduce variance from ~0.002 to ~0.0005

## Attention Mask Analysis

TabNet saves attention masks showing which features were used for each prediction. To extract and analyze:

```python
# During training, clf.explain() can extract masks
# This is sample-specific and shows:
# - Which features were selected (non-zero mask values)
# - Which decision step selected each feature
# - Overall sparsity (% of features used per sample)
```

## Blending with CatBoost

TabNet is orthogonal to CatBoost, making it ideal for blending:

1. **CatBoost**: Tree splits, fixed feature importance
2. **TabNet**: Attention mechanism, dynamic feature selection
3. **Blend strategy**: Simple average or weighted by CV scores

Expected blend improvement: +0.001 to +0.003 AUC

## Troubleshooting

### Training is slow
- Check GPU utilization: `nvidia-smi`
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU-bound

### Validation AUC is unstable
- Increase `n_steps` for more stable attention
- Increase `virtual_batch_size` for more stable batch norm
- Add more seeds (already using 5)

### OOF score << validation scores
- Check for data leakage in feature engineering
- Verify fold splits match exp012 (same seed=42 base)

### Feature importance is too uniform
- Increase `lambda_sparse` to enforce more sparsity
- Increase `gamma` for stronger feature selection pressure

## Next Steps After Training

1. **Check OOF score**: Should be competitive with exp012 (0.936)
2. **Inspect feature importance**: Compare top 20 features with CatBoost
3. **Blend with exp012**: Use `src/blend.py` for ensemble
4. **Submit if promising**: Use `src/submit.py` workflow

## Full Training Workflow

```bash
# 1. Train base TabNet
python gpu_train_tabnet.py --config experiments/exp046_tabnet.yaml

# 2. Check results
cat outputs/logs/exp046_tabnet/best_run_summary.json

# 3. Compare feature importance
head -20 outputs/logs/exp046_tabnet/feature_importance_comparison.csv

# 4. Blend with CatBoost
python -m src.blend --name exp046_cb_blend \
  --summaries outputs/logs/exp012_blend_bucket_features_fixed/best_run_summary.json \
               outputs/logs/exp046_tabnet/best_run_summary.json

# 5. Submit blend
python -m src.submit --submission outputs/submissions/exp046_cb_blend/submission.csv \
  --message "exp046 TabNet+CatBoost blend"
```

## Hyperparameter Tuning (Optional)

If initial results are promising, tune:

1. **n_steps**: [3, 5, 7, 9] - more steps = more complex reasoning
2. **lambda_sparse**: [1e-5, 1e-4, 1e-3] - controls sparsity
3. **gamma**: [1.0, 1.5, 2.0] - attention sharpness
4. **n_d/n_a**: [32, 64, 128] - model capacity

Priority order: n_steps > lambda_sparse > gamma > n_d/n_a

## References

- TabNet paper: https://arxiv.org/abs/1908.07442
- PyTorch TabNet: https://github.com/dreamquark-ai/tabnet
- Instance-wise feature selection is key differentiator from tree models
