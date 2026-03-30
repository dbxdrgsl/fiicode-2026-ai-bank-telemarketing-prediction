#!/usr/bin/env python3
"""
REPRODUCTION SCRIPT - FiiCode 2026 AI Competition
==================================================
Best Submission: 0.94839 Public LB

This script reproduces the best submission exactly.
Run from the repository root with the virtual environment activated.

Components:
1. CatBoost with blend_buckets features (3 seeds × 5 folds) - CV: 0.93638
2. Attention Neural Network (7 seeds × 5 folds) - CV: 0.93336  
3. 80/20 rank-based blend - CV: 0.93677 → LB: 0.94839
"""

import sys
from pathlib import Path

# Verify we're in the right directory
repo_root = Path(__file__).parent
if not (repo_root / "src" / "features.py").exists():
    print("ERROR: Run this script from the repository root")
    sys.exit(1)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rank_normalize(values):
    """Convert predictions to percentile ranks (0-1)"""
    return pd.Series(values).rank(method='average', pct=True).to_numpy()


def main():
    print("=" * 70)
    print("REPRODUCING BEST SUBMISSION (0.94839 Public LB)")
    print("=" * 70)
    
    # Paths
    oof_dir = repo_root / "outputs" / "oof"
    sub_dir = repo_root / "outputs" / "submissions"
    
    # Load OOF predictions
    print("\n[1] Loading OOF predictions...")
    
    cb_oof = pd.read_csv(oof_dir / "exp012_blend_bucket_features_fixed" / "oof_predictions.csv")
    nn_oof = pd.read_csv(oof_dir / "exp_nn" / "oof_predictions.csv")
    
    y_true = cb_oof['y_true']
    
    cb_auc = roc_auc_score(y_true, cb_oof['oof_pred'])
    nn_auc = roc_auc_score(y_true, nn_oof['oof_pred'])
    
    print(f"   CatBoost OOF AUC: {cb_auc:.6f}")
    print(f"   Neural Network OOF AUC: {nn_auc:.6f}")
    
    # Load test predictions
    print("\n[2] Loading test predictions...")
    
    cb_sub = pd.read_csv(sub_dir / "exp012_blend_bucket_features_fixed" / "submission.csv")
    nn_sub = pd.read_csv(sub_dir / "exp_nn" / "submission.csv")
    
    # Create 80/20 blend using direct probability average (not rank)
    print("\n[3] Creating 80/20 blend...")
    
    # OOF blend for CV verification
    blend_oof = 0.80 * cb_oof['oof_pred'] + 0.20 * nn_oof['oof_pred']
    blend_auc = roc_auc_score(y_true, blend_oof)
    print(f"   Blend OOF AUC: {blend_auc:.6f}")
    
    # Test blend (direct probability blend, not rank normalized)
    blend_test = 0.80 * cb_sub['Subscribed'] + 0.20 * nn_sub['Subscribed']
    
    # Create submission
    submission = pd.DataFrame({
        'id': cb_sub['id'],
        'Subscribed': blend_test
    })
    
    output_path = repo_root / "outputs" / "submissions" / "FINAL_BEST_094839" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n[4] Submission saved to: {output_path}")
    
    # Verify against original
    print("\n[5] Verification...")
    original = pd.read_csv(sub_dir / "exp_blend_80_20" / "submission.csv")
    
    diff = np.abs(submission['Subscribed'] - original['Subscribed']).max()
    print(f"   Max difference from original: {diff:.10f}")
    
    if diff < 1e-10:
        print("   ✓ EXACT MATCH with original 0.94839 submission")
    else:
        corr = np.corrcoef(submission['Subscribed'], original['Subscribed'])[0, 1]
        print(f"   Correlation with original: {corr:.6f}")
    
    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)
    print(f"\nSubmission file: {output_path}")
    print("Expected Public LB: 0.94839")


if __name__ == "__main__":
    main()
