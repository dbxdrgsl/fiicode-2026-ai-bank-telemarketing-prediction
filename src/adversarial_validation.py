"""Adversarial validation to detect train/test distribution shift."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features import prepare_data


def adversarial_validation(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Train a classifier to distinguish train from test data.
    
    Args:
        train_x: Training features
        test_x: Test features
        n_folds: Number of CV folds
        seed: Random seed
    
    Returns:
        train_probs: Probability of each train sample being from test
        results: Dict with AUC, feature importance, and diagnostics
    """
    # Create copies and encode categorical features
    train_adv = train_x.copy()
    test_adv = test_x.copy()
    
    # Identify categorical columns
    cat_cols = train_adv.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Found {len(cat_cols)} categorical columns")
    
    # Label encode categorical columns for LightGBM
    from sklearn.preprocessing import LabelEncoder
    
    for col in cat_cols:
        le = LabelEncoder()
        # Combine train and test to ensure consistent encoding
        combined_col = pd.concat([train_adv[col], test_adv[col]], axis=0)
        le.fit(combined_col)
        train_adv[col] = le.transform(train_adv[col])
        test_adv[col] = le.transform(test_adv[col])
    
    feature_cols = train_adv.columns.tolist()
    
    # Create adversarial target: train=0, test=1
    train_adv["is_test"] = 0
    test_adv["is_test"] = 1
    
    combined = pd.concat([train_adv, test_adv], axis=0, ignore_index=True)
    X = combined[feature_cols]
    y = combined["is_test"]
    
    print(f"Combined shape: {X.shape}")
    print(f"Train samples: {(y == 0).sum()}, Test samples: {(y == 1).sum()}")
    
    # Cross-validation predictions
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    feature_importance = np.zeros(len(feature_cols))
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train LightGBM classifier
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": seed,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        
        # Predict
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        feature_importance += model.feature_importance(importance_type="gain")
        
        # Fold AUC
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")
    
    # Overall AUC
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'=' * 60}")
    print(f"Overall Adversarial Validation AUC: {overall_auc:.5f}")
    print(f"{'=' * 60}")
    
    # Feature importance
    feature_importance /= n_folds
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 features with train/test distribution shift:")
    print(importance_df.head(10).to_string(index=False))
    
    # Extract train probabilities only
    train_probs = oof_preds[:len(train_x)]
    
    results = {
        "overall_auc": overall_auc,
        "fold_aucs": fold_aucs,
        "feature_importance": importance_df.to_dict(orient="records"),
        "train_probs_stats": {
            "min": float(train_probs.min()),
            "max": float(train_probs.max()),
            "mean": float(train_probs.mean()),
            "median": float(np.median(train_probs)),
            "std": float(train_probs.std()),
        },
    }
    
    return train_probs, results


def compute_sample_weights(
    train_probs: np.ndarray,
    max_weight: float = 10.0,
) -> np.ndarray:
    """
    Compute sample weights from adversarial validation probabilities.
    
    Args:
        train_probs: Probability of each train sample being from test
        max_weight: Maximum weight cap to prevent extreme outliers
    
    Returns:
        weights: Sample weights
    """
    # Weight = 1 / (1 - probability_of_being_test)
    # Higher probability of being test => higher weight
    weights = 1.0 / (1.0 - train_probs + 1e-6)
    
    # Cap weights
    weights = np.clip(weights, 1.0, max_weight)
    
    # Normalize to mean=1
    weights = weights / weights.mean()
    
    print(f"\nSample weight statistics:")
    print(f"  Min: {weights.min():.4f}")
    print(f"  Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Median: {np.median(weights):.4f}")
    print(f"  Std: {weights.std():.4f}")
    
    return weights


def main():
    """Run adversarial validation and save weights."""
    print("=" * 60)
    print("ADVERSARIAL VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Load data using the standard feature pipeline
    print("\nLoading data...")
    prepared = prepare_data(
        data_dir=Path("data/raw"),
        feature_set="blend_buckets",
        drop_columns=[],
        load_test=True,
    )
    
    print(f"Train shape: {prepared.x.shape}")
    print(f"Test shape: {prepared.x_test.shape}")
    print(f"Number of features: {len(prepared.x.columns)}")
    
    # Run adversarial validation
    print("\n" + "=" * 60)
    print("Running adversarial validation...")
    print("=" * 60)
    train_probs, results = adversarial_validation(
        train_x=prepared.x,
        test_x=prepared.x_test,
        n_folds=5,
        seed=42,
    )
    
    # Compute sample weights
    print("\n" + "=" * 60)
    print("Computing sample weights...")
    print("=" * 60)
    weights = compute_sample_weights(train_probs, max_weight=10.0)
    
    # Save weights
    output_dir = Path("outputs/weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_df = pd.DataFrame({
        "id": prepared.train_raw["id"],
        "weight": weights,
        "adv_prob": train_probs,
    })
    weights_path = output_dir / "adversarial_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"\nSaved weights to: {weights_path}")
    
    # Save analysis results
    analysis_dir = Path("outputs/logs/adversarial_validation")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_path = analysis_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")
    
    # Create human-readable report
    report_path = analysis_dir / "analysis.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ADVERSARIAL VALIDATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Overall AUC: {results['overall_auc']:.5f}\n")
        f.write(f"Fold AUCs: {[f'{auc:.5f}' for auc in results['fold_aucs']]}\n\n")
        
        if results['overall_auc'] > 0.6:
            f.write("⚠️  WARNING: AUC > 0.6 indicates SIGNIFICANT distribution shift!\n\n")
        elif results['overall_auc'] > 0.55:
            f.write("⚠️  NOTICE: AUC > 0.55 indicates MODERATE distribution shift.\n\n")
        else:
            f.write("✓ AUC < 0.55 indicates minimal distribution shift.\n\n")
        
        f.write("Top 10 features with train/test distribution shift:\n")
        f.write("-" * 60 + "\n")
        for i, feat in enumerate(results['feature_importance'][:10], 1):
            f.write(f"{i:2d}. {feat['feature']:40s} {feat['importance']:12.2f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("SAMPLE WEIGHT STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        stats = results['train_probs_stats']
        f.write(f"Adversarial probabilities:\n")
        f.write(f"  Min:    {stats['min']:.6f}\n")
        f.write(f"  Max:    {stats['max']:.6f}\n")
        f.write(f"  Mean:   {stats['mean']:.6f}\n")
        f.write(f"  Median: {stats['median']:.6f}\n")
        f.write(f"  Std:    {stats['std']:.6f}\n\n")
        
        f.write(f"Sample weights (capped at 10.0):\n")
        f.write(f"  Min:    {weights.min():.4f}\n")
        f.write(f"  Max:    {weights.max():.4f}\n")
        f.write(f"  Mean:   {weights.mean():.4f}\n")
        f.write(f"  Median: {np.median(weights):.4f}\n")
        f.write(f"  Std:    {weights.std():.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("Sample weights are computed as: weight = 1 / (1 - p_test)\n")
        f.write("Where p_test is the probability of being from the test set.\n\n")
        f.write("High weight => sample looks like test data => weight it more.\n")
        f.write("Low weight => sample looks unlike test data => weight it less.\n\n")
        f.write("This reweighting should help the model focus on training samples\n")
        f.write("that are most representative of the test distribution.\n")
    
    print(f"Saved report to: {report_path}")
    
    print("\n" + "=" * 60)
    print("ADVERSARIAL VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review analysis: {report_path}")
    print(f"2. Use weights in training: {weights_path}")
    print(f"3. Run exp043 with adversarial weights")


if __name__ == "__main__":
    main()
