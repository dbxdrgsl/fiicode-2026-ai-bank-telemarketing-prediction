from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
except ImportError:
    LGBMClassifier = None
    lgb_early_stopping = None
    lgb_log_evaluation = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


ModelFamily = Literal["catboost", "lightgbm", "xgboost", "mlp"]


def normalize_model_family(model_family: str) -> ModelFamily:
    normalized = model_family.strip().lower()
    allowed = {"catboost", "lightgbm", "xgboost", "mlp"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported model_family={model_family!r}. Expected one of: {sorted(allowed)}")
    return normalized  # type: ignore[return-value]


def _scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives <= 0:
        return 1.0
    return max(negatives / positives, 1.0)


def _prepare_one_hot_matrices(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    categorical_columns: list[str],
    *,
    x_test: pd.DataFrame | None = None,
):
    local_cats = [column for column in categorical_columns if column in x_train.columns]
    numeric_columns = [column for column in x_train.columns if column not in local_cats]
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), local_cats),
            ("numeric", "passthrough", numeric_columns),
        ],
        sparse_threshold=1.0,
    )
    x_train_encoded = transformer.fit_transform(x_train)
    x_valid_encoded = transformer.transform(x_valid)
    x_test_encoded = transformer.transform(x_test) if x_test is not None else None
    return x_train_encoded, x_valid_encoded, x_test_encoded


def compute_train_sample_weights(
    mode: str | None,
    x: pd.DataFrame,
    x_test: pd.DataFrame | None,
    categorical_columns: list[str],
    *,
    thread_count: int,
    weight_params: dict | None = None,
) -> np.ndarray | None:
    normalized_mode = (mode or "").strip().lower()
    if not normalized_mode:
        return None
    if normalized_mode != "adversarial":
        raise ValueError(f"Unsupported train_weight_mode={mode!r}. Expected 'adversarial' or null.")
    if x_test is None:
        raise ValueError("train_weight_mode='adversarial' requires test features to be loaded.")
    if CatBoostClassifier is None:
        raise ModuleNotFoundError("catboost not installed. Install with: pip install catboost")

    params = {
        "folds": 5,
        "seed": 42,
        "iterations": 250,
        "learning_rate": 0.05,
        "depth": 6,
        "early_stop": 50,
        "power": 0.5,
        "min_weight": 0.6,
        "max_weight": 2.5,
    }
    if weight_params:
        params.update(weight_params)

    combined = pd.concat([x, x_test], axis=0, ignore_index=True)
    labels = np.concatenate([np.zeros(len(x), dtype=int), np.ones(len(x_test), dtype=int)])
    cat_idx = [combined.columns.get_loc(column) for column in categorical_columns if column in combined.columns]
    oof = np.zeros(len(combined), dtype=float)
    splitter = StratifiedKFold(
        n_splits=int(params["folds"]),
        shuffle=True,
        random_state=int(params["seed"]),
    )

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(combined, labels), start=1):
        model = CatBoostClassifier(
            iterations=int(params["iterations"]),
            learning_rate=float(params["learning_rate"]),
            depth=int(params["depth"]),
            loss_function="Logloss",
            eval_metric="AUC",
            allow_writing_files=False,
            verbose=False,
            thread_count=thread_count,
            random_seed=int(params["seed"]) + fold,
        )
        model.fit(
            combined.iloc[train_idx],
            labels[train_idx],
            cat_features=cat_idx,
            eval_set=(combined.iloc[valid_idx], labels[valid_idx]),
            use_best_model=True,
            early_stopping_rounds=int(params["early_stop"]),
        )
        oof[valid_idx] = model.predict_proba(combined.iloc[valid_idx])[:, 1]

    train_probs = np.clip(oof[: len(x)], 1e-4, 1 - 1e-4)
    prior = len(x_test) / len(combined)
    prior_odds = prior / max(1.0 - prior, 1e-6)
    importance = (train_probs / (1.0 - train_probs)) / max(prior_odds, 1e-6)
    weights = np.power(importance, float(params["power"]))
    weights = np.clip(weights, float(params["min_weight"]), float(params["max_weight"]))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def suggest_params(model_family: ModelFamily, trial, thread_count: int) -> dict:
    family = normalize_model_family(model_family)
    if family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 2000, 6000, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "bootstrap_type": "Bayesian",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "allow_writing_files": False,
            "verbose": False,
            "thread_count": thread_count,
        }

    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 4000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_jobs": thread_count,
        }

    if family == "mlp":
        hidden_layer_sizes = trial.suggest_categorical(
            "hidden_layer_sizes", 
            ["128_64", "256_128", "128_64_32", "256_128_64", "64_32"]
        )
        return {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 500, step=50),
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }

    # XGBoost default
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 4000, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": thread_count,
    }


def materialize_params(model_family: ModelFamily, best_params: dict, thread_count: int) -> dict:
    family = normalize_model_family(model_family)
    if family == "catboost":
        params = {
            "iterations": int(best_params["iterations"]),
            "learning_rate": float(best_params["learning_rate"]),
            "depth": int(best_params["depth"]),
            "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
            "random_strength": float(best_params["random_strength"]),
            "bagging_temperature": float(best_params["bagging_temperature"]),
            "bootstrap_type": "Bayesian",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "allow_writing_files": False,
            "verbose": False,
            "thread_count": thread_count,
        }
        if "min_data_in_leaf" in best_params:
            params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"])
        return params

    if family == "lightgbm":
        return {
            "n_estimators": int(best_params["n_estimators"]),
            "learning_rate": float(best_params["learning_rate"]),
            "num_leaves": int(best_params["num_leaves"]),
            "max_depth": int(best_params["max_depth"]),
            "min_child_samples": int(best_params["min_child_samples"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "reg_alpha": float(best_params["reg_alpha"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "min_split_gain": float(best_params["min_split_gain"]),
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_jobs": thread_count,
        }

    if family == "mlp":
        hidden_str = best_params["hidden_layer_sizes"]
        hidden_tuple = tuple(int(x) for x in hidden_str.split("_"))
        return {
            "hidden_layer_sizes": hidden_tuple,
            "activation": best_params["activation"],
            "alpha": float(best_params["alpha"]),
            "learning_rate_init": float(best_params["learning_rate_init"]),
            "max_iter": int(best_params["max_iter"]),
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }

    # XGBoost default
    return {
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "max_depth": int(best_params["max_depth"]),
        "min_child_weight": float(best_params["min_child_weight"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "gamma": float(best_params["gamma"]),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": thread_count,
    }


def default_ablation_params(model_family: ModelFamily, thread_count: int) -> dict:
    family = normalize_model_family(model_family)
    if family == "catboost":
        return {
            "iterations": 3500,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 1.5,
            "bagging_temperature": 1.0,
            "bootstrap_type": "Bayesian",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "allow_writing_files": False,
            "verbose": False,
            "thread_count": thread_count,
        }

    if family == "lightgbm":
        return {
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 7,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_split_gain": 0.0,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_jobs": thread_count,
        }

    if family == "mlp":
        return {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }

    # XGBoost default
    return {
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 4.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "gamma": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": thread_count,
    }


def _require_backend(model_family: ModelFamily) -> None:
    family = normalize_model_family(model_family)
    if family == "catboost" and CatBoostClassifier is None:
        raise ModuleNotFoundError("catboost not installed. Install with: pip install catboost")
    if family == "lightgbm" and LGBMClassifier is None:
        raise ModuleNotFoundError("lightgbm not installed. Install with: pip install lightgbm")
    if family == "xgboost" and XGBClassifier is None:
        raise ModuleNotFoundError("xgboost not installed. Install with: pip install xgboost")


def run_model_cv(
    model_family: ModelFamily,
    x: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    seeds: list[int],
    n_splits: int,
    model_params: dict,
    *,
    x_test: pd.DataFrame | None = None,
    early_stop: int = 250,
    use_class_weight: bool = True,
    predict_test: bool = True,
    sample_weight: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    family = normalize_model_family(model_family)
    _require_backend(family)

    oof_preds_all = []
    test_preds_all = []
    should_predict_test = predict_test and x_test is not None

    if family == "catboost":
        cat_idx = [x.columns.get_loc(column) for column in categorical_columns]

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            oof_preds = np.zeros(len(x), dtype=float)
            test_preds = np.zeros(len(x_test), dtype=float) if should_predict_test else None

            for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
                x_train = x.iloc[train_idx]
                x_valid = x.iloc[valid_idx]
                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]
                train_sample_weight = sample_weight[train_idx] if sample_weight is not None else None

                params = model_params.copy()
                if use_class_weight:
                    params["auto_class_weights"] = "Balanced"

                model = CatBoostClassifier(**params, random_seed=seed + fold)
                model.fit(
                    x_train,
                    y_train,
                    cat_features=cat_idx,
                    eval_set=(x_valid, y_valid),
                    sample_weight=train_sample_weight,
                    use_best_model=True,
                    early_stopping_rounds=early_stop,
                )

                oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
                if test_preds is not None:
                    test_preds += model.predict_proba(x_test)[:, 1] / n_splits

            oof_preds_all.append(oof_preds)
            if test_preds is not None:
                test_preds_all.append(test_preds)

        results = {"oof": np.mean(np.vstack(oof_preds_all), axis=0)}
        if test_preds_all:
            results["test"] = np.mean(np.vstack(test_preds_all), axis=0)
        return results

    # MLP path - needs scaled features
    if family == "mlp":
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            oof_preds = np.zeros(len(x), dtype=float)
            test_preds = np.zeros(len(x_test), dtype=float) if should_predict_test else None

            for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
                x_train = x.iloc[train_idx]
                x_valid = x.iloc[valid_idx]
                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]

                # One-hot encode and scale
                x_train_encoded, x_valid_encoded, x_test_encoded = _prepare_one_hot_matrices(
                    x_train,
                    x_valid,
                    categorical_columns,
                    x_test=x_test if should_predict_test else None,
                )
                
                # Scale features for MLP
                scaler = StandardScaler(with_mean=False)
                x_train_scaled = scaler.fit_transform(x_train_encoded)
                x_valid_scaled = scaler.transform(x_valid_encoded)
                x_test_scaled = scaler.transform(x_test_encoded) if x_test_encoded is not None else None

                params = model_params.copy()
                model = MLPClassifier(**params, random_state=seed + fold)
                model.fit(x_train_scaled, y_train)

                oof_preds[valid_idx] = model.predict_proba(x_valid_scaled)[:, 1]
                if test_preds is not None and x_test_scaled is not None:
                    test_preds += model.predict_proba(x_test_scaled)[:, 1] / n_splits

            oof_preds_all.append(oof_preds)
            if test_preds is not None:
                test_preds_all.append(test_preds)

        results = {"oof": np.mean(np.vstack(oof_preds_all), axis=0)}
        if test_preds_all:
            results["test"] = np.mean(np.vstack(test_preds_all), axis=0)
        return results

    # LightGBM / XGBoost path
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(x), dtype=float)
        test_preds = np.zeros(len(x_test), dtype=float) if should_predict_test else None

        for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
            x_train = x.iloc[train_idx]
            x_valid = x.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            train_sample_weight = sample_weight[train_idx] if sample_weight is not None else None

            x_train_encoded, x_valid_encoded, x_test_encoded = _prepare_one_hot_matrices(
                x_train,
                x_valid,
                categorical_columns,
                x_test=x_test if should_predict_test else None,
            )

            params = model_params.copy()
            if use_class_weight:
                params["scale_pos_weight"] = _scale_pos_weight(y_train)

            if family == "lightgbm":
                model = LGBMClassifier(**params, random_state=seed + fold)
                model.fit(
                    x_train_encoded,
                    y_train,
                    sample_weight=train_sample_weight,
                    eval_set=[(x_valid_encoded, y_valid)],
                    callbacks=[
                        lgb_early_stopping(early_stop, verbose=False),
                        lgb_log_evaluation(0),
                    ],
                )
            else:
                model = XGBClassifier(**params, random_state=seed + fold)
                fit_kwargs = {
                    "eval_set": [(x_valid_encoded, y_valid)],
                    "sample_weight": train_sample_weight,
                    "verbose": False,
                }
                try:
                    model.fit(
                        x_train_encoded,
                        y_train,
                        early_stopping_rounds=early_stop,
                        **fit_kwargs,
                    )
                except TypeError:
                    model.set_params(early_stopping_rounds=early_stop)
                    model.fit(x_train_encoded, y_train, **fit_kwargs)

            oof_preds[valid_idx] = model.predict_proba(x_valid_encoded)[:, 1]
            if test_preds is not None and x_test_encoded is not None:
                test_preds += model.predict_proba(x_test_encoded)[:, 1] / n_splits

        oof_preds_all.append(oof_preds)
        if test_preds is not None:
            test_preds_all.append(test_preds)

    results = {"oof": np.mean(np.vstack(oof_preds_all), axis=0)}
    if test_preds_all:
        results["test"] = np.mean(np.vstack(test_preds_all), axis=0)
    return results
