from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

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


ModelFamily = Literal["catboost", "lightgbm", "xgboost"]


def normalize_model_family(model_family: str) -> ModelFamily:
    normalized = model_family.strip().lower()
    allowed = {"catboost", "lightgbm", "xgboost"}
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


def suggest_params(model_family: ModelFamily, trial, thread_count: int) -> dict:
    family = normalize_model_family(model_family)
    if family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 2500, 5000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),
            "depth": trial.suggest_int("depth", 5, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 2.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
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
        return {
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

                params = model_params.copy()
                if use_class_weight:
                    params["auto_class_weights"] = "Balanced"

                model = CatBoostClassifier(**params, random_seed=seed + fold)
                model.fit(
                    x_train,
                    y_train,
                    cat_features=cat_idx,
                    eval_set=(x_valid, y_valid),
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

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(x), dtype=float)
        test_preds = np.zeros(len(x_test), dtype=float) if should_predict_test else None

        for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
            x_train = x.iloc[train_idx]
            x_valid = x.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

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
