from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from clinical_readmission.models.xgboost_baseline import (
    BASELINE_COLSAMPLE_BYTREE,
    BASELINE_LEARNING_RATE,
    BASELINE_MAX_DEPTH,
    BASELINE_MIN_CHILD_WEIGHT,
    BASELINE_RANDOM_STATE,
    BASELINE_REG_ALPHA,
    BASELINE_REG_LAMBDA,
    BASELINE_SUBSAMPLE,
)

INTERNAL_VALIDATION_SIZE = 0.20
INTERNAL_SPLIT_RANDOM_STATE = 44

EARLY_STOPPING_N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50


def build_internal_early_stopping_split(
    target: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible stratified split within Train only."""

    values = np.asarray(target)

    indices = np.arange(
        len(values)
    )

    fit_indices, stop_indices = train_test_split(
        indices,
        test_size=INTERNAL_VALIDATION_SIZE,
        random_state=INTERNAL_SPLIT_RANDOM_STATE,
        stratify=values,
    )

    return (
        np.asarray(fit_indices),
        np.asarray(stop_indices),
    )


def build_early_stopping_xgboost() -> XGBClassifier:
    """Build the unweighted XGBoost development model."""

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=EARLY_STOPPING_N_ESTIMATORS,
        max_depth=BASELINE_MAX_DEPTH,
        learning_rate=BASELINE_LEARNING_RATE,
        min_child_weight=BASELINE_MIN_CHILD_WEIGHT,
        subsample=BASELINE_SUBSAMPLE,
        colsample_bytree=BASELINE_COLSAMPLE_BYTREE,
        reg_alpha=BASELINE_REG_ALPHA,
        reg_lambda=BASELINE_REG_LAMBDA,
        scale_pos_weight=1.0,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=BASELINE_RANDOM_STATE,
        n_jobs=-1,
    )


def build_refit_xgboost(
    n_estimators: int,
) -> XGBClassifier:
    """Build the full-Train model using a selected tree count."""

    if n_estimators <= 0:
        raise ValueError(
            "n_estimators must be greater than zero."
        )

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=BASELINE_MAX_DEPTH,
        learning_rate=BASELINE_LEARNING_RATE,
        min_child_weight=BASELINE_MIN_CHILD_WEIGHT,
        subsample=BASELINE_SUBSAMPLE,
        colsample_bytree=BASELINE_COLSAMPLE_BYTREE,
        reg_alpha=BASELINE_REG_ALPHA,
        reg_lambda=BASELINE_REG_LAMBDA,
        scale_pos_weight=1.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=BASELINE_RANDOM_STATE,
        n_jobs=-1,
    )