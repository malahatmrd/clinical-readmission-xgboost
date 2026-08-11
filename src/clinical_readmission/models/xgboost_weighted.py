from __future__ import annotations

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)

WEIGHTED_N_ESTIMATORS = 100
WEIGHTED_MAX_DEPTH = 6
WEIGHTED_LEARNING_RATE = 0.3
WEIGHTED_MIN_CHILD_WEIGHT = 1.0
WEIGHTED_SUBSAMPLE = 1.0
WEIGHTED_COLSAMPLE_BYTREE = 1.0
WEIGHTED_REG_ALPHA = 0.0
WEIGHTED_REG_LAMBDA = 1.0
WEIGHTED_RANDOM_STATE = 42


def build_weighted_xgboost(
    scale_pos_weight: float,
) -> Pipeline:
    """Build an XGBoost model with positive-class weighting."""

    if scale_pos_weight <= 0:
        raise ValueError(
            "scale_pos_weight must be greater than zero."
        )

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=WEIGHTED_N_ESTIMATORS,
        max_depth=WEIGHTED_MAX_DEPTH,
        learning_rate=WEIGHTED_LEARNING_RATE,
        min_child_weight=WEIGHTED_MIN_CHILD_WEIGHT,
        subsample=WEIGHTED_SUBSAMPLE,
        colsample_bytree=WEIGHTED_COLSAMPLE_BYTREE,
        reg_alpha=WEIGHTED_REG_ALPHA,
        reg_lambda=WEIGHTED_REG_LAMBDA,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=WEIGHTED_RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )