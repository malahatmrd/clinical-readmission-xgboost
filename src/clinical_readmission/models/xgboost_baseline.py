from __future__ import annotations

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)

BASELINE_N_ESTIMATORS = 100
BASELINE_MAX_DEPTH = 6
BASELINE_LEARNING_RATE = 0.3
BASELINE_MIN_CHILD_WEIGHT = 1.0
BASELINE_SUBSAMPLE = 1.0
BASELINE_COLSAMPLE_BYTREE = 1.0
BASELINE_REG_ALPHA = 0.0
BASELINE_REG_LAMBDA = 1.0
BASELINE_RANDOM_STATE = 42


def build_xgboost_baseline() -> Pipeline:
    """Build the untuned XGBoost classification baseline."""

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=BASELINE_N_ESTIMATORS,
        max_depth=BASELINE_MAX_DEPTH,
        learning_rate=BASELINE_LEARNING_RATE,
        min_child_weight=BASELINE_MIN_CHILD_WEIGHT,
        subsample=BASELINE_SUBSAMPLE,
        colsample_bytree=BASELINE_COLSAMPLE_BYTREE,
        reg_alpha=BASELINE_REG_ALPHA,
        reg_lambda=BASELINE_REG_LAMBDA,
        tree_method="hist",
        eval_metric="logloss",
        random_state=BASELINE_RANDOM_STATE,
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