from __future__ import annotations

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)

CV_N_SPLITS = 3
CV_RANDOM_STATE = 45

SEARCH_N_ITER = 24
SEARCH_RANDOM_STATE = 46
SEARCH_N_JOBS = 2

XGBOOST_RANDOM_STATE = 42
XGBOOST_N_JOBS = 1

PRIMARY_SCORER = "average_precision"

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
}

PARAMETER_DISTRIBUTIONS = {
    "model__n_estimators": [
        100,
        200,
        400,
    ],
    "model__learning_rate": [
        0.03,
        0.05,
        0.10,
        0.20,
    ],
    "model__max_depth": [
        2,
        3,
        4,
        5,
        6,
    ],
    "model__min_child_weight": [
        1.0,
        3.0,
        5.0,
        10.0,
    ],
    "model__subsample": [
        0.70,
        0.85,
        1.00,
    ],
    "model__colsample_bytree": [
        0.70,
        0.85,
        1.00,
    ],
    "model__reg_alpha": [
        0.0,
        0.1,
        0.5,
        1.0,
    ],
    "model__reg_lambda": [
        1.0,
        2.0,
        5.0,
        10.0,
    ],
    "model__scale_pos_weight": [
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
    ],
}


def build_tuning_pipeline() -> Pipeline:
    """Build leakage-safe XGBoost tuning pipeline."""

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=XGBOOST_RANDOM_STATE,
        n_jobs=XGBOOST_N_JOBS,
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


def build_xgboost_random_search() -> RandomizedSearchCV:
    """Build the frozen Train-only randomized-search protocol."""

    cross_validation = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    return RandomizedSearchCV(
        estimator=build_tuning_pipeline(),
        param_distributions=PARAMETER_DISTRIBUTIONS,
        n_iter=SEARCH_N_ITER,
        scoring=SCORING,
        refit=PRIMARY_SCORER,
        cv=cross_validation,
        random_state=SEARCH_RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        pre_dispatch=SEARCH_N_JOBS,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )