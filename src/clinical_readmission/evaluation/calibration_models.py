from __future__ import annotations

from sklearn.calibration import (
    CalibratedClassifierCV,
)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)
from clinical_readmission.models.xgboost_early_stopping import (
    build_refit_xgboost,
)
from clinical_readmission.models.xgboost_tuned import (
    build_tuned_refit_xgboost,
)

CALIBRATION_CV_SPLITS = 5
CALIBRATION_CV_RANDOM_STATE = 48

SUPPORTED_CALIBRATION_METHODS = (
    "sigmoid",
    "isotonic",
)


def build_calibration_cv() -> StratifiedKFold:
    """Build the frozen Train-only calibration splitter."""

    return StratifiedKFold(
        n_splits=CALIBRATION_CV_SPLITS,
        shuffle=True,
        random_state=(
            CALIBRATION_CV_RANDOM_STATE
        ),
    )


def build_early_stopped_pipeline(
    n_estimators: int,
) -> Pipeline:
    """Build the early-stopped XGBoost finalist pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                build_refit_xgboost(
                    n_estimators=(
                        n_estimators
                    ),
                ),
            ),
        ]
    )


def build_tuned_pipeline(
    parameters: dict,
    n_estimators: int,
) -> Pipeline:
    """Build the tuned XGBoost finalist pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                build_tuned_refit_xgboost(
                    parameters,
                    n_estimators=(
                        n_estimators
                    ),
                ),
            ),
        ]
    )


def build_calibrated_classifier(
    estimator,
    method: str,
) -> CalibratedClassifierCV:
    """Build Train-only cross-validated post-hoc calibration."""

    if method not in (
        SUPPORTED_CALIBRATION_METHODS
    ):
        raise ValueError(
            "Unsupported calibration method: "
            f"{method}"
        )

    return CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=build_calibration_cv(),
        ensemble=False,
        n_jobs=1,
    )