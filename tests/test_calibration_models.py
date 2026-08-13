from __future__ import annotations

import pytest
from sklearn.calibration import (
    CalibratedClassifierCV,
)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from clinical_readmission.evaluation.calibration_models import (
    CALIBRATION_CV_RANDOM_STATE,
    CALIBRATION_CV_SPLITS,
    build_calibrated_classifier,
    build_calibration_cv,
    build_early_stopped_pipeline,
    build_tuned_pipeline,
)


def tuned_parameters() -> dict:
    return {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "scale_pos_weight": 1.5,
    }


def test_builds_frozen_calibration_cv() -> None:
    cv = build_calibration_cv()

    assert isinstance(
        cv,
        StratifiedKFold,
    )

    assert (
        cv.n_splits
        == CALIBRATION_CV_SPLITS
    )

    assert cv.shuffle is True

    assert (
        cv.random_state
        == CALIBRATION_CV_RANDOM_STATE
    )


def test_builds_early_stopped_pipeline() -> None:
    pipeline = (
        build_early_stopped_pipeline(
            n_estimators=21
        )
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert (
        pipeline.named_steps[
            "model"
        ].n_estimators
        == 21
    )


def test_builds_tuned_pipeline() -> None:
    pipeline = build_tuned_pipeline(
        tuned_parameters(),
        n_estimators=155,
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    model = pipeline.named_steps[
        "model"
    ]

    assert model.n_estimators == 155
    assert model.learning_rate == 0.03
    assert model.max_depth == 5
    assert model.scale_pos_weight == 1.5


@pytest.mark.parametrize(
    "method",
    [
        "sigmoid",
        "isotonic",
    ],
)
def test_builds_supported_calibrator(
    method: str,
) -> None:
    estimator = (
        build_early_stopped_pipeline(
            n_estimators=21
        )
    )

    calibrated = (
        build_calibrated_classifier(
            estimator,
            method,
        )
    )

    assert isinstance(
        calibrated,
        CalibratedClassifierCV,
    )

    assert calibrated.method == method
    assert calibrated.ensemble is False
    assert calibrated.n_jobs == 1


def test_rejects_unknown_method() -> None:
    estimator = (
        build_early_stopped_pipeline(
            n_estimators=21
        )
    )

    with pytest.raises(
        ValueError,
        match="Unsupported calibration method",
    ):
        build_calibrated_classifier(
            estimator,
            "unknown",
        )