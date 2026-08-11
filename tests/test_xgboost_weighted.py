from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clinical_readmission.features.feature_schema import (
    LAB_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    STANDARD_CATEGORICAL_FEATURES,
)
from clinical_readmission.models.xgboost_weighted import (
    WEIGHTED_COLSAMPLE_BYTREE,
    WEIGHTED_LEARNING_RATE,
    WEIGHTED_MAX_DEPTH,
    WEIGHTED_N_ESTIMATORS,
    WEIGHTED_RANDOM_STATE,
    WEIGHTED_REG_ALPHA,
    WEIGHTED_REG_LAMBDA,
    WEIGHTED_SUBSAMPLE,
    build_weighted_xgboost,
)


def build_synthetic_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    rows = 20

    data: dict[str, list] = {}

    for index, column in enumerate(
        NUMERIC_FEATURES
    ):
        data[column] = [
            float((row + index) % 7)
            for row in range(rows)
        ]

    data["diag_1"] = [
        "250.02",
        "414",
        "486",
        "599",
        "715",
    ] * 4

    data["diag_2"] = [
        "786",
        "276",
        "820",
        "250",
        "174",
    ] * 4

    data["diag_3"] = [
        "599",
        "428",
        "V45",
        "250.8",
        "574",
    ] * 4

    for column in STANDARD_CATEGORICAL_FEATURES:
        data[column] = [
            "A" if row % 2 == 0 else "B"
            for row in range(rows)
        ]

    data["medical_specialty"] = [
        "InternalMedicine",
        "Cardiology",
    ] * 10

    for column in LAB_CATEGORICAL_FEATURES:
        data[column] = [
            None,
            "Norm",
            ">200",
            None,
            "Norm",
        ] * 4

    frame = pd.DataFrame(data)

    target = pd.Series(
        [0, 1] * 10,
        name="readmitted_30d",
    )

    return frame, target


def test_builds_expected_weighted_pipeline() -> None:
    scale_pos_weight = 4.5

    pipeline = build_weighted_xgboost(
        scale_pos_weight
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    model = pipeline.named_steps["model"]

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert (
        model.scale_pos_weight
        == scale_pos_weight
    )

    assert (
        model.n_estimators
        == WEIGHTED_N_ESTIMATORS
    )

    assert (
        model.max_depth
        == WEIGHTED_MAX_DEPTH
    )

    assert (
        model.learning_rate
        == WEIGHTED_LEARNING_RATE
    )

    assert (
        model.subsample
        == WEIGHTED_SUBSAMPLE
    )

    assert (
        model.colsample_bytree
        == WEIGHTED_COLSAMPLE_BYTREE
    )

    assert (
        model.reg_alpha
        == WEIGHTED_REG_ALPHA
    )

    assert (
        model.reg_lambda
        == WEIGHTED_REG_LAMBDA
    )

    assert (
        model.random_state
        == WEIGHTED_RANDOM_STATE
    )


def test_weighted_xgboost_fits_and_predicts() -> None:
    data, target = build_synthetic_data()

    pipeline = build_weighted_xgboost(
        scale_pos_weight=4.0
    )

    pipeline.fit(
        data,
        target,
    )

    probabilities = (
        pipeline.predict_proba(
            data
        )[:, 1]
    )

    assert len(probabilities) == len(data)

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()


def test_rejects_invalid_scale_pos_weight() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_weighted_xgboost(
            scale_pos_weight=0.0
        )