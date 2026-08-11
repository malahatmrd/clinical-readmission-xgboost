from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clinical_readmission.features.feature_schema import (
    DIAGNOSIS_FEATURES,
    LAB_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SPECIALTY_FEATURES,
    STANDARD_CATEGORICAL_FEATURES,
)
from clinical_readmission.models.xgboost_baseline import (
    BASELINE_COLSAMPLE_BYTREE,
    BASELINE_LEARNING_RATE,
    BASELINE_MAX_DEPTH,
    BASELINE_N_ESTIMATORS,
    BASELINE_RANDOM_STATE,
    BASELINE_REG_ALPHA,
    BASELINE_REG_LAMBDA,
    BASELINE_SUBSAMPLE,
    build_xgboost_baseline,
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


def test_builds_expected_xgboost_pipeline() -> None:
    pipeline = build_xgboost_baseline()

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert list(
        pipeline.named_steps
    ) == [
        "preprocessor",
        "model",
    ]

    model = pipeline.named_steps["model"]

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.n_estimators == BASELINE_N_ESTIMATORS
    assert model.max_depth == BASELINE_MAX_DEPTH
    assert model.learning_rate == BASELINE_LEARNING_RATE
    assert model.subsample == BASELINE_SUBSAMPLE
    assert (
        model.colsample_bytree
        == BASELINE_COLSAMPLE_BYTREE
    )
    assert model.reg_alpha == BASELINE_REG_ALPHA
    assert model.reg_lambda == BASELINE_REG_LAMBDA
    assert model.random_state == BASELINE_RANDOM_STATE


def test_xgboost_baseline_fits_and_predicts() -> None:
    data, target = build_synthetic_data()

    pipeline = build_xgboost_baseline()

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


def test_xgboost_feature_count_policy() -> None:
    expected = (
        set(NUMERIC_FEATURES)
        | set(DIAGNOSIS_FEATURES)
        | set(STANDARD_CATEGORICAL_FEATURES)
        | set(SPECIALTY_FEATURES)
        | set(LAB_CATEGORICAL_FEATURES)
    )

    assert len(expected) == 42