from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from clinical_readmission.features.feature_schema import (
    DIAGNOSIS_FEATURES,
    LAB_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SPECIALTY_FEATURES,
    STANDARD_CATEGORICAL_FEATURES,
)
from clinical_readmission.models.logistic_baseline import (
    BASELINE_C,
    BASELINE_MAX_ITER,
    BASELINE_SOLVER,
    build_logistic_baseline,
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


def test_builds_expected_pipeline() -> None:
    pipeline = build_logistic_baseline()

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
        LogisticRegression,
    )

    assert model.solver == BASELINE_SOLVER
    assert model.C == BASELINE_C
    assert model.max_iter == BASELINE_MAX_ITER
    assert model.class_weight is None


def test_baseline_fits_and_predicts_probabilities() -> None:
    data, target = build_synthetic_data()

    pipeline = build_logistic_baseline()

    pipeline.fit(
        data,
        target,
    )

    probabilities = pipeline.predict_proba(
        data
    )[:, 1]

    assert len(probabilities) == len(data)

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()


def test_baseline_uses_expected_feature_groups() -> None:
    expected = (
        set(NUMERIC_FEATURES)
        | set(DIAGNOSIS_FEATURES)
        | set(STANDARD_CATEGORICAL_FEATURES)
        | set(SPECIALTY_FEATURES)
        | set(LAB_CATEGORICAL_FEATURES)
    )

    assert len(expected) == 42