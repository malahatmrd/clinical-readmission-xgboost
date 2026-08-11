from __future__ import annotations

import pandas as pd

from clinical_readmission.features.feature_schema import (
    BASELINE_DROP_FEATURES,
    DIAGNOSIS_FEATURES,
    EXCLUDED_COLUMNS,
    LAB_CATEGORICAL_FEATURES,
    MODEL_INPUT_FEATURES,
    NUMERIC_FEATURES,
    SPECIALTY_FEATURES,
    STANDARD_CATEGORICAL_FEATURES,
)
from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)


def build_synthetic_frame() -> pd.DataFrame:
    data: dict[str, list] = {}

    for column in NUMERIC_FEATURES:
        data[column] = [
            1.0,
            2.0,
            float("nan"),
            4.0,
        ]

    data["diag_1"] = [
        "250.02",
        "414",
        None,
        "486",
    ]

    data["diag_2"] = [
        "599",
        "786",
        "715",
        "V45",
    ]

    data["diag_3"] = [
        "820",
        "276",
        None,
        "174",
    ]

    for column in STANDARD_CATEGORICAL_FEATURES:
        data[column] = [
            "A",
            "B",
            None,
            "A",
        ]

    data["medical_specialty"] = [
        "InternalMedicine",
        "Cardiology",
        None,
        "RareSpecialty",
    ]

    data["max_glu_serum"] = [
        None,
        "Norm",
        ">200",
        ">300",
    ]

    data["A1Cresult"] = [
        None,
        "Norm",
        ">7",
        ">8",
    ]

    frame = pd.DataFrame(data)

    assert set(frame.columns) == set(
        MODEL_INPUT_FEATURES
    )

    return frame


def test_preprocessor_fits_and_transforms() -> None:
    data = build_synthetic_frame()

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(
        data
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] == len(
        feature_names
    )

    assert any(
        name.endswith("diag_1_Diabetes")
        for name in feature_names
    )

    assert any(
        name.endswith(
            "max_glu_serum_NotMeasured"
        )
        for name in feature_names
    )


def test_preprocessor_handles_unseen_categories() -> None:
    train = build_synthetic_frame()

    preprocessor = build_preprocessor()

    preprocessor.fit(train)

    holdout = train.iloc[[0]].copy()

    holdout.loc[:, "race"] = "NeverSeenRace"
    holdout.loc[
        :,
        "medical_specialty",
    ] = "NeverSeenSpecialty"
    holdout.loc[
        :,
        "max_glu_serum",
    ] = "UnexpectedLabState"

    transformed = preprocessor.transform(
        holdout
    )

    assert transformed.shape[0] == 1

    assert transformed.shape[1] == len(
        preprocessor.get_feature_names_out()
    )


def test_excluded_columns_are_not_transformed() -> None:
    data = build_synthetic_frame()

    data["encounter_id"] = range(
        1,
        len(data) + 1,
    )

    data["patient_nbr"] = range(
        101,
        101 + len(data),
    )

    data["source_row"] = range(
        len(data)
    )

    data["readmitted"] = [
        "NO",
        "<30",
        ">30",
        "NO",
    ]

    data["readmitted_30d"] = [
        0,
        1,
        0,
        0,
    ]

    for column in BASELINE_DROP_FEATURES:
        data[column] = "Excluded"

    preprocessor = build_preprocessor()

    preprocessor.fit(data)

    selected_columns: set[str] = set()

    for (
        _name,
        transformer,
        columns,
    ) in preprocessor.transformers_:
        if transformer == "drop":
            continue

        if isinstance(
            columns,
            (list, tuple),
        ):
            selected_columns.update(
                columns
            )

    assert selected_columns == set(
        MODEL_INPUT_FEATURES
    )

    assert not (
        selected_columns
        & set(EXCLUDED_COLUMNS)
    )


def test_feature_group_partition() -> None:
    grouped = (
        set(NUMERIC_FEATURES)
        | set(DIAGNOSIS_FEATURES)
        | set(STANDARD_CATEGORICAL_FEATURES)
        | set(SPECIALTY_FEATURES)
        | set(LAB_CATEGORICAL_FEATURES)
    )

    assert grouped == set(
        MODEL_INPUT_FEATURES
    )