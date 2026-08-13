from __future__ import annotations

import numpy as np
import pytest

from clinical_readmission.evaluation.shap_utils import (
    aggregate_local_shap_to_source,
    build_transformed_feature_metadata,
    calculate_source_shap_importance,
    calculate_transformed_shap_importance,
    infer_source_feature,
    select_local_explanation_cases,
    validate_shap_matrix,
)


def test_infer_numeric_source_feature() -> None:
    assert (
        infer_source_feature(
            "numeric__time_in_hospital"
        )
        == "time_in_hospital"
    )


def test_infer_onehot_source_feature() -> None:
    assert (
        infer_source_feature(
            "categorical__admission_type_id_1"
        )
        == "admission_type_id"
    )


def test_build_feature_metadata() -> None:
    names = [
        "numeric__time_in_hospital",
        "categorical__race_Caucasian",
        "lab__A1Cresult_NotMeasured",
    ]

    table = (
        build_transformed_feature_metadata(
            names
        )
    )

    assert table[
        "source_feature"
    ].tolist() == [
        "time_in_hospital",
        "race",
        "A1Cresult",
    ]

    assert table[
        "transformed_index"
    ].tolist() == [
        0,
        1,
        2,
    ]


def test_transformed_importance_is_ranked() -> None:
    shap_values = np.array(
        [
            [1.0, 0.1],
            [2.0, -0.1],
        ]
    )

    table = (
        calculate_transformed_shap_importance(
            shap_values,
            [
                "numeric__time_in_hospital",
                "numeric__num_procedures",
            ],
        )
    )

    assert table.iloc[
        0
    ][
        "transformed_feature"
    ] == (
        "numeric__time_in_hospital"
    )

    assert table.iloc[
        0
    ][
        "rank"
    ] == 1


def test_source_importance_groups_before_absolute_value() -> None:
    names = [
        "categorical__race_A",
        "categorical__race_B",
        "numeric__time_in_hospital",
    ]

    metadata = (
        build_transformed_feature_metadata(
            names
        )
    )

    shap_values = np.array(
        [
            [1.0, -1.0, 2.0],
            [2.0, -1.0, 0.0],
        ]
    )

    table = (
        calculate_source_shap_importance(
            shap_values,
            metadata,
        )
    )

    race = table.loc[
        table[
            "source_feature"
        ].eq(
            "race"
        )
    ].iloc[
        0
    ]

    assert race[
        "mean_abs_shap"
    ] == pytest.approx(
        0.5
    )


def test_validate_shap_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="feature count",
    ):
        validate_shap_matrix(
            np.zeros(
                (
                    2,
                    3,
                )
            ),
            [
                "a",
                "b",
            ],
        )


def test_validate_shap_matrix_rejects_nonfinite_values() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_shap_matrix(
            np.array(
                [
                    [
                        1.0,
                        np.nan,
                    ]
                ]
            ),
            [
                "a",
                "b",
            ],
        )

def test_select_local_explanation_cases() -> None:
    target = np.array(
        [
            1,
            0,
            1,
            0,
            1,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.90,
            0.80,
            0.10,
            0.02,
            0.49,
            0.51,
        ]
    )

    table = (
        select_local_explanation_cases(
            target,
            probabilities,
            threshold=0.50,
        )
    )

    selected = dict(
        zip(
            table[
                "case_name"
            ],
            table[
                "validation_row"
            ],
            strict=True,
        )
    )

    assert selected[
        "high_confidence_true_positive"
    ] == 0

    assert selected[
        "high_confidence_false_positive"
    ] == 1

    assert selected[
        "near_threshold_false_negative"
    ] == 4

    assert selected[
        "low_risk_true_negative"
    ] == 3

    assert selected[
        "closest_unused_to_threshold"
    ] == 5

    assert len(
        set(
            selected.values()
        )
    ) == 5


def test_local_case_selection_rejects_missing_quadrant() -> None:
    with pytest.raises(
        ValueError,
        match="No eligible observation",
    ):
        select_local_explanation_cases(
            np.array(
                [
                    1,
                    1,
                    0,
                ]
            ),
            np.array(
                [
                    0.9,
                    0.8,
                    0.1,
                ]
            ),
            threshold=0.5,
        )


def test_aggregate_local_shap_to_source() -> None:
    names = [
        "categorical__race_A",
        "categorical__race_B",
        "numeric__time_in_hospital",
    ]

    metadata = (
        build_transformed_feature_metadata(
            names
        )
    )

    table = (
        aggregate_local_shap_to_source(
            np.array(
                [
                    1.0,
                    -0.5,
                    2.0,
                ]
            ),
            metadata,
        )
    )

    time_feature = table.loc[
        table[
            "source_feature"
        ].eq(
            "time_in_hospital"
        )
    ].iloc[
        0
    ]

    race = table.loc[
        table[
            "source_feature"
        ].eq(
            "race"
        )
    ].iloc[
        0
    ]

    assert time_feature[
        "shap_value"
    ] == pytest.approx(
        2.0
    )

    assert race[
        "shap_value"
    ] == pytest.approx(
        0.5
    )

    assert table.iloc[
        0
    ][
        "source_feature"
    ] == (
        "time_in_hospital"
    )


def test_aggregate_local_shap_rejects_wrong_length() -> None:
    metadata = (
        build_transformed_feature_metadata(
            [
                "numeric__time_in_hospital",
                "numeric__number_inpatient",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="length",
    ):
        aggregate_local_shap_to_source(
            np.array(
                [
                    1.0,
                ]
            ),
            metadata,
        )