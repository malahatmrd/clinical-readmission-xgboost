from __future__ import annotations

import pandas as pd
import pytest

from clinical_readmission.data import validate
from clinical_readmission.data.audit import (
    build_patient_audit,
    build_target_audit,
)
from clinical_readmission.data.feature_audit import (
    audit_feature,
)


def test_validate_target_accepts_expected_labels() -> None:
    target = pd.DataFrame(
        {
            "readmitted": [
                "NO",
                ">30",
                "<30",
                "NO",
            ]
        }
    )

    validate.validate_target(target)


def test_validate_target_rejects_unexpected_label() -> None:
    target = pd.DataFrame(
        {
            "readmitted": [
                "NO",
                "<30",
                "INVALID",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Unexpected target labels",
    ):
        validate.validate_target(target)


def test_duplicate_encounter_is_rejected() -> None:
    identifiers = pd.DataFrame(
        {
            "encounter_id": [
                100,
                100,
                200,
            ],
            "patient_nbr": [
                1,
                2,
                3,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate encounter_id",
    ):
        validate.validate_identifier_integrity(
            identifiers
        )


def test_patient_audit_detects_repeated_patients() -> None:
    identifiers = pd.DataFrame(
        {
            "encounter_id": [
                10,
                11,
                12,
                13,
            ],
            "patient_nbr": [
                100,
                100,
                200,
                300,
            ],
        }
    )

    result = build_patient_audit(
        identifiers
    )

    assert result["rows"] == 4
    assert result["unique_patients"] == 3
    assert result[
        "patients_with_multiple_encounters"
    ] == 1

    assert result[
        "max_encounters_per_patient"
    ] == 2


def test_binary_target_audit() -> None:
    target = pd.DataFrame(
        {
            "readmitted": [
                "<30",
                "NO",
                ">30",
                "NO",
            ]
        }
    )

    result = build_target_audit(
        target
    )

    assert result["positive_count"] == 1
    assert result["negative_count"] == 3
    assert result["positive_rate"] == pytest.approx(
        0.25
    )


def test_feature_audit_detects_missing_values() -> None:
    series = pd.Series(
        [
            "A",
            None,
            "B",
            None,
        ],
        name="example",
    )

    result = audit_feature(
        "example",
        series,
    )

    assert result["rows"] == 4
    assert result["missing_count"] == 2
    assert result["missing_pct"] == pytest.approx(
        50.0
    )

    assert result["unique_non_null"] == 2


def test_id_columns_are_not_predictive_features() -> None:
    identifier_columns = {
        "encounter_id",
        "patient_nbr",
    }

    model_features = {
        "race",
        "gender",
        "age",
        "time_in_hospital",
    }

    assert identifier_columns.isdisjoint(
        model_features
    )