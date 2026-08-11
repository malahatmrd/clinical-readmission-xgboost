from __future__ import annotations

import pandas as pd
import pytest

from clinical_readmission.data.cohort_builder import (
    add_binary_target,
    build_all_eligible_cohort,
    build_primary_cohort,
    build_sensitivity_cohort,
    validate_cohort,
)

TERMINAL_IDS = {
    11,
    13,
    14,
    19,
    20,
    21,
}


def make_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [
                10,
                11,
                20,
                21,
                30,
                31,
                40,
            ],
            "patient_nbr": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
            ],
            "discharge_disposition_id": [
                1,
                1,
                11,
                1,
                13,
                3,
                3,
            ],
            "readmitted": [
                "NO",
                "<30",
                "NO",
                "<30",
                "<30",
                "NO",
                ">30",
            ],
        }
    )


def test_primary_cohort_uses_first_observed_encounter() -> None:
    data = make_test_data()

    cohort = build_primary_cohort(
        data,
        TERMINAL_IDS,
    )

    assert len(cohort) == 2

    assert set(
        cohort["patient_nbr"]
    ) == {
        1,
        4,
    }

    assert not cohort[
        "patient_nbr"
    ].duplicated().any()

    assert not cohort[
        "discharge_disposition_id"
    ].isin(
        TERMINAL_IDS
    ).any()


def test_sensitivity_cohort_recovers_later_eligible_patients() -> None:
    data = make_test_data()

    cohort = build_sensitivity_cohort(
        data,
        TERMINAL_IDS,
    )

    assert len(cohort) == 4

    assert set(
        cohort["patient_nbr"]
    ) == {
        1,
        2,
        3,
        4,
    }

    assert not cohort[
        "patient_nbr"
    ].duplicated().any()


def test_all_eligible_cohort_retains_repeated_patients() -> None:
    data = make_test_data()

    cohort = build_all_eligible_cohort(
        data,
        TERMINAL_IDS,
    )

    assert len(cohort) == 5

    patient_counts = (
        cohort.groupby(
            "patient_nbr"
        )
        .size()
    )

    assert patient_counts.loc[1] == 2

    assert not cohort[
        "discharge_disposition_id"
    ].isin(
        TERMINAL_IDS
    ).any()


def test_binary_target_mapping_is_exact() -> None:
    data = pd.DataFrame(
        {
            "readmitted": [
                "<30",
                ">30",
                "NO",
            ]
        }
    )

    result = add_binary_target(
        data
    )

    assert result[
        "readmitted_30d"
    ].tolist() == [
        1,
        0,
        0,
    ]


def test_validate_cohort_rejects_repeated_patient_when_unique_required() -> None:
    data = pd.DataFrame(
        {
            "encounter_id": [
                1,
                2,
            ],
            "patient_nbr": [
                10,
                10,
            ],
            "discharge_disposition_id": [
                1,
                1,
            ],
            "readmitted": [
                "NO",
                "<30",
            ],
        }
    )

    data = add_binary_target(
        data
    )

    with pytest.raises(
        ValueError,
        match="repeated patient",
    ):
        validate_cohort(
            "test",
            data,
            TERMINAL_IDS,
            require_unique_patient=True,
        )


def test_validate_cohort_rejects_terminal_encounter() -> None:
    data = pd.DataFrame(
        {
            "encounter_id": [
                1,
            ],
            "patient_nbr": [
                10,
            ],
            "discharge_disposition_id": [
                11,
            ],
            "readmitted": [
                "NO",
            ],
        }
    )

    data = add_binary_target(
        data
    )

    with pytest.raises(
        ValueError,
        match="terminal/hospice",
    ):
        validate_cohort(
            "test",
            data,
            TERMINAL_IDS,
            require_unique_patient=True,
        )