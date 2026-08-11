from __future__ import annotations

import pandas as pd
import pytest

from clinical_readmission.data.split_builder import (
    build_split_assignments,
    build_split_summary,
    validate_assignments,
    validate_split_sizes,
)


def make_split_data() -> pd.DataFrame:
    rows = 100

    return pd.DataFrame(
        {
            "encounter_id": range(
                1,
                rows + 1,
            ),
            "patient_nbr": range(
                1001,
                1001 + rows,
            ),
            "readmitted_30d": (
                [1] * 20
                + [0] * 80
            ),
        }
    )


def test_split_sizes_must_sum_to_one() -> None:
    with pytest.raises(
        ValueError,
        match="must sum to 1.0",
    ):
        validate_split_sizes(
            0.70,
            0.20,
            0.20,
        )


def test_primary_split_has_expected_sizes() -> None:
    data = make_split_data()

    assignments = build_split_assignments(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    counts = (
        assignments["split"]
        .value_counts()
        .to_dict()
    )

    assert counts["train"] == 70
    assert counts["validation"] == 15
    assert counts["test"] == 15


def test_split_is_deterministic() -> None:
    data = make_split_data()

    first = build_split_assignments(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    second = build_split_assignments(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_no_patient_overlap_between_splits() -> None:
    data = make_split_data()

    assignments = build_split_assignments(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    validate_assignments(
        data,
        assignments,
    )

    train = set(
        assignments.loc[
            assignments["split"]
            == "train",
            "patient_nbr",
        ]
    )

    validation = set(
        assignments.loc[
            assignments["split"]
            == "validation",
            "patient_nbr",
        ]
    )

    test = set(
        assignments.loc[
            assignments["split"]
            == "test",
            "patient_nbr",
        ]
    )

    assert train.isdisjoint(
        validation
    )

    assert train.isdisjoint(
        test
    )

    assert validation.isdisjoint(
        test
    )


def test_stratification_preserves_positive_rate() -> None:
    data = make_split_data()

    assignments = build_split_assignments(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    overall_rate = (
        data["readmitted_30d"]
        .mean()
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        subset = assignments.loc[
            assignments["split"]
            == split_name
        ]

        split_rate = (
            subset["readmitted_30d"]
            .mean()
        )

        assert split_rate == pytest.approx(
            overall_rate,
            abs=0.04,
        )


def test_split_summary_covers_all_rows() -> None:
    data = make_split_data()

    assignments = build_split_assignments(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    summary = build_split_summary(
        assignments
    )

    assert (
        summary["encounters"].sum()
        == len(data)
    )

    assert set(
        summary["split"]
    ) == {
        "train",
        "validation",
        "test",
    }