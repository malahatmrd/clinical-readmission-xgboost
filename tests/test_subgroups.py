from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinical_readmission.evaluation.subgroups import (
    build_subgroup_performance_table,
    combine_subgroup_tables,
    determine_reporting_eligibility,
    normalize_subgroup_values,
)


def test_normalize_subgroup_values_preserves_missing() -> None:
    values = [
        "Male",
        None,
        "Female",
    ]

    result = normalize_subgroup_values(
        values
    )

    assert result.tolist() == [
        "Male",
        "Missing",
        "Female",
    ]


def test_reporting_eligibility_passes() -> None:
    eligible, reason = (
        determine_reporting_eligibility(
            group_size=100,
            positive_count=20,
            negative_count=80,
            minimum_group_size=50,
            minimum_positives=10,
            minimum_negatives=10,
        )
    )

    assert eligible is True
    assert reason == ""


def test_reporting_eligibility_records_reasons() -> None:
    eligible, reason = (
        determine_reporting_eligibility(
            group_size=20,
            positive_count=2,
            negative_count=18,
            minimum_group_size=50,
            minimum_positives=10,
            minimum_negatives=10,
        )
    )

    assert eligible is False
    assert "rows<50" in reason
    assert "positives<10" in reason


def test_build_subgroup_performance_table() -> None:
    target = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.8,
            0.2,
            0.7,
            0.3,
            0.9,
            0.4,
            0.6,
        ]
    )

    groups = [
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
    ]

    table = (
        build_subgroup_performance_table(
            target,
            probabilities,
            groups,
            subgroup_name="example",
            threshold=0.5,
            minimum_group_size=2,
            minimum_positives=1,
            minimum_negatives=1,
        )
    )

    assert table[
        "subgroup_value"
    ].tolist() == [
        "A",
        "B",
    ]

    assert (
        table[
            "reporting_eligible"
        ].all()
    )

    assert table[
        "roc_auc"
    ].tolist() == pytest.approx(
        [
            1.0,
            1.0,
        ]
    )

    assert table[
        "sensitivity"
    ].tolist() == pytest.approx(
        [
            1.0,
            1.0,
        ]
    )

    assert table[
        "specificity"
    ].tolist() == pytest.approx(
        [
            1.0,
            1.0,
        ]
    )


def test_ineligible_subgroup_has_nan_metrics() -> None:
    target = np.array(
        [
            0,
            1,
            0,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.8,
            0.2,
            0.3,
        ]
    )

    groups = [
        "A",
        "A",
        "B",
        "B",
    ]

    table = (
        build_subgroup_performance_table(
            target,
            probabilities,
            groups,
            subgroup_name="example",
            threshold=0.5,
            minimum_group_size=2,
            minimum_positives=1,
            minimum_negatives=1,
        )
    )

    group_b = table.loc[
        table[
            "subgroup_value"
        ].eq(
            "B"
        )
    ].iloc[
        0
    ]

    assert not group_b[
        "reporting_eligible"
    ]

    assert np.isnan(
        group_b[
            "roc_auc"
        ]
    )

    assert (
        "positives<1"
        in group_b[
            "exclusion_reason"
        ]
    )


def test_subgroup_length_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        build_subgroup_performance_table(
            np.array(
                [
                    0,
                    1,
                ]
            ),
            np.array(
                [
                    0.2,
                    0.8,
                ]
            ),
            [
                "A",
            ],
            subgroup_name="example",
            threshold=0.5,
            minimum_group_size=1,
            minimum_positives=1,
            minimum_negatives=1,
        )


def test_combine_subgroup_tables() -> None:
    table_a = pd.DataFrame(
        {
            "a": [
                1,
            ],
            "b": [
                2,
            ],
        }
    )

    table_b = pd.DataFrame(
        {
            "a": [
                3,
            ],
            "b": [
                4,
            ],
        }
    )

    result = combine_subgroup_tables(
        [
            table_a,
            table_b,
        ]
    )

    assert len(
        result
    ) == 2

    assert result[
        "a"
    ].tolist() == [
        1,
        3,
    ]