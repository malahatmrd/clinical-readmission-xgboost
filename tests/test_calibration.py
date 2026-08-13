from __future__ import annotations

import numpy as np
import pytest

from clinical_readmission.evaluation.calibration import (
    build_calibration_curve_table,
    calculate_calibration_intercept_slope,
    calculate_quantile_ece,
    validate_binary_probabilities,
)


def test_calibration_intercept_slope_are_finite() -> None:
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
            0.10,
            0.20,
            0.30,
            0.40,
            0.60,
            0.70,
            0.80,
            0.90,
        ]
    )

    intercept, slope = (
        calculate_calibration_intercept_slope(
            target,
            probabilities,
        )
    )

    assert np.isfinite(
        intercept
    )

    assert np.isfinite(
        slope
    )


def test_quantile_ece_matches_known_example() -> None:
    target = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.10,
            0.20,
            0.80,
            0.90,
        ]
    )

    result = calculate_quantile_ece(
        target,
        probabilities,
        n_bins=2,
    )

    assert result == pytest.approx(
        0.15
    )


def test_calibration_curve_table_has_expected_columns() -> None:
    target = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.60,
            0.75,
            0.90,
        ]
    )

    table = build_calibration_curve_table(
        target,
        probabilities,
        n_bins=3,
    )

    assert list(
        table.columns
    ) == [
        "bin",
        "mean_predicted_probability",
        "observed_event_rate",
        "calibration_error",
    ]

    assert len(table) <= 3


def test_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_binary_probabilities(
            [0, 1],
            [0.20, 1.10],
        )


def test_rejects_non_binary_target() -> None:
    with pytest.raises(
        ValueError,
        match="binary",
    ):
        validate_binary_probabilities(
            [0, 1, 2],
            [0.10, 0.50, 0.90],
        )


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        validate_binary_probabilities(
            [0, 1, 0],
            [0.10, 0.80],
        )