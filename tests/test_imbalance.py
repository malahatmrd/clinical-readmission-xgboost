from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinical_readmission.models.imbalance import (
    calculate_scale_pos_weight,
)


def test_calculates_expected_class_ratio() -> None:
    target = pd.Series(
        [
            0,
            0,
            0,
            0,
            1,
        ]
    )

    result = calculate_scale_pos_weight(
        target
    )

    assert result == 4.0


def test_accepts_numpy_array() -> None:
    target = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    result = calculate_scale_pos_weight(
        target
    )

    assert result == 1.0


def test_rejects_non_binary_target() -> None:
    target = pd.Series(
        [
            0,
            1,
            2,
        ]
    )

    with pytest.raises(
        ValueError,
        match="binary",
    ):
        calculate_scale_pos_weight(
            target
        )


def test_rejects_target_without_positive_class() -> None:
    target = pd.Series(
        [
            0,
            0,
            0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        calculate_scale_pos_weight(
            target
        )


def test_rejects_target_without_negative_class() -> None:
    target = pd.Series(
        [
            1,
            1,
            1,
        ]
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        calculate_scale_pos_weight(
            target
        )