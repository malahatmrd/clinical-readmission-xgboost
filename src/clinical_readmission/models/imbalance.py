from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_scale_pos_weight(
    target: pd.Series | np.ndarray,
) -> float:
    """Calculate negative-to-positive class ratio."""

    values = np.asarray(target)

    unique_values = set(
        np.unique(values).tolist()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "Target must contain binary values 0 and 1."
        )

    positive_count = int(
        np.sum(values == 1)
    )

    negative_count = int(
        np.sum(values == 0)
    )

    if positive_count == 0:
        raise ValueError(
            "Cannot calculate scale_pos_weight "
            "without positive samples."
        )

    if negative_count == 0:
        raise ValueError(
            "Cannot calculate scale_pos_weight "
            "without negative samples."
        )

    return (
        negative_count
        / positive_count
    )