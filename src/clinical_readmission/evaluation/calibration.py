from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

PROBABILITY_EPSILON = 1e-6
DEFAULT_CALIBRATION_BINS = 10


def validate_binary_probabilities(
    target,
    probabilities,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary targets and probability predictions."""

    target_array = np.asarray(
        target
    )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if target_array.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if probability_array.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if len(target_array) != len(
        probability_array
    ):
        raise ValueError(
            "Target and probabilities must have "
            "the same length."
        )

    unique_targets = set(
        np.unique(
            target_array
        ).tolist()
    )

    if not unique_targets.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Target must contain binary values "
            "0 and 1."
        )

    if not np.isfinite(
        probability_array
    ).all():
        raise ValueError(
            "Probabilities must be finite."
        )

    if (
        (probability_array < 0)
        | (probability_array > 1)
    ).any():
        raise ValueError(
            "Probabilities must be between "
            "0 and 1."
        )

    return (
        target_array.astype(int),
        probability_array,
    )


def calculate_calibration_intercept_slope(
    target,
    probabilities,
) -> tuple[float, float]:
    """Estimate logistic calibration intercept and slope."""

    (
        target_array,
        probability_array,
    ) = validate_binary_probabilities(
        target,
        probabilities,
    )

    clipped = np.clip(
        probability_array,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )

    logits = np.log(
        clipped
        / (1.0 - clipped)
    ).reshape(
        -1,
        1,
    )

    calibration_model = (
        LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=1000,
        )
    )

    calibration_model.fit(
        logits,
        target_array,
    )

    intercept = float(
        calibration_model
        .intercept_[0]
    )

    slope = float(
        calibration_model
        .coef_[0, 0]
    )

    return (
        intercept,
        slope,
    )


def calculate_quantile_ece(
    target,
    probabilities,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Calculate equal-count expected calibration error."""

    (
        target_array,
        probability_array,
    ) = validate_binary_probabilities(
        target,
        probabilities,
    )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2."
        )

    if n_bins > len(
        probability_array
    ):
        raise ValueError(
            "n_bins cannot exceed the "
            "number of observations."
        )

    sorted_indices = np.argsort(
        probability_array,
        kind="stable",
    )

    groups = np.array_split(
        sorted_indices,
        n_bins,
    )

    total_count = len(
        probability_array
    )

    error = 0.0

    for indices in groups:
        observed_rate = float(
            target_array[
                indices
            ].mean()
        )

        predicted_rate = float(
            probability_array[
                indices
            ].mean()
        )

        weight = (
            len(indices)
            / total_count
        )

        error += (
            weight
            * abs(
                observed_rate
                - predicted_rate
            )
        )

    return float(
        error
    )


def build_calibration_curve_table(
    target,
    probabilities,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Build a quantile-binned reliability table."""

    (
        target_array,
        probability_array,
    ) = validate_binary_probabilities(
        target,
        probabilities,
    )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2."
        )

    observed, predicted = (
        calibration_curve(
            target_array,
            probability_array,
            n_bins=n_bins,
            strategy="quantile",
        )
    )

    return pd.DataFrame(
        {
            "bin": range(
                1,
                len(observed) + 1,
            ),
            "mean_predicted_probability": (
                predicted
            ),
            "observed_event_rate": (
                observed
            ),
            "calibration_error": (
                observed
                - predicted
            ),
        }
    )