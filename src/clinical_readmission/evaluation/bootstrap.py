from __future__ import annotations

import numpy as np

from clinical_readmission.evaluation.metrics import (
    PROBABILITY_METRIC_NAMES,
    calculate_probability_metrics,
)

DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_RANDOM_STATE = 47
DEFAULT_CONFIDENCE_LEVEL = 0.95


def draw_stratified_bootstrap_indices(
    target,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one prevalence-preserving bootstrap sample."""

    target_array = np.asarray(
        target,
        dtype=int,
    )

    if target_array.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    unique_values = set(
        np.unique(
            target_array
        ).tolist()
    )

    if unique_values != {0, 1}:
        raise ValueError(
            "Target must contain both binary "
            "classes 0 and 1."
        )

    negative_indices = np.flatnonzero(
        target_array == 0
    )

    positive_indices = np.flatnonzero(
        target_array == 1
    )

    sampled_negative = rng.choice(
        negative_indices,
        size=len(negative_indices),
        replace=True,
    )

    sampled_positive = rng.choice(
        positive_indices,
        size=len(positive_indices),
        replace=True,
    )

    indices = np.concatenate(
        [
            sampled_negative,
            sampled_positive,
        ]
    )

    rng.shuffle(
        indices
    )

    return indices


def bootstrap_probability_metrics(
    target,
    probabilities,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Generate stratified bootstrap distributions."""

    target_array = np.asarray(
        target,
        dtype=int,
    )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(target_array) != len(
        probability_array
    ):
        raise ValueError(
            "Target and probabilities must have "
            "the same length."
        )

    if n_resamples < 100:
        raise ValueError(
            "n_resamples must be at least 100."
        )

    rng = np.random.default_rng(
        random_state
    )

    distributions = {
        metric: np.empty(
            n_resamples,
            dtype=float,
        )
        for metric in PROBABILITY_METRIC_NAMES
    }

    for iteration in range(
        n_resamples
    ):
        indices = (
            draw_stratified_bootstrap_indices(
                target_array,
                rng,
            )
        )

        metrics = (
            calculate_probability_metrics(
                target_array[indices],
                probability_array[indices],
            )
        )

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            distributions[
                metric
            ][
                iteration
            ] = metrics[metric]

    return distributions


def summarize_bootstrap_metrics(
    target,
    probabilities,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, dict[str, float]]:
    """Calculate point estimates and percentile bootstrap CIs."""

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be between "
            "0 and 1."
        )

    point_metrics = (
        calculate_probability_metrics(
            target,
            probabilities,
        )
    )

    distributions = (
        bootstrap_probability_metrics(
            target,
            probabilities,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    )

    alpha = (
        1.0
        - confidence_level
    )

    lower_quantile = (
        alpha / 2.0
    )

    upper_quantile = (
        1.0
        - alpha / 2.0
    )

    summary = {}

    for metric in (
        PROBABILITY_METRIC_NAMES
    ):
        values = distributions[
            metric
        ]

        summary[
            metric
        ] = {
            "estimate": float(
                point_metrics[metric]
            ),
            "ci_lower": float(
                np.quantile(
                    values,
                    lower_quantile,
                )
            ),
            "ci_upper": float(
                np.quantile(
                    values,
                    upper_quantile,
                )
            ),
            "bootstrap_standard_error": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        }

    return summary


def bootstrap_metric_difference(
    target,
    probabilities_a,
    probabilities_b,
    *,
    metric: str,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, float]:
    """Calculate paired bootstrap CI for a model-metric difference."""

    if metric not in (
        PROBABILITY_METRIC_NAMES
    ):
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    target_array = np.asarray(
        target,
        dtype=int,
    )

    probabilities_a = np.asarray(
        probabilities_a,
        dtype=float,
    )

    probabilities_b = np.asarray(
        probabilities_b,
        dtype=float,
    )

    if not (
        len(target_array)
        == len(probabilities_a)
        == len(probabilities_b)
    ):
        raise ValueError(
            "Target and probability arrays must "
            "have the same length."
        )

    if n_resamples < 100:
        raise ValueError(
            "n_resamples must be at least 100."
        )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be between "
            "0 and 1."
        )

    metrics_a = (
        calculate_probability_metrics(
            target_array,
            probabilities_a,
        )
    )

    metrics_b = (
        calculate_probability_metrics(
            target_array,
            probabilities_b,
        )
    )

    point_difference = (
        metrics_a[metric]
        - metrics_b[metric]
    )

    rng = np.random.default_rng(
        random_state
    )

    differences = np.empty(
        n_resamples,
        dtype=float,
    )

    for iteration in range(
        n_resamples
    ):
        indices = (
            draw_stratified_bootstrap_indices(
                target_array,
                rng,
            )
        )

        bootstrap_a = (
            calculate_probability_metrics(
                target_array[indices],
                probabilities_a[indices],
            )
        )

        bootstrap_b = (
            calculate_probability_metrics(
                target_array[indices],
                probabilities_b[indices],
            )
        )

        differences[
            iteration
        ] = (
            bootstrap_a[metric]
            - bootstrap_b[metric]
        )

    alpha = (
        1.0
        - confidence_level
    )

    return {
        "estimate": float(
            point_difference
        ),
        "ci_lower": float(
            np.quantile(
                differences,
                alpha / 2.0,
            )
        ),
        "ci_upper": float(
            np.quantile(
                differences,
                1.0 - alpha / 2.0,
            )
        ),
        "bootstrap_standard_error": float(
            np.std(
                differences,
                ddof=1,
            )
        ),
    }