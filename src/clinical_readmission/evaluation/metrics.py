from __future__ import annotations

from collections.abc import Mapping

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from clinical_readmission.evaluation.calibration import (
    validate_binary_probabilities,
)

PROBABILITY_METRIC_NAMES = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
)

DEFAULT_REPRODUCTION_TOLERANCE = 1e-9


def calculate_probability_metrics(
    target,
    probabilities,
) -> dict[str, float]:
    """Calculate threshold-independent probability metrics."""

    (
        target_array,
        probability_array,
    ) = validate_binary_probabilities(
        target,
        probabilities,
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                target_array,
                probability_array,
            )
        ),
        "average_precision": float(
            average_precision_score(
                target_array,
                probability_array,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                target_array,
                probability_array,
            )
        ),
        "log_loss": float(
            log_loss(
                target_array,
                probability_array,
            )
        ),
    }


def calculate_metric_deltas(
    current: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, float]:
    """Calculate current-minus-reference metric differences."""

    missing_current = sorted(
        set(PROBABILITY_METRIC_NAMES)
        - set(current)
    )

    missing_reference = sorted(
        set(PROBABILITY_METRIC_NAMES)
        - set(reference)
    )

    if missing_current:
        raise ValueError(
            "Current metrics are missing: "
            f"{missing_current}"
        )

    if missing_reference:
        raise ValueError(
            "Reference metrics are missing: "
            f"{missing_reference}"
        )

    return {
        metric: float(
            current[metric]
            - reference[metric]
        )
        for metric in PROBABILITY_METRIC_NAMES
    }


def assert_metric_reproduction(
    current: Mapping[str, float],
    reference: Mapping[str, float],
    tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE,
) -> None:
    """Fail if reproduced metrics differ from recorded metrics."""

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    deltas = calculate_metric_deltas(
        current,
        reference,
    )

    failures = {
        metric: delta
        for metric, delta in deltas.items()
        if abs(delta) > tolerance
    }

    if failures:
        details = ", ".join(
            f"{metric}={delta:+.12g}"
            for metric, delta in (
                failures.items()
            )
        )

        raise ValueError(
            "Metric reproduction failed: "
            f"{details}"
        )