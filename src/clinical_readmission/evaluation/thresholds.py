from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from clinical_readmission.evaluation.calibration import (
    validate_binary_probabilities,
)


def calculate_threshold_metrics(
    target,
    probabilities,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate clinically interpretable metrics at one threshold."""

    (
        target_array,
        probability_array,
    ) = validate_binary_probabilities(
        target,
        probabilities,
    )

    if not (
        0.0 < threshold < 1.0
    ):
        raise ValueError(
            "threshold must be strictly "
            "between 0 and 1."
        )

    predictions = (
        probability_array
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        target_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = len(
        target_array
    )

    positive_count = int(
        target_array.sum()
    )

    negative_count = (
        total
        - positive_count
    )

    predicted_positive = int(
        predictions.sum()
    )

    predicted_negative = (
        total
        - predicted_positive
    )

    sensitivity = (
        tp / positive_count
        if positive_count
        else 0.0
    )

    specificity = (
        tn / negative_count
        if negative_count
        else 0.0
    )

    ppv = (
        tp / predicted_positive
        if predicted_positive
        else 0.0
    )

    npv = (
        tn / predicted_negative
        if predicted_negative
        else 0.0
    )

    f1 = (
        2.0
        * ppv
        * sensitivity
        / (
            ppv
            + sensitivity
        )
        if (
            ppv
            + sensitivity
        )
        else 0.0
    )

    balanced_accuracy = (
        sensitivity
        + specificity
    ) / 2.0

    alert_rate = (
        predicted_positive
        / total
    )

    number_needed_to_evaluate = (
        predicted_positive
        / tp
        if tp
        else float("inf")
    )

    return {
        "threshold": float(
            threshold
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "ppv": float(
            ppv
        ),
        "npv": float(
            npv
        ),
        "f1": float(
            f1
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "alert_rate": float(
            alert_rate
        ),
        "alerts_per_100": float(
            100.0
            * alert_rate
        ),
        "true_positives_per_100": float(
            100.0
            * tp
            / total
        ),
        "false_positives_per_100": float(
            100.0
            * fp
            / total
        ),
        "number_needed_to_evaluate": float(
            number_needed_to_evaluate
        ),
    }


def calculate_net_benefit(
    target,
    probabilities,
    threshold: float,
) -> dict[str, float]:
    """Calculate model, treat-all, and treat-none net benefit."""

    metrics = calculate_threshold_metrics(
        target,
        probabilities,
        threshold,
    )

    target_array = np.asarray(
        target,
        dtype=int,
    )

    total = len(
        target_array
    )

    prevalence = float(
        target_array.mean()
    )

    threshold_odds = (
        threshold
        / (
            1.0
            - threshold
        )
    )

    model_net_benefit = (
        metrics[
            "true_positive"
        ]
        / total
        - (
            metrics[
                "false_positive"
            ]
            / total
        )
        * threshold_odds
    )

    treat_all_net_benefit = (
        prevalence
        - (
            1.0
            - prevalence
        )
        * threshold_odds
    )

    return {
        "threshold": float(
            threshold
        ),
        "model_net_benefit": float(
            model_net_benefit
        ),
        "treat_all_net_benefit": float(
            treat_all_net_benefit
        ),
        "treat_none_net_benefit": 0.0,
    }


def build_threshold_table(
    target,
    probabilities,
    thresholds,
) -> pd.DataFrame:
    """Build an operating-characteristic table."""

    threshold_array = np.asarray(
        thresholds,
        dtype=float,
    )

    if threshold_array.ndim != 1:
        raise ValueError(
            "thresholds must be "
            "one-dimensional."
        )

    if len(
        threshold_array
    ) == 0:
        raise ValueError(
            "thresholds cannot be empty."
        )

    rows = []

    for threshold in (
        threshold_array
    ):
        metrics = (
            calculate_threshold_metrics(
                target,
                probabilities,
                float(
                    threshold
                ),
            )
        )

        net_benefit = (
            calculate_net_benefit(
                target,
                probabilities,
                float(
                    threshold
                ),
            )
        )

        rows.append(
            {
                **metrics,
                "model_net_benefit": (
                    net_benefit[
                        "model_net_benefit"
                    ]
                ),
                "treat_all_net_benefit": (
                    net_benefit[
                        "treat_all_net_benefit"
                    ]
                ),
                "treat_none_net_benefit": (
                    net_benefit[
                        "treat_none_net_benefit"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )

def select_threshold_for_minimum_sensitivity(
    threshold_table: pd.DataFrame,
    minimum_sensitivity: float,
) -> pd.Series:
    """Select the highest threshold meeting a sensitivity target."""

    if not (
        0.0 < minimum_sensitivity <= 1.0
    ):
        raise ValueError(
            "minimum_sensitivity must be "
            "in (0, 1]."
        )

    eligible = threshold_table[
        threshold_table[
            "sensitivity"
        ]
        >= minimum_sensitivity
    ]

    if eligible.empty:
        raise ValueError(
            "No threshold meets the requested "
            "minimum sensitivity."
        )

    selected_index = (
        eligible[
            "threshold"
        ].idxmax()
    )

    return threshold_table.loc[
        selected_index
    ].copy()


def select_threshold_for_alert_capacity(
    threshold_table: pd.DataFrame,
    maximum_alerts_per_100: float,
) -> pd.Series:
    """Select the lowest threshold meeting an alert-capacity limit."""

    if not (
        0.0
        < maximum_alerts_per_100
        <= 100.0
    ):
        raise ValueError(
            "maximum_alerts_per_100 must be "
            "in (0, 100]."
        )

    eligible = threshold_table[
        threshold_table[
            "alerts_per_100"
        ]
        <= maximum_alerts_per_100
    ]

    if eligible.empty:
        raise ValueError(
            "No threshold meets the requested "
            "alert-capacity limit."
        )

    selected_index = (
        eligible[
            "threshold"
        ].idxmin()
    )

    return threshold_table.loc[
        selected_index
    ].copy()