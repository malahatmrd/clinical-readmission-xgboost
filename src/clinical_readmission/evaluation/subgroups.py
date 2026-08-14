from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.calibration import (
    validate_binary_probabilities,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
)
from clinical_readmission.evaluation.thresholds import (
    calculate_net_benefit,
    calculate_threshold_metrics,
)

DEFAULT_MIN_SUBGROUP_SIZE = 200
DEFAULT_MIN_SUBGROUP_POSITIVES = 20
DEFAULT_MIN_SUBGROUP_NEGATIVES = 20

MISSING_SUBGROUP_LABEL = "Missing"


def normalize_subgroup_values(
    values: Sequence,
    missing_label: str = MISSING_SUBGROUP_LABEL,
) -> np.ndarray:
    """Normalize subgroup values to reproducible string labels."""

    if not missing_label:
        raise ValueError(
            "missing_label cannot be empty."
        )

    series = pd.Series(
        values,
        dtype="object",
    )

    series = (
        series.astype("string")
        .fillna(
            missing_label
        )
    )

    return series.to_numpy(
        dtype=str
    )


def determine_reporting_eligibility(
    *,
    group_size: int,
    positive_count: int,
    negative_count: int,
    minimum_group_size: int,
    minimum_positives: int,
    minimum_negatives: int,
) -> tuple[bool, str]:
    """Determine whether subgroup performance is reportable."""

    if minimum_group_size <= 0:
        raise ValueError(
            "minimum_group_size must be positive."
        )

    if minimum_positives <= 0:
        raise ValueError(
            "minimum_positives must be positive."
        )

    if minimum_negatives <= 0:
        raise ValueError(
            "minimum_negatives must be positive."
        )

    reasons = []

    if group_size < minimum_group_size:
        reasons.append(
            (
                f"rows<{minimum_group_size}"
            )
        )

    if positive_count < minimum_positives:
        reasons.append(
            (
                f"positives<{minimum_positives}"
            )
        )

    if negative_count < minimum_negatives:
        reasons.append(
            (
                f"negatives<{minimum_negatives}"
            )
        )

    eligible = not reasons

    return (
        eligible,
        "; ".join(
            reasons
        ),
    )


def build_subgroup_performance_table(
    target,
    probabilities,
    subgroup_values: Sequence,
    subgroup_name: str,
    threshold: float,
    *,
    minimum_group_size: int = DEFAULT_MIN_SUBGROUP_SIZE,
    minimum_positives: int = DEFAULT_MIN_SUBGROUP_POSITIVES,
    minimum_negatives: int = DEFAULT_MIN_SUBGROUP_NEGATIVES,
) -> pd.DataFrame:
    """Evaluate frozen-model performance within one subgroup axis."""

    (
        target_array,
        probability_array,
    ) = validate_binary_probabilities(
        target,
        probabilities,
    )

    if not subgroup_name:
        raise ValueError(
            "subgroup_name cannot be empty."
        )

    labels = normalize_subgroup_values(
        subgroup_values
    )

    if len(
        labels
    ) != len(
        target_array
    ):
        raise ValueError(
            "subgroup_values must have the "
            "same length as target."
        )

    if not (
        0.0 < threshold < 1.0
    ):
        raise ValueError(
            "threshold must be strictly "
            "between zero and one."
        )

    rows = []

    unique_labels = sorted(
        np.unique(
            labels
        ).tolist()
    )

    for subgroup_value in (
        unique_labels
    ):
        mask = (
            labels
            == subgroup_value
        )

        subgroup_target = (
            target_array[
                mask
            ]
        )

        subgroup_probabilities = (
            probability_array[
                mask
            ]
        )

        group_size = int(
            len(
                subgroup_target
            )
        )

        positive_count = int(
            subgroup_target.sum()
        )

        negative_count = (
            group_size
            - positive_count
        )

        prevalence = float(
            subgroup_target.mean()
        )

        mean_probability = float(
            subgroup_probabilities.mean()
        )

        (
            reporting_eligible,
            exclusion_reason,
        ) = determine_reporting_eligibility(
            group_size=(
                group_size
            ),
            positive_count=(
                positive_count
            ),
            negative_count=(
                negative_count
            ),
            minimum_group_size=(
                minimum_group_size
            ),
            minimum_positives=(
                minimum_positives
            ),
            minimum_negatives=(
                minimum_negatives
            ),
        )

        row = {
            "subgroup_name": (
                subgroup_name
            ),
            "subgroup_value": (
                subgroup_value
            ),
            "rows": (
                group_size
            ),
            "positives": (
                positive_count
            ),
            "negatives": (
                negative_count
            ),
            "prevalence": (
                prevalence
            ),
            "mean_predicted_probability": (
                mean_probability
            ),
            "mean_probability_minus_prevalence": float(
                mean_probability
                - prevalence
            ),
            "threshold": float(
                threshold
            ),
            "reporting_eligible": bool(
                reporting_eligible
            ),
            "exclusion_reason": (
                exclusion_reason
            ),
        }

        if reporting_eligible:
            probability_metrics = (
                calculate_probability_metrics(
                    subgroup_target,
                    subgroup_probabilities,
                )
            )

            threshold_metrics = (
                calculate_threshold_metrics(
                    subgroup_target,
                    subgroup_probabilities,
                    threshold,
                )
            )

            net_benefit = (
                calculate_net_benefit(
                    subgroup_target,
                    subgroup_probabilities,
                    threshold,
                )
            )

            row.update(
                {
                    **probability_metrics,
                    "sensitivity": (
                        threshold_metrics[
                            "sensitivity"
                        ]
                    ),
                    "specificity": (
                        threshold_metrics[
                            "specificity"
                        ]
                    ),
                    "ppv": (
                        threshold_metrics[
                            "ppv"
                        ]
                    ),
                    "npv": (
                        threshold_metrics[
                            "npv"
                        ]
                    ),
                    "f1": (
                        threshold_metrics[
                            "f1"
                        ]
                    ),
                    "balanced_accuracy": (
                        threshold_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "alerts_per_100": (
                        threshold_metrics[
                            "alerts_per_100"
                        ]
                    ),
                    "number_needed_to_evaluate": (
                        threshold_metrics[
                            "number_needed_to_evaluate"
                        ]
                    ),
                    "model_net_benefit": (
                        net_benefit[
                            "model_net_benefit"
                        ]
                    ),
                }
            )

        else:
            row.update(
                {
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "brier_score": np.nan,
                    "log_loss": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                    "ppv": np.nan,
                    "npv": np.nan,
                    "f1": np.nan,
                    "balanced_accuracy": np.nan,
                    "alerts_per_100": np.nan,
                    "number_needed_to_evaluate": np.nan,
                    "model_net_benefit": np.nan,
                }
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def combine_subgroup_tables(
    tables: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Combine subgroup-performance tables with schema validation."""

    tables = list(
        tables
    )

    if not tables:
        raise ValueError(
            "tables cannot be empty."
        )

    reference_columns = list(
        tables[
            0
        ].columns
    )

    for table in tables:
        if list(
            table.columns
        ) != reference_columns:
            raise ValueError(
                "All subgroup tables must "
                "have the same column schema."
            )

    return pd.concat(
        tables,
        ignore_index=True,
    )