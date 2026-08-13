from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from clinical_readmission.features.feature_schema import (
    MODEL_INPUT_FEATURES,
)


def validate_shap_matrix(
    shap_values,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Validate a two-dimensional SHAP matrix."""

    values = np.asarray(
        shap_values,
        dtype=float,
    )

    if values.ndim != 2:
        raise ValueError(
            "SHAP values must be two-dimensional."
        )

    names = list(
        feature_names
    )

    if values.shape[1] != len(
        names
    ):
        raise ValueError(
            "SHAP feature count does not match "
            "feature_names."
        )

    if len(
        names
    ) != len(
        set(names)
    ):
        raise ValueError(
            "feature_names must be unique."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "SHAP values must be finite."
        )

    return values


def infer_source_feature(
    transformed_feature_name: str,
) -> str:
    """Map one transformed feature back to its source feature."""

    if "__" not in (
        transformed_feature_name
    ):
        raise ValueError(
            "Transformed feature name must "
            "contain a transformer prefix."
        )

    _, local_name = (
        transformed_feature_name.split(
            "__",
            maxsplit=1,
        )
    )

    candidates = [
        feature
        for feature in MODEL_INPUT_FEATURES
        if (
            local_name == feature
            or local_name.startswith(
                f"{feature}_"
            )
        )
    ]

    if not candidates:
        raise ValueError(
            "Could not map transformed feature "
            f"to source feature: "
            f"{transformed_feature_name}"
        )

    return max(
        candidates,
        key=len,
    )


def build_transformed_feature_metadata(
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Build metadata linking transformed and source features."""

    names = list(
        feature_names
    )

    if len(
        names
    ) != len(
        set(names)
    ):
        raise ValueError(
            "feature_names must be unique."
        )

    rows = []

    for index, name in enumerate(
        names
    ):
        if "__" not in name:
            raise ValueError(
                "Transformed feature name must "
                "contain a transformer prefix."
            )

        transformer, _ = (
            name.split(
                "__",
                maxsplit=1,
            )
        )

        rows.append(
            {
                "transformed_index": (
                    index
                ),
                "transformer": (
                    transformer
                ),
                "transformed_feature": (
                    name
                ),
                "source_feature": (
                    infer_source_feature(
                        name
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def calculate_transformed_shap_importance(
    shap_values,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Rank transformed features by mean absolute SHAP value."""

    values = validate_shap_matrix(
        shap_values,
        feature_names,
    )

    names = list(
        feature_names
    )

    mean_absolute = np.mean(
        np.abs(
            values
        ),
        axis=0,
    )

    mean_signed = np.mean(
        values,
        axis=0,
    )

    table = pd.DataFrame(
        {
            "transformed_feature": (
                names
            ),
            "mean_abs_shap": (
                mean_absolute
            ),
            "mean_signed_shap": (
                mean_signed
            ),
        }
    )

    table = (
        table.sort_values(
            "mean_abs_shap",
            ascending=False,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    table.insert(
        0,
        "rank",
        np.arange(
            1,
            len(table) + 1,
        ),
    )

    return table


def calculate_source_shap_importance(
    shap_values,
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transformed SHAP values to source features."""

    required_columns = {
        "transformed_index",
        "transformed_feature",
        "source_feature",
    }

    missing = (
        required_columns
        - set(
            feature_metadata.columns
        )
    )

    if missing:
        raise ValueError(
            "Feature metadata is missing columns: "
            f"{sorted(missing)}"
        )

    feature_names = (
        feature_metadata[
            "transformed_feature"
        ].tolist()
    )

    values = validate_shap_matrix(
        shap_values,
        feature_names,
    )

    expected_indices = np.arange(
        values.shape[1]
    )

    observed_indices = (
        feature_metadata[
            "transformed_index"
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        observed_indices,
        expected_indices,
    ):
        raise ValueError(
            "Feature metadata indices must "
            "match SHAP column order."
        )

    rows = []

    for source_feature in (
        feature_metadata[
            "source_feature"
        ].drop_duplicates()
    ):
        source_rows = (
            feature_metadata[
                feature_metadata[
                    "source_feature"
                ].eq(
                    source_feature
                )
            ]
        )

        indices = (
            source_rows[
                "transformed_index"
            ].to_numpy(
                dtype=int
            )
        )

        grouped_values = (
            values[
                :,
                indices,
            ].sum(
                axis=1
            )
        )

        rows.append(
            {
                "source_feature": (
                    source_feature
                ),
                "transformed_feature_count": int(
                    len(indices)
                ),
                "mean_abs_shap": float(
                    np.mean(
                        np.abs(
                            grouped_values
                        )
                    )
                ),
                "mean_signed_shap": float(
                    np.mean(
                        grouped_values
                    )
                ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    table = (
        table.sort_values(
            "mean_abs_shap",
            ascending=False,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    table.insert(
        0,
        "rank",
        np.arange(
            1,
            len(table) + 1,
        ),
    )

    return table

def select_local_explanation_cases(
    target,
    probabilities,
    threshold: float,
) -> pd.DataFrame:
    """Select reproducible Validation cases for local explanation."""

    target_array = np.asarray(
        target,
        dtype=int,
    )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if target_array.ndim != 1:
        raise ValueError(
            "target must be one-dimensional."
        )

    if probability_array.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional."
        )

    if len(target_array) != len(
        probability_array
    ):
        raise ValueError(
            "target and probabilities must "
            "have the same length."
        )

    if not (
        0.0 < threshold < 1.0
    ):
        raise ValueError(
            "threshold must be between zero and one."
        )

    if not np.isin(
        target_array,
        [
            0,
            1,
        ],
    ).all():
        raise ValueError(
            "target must be binary."
        )

    if not np.isfinite(
        probability_array
    ).all():
        raise ValueError(
            "probabilities must be finite."
        )

    if (
        (probability_array < 0.0).any()
        or (probability_array > 1.0).any()
    ):
        raise ValueError(
            "probabilities must be between "
            "zero and one."
        )

    predicted_positive = (
        probability_array
        >= threshold
    )

    masks = {
        "high_confidence_true_positive": (
            (target_array == 1)
            & predicted_positive
        ),
        "high_confidence_false_positive": (
            (target_array == 0)
            & predicted_positive
        ),
        "near_threshold_false_negative": (
            (target_array == 1)
            & ~predicted_positive
        ),
        "low_risk_true_negative": (
            (target_array == 0)
            & ~predicted_positive
        ),
    }

    for (
        case_name,
        mask,
    ) in masks.items():
        if not mask.any():
            raise ValueError(
                "No eligible observation for "
                f"local case: {case_name}"
            )

    selected = {}

    tp_indices = np.flatnonzero(
        masks[
            "high_confidence_true_positive"
        ]
    )

    selected[
        "high_confidence_true_positive"
    ] = int(
        tp_indices[
            np.argmax(
                probability_array[
                    tp_indices
                ]
            )
        ]
    )

    fp_indices = np.flatnonzero(
        masks[
            "high_confidence_false_positive"
        ]
    )

    selected[
        "high_confidence_false_positive"
    ] = int(
        fp_indices[
            np.argmax(
                probability_array[
                    fp_indices
                ]
            )
        ]
    )

    fn_indices = np.flatnonzero(
        masks[
            "near_threshold_false_negative"
        ]
    )

    selected[
        "near_threshold_false_negative"
    ] = int(
        fn_indices[
            np.argmax(
                probability_array[
                    fn_indices
                ]
            )
        ]
    )

    tn_indices = np.flatnonzero(
        masks[
            "low_risk_true_negative"
        ]
    )

    selected[
        "low_risk_true_negative"
    ] = int(
        tn_indices[
            np.argmin(
                probability_array[
                    tn_indices
                ]
            )
        ]
    )

    already_selected = set(
        selected.values()
    )

    remaining_indices = np.array(
        [
            index
            for index in range(
                len(
                    target_array
                )
            )
            if index not in (
                already_selected
            )
        ],
        dtype=int,
    )

    if not len(
        remaining_indices
    ):
        raise ValueError(
            "No unused observation remains "
            "for near-threshold selection."
        )

    distance = np.abs(
        probability_array[
            remaining_indices
        ]
        - threshold
    )

    selected[
        "closest_unused_to_threshold"
    ] = int(
        remaining_indices[
            np.argmin(
                distance
            )
        ]
    )

    rows = []

    for (
        case_name,
        index,
    ) in selected.items():
        rows.append(
            {
                "case_name": (
                    case_name
                ),
                "validation_row": int(
                    index
                ),
                "y_true": int(
                    target_array[
                        index
                    ]
                ),
                "calibrated_probability": float(
                    probability_array[
                        index
                    ]
                ),
                "predicted_positive": bool(
                    predicted_positive[
                        index
                    ]
                ),
                "threshold": float(
                    threshold
                ),
                "absolute_distance_to_threshold": float(
                    abs(
                        probability_array[
                            index
                        ]
                        - threshold
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def aggregate_local_shap_to_source(
    shap_values,
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one local SHAP vector to source clinical features."""

    values = np.asarray(
        shap_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Local SHAP values must be "
            "one-dimensional."
        )

    required_columns = {
        "transformed_index",
        "source_feature",
    }

    missing = (
        required_columns
        - set(
            feature_metadata.columns
        )
    )

    if missing:
        raise ValueError(
            "Feature metadata is missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        feature_metadata
    ) != len(
        values
    ):
        raise ValueError(
            "Feature metadata length does not "
            "match local SHAP vector."
        )

    indices = (
        feature_metadata[
            "transformed_index"
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        indices,
        np.arange(
            len(
                values
            )
        ),
    ):
        raise ValueError(
            "Feature metadata indices must "
            "match SHAP column order."
        )

    rows = []

    for source_feature in (
        feature_metadata[
            "source_feature"
        ].drop_duplicates()
    ):
        mask = (
            feature_metadata[
                "source_feature"
            ].eq(
                source_feature
            )
            .to_numpy()
        )

        contribution = float(
            values[
                mask
            ].sum()
        )

        rows.append(
            {
                "source_feature": (
                    source_feature
                ),
                "transformed_feature_count": int(
                    mask.sum()
                ),
                "shap_value": (
                    contribution
                ),
                "abs_shap_value": float(
                    abs(
                        contribution
                    )
                ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    table = (
        table.sort_values(
            "abs_shap_value",
            ascending=False,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    table.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                table
            ) + 1,
        ),
    )

    return table