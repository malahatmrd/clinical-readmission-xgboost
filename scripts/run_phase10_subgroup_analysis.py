from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
)
from clinical_readmission.evaluation.subgroups import (
    DEFAULT_MIN_SUBGROUP_NEGATIVES,
    DEFAULT_MIN_SUBGROUP_POSITIVES,
    DEFAULT_MIN_SUBGROUP_SIZE,
    build_subgroup_performance_table,
    combine_subgroup_tables,
)
from clinical_readmission.evaluation.thresholds import (
    calculate_net_benefit,
    calculate_threshold_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
    / "primary.csv"
)

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "primary_split_assignments.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)

PHASE8_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_threshold_selection.json"
)

PHASE9_SHAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_shap_validation.json"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_validation_subgroup_performance.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_subgroup_validation.json"
)

TARGET_COLUMN = "readmitted_30d"

CALIBRATED_PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_THRESHOLD = 0.105

FORBIDDEN_OUTPUT_IDENTIFIERS = {
    "encounter_id",
    "patient_nbr",
    "source_row",
}

AGE_GROUP_MAP = {
    "[0-10)": "<50",
    "[10-20)": "<50",
    "[20-30)": "<50",
    "[30-40)": "<50",
    "[40-50)": "<50",
    "[50-60)": "50-69",
    "[60-70)": "50-69",
    "[70-80)": "70-89",
    "[80-90)": "70-89",
    "[90-100)": "90+",
}


def load_json(
    path: Path,
) -> dict:
    """Load a JSON artifact."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def file_sha256(
    path: Path,
) -> str:
    """Calculate SHA256 for an artifact."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_partition(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Load one frozen split in deterministic cohort order."""

    ids = assignments.loc[
        assignments[
            "split"
        ].eq(
            split_name
        ),
        [
            "encounter_id",
            "patient_nbr",
            TARGET_COLUMN,
        ],
    ].copy()

    result = cohort.merge(
        ids,
        on=[
            "encounter_id",
            "patient_nbr",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_assignment",
        ),
    )

    assignment_target = (
        f"{TARGET_COLUMN}_assignment"
    )

    if not (
        result[
            TARGET_COLUMN
        ]
        == result[
            assignment_target
        ]
    ).all():
        raise ValueError(
            f"{split_name}: target mismatch."
        )

    return result


def normalize_missing_tokens(
    values,
) -> pd.Series:
    """Convert dataset missing markers to missing values."""

    result = pd.Series(
        values,
        dtype="string",
    )

    return result.replace(
        {
            "?": pd.NA,
            "": pd.NA,
        }
    )


def build_age_groups(
    values,
) -> np.ndarray:
    """Create predefined broad age groups."""

    age = normalize_missing_tokens(
        values
    )

    observed = set(
        age.dropna().unique()
    )

    unknown = sorted(
        observed
        - set(
            AGE_GROUP_MAP
        )
    )

    if unknown:
        raise ValueError(
            "Unexpected age categories: "
            f"{unknown}"
        )

    grouped = age.map(
        AGE_GROUP_MAP
    ).fillna(
        "Missing"
    )

    return grouped.to_numpy(
        dtype=str
    )


def build_demographic_labels(
    values,
) -> np.ndarray:
    """Normalize demographic subgroup labels."""

    return (
        normalize_missing_tokens(
            values
        )
        .fillna(
            "Missing"
        )
        .to_numpy(
            dtype=str
        )
    )


def main() -> None:
    phase8_selection = load_json(
        PHASE8_SELECTION_PATH
    )

    phase9_summary = load_json(
        PHASE9_SHAP_PATH
    )

    if (
        phase8_selection[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    if (
        phase8_selection[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test set must remain locked."
        )

    threshold = float(
        phase8_selection[
            "reference_threshold"
        ]
    )

    if abs(
        threshold
        - EXPECTED_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected frozen threshold."
        )

    expected_prediction_hash = (
        phase9_summary[
            "source_artifacts"
        ][
            "phase7_predictions"
        ][
            "sha256"
        ]
    )

    observed_prediction_hash = (
        file_sha256(
            PREDICTIONS_PATH
        )
    )

    if (
        observed_prediction_hash
        != expected_prediction_hash
    ):
        raise ValueError(
            "Phase 7 prediction artifact "
            "SHA256 mismatch."
        )

    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    if (
        FORBIDDEN_OUTPUT_IDENTIFIERS
        & set(
            predictions.columns
        )
    ):
        raise ValueError(
            "Prediction artifact contains "
            "forbidden identifiers."
        )

    validation = load_partition(
        cohort,
        assignments,
        "validation",
    ).reset_index(
        drop=True
    )

    expected_validation_rows = (
        np.arange(
            len(
                validation
            )
        )
    )

    observed_validation_rows = (
        predictions[
            "validation_row"
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        expected_validation_rows,
        observed_validation_rows,
    ):
        raise ValueError(
            "Validation row alignment "
            "check failed."
        )

    y_validation = (
        validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    recorded_target = (
        predictions[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        y_validation,
        recorded_target,
    ):
        raise ValueError(
            "Validation target order "
            "does not match predictions."
        )

    probabilities = (
        predictions[
            CALIBRATED_PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    print(
        "=" * 96
    )
    print(
        "PHASE 10 VALIDATION SUBGROUP ANALYSIS"
    )
    print(
        "=" * 96
    )

    print(
        "\nFrozen model        :",
        EXPECTED_MODEL,
    )

    print(
        "Frozen threshold    :",
        f"{threshold:.3f}",
    )

    print(
        "Validation rows     :",
        len(
            validation
        ),
    )

    print(
        "Validation positives:",
        int(
            y_validation.sum()
        ),
    )

    print(
        "Test used           : False"
    )

    print(
        "\nReporting eligibility:"
    )

    print(
        "  minimum rows      :",
        DEFAULT_MIN_SUBGROUP_SIZE,
    )

    print(
        "  minimum positives :",
        DEFAULT_MIN_SUBGROUP_POSITIVES,
    )

    print(
        "  minimum negatives :",
        DEFAULT_MIN_SUBGROUP_NEGATIVES,
    )

    gender_values = (
        build_demographic_labels(
            validation[
                "gender"
            ]
        )
    )

    race_values = (
        build_demographic_labels(
            validation[
                "race"
            ]
        )
    )

    age_values = (
        build_age_groups(
            validation[
                "age"
            ]
        )
    )

    gender_table = (
        build_subgroup_performance_table(
            y_validation,
            probabilities,
            gender_values,
            subgroup_name="gender",
            threshold=threshold,
        )
    )

    race_table = (
        build_subgroup_performance_table(
            y_validation,
            probabilities,
            race_values,
            subgroup_name="race",
            threshold=threshold,
        )
    )

    age_table = (
        build_subgroup_performance_table(
            y_validation,
            probabilities,
            age_values,
            subgroup_name="age_group",
            threshold=threshold,
        )
    )

    subgroup_table = (
        combine_subgroup_tables(
            [
                gender_table,
                race_table,
                age_table,
            ]
        )
    )

    if (
        FORBIDDEN_OUTPUT_IDENTIFIERS
        & set(
            subgroup_table.columns
        )
    ):
        raise ValueError(
            "Subgroup output contains "
            "forbidden identifiers."
        )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subgroup_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    probability_metrics = (
        calculate_probability_metrics(
            y_validation,
            probabilities,
        )
    )

    threshold_metrics = (
        calculate_threshold_metrics(
            y_validation,
            probabilities,
            threshold,
        )
    )

    net_benefit = (
        calculate_net_benefit(
            y_validation,
            probabilities,
            threshold,
        )
    )

    overall_reference = {
        **probability_metrics,
        "prevalence": float(
            y_validation.mean()
        ),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
        "sensitivity": float(
            threshold_metrics[
                "sensitivity"
            ]
        ),
        "specificity": float(
            threshold_metrics[
                "specificity"
            ]
        ),
        "ppv": float(
            threshold_metrics[
                "ppv"
            ]
        ),
        "npv": float(
            threshold_metrics[
                "npv"
            ]
        ),
        "f1": float(
            threshold_metrics[
                "f1"
            ]
        ),
        "balanced_accuracy": float(
            threshold_metrics[
                "balanced_accuracy"
            ]
        ),
        "alerts_per_100": float(
            threshold_metrics[
                "alerts_per_100"
            ]
        ),
        "model_net_benefit": float(
            net_benefit[
                "model_net_benefit"
            ]
        ),
    }

    subgroup_counts = {}

    for subgroup_name in (
        "gender",
        "race",
        "age_group",
    ):
        current = subgroup_table.loc[
            subgroup_table[
                "subgroup_name"
            ].eq(
                subgroup_name
            )
        ]

        subgroup_counts[
            subgroup_name
        ] = {
            "groups_total": int(
                len(
                    current
                )
            ),
            "groups_reportable": int(
                current[
                    "reporting_eligible"
                ].sum()
            ),
            "groups_not_reportable": int(
                (
                    ~current[
                        "reporting_eligible"
                    ]
                ).sum()
            ),
        }

    summary = {
        "phase": 10,
        "analysis": (
            "validation_subgroup_performance"
        ),
        "frozen_configuration": {
            "model": EXPECTED_MODEL,
            "reference_threshold": (
                threshold
            ),
            "model_retrained": False,
            "calibration_reselected": False,
            "threshold_reselected": False,
        },
        "subgroup_protocol": {
            "axes": [
                "gender",
                "race",
                "age_group",
            ],
            "age_groups": (
                AGE_GROUP_MAP
            ),
            "minimum_group_size": (
                DEFAULT_MIN_SUBGROUP_SIZE
            ),
            "minimum_positives": (
                DEFAULT_MIN_SUBGROUP_POSITIVES
            ),
            "minimum_negatives": (
                DEFAULT_MIN_SUBGROUP_NEGATIVES
            ),
            "subgroups_defined_before_metric_review": (
                True
            ),
            "small_groups_retained_in_table": (
                True
            ),
            "small_group_metrics_suppressed": (
                True
            ),
        },
        "sample_counts": {
            "validation": int(
                len(
                    validation
                )
            ),
            "validation_positive": int(
                y_validation.sum()
            ),
            "validation_negative": int(
                len(
                    validation
                )
                - y_validation.sum()
            ),
        },
        "overall_reference": (
            overall_reference
        ),
        "subgroup_counts": (
            subgroup_counts
        ),
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": (
                observed_prediction_hash
            ),
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "privacy": {
            "encounter_id_saved": False,
            "patient_nbr_saved": False,
            "source_row_saved": False,
            "validation_row_saved": False,
        },
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "test_used": False,
        },
    }

    OUTPUT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nSUBGROUP RESULTS"
    )
    print(
        "-" * 96
    )

    display_columns = [
        "subgroup_name",
        "subgroup_value",
        "rows",
        "positives",
        "prevalence",
        "reporting_eligible",
        "roc_auc",
        "average_precision",
        "brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "alerts_per_100",
    ]

    print(
        subgroup_table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nOverall Validation ROC-AUC :",
        f"{probability_metrics['roc_auc']:.6f}",
    )

    print(
        "Overall Validation AP      :",
        (
            f"{probability_metrics['average_precision']:.6f}"
        ),
    )

    print(
        "Overall Validation Brier   :",
        f"{probability_metrics['brier_score']:.6f}",
    )

    print(
        "\nSaved table:",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved summary:",
        OUTPUT_SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used: False"
    )


if __name__ == "__main__":
    main()