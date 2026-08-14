from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.calibration import (
    DEFAULT_CALIBRATION_BINS,
    calculate_calibration_intercept_slope,
    calculate_quantile_ece,
)
from clinical_readmission.evaluation.calibration_models import (
    build_calibrated_classifier,
    build_tuned_pipeline,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
)
from clinical_readmission.evaluation.thresholds import (
    calculate_net_benefit,
    calculate_threshold_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
    / "primary.csv"
)

ALL_ELIGIBLE_COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
    / "all_eligible_encounters.csv"
)

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "primary_split_assignments.csv"
)

TUNED_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_tuned_validation.json"
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

PHASE7_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_repeated_encounter_robustness.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_repeated_encounter_robustness.json"
)

TARGET_COLUMN = "readmitted_30d"

CALIBRATED_PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_THRESHOLD = 0.105

PROBABILITY_REPRODUCTION_TOLERANCE = 1e-7


def load_json(
    path: Path,
) -> dict:
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
    digest = hashlib.sha256()

    with path.open(
        "rb",
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


def evaluate_dataset(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target = (
        data[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    probability_metrics = (
        calculate_probability_metrics(
            target,
            probabilities,
        )
    )

    threshold_metrics = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold,
        )
    )

    net_benefit = (
        calculate_net_benefit(
            target,
            probabilities,
            threshold,
        )
    )

    intercept, slope = (
        calculate_calibration_intercept_slope(
            target,
            probabilities,
        )
    )

    ece = calculate_quantile_ece(
        target,
        probabilities,
        n_bins=DEFAULT_CALIBRATION_BINS,
    )

    prevalence = float(
        target.mean()
    )

    mean_probability = float(
        probabilities.mean()
    )

    return {
        "encounters": int(
            len(
                data
            )
        ),
        "unique_patients": int(
            data[
                "patient_nbr"
            ].nunique()
        ),
        "positives": int(
            target.sum()
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
        **probability_metrics,
        "calibration_intercept": float(
            intercept
        ),
        "calibration_slope": float(
            slope
        ),
        "quantile_ece": float(
            ece
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
        "number_needed_to_evaluate": float(
            threshold_metrics[
                "number_needed_to_evaluate"
            ]
        ),
        "model_net_benefit": float(
            net_benefit[
                "model_net_benefit"
            ]
        ),
    }


def main() -> None:
    phase8 = load_json(
        PHASE8_SELECTION_PATH
    )

    phase9 = load_json(
        PHASE9_SHAP_PATH
    )

    tuned_artifact = load_json(
        TUNED_METRICS_PATH
    )

    if (
        phase8[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    if (
        phase8[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test set must remain locked."
        )

    threshold = float(
        phase8[
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

    prediction_hash = file_sha256(
        PHASE7_PREDICTIONS_PATH
    )

    expected_prediction_hash = (
        phase9[
            "source_artifacts"
        ][
            "phase7_predictions"
        ][
            "sha256"
        ]
    )

    if (
        prediction_hash
        != expected_prediction_hash
    ):
        raise ValueError(
            "Phase 7 prediction artifact "
            "SHA256 mismatch."
        )

    primary = pd.read_csv(
        PRIMARY_COHORT_PATH,
        low_memory=False,
    )

    all_eligible = pd.read_csv(
        ALL_ELIGIBLE_COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    recorded_predictions = pd.read_csv(
        PHASE7_PREDICTIONS_PATH
    )

    train = load_partition(
        primary,
        assignments,
        "train",
    ).reset_index(
        drop=True
    )

    validation = load_partition(
        primary,
        assignments,
        "validation",
    ).reset_index(
        drop=True
    )

    y_train = train[
        TARGET_COLUMN
    ]

    y_validation = (
        validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        y_validation,
        recorded_predictions[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        ),
    ):
        raise ValueError(
            "Validation target order does not "
            "match Phase 7 predictions."
        )

    validation_patients = set(
        validation[
            "patient_nbr"
        ].tolist()
    )

    train_patients = set(
        train[
            "patient_nbr"
        ].tolist()
    )

    if (
        validation_patients
        & train_patients
    ):
        raise ValueError(
            "Train/Validation patient overlap."
        )

    repeated_validation = (
        all_eligible.loc[
            all_eligible[
                "patient_nbr"
            ].isin(
                validation_patients
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    observed_validation_patients = set(
        repeated_validation[
            "patient_nbr"
        ].unique()
    )

    if (
        observed_validation_patients
        != validation_patients
    ):
        raise ValueError(
            "Repeated-encounter robustness "
            "does not contain exactly the "
            "Validation patient set."
        )

    primary_encounter_ids = set(
        validation[
            "encounter_id"
        ].tolist()
    )

    subsequent_validation = (
        repeated_validation.loc[
            ~repeated_validation[
                "encounter_id"
            ].isin(
                primary_encounter_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        set(
            repeated_validation[
                "patient_nbr"
            ]
        )
        & train_patients
    ):
        raise ValueError(
            "Robustness cohort contains "
            "Train patients."
        )

    tuned_parameters = (
        tuned_artifact[
            "selected_hyperparameters"
        ]
    )

    tree_count = int(
        tuned_artifact[
            "development_protocol"
        ][
            "selected_tree_count"
        ]
    )

    print(
        "=" * 104
    )
    print(
        "PHASE 10 REPEATED-ENCOUNTER ROBUSTNESS"
    )
    print(
        "=" * 104
    )

    print(
        "\nFrozen model       :",
        EXPECTED_MODEL,
    )

    print(
        "Frozen tree count  :",
        tree_count,
    )

    print(
        "Frozen threshold   :",
        f"{threshold:.3f}",
    )

    print(
        "Train rows         :",
        len(
            train
        ),
    )

    print(
        "Validation patients:",
        len(
            validation_patients
        ),
    )

    print(
        "Test used          : False"
    )

    print(
        "\nRebuilding frozen "
        "Train-only sigmoid calibrated model..."
    )

    estimator = build_tuned_pipeline(
        tuned_parameters,
        n_estimators=tree_count,
    )

    model = build_calibrated_classifier(
        estimator,
        "sigmoid",
    )

    model.fit(
        train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            validation
        )[
            :,
            1
        ]
    )

    recorded_validation_probabilities = (
        recorded_predictions[
            CALIBRATED_PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    maximum_reproduction_error = float(
        np.max(
            np.abs(
                validation_probabilities
                - recorded_validation_probabilities
            )
        )
    )

    print(
        "Maximum Validation probability "
        "reproduction error:",
        f"{maximum_reproduction_error:.12g}",
    )

    if (
        maximum_reproduction_error
        > PROBABILITY_REPRODUCTION_TOLERANCE
    ):
        raise ValueError(
            "Frozen calibrated probability "
            "reproduction failed."
        )

    print(
        "Validation probability reproduction: PASS"
    )

    repeated_probabilities = (
        model.predict_proba(
            repeated_validation
        )[
            :,
            1
        ]
    )

    primary_metrics = evaluate_dataset(
        validation,
        validation_probabilities,
        threshold,
    )

    repeated_metrics = evaluate_dataset(
        repeated_validation,
        repeated_probabilities,
        threshold,
    )

    results = {
        "primary_validation": (
            primary_metrics
        ),
        "all_eligible_encounters_for_validation_patients": (
            repeated_metrics
        ),
    }

    if len(
        subsequent_validation
    ):
        subsequent_probabilities = (
            model.predict_proba(
                subsequent_validation
            )[
                :,
                1
            ]
        )

        subsequent_target = (
            subsequent_validation[
                TARGET_COLUMN
            ].to_numpy(
                dtype=int
            )
        )

        if len(
            np.unique(
                subsequent_target
            )
        ) == 2:
            subsequent_metrics = (
                evaluate_dataset(
                    subsequent_validation,
                    subsequent_probabilities,
                    threshold,
                )
            )

            results[
                "subsequent_eligible_encounters_only"
            ] = (
                subsequent_metrics
            )

    table_rows = []

    for (
        dataset_name,
        metrics,
    ) in results.items():
        table_rows.append(
            {
                "dataset": (
                    dataset_name
                ),
                **metrics,
            }
        )

    table = pd.DataFrame(
        table_rows
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    robustness_delta = {}

    for metric in (
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "calibration_intercept",
        "calibration_slope",
        "quantile_ece",
        "sensitivity",
        "specificity",
        "ppv",
        "alerts_per_100",
        "model_net_benefit",
    ):
        robustness_delta[
            metric
        ] = float(
            repeated_metrics[
                metric
            ]
            - primary_metrics[
                metric
            ]
        )

    output = {
        "phase": 10,
        "analysis": (
            "repeated_encounter_robustness"
        ),
        "analysis_role": (
            "distribution_and_encounter_position_robustness"
        ),
        "frozen_configuration": {
            "model": (
                EXPECTED_MODEL
            ),
            "tree_count": (
                tree_count
            ),
            "calibration_method": (
                "sigmoid"
            ),
            "reference_threshold": (
                threshold
            ),
            "model_specification_reselected": (
                False
            ),
            "calibration_reselected": (
                False
            ),
            "threshold_reselected": (
                False
            ),
        },
        "reproduction_audit": {
            "maximum_validation_probability_error": (
                maximum_reproduction_error
            ),
            "tolerance": (
                PROBABILITY_REPRODUCTION_TOLERANCE
            ),
            "passed": (
                maximum_reproduction_error
                <= PROBABILITY_REPRODUCTION_TOLERANCE
            ),
        },
        "cohort_protocol": {
            "training_cohort": (
                "primary_train"
            ),
            "reference_evaluation": (
                "primary_validation"
            ),
            "robustness_cohort_source": (
                "all_eligible_encounters"
            ),
            "robustness_patient_partition": (
                "original_validation_patients_only"
            ),
            "repeated_encounters_allowed": (
                True
            ),
            "new_split_created": (
                False
            ),
            "test_patients_used": (
                False
            ),
        },
        "results": (
            results
        ),
        "all_eligible_minus_primary_validation": (
            robustness_delta
        ),
        "source_artifacts": {
            "phase7_predictions": {
                "path": str(
                    PHASE7_PREDICTIONS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": (
                    prediction_hash
                ),
            },
            "all_eligible_cohort": {
                "path": str(
                    ALL_ELIGIBLE_COHORT_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": file_sha256(
                    ALL_ELIGIBLE_COHORT_PATH
                ),
            },
        },
        "privacy": {
            "row_level_predictions_saved": (
                False
            ),
            "patient_identifiers_saved": (
                False
            ),
        },
        "limitations": {
            "repeated_encounters_are_correlated_within_patient": (
                True
            ),
            "not_external_validation": (
                True
            ),
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "data_policy": {
            "fit_split": (
                "train"
            ),
            "robustness_partition": (
                "validation_patients"
            ),
            "test_used": (
                False
            ),
        },
    }

    OUTPUT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nROBUSTNESS RESULTS"
    )

    print(
        "-" * 104
    )

    display_columns = [
        "dataset",
        "encounters",
        "unique_patients",
        "positives",
        "prevalence",
        "roc_auc",
        "average_precision",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
        "sensitivity",
        "specificity",
        "ppv",
        "alerts_per_100",
    ]

    print(
        table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nAll-eligible minus primary:"
    )

    for metric in (
        "roc_auc",
        "average_precision",
        "brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "alerts_per_100",
    ):
        print(
            f"  {metric:<20}: "
            f"{robustness_delta[metric]:+.6f}"
        )

    print(
        "\nSaved table:",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved JSON :",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nNew split created    : False"
    )

    print(
        "Threshold reselected : False"
    )

    print(
        "Test used            : False"
    )


if __name__ == "__main__":
    main()