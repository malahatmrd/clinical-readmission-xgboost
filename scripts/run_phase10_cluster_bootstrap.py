from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.bootstrap import (
    DEFAULT_BOOTSTRAP_RANDOM_STATE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
)
from clinical_readmission.evaluation.calibration_models import (
    build_calibrated_classifier,
    build_tuned_pipeline,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
)
from clinical_readmission.evaluation.thresholds import (
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
    / "phase10_cluster_bootstrap.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_cluster_bootstrap.json"
)

TARGET_COLUMN = "readmitted_30d"

CALIBRATED_PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_THRESHOLD = 0.105

PROBABILITY_REPRODUCTION_TOLERANCE = 1e-7

BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "sensitivity",
    "specificity",
    "ppv",
    "alerts_per_100",
)


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


def calculate_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
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

    return {
        "roc_auc": float(
            probability_metrics[
                "roc_auc"
            ]
        ),
        "average_precision": float(
            probability_metrics[
                "average_precision"
            ]
        ),
        "brier_score": float(
            probability_metrics[
                "brier_score"
            ]
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
        "alerts_per_100": float(
            threshold_metrics[
                "alerts_per_100"
            ]
        ),
    }


def summarize_distribution(
    values: np.ndarray,
) -> dict[str, float]:
    alpha = (
        1.0
        - DEFAULT_CONFIDENCE_LEVEL
    )

    return {
        "ci_lower": float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        "ci_upper": float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
        "bootstrap_standard_error": float(
            np.std(
                values,
                ddof=1,
            )
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
            "Phase 7 prediction SHA256 mismatch."
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

    validation_patients = (
        validation[
            "patient_nbr"
        ].to_numpy()
    )

    if len(
        np.unique(
            validation_patients
        )
    ) != len(
        validation_patients
    ):
        raise ValueError(
            "Primary Validation must contain "
            "one encounter per patient."
        )

    repeated_validation = (
        all_eligible.loc[
            all_eligible[
                "patient_nbr"
            ].isin(
                set(
                    validation_patients
                )
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if set(
        repeated_validation[
            "patient_nbr"
        ].unique()
    ) != set(
        validation_patients
    ):
        raise ValueError(
            "Repeated cohort patient set mismatch."
        )

    y_train = train[
        TARGET_COLUMN
    ]

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
        "PHASE 10 PATIENT-CLUSTER BOOTSTRAP"
    )

    print(
        "=" * 104
    )

    print(
        "\nBootstrap unit      : patient"
    )

    print(
        "Bootstrap resamples:",
        DEFAULT_BOOTSTRAP_RESAMPLES,
    )

    print(
        "Bootstrap seed     :",
        DEFAULT_BOOTSTRAP_RANDOM_STATE,
    )

    print(
        "Confidence level   :",
        DEFAULT_CONFIDENCE_LEVEL,
    )

    print(
        "Frozen threshold   :",
        f"{threshold:.3f}",
    )

    print(
        "Validation patients:",
        len(
            validation_patients
        ),
    )

    print(
        "Repeated encounters:",
        len(
            repeated_validation
        ),
    )

    print(
        "Test used          : False"
    )

    estimator = build_tuned_pipeline(
        tuned_parameters,
        n_estimators=tree_count,
    )

    model = build_calibrated_classifier(
        estimator,
        "sigmoid",
    )

    print(
        "\nRebuilding frozen calibrated model..."
    )

    model.fit(
        train,
        y_train,
    )

    primary_probabilities = (
        model.predict_proba(
            validation
        )[
            :,
            1
        ]
    )

    recorded_probabilities = (
        recorded_predictions[
            CALIBRATED_PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    maximum_error = float(
        np.max(
            np.abs(
                primary_probabilities
                - recorded_probabilities
            )
        )
    )

    print(
        "Maximum Validation reproduction error:",
        f"{maximum_error:.12g}",
    )

    if (
        maximum_error
        > PROBABILITY_REPRODUCTION_TOLERANCE
    ):
        raise ValueError(
            "Frozen probability reproduction failed."
        )

    repeated_probabilities = (
        model.predict_proba(
            repeated_validation
        )[
            :,
            1
        ]
    )

    primary_target = (
        validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    repeated_target = (
        repeated_validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    patient_to_cluster = {
        patient: index
        for index, patient in enumerate(
            validation_patients
        )
    }

    repeated_cluster_index = (
        repeated_validation[
            "patient_nbr"
        ]
        .map(
            patient_to_cluster
        )
        .to_numpy(
            dtype=int
        )
    )

    if (
        np.any(
            repeated_cluster_index
            < 0
        )
    ):
        raise ValueError(
            "Unmapped repeated encounter."
        )

    primary_point = calculate_metrics(
        primary_target,
        primary_probabilities,
        threshold,
    )

    repeated_point = calculate_metrics(
        repeated_target,
        repeated_probabilities,
        threshold,
    )

    point_difference = {
        metric: float(
            repeated_point[
                metric
            ]
            - primary_point[
                metric
            ]
        )
        for metric in BOOTSTRAP_METRICS
    }

    rng = np.random.default_rng(
        DEFAULT_BOOTSTRAP_RANDOM_STATE
    )

    distributions = {
        metric: np.empty(
            DEFAULT_BOOTSTRAP_RESAMPLES,
            dtype=float,
        )
        for metric in BOOTSTRAP_METRICS
    }

    number_of_patients = len(
        validation_patients
    )

    primary_row_indices = np.arange(
        number_of_patients
    )

    repeated_row_indices = np.arange(
        len(
            repeated_validation
        )
    )

    print(
        "\nGenerating patient-cluster bootstrap..."
    )

    for iteration in range(
        DEFAULT_BOOTSTRAP_RESAMPLES
    ):
        sampled_clusters = rng.choice(
            number_of_patients,
            size=number_of_patients,
            replace=True,
        )

        cluster_counts = np.bincount(
            sampled_clusters,
            minlength=number_of_patients,
        )

        primary_bootstrap_indices = (
            np.repeat(
                primary_row_indices,
                cluster_counts,
            )
        )

        repeated_multiplicity = (
            cluster_counts[
                repeated_cluster_index
            ]
        )

        repeated_bootstrap_indices = (
            np.repeat(
                repeated_row_indices,
                repeated_multiplicity,
            )
        )

        primary_metrics = (
            calculate_metrics(
                primary_target[
                    primary_bootstrap_indices
                ],
                primary_probabilities[
                    primary_bootstrap_indices
                ],
                threshold,
            )
        )

        repeated_metrics = (
            calculate_metrics(
                repeated_target[
                    repeated_bootstrap_indices
                ],
                repeated_probabilities[
                    repeated_bootstrap_indices
                ],
                threshold,
            )
        )

        for metric in (
            BOOTSTRAP_METRICS
        ):
            distributions[
                metric
            ][
                iteration
            ] = (
                repeated_metrics[
                    metric
                ]
                - primary_metrics[
                    metric
                ]
            )

    results = {}
    table_rows = []

    print(
        "\nALL-ELIGIBLE MINUS PRIMARY"
    )

    print(
        "-" * 104
    )

    for metric in (
        BOOTSTRAP_METRICS
    ):
        summary = summarize_distribution(
            distributions[
                metric
            ]
        )

        result = {
            "estimate": (
                point_difference[
                    metric
                ]
            ),
            **summary,
        }

        results[
            metric
        ] = result

        table_rows.append(
            {
                "metric": (
                    metric
                ),
                **result,
                "ci_excludes_zero": bool(
                    result[
                        "ci_lower"
                    ]
                    > 0.0
                    or result[
                        "ci_upper"
                    ]
                    < 0.0
                ),
            }
        )

        print(
            f"{metric:<22}"
            f"{result['estimate']:+.6f} "
            f"[{result['ci_lower']:+.6f}, "
            f"{result['ci_upper']:+.6f}]"
        )

    output_table = pd.DataFrame(
        table_rows
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 10,
        "analysis": (
            "patient_cluster_bootstrap_"
            "repeated_encounter_robustness"
        ),
        "comparison": {
            "reference": (
                "primary_validation"
            ),
            "robustness": (
                "all_eligible_encounters_"
                "for_validation_patients"
            ),
            "difference_definition": (
                "robustness_minus_reference"
            ),
        },
        "bootstrap_protocol": {
            "unit": "patient",
            "method": (
                "paired_patient_cluster_"
                "bootstrap_with_replacement"
            ),
            "n_resamples": (
                DEFAULT_BOOTSTRAP_RESAMPLES
            ),
            "random_state": (
                DEFAULT_BOOTSTRAP_RANDOM_STATE
            ),
            "confidence_level": (
                DEFAULT_CONFIDENCE_LEVEL
            ),
            "interval_method": (
                "percentile"
            ),
            "all_encounters_for_sampled_"
            "patient_replicated_together": (
                True
            ),
        },
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
            "model_reselected": False,
            "calibration_reselected": False,
            "threshold_reselected": False,
        },
        "sample_counts": {
            "validation_patients": int(
                number_of_patients
            ),
            "primary_encounters": int(
                len(
                    validation
                )
            ),
            "all_eligible_encounters": int(
                len(
                    repeated_validation
                )
            ),
        },
        "point_metrics": {
            "primary": (
                primary_point
            ),
            "all_eligible": (
                repeated_point
            ),
        },
        "differences": (
            results
        ),
        "reproduction_audit": {
            "maximum_validation_probability_error": (
                maximum_error
            ),
            "tolerance": (
                PROBABILITY_REPRODUCTION_TOLERANCE
            ),
            "passed": (
                maximum_error
                <= PROBABILITY_REPRODUCTION_TOLERANCE
            ),
        },
        "prediction_artifact": {
            "path": str(
                PHASE7_PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": (
                prediction_hash
            ),
        },
        "privacy": {
            "patient_identifiers_saved": False,
            "row_level_predictions_saved": False,
        },
        "data_policy": {
            "fit_split": "train",
            "evaluation_patient_partition": (
                "validation"
            ),
            "new_split_created": False,
            "test_used": False,
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
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
        "\nBootstrap unit      : patient"
    )

    print(
        "Threshold reselected: False"
    )

    print(
        "Test used           : False"
    )


if __name__ == "__main__":
    main()