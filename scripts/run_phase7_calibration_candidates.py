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
    CALIBRATION_CV_RANDOM_STATE,
    CALIBRATION_CV_SPLITS,
    SUPPORTED_CALIBRATION_METHODS,
    build_calibrated_classifier,
    build_early_stopped_pipeline,
    build_tuned_pipeline,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
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

BASE_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_validation_probabilities.csv"
)

REPRODUCTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_prediction_reproduction.json"
)

EARLY_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_early_stopping_validation.json"
)

TUNED_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_tuned_validation.json"
)

OUTPUT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_calibration_candidates.csv"
)

TARGET_COLUMN = "readmitted_30d"

PRIMARY_CALIBRATION_METRIC = "brier_score"
SECONDARY_CALIBRATION_METRIC = "log_loss"


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_partition(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    ids = assignments.loc[
        assignments["split"].eq(split_name),
        [
            "encounter_id",
            "patient_nbr",
            "readmitted_30d",
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

    target_match = (
        result["readmitted_30d"]
        == result["readmitted_30d_assignment"]
    ).all()

    if not target_match:
        raise ValueError(
            f"{split_name}: target mismatch "
            "between cohort and assignments."
        )

    return result


def evaluate_probabilities(
    target,
    probabilities,
) -> dict[str, float]:
    metrics = calculate_probability_metrics(
        target,
        probabilities,
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
        np.mean(target)
    )

    mean_probability = float(
        np.mean(probabilities)
    )

    return {
        **metrics,
        "observed_prevalence": prevalence,
        "mean_predicted_probability": (
            mean_probability
        ),
        "mean_probability_minus_prevalence": (
            mean_probability
            - prevalence
        ),
        "calibration_intercept": float(
            intercept
        ),
        "calibration_slope": float(
            slope
        ),
        "quantile_ece": float(
            ece
        ),
    }


def main() -> None:
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    train = load_partition(
        cohort,
        assignments,
        "train",
    )

    validation = load_partition(
        cohort,
        assignments,
        "validation",
    )

    y_train = train[
        TARGET_COLUMN
    ]

    y_validation = validation[
        TARGET_COLUMN
    ].to_numpy()

    base_predictions = pd.read_csv(
        BASE_PREDICTIONS_PATH
    )

    reproduction = load_json(
        REPRODUCTION_PATH
    )

    expected_hash = (
        reproduction[
            "prediction_artifact"
        ]["sha256"]
    )

    observed_hash = file_sha256(
        BASE_PREDICTIONS_PATH
    )

    if observed_hash != expected_hash:
        raise ValueError(
            "Base prediction artifact SHA256 "
            "does not match reproduction summary."
        )

    if not np.array_equal(
        y_validation,
        base_predictions[
            TARGET_COLUMN
        ].to_numpy(),
    ):
        raise ValueError(
            "Validation target order does not "
            "match the prediction artifact."
        )

    forbidden_identifiers = {
        "encounter_id",
        "patient_nbr",
    }

    if forbidden_identifiers & set(
        base_predictions.columns
    ):
        raise ValueError(
            "Base prediction artifact contains "
            "forbidden identifiers."
        )

    early_artifact = load_json(
        EARLY_METRICS_PATH
    )

    tuned_artifact = load_json(
        TUNED_METRICS_PATH
    )

    early_tree_count = int(
        early_artifact[
            "development_protocol"
        ]["selected_tree_count"]
    )

    tuned_tree_count = int(
        tuned_artifact[
            "development_protocol"
        ]["selected_tree_count"]
    )

    tuned_parameters = (
        tuned_artifact[
            "selected_hyperparameters"
        ]
    )

    print("=" * 88)
    print("PHASE 7 TRAIN-ONLY CALIBRATION CANDIDATES")
    print("=" * 88)

    print(
        "\nTrain rows            :",
        len(train),
    )

    print(
        "Validation rows       :",
        len(validation),
    )

    print(
        "Validation positives  :",
        int(y_validation.sum()),
    )

    print(
        "Calibration CV folds  :",
        CALIBRATION_CV_SPLITS,
    )

    print(
        "Calibration CV seed   :",
        CALIBRATION_CV_RANDOM_STATE,
    )

    print(
        "Calibration methods   :",
        ", ".join(
            SUPPORTED_CALIBRATION_METHODS
        ),
    )

    print(
        "Primary selection metric  :",
        PRIMARY_CALIBRATION_METRIC,
    )

    print(
        "Secondary selection metric:",
        SECONDARY_CALIBRATION_METRIC,
    )

    print(
        "Phase-6 base family       : tuned_xgboost"
    )

    print(
        "Test used                 : False"
    )

    candidate_predictions = (
        base_predictions.copy()
    )

    candidate_metadata = {
        "logistic_regression": {
            "model_family": (
                "logistic_regression"
            ),
            "calibration_method": (
                "none"
            ),
            "probability_column": (
                "logistic_probability"
            ),
            "selection_eligible": False,
            "role": "reference",
        },
        "early_stopped_xgboost": {
            "model_family": (
                "early_stopped_xgboost"
            ),
            "calibration_method": (
                "none"
            ),
            "probability_column": (
                "early_stopped_xgboost_probability"
            ),
            "selection_eligible": False,
            "role": "reference",
        },
        "tuned_xgboost": {
            "model_family": (
                "tuned_xgboost"
            ),
            "calibration_method": (
                "none"
            ),
            "probability_column": (
                "tuned_xgboost_probability"
            ),
            "selection_eligible": True,
            "role": (
                "phase6_champion"
            ),
        },
    }

    calibration_jobs = (
        (
            "early_stopped_xgboost",
            "sigmoid",
        ),
        (
            "early_stopped_xgboost",
            "isotonic",
        ),
        (
            "tuned_xgboost",
            "sigmoid",
        ),
        (
            "tuned_xgboost",
            "isotonic",
        ),
    )

    print(
        "\nFitting Train-only calibration "
        "candidates..."
    )

    for (
        model_family,
        method,
    ) in calibration_jobs:
        candidate_name = (
            f"{model_family}_{method}"
        )

        print(
            f"\n{candidate_name}"
        )

        if model_family == (
            "early_stopped_xgboost"
        ):
            estimator = (
                build_early_stopped_pipeline(
                    n_estimators=(
                        early_tree_count
                    )
                )
            )
        elif model_family == (
            "tuned_xgboost"
        ):
            estimator = (
                build_tuned_pipeline(
                    tuned_parameters,
                    n_estimators=(
                        tuned_tree_count
                    ),
                )
            )
        else:
            raise ValueError(
                "Unexpected model family."
            )

        calibrated = (
            build_calibrated_classifier(
                estimator,
                method,
            )
        )

        print(
            "  fitting Train-only CV "
            "calibrator..."
        )

        calibrated.fit(
            train,
            y_train,
        )

        print(
            "  generating Validation "
            "probabilities..."
        )

        probabilities = (
            calibrated.predict_proba(
                validation
            )[:, 1]
        )

        probability_column = (
            f"{candidate_name}_probability"
        )

        candidate_predictions[
            probability_column
        ] = probabilities

        candidate_metadata[
            candidate_name
        ] = {
            "model_family": (
                model_family
            ),
            "calibration_method": (
                method
            ),
            "probability_column": (
                probability_column
            ),
            "selection_eligible": bool(
                model_family
                == "tuned_xgboost"
            ),
            "role": (
                "calibration_candidate"
            ),
        }

        print(
            "  complete"
        )

    results = {}
    table_rows = []

    print(
        "\nVALIDATION CALIBRATION RESULTS"
    )
    print("-" * 88)

    for (
        candidate_name,
        metadata,
    ) in candidate_metadata.items():
        probabilities = (
            candidate_predictions[
                metadata[
                    "probability_column"
                ]
            ].to_numpy()
        )

        result = (
            evaluate_probabilities(
                y_validation,
                probabilities,
            )
        )

        results[
            candidate_name
        ] = {
            **metadata,
            **result,
        }

        table_rows.append(
            {
                "candidate": (
                    candidate_name
                ),
                **metadata,
                **result,
            }
        )

        print(
            f"\n{candidate_name}"
        )

        print(
            "  ROC-AUC                    : "
            f"{result['roc_auc']:.6f}"
        )

        print(
            "  Average Precision          : "
            f"{result['average_precision']:.6f}"
        )

        print(
            "  Brier Score                : "
            f"{result['brier_score']:.6f}"
        )

        print(
            "  Log Loss                   : "
            f"{result['log_loss']:.6f}"
        )

        print(
            "  Mean predicted probability : "
            f"{result['mean_predicted_probability']:.6f}"
        )

        print(
            "  Mean probability - prevalence: "
            f"{result['mean_probability_minus_prevalence']:+.6f}"
        )

        print(
            "  Calibration intercept      : "
            f"{result['calibration_intercept']:+.6f}"
        )

        print(
            "  Calibration slope          : "
            f"{result['calibration_slope']:.6f}"
        )

        print(
            "  Quantile ECE               : "
            f"{result['quantile_ece']:.6f}"
        )

    result_table = pd.DataFrame(
        table_rows
    )

    result_table[
        "abs_calibration_intercept"
    ] = (
        result_table[
            "calibration_intercept"
        ].abs()
    )

    result_table[
        "abs_slope_minus_one"
    ] = (
        result_table[
            "calibration_slope"
        ]
        - 1.0
    ).abs()

    result_table = (
        result_table.sort_values(
            by=[
                "selection_eligible",
                PRIMARY_CALIBRATION_METRIC,
                SECONDARY_CALIBRATION_METRIC,
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    OUTPUT_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_predictions.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    output_hash = file_sha256(
        OUTPUT_PREDICTIONS_PATH
    )

    summary = {
        "phase": 7,
        "analysis": (
            "train_only_posthoc_calibration_candidates"
        ),
        "selection_policy": {
            "base_model_family": (
                "tuned_xgboost"
            ),
            "base_family_source": (
                "phase6_prespecified_average_precision_selection"
            ),
            "eligible_variants": [
                "tuned_xgboost",
                "tuned_xgboost_sigmoid",
                "tuned_xgboost_isotonic",
            ],
            "primary_calibration_metric": (
                PRIMARY_CALIBRATION_METRIC
            ),
            "primary_metric_direction": (
                "lower_is_better"
            ),
            "secondary_calibration_metric": (
                SECONDARY_CALIBRATION_METRIC
            ),
            "secondary_metric_direction": (
                "lower_is_better"
            ),
            "diagnostic_metrics": [
                "calibration_intercept",
                "calibration_slope",
                "quantile_ece",
            ],
            "discrimination_safeguards": [
                "roc_auc",
                "average_precision",
            ],
            "selection_split": (
                "validation"
            ),
            "test_used": False,
        },
        "calibration_protocol": {
            "fit_data": "train_only",
            "cv_strategy": (
                "StratifiedKFold"
            ),
            "cv_splits": (
                CALIBRATION_CV_SPLITS
            ),
            "cv_random_state": (
                CALIBRATION_CV_RANDOM_STATE
            ),
            "ensemble": False,
            "methods": list(
                SUPPORTED_CALIBRATION_METHODS
            ),
        },
        "sample_counts": {
            "train": int(
                len(train)
            ),
            "validation": int(
                len(validation)
            ),
            "validation_positive": int(
                y_validation.sum()
            ),
        },
        "model_configuration": {
            "early_stopped_tree_count": (
                early_tree_count
            ),
            "tuned_tree_count": (
                tuned_tree_count
            ),
            "tuned_parameters": (
                tuned_parameters
            ),
        },
        "base_prediction_artifact": {
            "path": str(
                BASE_PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": observed_hash,
        },
        "candidate_prediction_artifact": {
            "path": str(
                OUTPUT_PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": output_hash,
        },
        "candidates": results,
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
        "\nCandidate prediction SHA256:",
        output_hash,
    )

    print(
        "\nSaved candidate table      :",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved candidate predictions:",
        OUTPUT_PREDICTIONS_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved candidate summary    :",
        OUTPUT_SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used                  : False"
    )


if __name__ == "__main__":
    main()