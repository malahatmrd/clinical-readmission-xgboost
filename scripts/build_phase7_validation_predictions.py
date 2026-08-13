from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from clinical_readmission.evaluation.metrics import (
    assert_metric_reproduction,
    calculate_metric_deltas,
    calculate_probability_metrics,
)
from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)
from clinical_readmission.models.logistic_baseline import (
    build_logistic_baseline,
)
from clinical_readmission.models.xgboost_early_stopping import (
    build_refit_xgboost,
)
from clinical_readmission.models.xgboost_tuned import (
    build_tuned_refit_xgboost,
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

LOGISTIC_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "logistic_baseline_validation.json"
)

EARLY_STOPPING_METRICS_PATH = (
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

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_validation_probabilities.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_prediction_reproduction.json"
)

REPRODUCTION_TOLERANCE = 1e-7

def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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


def main() -> None:
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH,
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

    target_column = "readmitted_30d"

    y_train = train[
        target_column
    ]

    y_validation = validation[
        target_column
    ]

    logistic_artifact = load_json(
        LOGISTIC_METRICS_PATH
    )

    early_artifact = load_json(
        EARLY_STOPPING_METRICS_PATH
    )

    tuned_artifact = load_json(
        TUNED_METRICS_PATH
    )

    print("=" * 88)
    print("PHASE 7 VALIDATION PREDICTION REPRODUCTION")
    print("=" * 88)

    print(
        "\nTrain rows      :",
        len(train),
    )

    print(
        "Validation rows :",
        len(validation),
    )

    print(
        "Validation positives:",
        int(
            y_validation.sum()
        ),
    )

    print(
        "\nRebuilding Logistic Regression..."
    )

    logistic_model = (
        build_logistic_baseline()
    )

    logistic_model.fit(
        train,
        y_train,
    )

    logistic_probabilities = (
        logistic_model.predict_proba(
            validation
        )[:, 1]
    )

    logistic_metrics = (
        calculate_probability_metrics(
            y_validation,
            logistic_probabilities,
        )
    )

    assert_metric_reproduction(
        logistic_metrics,
        logistic_artifact["metrics"],
        tolerance=REPRODUCTION_TOLERANCE,
    )

    print(
        "Logistic reproduction : PASS"
    )

    print(
        "\nFitting shared full-Train "
        "preprocessor for XGBoost finalists..."
    )

    preprocessor = (
        build_preprocessor()
    )

    x_train = (
        preprocessor.fit_transform(
            train
        )
    )

    x_validation = (
        preprocessor.transform(
            validation
        )
    )

    feature_count = len(
        preprocessor
        .get_feature_names_out()
    )

    print(
        "Transformed features:",
        feature_count,
    )

    early_tree_count = int(
        early_artifact[
            "development_protocol"
        ][
            "selected_tree_count"
        ]
    )

    print(
        "\nRebuilding Early-Stopped XGBoost..."
    )

    print(
        "Tree count:",
        early_tree_count,
    )

    early_model = (
        build_refit_xgboost(
            n_estimators=(
                early_tree_count
            )
        )
    )

    early_model.fit(
        x_train,
        y_train,
    )

    early_probabilities = (
        early_model.predict_proba(
            x_validation
        )[:, 1]
    )

    early_metrics = (
        calculate_probability_metrics(
            y_validation,
            early_probabilities,
        )
    )

    assert_metric_reproduction(
        early_metrics,
        early_artifact["metrics"],
        tolerance=REPRODUCTION_TOLERANCE,
    )

    print(
        "Early-stop reproduction: PASS"
    )

    tuned_parameters = (
        tuned_artifact[
            "selected_hyperparameters"
        ]
    )

    tuned_tree_count = int(
        tuned_artifact[
            "development_protocol"
        ][
            "selected_tree_count"
        ]
    )

    print(
        "\nRebuilding Tuned XGBoost..."
    )

    print(
        "Tree count:",
        tuned_tree_count,
    )

    tuned_model = (
        build_tuned_refit_xgboost(
            tuned_parameters,
            n_estimators=(
                tuned_tree_count
            ),
        )
    )

    tuned_model.fit(
        x_train,
        y_train,
    )

    tuned_probabilities = (
        tuned_model.predict_proba(
            x_validation
        )[:, 1]
    )

    tuned_metrics = (
        calculate_probability_metrics(
            y_validation,
            tuned_probabilities,
        )
    )

    assert_metric_reproduction(
        tuned_metrics,
        tuned_artifact["metrics"],
        tolerance=REPRODUCTION_TOLERANCE,
    )

    print(
        "Tuned reproduction     : PASS"
    )

    predictions = pd.DataFrame(
        {
            "validation_row": range(
                len(validation)
            ),
            "readmitted_30d": (
                y_validation
                .to_numpy()
            ),
            "logistic_probability": (
                logistic_probabilities
            ),
            "early_stopped_xgboost_probability": (
                early_probabilities
            ),
            "tuned_xgboost_probability": (
                tuned_probabilities
            ),
        }
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    prediction_hash = file_sha256(
        PREDICTIONS_PATH
    )

    reproduced_metrics = {
        "logistic_regression": (
            logistic_metrics
        ),
        "early_stopped_xgboost": (
            early_metrics
        ),
        "tuned_xgboost": (
            tuned_metrics
        ),
    }

    reproduction_deltas = {
        "logistic_regression": (
            calculate_metric_deltas(
                logistic_metrics,
                logistic_artifact[
                    "metrics"
                ],
            )
        ),
        "early_stopped_xgboost": (
            calculate_metric_deltas(
                early_metrics,
                early_artifact[
                    "metrics"
                ],
            )
        ),
        "tuned_xgboost": (
            calculate_metric_deltas(
                tuned_metrics,
                tuned_artifact[
                    "metrics"
                ],
            )
        ),
    }

    summary = {
        "phase": 7,
        "experiment": (
            "validation_prediction_reproduction"
        ),
        "data_policy": {
            "fit_split": "train",
            "prediction_split": (
                "validation"
            ),
            "test_used": False,
            "identifiers_saved": False,
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
            "logistic_regression": {
                "source": str(
                    LOGISTIC_METRICS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
            },
            "early_stopped_xgboost": {
                "source": str(
                    EARLY_STOPPING_METRICS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "selected_tree_count": (
                    early_tree_count
                ),
            },
            "tuned_xgboost": {
                "source": str(
                    TUNED_METRICS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "selected_tree_count": (
                    tuned_tree_count
                ),
                "selected_hyperparameters": (
                    tuned_parameters
                ),
            },
        },
        "transformed_features": int(
            feature_count
        ),
        "reproduction_tolerance": (
            REPRODUCTION_TOLERANCE
        ),
        "reproduced_metrics": (
            reproduced_metrics
        ),
        "metric_deltas_vs_recorded": (
            reproduction_deltas
        ),
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": prediction_hash,
            "rows": int(
                len(predictions)
            ),
        },
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nREPRODUCED VALIDATION METRICS"
    )
    print("-" * 88)

    for (
        model_name,
        metrics,
    ) in reproduced_metrics.items():
        print(
            f"\n{model_name}"
        )

        for (
            metric_name,
            value,
        ) in metrics.items():
            print(
                f"  {metric_name:20}: "
                f"{value:.9f}"
            )

    print(
        "\nPrediction SHA256:",
        prediction_hash,
    )

    print(
        "\nSaved predictions:",
        PREDICTIONS_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved summary    :",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used        : False"
    )


if __name__ == "__main__":
    main()
