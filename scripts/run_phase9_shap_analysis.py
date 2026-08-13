from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from clinical_readmission.evaluation.shap_utils import (
    build_transformed_feature_metadata,
    calculate_source_shap_importance,
    calculate_transformed_shap_importance,
)
from clinical_readmission.features.preprocessing import (
    build_preprocessor,
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

TUNED_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_tuned_validation.json"
)

PHASE7_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)

PHASE7_CANDIDATE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

PHASE8_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_threshold_selection.json"
)

FEATURE_METADATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_transformed_feature_metadata.csv"
)

TRANSFORMED_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_transformed_shap_importance.csv"
)

SOURCE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_source_shap_importance.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_shap_validation.json"
)

TARGET_COLUMN = "readmitted_30d"

BASE_PROBABILITY_COLUMN = (
    "tuned_xgboost_probability"
)

EXPECTED_SELECTED_MODEL = (
    "tuned_xgboost_sigmoid"
)

EXPECTED_REFERENCE_THRESHOLD = 0.105

PROBABILITY_REPRODUCTION_TOLERANCE = 1e-7

ADDITIVITY_TOLERANCE = 1e-4

TOP_TRANSFORMED_FEATURES = 20
TOP_SOURCE_FEATURES = 20


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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

    target_match = (
        result[
            TARGET_COLUMN
        ]
        == result[
            f"{TARGET_COLUMN}_assignment"
        ]
    ).all()

    if not target_match:
        raise ValueError(
            f"{split_name}: target mismatch "
            "between cohort and assignments."
        )

    return result


def main() -> None:
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    tuned_artifact = load_json(
        TUNED_METRICS_PATH
    )

    phase7_summary = load_json(
        PHASE7_CANDIDATE_SUMMARY_PATH
    )

    phase8_selection = load_json(
        PHASE8_SELECTION_PATH
    )

    if (
        phase8_selection[
            "selected_model"
        ]
        != EXPECTED_SELECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    if not (
        phase8_selection[
            "data_policy"
        ][
            "model_frozen"
        ]
    ):
        raise ValueError(
            "Model must be frozen before SHAP."
        )

    if not (
        phase8_selection[
            "data_policy"
        ][
            "calibration_frozen"
        ]
    ):
        raise ValueError(
            "Calibration must be frozen before SHAP."
        )

    if not (
        phase8_selection[
            "data_policy"
        ][
            "threshold_frozen"
        ]
    ):
        raise ValueError(
            "Threshold must be frozen before SHAP."
        )

    if (
        phase8_selection[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
        )

    reference_threshold = float(
        phase8_selection[
            "reference_threshold"
        ]
    )

    if abs(
        reference_threshold
        - EXPECTED_REFERENCE_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected frozen reference threshold."
        )

    observed_prediction_hash = (
        file_sha256(
            PHASE7_PREDICTIONS_PATH
        )
    )

    expected_prediction_hash = (
        phase7_summary[
            "candidate_prediction_artifact"
        ][
            "sha256"
        ]
    )

    if (
        observed_prediction_hash
        != expected_prediction_hash
    ):
        raise ValueError(
            "Phase 7 prediction artifact "
            "SHA256 mismatch."
        )

    predictions = pd.read_csv(
        PHASE7_PREDICTIONS_PATH
    )

    forbidden_identifiers = {
        "encounter_id",
        "patient_nbr",
        "source_row",
    }

    present_identifiers = sorted(
        forbidden_identifiers
        & set(
            predictions.columns
        )
    )

    if present_identifiers:
        raise ValueError(
            "Prediction artifact contains "
            "forbidden identifiers: "
            f"{present_identifiers}"
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
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        y_validation,
        predictions[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        ),
    ):
        raise ValueError(
            "Validation target order does not "
            "match Phase 7 predictions."
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

    print("=" * 96)
    print("PHASE 9 SHAP VALIDATION ANALYSIS")
    print("=" * 96)

    print(
        "\nTrain rows          :",
        len(train),
    )

    print(
        "Validation rows     :",
        len(validation),
    )

    print(
        "Validation positives:",
        int(
            y_validation.sum()
        ),
    )

    print(
        "Frozen model        :",
        EXPECTED_SELECTED_MODEL,
    )

    print(
        "Reference threshold :",
        f"{reference_threshold:.3f}",
    )

    print(
        "XGBoost tree count  :",
        tree_count,
    )

    print(
        "\nFitting frozen full-Train preprocessor..."
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

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        "Transformed features:",
        len(
            feature_names
        ),
    )

    if (
        x_train.shape[
            1
        ]
        != len(
            feature_names
        )
    ):
        raise ValueError(
            "Train transformed width does not "
            "match feature-name count."
        )

    if (
        x_validation.shape[
            1
        ]
        != len(
            feature_names
        )
    ):
        raise ValueError(
            "Validation transformed width does "
            "not match feature-name count."
        )

    print(
        "\nFitting frozen Tuned XGBoost..."
    )

    model = (
        build_tuned_refit_xgboost(
            tuned_parameters,
            n_estimators=(
                tree_count
            ),
        )
    )

    model.fit(
        x_train,
        y_train,
    )

    reproduced_probabilities = (
        model.predict_proba(
            x_validation
        )[
            :,
            1
        ]
    )

    recorded_probabilities = (
        predictions[
            BASE_PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    probability_differences = np.abs(
        reproduced_probabilities
        - recorded_probabilities
    )

    maximum_probability_difference = (
        float(
            probability_differences.max()
        )
    )

    print(
        "Maximum probability reproduction error:",
        f"{maximum_probability_difference:.12g}",
    )

    if (
        maximum_probability_difference
        > PROBABILITY_REPRODUCTION_TOLERANCE
    ):
        raise ValueError(
            "Frozen XGBoost probabilities did "
            "not reproduce Phase 7 predictions."
        )

    print(
        "Probability reproduction: PASS"
    )

    print(
        "\nBuilding TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation=(
            "tree_path_dependent"
        ),
        model_output="raw",
        feature_names=(
            feature_names
        ),
    )

    print(
        "Calculating Validation SHAP values..."
    )

    shap_values = (
        explainer.shap_values(
            x_validation,
            check_additivity=True,
        )
    )

    shap_values = np.asarray(
        shap_values,
        dtype=float,
    )

    if shap_values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional SHAP "
            "values for binary XGBoost."
        )

    if shap_values.shape != (
        len(validation),
        len(feature_names),
    ):
        raise ValueError(
            "Unexpected SHAP matrix shape."
        )

    expected_value_array = np.asarray(
        explainer.expected_value,
        dtype=float,
    ).reshape(
        -1
    )

    if (
        expected_value_array.size
        != 1
    ):
        raise ValueError(
            "Expected scalar SHAP base value."
        )

    expected_value = float(
        expected_value_array[
            0
        ]
    )

    raw_margin = model.predict(
        x_validation,
        output_margin=True,
    )

    raw_margin = np.asarray(
        raw_margin,
        dtype=float,
    )

    reconstructed_margin = (
        expected_value
        + shap_values.sum(
            axis=1
        )
    )

    additivity_error = np.abs(
        reconstructed_margin
        - raw_margin
    )

    maximum_additivity_error = float(
        additivity_error.max()
    )

    mean_additivity_error = float(
        additivity_error.mean()
    )

    print(
        "SHAP expected value:",
        f"{expected_value:.9f}",
    )

    print(
        "Maximum additivity error:",
        f"{maximum_additivity_error:.12g}",
    )

    print(
        "Mean additivity error   :",
        f"{mean_additivity_error:.12g}",
    )

    if (
        maximum_additivity_error
        > ADDITIVITY_TOLERANCE
    ):
        raise ValueError(
            "SHAP additivity error exceeds "
            "the permitted tolerance."
        )

    probabilities_from_margin = (
        expit(
            raw_margin
        )
    )

    margin_probability_error = (
        np.abs(
            probabilities_from_margin
            - reproduced_probabilities
        )
    )

    maximum_margin_probability_error = (
        float(
            margin_probability_error.max()
        )
    )

    print(
        "Maximum expit(margin) probability error:",
        f"{maximum_margin_probability_error:.12g}",
    )

    if (
        maximum_margin_probability_error
        > PROBABILITY_REPRODUCTION_TOLERANCE
    ):
        raise ValueError(
            "Raw-margin inverse logistic "
            "transform did not reproduce "
            "XGBoost probabilities."
        )

    print(
        "\nBuilding transformed-feature metadata..."
    )

    feature_metadata = (
        build_transformed_feature_metadata(
            feature_names
        )
    )

    print(
        "Building transformed-feature "
        "SHAP importance..."
    )

    transformed_importance = (
        calculate_transformed_shap_importance(
            shap_values,
            feature_names,
        )
    )

    transformed_importance = (
        transformed_importance.merge(
            feature_metadata[
                [
                    "transformed_feature",
                    "transformer",
                    "source_feature",
                ]
            ],
            on="transformed_feature",
            how="left",
            validate="one_to_one",
        )
    )

    print(
        "Aggregating SHAP values to "
        "clinical source features..."
    )

    source_importance = (
        calculate_source_shap_importance(
            shap_values,
            feature_metadata,
        )
    )

    FEATURE_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_metadata.to_csv(
        FEATURE_METADATA_PATH,
        index=False,
    )

    transformed_importance.to_csv(
        TRANSFORMED_IMPORTANCE_PATH,
        index=False,
    )

    source_importance.to_csv(
        SOURCE_IMPORTANCE_PATH,
        index=False,
    )

    top_transformed = (
        transformed_importance.head(
            TOP_TRANSFORMED_FEATURES
        )
    )

    top_source = (
        source_importance.head(
            TOP_SOURCE_FEATURES
        )
    )

    summary = {
        "phase": 9,
        "analysis": (
            "validation_tree_shap_global_analysis"
        ),
        "explanation_target": {
            "model_family": (
                "tuned_xgboost"
            ),
            "final_development_variant": (
                EXPECTED_SELECTED_MODEL
            ),
            "explained_output": (
                "uncalibrated_xgboost_raw_margin"
            ),
            "raw_output_interpretation": (
                "binary_xgboost_log_odds_margin"
            ),
            "calibrated_probability_directly_explained": (
                False
            ),
            "reason": (
                "sigmoid calibration is retained "
                "for final probabilities and "
                "thresholding, while Tree SHAP "
                "decomposes the frozen base "
                "XGBoost raw margin"
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
            "tree_count": (
                tree_count
            ),
            "selected_hyperparameters": (
                tuned_parameters
            ),
            "reference_threshold": (
                reference_threshold
            ),
        },
        "transformed_feature_count": int(
            len(
                feature_names
            )
        ),
        "shap_matrix_shape": [
            int(
                shap_values.shape[
                    0
                ]
            ),
            int(
                shap_values.shape[
                    1
                ]
            ),
        ],
        "tree_explainer": {
            "model_output": (
                "raw"
            ),
            "feature_perturbation": (
                "tree_path_dependent"
            ),
            "check_additivity": True,
            "expected_value": (
                expected_value
            ),
        },
        "reproduction_audit": {
            "maximum_probability_difference": (
                maximum_probability_difference
            ),
            "probability_tolerance": (
                PROBABILITY_REPRODUCTION_TOLERANCE
            ),
            "maximum_additivity_error": (
                maximum_additivity_error
            ),
            "mean_additivity_error": (
                mean_additivity_error
            ),
            "additivity_tolerance": (
                ADDITIVITY_TOLERANCE
            ),
            "maximum_expit_margin_probability_error": (
                maximum_margin_probability_error
            ),
        },
        "top_transformed_features": (
            top_transformed.to_dict(
                orient="records"
            )
        ),
        "top_source_features": (
            top_source.to_dict(
                orient="records"
            )
        ),
        "source_artifacts": {
            "phase7_predictions": {
                "path": str(
                    PHASE7_PREDICTIONS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": (
                    observed_prediction_hash
                ),
            },
            "phase8_selection": str(
                PHASE8_SELECTION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "output_artifacts": {
            "feature_metadata": str(
                FEATURE_METADATA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "transformed_importance": str(
                TRANSFORMED_IMPORTANCE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "source_importance": str(
                SOURCE_IMPORTANCE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "data_policy": {
            "fit_split": "train",
            "explanation_split": (
                "validation"
            ),
            "identifiers_saved": False,
            "test_used": False,
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
        "\nTOP TRANSFORMED FEATURES"
    )

    print("-" * 96)

    print(
        top_transformed[
            [
                "rank",
                "transformed_feature",
                "source_feature",
                "mean_abs_shap",
                "mean_signed_shap",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nTOP SOURCE CLINICAL FEATURES"
    )

    print("-" * 96)

    print(
        top_source[
            [
                "rank",
                "source_feature",
                "transformed_feature_count",
                "mean_abs_shap",
                "mean_signed_shap",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nSaved metadata    :",
        FEATURE_METADATA_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved transformed :",
        TRANSFORMED_IMPORTANCE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved source      :",
        SOURCE_IMPORTANCE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved summary     :",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nSHAP output space : raw XGBoost margin/log-odds"
    )

    print(
        "Calibration       : frozen sigmoid; "
        "not directly decomposed"
    )

    print(
        "Test used         : False"
    )


if __name__ == "__main__":
    main()