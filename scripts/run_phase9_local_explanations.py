from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from clinical_readmission.evaluation.shap_utils import (
    aggregate_local_shap_to_source,
    build_transformed_feature_metadata,
    select_local_explanation_cases,
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

CASE_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_local_explanation_cases.csv"
)

SOURCE_CONTRIBUTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_local_source_contributions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_local_explanations.json"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

TARGET_COLUMN = "readmitted_30d"

CALIBRATED_PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

BASE_PROBABILITY_COLUMN = (
    "tuned_xgboost_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_THRESHOLD = 0.105

PROBABILITY_TOLERANCE = 1e-7
ADDITIVITY_TOLERANCE = 1e-4

WATERFALL_MAX_DISPLAY = 15
FIGURE_DPI = 300


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

    if not (
        result[
            TARGET_COLUMN
        ]
        == result[
            f"{TARGET_COLUMN}_assignment"
        ]
    ).all():
        raise ValueError(
            f"{split_name}: target mismatch."
        )

    return result


def confusion_label(
    y_true: int,
    predicted_positive: bool,
) -> str:
    if y_true == 1:
        return (
            "TP"
            if predicted_positive
            else "FN"
        )

    return (
        "FP"
        if predicted_positive
        else "TN"
    )


def format_source_value(
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "Missing"

    return str(
        value
    )


def save_waterfall(
    *,
    case_name: str,
    explanation: shap.Explanation,
    y_true: int,
    calibrated_probability: float,
    threshold: float,
    label: str,
) -> dict[str, str]:
    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            10.5,
            7.5,
        )
    )

    axis = shap.plots.waterfall(
        explanation,
        max_display=(
            WATERFALL_MAX_DISPLAY
        ),
        show=False,
    )

    axis.set_title(
        (
            f"{case_name}\n"
            f"{label} | y={y_true} | "
            f"calibrated p="
            f"{calibrated_probability:.4f} | "
            f"threshold={threshold:.3f}\n"
            "SHAP decomposition of raw "
            "XGBoost margin"
        ),
        pad=16,
    )

    figure = axis.get_figure()

    png_path = (
        FIGURE_ROOT
        / (
            "phase9_validation_shap_waterfall_"
            f"{case_name}.png"
        )
    )

    svg_path = (
        FIGURE_ROOT
        / (
            "phase9_validation_shap_waterfall_"
            f"{case_name}.svg"
        )
    )

    figure.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return {
        "png": str(
            png_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "svg": str(
            svg_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }


def main() -> None:
    phase8_selection = load_json(
        PHASE8_SELECTION_PATH
    )

    shap_summary = load_json(
        PHASE9_SHAP_PATH
    )

    tuned_artifact = load_json(
        TUNED_METRICS_PATH
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
            "Test data must remain locked."
        )

    if (
        shap_summary[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Phase 9 SHAP summary reports "
            "test usage."
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

    prediction_hash = file_sha256(
        PHASE7_PREDICTIONS_PATH
    )

    expected_prediction_hash = (
        shap_summary[
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

    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    predictions = pd.read_csv(
        PHASE7_PREDICTIONS_PATH
    )

    forbidden_identifiers = {
        "encounter_id",
        "patient_nbr",
        "source_row",
    }

    if (
        forbidden_identifiers
        & set(
            predictions.columns
        )
    ):
        raise ValueError(
            "Prediction artifact contains "
            "forbidden identifiers."
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

    calibrated_probabilities = (
        predictions[
            CALIBRATED_PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    print("=" * 96)
    print("PHASE 9 LOCAL SHAP EXPLANATIONS")
    print("=" * 96)

    print(
        "\nFrozen model       :",
        EXPECTED_MODEL,
    )

    print(
        "Reference threshold:",
        f"{threshold:.3f}",
    )

    print(
        "Validation rows    :",
        len(validation),
    )

    print(
        "Test used          : False"
    )

    print(
        "\nSelecting predefined "
        "local explanation cases..."
    )

    cases = select_local_explanation_cases(
        y_validation,
        calibrated_probabilities,
        threshold,
    )

    print(
        cases[
            [
                "case_name",
                "validation_row",
                "y_true",
                "calibrated_probability",
                "predicted_positive",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nFitting frozen full-Train "
        "preprocessor..."
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

    feature_metadata = (
        build_transformed_feature_metadata(
            feature_names
        )
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
        "Fitting frozen Tuned XGBoost..."
    )

    model = build_tuned_refit_xgboost(
        tuned_parameters,
        n_estimators=(
            tree_count
        ),
    )

    model.fit(
        x_train,
        y_train,
    )

    base_probabilities = (
        model.predict_proba(
            x_validation
        )[
            :,
            1
        ]
    )

    recorded_base_probabilities = (
        predictions[
            BASE_PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    maximum_probability_error = float(
        np.max(
            np.abs(
                base_probabilities
                - recorded_base_probabilities
            )
        )
    )

    print(
        "Maximum base probability "
        "reproduction error:",
        f"{maximum_probability_error:.12g}",
    )

    if (
        maximum_probability_error
        > PROBABILITY_TOLERANCE
    ):
        raise ValueError(
            "Frozen base model did not "
            "reproduce recorded probabilities."
        )

    selected_indices = (
        cases[
            "validation_row"
        ].to_numpy(
            dtype=int
        )
    )

    x_selected = (
        x_validation[
            selected_indices
        ]
    )

    print(
        "\nCalculating SHAP values for "
        "selected cases..."
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

    selected_shap_values = (
        explainer.shap_values(
            x_selected,
            check_additivity=True,
        )
    )

    selected_shap_values = np.asarray(
        selected_shap_values,
        dtype=float,
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

    raw_margins = np.asarray(
        model.predict(
            x_selected,
            output_margin=True,
        ),
        dtype=float,
    )

    reconstructed_margins = (
        expected_value
        + selected_shap_values.sum(
            axis=1
        )
    )

    additivity_errors = np.abs(
        raw_margins
        - reconstructed_margins
    )

    maximum_additivity_error = float(
        additivity_errors.max()
    )

    print(
        "Maximum local additivity error:",
        f"{maximum_additivity_error:.12g}",
    )

    if (
        maximum_additivity_error
        > ADDITIVITY_TOLERANCE
    ):
        raise ValueError(
            "Local SHAP additivity error "
            "exceeds tolerance."
        )

    if hasattr(
        x_selected,
        "toarray",
    ):
        dense_selected = np.asarray(
            x_selected.toarray(),
            dtype=float,
        )
    else:
        dense_selected = np.asarray(
            x_selected,
            dtype=float,
        )

    case_rows = []
    contribution_rows = []
    figure_artifacts = {}

    print(
        "\nBuilding local explanations..."
    )

    for local_position, case in (
        cases.reset_index(
            drop=True
        ).iterrows()
    ):
        case_name = str(
            case[
                "case_name"
            ]
        )

        validation_row = int(
            case[
                "validation_row"
            ]
        )

        y_true = int(
            case[
                "y_true"
            ]
        )

        calibrated_probability = float(
            case[
                "calibrated_probability"
            ]
        )

        predicted_positive = bool(
            case[
                "predicted_positive"
            ]
        )

        label = confusion_label(
            y_true,
            predicted_positive,
        )

        local_shap = (
            selected_shap_values[
                local_position
            ]
        )

        raw_margin = float(
            raw_margins[
                local_position
            ]
        )

        base_probability = float(
            base_probabilities[
                validation_row
            ]
        )

        local_additivity_error = float(
            additivity_errors[
                local_position
            ]
        )

        source_table = (
            aggregate_local_shap_to_source(
                local_shap,
                feature_metadata,
            )
        )

        raw_case = validation.iloc[
            validation_row
        ]

        source_table[
            "case_name"
        ] = (
            case_name
        )

        source_table[
            "validation_row"
        ] = (
            validation_row
        )

        source_table[
            "source_value"
        ] = (
            source_table[
                "source_feature"
            ].map(
                lambda feature, raw_case=raw_case: (
                    format_source_value(
                        raw_case[
                            feature
                        ]
                    )
                )
            )
        )

        contribution_rows.append(
            source_table[
                [
                    "case_name",
                    "validation_row",
                    "rank",
                    "source_feature",
                    "source_value",
                    "transformed_feature_count",
                    "shap_value",
                    "abs_shap_value",
                ]
            ]
        )

        positive_source = (
            source_table.sort_values(
                "shap_value",
                ascending=False,
            ).iloc[
                0
            ]
        )

        negative_source = (
            source_table.sort_values(
                "shap_value",
                ascending=True,
            ).iloc[
                0
            ]
        )

        explanation = shap.Explanation(
            values=(
                local_shap
            ),
            base_values=(
                expected_value
            ),
            data=(
                dense_selected[
                    local_position
                ]
            ),
            feature_names=(
                feature_names
            ),
        )

        print(
            f"  {case_name}: "
            f"{label}, "
            f"calibrated p="
            f"{calibrated_probability:.6f}"
        )

        figure_artifacts[
            case_name
        ] = save_waterfall(
            case_name=(
                case_name
            ),
            explanation=(
                explanation
            ),
            y_true=(
                y_true
            ),
            calibrated_probability=(
                calibrated_probability
            ),
            threshold=(
                threshold
            ),
            label=(
                label
            ),
        )

        case_rows.append(
            {
                "case_name": (
                    case_name
                ),
                "validation_row": (
                    validation_row
                ),
                "confusion_label": (
                    label
                ),
                "y_true": (
                    y_true
                ),
                "predicted_positive": (
                    predicted_positive
                ),
                "threshold": (
                    threshold
                ),
                "calibrated_probability": (
                    calibrated_probability
                ),
                "base_xgboost_probability": (
                    base_probability
                ),
                "raw_xgboost_margin": (
                    raw_margin
                ),
                "shap_expected_value": (
                    expected_value
                ),
                "local_additivity_error": (
                    local_additivity_error
                ),
                "top_positive_source_feature": str(
                    positive_source[
                        "source_feature"
                    ]
                ),
                "top_positive_source_shap": float(
                    positive_source[
                        "shap_value"
                    ]
                ),
                "top_negative_source_feature": str(
                    negative_source[
                        "source_feature"
                    ]
                ),
                "top_negative_source_shap": float(
                    negative_source[
                        "shap_value"
                    ]
                ),
            }
        )

    case_table = pd.DataFrame(
        case_rows
    )

    contribution_table = pd.concat(
        contribution_rows,
        ignore_index=True,
    )

    forbidden_output_columns = {
        "encounter_id",
        "patient_nbr",
        "source_row",
    }

    for (
        output_name,
        output_table,
    ) in (
        (
            "case table",
            case_table,
        ),
        (
            "source contribution table",
            contribution_table,
        ),
    ):
        if (
            forbidden_output_columns
            & set(
                output_table.columns
            )
        ):
            raise ValueError(
                f"{output_name} contains "
                "a forbidden identifier."
            )

    CASE_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_table.to_csv(
        CASE_TABLE_PATH,
        index=False,
    )

    contribution_table.to_csv(
        SOURCE_CONTRIBUTION_PATH,
        index=False,
    )

    summary = {
        "phase": 9,
        "analysis": (
            "predefined_local_shap_explanations"
        ),
        "selection_policy": {
            "probability_source": (
                CALIBRATED_PROBABILITY_COLUMN
            ),
            "threshold": (
                threshold
            ),
            "cases": [
                "high_confidence_true_positive",
                "high_confidence_false_positive",
                "near_threshold_false_negative",
                "low_risk_true_negative",
                "closest_unused_to_threshold",
            ],
            "selection_defined_before_case_review": True,
        },
        "explanation_target": {
            "model": (
                "tuned_xgboost"
            ),
            "final_development_variant": (
                EXPECTED_MODEL
            ),
            "output_space": (
                "raw_xgboost_margin_log_odds"
            ),
            "calibrated_probability_directly_decomposed": (
                False
            ),
        },
        "audit": {
            "phase7_prediction_sha256": (
                prediction_hash
            ),
            "maximum_base_probability_reproduction_error": (
                maximum_probability_error
            ),
            "maximum_local_additivity_error": (
                maximum_additivity_error
            ),
            "probability_tolerance": (
                PROBABILITY_TOLERANCE
            ),
            "additivity_tolerance": (
                ADDITIVITY_TOLERANCE
            ),
        },
        "selected_cases": (
            case_table.to_dict(
                orient="records"
            )
        ),
        "figure_artifacts": (
            figure_artifacts
        ),
        "output_artifacts": {
            "case_table": str(
                CASE_TABLE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "source_contributions": str(
                SOURCE_CONTRIBUTION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "privacy": {
            "encounter_id_saved": False,
            "patient_nbr_saved": False,
            "source_row_saved": False,
            "validation_row_is_positional_only": True,
        },
        "data_policy": {
            "fit_split": "train",
            "case_selection_split": (
                "validation"
            ),
            "explanation_split": (
                "validation"
            ),
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
        "\nLOCAL CASE SUMMARY"
    )

    print("-" * 96)

    print(
        case_table[
            [
                "case_name",
                "validation_row",
                "confusion_label",
                "y_true",
                "calibrated_probability",
                "base_xgboost_probability",
                "raw_xgboost_margin",
                "top_positive_source_feature",
                "top_negative_source_feature",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nSaved cases       :",
        CASE_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved contributions:",
        SOURCE_CONTRIBUTION_PATH.relative_to(
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
        "\nWaterfall output : raw XGBoost "
        "margin/log-odds"
    )

    print(
        "Case selection   : calibrated "
        "probability + frozen threshold"
    )

    print(
        "Identifiers saved: False"
    )

    print(
        "Test used        : False"
    )


if __name__ == "__main__":
    main()