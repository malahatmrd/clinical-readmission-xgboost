from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from clinical_readmission.features.feature_schema import (
    NUMERIC_FEATURES,
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

SHAP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_shap_validation.json"
)

DISCHARGE_AUDIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_discharge_disposition_audit.json"
)

SOURCE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_source_shap_importance.csv"
)

FEATURE_METADATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_transformed_feature_metadata.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_shap_figures.json"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

TARGET_COLUMN = "readmitted_30d"

EXPECTED_MODEL = (
    "tuned_xgboost_sigmoid"
)

EXPECTED_FEATURE_COUNT = 225

VISUAL_SAMPLE_SIZE = 3000
VISUAL_RANDOM_STATE = 49

TOP_BEESWARM_FEATURES = 20
TOP_SOURCE_FEATURES = 15
TOP_NUMERIC_DEPENDENCE = 3

FIGURE_DPI = 300


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


def draw_stratified_visual_sample(
    target: np.ndarray,
) -> np.ndarray:
    if (
        VISUAL_SAMPLE_SIZE
        > len(target)
    ):
        raise ValueError(
            "Visualization sample exceeds "
            "Validation size."
        )

    rng = np.random.default_rng(
        VISUAL_RANDOM_STATE
    )

    positive_indices = np.flatnonzero(
        target == 1
    )

    negative_indices = np.flatnonzero(
        target == 0
    )

    positive_count = int(
        round(
            VISUAL_SAMPLE_SIZE
            * target.mean()
        )
    )

    negative_count = (
        VISUAL_SAMPLE_SIZE
        - positive_count
    )

    sampled_positive = rng.choice(
        positive_indices,
        size=positive_count,
        replace=False,
    )

    sampled_negative = rng.choice(
        negative_indices,
        size=negative_count,
        replace=False,
    )

    indices = np.concatenate(
        [
            sampled_positive,
            sampled_negative,
        ]
    )

    rng.shuffle(
        indices
    )

    return indices


def to_dense(
    matrix,
) -> np.ndarray:
    if hasattr(
        matrix,
        "toarray",
    ):
        return np.asarray(
            matrix.toarray(),
            dtype=float,
        )

    return np.asarray(
        matrix,
        dtype=float,
    )


def save_figure(
    figure,
    stem: str,
) -> dict[str, str]:
    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_ROOT
        / f"{stem}.png"
    )

    svg_path = (
        FIGURE_ROOT
        / f"{stem}.svg"
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


def build_beeswarm(
    explanation: shap.Explanation,
) -> dict[str, str]:
    figure, axis = plt.subplots(
        figsize=(
            10.5,
            8.5,
        ),
    )

    shap.plots.beeswarm(
        explanation,
        max_display=(
            TOP_BEESWARM_FEATURES
        ),
        show=False,
        ax=axis,
        plot_size=None,
    )

    axis.set_title(
        "Validation SHAP Summary — "
        "Tuned XGBoost"
    )

    axis.set_xlabel(
        "SHAP value "
        "(raw margin / log-odds)"
    )

    figure.tight_layout()

    return save_figure(
        figure,
        "phase9_validation_shap_beeswarm",
    )


def build_source_importance_figure(
    source_importance: pd.DataFrame,
) -> dict[str, str]:
    top = (
        source_importance.head(
            TOP_SOURCE_FEATURES
        )
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            7.0,
        ),
        constrained_layout=True,
    )

    axis.barh(
        top[
            "source_feature"
        ],
        top[
            "mean_abs_shap"
        ],
    )

    axis.set_xlabel(
        "Mean absolute grouped SHAP value "
        "(raw margin / log-odds)"
    )

    axis.set_ylabel(
        "Clinical source feature"
    )

    axis.set_title(
        "Global Clinical Feature Importance"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    return save_figure(
        figure,
        "phase9_validation_source_shap_importance",
    )


def build_numeric_dependence_figure(
    *,
    source_feature: str,
    raw_values: np.ndarray,
    shap_values: np.ndarray,
) -> dict[str, str]:
    finite_mask = (
        np.isfinite(
            raw_values
        )
        & np.isfinite(
            shap_values
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            5.8,
        ),
        constrained_layout=True,
    )

    axis.scatter(
        raw_values[
            finite_mask
        ],
        shap_values[
            finite_mask
        ],
        alpha=0.25,
        s=14,
    )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        source_feature
    )

    axis.set_ylabel(
        "SHAP value "
        "(raw margin / log-odds)"
    )

    axis.set_title(
        "SHAP Dependence — "
        f"{source_feature}"
    )

    axis.grid(
        alpha=0.2,
    )

    safe_name = (
        source_feature
        .replace(
            " ",
            "_",
        )
    )

    return save_figure(
        figure,
        (
            "phase9_validation_shap_dependence_"
            f"{safe_name}"
        ),
    )


def main() -> None:
    shap_summary = load_json(
        SHAP_SUMMARY_PATH
    )

    discharge_audit = load_json(
        DISCHARGE_AUDIT_PATH
    )

    tuned_artifact = load_json(
        TUNED_METRICS_PATH
    )

    if (
        shap_summary[
            "explanation_target"
        ][
            "final_development_variant"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected Phase 9 model."
        )

    if (
        shap_summary[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
        )

    if not (
        discharge_audit[
            "audit_passed"
        ]
    ):
        raise ValueError(
            "Discharge audit must pass "
            "before SHAP visualization."
        )

    if (
        shap_summary[
            "transformed_feature_count"
        ]
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected transformed "
            "feature count."
        )

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
    ].to_numpy(
        dtype=int
    )

    sample_indices = (
        draw_stratified_visual_sample(
            y_validation
        )
    )

    sample_positive_count = int(
        y_validation[
            sample_indices
        ].sum()
    )

    print("=" * 96)
    print("PHASE 9 SHAP FIGURES")
    print("=" * 96)

    print(
        "\nValidation rows          :",
        len(validation),
    )

    print(
        "Visualization sample    :",
        len(sample_indices),
    )

    print(
        "Visualization positives :",
        sample_positive_count,
    )

    print(
        "Visualization seed      :",
        VISUAL_RANDOM_STATE,
    )

    print(
        "Test used               : False"
    )

    print(
        "\nFitting frozen preprocessor..."
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

    if (
        len(
            feature_names
        )
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Feature-name count mismatch."
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

    x_visual = (
        x_validation[
            sample_indices
        ]
    )

    print(
        "Calculating visualization SHAP values..."
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

    visual_shap_values = (
        explainer.shap_values(
            x_visual,
            check_additivity=True,
        )
    )

    visual_shap_values = (
        np.asarray(
            visual_shap_values,
            dtype=float,
        )
    )

    if visual_shap_values.shape != (
        len(
            sample_indices
        ),
        EXPECTED_FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected visualization "
            "SHAP matrix shape."
        )

    dense_visual_data = to_dense(
        x_visual
    )

    expected_value = float(
        np.asarray(
            explainer.expected_value,
            dtype=float,
        ).reshape(
            -1
        )[
            0
        ]
    )

    explanation = shap.Explanation(
        values=(
            visual_shap_values
        ),
        base_values=np.full(
            len(
                sample_indices
            ),
            expected_value,
            dtype=float,
        ),
        data=(
            dense_visual_data
        ),
        feature_names=(
            feature_names
        ),
    )

    source_importance = pd.read_csv(
        SOURCE_IMPORTANCE_PATH
    )

    feature_metadata = pd.read_csv(
        FEATURE_METADATA_PATH
    )

    print(
        "\nBuilding beeswarm..."
    )

    figure_artifacts = {
        "beeswarm": (
            build_beeswarm(
                explanation
            )
        ),
    }

    print(
        "Building source-feature "
        "importance bar chart..."
    )

    figure_artifacts[
        "source_importance"
    ] = (
        build_source_importance_figure(
            source_importance
        )
    )

    top_numeric_features = (
        source_importance[
            source_importance[
                "source_feature"
            ].isin(
                NUMERIC_FEATURES
            )
        ]
        .head(
            TOP_NUMERIC_DEPENDENCE
        )[
            "source_feature"
        ]
        .tolist()
    )

    if (
        len(
            top_numeric_features
        )
        != TOP_NUMERIC_DEPENDENCE
    ):
        raise ValueError(
            "Could not identify the requested "
            "number of numeric SHAP features."
        )

    print(
        "Top numeric dependence features:",
        ", ".join(
            top_numeric_features
        ),
    )

    dependence_artifacts = {}

    visual_validation = (
        validation.iloc[
            sample_indices
        ]
    )

    for source_feature in (
        top_numeric_features
    ):
        feature_rows = (
            feature_metadata[
                feature_metadata[
                    "source_feature"
                ].eq(
                    source_feature
                )
            ]
        )

        if len(
            feature_rows
        ) != 1:
            raise ValueError(
                "Numeric feature must map "
                "to exactly one transformed "
                f"feature: {source_feature}"
            )

        transformed_index = int(
            feature_rows.iloc[
                0
            ][
                "transformed_index"
            ]
        )

        raw_values = pd.to_numeric(
            visual_validation[
                source_feature
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        feature_shap_values = (
            visual_shap_values[
                :,
                transformed_index,
            ]
        )

        print(
            "Building dependence plot:",
            source_feature,
        )

        dependence_artifacts[
            source_feature
        ] = (
            build_numeric_dependence_figure(
                source_feature=(
                    source_feature
                ),
                raw_values=(
                    raw_values
                ),
                shap_values=(
                    feature_shap_values
                ),
            )
        )

    figure_artifacts[
        "numeric_dependence"
    ] = (
        dependence_artifacts
    )

    output = {
        "phase": 9,
        "analysis": (
            "shap_global_and_directional_figures"
        ),
        "model": (
            EXPECTED_MODEL
        ),
        "shap_output_space": (
            "uncalibrated_xgboost_raw_margin"
        ),
        "global_importance_population": {
            "split": (
                "validation"
            ),
            "rows": int(
                len(
                    validation
                )
            ),
            "source_importance_uses_full_validation": (
                True
            ),
        },
        "visualization_sample": {
            "split": (
                "validation"
            ),
            "size": int(
                len(
                    sample_indices
                )
            ),
            "positives": (
                sample_positive_count
            ),
            "random_state": (
                VISUAL_RANDOM_STATE
            ),
            "stratified": True,
            "used_for_ranking": False,
        },
        "beeswarm": {
            "max_display": (
                TOP_BEESWARM_FEATURES
            ),
            "transformed_features": True,
        },
        "source_importance": {
            "max_display": (
                TOP_SOURCE_FEATURES
            ),
            "grouped_source_features": True,
        },
        "numeric_dependence_features": (
            top_numeric_features
        ),
        "figure_artifacts": (
            figure_artifacts
        ),
        "data_policy": {
            "fit_split": (
                "train"
            ),
            "explanation_split": (
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
            output,
            file,
            indent=2,
        )

    print(
        "\nSaved figure summary:",
        OUTPUT_SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nSHAP output space: "
        "raw XGBoost margin/log-odds"
    )

    print(
        "Global ranking  : full Validation"
    )

    print(
        "Visual sample   : "
        f"{VISUAL_SAMPLE_SIZE} rows"
    )

    print(
        "Test used       : False"
    )


if __name__ == "__main__":
    main()