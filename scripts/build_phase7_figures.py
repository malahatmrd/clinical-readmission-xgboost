from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from clinical_readmission.evaluation.calibration import (
    DEFAULT_CALIBRATION_BINS,
    build_calibration_curve_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)

CANDIDATE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_selection.json"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

TARGET_COLUMN = "readmitted_30d"

MODEL_COLUMNS = {
    "Logistic Regression": (
        "logistic_probability"
    ),
    "Early-Stopped XGBoost": (
        "early_stopped_xgboost_probability"
    ),
    "Tuned XGBoost + Sigmoid": (
        "tuned_xgboost_sigmoid_probability"
    ),
}

EXPECTED_SELECTED_VARIANT = (
    "tuned_xgboost_sigmoid"
)

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


def save_figure(
    figure,
    stem: str,
) -> None:
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


def build_roc_figure(
    predictions: pd.DataFrame,
) -> None:
    target = predictions[
        TARGET_COLUMN
    ].to_numpy()

    figure, axis = plt.subplots(
        figsize=(7.2, 6.2),
        constrained_layout=True,
    )

    for (
        model_label,
        probability_column,
    ) in MODEL_COLUMNS.items():
        probabilities = predictions[
            probability_column
        ].to_numpy()

        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                target,
                probabilities,
            )
        )

        roc_auc = roc_auc_score(
            target,
            probabilities,
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.0,
            label=(
                f"{model_label} "
                f"(AUC = {roc_auc:.3f})"
            ),
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.2,
        label="Chance",
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        "Validation ROC Curves"
    )

    axis.legend(
        loc="lower right",
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    save_figure(
        figure,
        "phase7_validation_roc_curve",
    )


def build_precision_recall_figure(
    predictions: pd.DataFrame,
) -> None:
    target = predictions[
        TARGET_COLUMN
    ].to_numpy()

    prevalence = float(
        target.mean()
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 6.2),
        constrained_layout=True,
    )

    for (
        model_label,
        probability_column,
    ) in MODEL_COLUMNS.items():
        probabilities = predictions[
            probability_column
        ].to_numpy()

        precision, recall, _ = (
            precision_recall_curve(
                target,
                probabilities,
            )
        )

        average_precision = (
            average_precision_score(
                target,
                probabilities,
            )
        )

        axis.plot(
            recall,
            precision,
            linewidth=2.0,
            label=(
                f"{model_label} "
                f"(AP = {average_precision:.3f})"
            ),
        )

    axis.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Outcome prevalence "
            f"({prevalence:.3f})"
        ),
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "Recall"
    )

    axis.set_ylabel(
        "Precision"
    )

    axis.set_title(
        "Validation Precision–Recall Curves"
    )

    axis.legend(
        loc="upper right",
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    save_figure(
        figure,
        "phase7_validation_precision_recall_curve",
    )


def build_calibration_figure(
    predictions: pd.DataFrame,
) -> None:
    target = predictions[
        TARGET_COLUMN
    ].to_numpy()

    figure, axis = plt.subplots(
        figsize=(7.2, 6.2),
        constrained_layout=True,
    )

    for (
        model_label,
        probability_column,
    ) in MODEL_COLUMNS.items():
        probabilities = predictions[
            probability_column
        ].to_numpy()

        curve = (
            build_calibration_curve_table(
                target,
                probabilities,
                n_bins=(
                    DEFAULT_CALIBRATION_BINS
                ),
            )
        )

        axis.plot(
            curve[
                "mean_predicted_probability"
            ],
            curve[
                "observed_event_rate"
            ],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=model_label,
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.2,
        label="Perfect calibration",
    )

    axis.set_xlim(
        0.0,
        0.35,
    )

    axis.set_ylim(
        0.0,
        0.35,
    )

    axis.set_xlabel(
        "Mean Predicted Probability"
    )

    axis.set_ylabel(
        "Observed Event Rate"
    )

    axis.set_title(
        "Validation Calibration Curves"
    )

    axis.legend(
        loc="upper left",
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    save_figure(
        figure,
        "phase7_validation_calibration_curve",
    )


def main() -> None:
    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    candidate_summary = load_json(
        CANDIDATE_SUMMARY_PATH
    )

    selection = load_json(
        SELECTION_PATH
    )

    if (
        selection[
            "selected_variant"
        ]
        != EXPECTED_SELECTED_VARIANT
    ):
        raise ValueError(
            "Unexpected Phase 7 selected "
            "calibration variant."
        )

    observed_hash = file_sha256(
        PREDICTIONS_PATH
    )

    expected_hash = (
        candidate_summary[
            "candidate_prediction_artifact"
        ][
            "sha256"
        ]
    )

    if observed_hash != expected_hash:
        raise ValueError(
            "Calibration candidate prediction "
            "SHA256 does not match summary."
        )

    required_columns = {
        TARGET_COLUMN,
        *MODEL_COLUMNS.values(),
    }

    missing_columns = sorted(
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Figure input is missing columns: "
            f"{missing_columns}"
        )

    forbidden_identifiers = {
        "encounter_id",
        "patient_nbr",
    }

    present_identifiers = sorted(
        forbidden_identifiers
        & set(predictions.columns)
    )

    if present_identifiers:
        raise ValueError(
            "Figure input contains forbidden "
            f"identifiers: {present_identifiers}"
        )

    print("=" * 88)
    print("PHASE 7 PUBLICATION-GRADE FIGURES")
    print("=" * 88)

    print(
        "\nValidation rows :",
        len(predictions),
    )

    print(
        "Selected variant:",
        selection[
            "selected_variant"
        ],
    )

    print(
        "Prediction SHA256:",
        observed_hash,
    )

    print(
        "\nBuilding ROC figure..."
    )

    build_roc_figure(
        predictions
    )

    print(
        "Building Precision-Recall figure..."
    )

    build_precision_recall_figure(
        predictions
    )

    print(
        "Building calibration figure..."
    )

    build_calibration_figure(
        predictions
    )

    print(
        "\nSaved figures:"
    )

    for stem in (
        "phase7_validation_roc_curve",
        "phase7_validation_precision_recall_curve",
        "phase7_validation_calibration_curve",
    ):
        print(
            " ",
            (
                FIGURE_ROOT
                / f"{stem}.png"
            ).relative_to(
                PROJECT_ROOT
            ),
        )

        print(
            " ",
            (
                FIGURE_ROOT
                / f"{stem}.svg"
            ).relative_to(
                PROJECT_ROOT
            ),
        )

    print(
        "\nTest used: False"
    )


if __name__ == "__main__":
    main()