from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase8_validation_threshold_sweep.csv"
)

SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_threshold_selection.json"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

EXPECTED_MODEL = (
    "tuned_xgboost_sigmoid"
)

EXPECTED_THRESHOLD = 0.105

FIGURE_DPI = 300


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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


def build_threshold_tradeoff_figure(
    sweep: pd.DataFrame,
    reference_threshold: float,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
        constrained_layout=True,
    )

    axis.plot(
        sweep["threshold"],
        sweep["sensitivity"],
        linewidth=2.0,
        label="Sensitivity",
    )

    axis.plot(
        sweep["threshold"],
        sweep["specificity"],
        linewidth=2.0,
        label="Specificity",
    )

    axis.plot(
        sweep["threshold"],
        sweep["ppv"],
        linewidth=2.0,
        label="PPV",
    )

    axis.plot(
        sweep["threshold"],
        sweep["f1"],
        linewidth=2.0,
        label="F1",
    )

    axis.axvline(
        reference_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Reference threshold "
            f"({reference_threshold:.3f})"
        ),
    )

    axis.set_xlim(
        0.01,
        0.30,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "Risk Threshold"
    )

    axis.set_ylabel(
        "Metric Value"
    )

    axis.set_title(
        "Validation Threshold Trade-offs"
    )

    axis.legend(
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    save_figure(
        figure,
        "phase8_validation_threshold_tradeoffs",
    )


def build_alert_burden_figure(
    sweep: pd.DataFrame,
    reference_threshold: float,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
        constrained_layout=True,
    )

    axis.plot(
        sweep["threshold"],
        sweep["alerts_per_100"],
        linewidth=2.0,
        label="Alerts per 100 patients",
    )

    axis.axhline(
        20.0,
        linestyle="--",
        linewidth=1.5,
        label="Moderate-capacity limit (20/100)",
    )

    axis.axhline(
        10.0,
        linestyle=":",
        linewidth=1.5,
        label="Limited-capacity limit (10/100)",
    )

    axis.axvline(
        reference_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Reference threshold "
            f"({reference_threshold:.3f})"
        ),
    )

    axis.set_xlim(
        0.01,
        0.30,
    )

    axis.set_ylim(
        0.0,
        100.0,
    )

    axis.set_xlabel(
        "Risk Threshold"
    )

    axis.set_ylabel(
        "Alerts per 100 Patients"
    )

    axis.set_title(
        "Validation Alert Burden"
    )

    axis.legend(
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    save_figure(
        figure,
        "phase8_validation_alert_burden",
    )


def build_decision_curve_figure(
    sweep: pd.DataFrame,
    reference_threshold: float,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.0, 6.2),
        constrained_layout=True,
    )

    axis.plot(
        sweep["threshold"],
        sweep["model_net_benefit"],
        linewidth=2.0,
        label="Model",
    )

    axis.plot(
        sweep["threshold"],
        sweep["treat_all_net_benefit"],
        linewidth=1.8,
        label="Treat all",
    )

    axis.plot(
        sweep["threshold"],
        sweep["treat_none_net_benefit"],
        linewidth=1.8,
        label="Treat none",
    )

    axis.axvline(
        reference_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Reference threshold "
            f"({reference_threshold:.3f})"
        ),
    )

    axis.set_xlim(
        0.01,
        0.30,
    )

    axis.set_xlabel(
        "Risk Threshold"
    )

    axis.set_ylabel(
        "Net Benefit"
    )

    axis.set_title(
        "Validation Decision Curve"
    )

    axis.legend(
        frameon=False,
    )

    axis.grid(
        alpha=0.2
    )

    save_figure(
        figure,
        "phase8_validation_decision_curve",
    )


def main() -> None:
    sweep = pd.read_csv(
        SWEEP_PATH
    )

    selection = load_json(
        SELECTION_PATH
    )

    if (
        selection[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    if (
        selection[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
        )

    if not (
        selection[
            "data_policy"
        ][
            "threshold_frozen"
        ]
    ):
        raise ValueError(
            "Threshold must be frozen "
            "before figure generation."
        )

    reference_threshold = float(
        selection[
            "reference_threshold"
        ]
    )

    if abs(
        reference_threshold
        - EXPECTED_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected reference threshold."
        )

    print("=" * 96)
    print("PHASE 8 VALIDATION FIGURES")
    print("=" * 96)

    print(
        "\nSelected model     :",
        EXPECTED_MODEL,
    )

    print(
        "Reference threshold:",
        f"{reference_threshold:.3f}",
    )

    print(
        "\nBuilding threshold trade-off figure..."
    )

    build_threshold_tradeoff_figure(
        sweep,
        reference_threshold,
    )

    print(
        "Building alert-burden figure..."
    )

    build_alert_burden_figure(
        sweep,
        reference_threshold,
    )

    print(
        "Building decision-curve figure..."
    )

    build_decision_curve_figure(
        sweep,
        reference_threshold,
    )

    print(
        "\nSaved figures:"
    )

    stems = (
        "phase8_validation_threshold_tradeoffs",
        "phase8_validation_alert_burden",
        "phase8_validation_decision_curve",
    )

    for stem in stems:
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
        "\nThreshold frozen: True"
    )

    print(
        "Test used       : False"
    )


if __name__ == "__main__":
    main()