from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBGROUP_BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_validation_subgroup_bootstrap.csv"
)

ROBUSTNESS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_repeated_encounter_robustness.csv"
)

CLUSTER_BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_cluster_bootstrap.csv"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FIGURE_DPI = 300


def save_figure(
    figure,
    stem: str,
) -> None:
    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_ROOT / f"{stem}.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_ROOT / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def build_subgroup_label(
    row: pd.Series,
) -> str:
    return (
        f"{row['subgroup_name']} | "
        f"{row['subgroup_value']}"
    )


def build_subgroup_roc_figure(
    bootstrap: pd.DataFrame,
) -> None:
    data = bootstrap.loc[
        bootstrap[
            "metric"
        ].eq(
            "roc_auc"
        )
    ].copy()

    data["label"] = data.apply(
        build_subgroup_label,
        axis=1,
    )

    data = data.iloc[
        ::-1
    ].reset_index(
        drop=True
    )

    y_positions = np.arange(
        len(
            data
        )
    )

    lower_error = (
        data[
            "estimate"
        ]
        - data[
            "ci_lower"
        ]
    )

    upper_error = (
        data[
            "ci_upper"
        ]
        - data[
            "estimate"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            6.8,
        ),
        constrained_layout=True,
    )

    axis.errorbar(
        data[
            "estimate"
        ],
        y_positions,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        0.650820,
        linestyle="--",
        linewidth=1.3,
        label="Overall Validation ROC-AUC",
    )

    axis.axvline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        label="Chance discrimination",
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        data[
            "label"
        ]
    )

    axis.set_xlabel(
        "ROC-AUC with 95% Bootstrap CI"
    )

    axis.set_title(
        "Validation Discrimination by Subgroup"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    axis.legend(
        frameon=False,
    )

    save_figure(
        figure,
        "phase10_validation_subgroup_roc_auc",
    )


def build_subgroup_alert_figure(
    bootstrap: pd.DataFrame,
) -> None:
    data = bootstrap.loc[
        bootstrap[
            "metric"
        ].eq(
            "alerts_per_100"
        )
    ].copy()

    data["label"] = data.apply(
        build_subgroup_label,
        axis=1,
    )

    data = data.iloc[
        ::-1
    ].reset_index(
        drop=True
    )

    y_positions = np.arange(
        len(
            data
        )
    )

    lower_error = (
        data[
            "estimate"
        ]
        - data[
            "ci_lower"
        ]
    )

    upper_error = (
        data[
            "ci_upper"
        ]
        - data[
            "estimate"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            6.8,
        ),
        constrained_layout=True,
    )

    axis.errorbar(
        data[
            "estimate"
        ],
        y_positions,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        18.568979,
        linestyle="--",
        linewidth=1.3,
        label=(
            "Overall Validation "
            "alert burden"
        ),
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        data[
            "label"
        ]
    )

    axis.set_xlabel(
        "Alerts per 100 Encounters "
        "with 95% Bootstrap CI"
    )

    axis.set_title(
        "Frozen Threshold Alert Burden by Subgroup"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    axis.legend(
        frameon=False,
    )

    save_figure(
        figure,
        "phase10_validation_subgroup_alert_burden",
    )


def build_repeated_encounter_figure(
    robustness: pd.DataFrame,
) -> None:
    display_names = {
        "primary_validation": (
            "Primary Validation"
        ),
        (
            "all_eligible_encounters_"
            "for_validation_patients"
        ): (
            "All Eligible Encounters"
        ),
        (
            "subsequent_eligible_"
            "encounters_only"
        ): (
            "Subsequent Encounters Only"
        ),
    }

    data = robustness.copy()

    data["label"] = (
        data[
            "dataset"
        ]
        .map(
            display_names
        )
        .fillna(
            data[
                "dataset"
            ]
        )
    )

    metrics = [
        "sensitivity",
        "specificity",
        "ppv",
    ]

    x_positions = np.arange(
        len(
            metrics
        )
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            6.3,
        ),
        constrained_layout=True,
    )

    offsets = np.linspace(
        -width,
        width,
        len(
            data
        ),
    )

    for offset, (_, row) in zip(
        offsets,
        data.iterrows(),
        strict=True,
    ):
        axis.bar(
            x_positions + offset,
            [
                row[
                    metric
                ]
                for metric in metrics
            ],
            width=width,
            label=row[
                "label"
            ],
        )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            "Sensitivity",
            "Specificity",
            "PPV",
        ]
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_ylabel(
        "Metric Value"
    )

    axis.set_title(
        "Repeated-Encounter Robustness "
        "at Frozen Threshold 0.105"
    )

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    axis.legend(
        frameon=False,
    )

    save_figure(
        figure,
        "phase10_repeated_encounter_operating_metrics",
    )


def build_cluster_difference_figure(
    cluster_bootstrap: pd.DataFrame,
) -> None:
    metrics = [
        "roc_auc",
        "sensitivity",
        "specificity",
        "ppv",
    ]

    labels = {
        "roc_auc": "ROC-AUC",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "ppv": "PPV",
    }

    data = (
        cluster_bootstrap.loc[
            cluster_bootstrap[
                "metric"
            ].isin(
                metrics
            )
        ]
        .copy()
        .set_index(
            "metric"
        )
        .loc[
            metrics
        ]
        .reset_index()
    )

    y_positions = np.arange(
        len(
            data
        )
    )

    lower_error = (
        data[
            "estimate"
        ]
        - data[
            "ci_lower"
        ]
    )

    upper_error = (
        data[
            "ci_upper"
        ]
        - data[
            "estimate"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            5.4,
        ),
        constrained_layout=True,
    )

    axis.errorbar(
        data[
            "estimate"
        ],
        y_positions,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.3,
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        [
            labels[
                metric
            ]
            for metric in data[
                "metric"
            ]
        ]
    )

    axis.set_xlabel(
        "All-Eligible Minus Primary "
        "(95% Patient-Cluster Bootstrap CI)"
    )

    axis.set_title(
        "Repeated-Encounter Robustness Differences"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    save_figure(
        figure,
        "phase10_cluster_bootstrap_metric_differences",
    )


def build_cluster_alert_figure(
    cluster_bootstrap: pd.DataFrame,
) -> None:
    row = cluster_bootstrap.loc[
        cluster_bootstrap[
            "metric"
        ].eq(
            "alerts_per_100"
        )
    ].iloc[
        0
    ]

    figure, axis = plt.subplots(
        figsize=(
            7.0,
            4.8,
        ),
        constrained_layout=True,
    )

    estimate = float(
        row[
            "estimate"
        ]
    )

    lower = float(
        row[
            "ci_lower"
        ]
    )

    upper = float(
        row[
            "ci_upper"
        ]
    )

    axis.errorbar(
        [estimate],
        [0],
        xerr=np.array(
            [
                [
                    estimate
                    - lower
                ],
                [
                    upper
                    - estimate
                ],
            ]
        ),
        fmt="o",
        capsize=5,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.3,
    )

    axis.set_yticks(
        [0]
    )

    axis.set_yticklabels(
        [
            "Alert burden"
        ]
    )

    axis.set_xlabel(
        "Additional Alerts per 100 Encounters "
        "(95% Patient-Cluster Bootstrap CI)"
    )

    axis.set_title(
        "Repeated-Encounter Increase in Alert Burden"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    save_figure(
        figure,
        "phase10_cluster_bootstrap_alert_difference",
    )


def main() -> None:
    subgroup_bootstrap = pd.read_csv(
        SUBGROUP_BOOTSTRAP_PATH
    )

    robustness = pd.read_csv(
        ROBUSTNESS_PATH
    )

    cluster_bootstrap = pd.read_csv(
        CLUSTER_BOOTSTRAP_PATH
    )

    print(
        "=" * 88
    )

    print(
        "PHASE 10 FIGURES"
    )

    print(
        "=" * 88
    )

    build_subgroup_roc_figure(
        subgroup_bootstrap
    )

    print(
        "\nSubgroup ROC-AUC figure: complete"
    )

    build_subgroup_alert_figure(
        subgroup_bootstrap
    )

    print(
        "Subgroup alert-burden figure: complete"
    )

    build_repeated_encounter_figure(
        robustness
    )

    print(
        "Repeated-encounter operating figure: complete"
    )

    build_cluster_difference_figure(
        cluster_bootstrap
    )

    print(
        "Cluster-bootstrap difference figure: complete"
    )

    build_cluster_alert_figure(
        cluster_bootstrap
    )

    print(
        "Cluster-bootstrap alert figure: complete"
    )

    print(
        "\nFigures saved as PNG and SVG."
    )

    print(
        "Test used: False"
    )


if __name__ == "__main__":
    main()