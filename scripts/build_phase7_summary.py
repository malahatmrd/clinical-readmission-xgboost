from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_bootstrap_validation.json"
)

CALIBRATION_BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_bootstrap.json"
)

CALIBRATION_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

CALIBRATION_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_selection.json"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_final_validation_summary.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_final_validation_summary.json"
)

FINAL_MODELS = (
    "logistic_regression",
    "early_stopped_xgboost",
    "tuned_xgboost_sigmoid",
)

EXPECTED_SELECTED_VARIANT = (
    "tuned_xgboost_sigmoid"
)


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    base_bootstrap = load_json(
        BASE_BOOTSTRAP_PATH
    )

    calibration_bootstrap = load_json(
        CALIBRATION_BOOTSTRAP_PATH
    )

    calibration_candidates = load_json(
        CALIBRATION_CANDIDATES_PATH
    )

    selection = load_json(
        CALIBRATION_SELECTION_PATH
    )

    if (
        selection["selected_variant"]
        != EXPECTED_SELECTED_VARIANT
    ):
        raise ValueError(
            "Unexpected selected Phase 7 variant."
        )

    base_metrics = (
        base_bootstrap[
            "model_metrics"
        ]
    )

    calibrated_metrics = (
        calibration_bootstrap[
            "variant_metrics"
        ]
    )

    calibration_diagnostics = (
        calibration_candidates[
            "candidates"
        ]
    )

    rows = []
    model_payload = {}

    for model_name in FINAL_MODELS:
        if model_name in base_metrics:
            bootstrap_metrics = (
                base_metrics[
                    model_name
                ]
            )
        else:
            bootstrap_metrics = (
                calibrated_metrics[
                    model_name
                ]
            )

        diagnostics = (
            calibration_diagnostics[
                model_name
            ]
        )

        row = {
            "model": model_name,
            "roc_auc": (
                bootstrap_metrics[
                    "roc_auc"
                ]["estimate"]
            ),
            "roc_auc_ci_lower": (
                bootstrap_metrics[
                    "roc_auc"
                ]["ci_lower"]
            ),
            "roc_auc_ci_upper": (
                bootstrap_metrics[
                    "roc_auc"
                ]["ci_upper"]
            ),
            "average_precision": (
                bootstrap_metrics[
                    "average_precision"
                ]["estimate"]
            ),
            "average_precision_ci_lower": (
                bootstrap_metrics[
                    "average_precision"
                ]["ci_lower"]
            ),
            "average_precision_ci_upper": (
                bootstrap_metrics[
                    "average_precision"
                ]["ci_upper"]
            ),
            "brier_score": (
                bootstrap_metrics[
                    "brier_score"
                ]["estimate"]
            ),
            "brier_ci_lower": (
                bootstrap_metrics[
                    "brier_score"
                ]["ci_lower"]
            ),
            "brier_ci_upper": (
                bootstrap_metrics[
                    "brier_score"
                ]["ci_upper"]
            ),
            "log_loss": (
                bootstrap_metrics[
                    "log_loss"
                ]["estimate"]
            ),
            "log_loss_ci_lower": (
                bootstrap_metrics[
                    "log_loss"
                ]["ci_lower"]
            ),
            "log_loss_ci_upper": (
                bootstrap_metrics[
                    "log_loss"
                ]["ci_upper"]
            ),
            "calibration_intercept": (
                diagnostics[
                    "calibration_intercept"
                ]
            ),
            "calibration_slope": (
                diagnostics[
                    "calibration_slope"
                ]
            ),
            "quantile_ece": (
                diagnostics[
                    "quantile_ece"
                ]
            ),
            "mean_predicted_probability": (
                diagnostics[
                    "mean_predicted_probability"
                ]
            ),
        }

        rows.append(
            row
        )

        model_payload[
            model_name
        ] = row

    table = pd.DataFrame(
        rows
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 7,
        "analysis": (
            "final_validation_discrimination_and_calibration_summary"
        ),
        "selected_variant": (
            EXPECTED_SELECTED_VARIANT
        ),
        "selection_status": (
            "frozen_before_test_evaluation"
        ),
        "sample_counts": (
            base_bootstrap[
                "sample_counts"
            ]
        ),
        "bootstrap_protocol": (
            base_bootstrap[
                "bootstrap_protocol"
            ]
        ),
        "models": model_payload,
        "data_policy": {
            "development_evaluation_split": (
                "validation"
            ),
            "test_used": False,
            "threshold_optimized": False,
        },
        "source_artifacts": {
            "base_bootstrap": str(
                BASE_BOOTSTRAP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "calibration_bootstrap": str(
                CALIBRATION_BOOTSTRAP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "calibration_candidates": str(
                CALIBRATION_CANDIDATES_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "calibration_selection": str(
                CALIBRATION_SELECTION_PATH.relative_to(
                    PROJECT_ROOT
                )
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

    print("=" * 88)
    print("PHASE 7 FINAL VALIDATION SUMMARY")
    print("=" * 88)

    for row in rows:
        print(
            f"\n{row['model']}"
        )

        print(
            "  ROC-AUC           : "
            f"{row['roc_auc']:.6f} "
            f"[{row['roc_auc_ci_lower']:.6f}, "
            f"{row['roc_auc_ci_upper']:.6f}]"
        )

        print(
            "  Average Precision : "
            f"{row['average_precision']:.6f} "
            f"[{row['average_precision_ci_lower']:.6f}, "
            f"{row['average_precision_ci_upper']:.6f}]"
        )

        print(
            "  Brier Score       : "
            f"{row['brier_score']:.6f}"
        )

        print(
            "  Log Loss          : "
            f"{row['log_loss']:.6f}"
        )

        print(
            "  Calibration slope : "
            f"{row['calibration_slope']:.6f}"
        )

        print(
            "  ECE               : "
            f"{row['quantile_ece']:.6f}"
        )

    print(
        "\nSelected variant:",
        EXPECTED_SELECTED_VARIANT,
    )

    print(
        "Test used       : False"
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


if __name__ == "__main__":
    main()