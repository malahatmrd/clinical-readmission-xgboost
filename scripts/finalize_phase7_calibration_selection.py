from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_bootstrap.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_selection.json"
)

SELECTED_VARIANT = (
    "tuned_xgboost_sigmoid"
)

BASE_VARIANT = (
    "tuned_xgboost"
)

ALTERNATIVE_VARIANT = (
    "tuned_xgboost_isotonic"
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
    candidates = load_json(
        CANDIDATE_PATH
    )

    bootstrap = load_json(
        BOOTSTRAP_PATH
    )

    selected_metrics = (
        candidates[
            "candidates"
        ][
            SELECTED_VARIANT
        ]
    )

    sigmoid_vs_raw = (
        bootstrap[
            "paired_differences"
        ][
            "tuned_xgboost_sigmoid_minus_tuned_xgboost"
        ]
    )

    isotonic_vs_sigmoid = (
        bootstrap[
            "paired_differences"
        ][
            "tuned_xgboost_isotonic_minus_tuned_xgboost_sigmoid"
        ]
    )

    if (
        sigmoid_vs_raw[
            "brier_score"
        ][
            "ci_upper"
        ]
        >= 0.0
    ):
        raise ValueError(
            "Sigmoid Brier improvement "
            "is not consistently supported."
        )

    if (
        sigmoid_vs_raw[
            "log_loss"
        ][
            "ci_upper"
        ]
        >= 0.0
    ):
        raise ValueError(
            "Sigmoid Log Loss improvement "
            "is not consistently supported."
        )

    if (
        isotonic_vs_sigmoid[
            "average_precision"
        ][
            "ci_upper"
        ]
        >= 0.0
    ):
        raise ValueError(
            "Expected isotonic AP disadvantage "
            "was not reproduced."
        )

    isotonic_brier_uncertain = bool(
        isotonic_vs_sigmoid[
            "brier_score"
        ][
            "ci_lower"
        ]
        <= 0.0
        <= isotonic_vs_sigmoid[
            "brier_score"
        ][
            "ci_upper"
        ]
    )

    isotonic_log_loss_uncertain = bool(
        isotonic_vs_sigmoid[
            "log_loss"
        ][
            "ci_lower"
        ]
        <= 0.0
        <= isotonic_vs_sigmoid[
            "log_loss"
        ][
            "ci_upper"
        ]
    )

    if not (
        isotonic_brier_uncertain
        and isotonic_log_loss_uncertain
    ):
        raise ValueError(
            "Isotonic probability-quality "
            "advantage is no longer uncertain."
        )

    output = {
        "phase": 7,
        "decision": (
            "final_calibration_variant"
        ),
        "selected_variant": (
            SELECTED_VARIANT
        ),
        "base_variant": (
            BASE_VARIANT
        ),
        "rejected_alternative": (
            ALTERNATIVE_VARIANT
        ),
        "selection_rationale": {
            "sigmoid_preserves_roc_auc": True,
            "sigmoid_preserves_average_precision": True,
            "sigmoid_improves_brier_with_ci_excluding_zero": True,
            "sigmoid_improves_log_loss_with_ci_excluding_zero": True,
            "isotonic_brier_advantage_vs_sigmoid_uncertain": True,
            "isotonic_log_loss_advantage_vs_sigmoid_uncertain": True,
            "isotonic_average_precision_loss_vs_sigmoid_ci_excludes_zero": True,
        },
        "selected_validation_metrics": {
            "roc_auc": float(
                selected_metrics[
                    "roc_auc"
                ]
            ),
            "average_precision": float(
                selected_metrics[
                    "average_precision"
                ]
            ),
            "brier_score": float(
                selected_metrics[
                    "brier_score"
                ]
            ),
            "log_loss": float(
                selected_metrics[
                    "log_loss"
                ]
            ),
            "calibration_intercept": float(
                selected_metrics[
                    "calibration_intercept"
                ]
            ),
            "calibration_slope": float(
                selected_metrics[
                    "calibration_slope"
                ]
            ),
            "quantile_ece": float(
                selected_metrics[
                    "quantile_ece"
                ]
            ),
            "mean_predicted_probability": float(
                selected_metrics[
                    "mean_predicted_probability"
                ]
            ),
        },
        "data_policy": {
            "selection_split": (
                "validation"
            ),
            "test_used": False,
            "threshold_optimized": False,
        },
        "source_artifacts": {
            "candidate_analysis": str(
                CANDIDATE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "paired_bootstrap": str(
                BOOTSTRAP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("=" * 88)
    print("PHASE 7 CALIBRATION SELECTION")
    print("=" * 88)

    print(
        "\nSelected variant :",
        SELECTED_VARIANT,
    )

    print(
        "Base variant     :",
        BASE_VARIANT,
    )

    print(
        "Alternative      :",
        ALTERNATIVE_VARIANT,
    )

    print(
        "\nROC-AUC          :",
        f"{selected_metrics['roc_auc']:.6f}",
    )

    print(
        "Average Precision:",
        f"{selected_metrics['average_precision']:.6f}",
    )

    print(
        "Brier Score      :",
        f"{selected_metrics['brier_score']:.6f}",
    )

    print(
        "Log Loss         :",
        f"{selected_metrics['log_loss']:.6f}",
    )

    print(
        "Calibration slope:",
        f"{selected_metrics['calibration_slope']:.6f}",
    )

    print(
        "Calibration ECE  :",
        f"{selected_metrics['quantile_ece']:.6f}",
    )

    print(
        "\nTest used        : False"
    )

    print(
        "\nSaved selection  :",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()