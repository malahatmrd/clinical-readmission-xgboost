from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from clinical_readmission.evaluation.calibration import (
    DEFAULT_CALIBRATION_BINS,
    build_calibration_curve_table,
    calculate_calibration_intercept_slope,
    calculate_quantile_ece,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_validation.json"
)

SUMMARY_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_calibration_summary.csv"
)

CURVE_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_calibration_curves.csv"
)

TARGET_COLUMN = "readmitted_30d"

MODEL_COLUMNS = {
    "logistic_regression": (
        "logistic_probability"
    ),
    "early_stopped_xgboost": (
        "early_stopped_xgboost_probability"
    ),
    "tuned_xgboost": (
        "tuned_xgboost_probability"
    ),
}


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


def main() -> None:
    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    reproduction = load_json(
        REPRODUCTION_PATH
    )

    observed_hash = file_sha256(
        PREDICTIONS_PATH
    )

    expected_hash = (
        reproduction[
            "prediction_artifact"
        ][
            "sha256"
        ]
    )

    if observed_hash != expected_hash:
        raise ValueError(
            "Prediction artifact SHA256 "
            "does not match reproduction summary."
        )

    required_columns = {
        TARGET_COLUMN,
        *MODEL_COLUMNS.values(),
    }

    missing = sorted(
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Missing required prediction columns: "
            f"{missing}"
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
            "Prediction artifact contains "
            "forbidden identifiers: "
            f"{present_identifiers}"
        )

    target = predictions[
        TARGET_COLUMN
    ]

    prevalence = float(
        target.mean()
    )

    print("=" * 88)
    print("PHASE 7 VALIDATION CALIBRATION ANALYSIS")
    print("=" * 88)

    print(
        "\nValidation rows       :",
        len(predictions),
    )

    print(
        "Validation positives  :",
        int(
            target.sum()
        ),
    )

    print(
        "Observed prevalence   :",
        f"{prevalence:.6f}",
    )

    print(
        "Calibration bins      :",
        DEFAULT_CALIBRATION_BINS,
    )

    print(
        "Prediction SHA256     :",
        observed_hash,
    )

    model_results = {}
    summary_rows = []
    curve_tables = []

    for (
        model_name,
        probability_column,
    ) in MODEL_COLUMNS.items():
        probabilities = predictions[
            probability_column
        ]

        (
            intercept,
            slope,
        ) = (
            calculate_calibration_intercept_slope(
                target,
                probabilities,
            )
        )

        ece = (
            calculate_quantile_ece(
                target,
                probabilities,
                n_bins=(
                    DEFAULT_CALIBRATION_BINS
                ),
            )
        )

        mean_probability = float(
            probabilities.mean()
        )

        mean_probability_error = float(
            mean_probability
            - prevalence
        )

        curve = (
            build_calibration_curve_table(
                target,
                probabilities,
                n_bins=(
                    DEFAULT_CALIBRATION_BINS
                ),
            )
        )

        curve.insert(
            0,
            "model",
            model_name,
        )

        curve_tables.append(
            curve
        )

        result = {
            "observed_prevalence": (
                prevalence
            ),
            "mean_predicted_probability": (
                mean_probability
            ),
            "mean_probability_minus_prevalence": (
                mean_probability_error
            ),
            "calibration_intercept": (
                intercept
            ),
            "calibration_slope": (
                slope
            ),
            "quantile_ece": (
                ece
            ),
            "n_bins": (
                DEFAULT_CALIBRATION_BINS
            ),
        }

        model_results[
            model_name
        ] = result

        summary_rows.append(
            {
                "model": model_name,
                **result,
            }
        )

    summary_table = pd.DataFrame(
        summary_rows
    )

    curve_table = pd.concat(
        curve_tables,
        ignore_index=True,
    )

    SUMMARY_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_table.to_csv(
        SUMMARY_TABLE_PATH,
        index=False,
    )

    curve_table.to_csv(
        CURVE_TABLE_PATH,
        index=False,
    )

    summary = {
        "phase": 7,
        "analysis": (
            "validation_calibration_diagnostics"
        ),
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "test_used": False,
            "identifiers_present": False,
        },
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": observed_hash,
        },
        "sample_counts": {
            "validation": int(
                len(predictions)
            ),
            "positive": int(
                target.sum()
            ),
            "negative": int(
                len(target)
                - target.sum()
            ),
        },
        "calibration_protocol": {
            "curve_strategy": (
                "quantile"
            ),
            "n_bins": (
                DEFAULT_CALIBRATION_BINS
            ),
            "ideal_intercept": 0.0,
            "ideal_slope": 1.0,
            "ideal_ece": 0.0,
        },
        "models": model_results,
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
        "\nCALIBRATION DIAGNOSTICS"
    )

    print("-" * 88)

    for (
        model_name,
        result,
    ) in model_results.items():
        print(
            f"\n{model_name}"
        )

        print(
            "  mean predicted probability : "
            f"{result['mean_predicted_probability']:.6f}"
        )

        print(
            "  mean probability - prevalence: "
            f"{result['mean_probability_minus_prevalence']:+.6f}"
        )

        print(
            "  calibration intercept      : "
            f"{result['calibration_intercept']:+.6f}"
        )

        print(
            "  calibration slope          : "
            f"{result['calibration_slope']:.6f}"
        )

        print(
            "  quantile ECE               : "
            f"{result['quantile_ece']:.6f}"
        )

    print(
        "\nSaved calibration summary:",
        SUMMARY_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved calibration curves :",
        CURVE_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved calibration JSON   :",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used                 : False"
    )


if __name__ == "__main__":
    main()