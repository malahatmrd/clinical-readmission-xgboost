from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.thresholds import (
    build_threshold_table,
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

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase8_validation_threshold_sweep.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_validation_threshold_sweep.json"
)

TARGET_COLUMN = "readmitted_30d"

PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_SELECTED_VARIANT = (
    "tuned_xgboost_sigmoid"
)

THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.50
THRESHOLD_STEP = 0.005

REFERENCE_THRESHOLDS = (
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
)


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


def build_threshold_grid() -> np.ndarray:
    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP / 2.0,
        THRESHOLD_STEP,
    )

    return np.round(
        thresholds,
        3,
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
            "Unexpected frozen Phase 7 variant."
        )

    required_columns = {
        TARGET_COLUMN,
        PROBABILITY_COLUMN,
    }

    missing_columns = sorted(
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    forbidden_identifiers = {
        "encounter_id",
        "patient_nbr",
        "source_row",
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
            "Prediction artifact SHA256 "
            "does not match Phase 7 summary."
        )

    target = predictions[
        TARGET_COLUMN
    ].to_numpy()

    probabilities = predictions[
        PROBABILITY_COLUMN
    ].to_numpy()

    thresholds = (
        build_threshold_grid()
    )

    threshold_table = (
        build_threshold_table(
            target,
            probabilities,
            thresholds,
        )
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    probability_quantiles = {
        str(quantile): float(
            np.quantile(
                probabilities,
                quantile,
            )
        )
        for quantile in (
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        )
    }

    output = {
        "phase": 8,
        "analysis": (
            "validation_threshold_sweep"
        ),
        "selected_model": (
            EXPECTED_SELECTED_VARIANT
        ),
        "threshold_grid": {
            "minimum": THRESHOLD_MIN,
            "maximum": THRESHOLD_MAX,
            "step": THRESHOLD_STEP,
            "count": int(
                len(thresholds)
            ),
        },
        "validation_sample": {
            "rows": int(
                len(target)
            ),
            "positives": int(
                target.sum()
            ),
            "negatives": int(
                len(target)
                - target.sum()
            ),
            "prevalence": float(
                target.mean()
            ),
        },
        "probability_distribution": {
            "minimum": float(
                probabilities.min()
            ),
            "maximum": float(
                probabilities.max()
            ),
            "mean": float(
                probabilities.mean()
            ),
            "quantiles": (
                probability_quantiles
            ),
        },
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": observed_hash,
        },
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "threshold_selected": False,
            "test_used": False,
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

    print("=" * 96)
    print("PHASE 8 VALIDATION THRESHOLD SWEEP")
    print("=" * 96)

    print(
        "\nValidation rows      :",
        len(target),
    )

    print(
        "Validation positives :",
        int(
            target.sum()
        ),
    )

    print(
        "Prevalence           :",
        f"{target.mean():.6f}",
    )

    print(
        "Selected model       :",
        EXPECTED_SELECTED_VARIANT,
    )

    print(
        "Mean probability     :",
        f"{probabilities.mean():.6f}",
    )

    print(
        "Probability range    :",
        (
            f"{probabilities.min():.6f}"
            " to "
            f"{probabilities.max():.6f}"
        ),
    )

    print(
        "Threshold count      :",
        len(thresholds),
    )

    print(
        "\nREFERENCE OPERATING POINTS"
    )

    print("-" * 96)

    display_columns = [
        "threshold",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "alerts_per_100",
        "number_needed_to_evaluate",
        "model_net_benefit",
    ]

    reference_rows = (
        threshold_table[
            threshold_table[
                "threshold"
            ].isin(
                REFERENCE_THRESHOLDS
            )
        ][
            display_columns
        ]
        .copy()
    )

    print(
        reference_rows.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nSaved sweep table :",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved summary     :",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nThreshold selected: False"
    )

    print(
        "Test used         : False"
    )


if __name__ == "__main__":
    main()