from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.bootstrap import (
    DEFAULT_BOOTSTRAP_RANDOM_STATE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_probability_metrics,
)
from clinical_readmission.evaluation.metrics import (
    PROBABILITY_METRIC_NAMES,
    calculate_probability_metrics,
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
    / "phase7_bootstrap_validation.json"
)

METRICS_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_validation_bootstrap_metrics.csv"
)

DIFFERENCE_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_validation_paired_differences.csv"
)

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

COMPARISONS = (
    (
        "tuned_xgboost",
        "logistic_regression",
    ),
    (
        "tuned_xgboost",
        "early_stopped_xgboost",
    ),
)

HIGHER_IS_BETTER = {
    "roc_auc": True,
    "average_precision": True,
    "brier_score": False,
    "log_loss": False,
}

TARGET_COLUMN = "readmitted_30d"


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


def summarize_distribution(
    estimate: float,
    values: np.ndarray,
) -> dict[str, float]:
    alpha = (
        1.0
        - DEFAULT_CONFIDENCE_LEVEL
    )

    return {
        "estimate": float(
            estimate
        ),
        "ci_lower": float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        "ci_upper": float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
        "bootstrap_standard_error": float(
            np.std(
                values,
                ddof=1,
            )
        ),
    }


def main() -> None:
    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    reproduction = load_json(
        REPRODUCTION_PATH
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
            "Prediction artifact is missing: "
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
            "Prediction artifact contains "
            "forbidden identifiers: "
            f"{present_identifiers}"
        )

    if predictions[
        list(required_columns)
    ].isna().any().any():
        raise ValueError(
            "Prediction artifact contains "
            "missing values."
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
            "does not match reproduction "
            "summary."
        )

    target = predictions[
        TARGET_COLUMN
    ].to_numpy()

    print("=" * 88)
    print("PHASE 7 VALIDATION BOOTSTRAP ANALYSIS")
    print("=" * 88)

    print(
        "\nValidation rows      :",
        len(predictions),
    )

    print(
        "Validation positives :",
        int(
            target.sum()
        ),
    )

    print(
        "Bootstrap resamples  :",
        DEFAULT_BOOTSTRAP_RESAMPLES,
    )

    print(
        "Bootstrap seed       :",
        DEFAULT_BOOTSTRAP_RANDOM_STATE,
    )

    print(
        "Confidence level     :",
        DEFAULT_CONFIDENCE_LEVEL,
    )

    print(
        "Prediction SHA256    :",
        observed_hash,
    )

    point_metrics = {}
    distributions = {}
    model_summaries = {}
    metric_rows = []

    print(
        "\nGenerating paired bootstrap "
        "distributions..."
    )

    for (
        model_name,
        probability_column,
    ) in MODEL_COLUMNS.items():
        print(
            f"  {model_name} ..."
        )

        probabilities = predictions[
            probability_column
        ].to_numpy()

        point_metrics[
            model_name
        ] = (
            calculate_probability_metrics(
                target,
                probabilities,
            )
        )

        distributions[
            model_name
        ] = (
            bootstrap_probability_metrics(
                target,
                probabilities,
                n_resamples=(
                    DEFAULT_BOOTSTRAP_RESAMPLES
                ),
                random_state=(
                    DEFAULT_BOOTSTRAP_RANDOM_STATE
                ),
            )
        )

        model_summaries[
            model_name
        ] = {}

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            result = (
                summarize_distribution(
                    point_metrics[
                        model_name
                    ][
                        metric
                    ],
                    distributions[
                        model_name
                    ][
                        metric
                    ],
                )
            )

            model_summaries[
                model_name
            ][
                metric
            ] = result

            metric_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    **result,
                }
            )

        print(
            "    complete"
        )

    comparison_summaries = {}
    difference_rows = []

    print(
        "\nCalculating paired model "
        "differences..."
    )

    for (
        model_a,
        model_b,
    ) in COMPARISONS:
        comparison_name = (
            f"{model_a}_minus_{model_b}"
        )

        comparison_summaries[
            comparison_name
        ] = {}

        print(
            f"  {model_a} - {model_b}"
        )

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            point_difference = (
                point_metrics[
                    model_a
                ][
                    metric
                ]
                - point_metrics[
                    model_b
                ][
                    metric
                ]
            )

            difference_distribution = (
                distributions[
                    model_a
                ][
                    metric
                ]
                - distributions[
                    model_b
                ][
                    metric
                ]
            )

            result = (
                summarize_distribution(
                    point_difference,
                    difference_distribution,
                )
            )

            ci_excludes_zero = bool(
                result["ci_lower"] > 0.0
                or result["ci_upper"] < 0.0
            )

            if HIGHER_IS_BETTER[
                metric
            ]:
                ci_favors_model_a = bool(
                    result["ci_lower"]
                    > 0.0
                )
            else:
                ci_favors_model_a = bool(
                    result["ci_upper"]
                    < 0.0
                )

            enriched_result = {
                **result,
                "higher_is_better": (
                    HIGHER_IS_BETTER[
                        metric
                    ]
                ),
                "ci_excludes_zero": (
                    ci_excludes_zero
                ),
                "ci_consistently_favors_model_a": (
                    ci_favors_model_a
                ),
            }

            comparison_summaries[
                comparison_name
            ][
                metric
            ] = enriched_result

            difference_rows.append(
                {
                    "comparison": (
                        comparison_name
                    ),
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    **enriched_result,
                }
            )

    metrics_table = pd.DataFrame(
        metric_rows
    )

    difference_table = pd.DataFrame(
        difference_rows
    )

    METRICS_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_table.to_csv(
        METRICS_TABLE_PATH,
        index=False,
    )

    difference_table.to_csv(
        DIFFERENCE_TABLE_PATH,
        index=False,
    )

    summary = {
        "phase": 7,
        "analysis": (
            "validation_stratified_paired_bootstrap"
        ),
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "test_used": False,
            "identifiers_present": False,
        },
        "bootstrap_protocol": {
            "method": (
                "stratified_bootstrap_with_replacement"
            ),
            "paired_across_models": True,
            "n_resamples": (
                DEFAULT_BOOTSTRAP_RESAMPLES
            ),
            "random_state": (
                DEFAULT_BOOTSTRAP_RANDOM_STATE
            ),
            "confidence_level": (
                DEFAULT_CONFIDENCE_LEVEL
            ),
            "interval_method": (
                "percentile"
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
        "model_metrics": (
            model_summaries
        ),
        "paired_differences": (
            comparison_summaries
        ),
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
        "\nMODEL METRICS WITH 95% CI"
    )
    print("-" * 88)

    for model_name in (
        MODEL_COLUMNS
    ):
        print(
            f"\n{model_name}"
        )

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            result = (
                model_summaries[
                    model_name
                ][
                    metric
                ]
            )

            print(
                f"  {metric:20} "
                f"{result['estimate']:.6f} "
                f"[{result['ci_lower']:.6f}, "
                f"{result['ci_upper']:.6f}]"
            )

    print(
        "\nPAIRED DIFFERENCES WITH 95% CI"
    )
    print("-" * 88)

    for comparison_name, results in (
        comparison_summaries.items()
    ):
        print(
            f"\n{comparison_name}"
        )

        for metric, result in (
            results.items()
        ):
            print(
                f"  {metric:20} "
                f"{result['estimate']:+.6f} "
                f"[{result['ci_lower']:+.6f}, "
                f"{result['ci_upper']:+.6f}]"
            )

    print(
        "\nSaved metric table :",
        METRICS_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved difference table:",
        DIFFERENCE_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved summary      :",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used          : False"
    )


if __name__ == "__main__":
    main()