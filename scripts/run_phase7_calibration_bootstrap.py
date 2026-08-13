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
    / "phase7_calibration_candidate_probabilities.csv"
)

CANDIDATE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_bootstrap.json"
)

OUTPUT_METRICS_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_calibrated_variant_bootstrap_metrics.csv"
)

OUTPUT_DIFFERENCE_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase7_calibrated_variant_paired_differences.csv"
)

TARGET_COLUMN = "readmitted_30d"

VARIANT_COLUMNS = {
    "tuned_xgboost": (
        "tuned_xgboost_probability"
    ),
    "tuned_xgboost_sigmoid": (
        "tuned_xgboost_sigmoid_probability"
    ),
    "tuned_xgboost_isotonic": (
        "tuned_xgboost_isotonic_probability"
    ),
}

COMPARISONS = (
    (
        "tuned_xgboost_sigmoid",
        "tuned_xgboost",
    ),
    (
        "tuned_xgboost_isotonic",
        "tuned_xgboost",
    ),
    (
        "tuned_xgboost_isotonic",
        "tuned_xgboost_sigmoid",
    ),
)

HIGHER_IS_BETTER = {
    "roc_auc": True,
    "average_precision": True,
    "brier_score": False,
    "log_loss": False,
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

    candidate_summary = load_json(
        CANDIDATE_SUMMARY_PATH
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
        *VARIANT_COLUMNS.values(),
    }

    missing = sorted(
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Calibration prediction artifact "
            f"is missing columns: {missing}"
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

    target = predictions[
        TARGET_COLUMN
    ].to_numpy()

    print("=" * 88)
    print("PHASE 7 CALIBRATION CANDIDATE PAIRED BOOTSTRAP")
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
        "Candidate SHA256     :",
        observed_hash,
    )

    point_metrics = {}
    distributions = {}
    metric_summaries = {}
    metric_rows = []

    print(
        "\nGenerating paired bootstrap "
        "distributions..."
    )

    for (
        variant_name,
        probability_column,
    ) in VARIANT_COLUMNS.items():
        print(
            f"  {variant_name} ..."
        )

        probabilities = predictions[
            probability_column
        ].to_numpy()

        point_metrics[
            variant_name
        ] = (
            calculate_probability_metrics(
                target,
                probabilities,
            )
        )

        distributions[
            variant_name
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

        metric_summaries[
            variant_name
        ] = {}

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            result = summarize_distribution(
                point_metrics[
                    variant_name
                ][
                    metric
                ],
                distributions[
                    variant_name
                ][
                    metric
                ],
            )

            metric_summaries[
                variant_name
            ][
                metric
            ] = result

            metric_rows.append(
                {
                    "variant": (
                        variant_name
                    ),
                    "metric": metric,
                    **result,
                }
            )

        print(
            "    complete"
        )

    difference_summaries = {}
    difference_rows = []

    print(
        "\nCalculating paired differences..."
    )

    for (
        variant_a,
        variant_b,
    ) in COMPARISONS:
        comparison_name = (
            f"{variant_a}_minus_{variant_b}"
        )

        difference_summaries[
            comparison_name
        ] = {}

        print(
            f"  {variant_a} - {variant_b}"
        )

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            point_difference = (
                point_metrics[
                    variant_a
                ][
                    metric
                ]
                - point_metrics[
                    variant_b
                ][
                    metric
                ]
            )

            difference_distribution = (
                distributions[
                    variant_a
                ][
                    metric
                ]
                - distributions[
                    variant_b
                ][
                    metric
                ]
            )

            result = summarize_distribution(
                point_difference,
                difference_distribution,
            )

            ci_excludes_zero = bool(
                result[
                    "ci_lower"
                ] > 0.0
                or result[
                    "ci_upper"
                ] < 0.0
            )

            if HIGHER_IS_BETTER[
                metric
            ]:
                ci_favors_a = bool(
                    result[
                        "ci_lower"
                    ] > 0.0
                )

                ci_favors_b = bool(
                    result[
                        "ci_upper"
                    ] < 0.0
                )
            else:
                ci_favors_a = bool(
                    result[
                        "ci_upper"
                    ] < 0.0
                )

                ci_favors_b = bool(
                    result[
                        "ci_lower"
                    ] > 0.0
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
                "ci_consistently_favors_variant_a": (
                    ci_favors_a
                ),
                "ci_consistently_favors_variant_b": (
                    ci_favors_b
                ),
            }

            difference_summaries[
                comparison_name
            ][
                metric
            ] = enriched_result

            difference_rows.append(
                {
                    "comparison": (
                        comparison_name
                    ),
                    "variant_a": (
                        variant_a
                    ),
                    "variant_b": (
                        variant_b
                    ),
                    "metric": metric,
                    **enriched_result,
                }
            )

    metric_table = pd.DataFrame(
        metric_rows
    )

    difference_table = pd.DataFrame(
        difference_rows
    )

    OUTPUT_METRICS_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_table.to_csv(
        OUTPUT_METRICS_TABLE_PATH,
        index=False,
    )

    difference_table.to_csv(
        OUTPUT_DIFFERENCE_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 7,
        "analysis": (
            "calibration_candidate_paired_bootstrap"
        ),
        "selection_context": {
            "base_family": (
                "tuned_xgboost"
            ),
            "primary_calibration_metric": (
                "brier_score"
            ),
            "secondary_calibration_metric": (
                "log_loss"
            ),
            "discrimination_safeguards": [
                "roc_auc",
                "average_precision",
            ],
            "decision_deferred_until_bootstrap_review": (
                True
            ),
        },
        "bootstrap_protocol": {
            "method": (
                "stratified_bootstrap_with_replacement"
            ),
            "paired_across_variants": True,
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
        "variant_metrics": (
            metric_summaries
        ),
        "paired_differences": (
            difference_summaries
        ),
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
        "\nVARIANT METRICS WITH 95% CI"
    )

    print("-" * 88)

    for variant_name in (
        VARIANT_COLUMNS
    ):
        print(
            f"\n{variant_name}"
        )

        for metric in (
            PROBABILITY_METRIC_NAMES
        ):
            result = (
                metric_summaries[
                    variant_name
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

    for (
        comparison_name,
        results,
    ) in difference_summaries.items():
        print(
            f"\n{comparison_name}"
        )

        for (
            metric,
            result,
        ) in results.items():
            print(
                f"  {metric:20} "
                f"{result['estimate']:+.6f} "
                f"[{result['ci_lower']:+.6f}, "
                f"{result['ci_upper']:+.6f}]"
            )

    print(
        "\nSaved metric table     :",
        OUTPUT_METRICS_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved difference table :",
        OUTPUT_DIFFERENCE_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved bootstrap summary:",
        OUTPUT_SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used              : False"
    )


if __name__ == "__main__":
    main()