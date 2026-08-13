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
    draw_stratified_bootstrap_indices,
)
from clinical_readmission.evaluation.thresholds import (
    calculate_net_benefit,
    calculate_threshold_metrics,
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

SCENARIO_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_operating_scenarios.json"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase8_operating_scenario_bootstrap.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_operating_scenario_bootstrap.json"
)

TARGET_COLUMN = "readmitted_30d"

PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = (
    "tuned_xgboost_sigmoid"
)

BOOTSTRAP_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "alerts_per_100",
    "number_needed_to_evaluate",
    "model_net_benefit",
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


def summarize_distribution(
    values: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float]:
    alpha = (
        1.0
        - confidence_level
    )

    return {
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


def bootstrap_fixed_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    point_metrics = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold,
        )
    )

    point_net_benefit = (
        calculate_net_benefit(
            target,
            probabilities,
            threshold,
        )
    )

    rng = np.random.default_rng(
        DEFAULT_BOOTSTRAP_RANDOM_STATE
    )

    distributions = {
        metric: np.empty(
            DEFAULT_BOOTSTRAP_RESAMPLES,
            dtype=float,
        )
        for metric in BOOTSTRAP_METRICS
    }

    for iteration in range(
        DEFAULT_BOOTSTRAP_RESAMPLES
    ):
        indices = (
            draw_stratified_bootstrap_indices(
                target,
                rng,
            )
        )

        bootstrap_target = (
            target[
                indices
            ]
        )

        bootstrap_probabilities = (
            probabilities[
                indices
            ]
        )

        metrics = (
            calculate_threshold_metrics(
                bootstrap_target,
                bootstrap_probabilities,
                threshold,
            )
        )

        net_benefit = (
            calculate_net_benefit(
                bootstrap_target,
                bootstrap_probabilities,
                threshold,
            )
        )

        for metric in (
            BOOTSTRAP_METRICS
        ):
            if (
                metric
                == "model_net_benefit"
            ):
                value = (
                    net_benefit[
                        metric
                    ]
                )
            else:
                value = (
                    metrics[
                        metric
                    ]
                )

            distributions[
                metric
            ][
                iteration
            ] = value

    summary = {}

    for metric in (
        BOOTSTRAP_METRICS
    ):
        if (
            metric
            == "model_net_benefit"
        ):
            estimate = (
                point_net_benefit[
                    metric
                ]
            )
        else:
            estimate = (
                point_metrics[
                    metric
                ]
            )

        summary[
            metric
        ] = {
            "estimate": float(
                estimate
            ),
            **summarize_distribution(
                distributions[
                    metric
                ],
                confidence_level=(
                    DEFAULT_CONFIDENCE_LEVEL
                ),
            ),
        }

    return summary


def main() -> None:
    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    candidate_summary = load_json(
        CANDIDATE_SUMMARY_PATH
    )

    scenario_summary = load_json(
        SCENARIO_PATH
    )

    if (
        scenario_summary[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected Phase 8 model."
        )

    if (
        scenario_summary[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
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
            "Prediction artifact SHA256 mismatch."
        )

    target = predictions[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    probabilities = predictions[
        PROBABILITY_COLUMN
    ].to_numpy(
        dtype=float
    )

    scenario_results = {}
    table_rows = []

    print("=" * 104)
    print(
        "PHASE 8 OPERATING-SCENARIO BOOTSTRAP"
    )
    print("=" * 104)

    print(
        "\nBootstrap resamples:",
        DEFAULT_BOOTSTRAP_RESAMPLES,
    )

    print(
        "Bootstrap seed     :",
        DEFAULT_BOOTSTRAP_RANDOM_STATE,
    )

    print(
        "Confidence level   :",
        DEFAULT_CONFIDENCE_LEVEL,
    )

    print(
        "\nGenerating fixed-threshold "
        "bootstrap intervals..."
    )

    for (
        scenario_name,
        scenario,
    ) in scenario_summary[
        "scenarios"
    ].items():
        threshold = float(
            scenario[
                "threshold"
            ]
        )

        print(
            f"  {scenario_name} "
            f"(threshold={threshold:.3f}) ..."
        )

        metrics = (
            bootstrap_fixed_threshold(
                target,
                probabilities,
                threshold,
            )
        )

        scenario_results[
            scenario_name
        ] = {
            "threshold": threshold,
            "metrics": metrics,
        }

        for metric in (
            BOOTSTRAP_METRICS
        ):
            result = (
                metrics[
                    metric
                ]
            )

            table_rows.append(
                {
                    "scenario": (
                        scenario_name
                    ),
                    "threshold": (
                        threshold
                    ),
                    "metric": (
                        metric
                    ),
                    "estimate": (
                        result[
                            "estimate"
                        ]
                    ),
                    "ci_lower": (
                        result[
                            "ci_lower"
                        ]
                    ),
                    "ci_upper": (
                        result[
                            "ci_upper"
                        ]
                    ),
                    "bootstrap_standard_error": (
                        result[
                            "bootstrap_standard_error"
                        ]
                    ),
                }
            )

        print(
            "    complete"
        )

    table = pd.DataFrame(
        table_rows
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
        "phase": 8,
        "analysis": (
            "fixed_operating_scenario_bootstrap"
        ),
        "selected_model": (
            EXPECTED_MODEL
        ),
        "bootstrap_protocol": {
            "method": (
                "stratified_bootstrap_with_replacement"
            ),
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
            "thresholds_reselected_per_resample": (
                False
            ),
        },
        "scenarios": (
            scenario_results
        ),
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": (
                observed_hash
            ),
        },
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "reference_threshold_selected": (
                False
            ),
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

    print(
        "\nOPERATING METRICS WITH 95% CI"
    )

    print("-" * 104)

    for (
        scenario_name,
        result,
    ) in scenario_results.items():
        print(
            f"\n{scenario_name}"
        )

        print(
            "  threshold:",
            f"{result['threshold']:.3f}",
        )

        for metric in (
            BOOTSTRAP_METRICS
        ):
            values = (
                result[
                    "metrics"
                ][
                    metric
                ]
            )

            print(
                f"  {metric:<28}"
                f"{values['estimate']:.4f} "
                f"[{values['ci_lower']:.4f}, "
                f"{values['ci_upper']:.4f}]"
            )

    print(
        "\nSaved table :",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved JSON  :",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nReference threshold selected: False"
    )

    print(
        "Test used                   : False"
    )


if __name__ == "__main__":
    main()