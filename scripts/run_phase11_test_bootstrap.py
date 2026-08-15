from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.bootstrap import (
    DEFAULT_BOOTSTRAP_RANDOM_STATE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    draw_stratified_bootstrap_indices,
)
from clinical_readmission.evaluation.calibration import (
    DEFAULT_CALIBRATION_BINS,
    calculate_calibration_intercept_slope,
    calculate_quantile_ece,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
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
    / "phase11_locked_test_probabilities.csv"
)

EVALUATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase11_locked_test_evaluation.json"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase11_locked_test_bootstrap.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase11_locked_test_bootstrap.json"
)

LOCKED_TEST_EVALUATION_COMMIT = (
    "3551c215dadc65930e11865ca062ff18e1f107b6"
)

TARGET_COLUMN = "readmitted_30d"

PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_THRESHOLD = 0.105
EXPECTED_ROWS = 10496
EXPECTED_POSITIVES = 941
EXPECTED_NEGATIVES = 9555

POINT_REPRODUCTION_TOLERANCE = 1e-10

BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "quantile_ece",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "alerts_per_100",
    "number_needed_to_evaluate",
    "model_net_benefit",
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def run_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


def verify_repository_state() -> str:
    branch = run_git(
        "branch",
        "--show-current",
    )

    head = run_git(
        "rev-parse",
        "HEAD",
    )

    origin_main = run_git(
        "rev-parse",
        "origin/main",
    )

    status = run_git(
        "status",
        "--porcelain",
    )

    if branch != "main":
        raise ValueError(
            "Bootstrap must run from main."
        )

    if head != origin_main:
        raise ValueError(
            "HEAD must match origin/main."
        )

    if status:
        raise ValueError(
            "Working tree must be clean."
        )

    if not git_is_ancestor(
        LOCKED_TEST_EVALUATION_COMMIT,
        head,
    ):
        raise ValueError(
            "Locked Test evaluation commit "
            "is not an ancestor of HEAD."
        )

    return head


def ensure_no_prior_output() -> None:
    existing = [
        path
        for path in (
            OUTPUT_TABLE_PATH,
            OUTPUT_JSON_PATH,
        )
        if path.exists()
    ]

    if existing:
        names = ", ".join(
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
            for path in existing
        )

        raise FileExistsError(
            "Phase 11 bootstrap outputs "
            f"already exist: {names}"
        )


def evaluate_sample(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    probability_metrics = (
        calculate_probability_metrics(
            target,
            probabilities,
        )
    )

    threshold_metrics = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold,
        )
    )

    net_benefit = calculate_net_benefit(
        target,
        probabilities,
        threshold,
    )

    intercept, slope = (
        calculate_calibration_intercept_slope(
            target,
            probabilities,
        )
    )

    ece = calculate_quantile_ece(
        target,
        probabilities,
        n_bins=DEFAULT_CALIBRATION_BINS,
    )

    return {
        **probability_metrics,
        "calibration_intercept": float(
            intercept
        ),
        "calibration_slope": float(
            slope
        ),
        "quantile_ece": float(ece),
        "sensitivity": float(
            threshold_metrics[
                "sensitivity"
            ]
        ),
        "specificity": float(
            threshold_metrics[
                "specificity"
            ]
        ),
        "ppv": float(
            threshold_metrics["ppv"]
        ),
        "npv": float(
            threshold_metrics["npv"]
        ),
        "f1": float(
            threshold_metrics["f1"]
        ),
        "balanced_accuracy": float(
            threshold_metrics[
                "balanced_accuracy"
            ]
        ),
        "alerts_per_100": float(
            threshold_metrics[
                "alerts_per_100"
            ]
        ),
        "number_needed_to_evaluate": float(
            threshold_metrics[
                "number_needed_to_evaluate"
            ]
        ),
        "model_net_benefit": float(
            net_benefit[
                "model_net_benefit"
            ]
        ),
    }


def verify_point_estimates(
    point_metrics: dict[str, float],
    recorded_metrics: dict,
) -> None:
    failures = {}

    for metric in BOOTSTRAP_METRICS:
        difference = float(
            point_metrics[metric]
            - recorded_metrics[metric]
        )

        if (
            abs(difference)
            > POINT_REPRODUCTION_TOLERANCE
        ):
            failures[metric] = difference

    if failures:
        details = ", ".join(
            (
                f"{metric}="
                f"{difference:+.12g}"
            )
            for (
                metric,
                difference,
            ) in failures.items()
        )

        raise ValueError(
            "Locked Test point-metric "
            "reproduction failed: "
            f"{details}"
        )


def main() -> None:
    ensure_no_prior_output()

    bootstrap_commit = (
        verify_repository_state()
    )

    evaluation = load_json(
        EVALUATION_PATH
    )

    if (
        evaluation["phase"] != 11
        or evaluation["stage"]
        != "locked_test_evaluation"
    ):
        raise ValueError(
            "Unexpected locked Test "
            "evaluation artifact."
        )

    policy = evaluation[
        "data_policy"
    ]

    if not policy["test_used"]:
        raise ValueError(
            "Locked Test evaluation "
            "artifact does not record "
            "Test use."
        )

    if (
        policy["model_reselected"]
        or policy[
            "calibration_reselected"
        ]
        or policy[
            "threshold_reselected"
        ]
        or policy["feature_reselected"]
        or policy[
            "test_metrics_used_for_selection"
        ]
    ):
        raise ValueError(
            "Selection policy changed "
            "after Test evaluation."
        )

    configuration = evaluation[
        "frozen_configuration"
    ]

    if (
        configuration["model"]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    threshold = float(
        configuration[
            "reference_threshold"
        ]
    )

    if (
        abs(
            threshold
            - EXPECTED_THRESHOLD
        )
        > 1e-12
    ):
        raise ValueError(
            "Unexpected frozen threshold."
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    expected_columns = [
        "test_row",
        TARGET_COLUMN,
        PROBABILITY_COLUMN,
    ]

    if (
        predictions.columns.tolist()
        != expected_columns
    ):
        raise ValueError(
            "Unexpected locked Test "
            "prediction columns."
        )

    observed_hash = file_sha256(
        PREDICTIONS_PATH
    )

    expected_hash = evaluation[
        "prediction_artifact"
    ][
        "sha256"
    ]

    if observed_hash != expected_hash:
        raise ValueError(
            "Locked Test prediction "
            "SHA256 mismatch."
        )

    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected prediction row count."
        )

    if not np.array_equal(
        predictions[
            "test_row"
        ].to_numpy(),
        np.arange(
            EXPECTED_ROWS
        ),
    ):
        raise ValueError(
            "Unexpected test_row sequence."
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

    positives = int(
        target.sum()
    )

    negatives = int(
        len(target)
        - positives
    )

    if (
        positives
        != EXPECTED_POSITIVES
        or negatives
        != EXPECTED_NEGATIVES
    ):
        raise ValueError(
            "Locked Test class counts "
            "do not match recorded counts."
        )

    point_metrics = evaluate_sample(
        target,
        probabilities,
        threshold,
    )

    verify_point_estimates(
        point_metrics,
        evaluation[
            "locked_test_metrics"
        ],
    )

    print("=" * 100)
    print(
        "PHASE 11 LOCKED TEST BOOTSTRAP"
    )
    print("=" * 100)

    print(
        "\nBootstrap code commit :",
        bootstrap_commit,
    )

    print(
        "Prediction SHA256     :",
        observed_hash,
    )

    print(
        "Rows                  :",
        len(target),
    )

    print(
        "Positives             :",
        positives,
    )

    print(
        "Threshold             :",
        f"{threshold:.3f}",
    )

    print(
        "Point reproduction    : PASS"
    )

    print(
        "\nBootstrap resamples   :",
        DEFAULT_BOOTSTRAP_RESAMPLES,
    )

    print(
        "Random seed           :",
        DEFAULT_BOOTSTRAP_RANDOM_STATE,
    )

    print(
        "Confidence level      :",
        DEFAULT_CONFIDENCE_LEVEL,
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

        sample_metrics = (
            evaluate_sample(
                target[indices],
                probabilities[indices],
                threshold,
            )
        )

        for metric in (
            BOOTSTRAP_METRICS
        ):
            distributions[
                metric
            ][
                iteration
            ] = sample_metrics[
                metric
            ]

        if (
            iteration + 1
        ) % 250 == 0:
            print(
                "Completed resamples:",
                iteration + 1,
            )

    alpha = (
        1.0
        - DEFAULT_CONFIDENCE_LEVEL
    )

    lower_quantile = (
        alpha / 2.0
    )

    upper_quantile = (
        1.0
        - alpha / 2.0
    )

    rows = []
    summary = {}

    for metric in (
        BOOTSTRAP_METRICS
    ):
        values = distributions[
            metric
        ]

        result = {
            "estimate": float(
                point_metrics[
                    metric
                ]
            ),
            "ci_lower": float(
                np.quantile(
                    values,
                    lower_quantile,
                )
            ),
            "ci_upper": float(
                np.quantile(
                    values,
                    upper_quantile,
                )
            ),
            "bootstrap_standard_error": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        }

        summary[metric] = result

        rows.append(
            {
                "metric": metric,
                **result,
            }
        )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 11,
        "stage": (
            "locked_test_bootstrap"
        ),
        "analysis_role": (
            "uncertainty_characterization_"
            "after_final_locked_test"
        ),
        "source_evaluation": {
            "path": str(
                EVALUATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "commit": (
                LOCKED_TEST_EVALUATION_COMMIT
            ),
        },
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": observed_hash,
            "rows": int(
                len(predictions)
            ),
            "committed_to_git": False,
            "identifiers_saved": False,
        },
        "frozen_configuration": {
            "model": EXPECTED_MODEL,
            "threshold": threshold,
            "model_reselected": False,
            "calibration_reselected": False,
            "threshold_reselected": False,
        },
        "bootstrap_protocol": {
            "method": (
                "stratified_bootstrap_"
                "with_replacement"
            ),
            "unit": (
                "primary_test_encounter_"
                "one_per_patient"
            ),
            "prevalence_preserved": True,
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
        "sample_counts": {
            "test": int(
                len(target)
            ),
            "positive": positives,
            "negative": negatives,
        },
        "point_estimate_reproduction": {
            "passed": True,
            "tolerance": (
                POINT_REPRODUCTION_TOLERANCE
            ),
        },
        "metrics": summary,
        "data_policy": {
            "model_refit": False,
            "new_test_predictions": False,
            "test_prediction_artifact_reused": True,
            "model_reselected": False,
            "calibration_reselected": False,
            "threshold_reselected": False,
            "test_metrics_used_for_selection": False,
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
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
        file.write("\n")

    print(
        "\nLOCKED TEST BOOTSTRAP 95% CI"
    )

    print("-" * 100)

    for metric in (
        BOOTSTRAP_METRICS
    ):
        result = summary[
            metric
        ]

        print(
            f"{metric:34}: "
            f"{result['estimate']:.6f} "
            f"[{result['ci_lower']:.6f}, "
            f"{result['ci_upper']:.6f}]"
        )

    print(
        "\nModel refit             : False"
    )

    print(
        "New Test predictions    : False"
    )

    print(
        "Threshold reselected     : False"
    )

    print(
        "\nSaved table:",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved summary:",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()