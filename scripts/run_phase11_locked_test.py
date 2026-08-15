from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.calibration import (
    DEFAULT_CALIBRATION_BINS,
    calculate_calibration_intercept_slope,
    calculate_quantile_ece,
)
from clinical_readmission.evaluation.calibration_models import (
    build_calibrated_classifier,
    build_tuned_pipeline,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
)
from clinical_readmission.evaluation.thresholds import (
    calculate_net_benefit,
    calculate_threshold_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_COHORT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "cohorts" / "primary.csv"
)
ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "primary_split_assignments.csv"
)
PRETEST_FREEZE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase11_pretest_freeze.json"
)
PHASE7_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)
PHASE10_SUBGROUP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_subgroup_validation.json"
)

TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase11_locked_test_probabilities.csv"
)
OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase11_locked_test_metrics.csv"
)
OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase11_locked_test_evaluation.json"
)

PRETEST_MANIFEST_COMMIT = (
    "29f72f77594ad9149c374e642683a22344d6f982"
)

TARGET_COLUMN = "readmitted_30d"
CALIBRATED_PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_TREE_COUNT = 155
EXPECTED_THRESHOLD = 0.105

EXPECTED_TEST_ROWS = 10496
EXPECTED_TEST_POSITIVES = 941
EXPECTED_TEST_NEGATIVES = 9555

PROBABILITY_REPRODUCTION_TOLERANCE = 1e-7
METRIC_REPRODUCTION_TOLERANCE = 1e-7

COMPARISON_METRICS = (
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
    "model_net_benefit",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
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


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
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


def load_partition(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    ids = assignments.loc[
        assignments["split"].eq(split_name),
        [
            "encounter_id",
            "patient_nbr",
            TARGET_COLUMN,
        ],
    ].copy()

    result = cohort.merge(
        ids,
        on=[
            "encounter_id",
            "patient_nbr",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_assignment",
        ),
    )

    assignment_target = f"{TARGET_COLUMN}_assignment"

    if not (
        result[TARGET_COLUMN]
        == result[assignment_target]
    ).all():
        raise ValueError(
            f"{split_name}: target mismatch "
            "between cohort and assignments."
        )

    return result.reset_index(drop=True)


def evaluate_dataset(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target = data[TARGET_COLUMN].to_numpy(dtype=int)

    probability_metrics = calculate_probability_metrics(
        target,
        probabilities,
    )

    threshold_metrics = calculate_threshold_metrics(
        target,
        probabilities,
        threshold,
    )

    net_benefit = calculate_net_benefit(
        target,
        probabilities,
        threshold,
    )

    intercept, slope = calculate_calibration_intercept_slope(
        target,
        probabilities,
    )

    ece = calculate_quantile_ece(
        target,
        probabilities,
        n_bins=DEFAULT_CALIBRATION_BINS,
    )

    prevalence = float(target.mean())
    mean_probability = float(probabilities.mean())

    return {
        "encounters": int(len(data)),
        "unique_patients": int(
            data["patient_nbr"].nunique()
        ),
        "positives": int(target.sum()),
        "negatives": int(
            len(target) - target.sum()
        ),
        "prevalence": prevalence,
        "mean_predicted_probability": mean_probability,
        "mean_probability_minus_prevalence": float(
            mean_probability - prevalence
        ),
        **probability_metrics,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "quantile_ece": float(ece),
        **threshold_metrics,
        "model_net_benefit": float(
            net_benefit["model_net_benefit"]
        ),
        "treat_all_net_benefit": float(
            net_benefit["treat_all_net_benefit"]
        ),
        "treat_none_net_benefit": float(
            net_benefit["treat_none_net_benefit"]
        ),
    }


def assert_reference_metrics(
    current: dict[str, float | int],
    reference: dict,
) -> None:
    metrics = (
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "alerts_per_100",
        "model_net_benefit",
    )

    failures = {}

    for metric in metrics:
        difference = float(
            current[metric] - reference[metric]
        )

        if abs(difference) > METRIC_REPRODUCTION_TOLERANCE:
            failures[metric] = difference

    if failures:
        details = ", ".join(
            f"{metric}={difference:+.12g}"
            for metric, difference in failures.items()
        )

        raise ValueError(
            "Validation metric reproduction failed: "
            f"{details}"
        )


def verify_pretest_state() -> tuple[dict, str]:
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

    worktree_status = run_git(
        "status",
        "--porcelain",
    )

    if branch != "main":
        raise ValueError(
            "Locked Test evaluation must run from main."
        )

    if head != origin_main:
        raise ValueError(
            "HEAD must match origin/main before Test access."
        )

    if worktree_status:
        raise ValueError(
            "Working tree must be clean before Test access."
        )

    if not git_is_ancestor(
        PRETEST_MANIFEST_COMMIT,
        head,
    ):
        raise ValueError(
            "Pre-Test freeze commit is not an ancestor "
            "of the evaluation code commit."
        )

    freeze = load_json(
        PRETEST_FREEZE_PATH
    )

    if freeze["phase"] != 11:
        raise ValueError(
            "Unexpected freeze manifest phase."
        )

    if freeze["stage"] != "pretest_freeze":
        raise ValueError(
            "Unexpected freeze manifest stage."
        )

    frozen_model = freeze["frozen_model"]

    if frozen_model["name"] != EXPECTED_MODEL:
        raise ValueError(
            "Unexpected frozen model."
        )

    if (
        int(frozen_model["tree_count"])
        != EXPECTED_TREE_COUNT
    ):
        raise ValueError(
            "Unexpected frozen tree count."
        )

    threshold = float(
        frozen_model["reference_threshold"]
    )

    if abs(
        threshold - EXPECTED_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected frozen threshold."
        )

    calibration = frozen_model["calibration"]

    if (
        calibration["method"] != "sigmoid"
        or calibration["fit_data"] != "train_only"
        or calibration["cv_splits"] != 5
        or calibration["cv_random_state"] != 48
        or calibration["ensemble"]
    ):
        raise ValueError(
            "Unexpected frozen calibration protocol."
        )

    locked_test = freeze["locked_test"]

    if (
        locked_test["encounters"]
        != EXPECTED_TEST_ROWS
        or locked_test["positive_30d"]
        != EXPECTED_TEST_POSITIVES
        or locked_test["negative_30d"]
        != EXPECTED_TEST_NEGATIVES
    ):
        raise ValueError(
            "Locked Test counts do not match "
            "the Pre-Test freeze."
        )

    if (
        locked_test["test_rows_read_by_this_script"]
        or locked_test["test_predictions_generated"]
        or locked_test["test_metrics_calculated"]
    ):
        raise ValueError(
            "Pre-Test freeze unexpectedly records "
            "prior Test access."
        )

    policy = freeze["test_access_policy"]

    if not (
        policy["development_complete"]
        and not policy["model_reselection_after_test"]
        and not policy["calibration_reselection_after_test"]
        and not policy["threshold_reselection_after_test"]
    ):
        raise ValueError(
            "Unexpected Test-access policy."
        )

    for relative_path, expected_hash in (
        freeze["source_file_sha256"].items()
    ):
        current_hash = file_sha256(
            PROJECT_ROOT / relative_path
        )

        if current_hash != expected_hash:
            raise ValueError(
                "Frozen source file changed after "
                f"Pre-Test freeze: {relative_path}"
            )

    return freeze, head


def ensure_no_prior_test_output() -> None:
    existing = [
        path
        for path in (
            TEST_PREDICTIONS_PATH,
            OUTPUT_TABLE_PATH,
            OUTPUT_JSON_PATH,
        )
        if path.exists()
    ]

    if existing:
        paths = ", ".join(
            str(path.relative_to(PROJECT_ROOT))
            for path in existing
        )

        raise FileExistsError(
            "Locked Test outputs already exist. "
            "Refusing accidental re-evaluation: "
            f"{paths}"
        )


def main() -> None:
    ensure_no_prior_test_output()

    freeze, evaluation_commit = verify_pretest_state()

    phase10_subgroup = load_json(
        PHASE10_SUBGROUP_PATH
    )

    recorded_predictions = pd.read_csv(
        PHASE7_PREDICTIONS_PATH
    )

    expected_phase7_hash = (
        phase10_subgroup[
            "prediction_artifact"
        ][
            "sha256"
        ]
    )

    observed_phase7_hash = file_sha256(
        PHASE7_PREDICTIONS_PATH
    )

    if observed_phase7_hash != expected_phase7_hash:
        raise ValueError(
            "Phase 7 calibrated prediction artifact "
            "SHA256 mismatch."
        )

    print("=" * 100)
    print("PHASE 11 LOCKED TEST EVALUATION")
    print("=" * 100)

    print(
        "\nPre-Test freeze commit :",
        PRETEST_MANIFEST_COMMIT,
    )
    print(
        "Evaluation code commit:",
        evaluation_commit,
    )
    print(
        "Frozen model           :",
        EXPECTED_MODEL,
    )
    print(
        "Frozen tree count      :",
        EXPECTED_TREE_COUNT,
    )
    print(
        "Frozen threshold       :",
        f"{EXPECTED_THRESHOLD:.3f}",
    )

    print(
        "\nStatic freeze audit    : PASS"
    )
    print(
        "Test metrics inspected : False"
    )

    cohort = pd.read_csv(
        PRIMARY_COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    train = load_partition(
        cohort,
        assignments,
        "train",
    )

    validation = load_partition(
        cohort,
        assignments,
        "validation",
    )

    test = load_partition(
        cohort,
        assignments,
        "test",
    )

    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            "Unexpected Test row count."
        )

    test_positives = int(
        test[TARGET_COLUMN].sum()
    )

    test_negatives = int(
        len(test) - test_positives
    )

    if (
        test_positives != EXPECTED_TEST_POSITIVES
        or test_negatives != EXPECTED_TEST_NEGATIVES
    ):
        raise ValueError(
            "Unexpected Test outcome counts."
        )

    train_patients = set(
        train["patient_nbr"]
    )
    validation_patients = set(
        validation["patient_nbr"]
    )
    test_patients = set(
        test["patient_nbr"]
    )

    if (
        train_patients & validation_patients
        or train_patients & test_patients
        or validation_patients & test_patients
    ):
        raise ValueError(
            "Patient overlap detected between splits."
        )

    y_train = train[TARGET_COLUMN]

    y_validation = validation[
        TARGET_COLUMN
    ].to_numpy(dtype=int)

    y_test = test[
        TARGET_COLUMN
    ].to_numpy(dtype=int)

    recorded_validation_target = (
        recorded_predictions[
            TARGET_COLUMN
        ].to_numpy(dtype=int)
    )

    if not np.array_equal(
        y_validation,
        recorded_validation_target,
    ):
        raise ValueError(
            "Validation target order does not match "
            "the recorded Phase 7 predictions."
        )

    frozen_model = freeze[
        "frozen_model"
    ]

    tuned_parameters = frozen_model[
        "hyperparameters"
    ]

    tree_count = int(
        frozen_model["tree_count"]
    )

    threshold = float(
        frozen_model["reference_threshold"]
    )

    print(
        "\nTrain rows             :",
        len(train),
    )
    print(
        "Validation rows        :",
        len(validation),
    )
    print(
        "Locked Test rows       :",
        len(test),
    )
    print(
        "Locked Test positives  :",
        test_positives,
    )

    print(
        "\nRebuilding frozen Train-only "
        "sigmoid-calibrated model..."
    )

    estimator = build_tuned_pipeline(
        tuned_parameters,
        n_estimators=tree_count,
    )

    model = build_calibrated_classifier(
        estimator,
        "sigmoid",
    )

    model.fit(
        train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            validation
        )[:, 1]
    )

    recorded_validation_probabilities = (
        recorded_predictions[
            CALIBRATED_PROBABILITY_COLUMN
        ].to_numpy(dtype=float)
    )

    maximum_validation_error = float(
        np.max(
            np.abs(
                validation_probabilities
                - recorded_validation_probabilities
            )
        )
    )

    print(
        "Maximum Validation probability "
        "reproduction error:",
        f"{maximum_validation_error:.12g}",
    )

    if (
        maximum_validation_error
        > PROBABILITY_REPRODUCTION_TOLERANCE
    ):
        raise ValueError(
            "Frozen calibrated model reproduction failed."
        )

    validation_metrics = evaluate_dataset(
        validation,
        validation_probabilities,
        threshold,
    )

    assert_reference_metrics(
        validation_metrics,
        phase10_subgroup[
            "overall_reference"
        ],
    )

    print(
        "Validation reproduction : PASS"
    )

    print(
        "\nFrozen model verified."
    )
    print(
        "Beginning one-time locked Test prediction..."
    )

    test_probabilities = (
        model.predict_proba(
            test
        )[:, 1]
    )

    test_metrics = evaluate_dataset(
        test,
        test_probabilities,
        threshold,
    )

    test_minus_validation = {
        metric: float(
            test_metrics[metric]
            - validation_metrics[metric]
        )
        for metric in COMPARISON_METRICS
    }

    predictions = pd.DataFrame(
        {
            "test_row": range(
                len(test)
            ),
            TARGET_COLUMN: y_test,
            CALIBRATED_PROBABILITY_COLUMN: (
                test_probabilities
            ),
        }
    )

    TEST_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    prediction_hash = file_sha256(
        TEST_PREDICTIONS_PATH
    )

    table_row = {
        "dataset": "locked_test",
        "model": EXPECTED_MODEL,
        "threshold": threshold,
        **test_metrics,
    }

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [table_row]
    ).to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 11,
        "stage": "locked_test_evaluation",
        "evaluation_role": (
            "one_time_final_locked_test_evaluation"
        ),
        "pretest_freeze": {
            "manifest_path": str(
                PRETEST_FREEZE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "manifest_sha256": file_sha256(
                PRETEST_FREEZE_PATH
            ),
            "manifest_commit": (
                PRETEST_MANIFEST_COMMIT
            ),
            "frozen_development_commit": (
                freeze[
                    "freeze_state"
                ][
                    "git_commit"
                ]
            ),
        },
        "evaluation_code_commit": (
            evaluation_commit
        ),
        "frozen_configuration": {
            "model": EXPECTED_MODEL,
            "tree_count": tree_count,
            "hyperparameters": tuned_parameters,
            "calibration_method": "sigmoid",
            "calibration_fit_split": "train_only",
            "calibration_cv_splits": 5,
            "calibration_cv_random_state": 48,
            "reference_threshold": threshold,
            "threshold_scenario": (
                frozen_model[
                    "threshold_scenario"
                ]
            ),
        },
        "sample_counts": {
            "train": int(len(train)),
            "validation": int(
                len(validation)
            ),
            "test": int(len(test)),
            "test_positive": (
                test_positives
            ),
            "test_negative": (
                test_negatives
            ),
        },
        "reproduction_audit": {
            "maximum_validation_probability_error": (
                maximum_validation_error
            ),
            "probability_tolerance": (
                PROBABILITY_REPRODUCTION_TOLERANCE
            ),
            "validation_metric_tolerance": (
                METRIC_REPRODUCTION_TOLERANCE
            ),
            "validation_probability_reproduction_passed": (
                True
            ),
            "validation_metric_reproduction_passed": (
                True
            ),
        },
        "validation_reference": (
            validation_metrics
        ),
        "locked_test_metrics": (
            test_metrics
        ),
        "test_minus_validation": (
            test_minus_validation
        ),
        "prediction_artifact": {
            "path": str(
                TEST_PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": prediction_hash,
            "rows": int(
                len(predictions)
            ),
            "encounter_id_saved": False,
            "patient_nbr_saved": False,
            "intended_for_git": False,
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "data_policy": {
            "model_fit_split": "train",
            "calibration_fit_split": (
                "train_only"
            ),
            "evaluation_split": "test",
            "test_used": True,
            "model_reselected": False,
            "calibration_reselected": False,
            "threshold_reselected": False,
            "feature_reselected": False,
            "test_metrics_used_for_selection": False,
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
        file.write("\n")

    print(
        "\nLOCKED TEST RESULTS"
    )
    print("-" * 100)

    display_metrics = (
        "prevalence",
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
        "model_net_benefit",
    )

    for metric in display_metrics:
        print(
            f"{metric:34}: "
            f"{test_metrics[metric]:.9f}"
        )

    print(
        "\nConfusion matrix"
    )
    print(
        "  TP:",
        test_metrics["true_positive"],
    )
    print(
        "  FP:",
        test_metrics["false_positive"],
    )
    print(
        "  TN:",
        test_metrics["true_negative"],
    )
    print(
        "  FN:",
        test_metrics["false_negative"],
    )

    print(
        "\nTest prediction SHA256:",
        prediction_hash,
    )

    print(
        "\nSaved local predictions:",
        TEST_PREDICTIONS_PATH.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Saved metric table     :",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Saved summary          :",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nModel reselected       : False"
    )
    print(
        "Calibration reselected : False"
    )
    print(
        "Threshold reselected   : False"
    )
    print(
        "Locked Test used       : True"
    )


if __name__ == "__main__":
    main()