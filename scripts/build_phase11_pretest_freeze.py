from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase11_pretest_freeze.json"
)

SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "primary_split_manifest.json"
)

TUNED_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_tuned_validation.json"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase7_calibration_candidates_validation.json"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_threshold_selection.json"
)

PHASE10_SUBGROUP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_subgroup_validation.json"
)

PHASE10_ROBUSTNESS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_cluster_bootstrap.json"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_TREE_COUNT = 155
EXPECTED_THRESHOLD = 0.105

EXPECTED_COHORT_SHA256 = (
    "f1de76f15beed99c154e81c078b99b3840b95aae01e18354"
    "b570363d8cc8ddd8"
)

EXPECTED_ASSIGNMENT_SHA256 = (
    "e8201f2f411995f47fb5b525061762dca8cd65831578f12f"
    "a9ce2eff05bc9482"
)

EXPECTED_TEST_ROWS = 10496
EXPECTED_TEST_POSITIVES = 941
EXPECTED_TEST_NEGATIVES = 9555

SOURCE_FILES = (
    "pyproject.toml",
    "requirements-lock.txt",
    "configs/data.yaml",
    "docs/split_protocol.md",
    "artifacts/metrics/primary_split_manifest.json",
    "artifacts/metrics/xgboost_tuned_validation.json",
    (
        "artifacts/metrics/"
        "phase7_calibration_candidates_validation.json"
    ),
    "artifacts/metrics/phase8_threshold_selection.json",
    "artifacts/metrics/phase10_subgroup_validation.json",
    "artifacts/metrics/phase10_cluster_bootstrap.json",
    "src/clinical_readmission/features/preprocessing.py",
    "src/clinical_readmission/models/xgboost_tuned.py",
    (
        "src/clinical_readmission/evaluation/"
        "calibration_models.py"
    ),
    "src/clinical_readmission/evaluation/metrics.py",
    "src/clinical_readmission/evaluation/thresholds.py",
    "scripts/build_phase11_pretest_freeze.py",
)


def run_git(
    *arguments: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


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


def get_package_version(
    package: str,
) -> str:
    return importlib.metadata.version(
        package
    )


def find_split_summary(
    split_manifest: dict,
    split_name: str,
) -> dict:
    matches = [
        row
        for row in split_manifest[
            "summary"
        ]
        if row[
            "split"
        ]
        == split_name
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"Expected exactly one {split_name} "
            "split summary."
        )

    return matches[
        0
    ]


def main() -> None:
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
            "Pre-Test freeze must be created "
            "from branch main."
        )

    if head != origin_main:
        raise ValueError(
            "Local HEAD and origin/main must "
            "match before Test access."
        )

    if worktree_status:
        raise ValueError(
            "Working tree must be clean before "
            "creating the Pre-Test freeze manifest."
        )

    split_manifest = load_json(
        SPLIT_MANIFEST_PATH
    )

    tuned_model = load_json(
        TUNED_MODEL_PATH
    )

    calibration = load_json(
        CALIBRATION_PATH
    )

    threshold_selection = load_json(
        THRESHOLD_PATH
    )

    phase10_subgroup = load_json(
        PHASE10_SUBGROUP_PATH
    )

    phase10_robustness = load_json(
        PHASE10_ROBUSTNESS_PATH
    )

    if (
        split_manifest[
            "cohort_sha256"
        ]
        != EXPECTED_COHORT_SHA256
    ):
        raise ValueError(
            "Unexpected primary cohort SHA256."
        )

    if (
        split_manifest[
            "assignment_sha256"
        ]
        != EXPECTED_ASSIGNMENT_SHA256
    ):
        raise ValueError(
            "Unexpected split assignment SHA256."
        )

    test_summary = find_split_summary(
        split_manifest,
        "test",
    )

    if (
        test_summary[
            "encounters"
        ]
        != EXPECTED_TEST_ROWS
    ):
        raise ValueError(
            "Unexpected locked Test row count."
        )

    if (
        test_summary[
            "positive_30d"
        ]
        != EXPECTED_TEST_POSITIVES
    ):
        raise ValueError(
            "Unexpected locked Test positive count."
        )

    if (
        test_summary[
            "negative_30d"
        ]
        != EXPECTED_TEST_NEGATIVES
    ):
        raise ValueError(
            "Unexpected locked Test negative count."
        )

    if (
        threshold_selection[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    threshold = float(
        threshold_selection[
            "reference_threshold"
        ]
    )

    if abs(
        threshold
        - EXPECTED_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected frozen threshold."
        )

    threshold_policy = (
        threshold_selection[
            "data_policy"
        ]
    )

    if not (
        threshold_policy[
            "model_frozen"
        ]
        and threshold_policy[
            "calibration_frozen"
        ]
        and threshold_policy[
            "threshold_frozen"
        ]
    ):
        raise ValueError(
            "Phase 8 freeze flags are not all True."
        )

    if threshold_policy[
        "test_used"
    ]:
        raise ValueError(
            "Phase 8 unexpectedly used Test data."
        )

    model_configuration = (
        calibration[
            "model_configuration"
        ]
    )

    tree_count = int(
        model_configuration[
            "tuned_tree_count"
        ]
    )

    if (
        tree_count
        != EXPECTED_TREE_COUNT
    ):
        raise ValueError(
            "Unexpected frozen tree count."
        )

    tuned_parameters = (
        model_configuration[
            "tuned_parameters"
        ]
    )

    if (
        tuned_parameters
        != tuned_model[
            "selected_hyperparameters"
        ]
    ):
        raise ValueError(
            "Frozen tuned parameters do not match "
            "the Phase 6 model artifact."
        )

    calibration_protocol = (
        calibration[
            "calibration_protocol"
        ]
    )

    if (
        calibration_protocol[
            "fit_data"
        ]
        != "train_only"
        or calibration_protocol[
            "cv_splits"
        ]
        != 5
        or calibration_protocol[
            "cv_random_state"
        ]
        != 48
        or calibration_protocol[
            "ensemble"
        ]
    ):
        raise ValueError(
            "Unexpected calibration protocol."
        )

    selected_candidate = (
        calibration[
            "candidates"
        ][
            EXPECTED_MODEL
        ]
    )

    if (
        selected_candidate[
            "calibration_method"
        ]
        != "sigmoid"
    ):
        raise ValueError(
            "Unexpected calibration method."
        )

    if (
        phase10_subgroup[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Phase 10 subgroup analysis used Test."
        )

    if (
        phase10_robustness[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Phase 10 robustness analysis used Test."
        )

    source_hashes = {}

    for relative_path in SOURCE_FILES:
        path = (
            PROJECT_ROOT
            / relative_path
        )

        if not path.exists():
            raise FileNotFoundError(
                relative_path
            )

        source_hashes[
            relative_path
        ] = file_sha256(
            path
        )

    output = {
        "phase": 11,
        "stage": "pretest_freeze",
        "freeze_state": {
            "git_branch": branch,
            "git_commit": head,
            "origin_main_commit": (
                origin_main
            ),
            "head_matches_origin_main": (
                head
                == origin_main
            ),
            "worktree_clean_before_manifest": (
                True
            ),
        },
        "frozen_model": {
            "name": (
                EXPECTED_MODEL
            ),
            "base_model_family": (
                "tuned_xgboost"
            ),
            "tree_count": (
                tree_count
            ),
            "hyperparameters": (
                tuned_parameters
            ),
            "calibration": {
                "method": (
                    "sigmoid"
                ),
                "fit_data": (
                    "train_only"
                ),
                "cv_strategy": (
                    calibration_protocol[
                        "cv_strategy"
                    ]
                ),
                "cv_splits": (
                    calibration_protocol[
                        "cv_splits"
                    ]
                ),
                "cv_random_state": (
                    calibration_protocol[
                        "cv_random_state"
                    ]
                ),
                "ensemble": (
                    calibration_protocol[
                        "ensemble"
                    ]
                ),
            },
            "reference_threshold": (
                threshold
            ),
            "threshold_scenario": (
                threshold_selection[
                    "selected_scenario"
                ]
            ),
        },
        "locked_test": {
            "split": "test",
            "encounters": (
                test_summary[
                    "encounters"
                ]
            ),
            "positive_30d": (
                test_summary[
                    "positive_30d"
                ]
            ),
            "negative_30d": (
                test_summary[
                    "negative_30d"
                ]
            ),
            "assignment_sha256": (
                split_manifest[
                    "assignment_sha256"
                ]
            ),
            "source_cohort_sha256": (
                split_manifest[
                    "cohort_sha256"
                ]
            ),
            "row_level_assignment_read_by_this_script": (
                False
            ),
            "test_rows_read_by_this_script": (
                False
            ),
            "test_predictions_generated": (
                False
            ),
            "test_metrics_calculated": (
                False
            ),
        },
        "development_history": {
            "phase7_test_used": (
                calibration[
                    "selection_policy"
                ][
                    "test_used"
                ]
            ),
            "phase8_test_used": (
                threshold_policy[
                    "test_used"
                ]
            ),
            "phase10_subgroup_test_used": (
                phase10_subgroup[
                    "data_policy"
                ][
                    "test_used"
                ]
            ),
            "phase10_robustness_test_used": (
                phase10_robustness[
                    "data_policy"
                ][
                    "test_used"
                ]
            ),
        },
        "environment": {
            "python": (
                sys.version.split()[0]
            ),
            "numpy": get_package_version(
                "numpy"
            ),
            "pandas": get_package_version(
                "pandas"
            ),
            "scikit_learn": (
                get_package_version(
                    "scikit-learn"
                )
            ),
            "xgboost": get_package_version(
                "xgboost"
            ),
            "shap": get_package_version(
                "shap"
            ),
        },
        "source_file_sha256": (
            source_hashes
        ),
        "test_access_policy": {
            "development_complete": True,
            "model_reselection_after_test": (
                False
            ),
            "calibration_reselection_after_test": (
                False
            ),
            "threshold_reselection_after_test": (
                False
            ),
            "test_role": (
                "one_time_final_locked_evaluation"
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

        file.write(
            "\n"
        )

    print(
        "=" * 96
    )

    print(
        "PHASE 11 PRE-TEST FREEZE"
    )

    print(
        "=" * 96
    )

    print(
        "\nGit commit        :",
        head,
    )

    print(
        "HEAD == origin/main:",
        True,
    )

    print(
        "Worktree pre-freeze:",
        "clean",
    )

    print(
        "\nFrozen model      :",
        EXPECTED_MODEL,
    )

    print(
        "Tree count        :",
        tree_count,
    )

    print(
        "Calibration       :",
        "sigmoid, Train-only 5-fold CV",
    )

    print(
        "Reference threshold:",
        f"{threshold:.3f}",
    )

    print(
        "\nLocked Test rows  :",
        test_summary[
            "encounters"
        ],
    )

    print(
        "Locked positives  :",
        test_summary[
            "positive_30d"
        ],
    )

    print(
        "Test rows read    : False"
    )

    print(
        "Test predictions  : False"
    )

    print(
        "Test metrics      : False"
    )

    print(
        "\nSaved manifest:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()