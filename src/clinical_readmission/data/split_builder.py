from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "data.yaml"
)

PRIMARY_COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
    / "primary.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

ASSIGNMENT_PATH = (
    SPLIT_DIR
    / "primary_split_assignments.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "primary_split_summary.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "primary_split_manifest.json"
)


def calculate_sha256(
    file_path: Path,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_split_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return {
        "random_seed": int(
            config["project"]["random_seed"]
        ),
        "group_column": str(
            config["split"]["group_column"]
        ),
        "train_size": float(
            config["split"]["train_size"]
        ),
        "validation_size": float(
            config["split"]["validation_size"]
        ),
        "test_size": float(
            config["split"]["test_size"]
        ),
    }


def load_primary_cohort() -> pd.DataFrame:
    if not PRIMARY_COHORT_PATH.exists():
        raise FileNotFoundError(
            "Primary cohort not found. "
            "Run scripts/build_cohorts.py first."
        )

    return pd.read_csv(
        PRIMARY_COHORT_PATH,
        low_memory=False,
    )


def validate_split_sizes(
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    total = (
        train_size
        + validation_size
        + test_size
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train/validation/test fractions "
            f"must sum to 1.0, got {total}."
        )

    for name, value in {
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
    }.items():
        if not 0 < value < 1:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )


def build_split_assignments(
    data: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> pd.DataFrame:
    validate_split_sizes(
        train_size,
        validation_size,
        test_size,
    )

    required_columns = {
        "encounter_id",
        "patient_nbr",
        "readmitted_30d",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data["patient_nbr"].duplicated().any():
        raise ValueError(
            "Primary cohort contains repeated patients."
        )

    if data["encounter_id"].duplicated().any():
        raise ValueError(
            "Primary cohort contains duplicate encounters."
        )

    target = data["readmitted_30d"]

    first_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        test_size=(
            validation_size
            + test_size
        ),
        random_state=random_seed,
    )

    train_indices, holdout_indices = next(
        first_split.split(
            data,
            target,
        )
    )

    holdout = data.iloc[
        holdout_indices
    ]

    holdout_target = target.iloc[
        holdout_indices
    ]

    relative_test_size = (
        test_size
        / (
            validation_size
            + test_size
        )
    )

    second_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=random_seed + 1,
    )

    validation_relative, test_relative = next(
        second_split.split(
            holdout,
            holdout_target,
        )
    )

    validation_indices = (
        holdout_indices[
            validation_relative
        ]
    )

    test_indices = (
        holdout_indices[
            test_relative
        ]
    )

    assignments = data[
        [
            "encounter_id",
            "patient_nbr",
            "readmitted_30d",
        ]
    ].copy()

    assignments["split"] = ""

    assignments.loc[
        assignments.index[
            train_indices
        ],
        "split",
    ] = "train"

    assignments.loc[
        assignments.index[
            validation_indices
        ],
        "split",
    ] = "validation"

    assignments.loc[
        assignments.index[
            test_indices
        ],
        "split",
    ] = "test"

    return assignments


def validate_assignments(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    if len(assignments) != len(data):
        raise ValueError(
            "Split assignment row count does not "
            "match cohort row count."
        )

    if assignments[
        "encounter_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate encounters in split assignments."
        )

    if assignments[
        "patient_nbr"
    ].duplicated().any():
        raise ValueError(
            "Patient appears more than once in "
            "primary split assignments."
        )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        assignments["split"].unique()
    )

    if actual_splits != expected_splits:
        raise ValueError(
            "Unexpected split labels: "
            f"{sorted(actual_splits)}"
        )

    if (
        assignments["split"]
        .eq("")
        .any()
    ):
        raise ValueError(
            "Unassigned rows detected."
        )

    patient_sets = {
        split_name: set(
            assignments.loc[
                assignments["split"]
                == split_name,
                "patient_nbr",
            ]
        )
        for split_name in expected_splits
    }

    if (
        patient_sets["train"]
        & patient_sets["validation"]
    ):
        raise ValueError(
            "Patient overlap between train "
            "and validation."
        )

    if (
        patient_sets["train"]
        & patient_sets["test"]
    ):
        raise ValueError(
            "Patient overlap between train and test."
        )

    if (
        patient_sets["validation"]
        & patient_sets["test"]
    ):
        raise ValueError(
            "Patient overlap between validation "
            "and test."
        )

    expected_target = (
        data.set_index("encounter_id")[
            "readmitted_30d"
        ]
    )

    observed_target = (
        assignments.set_index(
            "encounter_id"
        )["readmitted_30d"]
    )

    observed_target = observed_target.loc[
        expected_target.index
    ]

    if not expected_target.equals(
        observed_target
    ):
        raise ValueError(
            "Target mismatch in split assignments."
        )


def build_split_summary(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        subset = assignments.loc[
            assignments["split"]
            == split_name
        ]

        positives = int(
            subset[
                "readmitted_30d"
            ].sum()
        )

        rows = int(
            len(subset)
        )

        records.append(
            {
                "split": split_name,
                "encounters": rows,
                "unique_patients": int(
                    subset[
                        "patient_nbr"
                    ].nunique()
                ),
                "positive_30d": positives,
                "negative_30d": int(
                    rows - positives
                ),
                "positive_30d_pct": float(
                    positives
                    / rows
                    * 100
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def save_outputs(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict,
) -> None:
    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        ASSIGNMENT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    manifest = {
        "cohort": "primary",
        "cohort_file": (
            "data/interim/cohorts/primary.csv"
        ),
        "cohort_sha256": calculate_sha256(
            PRIMARY_COHORT_PATH
        ),
        "assignment_file": (
            "data/processed/splits/"
            "primary_split_assignments.csv"
        ),
        "assignment_sha256": calculate_sha256(
            ASSIGNMENT_PATH
        ),
        "target_column": "readmitted_30d",
        "group_column": config[
            "group_column"
        ],
        "random_seed": config[
            "random_seed"
        ],
        "second_stage_seed": (
            config["random_seed"]
            + 1
        ),
        "split_fractions": {
            "train": config[
                "train_size"
            ],
            "validation": config[
                "validation_size"
            ],
            "test": config[
                "test_size"
            ],
        },
        "sklearn_version": (
            sklearn.__version__
        ),
        "test_set_policy": (
            "Locked after creation. "
            "Not used for preprocessing decisions, "
            "feature selection, hyperparameter tuning, "
            "calibration fitting, or threshold selection."
        ),
        "summary": (
            summary.to_dict(
                orient="records"
            )
        ),
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    config = load_split_config()

    data = load_primary_cohort()

    assignments = build_split_assignments(
        data=data,
        train_size=config[
            "train_size"
        ],
        validation_size=config[
            "validation_size"
        ],
        test_size=config[
            "test_size"
        ],
        random_seed=config[
            "random_seed"
        ],
    )

    validate_assignments(
        data,
        assignments,
    )

    summary = build_split_summary(
        assignments
    )

    save_outputs(
        assignments,
        summary,
        config,
    )

    print("=" * 88)
    print("PRIMARY COHORT SPLIT BUILDER")
    print("=" * 88)

    print(
        f"\nPrimary cohort rows : "
        f"{len(data):,}"
    )

    print(
        f"Random seed         : "
        f"{config['random_seed']}"
    )

    print(
        f"Second-stage seed   : "
        f"{config['random_seed'] + 1}"
    )

    print("\nSPLIT SUMMARY")
    print("-" * 88)

    display = summary.copy()

    display[
        "positive_30d_pct"
    ] = (
        display[
            "positive_30d_pct"
        ]
        .map(
            lambda value: f"{value:.2f}%"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print("\nLOCKED TEST SET")
    print("-" * 88)

    print(
        "The test split must not be used for "
        "training, tuning, calibration, "
        "feature selection, or threshold selection."
    )

    print("\nOUTPUTS")
    print("-" * 88)

    print(
        "Assignments:",
        ASSIGNMENT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Summary    :",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Manifest   :",
        MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nSplit construction completed successfully."
    )


if __name__ == "__main__":
    main()