from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IDENTIFIERS_PATH = PROJECT_ROOT / "data" / "raw" / "identifiers.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "raw" / "features.csv"
TARGET_PATH = PROJECT_ROOT / "data" / "raw" / "target.csv"

PROVENANCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "data_provenance.json"
)

EXPECTED_ROWS = 101_766
EXPECTED_FEATURES = 47

EXPECTED_ID_COLUMNS = {
    "encounter_id",
    "patient_nbr",
}


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_files = (
        IDENTIFIERS_PATH,
        FEATURES_PATH,
        TARGET_PATH,
    )

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing raw file: {file_path}"
            )

    identifiers = pd.read_csv(
        IDENTIFIERS_PATH,
        low_memory=False,
    )

    features = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
        dtype={"payer_code": "string"},
    )

    target = pd.read_csv(
        TARGET_PATH,
        low_memory=False,
    )

    return identifiers, features, target


def validate_shapes(
    identifiers: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
    row_counts = {
        "identifiers": len(identifiers),
        "features": len(features),
        "target": len(target),
    }

    if len(set(row_counts.values())) != 1:
        raise ValueError(
            f"Row-count mismatch: {row_counts}"
        )

    if len(features) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected row count: {len(features)}"
        )

    if features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Unexpected feature count: {features.shape[1]}"
        )

    if identifiers.shape[1] != 2:
        raise ValueError(
            "Expected 2 identifier columns, "
            f"got {identifiers.shape[1]}"
        )

    if target.shape[1] != 1:
        raise ValueError(
            "Expected 1 target column, "
            f"got {target.shape[1]}"
        )


def validate_columns(
    identifiers: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
    missing_ids = (
        EXPECTED_ID_COLUMNS
        - set(identifiers.columns)
    )

    if missing_ids:
        raise ValueError(
            f"Missing ID columns: {sorted(missing_ids)}"
        )

    if "readmitted" not in target.columns:
        raise ValueError(
            "Missing target column: readmitted"
        )


def validate_identifier_integrity(
    identifiers: pd.DataFrame,
) -> None:
    if identifiers["encounter_id"].isna().any():
        raise ValueError(
            "Missing encounter_id values detected."
        )

    if identifiers["patient_nbr"].isna().any():
        raise ValueError(
            "Missing patient_nbr values detected."
        )

    duplicate_encounters = (
        identifiers["encounter_id"]
        .duplicated()
        .sum()
    )

    if duplicate_encounters:
        raise ValueError(
            "Duplicate encounter_id values detected: "
            f"{duplicate_encounters}"
        )


def validate_target(
    target: pd.DataFrame,
) -> None:
    expected_labels = {
        "NO",
        ">30",
        "<30",
    }

    actual_labels = set(
        target["readmitted"]
        .dropna()
        .unique()
    )

    if actual_labels != expected_labels:
        raise ValueError(
            "Unexpected target labels. "
            f"Expected {sorted(expected_labels)}, "
            f"got {sorted(actual_labels)}"
        )


def build_provenance(
    identifiers: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> dict:
    counts = (
        target["readmitted"]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "dataset": {
            "name": (
                "Diabetes 130-US Hospitals "
                "for Years 1999-2008"
            ),
            "uci_dataset_id": 296,
        },
        "snapshot": {
            "created_at_utc": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "identifiers_file": (
                "data/raw/identifiers.csv"
            ),
            "features_file": (
                "data/raw/features.csv"
            ),
            "target_file": (
                "data/raw/target.csv"
            ),
            "identifiers_sha256": (
                calculate_sha256(
                    IDENTIFIERS_PATH
                )
            ),
            "features_sha256": (
                calculate_sha256(
                    FEATURES_PATH
                )
            ),
            "target_sha256": (
                calculate_sha256(
                    TARGET_PATH
                )
            ),
        },
        "shape": {
            "rows": len(features),
            "identifier_columns": (
                identifiers.shape[1]
            ),
            "features": features.shape[1],
            "target_columns": target.shape[1],
        },
        "identifiers": {
            "unique_patients": int(
                identifiers[
                    "patient_nbr"
                ].nunique()
            ),
            "unique_encounters": int(
                identifiers[
                    "encounter_id"
                ].nunique()
            ),
        },
        "target_distribution": {
            str(key): int(value)
            for key, value in counts.items()
        },
    }


def save_provenance(
    provenance: dict,
) -> None:
    PROVENANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PROVENANCE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            provenance,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    print("=" * 72)
    print("RAW DATA VALIDATION & PROVENANCE")
    print("=" * 72)

    identifiers, features, target = (
        load_raw_data()
    )

    validate_shapes(
        identifiers,
        features,
        target,
    )

    validate_columns(
        identifiers,
        target,
    )

    validate_identifier_integrity(
        identifiers
    )

    validate_target(target)

    provenance = build_provenance(
        identifiers,
        features,
        target,
    )

    save_provenance(provenance)

    print(f"Rows               : {len(features):,}")
    print(f"Features           : {features.shape[1]}")
    print(
        "Identifier columns : "
        f"{identifiers.shape[1]}"
    )
    print(
        "Target columns     : "
        f"{target.shape[1]}"
    )

    print(
        "Unique patients    : "
        f"{identifiers['patient_nbr'].nunique():,}"
    )

    print(
        "Unique encounters  : "
        f"{identifiers['encounter_id'].nunique():,}"
    )

    print("\nTarget distribution")
    print(
        target["readmitted"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSHA-256")
    print(
        "Identifiers:",
        provenance["snapshot"][
            "identifiers_sha256"
        ],
    )
    print(
        "Features   :",
        provenance["snapshot"][
            "features_sha256"
        ],
    )
    print(
        "Target     :",
        provenance["snapshot"][
            "target_sha256"
        ],
    )

    print(
        "\nProvenance saved to:",
        PROVENANCE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nValidation completed successfully."
    )


if __name__ == "__main__":
    main()