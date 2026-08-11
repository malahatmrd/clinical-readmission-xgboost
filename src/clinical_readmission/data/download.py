from __future__ import annotations

import hashlib
from pathlib import Path

from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_ID = 296


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def download_dataset() -> tuple[Path, Path, Path]:
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("UCI DATA ACQUISITION")
    print("=" * 72)

    print(f"Dataset ID : {DATASET_ID}")
    print("Downloading dataset...")

    dataset = fetch_ucirepo(id=DATASET_ID)

    identifiers = dataset.data.ids
    features = dataset.data.features
    target = dataset.data.targets

    if identifiers is None:
        raise ValueError(
            "Dataset did not return identifier columns."
        )

    if features is None:
        raise ValueError(
            "Dataset did not return feature columns."
        )

    if target is None:
        raise ValueError(
            "Dataset did not return target columns."
        )

    identifiers_path = RAW_DATA_DIR / "identifiers.csv"
    features_path = RAW_DATA_DIR / "features.csv"
    target_path = RAW_DATA_DIR / "target.csv"

    identifiers.to_csv(
        identifiers_path,
        index=False,
    )

    features.to_csv(
        features_path,
        index=False,
    )

    target.to_csv(
        target_path,
        index=False,
    )

    print("\nDataset saved successfully.")

    print(f"Identifiers : {identifiers_path}")
    print(f"Features    : {features_path}")
    print(f"Target      : {target_path}")

    print("\nShapes")
    print(f"Identifiers : {identifiers.shape}")
    print(f"Features    : {features.shape}")
    print(f"Target      : {target.shape}")

    print("\nIdentifier columns")
    print(identifiers.columns.tolist())

    print("\nSHA-256")
    print(
        "Identifiers:",
        calculate_sha256(identifiers_path),
    )
    print(
        "Features   :",
        calculate_sha256(features_path),
    )
    print(
        "Target     :",
        calculate_sha256(target_path),
    )

    return (
        identifiers_path,
        features_path,
        target_path,
    )


if __name__ == "__main__":
    download_dataset()