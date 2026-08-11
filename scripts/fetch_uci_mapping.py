from __future__ import annotations

import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "reference"
)

OUTPUT_PATH = (
    REFERENCE_DIR
    / "IDS_mapping.csv"
)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
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


def main() -> None:
    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("UCI REFERENCE MAPPING ACQUISITION")
    print("=" * 72)

    print("Downloading official UCI archive...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        archive_path = (
            temp_path
            / "uci_296.zip"
        )

        urllib.request.urlretrieve(
            DATASET_URL,
            archive_path,
        )

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if Path(name).name
                == "IDS_mapping.csv"
            ]

            if len(candidates) != 1:
                raise RuntimeError(
                    "Could not uniquely locate "
                    "IDS_mapping.csv in UCI archive. "
                    f"Found: {candidates}"
                )

            mapping_bytes = archive.read(
                candidates[0]
            )

    OUTPUT_PATH.write_bytes(
        mapping_bytes
    )

    print(
        "\nSaved:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Size:",
        OUTPUT_PATH.stat().st_size,
        "bytes",
    )

    print(
        "SHA-256:",
        calculate_sha256(
            OUTPUT_PATH
        ),
    )

    print(
        "\nMapping acquisition completed successfully."
    )


if __name__ == "__main__":
    main()