from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from clinical_readmission.data.validate import load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "IDS_mapping.csv"
)

TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "discharge_disposition_audit.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "discharge_disposition_audit.json"
)

SECTION_HEADERS = {
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
}


def load_discharge_mapping() -> dict[int, str]:
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Missing mapping file: {MAPPING_PATH}"
        )

    mapping: dict[int, str] = {}
    current_section: str | None = None

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            first_value = row[0].strip()

            if not first_value:
                continue

            if first_value in SECTION_HEADERS:
                current_section = first_value
                continue

            if current_section != "discharge_disposition_id":
                continue

            try:
                disposition_id = int(first_value)
            except ValueError:
                continue

            description = (
                row[1].strip()
                if len(row) > 1
                else ""
            )

            mapping[disposition_id] = description

    if not mapping:
        raise ValueError(
            "No discharge-disposition mappings were parsed."
        )

    return mapping


def is_terminal_disposition(
    description: str,
) -> bool:
    normalized = description.lower()

    return (
        "expired" in normalized
        or "hospice" in normalized
    )


def build_disposition_audit() -> tuple[
    pd.DataFrame,
    dict,
]:
    identifiers, features, target = load_raw_data()

    mapping = load_discharge_mapping()

    data = pd.DataFrame(
        {
            "encounter_id": identifiers["encounter_id"],
            "patient_nbr": identifiers["patient_nbr"],
            "discharge_disposition_id": features[
                "discharge_disposition_id"
            ],
            "readmitted": target["readmitted"],
        }
    )

    summary = (
        data
        .groupby(
            "discharge_disposition_id",
            dropna=False,
        )
        .agg(
            encounters=(
                "encounter_id",
                "size",
            ),
            unique_patients=(
                "patient_nbr",
                "nunique",
            ),
            positive_30d=(
                "readmitted",
                lambda values: int(
                    (values == "<30").sum()
                ),
            ),
        )
        .reset_index()
    )

    summary["positive_30d_pct"] = (
        summary["positive_30d"]
        / summary["encounters"]
        * 100
    )

    summary["description"] = (
        summary["discharge_disposition_id"]
        .map(mapping)
        .fillna("Unmapped")
    )

    summary["terminal_or_hospice"] = (
        summary["description"]
        .map(is_terminal_disposition)
    )

    summary = summary[
        [
            "discharge_disposition_id",
            "description",
            "encounters",
            "unique_patients",
            "positive_30d",
            "positive_30d_pct",
            "terminal_or_hospice",
        ]
    ].sort_values(
        "discharge_disposition_id"
    )

    terminal_rows = summary[
        summary["terminal_or_hospice"]
    ]

    terminal_ids = [
        int(value)
        for value in terminal_rows[
            "discharge_disposition_id"
        ].tolist()
    ]

    summary_json = {
        "raw_encounters": int(len(data)),
        "raw_unique_patients": int(
            data["patient_nbr"].nunique()
        ),
        "observed_disposition_ids": int(
            data[
                "discharge_disposition_id"
            ].nunique()
        ),
        "terminal_or_hospice_ids_observed": (
            terminal_ids
        ),
        "terminal_or_hospice_encounters": int(
            terminal_rows["encounters"].sum()
        ),
        "terminal_or_hospice_unique_patients": int(
            data.loc[
                data[
                    "discharge_disposition_id"
                ].isin(terminal_ids),
                "patient_nbr",
            ].nunique()
        ),
        "terminal_or_hospice_positive_30d": int(
            terminal_rows["positive_30d"].sum()
        ),
    }

    return summary, summary_json


def save_outputs(
    audit: pd.DataFrame,
    summary: dict,
) -> None:
    TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        TABLE_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    audit, summary = build_disposition_audit()

    save_outputs(
        audit,
        summary,
    )

    print("=" * 100)
    print("DISCHARGE DISPOSITION AUDIT")
    print("=" * 100)

    display = audit.copy()

    display["positive_30d_pct"] = (
        display["positive_30d_pct"]
        .map(lambda value: f"{value:.2f}%")
    )

    print(
        display.to_string(
            index=False,
        )
    )

    print("\nTERMINAL / HOSPICE SUMMARY")
    print("-" * 100)

    print(
        "Candidate IDs          :",
        summary[
            "terminal_or_hospice_ids_observed"
        ],
    )

    print(
        "Affected encounters    :",
        f"{summary['terminal_or_hospice_encounters']:,}",
    )

    print(
        "Affected patients      :",
        f"{summary['terminal_or_hospice_unique_patients']:,}",
    )

    print(
        "Positive <30 outcomes  :",
        f"{summary['terminal_or_hospice_positive_30d']:,}",
    )

    print(
        "\nTable saved to:",
        TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Summary saved to:",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nDischarge audit completed successfully."
    )


if __name__ == "__main__":
    main()