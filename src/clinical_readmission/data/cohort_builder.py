from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clinical_readmission.data.cohort_audit import (
    build_combined_data,
    get_terminal_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

COHORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
)

FLOW_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "cohort_flow.json"
)

FLOW_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "cohort_flow.csv"
)


def add_binary_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result["readmitted_30d"] = (
        result["readmitted"]
        .eq("<30")
        .astype("int8")
    )

    return result


def summarize_cohort(
    name: str,
    data: pd.DataFrame,
) -> dict:
    positive = int(
        data["readmitted_30d"].sum()
    )

    rows = int(
        len(data)
    )

    return {
        "cohort": name,
        "encounters": rows,
        "unique_patients": int(
            data["patient_nbr"].nunique()
        ),
        "positive_30d": positive,
        "negative_30d": int(
            rows - positive
        ),
        "positive_30d_pct": (
            positive / rows * 100
            if rows
            else 0.0
        ),
    }


def build_primary_cohort(
    data: pd.DataFrame,
    terminal_ids: set[int],
) -> pd.DataFrame:
    first_observed = (
        data
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )

    primary = first_observed.loc[
        ~first_observed[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    return add_binary_target(
        primary
    )


def build_sensitivity_cohort(
    data: pd.DataFrame,
    terminal_ids: set[int],
) -> pd.DataFrame:
    eligible = data.loc[
        ~data[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    first_eligible = (
        eligible
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )

    return add_binary_target(
        first_eligible
    )


def build_all_eligible_cohort(
    data: pd.DataFrame,
    terminal_ids: set[int],
) -> pd.DataFrame:
    eligible = data.loc[
        ~data[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    return add_binary_target(
        eligible
    )


def validate_cohort(
    name: str,
    data: pd.DataFrame,
    terminal_ids: set[int],
    require_unique_patient: bool,
) -> None:
    if data.empty:
        raise ValueError(
            f"{name}: cohort is empty."
        )

    if data["encounter_id"].duplicated().any():
        raise ValueError(
            f"{name}: duplicate encounter_id detected."
        )

    if data["patient_nbr"].isna().any():
        raise ValueError(
            f"{name}: missing patient_nbr detected."
        )

    if data[
        "discharge_disposition_id"
    ].isin(terminal_ids).any():
        raise ValueError(
            f"{name}: terminal/hospice encounter detected."
        )

    if require_unique_patient:
        if data[
            "patient_nbr"
        ].duplicated().any():
            raise ValueError(
                f"{name}: repeated patient detected."
            )

    expected_binary = (
        data["readmitted"]
        .eq("<30")
        .astype("int8")
    )

    if not data[
        "readmitted_30d"
    ].equals(expected_binary):
        raise ValueError(
            f"{name}: binary target mismatch."
        )


def save_cohort(
    name: str,
    data: pd.DataFrame,
) -> Path:
    COHORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        COHORT_DIR
        / f"{name}.csv"
    )

    data.to_csv(
        path,
        index=False,
    )

    return path


def main() -> None:
    data = build_combined_data()

    terminal_ids = get_terminal_ids()

    primary = build_primary_cohort(
        data,
        terminal_ids,
    )

    sensitivity = (
        build_sensitivity_cohort(
            data,
            terminal_ids,
        )
    )

    all_eligible = (
        build_all_eligible_cohort(
            data,
            terminal_ids,
        )
    )

    validate_cohort(
        "primary",
        primary,
        terminal_ids,
        require_unique_patient=True,
    )

    validate_cohort(
        "sensitivity_first_eligible",
        sensitivity,
        terminal_ids,
        require_unique_patient=True,
    )

    validate_cohort(
        "all_eligible_encounters",
        all_eligible,
        terminal_ids,
        require_unique_patient=False,
    )

    primary_path = save_cohort(
        "primary",
        primary,
    )

    sensitivity_path = save_cohort(
        "sensitivity_first_eligible",
        sensitivity,
    )

    all_eligible_path = save_cohort(
        "all_eligible_encounters",
        all_eligible,
    )

    summaries = [
        summarize_cohort(
            "raw",
            add_binary_target(
                data
            ),
        ),
        summarize_cohort(
            "primary",
            primary,
        ),
        summarize_cohort(
            "sensitivity_first_eligible",
            sensitivity,
        ),
        summarize_cohort(
            "all_eligible_encounters",
            all_eligible,
        ),
    ]

    flow = {
        "terminal_or_hospice_ids": sorted(
            terminal_ids
        ),
        "primary_definition": (
            "first observed encounter per patient, "
            "then exclude terminal/hospice"
        ),
        "sensitivity_definition": (
            "exclude terminal/hospice, then retain "
            "first eligible observed encounter per patient"
        ),
        "all_eligible_definition": (
            "all encounters excluding terminal/hospice"
        ),
        "target_definition": {
            "column": "readmitted_30d",
            "positive": "readmitted == <30",
            "negative": (
                "readmitted == NO or readmitted == >30"
            ),
        },
        "cohorts": summaries,
    }

    FLOW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FLOW_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FLOW_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            flow,
            file,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(
        summaries
    ).to_csv(
        FLOW_TABLE_PATH,
        index=False,
    )

    print("=" * 88)
    print("CLINICAL COHORT BUILDER")
    print("=" * 88)

    display = pd.DataFrame(
        summaries
    )

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
        "\n"
        + display.to_string(
            index=False
        )
    )

    print("\nTERMINAL / HOSPICE IDS")
    print("-" * 88)

    print(
        sorted(
            terminal_ids
        )
    )

    print("\nCOHORT FILES")
    print("-" * 88)

    print(
        "Primary     :",
        primary_path.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Sensitivity :",
        sensitivity_path.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "All eligible:",
        all_eligible_path.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nFlow JSON:",
        FLOW_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Flow table:",
        FLOW_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nCohort construction completed successfully."
    )


if __name__ == "__main__":
    main()