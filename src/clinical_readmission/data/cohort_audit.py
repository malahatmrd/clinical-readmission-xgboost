from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clinical_readmission.data.disposition_audit import (
    is_terminal_disposition,
    load_discharge_mapping,
)
from clinical_readmission.data.validate import load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "cohort_rule_audit.json"
)

TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "cohort_rule_audit.csv"
)


def build_combined_data() -> pd.DataFrame:
    identifiers, features, target = load_raw_data()

    data = pd.concat(
        [
            identifiers.reset_index(drop=True),
            features.reset_index(drop=True),
            target.reset_index(drop=True),
        ],
        axis=1,
    )

    data["source_row"] = range(len(data))

    return data


def get_terminal_ids() -> set[int]:
    mapping = load_discharge_mapping()

    return {
        disposition_id
        for disposition_id, description in mapping.items()
        if is_terminal_disposition(description)
    }


def summarize_cohort(
    name: str,
    data: pd.DataFrame,
) -> dict:
    positive_count = int(
        (data["readmitted"] == "<30").sum()
    )

    rows = int(len(data))

    return {
        "strategy": name,
        "encounters": rows,
        "unique_patients": int(
            data["patient_nbr"].nunique()
        ),
        "positive_30d": positive_count,
        "positive_30d_pct": (
            positive_count / rows * 100
            if rows
            else 0.0
        ),
    }


def main() -> None:
    data = build_combined_data()
    terminal_ids = get_terminal_ids()

    print("=" * 84)
    print("COHORT RULE AUDIT")
    print("=" * 84)

    print(
        "\nTerminal / hospice IDs:",
        sorted(terminal_ids),
    )

    # ---------------------------------------------------------
    # Strategy 1:
    # first occurrence in original UCI row order
    # ---------------------------------------------------------

    source_first = (
        data
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )

    # ---------------------------------------------------------
    # Strategy 2:
    # minimum encounter_id per patient
    # ---------------------------------------------------------

    min_id_first = (
        data
        .sort_values(
            [
                "patient_nbr",
                "encounter_id",
            ]
        )
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )

    comparison = (
        source_first[
            [
                "patient_nbr",
                "encounter_id",
            ]
        ]
        .merge(
            min_id_first[
                [
                    "patient_nbr",
                    "encounter_id",
                ]
            ],
            on="patient_nbr",
            suffixes=(
                "_source_first",
                "_min_id",
            ),
            validate="one_to_one",
        )
    )

    mismatch_mask = (
        comparison[
            "encounter_id_source_first"
        ]
        != comparison[
            "encounter_id_min_id"
        ]
    )

    mismatch_count = int(
        mismatch_mask.sum()
    )

    # ---------------------------------------------------------
    # Paper-like ordering:
    # first encounter per patient, then terminal exclusion
    # ---------------------------------------------------------

    paper_like = source_first.loc[
        ~source_first[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    # ---------------------------------------------------------
    # Alternative:
    # remove terminal encounters first,
    # then retain first eligible occurrence per patient
    # ---------------------------------------------------------

    eligible_all = data.loc[
        ~data[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    eligible_then_first = (
        eligible_all
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )

    # ---------------------------------------------------------
    # Robustness cohort:
    # all non-terminal eligible encounters
    # ---------------------------------------------------------

    all_eligible = eligible_all.copy()

    summaries = [
        summarize_cohort(
            "raw",
            data,
        ),
        summarize_cohort(
            "source_first_per_patient",
            source_first,
        ),
        summarize_cohort(
            "min_encounter_id_per_patient",
            min_id_first,
        ),
        summarize_cohort(
            "first_then_remove_terminal",
            paper_like,
        ),
        summarize_cohort(
            "remove_terminal_then_first",
            eligible_then_first,
        ),
        summarize_cohort(
            "all_non_terminal_encounters",
            all_eligible,
        ),
    ]

    summary_table = pd.DataFrame(
        summaries
    )

    source_first_terminal = int(
        source_first[
            "discharge_disposition_id"
        ]
        .isin(terminal_ids)
        .sum()
    )

    patients_recovered_by_eligible_first = int(
        eligible_then_first[
            "patient_nbr"
        ]
        .nunique()
        - paper_like[
            "patient_nbr"
        ]
        .nunique()
    )

    audit = {
        "terminal_or_hospice_ids": sorted(
            terminal_ids
        ),
        "raw_rows": int(len(data)),
        "raw_unique_patients": int(
            data["patient_nbr"].nunique()
        ),
        "source_first_vs_min_encounter_id": {
            "patients_compared": int(
                len(comparison)
            ),
            "mismatched_first_encounters": (
                mismatch_count
            ),
            "match_percentage": float(
                (
                    len(comparison)
                    - mismatch_count
                )
                / len(comparison)
                * 100
            ),
        },
        "source_first_terminal_encounters": (
            source_first_terminal
        ),
        "patients_recovered_if_terminal_removed_first": (
            patients_recovered_by_eligible_first
        ),
        "strategies": summaries,
    }

    TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_table.to_csv(
        TABLE_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSOURCE ORDER VS MINIMUM ENCOUNTER ID")
    print("-" * 84)

    print(
        "Patients compared         :",
        f"{len(comparison):,}",
    )

    print(
        "Mismatched first records  :",
        f"{mismatch_count:,}",
    )

    print(
        "Match percentage          :",
        f"{audit['source_first_vs_min_encounter_id']['match_percentage']:.4f}%",
    )

    print("\nCOHORT STRATEGY COMPARISON")
    print("-" * 84)

    display = summary_table.copy()

    display["positive_30d_pct"] = (
        display["positive_30d_pct"]
        .map(
            lambda value: f"{value:.2f}%"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print("\nORDER-SENSITIVITY")
    print("-" * 84)

    print(
        "Terminal/hospice among source-first encounters :",
        f"{source_first_terminal:,}",
    )

    print(
        "Patients recovered if exclusion happens first :",
        f"{patients_recovered_by_eligible_first:,}",
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
        "\nCohort-rule audit completed successfully."
    )


if __name__ == "__main__":
    main()