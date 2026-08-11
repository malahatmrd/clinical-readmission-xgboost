from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from clinical_readmission.data.cohort_audit import (
    build_combined_data,
    get_terminal_ids,
)
from clinical_readmission.data.disposition_audit import (
    load_discharge_mapping,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "cohort_reconciliation_candidates.csv"
)

RECOVERED_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "audits"
    / "recovered_patient_audit.csv"
)

BREAKDOWN_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "source_first_terminal_breakdown.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "cohort_reconciliation_summary.json"
)

REFERENCE_COHORT_SIZE = 69_984
REFERENCE_POSITIVE_COUNT = 6_459


def build_source_first(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return (
        data
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )


def build_terminal_breakdown(
    source_first: pd.DataFrame,
    terminal_ids: set[int],
) -> pd.DataFrame:
    mapping = load_discharge_mapping()

    terminal = source_first.loc[
        source_first[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    breakdown = (
        terminal
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

    breakdown["description"] = (
        breakdown[
            "discharge_disposition_id"
        ]
        .map(mapping)
        .fillna("Unmapped")
    )

    return breakdown[
        [
            "discharge_disposition_id",
            "description",
            "encounters",
            "unique_patients",
            "positive_30d",
        ]
    ].sort_values(
        "discharge_disposition_id"
    )


def generate_candidate_rules(
    source_first: pd.DataFrame,
    terminal_ids: set[int],
) -> pd.DataFrame:
    ids = sorted(terminal_ids)

    records: list[dict] = []

    for subset_size in range(
        len(ids) + 1
    ):
        for subset_tuple in itertools.combinations(
            ids,
            subset_size,
        ):
            subset = set(subset_tuple)

            cohort = source_first.loc[
                ~source_first[
                    "discharge_disposition_id"
                ].isin(subset)
            ]

            encounters = int(
                len(cohort)
            )

            positives = int(
                (
                    cohort["readmitted"]
                    == "<30"
                ).sum()
            )

            records.append(
                {
                    "excluded_ids": (
                        ",".join(
                            str(value)
                            for value in sorted(
                                subset
                            )
                        )
                        if subset
                        else "None"
                    ),
                    "excluded_id_count": len(
                        subset
                    ),
                    "encounters": encounters,
                    "positive_30d": positives,
                    "delta_vs_reference_n": (
                        encounters
                        - REFERENCE_COHORT_SIZE
                    ),
                    "delta_vs_reference_positive": (
                        positives
                        - REFERENCE_POSITIVE_COUNT
                    ),
                }
            )

    candidates = pd.DataFrame(
        records
    )

    candidates[
        "absolute_n_difference"
    ] = (
        candidates[
            "delta_vs_reference_n"
        ].abs()
    )

    candidates[
        "absolute_positive_difference"
    ] = (
        candidates[
            "delta_vs_reference_positive"
        ].abs()
    )

    return candidates.sort_values(
        by=[
            "absolute_n_difference",
            "absolute_positive_difference",
            "excluded_id_count",
        ]
    ).reset_index(
        drop=True
    )


def build_recovered_patient_audit(
    data: pd.DataFrame,
    terminal_ids: set[int],
) -> pd.DataFrame:
    source_first = build_source_first(
        data
    )

    first_then_remove = (
        source_first.loc[
            ~source_first[
                "discharge_disposition_id"
            ].isin(terminal_ids)
        ]
        .copy()
    )

    eligible = data.loc[
        ~data[
            "discharge_disposition_id"
        ].isin(terminal_ids)
    ].copy()

    eligible_first = (
        eligible
        .drop_duplicates(
            subset="patient_nbr",
            keep="first",
        )
        .copy()
    )

    recovered_ids = sorted(
        set(
            eligible_first[
                "patient_nbr"
            ]
        )
        - set(
            first_then_remove[
                "patient_nbr"
            ]
        )
    )

    original = source_first.loc[
        source_first[
            "patient_nbr"
        ].isin(recovered_ids),
        [
            "patient_nbr",
            "encounter_id",
            "discharge_disposition_id",
            "readmitted",
        ],
    ].rename(
        columns={
            "encounter_id": (
                "original_first_encounter_id"
            ),
            "discharge_disposition_id": (
                "original_first_disposition_id"
            ),
            "readmitted": (
                "original_first_readmitted"
            ),
        }
    )

    replacement = eligible_first.loc[
        eligible_first[
            "patient_nbr"
        ].isin(recovered_ids),
        [
            "patient_nbr",
            "encounter_id",
            "discharge_disposition_id",
            "readmitted",
        ],
    ].rename(
        columns={
            "encounter_id": (
                "first_eligible_encounter_id"
            ),
            "discharge_disposition_id": (
                "first_eligible_disposition_id"
            ),
            "readmitted": (
                "first_eligible_readmitted"
            ),
        }
    )

    recovered = original.merge(
        replacement,
        on="patient_nbr",
        validate="one_to_one",
    )

    return recovered.sort_values(
        "patient_nbr"
    ).reset_index(
        drop=True
    )


def main() -> None:
    data = build_combined_data()

    terminal_ids = get_terminal_ids()

    source_first = build_source_first(
        data
    )

    breakdown = build_terminal_breakdown(
        source_first,
        terminal_ids,
    )

    candidates = generate_candidate_rules(
        source_first,
        terminal_ids,
    )

    recovered = (
        build_recovered_patient_audit(
            data,
            terminal_ids,
        )
    )

    CANDIDATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        CANDIDATES_PATH,
        index=False,
    )

    recovered.to_csv(
        RECOVERED_PATH,
        index=False,
    )

    breakdown.to_csv(
        BREAKDOWN_PATH,
        index=False,
    )

    exact_size_matches = candidates.loc[
        candidates[
            "encounters"
        ] == REFERENCE_COHORT_SIZE
    ]

    exact_full_matches = candidates.loc[
        (
            candidates["encounters"]
            == REFERENCE_COHORT_SIZE
        )
        & (
            candidates["positive_30d"]
            == REFERENCE_POSITIVE_COUNT
        )
    ]

    summary = {
        "reference_paper": {
            "cohort_size": (
                REFERENCE_COHORT_SIZE
            ),
            "positive_30d": (
                REFERENCE_POSITIVE_COUNT
            ),
        },
        "current_snapshot": {
            "raw_encounters": int(
                len(data)
            ),
            "unique_patients": int(
                data[
                    "patient_nbr"
                ].nunique()
            ),
        },
        "terminal_ids": sorted(
            terminal_ids
        ),
        "source_first_terminal_total": int(
            breakdown[
                "encounters"
            ].sum()
        ),
        "exact_size_rule_matches": int(
            len(exact_size_matches)
        ),
        "exact_size_and_positive_matches": int(
            len(exact_full_matches)
        ),
        "recovered_patients_if_eligibility_first": int(
            len(recovered)
        ),
    }

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

    print("=" * 92)
    print("COHORT RECONCILIATION AUDIT")
    print("=" * 92)

    print("\nSOURCE-FIRST TERMINAL BREAKDOWN")
    print("-" * 92)

    print(
        breakdown.to_string(
            index=False
        )
    )

    print("\nREFERENCE PAPER")
    print("-" * 92)

    print(
        "Reported cohort size   :",
        f"{REFERENCE_COHORT_SIZE:,}",
    )

    print(
        "Reported positive <30  :",
        f"{REFERENCE_POSITIVE_COUNT:,}",
    )

    print("\nTOP CANDIDATE TERMINAL-ID RULES")
    print("-" * 92)

    print(
        candidates.head(15).to_string(
            index=False
        )
    )

    print("\nEXACT MATCH CHECK")
    print("-" * 92)

    print(
        "Rules matching reported N:",
        len(exact_size_matches),
    )

    print(
        "Rules matching N and positives:",
        len(exact_full_matches),
    )

    if not exact_size_matches.empty:
        print(
            "\nRules with exact N:"
        )

        print(
            exact_size_matches[
                [
                    "excluded_ids",
                    "encounters",
                    "positive_30d",
                    "delta_vs_reference_positive",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nRECOVERED PATIENTS")
    print("-" * 92)

    print(
        "Patients recovered when eligibility "
        "is applied before first-encounter selection:",
        len(recovered),
    )

    if not recovered.empty:
        print(
            recovered.to_string(
                index=False
            )
        )

    print(
        "\nCandidate table:",
        CANDIDATES_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Recovered-patient table:",
        RECOVERED_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Terminal breakdown:",
        BREAKDOWN_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Summary:",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nReconciliation audit completed successfully."
    )


if __name__ == "__main__":
    main()