from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clinical_readmission.data.cohort_audit import (
    build_combined_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "audits"
    / "post_terminal_encounters.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "temporal_consistency_audit.json"
)

EXPIRED_IDS = {
    11,
    19,
    20,
    21,
}

HOSPICE_IDS = {
    13,
    14,
}

TERMINAL_IDS = (
    EXPIRED_IDS
    | HOSPICE_IDS
)


def build_post_terminal_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []

    ordered = data.sort_values(
        [
            "patient_nbr",
            "encounter_id",
        ]
    )

    for patient_nbr, group in ordered.groupby(
        "patient_nbr",
        sort=False,
    ):
        group = group.reset_index(
            drop=True
        )

        dispositions = group[
            "discharge_disposition_id"
        ]

        terminal_positions = group.index[
            dispositions.isin(
                TERMINAL_IDS
            )
        ].tolist()

        for position in terminal_positions:
            later = group.iloc[
                position + 1 :
            ]

            if later.empty:
                continue

            terminal_row = group.iloc[
                position
            ]

            first_later = later.iloc[0]

            disposition_id = int(
                terminal_row[
                    "discharge_disposition_id"
                ]
            )

            records.append(
                {
                    "patient_nbr": int(
                        patient_nbr
                    ),
                    "terminal_encounter_id": int(
                        terminal_row[
                            "encounter_id"
                        ]
                    ),
                    "terminal_disposition_id": (
                        disposition_id
                    ),
                    "terminal_type": (
                        "expired"
                        if disposition_id
                        in EXPIRED_IDS
                        else "hospice"
                    ),
                    "terminal_readmitted": (
                        terminal_row[
                            "readmitted"
                        ]
                    ),
                    "later_encounter_count": int(
                        len(later)
                    ),
                    "first_later_encounter_id": int(
                        first_later[
                            "encounter_id"
                        ]
                    ),
                    "first_later_disposition_id": int(
                        first_later[
                            "discharge_disposition_id"
                        ]
                    ),
                    "first_later_readmitted": (
                        first_later[
                            "readmitted"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def main() -> None:
    data = build_combined_data()

    encounter_ids = data[
        "encounter_id"
    ]

    globally_monotonic = bool(
        encounter_ids.is_monotonic_increasing
    )

    global_inversions = int(
        (
            encounter_ids.diff()
            .fillna(0)
            < 0
        ).sum()
    )

    expired_rows = data[
        data[
            "discharge_disposition_id"
        ].isin(EXPIRED_IDS)
    ]

    hospice_rows = data[
        data[
            "discharge_disposition_id"
        ].isin(HOSPICE_IDS)
    ]

    post_terminal = (
        build_post_terminal_table(
            data
        )
    )

    if post_terminal.empty:
        post_expired = (
            post_terminal.copy()
        )
        post_hospice = (
            post_terminal.copy()
        )
    else:
        post_expired = post_terminal[
            post_terminal[
                "terminal_type"
            ]
            == "expired"
        ]

        post_hospice = post_terminal[
            post_terminal[
                "terminal_type"
            ]
            == "hospice"
        ]

    summary = {
        "raw_encounters": int(
            len(data)
        ),
        "unique_patients": int(
            data[
                "patient_nbr"
            ].nunique()
        ),
        "encounter_id": {
            "globally_monotonic_increasing": (
                globally_monotonic
            ),
            "global_inversions": (
                global_inversions
            ),
        },
        "expired": {
            "rows": int(
                len(expired_rows)
            ),
            "unique_patients": int(
                expired_rows[
                    "patient_nbr"
                ].nunique()
            ),
            "patients_with_observed_later_encounter": int(
                post_expired[
                    "patient_nbr"
                ].nunique()
                if not post_expired.empty
                else 0
            ),
        },
        "hospice": {
            "rows": int(
                len(hospice_rows)
            ),
            "unique_patients": int(
                hospice_rows[
                    "patient_nbr"
                ].nunique()
            ),
            "patients_with_observed_later_encounter": int(
                post_hospice[
                    "patient_nbr"
                ].nunique()
                if not post_hospice.empty
                else 0
            ),
        },
        "terminal_overall": {
            "rows": int(
                data[
                    "discharge_disposition_id"
                ]
                .isin(TERMINAL_IDS)
                .sum()
            ),
            "patients_with_observed_later_encounter": int(
                post_terminal[
                    "patient_nbr"
                ].nunique()
                if not post_terminal.empty
                else 0
            ),
        },
    }

    TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    post_terminal.to_csv(
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

    print("=" * 88)
    print("TEMPORAL / TERMINAL CONSISTENCY AUDIT")
    print("=" * 88)

    print("\nENCOUNTER ID ORDER")
    print("-" * 88)

    print(
        "Globally monotonic increasing :",
        globally_monotonic,
    )

    print(
        "Global order inversions        :",
        global_inversions,
    )

    print("\nEXPIRED")
    print("-" * 88)

    print(
        "Expired rows                   :",
        f"{len(expired_rows):,}",
    )

    print(
        "Expired unique patients        :",
        f"{expired_rows['patient_nbr'].nunique():,}",
    )

    print(
        "Patients with later encounter  :",
        f"{summary['expired']['patients_with_observed_later_encounter']:,}",
    )

    print("\nHOSPICE")
    print("-" * 88)

    print(
        "Hospice rows                   :",
        f"{len(hospice_rows):,}",
    )

    print(
        "Hospice unique patients        :",
        f"{hospice_rows['patient_nbr'].nunique():,}",
    )

    print(
        "Patients with later encounter  :",
        f"{summary['hospice']['patients_with_observed_later_encounter']:,}",
    )

    print("\nPOST-TERMINAL RECORDS")
    print("-" * 88)

    if post_terminal.empty:
        print("None")
    else:
        print(
            post_terminal.to_string(
                index=False
            )
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
        "\nTemporal consistency audit completed successfully."
    )


if __name__ == "__main__":
    main()