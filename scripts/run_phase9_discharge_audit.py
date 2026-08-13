from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
    / "primary.csv"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase9_discharge_disposition_audit.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_discharge_disposition_audit.json"
)

TARGET_COLUMN = "readmitted_30d"
DISPOSITION_COLUMN = "discharge_disposition_id"

AUDITED_NONREADMITTABLE_IDS = (
    11,
    13,
    14,
    19,
    20,
    21,
)


def main() -> None:
    cohort = pd.read_csv(
        COHORT_PATH,
        usecols=[
            DISPOSITION_COLUMN,
            TARGET_COLUMN,
        ],
    )

    summary = (
        cohort.groupby(
            DISPOSITION_COLUMN,
            dropna=False,
        )[
            TARGET_COLUMN
        ]
        .agg(
            count="count",
            readmitted_30d_count="sum",
            readmitted_30d_rate="mean",
        )
        .reset_index()
        .sort_values(
            DISPOSITION_COLUMN,
        )
        .reset_index(
            drop=True
        )
    )

    audited_rows = cohort[
        cohort[
            DISPOSITION_COLUMN
        ].isin(
            AUDITED_NONREADMITTABLE_IDS
        )
    ]

    audited_count = int(
        len(
            audited_rows
        )
    )

    audit_passed = (
        audited_count == 0
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 9,
        "analysis": (
            "discharge_disposition_leakage_audit"
        ),
        "audited_disposition_ids": list(
            AUDITED_NONREADMITTABLE_IDS
        ),
        "cohort_rows": int(
            len(
                cohort
            )
        ),
        "audited_rows_present": (
            audited_count
        ),
        "audit_passed": (
            audit_passed
        ),
        "interpretation": {
            "obvious_nonreadmittable_dispositions_present": (
                False
                if audit_passed
                else True
            ),
            "model_timing": (
                "at_or_near_discharge"
            ),
            "admission_time_model_claimed": False,
            "discharge_disposition_retained": True,
        },
        "data_policy": {
            "identifiers_saved": False,
            "test_used": False,
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    OUTPUT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("=" * 96)
    print("PHASE 9 DISCHARGE-DISPOSITION AUDIT")
    print("=" * 96)

    print(
        "\nCohort rows:",
        len(
            cohort
        ),
    )

    print(
        "Audited IDs:",
        ", ".join(
            str(value)
            for value in (
                AUDITED_NONREADMITTABLE_IDS
            )
        ),
    )

    print(
        "Rows with audited IDs:",
        audited_count,
    )

    print(
        "\nAUDIT RESULT:",
        (
            "PASS"
            if audit_passed
            else "FAIL"
        ),
    )

    print(
        "Model timing interpretation: "
        "at/near discharge"
    )

    print(
        "Admission-time model claimed: False"
    )

    print(
        "\nSaved table:",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved JSON :",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nTest used: False"
    )

    if not audit_passed:
        raise ValueError(
            "Audited non-readmittable "
            "dispositions remain in cohort."
        )


if __name__ == "__main__":
    main()