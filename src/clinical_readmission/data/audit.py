from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clinical_readmission.data.validate import load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AUDIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "raw_data_audit.json"
)


def build_patient_audit(
    identifiers: pd.DataFrame,
) -> dict:
    encounters_per_patient = (
        identifiers
        .groupby("patient_nbr")
        .size()
    )

    repeated_patients = int(
        (encounters_per_patient > 1).sum()
    )

    return {
        "rows": int(len(identifiers)),
        "unique_patients": int(
            identifiers["patient_nbr"].nunique()
        ),
        "unique_encounters": int(
            identifiers["encounter_id"].nunique()
        ),
        "patients_with_multiple_encounters": (
            repeated_patients
        ),
        "patients_with_multiple_encounters_pct": float(
            repeated_patients
            / identifiers["patient_nbr"].nunique()
            * 100
        ),
        "mean_encounters_per_patient": float(
            encounters_per_patient.mean()
        ),
        "median_encounters_per_patient": float(
            encounters_per_patient.median()
        ),
        "max_encounters_per_patient": int(
            encounters_per_patient.max()
        ),
    }


def build_target_audit(
    target: pd.DataFrame,
) -> dict:
    counts = (
        target["readmitted"]
        .value_counts(dropna=False)
    )

    percentages = (
        target["readmitted"]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .mul(100)
    )

    positive_count = int(
        (target["readmitted"] == "<30").sum()
    )

    positive_rate = float(
        positive_count / len(target)
    )

    return {
        "original_distribution": {
            str(label): {
                "count": int(count),
                "percentage": float(
                    percentages[label]
                ),
            }
            for label, count in counts.items()
        },
        "binary_target_definition": {
            "positive": "<30",
            "negative": ["NO", ">30"],
        },
        "positive_count": positive_count,
        "negative_count": int(
            len(target) - positive_count
        ),
        "positive_rate": positive_rate,
        "positive_percentage": float(
            positive_rate * 100
        ),
    }


def build_payer_code_audit(
    features: pd.DataFrame,
) -> dict:
    payer = features["payer_code"]

    top_values = (
        payer
        .value_counts(dropna=False)
        .head(20)
    )

    return {
        "dtype": str(payer.dtype),
        "missing_count": int(
            payer.isna().sum()
        ),
        "missing_percentage": float(
            payer.isna().mean() * 100
        ),
        "unique_non_null": int(
            payer.nunique(dropna=True)
        ),
        "top_values": {
            str(key): int(value)
            for key, value in top_values.items()
        },
    }


def save_audit(
    audit: dict,
) -> None:
    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    identifiers, features, target = (
        load_raw_data()
    )

    patient_audit = build_patient_audit(
        identifiers
    )

    target_audit = build_target_audit(
        target
    )

    payer_audit = build_payer_code_audit(
        features
    )

    audit = {
        "patient_encounter_audit": patient_audit,
        "target_audit": target_audit,
        "payer_code_audit": payer_audit,
    }

    save_audit(audit)

    print("=" * 72)
    print("RAW DATA AUDIT")
    print("=" * 72)

    print("\nPATIENT / ENCOUNTER")
    print("-" * 72)

    print(
        "Rows                         : "
        f"{patient_audit['rows']:,}"
    )

    print(
        "Unique patients              : "
        f"{patient_audit['unique_patients']:,}"
    )

    print(
        "Unique encounters            : "
        f"{patient_audit['unique_encounters']:,}"
    )

    print(
        "Patients with >1 encounter   : "
        f"{patient_audit['patients_with_multiple_encounters']:,}"
    )

    print(
        "Repeated-patient percentage  : "
        f"{patient_audit['patients_with_multiple_encounters_pct']:.2f}%"
    )

    print(
        "Mean encounters / patient    : "
        f"{patient_audit['mean_encounters_per_patient']:.3f}"
    )

    print(
        "Median encounters / patient  : "
        f"{patient_audit['median_encounters_per_patient']:.1f}"
    )

    print(
        "Maximum encounters / patient : "
        f"{patient_audit['max_encounters_per_patient']}"
    )

    print("\nTARGET")
    print("-" * 72)

    for label, values in target_audit[
        "original_distribution"
    ].items():
        print(
            f"{label:<5} "
            f"{values['count']:>8,} "
            f"({values['percentage']:6.2f}%)"
        )

    print("\nBINARY 30-DAY TARGET")
    print("-" * 72)

    print(
        "Positive definition : <30"
    )

    print(
        "Positive count      : "
        f"{target_audit['positive_count']:,}"
    )

    print(
        "Negative count      : "
        f"{target_audit['negative_count']:,}"
    )

    print(
        "Positive percentage : "
        f"{target_audit['positive_percentage']:.2f}%"
    )

    print("\nPAYER CODE")
    print("-" * 72)

    print(
        "dtype              : "
        f"{payer_audit['dtype']}"
    )

    print(
        "Missing count      : "
        f"{payer_audit['missing_count']:,}"
    )

    print(
        "Missing percentage : "
        f"{payer_audit['missing_percentage']:.2f}%"
    )

    print(
        "Unique non-null    : "
        f"{payer_audit['unique_non_null']:,}"
    )

    print("\nTop payer_code values")

    for key, value in (
        payer_audit["top_values"].items()
    ):
        print(
            f"{key:<15} {value:>8,}"
        )

    print(
        "\nAudit saved to:",
        AUDIT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nAudit completed successfully."
    )


if __name__ == "__main__":
    main()