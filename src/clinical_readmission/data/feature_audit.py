from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from clinical_readmission.data.validate import load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "feature_quality_audit.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "feature_quality_summary.json"
)


def infer_feature_type(
    column_name: str,
    series: pd.Series,
) -> str:
    if column_name.endswith("_id"):
        return "categorical_code"

    if is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def audit_feature(
    column_name: str,
    series: pd.Series,
) -> dict:
    rows = len(series)

    missing_count = int(
        series.isna().sum()
    )

    unique_non_null = int(
        series.nunique(dropna=True)
    )

    cardinality_ratio = (
        unique_non_null / rows
        if rows
        else 0.0
    )

    as_string = (
        series.astype("string")
        .str.strip()
    )

    question_mark_count = int(
        (as_string == "?").sum()
    )

    unknown_invalid_count = int(
        (as_string == "Unknown/Invalid").sum()
    )

    constant = (
        unique_non_null <= 1
    )

    near_unique = (
        cardinality_ratio >= 0.95
    )

    return {
        "feature": column_name,
        "dtype": str(series.dtype),
        "feature_type_guess": infer_feature_type(
            column_name,
            series,
        ),
        "rows": rows,
        "missing_count": missing_count,
        "missing_pct": (
            missing_count
            / rows
            * 100
        ),
        "unique_non_null": unique_non_null,
        "cardinality_ratio": cardinality_ratio,
        "question_mark_count": question_mark_count,
        "unknown_invalid_count": (
            unknown_invalid_count
        ),
        "constant": constant,
        "near_unique": near_unique,
    }


def build_feature_audit(
    features: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for column in features.columns:
        records.append(
            audit_feature(
                column,
                features[column],
            )
        )

    audit = pd.DataFrame(records)

    return audit.sort_values(
        by=[
            "missing_pct",
            "cardinality_ratio",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def build_summary(
    audit: pd.DataFrame,
) -> dict:
    high_missing = audit[
        audit["missing_pct"] >= 20
    ]

    moderate_missing = audit[
        (audit["missing_pct"] >= 5)
        & (audit["missing_pct"] < 20)
    ]

    near_unique = audit[
        audit["near_unique"]
    ]

    constants = audit[
        audit["constant"]
    ]

    question_mark_features = audit[
        audit["question_mark_count"] > 0
    ]

    unknown_features = audit[
        audit["unknown_invalid_count"] > 0
    ]

    return {
        "total_features": int(
            len(audit)
        ),
        "features_with_missing": int(
            (audit["missing_count"] > 0).sum()
        ),
        "high_missing_features_ge_20_pct": (
            high_missing["feature"].tolist()
        ),
        "moderate_missing_features_5_to_20_pct": (
            moderate_missing[
                "feature"
            ].tolist()
        ),
        "near_unique_features": (
            near_unique[
                "feature"
            ].tolist()
        ),
        "constant_features": (
            constants[
                "feature"
            ].tolist()
        ),
        "features_with_question_mark": (
            question_mark_features[
                "feature"
            ].tolist()
        ),
        "features_with_unknown_invalid": (
            unknown_features[
                "feature"
            ].tolist()
        ),
    }


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
    _, features, _ = load_raw_data()

    audit = build_feature_audit(
        features
    )

    summary = build_summary(
        audit
    )

    save_outputs(
        audit,
        summary,
    )

    print("=" * 72)
    print("FEATURE DATA QUALITY AUDIT")
    print("=" * 72)

    print(
        "\nTotal features       :",
        summary["total_features"],
    )

    print(
        "Features with missing:",
        summary["features_with_missing"],
    )

    print("\nTOP FEATURES BY MISSINGNESS")
    print("-" * 72)

    columns = [
        "feature",
        "dtype",
        "feature_type_guess",
        "missing_count",
        "missing_pct",
        "unique_non_null",
    ]

    print(
        audit[columns]
        .head(20)
        .to_string(
            index=False,
            formatters={
                "missing_pct": (
                    lambda value: f"{value:.2f}%"
                )
            },
        )
    )

    print("\nHIGH MISSINGNESS >= 20%")
    print("-" * 72)

    high_missing = audit[
        audit["missing_pct"] >= 20
    ]

    if high_missing.empty:
        print("None")
    else:
        print(
            high_missing[
                [
                    "feature",
                    "missing_pct",
                ]
            ].to_string(
                index=False,
                formatters={
                    "missing_pct": (
                        lambda value: (
                            f"{value:.2f}%"
                        )
                    )
                },
            )
        )

    print("\nNEAR-UNIQUE FEATURES")
    print("-" * 72)

    near_unique = audit[
        audit["near_unique"]
    ]

    if near_unique.empty:
        print("None")
    else:
        print(
            near_unique[
                [
                    "feature",
                    "unique_non_null",
                    "cardinality_ratio",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nPLACEHOLDER '?' FEATURES")
    print("-" * 72)

    question_marks = audit[
        audit[
            "question_mark_count"
        ] > 0
    ]

    if question_marks.empty:
        print("None")
    else:
        print(
            question_marks[
                [
                    "feature",
                    "question_mark_count",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nUNKNOWN / INVALID FEATURES")
    print("-" * 72)

    unknown = audit[
        audit[
            "unknown_invalid_count"
        ] > 0
    ]

    if unknown.empty:
        print("None")
    else:
        print(
            unknown[
                [
                    "feature",
                    "unknown_invalid_count",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nAudit table saved to:",
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
        "\nFeature audit completed successfully."
    )


if __name__ == "__main__":
    main()