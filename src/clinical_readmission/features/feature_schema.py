from __future__ import annotations

IDENTIFIER_COLUMNS = (
    "encounter_id",
    "patient_nbr",
    "source_row",
)

TARGET_COLUMNS = (
    "readmitted",
    "readmitted_30d",
)

BASELINE_DROP_FEATURES = (
    "weight",
    "examide",
    "citoglipton",
    "glimepiride-pioglitazone",
    "metformin-pioglitazone",
)

NUMERIC_FEATURES = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)

DIAGNOSIS_FEATURES = (
    "diag_1",
    "diag_2",
    "diag_3",
)

SPECIALTY_FEATURES = (
    "medical_specialty",
)

LAB_CATEGORICAL_FEATURES = (
    "max_glu_serum",
    "A1Cresult",
)

STANDARD_CATEGORICAL_FEATURES = (
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "metformin-rosiglitazone",
    "change",
    "diabetesMed",
)

MODEL_INPUT_FEATURES = (
    NUMERIC_FEATURES
    + DIAGNOSIS_FEATURES
    + STANDARD_CATEGORICAL_FEATURES
    + SPECIALTY_FEATURES
    + LAB_CATEGORICAL_FEATURES
)

EXCLUDED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + TARGET_COLUMNS
    + BASELINE_DROP_FEATURES
)


def validate_feature_schema(
    available_columns: list[str],
) -> None:
    available = set(available_columns)

    required = (
        set(MODEL_INPUT_FEATURES)
        | set(EXCLUDED_COLUMNS)
    )

    missing = sorted(
        required - available
    )

    if missing:
        raise ValueError(
            "Feature schema references missing columns: "
            f"{missing}"
        )

    if len(MODEL_INPUT_FEATURES) != len(
        set(MODEL_INPUT_FEATURES)
    ):
        raise ValueError(
            "Duplicate model-input feature detected."
        )

    overlap = (
        set(MODEL_INPUT_FEATURES)
        & set(EXCLUDED_COLUMNS)
    )

    if overlap:
        raise ValueError(
            "Columns cannot be both model inputs "
            f"and excluded: {sorted(overlap)}"
        )