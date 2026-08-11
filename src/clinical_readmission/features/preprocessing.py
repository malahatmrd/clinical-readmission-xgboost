from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from clinical_readmission.features.diagnosis_mapping import (
    DIAGNOSIS_CATEGORIES,
    map_diagnosis_frame,
)
from clinical_readmission.features.feature_schema import (
    DIAGNOSIS_FEATURES,
    LAB_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SPECIALTY_FEATURES,
    STANDARD_CATEGORICAL_FEATURES,
)

MISSING_CATEGORY = "Missing"
NOT_MEASURED_CATEGORY = "NotMeasured"

SPECIALTY_MIN_FREQUENCY = 0.001


def prepare_categorical_frame(
    data: pd.DataFrame,
    missing_label: str,
) -> pd.DataFrame:
    """Convert categorical values to strings and preserve missingness."""

    result = data.copy()

    for column in result.columns:
        result[column] = (
            result[column]
            .astype("string")
            .fillna(missing_label)
        )

    return result


def build_categorical_preparer(
    missing_label: str,
) -> FunctionTransformer:
    return FunctionTransformer(
        prepare_categorical_frame,
        kw_args={
            "missing_label": missing_label,
        },
        validate=False,
        feature_names_out="one-to-one",
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the leakage-safe baseline preprocessing transformer."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "prepare",
                build_categorical_preparer(
                    MISSING_CATEGORY
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    specialty_pipeline = Pipeline(
        steps=[
            (
                "prepare",
                build_categorical_preparer(
                    MISSING_CATEGORY
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=SPECIALTY_MIN_FREQUENCY,
                    sparse_output=True,
                ),
            ),
        ]
    )

    lab_pipeline = Pipeline(
        steps=[
            (
                "prepare",
                build_categorical_preparer(
                    NOT_MEASURED_CATEGORY
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    diagnosis_pipeline = Pipeline(
        steps=[
            (
                "group",
                FunctionTransformer(
                    map_diagnosis_frame,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    categories=[
                        list(DIAGNOSIS_CATEGORIES)
                        for _ in DIAGNOSIS_FEATURES
                    ],
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_FEATURES),
            ),
            (
                "diagnosis",
                diagnosis_pipeline,
                list(DIAGNOSIS_FEATURES),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(STANDARD_CATEGORICAL_FEATURES),
            ),
            (
                "specialty",
                specialty_pipeline,
                list(SPECIALTY_FEATURES),
            ),
            (
                "lab",
                lab_pipeline,
                list(LAB_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )