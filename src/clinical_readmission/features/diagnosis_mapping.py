from __future__ import annotations

from typing import Any

import pandas as pd

from clinical_readmission.features.feature_schema import (
    DIAGNOSIS_FEATURES,
)

MISSING_CATEGORY = "Missing"
OTHER_CATEGORY = "Other"

DIAGNOSIS_CATEGORIES = (
    "Circulatory",
    "Diabetes",
    "Respiratory",
    "Digestive",
    "Genitourinary",
    "Musculoskeletal",
    "Neoplasms",
    "Injury",
    OTHER_CATEGORY,
    MISSING_CATEGORY,
)


def map_icd9_category(
    value: Any,
) -> str:
    """Map an ICD-9 diagnosis code to a clinical category."""

    if pd.isna(value):
        return MISSING_CATEGORY

    text = str(value).strip()

    if text.lower() in {
        "",
        "?",
        "nan",
        "none",
        "<na>",
    }:
        return MISSING_CATEGORY

    # ICD-9 supplementary V and E codes fall outside the
    # numeric disease groups used by the reference paper.
    if text.upper().startswith(("V", "E")):
        return OTHER_CATEGORY

    try:
        code = float(text)
    except (TypeError, ValueError):
        return OTHER_CATEGORY

    # Diabetes mellitus: ICD-9 250.xx
    if 250 <= code < 251:
        return "Diabetes"

    # Diseases of the circulatory system:
    # ICD-9 390-459 plus 785.
    if 390 <= code <= 459 or code == 785:
        return "Circulatory"

    # Diseases of the respiratory system:
    # ICD-9 460-519 plus 786.
    if 460 <= code <= 519 or code == 786:
        return "Respiratory"

    # Diseases of the digestive system:
    # ICD-9 520-579 plus 787.
    if 520 <= code <= 579 or code == 787:
        return "Digestive"

    # Diseases of the genitourinary system:
    # ICD-9 580-629 plus 788.
    if 580 <= code <= 629 or code == 788:
        return "Genitourinary"

    # Diseases of the musculoskeletal system
    # and connective tissue.
    if 710 <= code <= 739:
        return "Musculoskeletal"

    # Neoplasms.
    if 140 <= code <= 239:
        return "Neoplasms"

    # Injury and poisoning.
    if 800 <= code <= 999:
        return "Injury"

    return OTHER_CATEGORY


def map_diagnosis_frame(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Map all diagnosis columns while preserving row order."""

    missing_columns = (
        set(DIAGNOSIS_FEATURES)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing diagnosis columns: "
            f"{sorted(missing_columns)}"
        )

    result = data.loc[
        :,
        list(DIAGNOSIS_FEATURES),
    ].copy()

    for column in DIAGNOSIS_FEATURES:
        result[column] = (
            result[column]
            .map(map_icd9_category)
        )

    return result