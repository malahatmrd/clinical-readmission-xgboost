from __future__ import annotations

import pandas as pd

from clinical_readmission.features.diagnosis_mapping import (
    map_diagnosis_frame,
    map_icd9_category,
)


def test_maps_reference_diagnosis_categories() -> None:
    cases = {
        "250": "Diabetes",
        "250.02": "Diabetes",
        "414": "Circulatory",
        "785": "Circulatory",
        "486": "Respiratory",
        "786": "Respiratory",
        "574": "Digestive",
        "787": "Digestive",
        "599": "Genitourinary",
        "788": "Genitourinary",
        "715": "Musculoskeletal",
        "174": "Neoplasms",
        "820": "Injury",
    }

    for code, expected in cases.items():
        assert map_icd9_category(code) == expected


def test_maps_other_codes() -> None:
    assert map_icd9_category("276") == "Other"
    assert map_icd9_category("V45") == "Other"
    assert map_icd9_category("E849") == "Other"
    assert map_icd9_category("invalid") == "Other"


def test_maps_missing_values() -> None:
    assert map_icd9_category(None) == "Missing"
    assert map_icd9_category(float("nan")) == "Missing"
    assert map_icd9_category("?") == "Missing"
    assert map_icd9_category("") == "Missing"


def test_maps_diagnosis_dataframe() -> None:
    data = pd.DataFrame(
        {
            "diag_1": [
                "250.8",
                "414",
                None,
            ],
            "diag_2": [
                "786",
                "715",
                "V45",
            ],
            "diag_3": [
                "599",
                "820",
                "276",
            ],
        }
    )

    result = map_diagnosis_frame(data)

    assert result.to_dict(
        orient="list"
    ) == {
        "diag_1": [
            "Diabetes",
            "Circulatory",
            "Missing",
        ],
        "diag_2": [
            "Respiratory",
            "Musculoskeletal",
            "Other",
        ],
        "diag_3": [
            "Genitourinary",
            "Injury",
            "Other",
        ],
    }