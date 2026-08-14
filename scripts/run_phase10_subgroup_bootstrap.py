from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.evaluation.bootstrap import (
    DEFAULT_BOOTSTRAP_RANDOM_STATE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    draw_stratified_bootstrap_indices,
)
from clinical_readmission.evaluation.metrics import (
    calculate_probability_metrics,
)
from clinical_readmission.evaluation.thresholds import (
    calculate_threshold_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cohorts"
    / "primary.csv"
)

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "primary_split_assignments.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "phase7_calibration_candidate_probabilities.csv"
)

PHASE8_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_threshold_selection.json"
)

PHASE9_SHAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase9_shap_validation.json"
)

SUBGROUP_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_validation_subgroup_performance.csv"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase10_validation_subgroup_bootstrap.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase10_subgroup_bootstrap.json"
)

TARGET_COLUMN = "readmitted_30d"

PROBABILITY_COLUMN = (
    "tuned_xgboost_sigmoid_probability"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"
EXPECTED_THRESHOLD = 0.105

BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "sensitivity",
    "specificity",
    "ppv",
    "alerts_per_100",
)

AGE_GROUP_MAP = {
    "[0-10)": "<50",
    "[10-20)": "<50",
    "[20-30)": "<50",
    "[30-40)": "<50",
    "[40-50)": "<50",
    "[50-60)": "50-69",
    "[60-70)": "50-69",
    "[70-80)": "70-89",
    "[80-90)": "70-89",
    "[90-100)": "90+",
}


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_partition(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    ids = assignments.loc[
        assignments[
            "split"
        ].eq(
            split_name
        ),
        [
            "encounter_id",
            "patient_nbr",
            TARGET_COLUMN,
        ],
    ].copy()

    result = cohort.merge(
        ids,
        on=[
            "encounter_id",
            "patient_nbr",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_assignment",
        ),
    )

    assignment_target = (
        f"{TARGET_COLUMN}_assignment"
    )

    if not (
        result[
            TARGET_COLUMN
        ]
        == result[
            assignment_target
        ]
    ).all():
        raise ValueError(
            f"{split_name}: target mismatch."
        )

    return result


def normalize_missing_tokens(
    values,
) -> pd.Series:
    result = pd.Series(
        values,
        dtype="string",
    )

    return result.replace(
        {
            "?": pd.NA,
            "": pd.NA,
        }
    )


def build_demographic_labels(
    values,
) -> np.ndarray:
    return (
        normalize_missing_tokens(
            values
        )
        .fillna(
            "Missing"
        )
        .to_numpy(
            dtype=str
        )
    )


def build_age_groups(
    values,
) -> np.ndarray:
    age = normalize_missing_tokens(
        values
    )

    observed = set(
        age.dropna().unique()
    )

    unexpected = sorted(
        observed
        - set(
            AGE_GROUP_MAP
        )
    )

    if unexpected:
        raise ValueError(
            "Unexpected age categories: "
            f"{unexpected}"
        )

    return (
        age.map(
            AGE_GROUP_MAP
        )
        .fillna(
            "Missing"
        )
        .to_numpy(
            dtype=str
        )
    )


def summarize_distribution(
    values: np.ndarray,
) -> dict[str, float]:
    alpha = (
        1.0
        - DEFAULT_CONFIDENCE_LEVEL
    )

    return {
        "ci_lower": float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        "ci_upper": float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
        "bootstrap_standard_error": float(
            np.std(
                values,
                ddof=1,
            )
        ),
    }


def calculate_selected_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    probability_metrics = (
        calculate_probability_metrics(
            target,
            probabilities,
        )
    )

    threshold_metrics = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold,
        )
    )

    return {
        "roc_auc": float(
            probability_metrics[
                "roc_auc"
            ]
        ),
        "average_precision": float(
            probability_metrics[
                "average_precision"
            ]
        ),
        "brier_score": float(
            probability_metrics[
                "brier_score"
            ]
        ),
        "sensitivity": float(
            threshold_metrics[
                "sensitivity"
            ]
        ),
        "specificity": float(
            threshold_metrics[
                "specificity"
            ]
        ),
        "ppv": float(
            threshold_metrics[
                "ppv"
            ]
        ),
        "alerts_per_100": float(
            threshold_metrics[
                "alerts_per_100"
            ]
        ),
    }


def bootstrap_subgroup(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    random_state: int,
) -> dict[str, dict[str, float]]:
    point_metrics = (
        calculate_selected_metrics(
            target,
            probabilities,
            threshold,
        )
    )

    rng = np.random.default_rng(
        random_state
    )

    distributions = {
        metric: np.empty(
            DEFAULT_BOOTSTRAP_RESAMPLES,
            dtype=float,
        )
        for metric in BOOTSTRAP_METRICS
    }

    for iteration in range(
        DEFAULT_BOOTSTRAP_RESAMPLES
    ):
        indices = (
            draw_stratified_bootstrap_indices(
                target,
                rng,
            )
        )

        metrics = (
            calculate_selected_metrics(
                target[
                    indices
                ],
                probabilities[
                    indices
                ],
                threshold,
            )
        )

        for metric in (
            BOOTSTRAP_METRICS
        ):
            distributions[
                metric
            ][
                iteration
            ] = metrics[
                metric
            ]

    return {
        metric: {
            "estimate": float(
                point_metrics[
                    metric
                ]
            ),
            **summarize_distribution(
                distributions[
                    metric
                ]
            ),
        }
        for metric in BOOTSTRAP_METRICS
    }


def main() -> None:
    phase8 = load_json(
        PHASE8_SELECTION_PATH
    )

    phase9 = load_json(
        PHASE9_SHAP_PATH
    )

    if (
        phase8[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected frozen model."
        )

    threshold = float(
        phase8[
            "reference_threshold"
        ]
    )

    if abs(
        threshold
        - EXPECTED_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected frozen threshold."
        )

    if (
        phase8[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test set must remain locked."
        )

    observed_hash = file_sha256(
        PREDICTIONS_PATH
    )

    expected_hash = (
        phase9[
            "source_artifacts"
        ][
            "phase7_predictions"
        ][
            "sha256"
        ]
    )

    if (
        observed_hash
        != expected_hash
    ):
        raise ValueError(
            "Prediction artifact SHA256 mismatch."
        )

    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    subgroup_table = pd.read_csv(
        SUBGROUP_TABLE_PATH
    )

    validation = load_partition(
        cohort,
        assignments,
        "validation",
    ).reset_index(
        drop=True
    )

    y_validation = (
        validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        y_validation,
        predictions[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        ),
    ):
        raise ValueError(
            "Validation target alignment failed."
        )

    probabilities = (
        predictions[
            PROBABILITY_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    subgroup_axes = {
        "gender": (
            build_demographic_labels(
                validation[
                    "gender"
                ]
            )
        ),
        "race": (
            build_demographic_labels(
                validation[
                    "race"
                ]
            )
        ),
        "age_group": (
            build_age_groups(
                validation[
                    "age"
                ]
            )
        ),
    }

    reportable = subgroup_table.loc[
        subgroup_table[
            "reporting_eligible"
        ].astype(
            bool
        )
    ].copy()

    results = {}
    table_rows = []

    print(
        "=" * 104
    )
    print(
        "PHASE 10 SUBGROUP BOOTSTRAP"
    )
    print(
        "=" * 104
    )

    print(
        "\nBootstrap resamples:",
        DEFAULT_BOOTSTRAP_RESAMPLES,
    )

    print(
        "Bootstrap seed     :",
        DEFAULT_BOOTSTRAP_RANDOM_STATE,
    )

    print(
        "Confidence level   :",
        DEFAULT_CONFIDENCE_LEVEL,
    )

    print(
        "Frozen threshold   :",
        f"{threshold:.3f}",
    )

    print(
        "Test used          : False"
    )

    for subgroup_index, row in (
        reportable.reset_index(
            drop=True
        ).iterrows()
    ):
        subgroup_name = str(
            row[
                "subgroup_name"
            ]
        )

        subgroup_value = str(
            row[
                "subgroup_value"
            ]
        )

        values = subgroup_axes[
            subgroup_name
        ]

        mask = (
            values
            == subgroup_value
        )

        subgroup_target = (
            y_validation[
                mask
            ]
        )

        subgroup_probabilities = (
            probabilities[
                mask
            ]
        )

        seed = (
            DEFAULT_BOOTSTRAP_RANDOM_STATE
            + subgroup_index
        )

        print(
            f"\n{subgroup_name} = "
            f"{subgroup_value}"
        )

        print(
            "  rows      :",
            len(
                subgroup_target
            ),
        )

        print(
            "  positives :",
            int(
                subgroup_target.sum()
            ),
        )

        metrics = bootstrap_subgroup(
            subgroup_target,
            subgroup_probabilities,
            threshold,
            random_state=seed,
        )

        key = (
            f"{subgroup_name}::{subgroup_value}"
        )

        results[
            key
        ] = {
            "subgroup_name": (
                subgroup_name
            ),
            "subgroup_value": (
                subgroup_value
            ),
            "rows": int(
                len(
                    subgroup_target
                )
            ),
            "positives": int(
                subgroup_target.sum()
            ),
            "random_state": int(
                seed
            ),
            "metrics": metrics,
        }

        for metric in (
            BOOTSTRAP_METRICS
        ):
            result = metrics[
                metric
            ]

            table_rows.append(
                {
                    "subgroup_name": (
                        subgroup_name
                    ),
                    "subgroup_value": (
                        subgroup_value
                    ),
                    "metric": (
                        metric
                    ),
                    "estimate": (
                        result[
                            "estimate"
                        ]
                    ),
                    "ci_lower": (
                        result[
                            "ci_lower"
                        ]
                    ),
                    "ci_upper": (
                        result[
                            "ci_upper"
                        ]
                    ),
                    "bootstrap_standard_error": (
                        result[
                            "bootstrap_standard_error"
                        ]
                    ),
                    "bootstrap_random_state": (
                        seed
                    ),
                }
            )

        roc = metrics[
            "roc_auc"
        ]

        print(
            "  ROC-AUC   : "
            f"{roc['estimate']:.4f} "
            f"[{roc['ci_lower']:.4f}, "
            f"{roc['ci_upper']:.4f}]"
        )

    output_table = pd.DataFrame(
        table_rows
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 10,
        "analysis": (
            "validation_subgroup_bootstrap"
        ),
        "frozen_configuration": {
            "model": EXPECTED_MODEL,
            "reference_threshold": (
                threshold
            ),
            "model_retrained": False,
            "calibration_reselected": False,
            "threshold_reselected": False,
        },
        "bootstrap_protocol": {
            "method": (
                "stratified_bootstrap_with_replacement"
            ),
            "n_resamples": (
                DEFAULT_BOOTSTRAP_RESAMPLES
            ),
            "base_random_state": (
                DEFAULT_BOOTSTRAP_RANDOM_STATE
            ),
            "confidence_level": (
                DEFAULT_CONFIDENCE_LEVEL
            ),
            "interval_method": (
                "percentile"
            ),
            "prevalence_preserved_within_subgroup": (
                True
            ),
        },
        "metrics": list(
            BOOTSTRAP_METRICS
        ),
        "reportable_subgroups": (
            results
        ),
        "prediction_artifact": {
            "path": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": (
                observed_hash
            ),
        },
        "output_table": str(
            OUTPUT_TABLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "test_used": False,
        },
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
        "\nThreshold reselected: False"
    )

    print(
        "Test used           : False"
    )


if __name__ == "__main__":
    main()