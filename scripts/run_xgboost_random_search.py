from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_readmission.models.xgboost_tuning import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    PARAMETER_DISTRIBUTIONS,
    PRIMARY_SCORER,
    SEARCH_N_ITER,
    SEARCH_N_JOBS,
    SEARCH_RANDOM_STATE,
    build_xgboost_random_search,
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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_random_search_cv.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "xgboost_random_search_cv_results.csv"
)

TOP_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "xgboost_random_search_top10.csv"
)


def load_train_partition(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    train_ids = assignments.loc[
        assignments["split"].eq("train"),
        [
            "encounter_id",
            "patient_nbr",
            "readmitted_30d",
        ],
    ].copy()

    train = cohort.merge(
        train_ids,
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

    target_match = (
        train["readmitted_30d"]
        == train["readmitted_30d_assignment"]
    ).all()

    if not target_match:
        raise ValueError(
            "Train target mismatch between "
            "cohort and split assignments."
        )

    return train


def to_builtin(value):
    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    return value


def clean_parameter_dict(
    parameters: dict,
) -> dict:
    return {
        key: to_builtin(value)
        for key, value in parameters.items()
    }


def build_results_table(
    cv_results: dict,
) -> pd.DataFrame:
    results = pd.DataFrame(
        cv_results
    ).copy()

    parameter_columns = [
        f"param_{parameter}"
        for parameter
        in PARAMETER_DISTRIBUTIONS
    ]

    selected_columns = (
        [
            "rank_test_average_precision",
            "mean_test_average_precision",
            "std_test_average_precision",
            "mean_train_average_precision",
            "std_train_average_precision",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_train_roc_auc",
            "std_train_roc_auc",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
        ]
        + parameter_columns
    )

    results = results[
        selected_columns
    ].copy()

    results[
        "average_precision_gap"
    ] = (
        results[
            "mean_train_average_precision"
        ]
        - results[
            "mean_test_average_precision"
        ]
    )

    results[
        "roc_auc_gap"
    ] = (
        results[
            "mean_train_roc_auc"
        ]
        - results[
            "mean_test_roc_auc"
        ]
    )

    results = results.sort_values(
        by=[
            "rank_test_average_precision",
            "mean_test_average_precision",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return results


def top_candidate_records(
    results: pd.DataFrame,
    count: int = 5,
) -> list[dict]:
    records = []

    for _, row in (
        results.head(count).iterrows()
    ):
        parameters = {}

        for parameter in (
            PARAMETER_DISTRIBUTIONS
        ):
            column = (
                f"param_{parameter}"
            )

            parameters[parameter] = (
                to_builtin(
                    row[column]
                )
            )

        records.append(
            {
                "rank_average_precision": int(
                    row[
                        "rank_test_average_precision"
                    ]
                ),
                "mean_cv_average_precision": float(
                    row[
                        "mean_test_average_precision"
                    ]
                ),
                "std_cv_average_precision": float(
                    row[
                        "std_test_average_precision"
                    ]
                ),
                "mean_train_average_precision": float(
                    row[
                        "mean_train_average_precision"
                    ]
                ),
                "average_precision_gap": float(
                    row[
                        "average_precision_gap"
                    ]
                ),
                "mean_cv_roc_auc": float(
                    row[
                        "mean_test_roc_auc"
                    ]
                ),
                "std_cv_roc_auc": float(
                    row[
                        "std_test_roc_auc"
                    ]
                ),
                "roc_auc_gap": float(
                    row[
                        "roc_auc_gap"
                    ]
                ),
                "parameters": parameters,
            }
        )

    return records


def main() -> None:
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH,
    )

    train = load_train_partition(
        cohort,
        assignments,
    )

    target_column = (
        "readmitted_30d"
    )

    target = train[
        target_column
    ]

    positive_count = int(
        target.sum()
    )

    negative_count = int(
        (target == 0).sum()
    )

    print("=" * 88)
    print("TRAIN-ONLY XGBOOST RANDOMIZED SEARCH")
    print("=" * 88)

    print(
        "\nTrain rows          :",
        len(train),
    )

    print(
        "Train positives     :",
        positive_count,
    )

    print(
        "Train negatives     :",
        negative_count,
    )

    print(
        "Positive prevalence :",
        f"{target.mean():.6f}",
    )

    print(
        "\nCV folds            :",
        CV_N_SPLITS,
    )

    print(
        "CV random state     :",
        CV_RANDOM_STATE,
    )

    print(
        "Search iterations   :",
        SEARCH_N_ITER,
    )

    print(
        "Search random state :",
        SEARCH_RANDOM_STATE,
    )

    print(
        "Parallel search jobs:",
        SEARCH_N_JOBS,
    )

    print(
        "Primary scorer      :",
        PRIMARY_SCORER,
    )

    print(
        "Validation loaded   : False"
    )

    print(
        "Test loaded         : False"
    )

    search = (
        build_xgboost_random_search()
    )

    print(
        "\nRunning Train-only "
        "RandomizedSearchCV..."
    )

    search.fit(
        train,
        target,
    )

    results = build_results_table(
        search.cv_results_
    )

    best_parameters = (
        clean_parameter_dict(
            search.best_params_
        )
    )

    best_index = int(
        search.best_index_
    )

    best_score = float(
        search.best_score_
    )

    best_cv_roc_auc = float(
        search.cv_results_[
            "mean_test_roc_auc"
        ][best_index]
    )

    best_cv_roc_auc_std = float(
        search.cv_results_[
            "std_test_roc_auc"
        ][best_index]
    )

    best_ap_std = float(
        search.cv_results_[
            "std_test_average_precision"
        ][best_index]
    )

    best_train_ap = float(
        search.cv_results_[
            "mean_train_average_precision"
        ][best_index]
    )

    best_ap_gap = (
        best_train_ap
        - best_score
    )

    print(
        "\nBEST TRAIN-CV CONFIGURATION"
    )
    print("-" * 88)

    print(
        "Best CV Average Precision:",
        f"{best_score:.6f}",
    )

    print(
        "CV AP std                :",
        f"{best_ap_std:.6f}",
    )

    print(
        "Best configuration ROC-AUC:",
        f"{best_cv_roc_auc:.6f}",
    )

    print(
        "CV ROC-AUC std            :",
        f"{best_cv_roc_auc_std:.6f}",
    )

    print(
        "Mean Train AP             :",
        f"{best_train_ap:.6f}",
    )

    print(
        "Train-CV AP gap           :",
        f"{best_ap_gap:.6f}",
    )

    print(
        "\nBest parameters:"
    )

    for name, value in sorted(
        best_parameters.items()
    ):
        print(
            f"  {name}: {value}"
        )

    top_candidates = (
        top_candidate_records(
            results,
            count=5,
        )
    )

    print(
        "\nTOP 5 CONFIGURATIONS"
    )
    print("-" * 88)

    for candidate in (
        top_candidates
    ):
        print(
            "Rank",
            candidate[
                "rank_average_precision"
            ],
            "| AP",
            f"{candidate['mean_cv_average_precision']:.6f}",
            "+/-",
            f"{candidate['std_cv_average_precision']:.6f}",
            "| ROC-AUC",
            f"{candidate['mean_cv_roc_auc']:.6f}",
            "| AP gap",
            f"{candidate['average_precision_gap']:.6f}",
        )

    summary = {
        "experiment": (
            "xgboost_randomized_search_cv"
        ),
        "selection_policy": {
            "search_data": (
                "train_only"
            ),
            "validation_used": False,
            "test_used": False,
            "cv_strategy": (
                "StratifiedKFold"
            ),
            "cv_splits": CV_N_SPLITS,
            "cv_random_state": (
                CV_RANDOM_STATE
            ),
            "search_iterations": (
                SEARCH_N_ITER
            ),
            "search_random_state": (
                SEARCH_RANDOM_STATE
            ),
            "primary_metric": (
                PRIMARY_SCORER
            ),
            "secondary_metric": (
                "roc_auc"
            ),
        },
        "sample_counts": {
            "train": int(
                len(train)
            ),
            "positive": positive_count,
            "negative": negative_count,
            "positive_prevalence": float(
                target.mean()
            ),
        },
        "best_result": {
            "best_index": best_index,
            "mean_cv_average_precision": (
                best_score
            ),
            "std_cv_average_precision": (
                best_ap_std
            ),
            "mean_train_average_precision": (
                best_train_ap
            ),
            "average_precision_gap": float(
                best_ap_gap
            ),
            "mean_cv_roc_auc": (
                best_cv_roc_auc
            ),
            "std_cv_roc_auc": (
                best_cv_roc_auc_std
            ),
            "parameters": (
                best_parameters
            ),
        },
        "top_5_candidates": (
            top_candidates
        ),
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    results.head(10).to_csv(
        TOP_RESULTS_PATH,
        index=False,
    )

    print(
        "\nSaved summary:",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved full CV table:",
        RESULTS_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved Top-10 table :",
        TOP_RESULTS_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nValidation used: False"
    )

    print(
        "Test used      : False"
    )


if __name__ == "__main__":
    main()