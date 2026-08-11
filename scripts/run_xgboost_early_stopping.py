from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)
from clinical_readmission.models.xgboost_early_stopping import (
    EARLY_STOPPING_N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    INTERNAL_SPLIT_RANDOM_STATE,
    INTERNAL_VALIDATION_SIZE,
    build_early_stopping_xgboost,
    build_internal_early_stopping_split,
    build_refit_xgboost,
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_early_stopping_validation.json"
)

LOGISTIC_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "logistic_baseline_validation.json"
)

XGBOOST_BASELINE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_baseline_validation.json"
)

WEIGHTED_XGBOOST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_weighted_validation.json"
)


def load_partition(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    ids = assignments.loc[
        assignments["split"].eq(split_name),
        [
            "encounter_id",
            "patient_nbr",
            "readmitted_30d",
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

    target_match = (
        result["readmitted_30d"]
        == result["readmitted_30d_assignment"]
    ).all()

    if not target_match:
        raise ValueError(
            f"{split_name}: target mismatch "
            "between cohort and assignments."
        )

    return result


def calculate_metrics(
    target: pd.Series,
    probabilities,
) -> dict:
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(
                target,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                target,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                target,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                target,
                probabilities,
            )
        ),
        "threshold": 0.5,
        "precision_at_0_5": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall_at_0_5": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def load_metrics(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)["metrics"]


def metric_deltas(
    current: dict,
    reference: dict,
) -> dict:
    return {
        "roc_auc_delta": float(
            current["roc_auc"]
            - reference["roc_auc"]
        ),
        "average_precision_delta": float(
            current["average_precision"]
            - reference[
                "average_precision"
            ]
        ),
        "brier_score_delta": float(
            current["brier_score"]
            - reference["brier_score"]
        ),
        "log_loss_delta": float(
            current["log_loss"]
            - reference["log_loss"]
        ),
    }


def main() -> None:
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH,
    )

    train = load_partition(
        cohort,
        assignments,
        "train",
    )

    validation = load_partition(
        cohort,
        assignments,
        "validation",
    )

    target_column = "readmitted_30d"

    train_target = train[
        target_column
    ]

    fit_indices, stop_indices = (
        build_internal_early_stopping_split(
            train_target
        )
    )

    internal_fit = train.iloc[
        fit_indices
    ].copy()

    internal_stop = train.iloc[
        stop_indices
    ].copy()

    fit_target = internal_fit[
        target_column
    ]

    stop_target = internal_stop[
        target_column
    ]

    print("=" * 88)
    print("XGBOOST EARLY-STOPPING DEVELOPMENT")
    print("=" * 88)

    print(
        "\nFull Train rows      :",
        len(train),
    )

    print(
        "Internal Fit rows    :",
        len(internal_fit),
    )

    print(
        "Internal Stop rows   :",
        len(internal_stop),
    )

    print(
        "Validation rows      :",
        len(validation),
    )

    print(
        "\nInternal Fit positives :",
        int(fit_target.sum()),
    )

    print(
        "Internal Stop positives:",
        int(stop_target.sum()),
    )

    print(
        "\nInternal validation size:",
        INTERNAL_VALIDATION_SIZE,
    )

    print(
        "Internal split seed     :",
        INTERNAL_SPLIT_RANDOM_STATE,
    )

    print(
        "Maximum boosting rounds :",
        EARLY_STOPPING_N_ESTIMATORS,
    )

    print(
        "Early stopping rounds   :",
        EARLY_STOPPING_ROUNDS,
    )

    print(
        "\nFitting preprocessing "
        "on INTERNAL FIT only..."
    )

    development_preprocessor = (
        build_preprocessor()
    )

    x_fit = (
        development_preprocessor
        .fit_transform(
            internal_fit
        )
    )

    x_stop = (
        development_preprocessor
        .transform(
            internal_stop
        )
    )

    print(
        "Internal transformed shape:",
        x_fit.shape,
    )

    development_model = (
        build_early_stopping_xgboost()
    )

    print(
        "\nRunning Train-only "
        "early stopping..."
    )

    development_model.fit(
        x_fit,
        fit_target,
        eval_set=[
            (
                x_stop,
                stop_target,
            )
        ],
        verbose=False,
    )

    best_iteration = int(
        development_model.best_iteration
    )

    best_score = float(
        development_model.best_score
    )

    selected_tree_count = (
        best_iteration + 1
    )

    boosted_rounds = int(
        development_model
        .get_booster()
        .num_boosted_rounds()
    )

    print(
        "\nEARLY-STOPPING RESULT"
    )
    print("-" * 88)

    print(
        "Best iteration       :",
        best_iteration,
    )

    print(
        "Selected tree count  :",
        selected_tree_count,
    )

    print(
        "Best internal AUCPR  :",
        f"{best_score:.6f}",
    )

    print(
        "Rounds actually built:",
        boosted_rounds,
    )

    print(
        "\nRefitting preprocessing "
        "on FULL TRAIN..."
    )

    final_preprocessor = (
        build_preprocessor()
    )

    x_train_full = (
        final_preprocessor
        .fit_transform(
            train
        )
    )

    x_validation = (
        final_preprocessor
        .transform(
            validation
        )
    )

    transformed_features = len(
        final_preprocessor
        .get_feature_names_out()
    )

    final_model = (
        build_refit_xgboost(
            n_estimators=(
                selected_tree_count
            )
        )
    )

    print(
        "Refitting XGBoost on "
        "FULL TRAIN..."
    )

    final_model.fit(
        x_train_full,
        train_target,
    )

    print(
        "Generating VALIDATION "
        "probabilities..."
    )

    probabilities = (
        final_model.predict_proba(
            x_validation
        )[:, 1]
    )

    metrics = calculate_metrics(
        validation[target_column],
        probabilities,
    )

    logistic_metrics = load_metrics(
        LOGISTIC_PATH
    )

    xgboost_baseline_metrics = (
        load_metrics(
            XGBOOST_BASELINE_PATH
        )
    )

    weighted_metrics = load_metrics(
        WEIGHTED_XGBOOST_PATH
    )

    comparison = {
        "vs_logistic": metric_deltas(
            metrics,
            logistic_metrics,
        ),
        "vs_xgboost_baseline": (
            metric_deltas(
                metrics,
                xgboost_baseline_metrics,
            )
        ),
        "vs_weighted_xgboost": (
            metric_deltas(
                metrics,
                weighted_metrics,
            )
        ),
    }

    output = {
        "model": (
            "xgboost_early_stopping_refit"
        ),
        "development_protocol": {
            "internal_split_source": (
                "train_only"
            ),
            "internal_validation_size": (
                INTERNAL_VALIDATION_SIZE
            ),
            "internal_split_random_state": (
                INTERNAL_SPLIT_RANDOM_STATE
            ),
            "internal_fit_rows": int(
                len(internal_fit)
            ),
            "internal_stop_rows": int(
                len(internal_stop)
            ),
            "internal_fit_positive": int(
                fit_target.sum()
            ),
            "internal_stop_positive": int(
                stop_target.sum()
            ),
            "preprocessor_fit_for_stopping": (
                "internal_fit_only"
            ),
            "early_stopping_metric": (
                "aucpr"
            ),
            "early_stopping_rounds": (
                EARLY_STOPPING_ROUNDS
            ),
            "maximum_boosting_rounds": (
                EARLY_STOPPING_N_ESTIMATORS
            ),
            "best_iteration_zero_based": (
                best_iteration
            ),
            "selected_tree_count": (
                selected_tree_count
            ),
            "best_internal_aucpr": (
                best_score
            ),
            "rounds_actually_built": (
                boosted_rounds
            ),
        },
        "refit_policy": {
            "preprocessor_fit_split": (
                "full_train"
            ),
            "model_fit_split": (
                "full_train"
            ),
            "evaluation_split": (
                "validation"
            ),
            "test_used": False,
            "scale_pos_weight": 1.0,
            "hyperparameter_tuning": False,
        },
        "sample_counts": {
            "train": int(
                len(train)
            ),
            "validation": int(
                len(validation)
            ),
            "train_positive": int(
                train_target.sum()
            ),
            "validation_positive": int(
                validation[
                    target_column
                ].sum()
            ),
        },
        "transformed_features": int(
            transformed_features
        ),
        "metrics": metrics,
        "comparison": comparison,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nVALIDATION METRICS"
    )
    print("-" * 88)

    for name, value in metrics.items():
        if isinstance(
            value,
            float,
        ):
            print(
                f"{name:22}: "
                f"{value:.6f}"
            )
        else:
            print(
                f"{name:22}: "
                f"{value}"
            )

    for comparison_name, values in (
        comparison.items()
    ):
        print(
            "\n"
            + comparison_name.upper()
        )
        print("-" * 88)

        for name, value in (
            values.items()
        ):
            print(
                f"{name:32}: "
                f"{value:+.6f}"
            )

    print(
        "\nTransformed features:",
        transformed_features,
    )

    print(
        "Test set used        : False"
    )

    print(
        "\nSaved metrics:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()