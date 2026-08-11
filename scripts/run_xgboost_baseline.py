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

from clinical_readmission.models.xgboost_baseline import (
    build_xgboost_baseline,
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

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_baseline_validation.json"
)

LOGISTIC_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "logistic_baseline_validation.json"
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


def load_logistic_metrics() -> dict:
    if not LOGISTIC_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Logistic baseline metrics not found. "
            "Run scripts/run_logistic_baseline.py first."
        )

    with LOGISTIC_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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

    model = build_xgboost_baseline()

    print("=" * 88)
    print("XGBOOST BASELINE")
    print("=" * 88)

    print(
        "\nTrain rows          :",
        len(train),
    )

    print(
        "Validation rows     :",
        len(validation),
    )

    print(
        "Train positives     :",
        int(train[target_column].sum()),
    )

    print(
        "Validation positives:",
        int(validation[target_column].sum()),
    )

    print(
        "\nFitting XGBoost on TRAIN only..."
    )

    model.fit(
        train,
        train[target_column],
    )

    print(
        "Generating VALIDATION probabilities..."
    )

    probabilities = (
        model.predict_proba(
            validation
        )[:, 1]
    )

    metrics = calculate_metrics(
        validation[target_column],
        probabilities,
    )

    fitted_preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    logistic_output = (
        load_logistic_metrics()
    )

    logistic_metrics = (
        logistic_output["metrics"]
    )

    comparison = {
        "roc_auc_delta_vs_logistic": float(
            metrics["roc_auc"]
            - logistic_metrics["roc_auc"]
        ),
        "average_precision_delta_vs_logistic": float(
            metrics["average_precision"]
            - logistic_metrics["average_precision"]
        ),
        "brier_score_delta_vs_logistic": float(
            metrics["brier_score"]
            - logistic_metrics["brier_score"]
        ),
        "log_loss_delta_vs_logistic": float(
            metrics["log_loss"]
            - logistic_metrics["log_loss"]
        ),
    }

    output = {
        "model": "xgboost_baseline",
        "training_policy": {
            "fit_split": "train",
            "evaluation_split": "validation",
            "test_used": False,
            "class_weighting": False,
            "scale_pos_weight": None,
            "early_stopping": False,
            "hyperparameter_tuning": False,
            "threshold_policy": (
                "0.5 descriptive only; "
                "not optimized"
            ),
        },
        "sample_counts": {
            "train": int(
                len(train)
            ),
            "validation": int(
                len(validation)
            ),
            "train_positive": int(
                train[
                    target_column
                ].sum()
            ),
            "validation_positive": int(
                validation[
                    target_column
                ].sum()
            ),
        },
        "transformed_features": int(
            len(feature_names)
        ),
        "metrics": metrics,
        "comparison_with_logistic": (
            comparison
        ),
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\nVALIDATION METRICS")
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

    print(
        "\nCOMPARISON VS LOGISTIC"
    )
    print("-" * 88)

    for name, value in comparison.items():
        print(
            f"{name:40}: "
            f"{value:+.6f}"
        )

    print(
        "\nTransformed features:",
        len(feature_names),
    )

    print(
        "Test set used        : False"
    )

    print(
        "\nSaved metrics:",
        METRICS_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()