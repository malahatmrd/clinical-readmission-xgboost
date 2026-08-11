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

from clinical_readmission.models.logistic_baseline import (
    build_logistic_baseline,
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

    model = build_logistic_baseline()

    target_column = "readmitted_30d"

    print("=" * 88)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 88)

    print(
        "\nTrain rows      :",
        len(train),
    )

    print(
        "Validation rows :",
        len(validation),
    )

    print(
        "Train positives :",
        int(train[target_column].sum()),
    )

    print(
        "Validation positives:",
        int(validation[target_column].sum()),
    )

    print(
        "\nFitting baseline on TRAIN only..."
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

    output = {
        "model": (
            "logistic_regression_baseline"
        ),
        "training_policy": {
            "fit_split": "train",
            "evaluation_split": (
                "validation"
            ),
            "test_used": False,
            "class_weight": None,
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