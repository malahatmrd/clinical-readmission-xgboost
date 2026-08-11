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

from clinical_readmission.models.imbalance import (
    calculate_scale_pos_weight,
)
from clinical_readmission.models.xgboost_weighted import (
    build_weighted_xgboost,
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
    / "xgboost_weighted_validation.json"
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

    scale_pos_weight = (
        calculate_scale_pos_weight(
            train_target
        )
    )

    print("=" * 88)
    print("WEIGHTED XGBOOST")
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
        int(train_target.sum()),
    )

    print(
        "Train negatives     :",
        int(
            (train_target == 0).sum()
        ),
    )

    print(
        "scale_pos_weight    :",
        f"{scale_pos_weight:.6f}",
    )

    model = build_weighted_xgboost(
        scale_pos_weight=scale_pos_weight,
    )

    print(
        "\nFitting weighted XGBoost "
        "on TRAIN only..."
    )

    model.fit(
        train,
        train_target,
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

    logistic_metrics = load_metrics(
        LOGISTIC_PATH
    )

    baseline_metrics = load_metrics(
        XGBOOST_BASELINE_PATH
    )

    comparison = {
        "vs_logistic": {
            "roc_auc_delta": float(
                metrics["roc_auc"]
                - logistic_metrics["roc_auc"]
            ),
            "average_precision_delta": float(
                metrics["average_precision"]
                - logistic_metrics[
                    "average_precision"
                ]
            ),
            "brier_score_delta": float(
                metrics["brier_score"]
                - logistic_metrics[
                    "brier_score"
                ]
            ),
            "log_loss_delta": float(
                metrics["log_loss"]
                - logistic_metrics[
                    "log_loss"
                ]
            ),
        },
        "vs_unweighted_xgboost": {
            "roc_auc_delta": float(
                metrics["roc_auc"]
                - baseline_metrics["roc_auc"]
            ),
            "average_precision_delta": float(
                metrics["average_precision"]
                - baseline_metrics[
                    "average_precision"
                ]
            ),
            "brier_score_delta": float(
                metrics["brier_score"]
                - baseline_metrics[
                    "brier_score"
                ]
            ),
            "log_loss_delta": float(
                metrics["log_loss"]
                - baseline_metrics[
                    "log_loss"
                ]
            ),
        },
    }

    fitted_preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    transformed_features = len(
        fitted_preprocessor
        .get_feature_names_out()
    )

    output = {
        "model": "xgboost_weighted",
        "training_policy": {
            "fit_split": "train",
            "evaluation_split": "validation",
            "test_used": False,
            "scale_pos_weight_source": (
                "train_negative_count/"
                "train_positive_count"
            ),
            "early_stopping": False,
            "hyperparameter_tuning": False,
        },
        "scale_pos_weight": float(
            scale_pos_weight
        ),
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
        "\nVS LOGISTIC"
    )
    print("-" * 88)

    for name, value in (
        comparison[
            "vs_logistic"
        ].items()
    ):
        print(
            f"{name:32}: "
            f"{value:+.6f}"
        )

    print(
        "\nVS UNWEIGHTED XGBOOST"
    )
    print("-" * 88)

    for name, value in (
        comparison[
            "vs_unweighted_xgboost"
        ].items()
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