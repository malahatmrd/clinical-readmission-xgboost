from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase6_model_comparison.csv"
)

CHAMPION_PATH = (
    METRICS_ROOT
    / "phase6_champion.json"
)

SELECTION_METRIC = "average_precision"

MODEL_SOURCES = {
    "logistic_regression": (
        METRICS_ROOT
        / "logistic_baseline_validation.json"
    ),
    "xgboost_baseline": (
        METRICS_ROOT
        / "xgboost_baseline_validation.json"
    ),
    "xgboost_weighted": (
        METRICS_ROOT
        / "xgboost_weighted_validation.json"
    ),
    "xgboost_early_stopping": (
        METRICS_ROOT
        / "xgboost_early_stopping_validation.json"
    ),
    "xgboost_tuned": (
        METRICS_ROOT
        / "xgboost_tuned_validation.json"
    ),
}


def load_metrics(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return payload["metrics"]


def build_comparison_table() -> pd.DataFrame:
    rows = []

    for (
        model_name,
        metrics_path,
    ) in MODEL_SOURCES.items():
        metrics = load_metrics(
            metrics_path
        )

        rows.append(
            {
                "model": model_name,
                "roc_auc": float(
                    metrics["roc_auc"]
                ),
                "average_precision": float(
                    metrics[
                        "average_precision"
                    ]
                ),
                "brier_score": float(
                    metrics["brier_score"]
                ),
                "log_loss": float(
                    metrics["log_loss"]
                ),
                "precision_at_0_5": float(
                    metrics[
                        "precision_at_0_5"
                    ]
                ),
                "recall_at_0_5": float(
                    metrics[
                        "recall_at_0_5"
                    ]
                ),
                "f1_at_0_5": float(
                    metrics[
                        "f1_at_0_5"
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison = comparison.sort_values(
        by=SELECTION_METRIC,
        ascending=False,
    ).reset_index(
        drop=True
    )

    comparison.insert(
        0,
        "development_rank",
        range(
            1,
            len(comparison) + 1,
        ),
    )

    return comparison


def main() -> None:
    comparison = (
        build_comparison_table()
    )

    champion = comparison.iloc[0]

    COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    champion_payload = {
        "phase": 6,
        "selection_policy": {
            "primary_metric": (
                SELECTION_METRIC
            ),
            "higher_is_better": True,
            "selection_split": (
                "validation"
            ),
            "test_used": False,
            "threshold_optimized": False,
        },
        "champion": {
            "model": str(
                champion["model"]
            ),
            "average_precision": float(
                champion[
                    "average_precision"
                ]
            ),
            "roc_auc": float(
                champion["roc_auc"]
            ),
            "brier_score": float(
                champion["brier_score"]
            ),
            "log_loss": float(
                champion["log_loss"]
            ),
        },
        "candidate_count": int(
            len(comparison)
        ),
        "comparison_table": str(
            COMPARISON_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "source_metrics": {
            model: str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
            for model, path in (
                MODEL_SOURCES.items()
            )
        },
    }

    with CHAMPION_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            champion_payload,
            file,
            indent=2,
        )

    print("=" * 88)
    print("PHASE 6 MODEL COMPARISON")
    print("=" * 88)

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nChampion model      :",
        champion["model"],
    )

    print(
        "Selection metric    :",
        SELECTION_METRIC,
    )

    print(
        "Champion AP         :",
        f"{champion['average_precision']:.6f}",
    )

    print(
        "Champion ROC-AUC    :",
        f"{champion['roc_auc']:.6f}",
    )

    print(
        "Test used           : False"
    )

    print(
        "\nSaved comparison:",
        COMPARISON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved champion  :",
        CHAMPION_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()