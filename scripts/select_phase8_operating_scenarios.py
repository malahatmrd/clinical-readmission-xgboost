from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clinical_readmission.evaluation.thresholds import (
    select_threshold_for_alert_capacity,
    select_threshold_for_minimum_sensitivity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase8_validation_threshold_sweep.csv"
)

SWEEP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_validation_threshold_sweep.json"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "phase8_operating_scenarios.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_operating_scenarios.json"
)

EXPECTED_MODEL = "tuned_xgboost_sigmoid"

HIGH_SENSITIVITY_TARGET = 0.80
MODERATE_CAPACITY_LIMIT = 20.0
LIMITED_CAPACITY_LIMIT = 10.0


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def row_to_dict(
    row: pd.Series,
) -> dict:
    output = {}

    for key, value in row.items():
        if pd.isna(value):
            output[key] = None
        elif isinstance(
            value,
            (int, float),
        ):
            output[key] = float(
                value
            )
        else:
            output[key] = value

    return output


def main() -> None:
    sweep_table = pd.read_csv(
        SWEEP_TABLE_PATH
    )

    sweep_summary = load_json(
        SWEEP_SUMMARY_PATH
    )

    if (
        sweep_summary[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected Phase 8 model."
        )

    if (
        sweep_summary[
            "data_policy"
        ][
            "threshold_selected"
        ]
    ):
        raise ValueError(
            "Threshold was already marked "
            "as selected unexpectedly."
        )

    if (
        sweep_summary[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
        )

    high_sensitivity = (
        select_threshold_for_minimum_sensitivity(
            sweep_table,
            minimum_sensitivity=(
                HIGH_SENSITIVITY_TARGET
            ),
        )
    )

    moderate_capacity = (
        select_threshold_for_alert_capacity(
            sweep_table,
            maximum_alerts_per_100=(
                MODERATE_CAPACITY_LIMIT
            ),
        )
    )

    limited_capacity = (
        select_threshold_for_alert_capacity(
            sweep_table,
            maximum_alerts_per_100=(
                LIMITED_CAPACITY_LIMIT
            ),
        )
    )

    scenarios = [
        {
            "scenario": (
                "high_sensitivity"
            ),
            "selection_rule": (
                "highest_threshold_with_"
                "sensitivity_at_least_0.80"
            ),
            "constraint_value": (
                HIGH_SENSITIVITY_TARGET
            ),
            **row_to_dict(
                high_sensitivity
            ),
        },
        {
            "scenario": (
                "moderate_capacity"
            ),
            "selection_rule": (
                "lowest_threshold_with_"
                "alerts_per_100_at_most_20"
            ),
            "constraint_value": (
                MODERATE_CAPACITY_LIMIT
            ),
            **row_to_dict(
                moderate_capacity
            ),
        },
        {
            "scenario": (
                "limited_capacity"
            ),
            "selection_rule": (
                "lowest_threshold_with_"
                "alerts_per_100_at_most_10"
            ),
            "constraint_value": (
                LIMITED_CAPACITY_LIMIT
            ),
            **row_to_dict(
                limited_capacity
            ),
        },
    ]

    scenario_table = pd.DataFrame(
        scenarios
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_table.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    output = {
        "phase": 8,
        "analysis": (
            "validation_operating_scenarios"
        ),
        "selected_model": (
            EXPECTED_MODEL
        ),
        "scenario_definitions": {
            "high_sensitivity": {
                "minimum_sensitivity": (
                    HIGH_SENSITIVITY_TARGET
                ),
                "selection_rule": (
                    "highest eligible threshold"
                ),
            },
            "moderate_capacity": {
                "maximum_alerts_per_100": (
                    MODERATE_CAPACITY_LIMIT
                ),
                "selection_rule": (
                    "lowest eligible threshold"
                ),
            },
            "limited_capacity": {
                "maximum_alerts_per_100": (
                    LIMITED_CAPACITY_LIMIT
                ),
                "selection_rule": (
                    "lowest eligible threshold"
                ),
            },
        },
        "scenarios": {
            scenario[
                "scenario"
            ]: scenario
            for scenario in scenarios
        },
        "data_policy": {
            "evaluation_split": (
                "validation"
            ),
            "scenarios_predefined": True,
            "reference_threshold_selected": False,
            "test_used": False,
        },
        "source_artifacts": {
            "threshold_sweep_table": str(
                SWEEP_TABLE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "threshold_sweep_summary": str(
                SWEEP_SUMMARY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
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

    display_columns = [
        "scenario",
        "threshold",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "alerts_per_100",
        "number_needed_to_evaluate",
        "model_net_benefit",
        "treat_all_net_benefit",
    ]

    print("=" * 104)
    print("PHASE 8 VALIDATION OPERATING SCENARIOS")
    print("=" * 104)

    print(
        "\nSelected model:",
        EXPECTED_MODEL,
    )

    print(
        "\nPREDEFINED OPERATING SCENARIOS"
    )

    print("-" * 104)

    print(
        scenario_table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nSaved table :",
        OUTPUT_TABLE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved JSON  :",
        OUTPUT_JSON_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "\nReference threshold selected: False"
    )

    print(
        "Test used                   : False"
    )


if __name__ == "__main__":
    main()