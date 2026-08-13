from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_operating_scenarios.json"
)

BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_operating_scenario_bootstrap.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase8_threshold_selection.json"
)

EXPECTED_MODEL = (
    "tuned_xgboost_sigmoid"
)

SELECTED_SCENARIO = (
    "moderate_capacity"
)

EXPECTED_THRESHOLD = 0.105
CAPACITY_LIMIT = 20.0


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    scenarios = load_json(
        SCENARIOS_PATH
    )

    bootstrap = load_json(
        BOOTSTRAP_PATH
    )

    if (
        scenarios[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected selected model."
        )

    if (
        bootstrap[
            "selected_model"
        ]
        != EXPECTED_MODEL
    ):
        raise ValueError(
            "Unexpected bootstrap model."
        )

    if (
        scenarios[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
        )

    if (
        bootstrap[
            "data_policy"
        ][
            "test_used"
        ]
    ):
        raise ValueError(
            "Test data must remain locked."
        )

    scenario = (
        scenarios[
            "scenarios"
        ][
            SELECTED_SCENARIO
        ]
    )

    bootstrap_scenario = (
        bootstrap[
            "scenarios"
        ][
            SELECTED_SCENARIO
        ]
    )

    threshold = float(
        scenario[
            "threshold"
        ]
    )

    if abs(
        threshold
        - EXPECTED_THRESHOLD
    ) > 1e-12:
        raise ValueError(
            "Unexpected moderate-capacity "
            "threshold."
        )

    if (
        scenario[
            "alerts_per_100"
        ]
        > CAPACITY_LIMIT
    ):
        raise ValueError(
            "Selected threshold exceeds "
            "the predefined alert capacity."
        )

    alert_ci_upper = (
        bootstrap_scenario[
            "metrics"
        ][
            "alerts_per_100"
        ][
            "ci_upper"
        ]
    )

    if (
        alert_ci_upper
        > CAPACITY_LIMIT
    ):
        raise ValueError(
            "Upper bootstrap alert-rate CI "
            "exceeds predefined capacity."
        )

    output = {
        "phase": 8,
        "decision": (
            "reference_operating_threshold"
        ),
        "selected_model": (
            EXPECTED_MODEL
        ),
        "selected_scenario": (
            SELECTED_SCENARIO
        ),
        "reference_threshold": (
            threshold
        ),
        "selection_interpretation": (
            "reference moderate-capacity "
            "operating point; not claimed "
            "to be universally clinically optimal"
        ),
        "predefined_constraint": {
            "metric": (
                "alerts_per_100"
            ),
            "maximum": (
                CAPACITY_LIMIT
            ),
            "selection_rule": (
                "lowest threshold satisfying "
                "the alert-capacity constraint"
            ),
        },
        "validation_operating_metrics": {
            "sensitivity": float(
                scenario[
                    "sensitivity"
                ]
            ),
            "specificity": float(
                scenario[
                    "specificity"
                ]
            ),
            "ppv": float(
                scenario[
                    "ppv"
                ]
            ),
            "npv": float(
                scenario[
                    "npv"
                ]
            ),
            "f1": float(
                scenario[
                    "f1"
                ]
            ),
            "alerts_per_100": float(
                scenario[
                    "alerts_per_100"
                ]
            ),
            "number_needed_to_evaluate": float(
                scenario[
                    "number_needed_to_evaluate"
                ]
            ),
            "model_net_benefit": float(
                scenario[
                    "model_net_benefit"
                ]
            ),
        },
        "bootstrap_95_ci": {
            metric: {
                "ci_lower": float(
                    values[
                        "ci_lower"
                    ]
                ),
                "ci_upper": float(
                    values[
                        "ci_upper"
                    ]
                ),
            }
            for metric, values in (
                bootstrap_scenario[
                    "metrics"
                ].items()
            )
        },
        "data_policy": {
            "selection_split": (
                "validation"
            ),
            "model_frozen": True,
            "calibration_frozen": True,
            "threshold_frozen": True,
            "test_used": False,
        },
        "source_artifacts": {
            "operating_scenarios": str(
                SCENARIOS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "operating_bootstrap": str(
                BOOTSTRAP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
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

    print("=" * 96)
    print("PHASE 8 REFERENCE THRESHOLD SELECTION")
    print("=" * 96)

    print(
        "\nSelected model    :",
        EXPECTED_MODEL,
    )

    print(
        "Selected scenario :",
        SELECTED_SCENARIO,
    )

    print(
        "Reference threshold:",
        f"{threshold:.3f}",
    )

    print(
        "\nValidation operating metrics"
    )

    print(
        "  Sensitivity :",
        f"{scenario['sensitivity']:.4f}",
    )

    print(
        "  Specificity :",
        f"{scenario['specificity']:.4f}",
    )

    print(
        "  PPV         :",
        f"{scenario['ppv']:.4f}",
    )

    print(
        "  NPV         :",
        f"{scenario['npv']:.4f}",
    )

    print(
        "  Alerts / 100:",
        f"{scenario['alerts_per_100']:.4f}",
    )

    print(
        "  NNE         :",
        f"{scenario['number_needed_to_evaluate']:.4f}",
    )

    print(
        "  Net benefit :",
        f"{scenario['model_net_benefit']:.4f}",
    )

    print(
        "\nAlert-rate 95% CI:",
        (
            f"[{bootstrap_scenario['metrics']['alerts_per_100']['ci_lower']:.4f}, "
            f"{bootstrap_scenario['metrics']['alerts_per_100']['ci_upper']:.4f}]"
        ),
    )

    print(
        "\nThreshold frozen : True"
    )

    print(
        "Test used        : False"
    )

    print(
        "\nSaved selection :",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()