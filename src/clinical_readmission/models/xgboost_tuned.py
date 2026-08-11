from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier

TUNED_EARLY_STOPPING_ROUNDS = 50
TUNED_EARLY_STOPPING_METRIC = "aucpr"
TUNED_RANDOM_STATE = 42

SEARCH_PARAMETER_PREFIX = "model__"

REQUIRED_TUNED_PARAMETERS = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
)


def extract_tuned_parameters(
    search_summary: dict[str, Any],
) -> dict[str, Any]:
    """Extract the selected XGBoost parameters from search artifact."""

    try:
        raw_parameters = (
            search_summary[
                "best_result"
            ][
                "parameters"
            ]
        )
    except KeyError as error:
        raise ValueError(
            "Search summary does not contain "
            "best_result.parameters."
        ) from error

    parameters = {}

    for name, value in (
        raw_parameters.items()
    ):
        if not name.startswith(
            SEARCH_PARAMETER_PREFIX
        ):
            continue

        clean_name = name.removeprefix(
            SEARCH_PARAMETER_PREFIX
        )

        parameters[
            clean_name
        ] = value

    missing = sorted(
        set(REQUIRED_TUNED_PARAMETERS)
        - set(parameters)
    )

    if missing:
        raise ValueError(
            "Selected tuning parameters are missing: "
            f"{missing}"
        )

    return {
        name: parameters[name]
        for name in REQUIRED_TUNED_PARAMETERS
    }


def build_tuned_early_stopping_xgboost(
    parameters: dict[str, Any],
) -> XGBClassifier:
    """Build the selected model for Train-only early stopping."""

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(
            parameters[
                "n_estimators"
            ]
        ),
        learning_rate=float(
            parameters[
                "learning_rate"
            ]
        ),
        max_depth=int(
            parameters[
                "max_depth"
            ]
        ),
        min_child_weight=float(
            parameters[
                "min_child_weight"
            ]
        ),
        subsample=float(
            parameters[
                "subsample"
            ]
        ),
        colsample_bytree=float(
            parameters[
                "colsample_bytree"
            ]
        ),
        reg_alpha=float(
            parameters[
                "reg_alpha"
            ]
        ),
        reg_lambda=float(
            parameters[
                "reg_lambda"
            ]
        ),
        scale_pos_weight=float(
            parameters[
                "scale_pos_weight"
            ]
        ),
        tree_method="hist",
        eval_metric=TUNED_EARLY_STOPPING_METRIC,
        early_stopping_rounds=(
            TUNED_EARLY_STOPPING_ROUNDS
        ),
        random_state=TUNED_RANDOM_STATE,
        n_jobs=-1,
    )


def build_tuned_refit_xgboost(
    parameters: dict[str, Any],
    n_estimators: int,
) -> XGBClassifier:
    """Build the selected model for refitting on full Train."""

    if n_estimators <= 0:
        raise ValueError(
            "n_estimators must be greater than zero."
        )

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        learning_rate=float(
            parameters[
                "learning_rate"
            ]
        ),
        max_depth=int(
            parameters[
                "max_depth"
            ]
        ),
        min_child_weight=float(
            parameters[
                "min_child_weight"
            ]
        ),
        subsample=float(
            parameters[
                "subsample"
            ]
        ),
        colsample_bytree=float(
            parameters[
                "colsample_bytree"
            ]
        ),
        reg_alpha=float(
            parameters[
                "reg_alpha"
            ]
        ),
        reg_lambda=float(
            parameters[
                "reg_lambda"
            ]
        ),
        scale_pos_weight=float(
            parameters[
                "scale_pos_weight"
            ]
        ),
        tree_method="hist",
        eval_metric="logloss",
        random_state=TUNED_RANDOM_STATE,
        n_jobs=-1,
    )