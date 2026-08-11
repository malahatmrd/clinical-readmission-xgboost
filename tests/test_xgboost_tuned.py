from __future__ import annotations

import pytest
from xgboost import XGBClassifier

from clinical_readmission.models.xgboost_tuned import (
    TUNED_EARLY_STOPPING_METRIC,
    TUNED_EARLY_STOPPING_ROUNDS,
    build_tuned_early_stopping_xgboost,
    build_tuned_refit_xgboost,
    extract_tuned_parameters,
)


def build_search_summary() -> dict:
    return {
        "best_result": {
            "parameters": {
                "model__n_estimators": 400,
                "model__learning_rate": 0.03,
                "model__max_depth": 5,
                "model__min_child_weight": 1.0,
                "model__subsample": 1.0,
                "model__colsample_bytree": 0.7,
                "model__reg_alpha": 0.1,
                "model__reg_lambda": 2.0,
                "model__scale_pos_weight": 1.5,
            }
        }
    }


def test_extracts_selected_search_parameters() -> None:
    parameters = extract_tuned_parameters(
        build_search_summary()
    )

    assert parameters[
        "n_estimators"
    ] == 400

    assert parameters[
        "learning_rate"
    ] == 0.03

    assert parameters[
        "max_depth"
    ] == 5

    assert parameters[
        "scale_pos_weight"
    ] == 1.5


def test_rejects_incomplete_search_parameters() -> None:
    summary = build_search_summary()

    del summary[
        "best_result"
    ][
        "parameters"
    ][
        "model__max_depth"
    ]

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        extract_tuned_parameters(
            summary
        )


def test_builds_tuned_early_stopping_model() -> None:
    parameters = extract_tuned_parameters(
        build_search_summary()
    )

    model = (
        build_tuned_early_stopping_xgboost(
            parameters
        )
    )

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.n_estimators == 400
    assert model.learning_rate == 0.03
    assert model.max_depth == 5
    assert model.scale_pos_weight == 1.5

    assert (
        model.eval_metric
        == TUNED_EARLY_STOPPING_METRIC
    )

    assert (
        model.early_stopping_rounds
        == TUNED_EARLY_STOPPING_ROUNDS
    )


def test_builds_tuned_refit_model() -> None:
    parameters = extract_tuned_parameters(
        build_search_summary()
    )

    model = build_tuned_refit_xgboost(
        parameters,
        n_estimators=137,
    )

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.n_estimators == 137
    assert model.learning_rate == 0.03
    assert model.max_depth == 5
    assert model.scale_pos_weight == 1.5
    assert model.early_stopping_rounds is None


def test_rejects_invalid_refit_tree_count() -> None:
    parameters = extract_tuned_parameters(
        build_search_summary()
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_tuned_refit_xgboost(
            parameters,
            n_estimators=0,
        )