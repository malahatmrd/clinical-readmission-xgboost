from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from clinical_readmission.models.xgboost_baseline import (
    BASELINE_LEARNING_RATE,
    BASELINE_MAX_DEPTH,
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


def test_internal_split_is_reproducible() -> None:
    target = pd.Series(
        [0] * 80
        + [1] * 20
    )

    fit_a, stop_a = (
        build_internal_early_stopping_split(
            target
        )
    )

    fit_b, stop_b = (
        build_internal_early_stopping_split(
            target
        )
    )

    assert np.array_equal(
        fit_a,
        fit_b,
    )

    assert np.array_equal(
        stop_a,
        stop_b,
    )


def test_internal_split_is_stratified_and_disjoint() -> None:
    target = np.array(
        [0] * 80
        + [1] * 20
    )

    fit_indices, stop_indices = (
        build_internal_early_stopping_split(
            target
        )
    )

    assert len(fit_indices) == 80
    assert len(stop_indices) == 20

    assert not (
        set(fit_indices)
        & set(stop_indices)
    )

    assert len(
        set(fit_indices)
        | set(stop_indices)
    ) == 100

    assert int(
        target[fit_indices].sum()
    ) == 16

    assert int(
        target[stop_indices].sum()
    ) == 4


def test_builds_expected_early_stopping_model() -> None:
    model = (
        build_early_stopping_xgboost()
    )

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert (
        model.n_estimators
        == EARLY_STOPPING_N_ESTIMATORS
    )

    assert (
        model.early_stopping_rounds
        == EARLY_STOPPING_ROUNDS
    )

    assert model.eval_metric == "aucpr"
    assert model.scale_pos_weight == 1.0

    assert (
        model.max_depth
        == BASELINE_MAX_DEPTH
    )

    assert (
        model.learning_rate
        == BASELINE_LEARNING_RATE
    )


def test_builds_expected_refit_model() -> None:
    model = build_refit_xgboost(
        n_estimators=137
    )

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.n_estimators == 137
    assert model.scale_pos_weight == 1.0
    assert model.early_stopping_rounds is None


def test_rejects_invalid_refit_tree_count() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_refit_xgboost(
            n_estimators=0
        )


def test_protocol_constants_are_frozen() -> None:
    assert INTERNAL_VALIDATION_SIZE == 0.20
    assert INTERNAL_SPLIT_RANDOM_STATE == 44