from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clinical_readmission.models.xgboost_tuning import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    PARAMETER_DISTRIBUTIONS,
    PRIMARY_SCORER,
    SCORING,
    SEARCH_N_ITER,
    SEARCH_N_JOBS,
    SEARCH_RANDOM_STATE,
    XGBOOST_N_JOBS,
    build_tuning_pipeline,
    build_xgboost_random_search,
)


def test_builds_expected_tuning_pipeline() -> None:
    pipeline = build_tuning_pipeline()

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert list(
        pipeline.named_steps
    ) == [
        "preprocessor",
        "model",
    ]

    model = pipeline.named_steps["model"]

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.tree_method == "hist"
    assert model.n_jobs == XGBOOST_N_JOBS

    assert (
        model.early_stopping_rounds
        is None
    )


def test_builds_expected_random_search() -> None:
    search = (
        build_xgboost_random_search()
    )

    assert isinstance(
        search,
        RandomizedSearchCV,
    )

    assert search.n_iter == SEARCH_N_ITER

    assert (
        search.random_state
        == SEARCH_RANDOM_STATE
    )

    assert search.n_jobs == SEARCH_N_JOBS

    assert search.refit == PRIMARY_SCORER

    assert search.scoring == SCORING

    assert search.error_score == "raise"

    assert search.return_train_score is True


def test_search_uses_frozen_stratified_cv() -> None:
    search = (
        build_xgboost_random_search()
    )

    cross_validation = search.cv

    assert isinstance(
        cross_validation,
        StratifiedKFold,
    )

    assert (
        cross_validation.n_splits
        == CV_N_SPLITS
    )

    assert cross_validation.shuffle is True

    assert (
        cross_validation.random_state
        == CV_RANDOM_STATE
    )


def test_cross_validation_is_reproducible() -> None:
    target = np.array(
        [0] * 80
        + [1] * 20
    )

    search_a = (
        build_xgboost_random_search()
    )

    search_b = (
        build_xgboost_random_search()
    )

    splits_a = list(
        search_a.cv.split(
            np.zeros(
                (len(target), 1)
            ),
            target,
        )
    )

    splits_b = list(
        search_b.cv.split(
            np.zeros(
                (len(target), 1)
            ),
            target,
        )
    )

    assert len(splits_a) == CV_N_SPLITS

    for (
        train_a,
        validation_a,
    ), (
        train_b,
        validation_b,
    ) in zip(
        splits_a,
        splits_b,
        strict=True,
    ):
        assert np.array_equal(
            train_a,
            train_b,
        )

        assert np.array_equal(
            validation_a,
            validation_b,
        )


def test_search_space_contains_expected_parameters() -> None:
    expected = {
        "model__n_estimators",
        "model__learning_rate",
        "model__max_depth",
        "model__min_child_weight",
        "model__subsample",
        "model__colsample_bytree",
        "model__reg_alpha",
        "model__reg_lambda",
        "model__scale_pos_weight",
    }

    assert (
        set(PARAMETER_DISTRIBUTIONS)
        == expected
    )


def test_search_space_includes_unweighted_model() -> None:
    weights = (
        PARAMETER_DISTRIBUTIONS[
            "model__scale_pos_weight"
        ]
    )

    assert 1.0 in weights
    assert 10.0 in weights