from __future__ import annotations

import numpy as np
import pytest

from clinical_readmission.evaluation.bootstrap import (
    bootstrap_metric_difference,
    bootstrap_probability_metrics,
    draw_stratified_bootstrap_indices,
    summarize_bootstrap_metrics,
)


def example_data():
    target = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.15,
            0.25,
            0.35,
            0.55,
            0.65,
            0.75,
            0.85,
            0.95,
        ]
    )

    return (
        target,
        probabilities,
    )


def test_stratified_bootstrap_preserves_class_counts() -> None:
    target, _ = example_data()

    rng = np.random.default_rng(
        42
    )

    indices = (
        draw_stratified_bootstrap_indices(
            target,
            rng,
        )
    )

    sampled_target = target[
        indices
    ]

    assert len(sampled_target) == len(
        target
    )

    assert int(
        sampled_target.sum()
    ) == int(
        target.sum()
    )


def test_bootstrap_is_reproducible() -> None:
    target, probabilities = (
        example_data()
    )

    first = (
        bootstrap_probability_metrics(
            target,
            probabilities,
            n_resamples=100,
            random_state=77,
        )
    )

    second = (
        bootstrap_probability_metrics(
            target,
            probabilities,
            n_resamples=100,
            random_state=77,
        )
    )

    for metric in first:
        np.testing.assert_array_equal(
            first[metric],
            second[metric],
        )


def test_bootstrap_summary_has_valid_intervals() -> None:
    target, probabilities = (
        example_data()
    )

    summary = (
        summarize_bootstrap_metrics(
            target,
            probabilities,
            n_resamples=100,
            random_state=77,
        )
    )

    assert {
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    } == set(summary)

    for result in summary.values():
        assert np.isfinite(
            result["estimate"]
        )

        assert (
            result["ci_lower"]
            <= result["ci_upper"]
        )

        assert (
            result[
                "bootstrap_standard_error"
            ]
            >= 0
        )


def test_paired_difference_identical_models_is_zero() -> None:
    target, probabilities = (
        example_data()
    )

    result = (
        bootstrap_metric_difference(
            target,
            probabilities,
            probabilities,
            metric="roc_auc",
            n_resamples=100,
            random_state=77,
        )
    )

    assert result[
        "estimate"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "ci_lower"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "ci_upper"
    ] == pytest.approx(
        0.0
    )


def test_rejects_too_few_resamples() -> None:
    target, probabilities = (
        example_data()
    )

    with pytest.raises(
        ValueError,
        match="at least 100",
    ):
        bootstrap_probability_metrics(
            target,
            probabilities,
            n_resamples=20,
        )


def test_rejects_unknown_difference_metric() -> None:
    target, probabilities = (
        example_data()
    )

    with pytest.raises(
        ValueError,
        match="Unsupported metric",
    ):
        bootstrap_metric_difference(
            target,
            probabilities,
            probabilities,
            metric="accuracy",
            n_resamples=100,
        )