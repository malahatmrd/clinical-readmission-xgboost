from __future__ import annotations

import pytest

from clinical_readmission.evaluation.metrics import (
    assert_metric_reproduction,
    calculate_metric_deltas,
    calculate_probability_metrics,
)


def test_calculates_probability_metrics() -> None:
    target = [
        0,
        0,
        1,
        1,
    ]

    probabilities = [
        0.10,
        0.20,
        0.80,
        0.90,
    ]

    metrics = calculate_probability_metrics(
        target,
        probabilities,
    )

    assert metrics["roc_auc"] == 1.0

    assert (
        metrics["average_precision"]
        == 1.0
    )

    assert metrics["brier_score"] > 0
    assert metrics["log_loss"] > 0


def test_calculates_metric_deltas() -> None:
    current = {
        "roc_auc": 0.70,
        "average_precision": 0.20,
        "brier_score": 0.08,
        "log_loss": 0.30,
    }

    reference = {
        "roc_auc": 0.65,
        "average_precision": 0.18,
        "brier_score": 0.09,
        "log_loss": 0.31,
    }

    deltas = calculate_metric_deltas(
        current,
        reference,
    )

    assert deltas[
        "roc_auc"
    ] == pytest.approx(
        0.05
    )

    assert deltas[
        "average_precision"
    ] == pytest.approx(
        0.02
    )

    assert deltas[
        "brier_score"
    ] == pytest.approx(
        -0.01
    )


def test_exact_reproduction_passes() -> None:
    metrics = {
        "roc_auc": 0.65,
        "average_precision": 0.17,
        "brier_score": 0.08,
        "log_loss": 0.29,
    }

    assert_metric_reproduction(
        metrics,
        metrics,
    )


def test_small_numeric_difference_passes() -> None:
    current = {
        "roc_auc": 0.6500000001,
        "average_precision": 0.17,
        "brier_score": 0.08,
        "log_loss": 0.29,
    }

    reference = {
        "roc_auc": 0.65,
        "average_precision": 0.17,
        "brier_score": 0.08,
        "log_loss": 0.29,
    }

    assert_metric_reproduction(
        current,
        reference,
        tolerance=1e-8,
    )


def test_material_reproduction_difference_fails() -> None:
    current = {
        "roc_auc": 0.66,
        "average_precision": 0.17,
        "brier_score": 0.08,
        "log_loss": 0.29,
    }

    reference = {
        "roc_auc": 0.65,
        "average_precision": 0.17,
        "brier_score": 0.08,
        "log_loss": 0.29,
    }

    with pytest.raises(
        ValueError,
        match="reproduction failed",
    ):
        assert_metric_reproduction(
            current,
            reference,
        )


def test_rejects_negative_tolerance() -> None:
    metrics = {
        "roc_auc": 0.65,
        "average_precision": 0.17,
        "brier_score": 0.08,
        "log_loss": 0.29,
    }

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        assert_metric_reproduction(
            metrics,
            metrics,
            tolerance=-1.0,
        )