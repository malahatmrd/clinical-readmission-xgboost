from __future__ import annotations

import numpy as np
import pytest

from clinical_readmission.evaluation.thresholds import (
    build_threshold_table,
    calculate_net_benefit,
    calculate_threshold_metrics,
    select_threshold_for_alert_capacity,
    select_threshold_for_minimum_sensitivity,
)


def example_data():
    target = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.10,
            0.20,
            0.40,
            0.60,
            0.80,
            0.90,
        ]
    )

    return (
        target,
        probabilities,
    )


def test_threshold_metrics_known_example() -> None:
    target, probabilities = (
        example_data()
    )

    result = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold=0.50,
        )
    )

    assert result[
        "true_positive"
    ] == 3

    assert result[
        "false_positive"
    ] == 0

    assert result[
        "true_negative"
    ] == 3

    assert result[
        "false_negative"
    ] == 0

    assert result[
        "sensitivity"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "specificity"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "ppv"
    ] == pytest.approx(
        1.0
    )


def test_alert_rate_is_calculated() -> None:
    target, probabilities = (
        example_data()
    )

    result = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold=0.70,
        )
    )

    assert result[
        "alert_rate"
    ] == pytest.approx(
        2.0 / 6.0
    )

    assert result[
        "alerts_per_100"
    ] == pytest.approx(
        100.0 / 3.0
    )


def test_number_needed_to_evaluate() -> None:
    target, probabilities = (
        example_data()
    )

    result = (
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold=0.30,
        )
    )

    assert result[
        "number_needed_to_evaluate"
    ] == pytest.approx(
        4.0 / 3.0
    )


def test_net_benefit_known_example() -> None:
    target, probabilities = (
        example_data()
    )

    result = (
        calculate_net_benefit(
            target,
            probabilities,
            threshold=0.50,
        )
    )

    assert result[
        "model_net_benefit"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "treat_none_net_benefit"
    ] == 0.0


def test_threshold_table_has_expected_rows() -> None:
    target, probabilities = (
        example_data()
    )

    table = build_threshold_table(
        target,
        probabilities,
        thresholds=[
            0.30,
            0.50,
            0.70,
        ],
    )

    assert len(table) == 3

    assert {
        "sensitivity",
        "specificity",
        "ppv",
        "alert_rate",
        "model_net_benefit",
    }.issubset(
        table.columns
    )


def test_rejects_invalid_threshold() -> None:
    target, probabilities = (
        example_data()
    )

    with pytest.raises(
        ValueError,
        match="strictly",
    ):
        calculate_threshold_metrics(
            target,
            probabilities,
            threshold=1.0,
        )


def test_select_threshold_for_minimum_sensitivity() -> None:
    target, probabilities = (
        example_data()
    )

    table = build_threshold_table(
        target,
        probabilities,
        thresholds=[
            0.30,
            0.50,
            0.70,
            0.85,
        ],
    )

    selected = (
        select_threshold_for_minimum_sensitivity(
            table,
            minimum_sensitivity=0.80,
        )
    )

    assert selected[
        "threshold"
    ] == pytest.approx(
        0.50
    )


def test_select_threshold_for_alert_capacity() -> None:
    target, probabilities = (
        example_data()
    )

    table = build_threshold_table(
        target,
        probabilities,
        thresholds=[
            0.30,
            0.50,
            0.70,
            0.85,
        ],
    )

    selected = (
        select_threshold_for_alert_capacity(
            table,
            maximum_alerts_per_100=50.0,
        )
    )

    assert selected[
        "threshold"
    ] == pytest.approx(
        0.50
    )


def test_sensitivity_selector_rejects_impossible_target() -> None:
    target, probabilities = (
        example_data()
    )

    table = build_threshold_table(
        target,
        probabilities,
        thresholds=[
            0.85,
            0.90,
        ],
    )

    with pytest.raises(
        ValueError,
        match="No threshold",
    ):
        select_threshold_for_minimum_sensitivity(
            table,
            minimum_sensitivity=1.0,
        )


def test_alert_selector_rejects_impossible_capacity() -> None:
    target, probabilities = (
        example_data()
    )

    table = build_threshold_table(
        target,
        probabilities,
        thresholds=[
            0.10,
            0.20,
        ],
    )

    with pytest.raises(
        ValueError,
        match="No threshold",
    ):
        select_threshold_for_alert_capacity(
            table,
            maximum_alerts_per_100=1.0,
        )