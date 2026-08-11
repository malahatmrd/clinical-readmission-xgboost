from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from clinical_readmission.features.preprocessing import (
    build_preprocessor,
)

BASELINE_SOLVER = "newton-cholesky"
BASELINE_C = 1.0
BASELINE_MAX_ITER = 1000


def build_logistic_baseline() -> Pipeline:
    """Build the unweighted logistic-regression baseline."""

    model = LogisticRegression(
        solver=BASELINE_SOLVER,
        C=BASELINE_C,
        max_iter=BASELINE_MAX_ITER,
        class_weight=None,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )