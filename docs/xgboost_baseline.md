# XGBoost Baseline

## Purpose

This phase establishes an untuned XGBoost baseline for 30-day hospital
readmission prediction.

The objective is to determine whether a standard gradient-boosted tree
model improves upon the leakage-safe logistic-regression baseline before
class-imbalance handling or hyperparameter optimization is introduced.

## Data Splits

The frozen data partitions from Phase 3 are retained.

| Split | Encounters | Positive Outcomes |
|---|---:|---:|
| Train | 48,981 | 4,394 |
| Validation | 10,496 | 942 |
| Test | 10,496 | 941 |

The XGBoost model is fitted exclusively on the training partition.

Development performance is evaluated exclusively on the validation
partition.

The locked test set is not used.

## Preprocessing

The same leakage-safe preprocessing policy established for the logistic
regression baseline is reused.

The preprocessing pipeline produces 225 transformed features from
42 raw model inputs.

All data-dependent preprocessing transformations are fitted using
training data only.

## Baseline XGBoost Configuration

The baseline uses an XGBoost binary classifier with:

- objective: binary logistic
- estimators: 100
- maximum tree depth: 6
- learning rate: 0.3
- minimum child weight: 1
- subsample: 1.0
- column subsampling: 1.0
- L1 regularization: 0
- L2 regularization: 1
- tree method: histogram
- random state: 42

No class weighting is applied.

No `scale_pos_weight` adjustment is applied.

No early stopping is used.

No hyperparameter optimization is performed.

These decisions preserve the model as a genuine untuned baseline.

## Validation Performance

| Metric | XGBoost | Logistic Regression | Difference |
|---|---:|---:|---:|
| ROC-AUC | 0.636331 | 0.642876 | -0.006544 |
| Average precision | 0.153002 | 0.159687 | -0.006685 |
| Brier score | 0.081284 | 0.079809 | +0.001475 |
| Log loss | 0.296462 | 0.290961 | +0.005501 |

Lower values are preferable for Brier score and log loss.

The untuned XGBoost baseline therefore does not outperform the logistic
regression baseline on the principal validation metrics.

## Conventional Threshold Performance

At the non-optimized probability threshold of 0.5:

| Metric | Value |
|---|---:|
| Precision | 0.344262 |
| Recall | 0.022293 |
| F1 | 0.041874 |
| True negatives | 9,514 |
| False positives | 40 |
| False negatives | 921 |
| True positives | 21 |

The conventional threshold identifies only a small proportion of
positive 30-day readmissions.

These threshold-dependent results are descriptive only.

No operating threshold is selected during this phase.

## Interpretation

The untuned XGBoost model demonstrates predictive discrimination above
chance but does not improve upon the logistic-regression baseline.

This result does not indicate that gradient boosting is unsuitable for
the prediction task.

The current experiment intentionally excludes class-imbalance
adjustments, early stopping, and hyperparameter optimization.

The comparison therefore establishes a reference point for subsequent
XGBoost development rather than a final model comparison.

## Leakage Policy

The test partition remains locked and has not been used for:

- preprocessing fitting
- model fitting
- hyperparameter selection
- class-imbalance strategy selection
- early stopping
- calibration
- threshold selection
- baseline model comparison

## Next Stage

The next phase evaluates class-imbalance strategies and performs
validation-based hyperparameter optimization for XGBoost.

The locked test set remains unavailable for model-development decisions.