# XGBoost Optimization and Class-Imbalance Analysis

## Purpose

Phase 6 develops the XGBoost model beyond the untuned baseline while preserving strict separation between model development and the locked test set.

The phase evaluates class weighting, Train-only early stopping, randomized hyperparameter search, and a final tuned XGBoost development model.

Average precision was defined in advance as the primary model-selection metric because 30-day readmission occurs in approximately 9% of the development cohort.

ROC-AUC is treated as a secondary discrimination metric.

Brier score and log loss are retained to assess probability quality.

The probability threshold of 0.5 remains descriptive only and is not optimized during this phase.

## Class Imbalance

The training partition contains:

| Class | Count |
|---|---:|
| Positive | 4,394 |
| Negative | 44,587 |

The negative-to-positive ratio is approximately 10.147.

An initial weighted XGBoost model therefore evaluated:

`scale_pos_weight = 10.147246`

This substantially increased recall at the conventional threshold of 0.5 but degraded discrimination and probability quality.

The experiment demonstrates that inverse-frequency class weighting is not automatically optimal for this clinical prediction problem.

## Train-Only Early Stopping

To avoid using the main validation partition for early stopping, the original training partition was divided reproducibly into:

| Internal partition | Rows | Positive outcomes |
|---|---:|---:|
| Internal fit | 39,184 | 3,515 |
| Internal stopping set | 9,797 | 879 |

The internal split was stratified and used random seed 44.

Preprocessing was fitted exclusively on the internal-fit subset during early-stopping development.

The initial unweighted early-stopping experiment selected 21 trees and improved validation performance over the 100-tree XGBoost baseline.

## Train-Only Hyperparameter Search

Hyperparameter optimization used only the original training partition.

The search protocol consisted of:

- 3-fold stratified cross-validation
- 24 randomized configurations
- 72 total cross-validation fits
- Average Precision as the primary selection metric
- ROC-AUC as a secondary metric
- Validation excluded from hyperparameter search
- Test excluded from hyperparameter search

The best Train-CV configuration achieved:

| Metric | Value |
|---|---:|
| Mean CV Average Precision | 0.185535 |
| CV AP standard deviation | 0.004979 |
| Mean CV ROC-AUC | 0.653620 |
| CV ROC-AUC standard deviation | 0.004105 |

The selected hyperparameters were:

| Hyperparameter | Value |
|---|---:|
| learning_rate | 0.03 |
| max_depth | 5 |
| min_child_weight | 1.0 |
| n_estimators | 400 |
| subsample | 1.0 |
| colsample_bytree | 0.7 |
| reg_alpha | 0.1 |
| reg_lambda | 2.0 |
| scale_pos_weight | 1.5 |

The milder positive-class weight of 1.5 outperformed the much larger inverse-frequency reference weight during Train-only model selection.

## Overfitting Assessment

The best randomized-search configuration produced:

| Metric | Value |
|---|---:|
| Mean Train Average Precision | 0.339559 |
| Mean CV Average Precision | 0.185535 |
| Train-CV AP gap | 0.154024 |

The observed gap indicated substantial model capacity and justified a separate early-stopping stage before final validation evaluation.

## Tuned Early Stopping

The selected Train-CV hyperparameters were passed unchanged into the internal Train-only early-stopping protocol.

The maximum tree count remained 400, matching the configuration selected during randomized search.

Early stopping produced:

| Property | Value |
|---|---:|
| Best zero-based iteration | 154 |
| Selected trees | 155 |
| Best internal AUCPR | 0.174803 |
| Rounds actually built | 205 |

The final development model was then refitted from scratch using all 48,981 training encounters and 155 boosting trees.

## Development Model Comparison

| Model | ROC-AUC | Average Precision | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.642876 | 0.159687 | 0.079809 | 0.290961 |
| Untuned XGBoost | 0.636331 | 0.153002 | 0.081284 | 0.296462 |
| Inverse-frequency Weighted XGBoost | 0.620873 | 0.150614 | 0.188640 | 0.559007 |
| Early-Stopped XGBoost | 0.645781 | 0.164314 | 0.079631 | 0.290352 |
| Tuned + Early-Stopped XGBoost | 0.650820 | 0.171639 | 0.080553 | 0.295975 |

## Phase 6 Champion

Average Precision was specified before optimization as the primary development metric.

Under this policy, the Phase 6 champion is:

**Tuned XGBoost with Train-only hyperparameter search and Train-only early stopping.**

Its validation performance is:

- ROC-AUC: 0.650820
- Average Precision: 0.171639
- Brier Score: 0.080553
- Log Loss: 0.295975

The tuned model provides the strongest discrimination and ranking performance observed so far.

However, it does not provide the best probability-quality metrics.

The simpler early-stopped XGBoost currently achieves a lower Brier score and lower log loss.

This distinction motivates the dedicated calibration analysis in the next phase.

## Threshold Policy

Threshold-dependent metrics at 0.5 are reported only for descriptive purposes.

No threshold was selected or optimized during Phase 6.

Clinical operating-point selection is deferred to a later validation-only phase.

## Leakage Control

The locked test partition has not been used for:

- preprocessing decisions
- class-weight selection
- early stopping
- hyperparameter optimization
- model selection
- calibration
- threshold selection

Randomized hyperparameter search used Train only.

Early stopping used an internal holdout created from Train only.

The main validation partition was used for development-model evaluation and comparison after model-development decisions were established.

## Next Stage

Phase 7 evaluates discrimination and calibration more formally.

The analysis will compare probability calibration across candidate models before any final locked-test evaluation is performed.