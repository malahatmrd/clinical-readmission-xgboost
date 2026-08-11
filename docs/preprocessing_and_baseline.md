# Leakage-Safe Preprocessing and Logistic Regression Baseline

## Purpose

This phase establishes the first leakage-safe modeling baseline for
30-day hospital readmission prediction.

All preprocessing parameters and model coefficients are fitted using
the training partition only.

The validation partition is used for development evaluation.

The locked test partition is not used.

## Training Data

| Property | Value |
|---|---:|
| Training encounters | 48,981 |
| Training patients | 48,981 |
| Positive outcomes | 4,394 |
| Positive rate | 8.971% |

The validation partition contains:

| Property | Value |
|---|---:|
| Validation encounters | 10,496 |
| Positive outcomes | 942 |
| Positive rate | 8.975% |

## Feature Policy

Identifiers and metadata are excluded from prediction:

- encounter_id
- patient_nbr
- source_row

Target columns are excluded from predictors:

- readmitted
- readmitted_30d

The weight feature is excluded because approximately 96% of training
values are missing.

The following training-constant features are also excluded:

- examide
- citoglipton
- glimepiride-pioglitazone
- metformin-pioglitazone

## Numerical Features

Eight numerical predictors are retained.

Numerical preprocessing consists of:

1. median imputation
2. standard scaling

Both transformations are fitted using training data only.

## Categorical Features

Categorical variables are processed using explicit missing-value
categories followed by one-hot encoding.

Previously unseen categories in validation data are handled without
refitting the encoder.

## Medical Specialty

Missing medical-specialty values are represented explicitly.

Low-frequency specialties are grouped using a frequency threshold learned
from training data only.

## Laboratory Features

Missing values in:

- A1Cresult
- max_glu_serum

are represented as `NotMeasured`.

This preserves the distinction between an unperformed laboratory test
and an observed laboratory result.

## Diagnosis Features

Raw ICD-9 diagnosis codes in:

- diag_1
- diag_2
- diag_3

are mapped into clinically interpretable groups:

- Circulatory
- Diabetes
- Respiratory
- Digestive
- Genitourinary
- Musculoskeletal
- Neoplasms
- Injury
- Other
- Missing

The grouped diagnosis variables are subsequently one-hot encoded.

## Final Feature Matrix

The training preprocessing pipeline produced:

| Property | Value |
|---|---:|
| Training rows | 48,981 |
| Raw model inputs | 42 |
| Transformed features | 225 |
| Missing transformed values | 0 |
| Matrix representation | Sparse |

## Logistic Regression Baseline

The initial baseline uses:

- logistic regression
- solver: newton-cholesky
- C: 1.0
- class weighting: none
- no hyperparameter optimization

Class-imbalance optimization is intentionally deferred to a later phase.

## Validation Results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.642876 |
| Average precision | 0.159687 |
| Brier score | 0.079809 |
| Log loss | 0.290961 |

At the conventional threshold of 0.5:

| Metric | Value |
|---|---:|
| Precision | 0.076923 |
| Recall | 0.001062 |
| F1 | 0.002094 |
| True negative | 9,542 |
| False positive | 12 |
| False negative | 941 |
| True positive | 1 |

The 0.5 threshold is reported only as a descriptive baseline.

It is not considered an optimized clinical operating point.

Threshold optimization remains reserved for the dedicated validation-only
threshold-selection stage.

## Interpretation

The baseline demonstrates modest discrimination above chance.

Average precision exceeds the validation event prevalence, indicating
that the model contains predictive ranking information despite substantial
class imbalance.

Performance at the conventional 0.5 threshold is poor because very few
encounters receive probabilities above 0.5.

No threshold adjustment is performed during this phase.

## Leakage Policy

The test set remains locked.

It has not been used for:

- preprocessing fitting
- feature engineering decisions
- categorical vocabulary learning
- scaling
- imputation
- model fitting
- class-weight selection
- threshold selection
- baseline evaluation

## Next Stage

The next modeling phase is XGBoost development using the same frozen
training, validation, and test partitions.