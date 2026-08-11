# Clinical Readmission Prediction with XGBoost

Reliable and reproducible 30-day hospital readmission prediction using
real-world clinical encounter data.

## Overview

This project develops a leakage-aware clinical machine-learning pipeline
for predicting hospital readmission within 30 days.

The project is designed as a research-oriented medical data science
repository rather than a single modeling notebook.

Key priorities include:

- reproducible data acquisition
- explicit data provenance
- patient-level leakage prevention
- clinical cohort definition
- robust preprocessing
- interpretable baseline models
- XGBoost modeling
- class-imbalance handling
- probability calibration
- threshold analysis
- SHAP explainability
- subgroup evaluation
- robustness analysis
- automated testing

## Dataset

Diabetes 130-US Hospitals for Years 1999-2008.

Current validated snapshot:

| Property | Value |
|---|---:|
| Encounters | 101,766 |
| Unique patients | 71,518 |
| Predictive features | 47 |
| Positive 30-day readmissions | 11,357 |
| Positive rate | 11.16% |

Detailed dataset documentation is available in:

`docs/data_card.md`

## Prediction Target

Positive:

`readmitted == "<30"`

Negative:

`readmitted == "NO"` or `readmitted == ">30"`

## Leakage Prevention

The UCI dataset provides `encounter_id` and `patient_nbr` separately from
the predictive feature matrix.

These identifiers are never exposed to the predictive model.

Because 16,773 patients have multiple encounters, ordinary row-level
random splitting could place encounters from the same patient in both
training and evaluation data.

This project therefore uses patient-aware cohort construction and
group-based validation.

## Repository Structure

```text
clinical-readmission-xgboost/
│
├── artifacts/
│   └── metrics/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   └── data_card.md
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── scripts/
│
├── src/
│   └── clinical_readmission/
│       ├── data/
│       ├── evaluation/
│       ├── explainability/
│       ├── features/
│       └── models/
│
├── tests/
│
├── pyproject.toml
├── requirements.txt
└── requirements-lock.txt