# Clinical Readmission Prediction with XGBoost

A reproducible clinical machine-learning project for predicting 30-day hospital readmission using real-world diabetes inpatient encounters.

## Project Goal

The project investigates whether routinely available clinical information can be used to estimate the risk of hospital readmission within 30 days of discharge.

The emphasis is not limited to predictive performance. The pipeline is designed around:

- leakage-aware cohort construction
- patient-level train/validation/test splitting
- reproducible preprocessing
- interpretable baseline modeling
- gradient-boosted decision trees
- probability calibration
- threshold selection
- SHAP-based explainability
- subgroup evaluation
- robustness analysis
- reproducible testing

## Dataset

Diabetes 130-US Hospitals for Years 1999-2008  
UCI Machine Learning Repository — Dataset ID 296.

Raw clinical data are not committed to this repository.

## Prediction Target

Positive:
- readmission within 30 days

Negative:
- readmission after 30 days
- no recorded readmission

## Repository Status

Phase 0 — project bootstrap and reproducible environment.

## Disclaimer

This project is intended for research and educational purposes only.

It is not a clinical decision-support system and must not be used for patient care without appropriate external validation, prospective evaluation, and regulatory review.
