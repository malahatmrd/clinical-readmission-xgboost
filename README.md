# Clinical Readmission Prediction with XGBoost

A leakage-aware and reproducible clinical machine-learning pipeline for
predicting 30-day hospital readmission using structured hospital encounter data.

The project emphasizes rigorous cohort construction, patient-independent data
splitting, probability calibration, validation-only threshold selection,
explainability, subgroup analysis, robustness evaluation, and a one-time locked
Test assessment.

> **Research use only.** This model has not undergone external or prospective
> clinical validation and is not intended for clinical deployment.

## Final Model

Final model:

`tuned_xgboost_sigmoid`

Configuration:

- tuned XGBoost classifier
- 155 boosting trees
- Train-only sigmoid calibration
- reference threshold: `0.105`
- prediction timing: at or near discharge

The model, calibration procedure, and threshold were frozen before the held-out
Test set was accessed.

## Final Locked Test Results

Test cohort:

- encounters: 10,496
- positive 30-day readmissions: 941
- prevalence: 8.97%

| Metric | Estimate | 95% Bootstrap CI |
|---|---:|---:|
| ROC-AUC | 0.6655 | 0.6470–0.6826 |
| Average precision | 0.1954 | 0.1771–0.2173 |
| Brier score | 0.0782 | 0.0773–0.0791 |
| Log loss | 0.2857 | 0.2826–0.2890 |
| Calibration intercept | 0.2417 | -0.0074–0.4892 |
| Calibration slope | 1.1057 | 0.9942–1.2142 |
| Sensitivity | 0.3879 | 0.3560–0.4187 |
| Specificity | 0.8322 | 0.8247–0.8397 |
| PPV | 0.1855 | 0.1716–0.1991 |
| NPV | 0.9325 | 0.9292–0.9357 |
| Alerts per 100 | 18.75 | 18.04–19.51 |
| Model net benefit | 0.0169 | 0.0139–0.0197 |

Uncertainty was estimated with 2,000 stratified bootstrap resamples using
a fixed model and fixed threshold.

No model refitting, probability recalibration, feature reselection, or
threshold reselection was performed after Test access.

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008**
dataset from the UCI Machine Learning Repository.

Raw dataset:

- 101,766 encounters
- 71,518 unique patients
- 11,357 readmissions within 30 days
- raw prevalence: 11.16%

Primary modeling cohort:

- 69,973 encounters
- 69,973 unique patients
- 6,277 positive outcomes
- prevalence: 8.97%

The primary analysis retains one encounter per patient and excludes terminal
and hospice discharge dispositions.

Dataset DOI:

`10.24432/C5230J`

See:

- `docs/data_card.md`
- `docs/cohort_definition.md`
- `DATA_LICENSE.md`

## Data Split

| Split | Encounters | Positive outcomes |
|---|---:|---:|
| Train | 48,981 | 4,394 |
| Validation | 10,496 | 942 |
| Test | 10,496 | 941 |

The Test partition was locked during:

- preprocessing development
- model selection
- hyperparameter tuning
- calibration selection
- threshold selection
- SHAP analysis
- subgroup analysis
- robustness analysis

See `docs/split_protocol.md`.

## Modeling Pipeline

The development workflow includes:

1. dataset acquisition and integrity checks
2. clinical cohort definition
3. leakage-safe Train/Validation/Test splitting
4. preprocessing and logistic-regression baseline
5. XGBoost baseline development
6. imbalance analysis and hyperparameter optimization
7. probability calibration
8. validation-only operating-threshold selection
9. SHAP explainability
10. subgroup and repeated-encounter robustness analysis
11. frozen final Test evaluation and uncertainty estimation

The final preprocessing representation contains 225 transformed model
features.

Patient and encounter identifiers are excluded from predictive inputs.

## Prediction Timing

`discharge_disposition_id` is included in the final feature set and is the
largest source-level SHAP contributor.

The model must therefore be interpreted as an **at-discharge or
near-discharge risk model**, not as an admission-time prediction model.

## Explainability

SHAP analysis was performed on the Validation partition before Test access.

Leading source-level contributors included:

1. `discharge_disposition_id`
2. `number_inpatient`
3. `time_in_hospital`
4. `diag_1`
5. `payer_code`
6. `age`
7. `diabetesMed`
8. `number_diagnoses`

SHAP values describe model dependence and must not be interpreted as causal
effects.

See `docs/shap_explainability.md`.

## Subgroup and Robustness Analysis

Validation-only subgroup analyses evaluated:

- gender
- race
- age

No evaluated pairwise subgroup ROC-AUC confidence interval provided clear
evidence of a discrimination difference.

However, the fixed threshold produced meaningfully different operating
characteristics across age groups.

Repeated-encounter analysis also showed substantial shifts in prevalence,
calibration, and alert burden.

These analyses are exploratory and do not constitute fairness certification
or external validation.

See `docs/subgroup_robustness.md`.

## Reproducibility

Python version:

`3.12`

Create an environment:

```bash
python -m venv .venv