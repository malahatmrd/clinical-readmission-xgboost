# Model Card — 30-Day Hospital Readmission Prediction

## Model Overview

This project develops a research-oriented machine-learning model for
predicting 30-day hospital readmission from structured clinical encounter
data.

Final model:

`tuned_xgboost_sigmoid`

The final prediction system consists of:

- a leakage-controlled preprocessing pipeline
- a tuned XGBoost classifier
- 155 fitted boosting trees
- Train-only sigmoid probability calibration
- a frozen reference threshold of 0.105

The model was developed with a strict separation between model development
and final Test evaluation.

The held-out Test partition was not used for:

- feature engineering
- preprocessing decisions
- model-family selection
- hyperparameter optimization
- calibration selection
- threshold selection
- SHAP interpretation
- subgroup analysis
- robustness analysis

The final Test set was accessed only after the model configuration,
calibration procedure, and operating threshold had been frozen and recorded.

---

## Intended Purpose

The model is intended as a research example of rigorous clinical
machine-learning development for binary risk prediction.

The primary prediction task is:

> Estimate the probability that a patient will experience hospital
> readmission within 30 days after the index hospitalization.

The project emphasizes:

- leakage prevention
- reproducible cohort construction
- patient-independent data splitting
- probabilistic prediction
- calibration
- class-imbalance-aware evaluation
- transparent operating-threshold analysis
- explainability
- subgroup characterization
- repeated-encounter robustness
- locked final Test evaluation
- uncertainty estimation

The project is intended for methodological demonstration, research,
education, and portfolio use.

---

## Not Intended for Clinical Deployment

This model must not be interpreted as a clinically validated decision-support
system.

It has not undergone:

- prospective clinical validation
- external validation in an independent health system
- temporal validation on contemporary clinical data
- clinical workflow evaluation
- randomized impact evaluation
- regulatory review
- formal health-economic evaluation
- formal fairness certification

The model should therefore not be used independently to determine patient
care, discharge decisions, follow-up intensity, resource allocation, or
clinical treatment.

The reference threshold described in this project is an illustrative
operating point rather than a universally optimal clinical threshold.

---

## Dataset

The project uses the:

**Diabetes 130-US Hospitals for Years 1999-2008**

dataset from the UCI Machine Learning Repository.

The raw dataset contains:

- 101,766 hospital encounters
- 71,518 unique patients
- 11,357 `<30` readmission outcomes
- raw positive rate: approximately 11.16%

Because the dataset represents hospital care from 1999-2008, its clinical
practice patterns, coding conventions, patient populations, and healthcare
delivery processes may differ materially from contemporary practice.

This historical setting is an important limitation on transportability.

Detailed dataset documentation:

`docs/data_card.md`

---

## Outcome Definition

The target is binary 30-day hospital readmission.

Positive outcome:

`readmitted == "<30"`

Negative outcome:

- `readmitted == "NO"`
- `readmitted == ">30"`

The binary modeling target is stored as:

`readmitted_30d`

---

## Primary Cohort

The primary modeling cohort was defined before model development.

Construction:

1. start with 101,766 raw encounters
2. retain the first observed encounter for each patient
3. obtain 71,518 patient-independent encounters
4. exclude terminal and hospice discharge dispositions
5. obtain the final primary cohort

Final primary cohort:

- encounters: 69,973
- unique patients: 69,973
- positive 30-day readmissions: 6,277
- prevalence: approximately 8.97%

Using one primary encounter per patient prevents repeated observations from
the same patient from crossing the primary data partitions.

Detailed cohort documentation:

`docs/cohort_definition.md`

---

## Data Split

The primary cohort was divided into Train, Validation, and locked Test
partitions.

| Split | Encounters | Positive outcomes |
|---|---:|---:|
| Train | 48,981 | 4,394 |
| Validation | 10,496 | 942 |
| Test | 10,496 | 941 |

The split was stratified by the binary outcome.

The final Test set remained locked throughout development.

Detailed split protocol:

`docs/split_protocol.md`

---

## Leakage Controls

The dataset contains both:

- `encounter_id`
- `patient_nbr`

Neither identifier is used as a predictive model feature.

Patient identifiers are used only for:

- cohort construction
- split-boundary verification
- repeated-encounter robustness analysis
- patient-cluster bootstrap procedures

Row-level split assignments are excluded from Git.

Row-level final Test predictions are also excluded from Git.

The final Test prediction artifact contains no encounter or patient
identifiers and remains local.

---

## Prediction Timing

The final feature set includes `discharge_disposition_id`.

Therefore, this model should be interpreted as an **at-discharge or
near-discharge readmission-risk model**, not as an admission-time prediction
model.

This distinction is important because several variables used by the model
are only available during or near completion of the hospitalization.

Any deployment interpretation must preserve this prediction-time definition.

---

## Preprocessing

The preprocessing pipeline is fitted using Train data only.

The final transformed representation contains:

225 model features.

Major preprocessing components include:

- numeric feature processing
- categorical one-hot encoding
- diagnosis grouping
- explicit missing-value categories where appropriate
- controlled treatment of high-cardinality categorical variables

Identifiers and target columns are excluded from predictive inputs.

Detailed preprocessing documentation:

`docs/preprocessing_and_baseline.md`

---

## Model Development

Several model stages were evaluated during development.

These included:

- logistic-regression baseline
- untuned XGBoost
- class-weighted XGBoost
- early-stopped XGBoost
- tuned XGBoost
- calibrated tuned XGBoost

Hyperparameter optimization was conducted using Train data only.

The final tuned XGBoost configuration uses:

- learning rate: 0.03
- maximum depth: 5
- minimum child weight: 1
- subsample: 1.0
- column subsample by tree: 0.7
- L1 regularization: 0.1
- L2 regularization: 2.0
- positive-class weight: 1.5
- selected tree count: 155

The selected tree count was determined during Train-only development and was
frozen before the final model evaluation.

Detailed optimization documentation:

`docs/xgboost_optimization.md`

---

## Probability Calibration

The final XGBoost model uses sigmoid probability calibration.

Calibration protocol:

- calibration method: sigmoid
- calibration fitting data: Train only
- cross-validation strategy: stratified 5-fold
- random seed: 48
- ensemble mode: disabled

Calibration strategy selection occurred before threshold selection and before
the Test set was accessed.

Detailed discrimination and calibration documentation:

`docs/discrimination_and_calibration.md`

---

## Validation Performance

Before final Test evaluation, the frozen calibrated model achieved the
following Validation performance:

| Metric | Validation |
|---|---:|
| ROC-AUC | 0.6508 |
| Average precision | 0.1716 |
| Brier score | 0.0794 |
| Log loss | 0.2897 |
| Calibration intercept | -0.1182 |
| Calibration slope | 0.9484 |

These values represent development-set performance and were not final
generalization estimates.

---

## Reference Operating Threshold

Threshold analysis was performed only on Validation data.

Three operating scenarios were examined:

- high sensitivity
- moderate capacity
- limited capacity

The final frozen reference threshold was:

`0.105`

Selected scenario:

`moderate_capacity`

The threshold was chosen using a predefined alert-capacity framework rather
than by maximizing Test performance.

Validation operating characteristics at threshold 0.105:

- sensitivity: 0.3567
- specificity: 0.8312
- PPV: 0.1724
- NPV: 0.9291
- alerts per 100: 18.57
- model net benefit: 0.0140

The model family, calibration method, and threshold were frozen before the
Test partition was accessed.

Detailed threshold documentation:

`docs/threshold_selection.md`

---

## Pre-Test Freeze

Before Test evaluation, a machine-readable Pre-Test freeze manifest was
created and committed.

The manifest recorded:

- model identity
- selected XGBoost hyperparameters
- tree count
- calibration strategy
- reference threshold
- data hashes
- environment versions
- source-file hashes
- locked Test aggregate counts
- explicit confirmation that Test predictions and Test metrics had not yet
  been generated

Pre-Test development commit:

`135936db144f8ff878c40f764be2a5ed92acf803`

Pre-Test freeze manifest commit:

`29f72f77594ad9149c374e642683a22344d6f982`

Evaluation-code commit:

`12ca5b45954ebaaa2e750f40981eba70dfedaf60`

Artifact:

`artifacts/metrics/phase11_pretest_freeze.json`

---

## Final Locked Test Evaluation

The frozen model was evaluated once on the locked Test partition.

Test sample:

- encounters: 10,496
- unique patients: 10,496
- positive outcomes: 941
- negative outcomes: 9,555
- prevalence: 8.97%

Before Test scoring, the model reconstruction was verified against previously
stored Validation predictions.

Maximum absolute Validation probability reproduction error:

`9.71445146547e-17`

Therefore:

`Validation probability reproduction: PASS`

No model, calibration, feature, or threshold decision was changed after Test
results became available.

---

## Final Test Performance

### Discrimination and Probability Accuracy

| Metric | Estimate | 95% Bootstrap CI |
|---|---:|---:|
| ROC-AUC | 0.6655 | 0.6470 to 0.6826 |
| Average precision | 0.1954 | 0.1771 to 0.2173 |
| Brier score | 0.07818 | 0.07734 to 0.07906 |
| Log loss | 0.28571 | 0.28261 to 0.28897 |

The model demonstrates moderate discrimination.

Average precision should be interpreted relative to the approximately 8.97%
outcome prevalence.

---

## Final Test Calibration

| Metric | Estimate | 95% Bootstrap CI |
|---|---:|---:|
| Calibration intercept | 0.2417 | -0.0074 to 0.4892 |
| Calibration slope | 1.1057 | 0.9942 to 1.2142 |
| Quantile ECE | 0.01124 | 0.00901 to 0.01729 |

The calibration-intercept confidence interval includes zero.

The calibration-slope confidence interval is close to and includes values
near the ideal slope of one.

These results do not establish perfect calibration, but they also do not
provide strong evidence of major global calibration failure in the locked
Test sample.

Calibration should be reassessed before use in any new population.

---

## Final Operating-Point Performance

At the frozen threshold of 0.105:

| Metric | Estimate | 95% Bootstrap CI |
|---|---:|---:|
| Sensitivity | 0.3879 | 0.3560 to 0.4187 |
| Specificity | 0.8322 | 0.8247 to 0.8397 |
| PPV | 0.1855 | 0.1716 to 0.1991 |
| NPV | 0.9325 | 0.9292 to 0.9357 |
| F1 | 0.2509 | 0.2323 to 0.2691 |
| Balanced accuracy | 0.6101 | 0.5941 to 0.6258 |
| Alerts per 100 | 18.75 | 18.04 to 19.51 |
| Number needed to evaluate | 5.39 | 5.02 to 5.83 |
| Model net benefit | 0.01686 | 0.01394 to 0.01969 |

The Test alert burden remained consistent with the predefined
moderate-capacity operating scenario.

The positive net-benefit estimate should not be interpreted as prospective
evidence of clinical utility.

---

## Final Test Confusion Matrix

At threshold 0.105:

- true positives: 365
- false positives: 1,603
- true negatives: 7,952
- false negatives: 576

The relatively modest sensitivity is an important limitation of this
operating point.

The threshold reflects an alert-capacity trade-off rather than a requirement
to detect every readmission.

---

## Test Uncertainty Protocol

Final Test uncertainty was characterized after the one-time locked Test
evaluation.

Bootstrap protocol:

- resamples: 2,000
- random seed: 47
- sampling: stratified with replacement
- prevalence preserved within each resample
- confidence level: 95%
- interval method: percentile
- threshold re-selection: no
- model refitting: no
- new Test prediction generation: no

The bootstrap reused the previously saved locked Test probability artifact.

Artifacts:

- `artifacts/metrics/phase11_locked_test_evaluation.json`
- `artifacts/metrics/phase11_locked_test_bootstrap.json`
- `reports/tables/phase11_locked_test_metrics.csv`
- `reports/tables/phase11_locked_test_bootstrap.csv`

---

## Explainability

SHAP analysis was performed before Test access using the Validation
partition.

The SHAP decomposition explains the uncalibrated XGBoost raw margin rather
than the post-calibration probability.

Leading source features by grouped mean absolute SHAP included:

1. `discharge_disposition_id`
2. `number_inpatient`
3. `time_in_hospital`
4. `diag_1`
5. `payer_code`
6. `age`
7. `diabetesMed`
8. `number_diagnoses`

The importance ranking reflects model dependence rather than causal effects.

Because `discharge_disposition_id` was the dominant feature, a targeted
disposition audit was performed.

Terminal and hospice-related disposition IDs audited in the primary cohort
were absent after cohort filtering.

SHAP analysis did not trigger any model, feature, calibration, or threshold
change.

Detailed documentation:

`docs/shap_explainability.md`

---

## Subgroup Evaluation

Subgroup evaluation was performed on Validation before Test access.

Predefined subgroup axes:

- gender
- race
- age

Subgroup estimates were suppressed when groups failed predefined minimum
sample and outcome-count requirements.

No evaluated pairwise subgroup ROC-AUC 95% confidence interval provided clear
evidence of a discrimination difference.

However, the same frozen threshold produced materially different operating
characteristics across some age groups.

For example, older age groups generated substantially larger alert burdens
than younger groups at the same threshold.

Therefore:

- similar discrimination does not imply equal operating characteristics
- this analysis is not a fairness certification
- subgroup-specific thresholds were not selected
- subgroup findings did not trigger model modification

Detailed documentation:

`docs/subgroup_robustness.md`

---

## Repeated-Encounter Robustness

The primary analysis uses one encounter per patient.

A separate robustness analysis evaluated all eligible encounters belonging
to patients from the original Validation partition.

This analysis identified meaningful changes in:

- outcome prevalence
- calibration
- sensitivity
- specificity
- alert burden
- average precision

For all eligible encounters from Validation patients, alerts increased from
approximately:

`18.57 per 100`

to:

`31.64 per 100`

For subsequent eligible encounters only, alert burden increased further to
approximately:

`63.13 per 100`

These results demonstrate that encounter position and case mix materially
affect model behavior.

They should be interpreted as a transportability and operational-stability
warning rather than evidence of improved performance on repeated encounters.

---

## Known Limitations

### Historical data

The source dataset represents care delivered during 1999-2008.

Performance may not transfer to modern clinical practice.

### Internal validation only

The final Test set is held out from the same source dataset and therefore
represents internal rather than external validation.

### Moderate discrimination

The final Test ROC-AUC is approximately 0.665.

The model should therefore not be interpreted as highly discriminative.

### Limited sensitivity at the reference threshold

At threshold 0.105, sensitivity is approximately 38.8%.

A substantial proportion of positive outcomes are not flagged.

### Threshold dependence

Operating characteristics depend strongly on the chosen threshold.

The selected threshold is illustrative and capacity-based rather than
universally optimal.

### Prediction timing

The inclusion of discharge disposition makes the model appropriate only for
an at-discharge or near-discharge prediction setting.

### Dataset-specific coding

Several predictors are administrative or coding-related variables.

Their meaning and availability may differ across hospitals and eras.

### Calibration transportability

Even when discrimination transports, predicted probabilities may require
recalibration in a new population.

### Subgroup uncertainty

Several demographic subgroups contain limited numbers of positive outcomes.

The current analysis cannot establish algorithmic fairness.

### Repeated encounters

Performance changes materially when repeated encounters are included.

The primary model should not be assumed to behave identically for subsequent
hospitalizations.

### No prospective utility evidence

Positive retrospective net benefit does not demonstrate that model use
improves clinical outcomes.

---

## Ethical and Clinical Considerations

Predicted readmission risk may be associated with demographic,
socioeconomic, administrative, utilization, and healthcare-access factors.

Model predictions should therefore not be interpreted as direct measures of
patient behavior or preventability.

High predicted risk should not be used to:

- deny care
- reduce access to treatment
- penalize patients
- determine insurance eligibility
- infer patient responsibility for readmission

Any future clinical use would require careful evaluation of:

- subgroup performance
- calibration
- workflow integration
- intervention availability
- alert burden
- unintended consequences
- fairness
- patient benefit

---

## Data Privacy

No raw clinical dataset is committed to Git.

The following are also excluded from version control:

- patient-level raw data
- primary row-level split assignments
- local row-level locked Test predictions

Published metric artifacts contain aggregate results only.

The final local Test prediction artifact contains:

- Test row index
- binary outcome
- calibrated probability

It does not contain:

- `encounter_id`
- `patient_nbr`

---

## Reproducibility

The project records:

- raw-data provenance
- cohort construction
- split hashes
- preprocessing configuration
- selected hyperparameters
- selected tree count
- calibration procedure
- reference threshold
- source-code hashes before Test access
- environment package versions
- final Test point estimates
- final Test uncertainty intervals

The final Pre-Test freeze allows the development state to be distinguished
from all post-Test reporting work.

Key Phase 11 artifacts:

- `artifacts/metrics/phase11_pretest_freeze.json`
- `artifacts/metrics/phase11_locked_test_evaluation.json`
- `artifacts/metrics/phase11_locked_test_bootstrap.json`

---

## Interpretation Summary

The final frozen calibrated XGBoost model provides reproducible but moderate
discrimination for 30-day readmission prediction in the primary cohort.

On the locked Test set:

- ROC-AUC = 0.6655
- average precision = 0.1954
- Brier score = 0.0782
- calibration slope = 1.1057
- threshold sensitivity = 38.8%
- threshold specificity = 83.2%
- PPV = 18.5%
- alert burden = 18.75 per 100
- model net benefit = 0.0169

The Test results are broadly consistent with the development findings and do
not indicate an unexpected collapse in generalization.

However, the model remains a retrospective research model.

The strongest evidence produced by this project is methodological:
development decisions were isolated from the locked Test set, uncertainty
was quantified, model behavior was characterized across subgroups and
repeated encounters, and limitations were preserved rather than optimized
away after final evaluation.

External and prospective validation would be required before any clinical
deployment claim.