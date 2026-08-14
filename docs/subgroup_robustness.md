# Subgroup Evaluation and Robustness Analysis

## Overview

Phase 10 characterizes the frozen development model across predefined
demographic subgroups and evaluates robustness to repeated eligible
encounters.

This phase is descriptive and exploratory. It does not modify the model,
calibration procedure, or reference operating threshold.

Frozen configuration:

- model: `tuned_xgboost_sigmoid`
- XGBoost tree count: 155
- calibration: Train-only 5-fold sigmoid calibration
- reference threshold: 0.105
- primary evaluation split: Validation
- Test used: False

The locked Test set remains untouched throughout Phase 10.

## Analysis Principles

Phase 10 follows four constraints:

1. subgroup definitions are established before reviewing subgroup metrics
2. small subgroups remain visible descriptively but unstable performance
   estimates are suppressed
3. subgroup findings do not trigger model or threshold re-selection
4. robustness analyses retain the original patient-level split boundaries

The subgroup analysis is not a formal fairness certification and does not
establish causal differences between demographic groups.

## Subgroup Definitions

Three predefined demographic axes are evaluated.

### Gender

Observed dataset categories are evaluated directly after normalizing missing
values.

### Race

Observed race categories are retained without combining uncommon groups merely
to increase reportability.

Missing race values are reported as a separate descriptive category.

### Age

The original decade-based age intervals are combined into four predefined
groups:

- `<50`
- `50-69`
- `70-89`
- `90+`

These broad categories are intended to provide clinically interpretable
age-stratified performance summaries while preserving adequate sample sizes
where possible.

## Reporting Eligibility

Subgroup performance metrics are reported only when all of the following are
satisfied:

- at least 200 observations
- at least 20 positive outcomes
- at least 20 negative outcomes

Groups below these thresholds remain in the descriptive table, but their
performance metrics are suppressed.

These thresholds are pragmatic reporting safeguards rather than guarantees of
statistical precision.

## Overall Validation Reference

The frozen calibrated model on the primary Validation cohort contains:

- encounters: 10,496
- positive 30-day readmissions: 942
- prevalence: 8.97%
- ROC-AUC: 0.6508
- average precision: 0.1716
- Brier score: 0.0794
- reference threshold: 0.105
- sensitivity: 0.3567
- specificity: 0.8312
- PPV: 0.1724
- alerts per 100 encounters: 18.57

These values are used only as reference values for subgroup characterization.

## Subgroup Performance

### Gender

Female:

- n = 5,611
- positives = 508
- prevalence = 9.05%
- ROC-AUC = 0.6619
- 95% bootstrap CI = 0.6365 to 0.6859
- sensitivity = 0.3701
- specificity = 0.8223
- PPV = 0.1717
- alerts per 100 = 19.52

Male:

- n = 4,885
- positives = 434
- prevalence = 8.88%
- ROC-AUC = 0.6375
- 95% bootstrap CI = 0.6092 to 0.6660
- sensitivity = 0.3410
- specificity = 0.8414
- PPV = 0.1733
- alerts per 100 = 17.48

The estimated ROC-AUC difference for Female minus Male is:

`+0.0244 [95% CI -0.0141, +0.0607]`

The interval includes zero, so the Validation data do not provide clear
evidence of a difference in discrimination by gender.

At the frozen threshold, Female encounters have slightly lower specificity and
approximately 2 additional alerts per 100 encounters compared with Male
encounters.

These operating-point differences are descriptive and should not be interpreted
as proof of demographic unfairness.

## Race

Reportable groups include African American, Caucasian, and observations with
missing race.

African American:

- n = 1,896
- positives = 169
- prevalence = 8.91%
- ROC-AUC = 0.6761
- 95% bootstrap CI = 0.6349 to 0.7173

Caucasian:

- n = 7,885
- positives = 727
- prevalence = 9.22%
- ROC-AUC = 0.6445
- 95% bootstrap CI = 0.6217 to 0.6657

Missing race:

- n = 288
- positives = 21
- prevalence = 7.29%
- ROC-AUC = 0.6740
- 95% bootstrap CI = 0.5523 to 0.7862

Asian, Hispanic, and Other groups do not satisfy the predefined reporting
criteria and therefore retain descriptive counts while performance estimates
are suppressed.

The estimated African American minus Caucasian ROC-AUC difference is:

`+0.0316 [95% CI -0.0161, +0.0740]`

The interval includes zero.

The missing-race subgroup has only 21 positive outcomes and correspondingly
wide uncertainty. Its estimates should therefore be treated as highly
exploratory.

Differences in average precision, PPV, and Brier score across race groups
require additional caution because these metrics are affected by outcome
prevalence and case mix.

## Age

### Younger than 50 years

- n = 1,653
- positives = 115
- prevalence = 6.96%
- ROC-AUC = 0.6692
- 95% bootstrap CI = 0.6191 to 0.7203
- sensitivity = 0.1913
- specificity = 0.9194
- alerts per 100 = 8.83

### Age 50-69

- n = 4,211
- positives = 332
- prevalence = 7.88%
- ROC-AUC = 0.6569
- 95% bootstrap CI = 0.6238 to 0.6875
- sensitivity = 0.3343
- specificity = 0.8775
- alerts per 100 = 13.92

### Age 70-89

- n = 4,360
- positives = 462
- prevalence = 10.60%
- ROC-AUC = 0.6191
- 95% bootstrap CI = 0.5911 to 0.6462
- sensitivity = 0.4069
- specificity = 0.7555
- alerts per 100 = 26.17

### Age 90+

- n = 272
- positives = 33
- prevalence = 12.13%
- ROC-AUC = 0.6042
- 95% bootstrap CI = 0.4945 to 0.7095
- sensitivity = 0.4545
- specificity = 0.7448
- alerts per 100 = 27.94

The oldest subgroup has substantial statistical uncertainty because of its
small sample and event count.

## Pairwise Discrimination Differences

Independent stratified bootstrap comparisons are used for disjoint subgroups.

All evaluated pairwise ROC-AUC 95% confidence intervals include zero.

Selected comparisons:

- Female minus Male:
  `+0.0244 [-0.0141, +0.0607]`
- African American minus Caucasian:
  `+0.0316 [-0.0161, +0.0740]`
- age 50-69 minus age 70-89:
  `+0.0378 [-0.0049, +0.0805]`
- age 70-89 minus age <50:
  `-0.0501 [-0.1099, +0.0089]`
- age 90+ minus age <50:
  `-0.0650 [-0.1794, +0.0545]`

Therefore, the current Validation data do not establish a clear subgroup
difference in discrimination.

The comparisons are exploratory and no multiple-comparison adjustment is
applied.

## Operating-Point Differences

Although discrimination differences are uncertain, the fixed threshold of
0.105 produces different operating characteristics across some subgroups.

The strongest pattern occurs across age groups.

Age 70-89 minus age <50:

- sensitivity difference: +0.2156
- 95% CI: +0.1266 to +0.2980
- specificity difference: -0.1639
- 95% CI: -0.1819 to -0.1449
- alerts per 100 difference: +17.34
- 95% CI: +15.41 to +19.12

Age 90+ minus age <50:

- sensitivity difference: +0.2632
- 95% CI: +0.0899 to +0.4454
- specificity difference: -0.1746
- 95% CI: -0.2306 to -0.1167
- alerts per 100 difference: +19.11
- 95% CI: +13.60 to +24.44

These findings indicate that the same frozen risk threshold translates into
substantially different alert burdens and sensitivity/specificity trade-offs
across age strata.

They do not justify subgroup-specific threshold optimization within this
development analysis.

## Bootstrap Uncertainty

Reportable subgroup metrics use:

- 2,000 bootstrap resamples
- stratified sampling with replacement within each subgroup
- base random seed: 47
- percentile 95% confidence intervals

Stratification preserves the positive and negative class counts within each
subgroup bootstrap sample.

For comparisons between disjoint demographic groups, each group is resampled
independently before calculating the metric difference.

## Repeated-Encounter Robustness

The primary analysis intentionally retains one encounter per patient.

A predefined robustness cohort contains all eligible encounters after excluding
terminal and hospice dispositions.

To preserve the frozen split structure, repeated-encounter robustness is
restricted to patients already assigned to the original Validation partition.

No new Train/Validation/Test split is created.

The frozen Train-only calibrated model is reconstructed before this analysis.

Maximum difference between reconstructed and previously saved Validation
probabilities:

`9.71445146547e-17`

This is well below the predefined reproduction tolerance and confirms exact
reproduction of the frozen development model.

## Repeated-Encounter Results

### Primary Validation

- encounters = 10,496
- unique patients = 10,496
- positives = 942
- prevalence = 8.97%
- ROC-AUC = 0.6508
- average precision = 0.1716
- Brier score = 0.0794
- calibration intercept = -0.1182
- calibration slope = 0.9484
- sensitivity = 0.3567
- specificity = 0.8312
- PPV = 0.1724
- alerts per 100 = 18.57

### All Eligible Encounters From Validation Patients

- encounters = 14,854
- unique patients = 10,496
- positives = 1,727
- prevalence = 11.63%
- ROC-AUC = 0.6723
- average precision = 0.2238
- Brier score = 0.0988
- calibration intercept = -0.2761
- calibration slope = 0.8349
- sensitivity = 0.5333
- specificity = 0.7121
- PPV = 0.1960
- alerts per 100 = 31.64

### Subsequent Eligible Encounters Only

- encounters = 4,358
- unique patients = 2,453
- positives = 785
- prevalence = 18.01%
- ROC-AUC = 0.6165
- average precision = 0.2725
- Brier score = 0.1456
- calibration intercept = -0.5413
- calibration slope = 0.5759
- sensitivity = 0.7452
- specificity = 0.3938
- PPV = 0.2127
- alerts per 100 = 63.13

The repeated-encounter populations have materially different prevalence,
calibration, and operating characteristics.

The increased average precision in repeated encounters must not be interpreted
as an unqualified performance improvement because average precision is strongly
affected by the higher outcome prevalence.

## Patient-Cluster Bootstrap

Repeated encounters from the same patient are correlated.

Therefore, robustness uncertainty is estimated using a paired patient-cluster
bootstrap rather than a row-level bootstrap.

Protocol:

- bootstrap unit: patient
- 2,000 resamples
- random seed: 47
- percentile 95% confidence intervals
- all encounters belonging to a sampled patient are replicated together
- difference definition:
  `all eligible encounters minus primary Validation`

Results:

| Metric | Difference | 95% CI |
|---|---:|---:|
| ROC-AUC | +0.0215 | +0.0080 to +0.0358 |
| Average precision | +0.0521 | +0.0291 to +0.0781 |
| Brier score | +0.0194 | +0.0163 to +0.0226 |
| Sensitivity | +0.1766 | +0.1529 to +0.2016 |
| Specificity | -0.1191 | -0.1265 to -0.1118 |
| PPV | +0.0236 | +0.0086 to +0.0389 |
| Alerts per 100 | +13.07 | +12.29 to +13.93 |

All reported intervals exclude zero.

These findings support a reproducible encounter-position and case-mix shift
when the frozen model is applied beyond the one-encounter-per-patient primary
cohort.

This is a transportability and operational-stability finding rather than
evidence that repeated encounters improve the underlying model.

## Interpretation

The Phase 10 analyses support four main conclusions.

First, the frozen model shows moderate discrimination across the reportable
demographic subgroups, with substantial uncertainty in smaller groups.

Second, no pairwise subgroup ROC-AUC comparison provides clear evidence of a
difference in discrimination because every 95% confidence interval includes
zero.

Third, a single fixed threshold can produce meaningfully different operational
behavior across subgroups, particularly across age groups. Similar
discrimination therefore does not imply similar sensitivity, specificity, or
alert burden.

Fourth, repeated encounters represent a materially different evaluation
setting. Their prevalence, calibration, and threshold behavior differ from the
primary one-encounter-per-patient Validation cohort.

## What Phase 10 Does Not Show

These analyses do not establish:

- causal demographic effects
- absence or presence of algorithmic fairness
- external transportability
- clinical effectiveness
- clinical utility in prospective deployment
- an optimal threshold for any subgroup
- superiority of the model on repeated encounters

No subgroup-specific threshold is selected.

No model is retrained in response to subgroup findings.

The Test partition remains locked.

## Figures

Phase 10 produces the following publication-quality figures in both PNG and SVG
formats:

- `phase10_validation_subgroup_roc_auc`
- `phase10_validation_subgroup_alert_burden`
- `phase10_repeated_encounter_operating_metrics`
- `phase10_cluster_bootstrap_metric_differences`
- `phase10_cluster_bootstrap_alert_difference`

Figures are stored under:

`reports/figures/`

## Main Artifacts

Subgroup performance:

`reports/tables/phase10_validation_subgroup_performance.csv`

Subgroup bootstrap intervals:

`reports/tables/phase10_validation_subgroup_bootstrap.csv`

Independent subgroup differences:

`reports/tables/phase10_validation_subgroup_differences.csv`

Repeated-encounter robustness:

`reports/tables/phase10_repeated_encounter_robustness.csv`

Patient-cluster bootstrap:

`reports/tables/phase10_cluster_bootstrap.csv`

Machine-readable summaries are stored under:

`artifacts/metrics/`

## Data Governance

The Phase 10 outputs do not save:

- `encounter_id`
- `patient_nbr`
- row-level patient predictions

Patient identifiers are used only transiently to preserve patient-level split
boundaries and construct the patient-cluster bootstrap.

The row-level split assignment artifact remains excluded from Git.

## Model Freeze

Phase 10 does not change:

- model family
- XGBoost hyperparameters
- tree count
- preprocessing
- calibration method
- calibration fitting protocol
- reference threshold

The model remains frozen before final Test evaluation.

## Test-Set Policy

The Test partition is not accessed during subgroup analysis, uncertainty
estimation, robustness analysis, or figure generation.

`Test used: False`

Final locked Test evaluation remains reserved for the final release phase.