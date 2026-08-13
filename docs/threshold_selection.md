# Validation-Only Threshold Selection

## Purpose

This document records the operating-threshold analysis performed after
model-family selection and probability calibration.

The selected model family had already been frozen before threshold
selection:

- base learner: tuned XGBoost
- calibration: sigmoid
- final validation model label: `tuned_xgboost_sigmoid`

Threshold selection was performed on the validation partition only.

The test set remained locked and was not used.

## Selected Validation Model

The selected calibrated model is:

`tuned_xgboost_sigmoid`

This model was selected at the end of Phase 7 after discrimination and
calibration analysis.

## Threshold-Selection Policy

Threshold selection was treated as a separate operating-point decision,
not as a model-training step.

We did **not** claim to identify a universally clinically optimal
threshold.

Instead, we defined a **reference operating threshold** based on a
predefined alert-capacity constraint.

## Candidate Threshold Sweep

A dense validation threshold sweep was performed over 99 thresholds.

Artifacts:

- `reports/tables/phase8_validation_threshold_sweep.csv`
- `artifacts/metrics/phase8_validation_threshold_sweep.json`

The sweep quantified, for each threshold:

- sensitivity
- specificity
- positive predictive value (PPV)
- negative predictive value (NPV)
- F1 score
- alerts per 100 patients
- number needed to evaluate
- model net benefit
- treat-all net benefit
- treat-none net benefit

## Predefined Operating Scenarios

Three reference scenarios were extracted from the validation threshold
sweep:

1. high_sensitivity
2. moderate_capacity
3. limited_capacity

Artifacts:

- `reports/tables/phase8_operating_scenarios.csv`
- `artifacts/metrics/phase8_operating_scenarios.json`

### Scenario Summary

| Scenario | Threshold | Sensitivity | Specificity | PPV | NPV | F1 | Alerts/100 | NNE | Net Benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_sensitivity | 0.065 | 0.8025 | 0.3552 | 0.1093 | 0.9480 | 0.1924 | 65.8918 | 9.1481 | 0.0312 |
| moderate_capacity | 0.105 | 0.3567 | 0.8312 | 0.1724 | 0.9291 | 0.2324 | 18.5690 | 5.8006 | 0.0140 |
| limited_capacity | 0.130 | 0.2346 | 0.9140 | 0.2119 | 0.9237 | 0.2227 | 9.9371 | 4.7195 | 0.0094 |

## Bootstrap Uncertainty Analysis

We then estimated the uncertainty of the three operating scenarios using
fixed-threshold stratified bootstrap resampling.

Artifacts:

- `reports/tables/phase8_operating_scenario_bootstrap.csv`
- `artifacts/metrics/phase8_operating_scenario_bootstrap.json`

Bootstrap protocol:

- resamples: 2000
- random seed: 47
- confidence level: 0.95
- thresholds reselected per resample: no

### Bootstrap Results

#### high_sensitivity (threshold = 0.065)

- sensitivity: 0.8025 [0.7771, 0.8280]
- specificity: 0.3552 [0.3457, 0.3648]
- PPV: 0.1093 [0.1058, 0.1128]
- NPV: 0.9480 [0.9415, 0.9545]
- F1: 0.1924 [0.1862, 0.1984]
- alerts per 100: 65.8918 [64.9769, 66.7969]
- number needed to evaluate: 9.1481 [8.8684, 9.4526]
- model net benefit: 0.0312 [0.0288, 0.0335]

#### moderate_capacity (threshold = 0.105)

- sensitivity: 0.3567 [0.3248, 0.3864]
- specificity: 0.8312 [0.8237, 0.8387]
- PPV: 0.1724 [0.1586, 0.1859]
- NPV: 0.9291 [0.9259, 0.9323]
- F1: 0.2324 [0.2140, 0.2509]
- alerts per 100: 18.5690 [17.8256, 19.2835]
- number needed to evaluate: 5.8006 [5.3778, 6.3056]
- model net benefit: 0.0140 [0.0112, 0.0168]

#### limited_capacity (threshold = 0.130)

- sensitivity: 0.2346 [0.2070, 0.2601]
- specificity: 0.9140 [0.9084, 0.9193]
- PPV: 0.2119 [0.1901, 0.2332]
- NPV: 0.9237 [0.9212, 0.9261]
- F1: 0.2227 [0.1983, 0.2450]
- alerts per 100: 9.9371 [9.3655, 10.4802]
- number needed to evaluate: 4.7195 [4.2880, 5.2599]
- model net benefit: 0.0094 [0.0068, 0.0118]

## Reference Threshold Selection

The final reference operating threshold was selected as:

`0.105`

Selected scenario:

`moderate_capacity`

Reasoning:

- it respects the predefined operational capacity target
- it produces 18.569 alerts per 100 patients on validation
- the upper bound of the alert-rate bootstrap interval remains below 20
  alerts per 100 patients
- it offers a clinically plausible compromise between recall and alert
  burden

## Final Frozen Operating Point

Selected model:

`tuned_xgboost_sigmoid`

Selected threshold:

`0.105`

Validation operating metrics:

- sensitivity: 0.3567
- specificity: 0.8312
- PPV: 0.1724
- NPV: 0.9291
- F1: 0.2324
- alerts per 100: 18.5690
- number needed to evaluate: 5.8006
- model net benefit: 0.0140

Alert-rate 95% CI:

`[17.8256, 19.2835]`

Artifact:

`artifacts/metrics/phase8_threshold_selection.json`

## Figures

Phase 8 figures:

- `reports/figures/phase8_validation_threshold_tradeoffs.png`
- `reports/figures/phase8_validation_threshold_tradeoffs.svg`
- `reports/figures/phase8_validation_alert_burden.png`
- `reports/figures/phase8_validation_alert_burden.svg`
- `reports/figures/phase8_validation_decision_curve.png`
- `reports/figures/phase8_validation_decision_curve.svg`

## Data-Use Policy

The following components are now frozen:

- model family
- calibration strategy
- reference operating threshold

At the end of Phase 8:

- model frozen = True
- calibration frozen = True
- threshold frozen = True
- test used = False

No held-out test data were used at any point during threshold selection.

## Next Phase

The next stage is Phase 9:

**SHAP explainability** using the frozen calibrated XGBoost model.