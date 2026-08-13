# SHAP Explainability Analysis

## Purpose

Phase 9 evaluates the interpretability of the frozen readmission model using
SHAP (SHapley Additive exPlanations).

The objective of this phase is to characterize:

- which transformed model features contribute most strongly to predictions
- which original clinical source variables are most influential
- how important numeric features affect the model output
- how individual predictions are formed
- whether local and global SHAP explanations satisfy additivity
- whether highly influential variables reveal potential leakage or
  target-adjacent information

No model training decisions, calibration decisions, or threshold decisions
were changed during this phase.

The test set remained locked throughout Phase 9.

---

## Frozen Model Configuration

The final development model entering Phase 9 was already frozen:

- base model: tuned XGBoost
- selected tree count: 155
- calibration method: sigmoid
- final development variant: `tuned_xgboost_sigmoid`
- reference operating threshold: `0.105`

The model family, calibration strategy, and threshold were all selected
before SHAP analysis.

Phase 9 therefore performs model interpretation only.

---

## Explanation Target

SHAP explanations were generated for the frozen base XGBoost model.

The calibrated sigmoid probabilities were retained for prediction and
threshold-based classification, but the calibrated probability was not
directly decomposed with SHAP.

The SHAP explanation space is:

`uncalibrated XGBoost raw margin / log-odds`

For each encounter:

`expected raw margin + sum(SHAP values) = model raw margin`

Therefore, positive SHAP values increase the XGBoost raw risk score and
negative SHAP values decrease it.

These SHAP contributions should not be interpreted as causal effects.

---

## Data-Use Policy

The following policy was maintained throughout Phase 9:

- model fitting split: Train
- explanation split: Validation
- test set used: no
- model changed after SHAP review: no
- calibration changed after SHAP review: no
- threshold changed after SHAP review: no
- patient identifiers saved in SHAP outputs: no

The Validation partition contained:

- 10,496 encounters
- 942 positive 30-day readmissions

---

## Model Reproduction Before SHAP

Before SHAP values were calculated, the frozen Tuned XGBoost model was
reconstructed from the previously selected hyperparameters and tree count.

The full-Train preprocessing pipeline produced:

- 225 transformed features

The reconstructed model reproduced the previously stored Validation
probabilities with a maximum absolute difference of:

`2.73e-08`

This was below the predefined reproduction tolerance.

Therefore:

`Probability reproduction: PASS`

---

## SHAP Additivity Audit

Tree SHAP values were generated using:

- `TreeExplainer`
- `model_output="raw"`
- `feature_perturbation="tree_path_dependent"`
- `check_additivity=True`

The expected SHAP value was approximately:

`-1.914126`

The global Validation additivity audit produced:

- maximum absolute additivity error: `2.96e-06`
- mean absolute additivity error: `4.76e-07`

The inverse-logistic transformation of the raw XGBoost margin also
reproduced the base XGBoost probability with a maximum error of:

`4.58e-08`

These results confirm that the SHAP decomposition accurately reconstructed
the frozen XGBoost raw output.

---

## Global Transformed-Feature Importance

Global transformed-feature importance was measured using mean absolute SHAP
value across the complete Validation partition.

The leading transformed features were:

| Rank | Transformed feature | Source feature | Mean absolute SHAP |
|---:|---|---|---:|
| 1 | `categorical__discharge_disposition_id_1` | `discharge_disposition_id` | 0.218130 |
| 2 | `numeric__number_inpatient` | `number_inpatient` | 0.123497 |
| 3 | `numeric__time_in_hospital` | `time_in_hospital` | 0.066874 |
| 4 | `categorical__payer_code_Missing` | `payer_code` | 0.043570 |
| 5 | `diagnosis__diag_1_Circulatory` | `diag_1` | 0.041111 |
| 6 | `categorical__discharge_disposition_id_22` | `discharge_disposition_id` | 0.038219 |
| 7 | `categorical__age_[50-60)` | `age` | 0.037650 |
| 8 | `numeric__number_diagnoses` | `number_diagnoses` | 0.030721 |
| 9 | `categorical__discharge_disposition_id_3` | `discharge_disposition_id` | 0.029586 |
| 10 | `categorical__diabetesMed_No` | `diabetesMed` | 0.027777 |

The importance ranking describes model dependence, not causal or clinical
effect size.

---

## Aggregation to Clinical Source Features

One-hot encoded and otherwise transformed features were also aggregated
back to their original clinical source variables.

For each source feature, transformed SHAP contributions were first summed
within each encounter and then summarized across Validation.

The leading source features were:

| Rank | Source feature | Transformed features | Mean absolute grouped SHAP |
|---:|---|---:|---:|
| 1 | `discharge_disposition_id` | 21 | 0.256966 |
| 2 | `number_inpatient` | 1 | 0.123497 |
| 3 | `time_in_hospital` | 1 | 0.066874 |
| 4 | `diag_1` | 10 | 0.049384 |
| 5 | `payer_code` | 18 | 0.047564 |
| 6 | `age` | 10 | 0.040669 |
| 7 | `diabetesMed` | 2 | 0.039044 |
| 8 | `number_diagnoses` | 1 | 0.030721 |
| 9 | `num_lab_procedures` | 1 | 0.024690 |
| 10 | `number_emergency` | 1 | 0.023143 |

Additional important source features included:

- `num_medications`
- `diag_2`
- `medical_specialty`
- `insulin`
- `admission_source_id`
- `diag_3`
- `admission_type_id`
- `metformin`
- `number_outpatient`
- `A1Cresult`

---

## Discharge-Disposition Audit

`discharge_disposition_id` was the dominant SHAP source feature.

Because discharge disposition can contain target-adjacent information, a
specific audit was performed before further interpretation.

The following disposition IDs were audited:

- 11
- 13
- 14
- 19
- 20
- 21

The number of encounters in the final primary cohort containing these
audited disposition IDs was:

`0`

Therefore:

`Discharge-disposition audit: PASS`

This finding reduced concern that the observed SHAP dominance was caused by
the audited death/hospice-related or otherwise non-readmittable categories.

However, `discharge_disposition_id` is still information that becomes
available at discharge rather than at initial hospital admission.

Consequently, the current model should be interpreted as an:

`at-or-near-discharge 30-day readmission prediction model`

It should not be presented as an admission-time prediction model.

A future admission-time model would require a more restrictive feature
availability policy.

---

## Global SHAP Figures

The following global SHAP figures were generated:

- `reports/figures/phase9_validation_shap_beeswarm.png`
- `reports/figures/phase9_validation_shap_beeswarm.svg`
- `reports/figures/phase9_validation_source_shap_importance.png`
- `reports/figures/phase9_validation_source_shap_importance.svg`

The source-feature importance ranking was based on the complete Validation
partition.

For visualization only, a reproducible stratified sample of 3,000
Validation encounters was used for the beeswarm and dependence plots.

Visualization sample:

- rows: 3,000
- positives: 269
- random seed: 49
- used for global ranking: no

---

## Numeric Dependence Analysis

The three highest-ranked numeric source features were selected for SHAP
dependence visualization:

1. `number_inpatient`
2. `time_in_hospital`
3. `number_diagnoses`

The corresponding figures are:

- `reports/figures/phase9_validation_shap_dependence_number_inpatient.png`
- `reports/figures/phase9_validation_shap_dependence_time_in_hospital.png`
- `reports/figures/phase9_validation_shap_dependence_number_diagnoses.png`

SVG versions were also generated.

These plots characterize nonlinear model behavior and variation in local
feature contributions.

They do not establish causal relationships.

---

## Predefined Local Explanation Protocol

Local cases were selected using a predefined protocol before reviewing
individual explanations.

Case selection used:

- calibrated probability:
  `tuned_xgboost_sigmoid_probability`
- frozen reference threshold:
  `0.105`

Five Validation cases were selected:

1. highest-probability true positive
2. highest-probability false positive
3. false negative closest to the threshold from below
4. lowest-risk true negative
5. closest unused observation to the threshold

No encounter identifier, patient identifier, or source-row identifier was
saved.

Only positional Validation row numbers were retained for reproducibility.

---

## Selected Local Cases

| Case | Validation row | Classification | Calibrated probability |
|---|---:|---|---:|
| High-confidence true positive | 7260 | TP | 0.817505 |
| High-confidence false positive | 4559 | FP | 0.808658 |
| Near-threshold false negative | 3283 | FN | 0.104979 |
| Low-risk true negative | 8565 | TN | 0.043255 |
| Closest unused case to threshold | 6600 | TN | 0.104993 |

The maximum local SHAP additivity error across these cases was:

`4.81e-07`

Therefore, local SHAP decomposition also passed the additivity audit.

---

## High-Confidence True Positive

The high-confidence true positive had:

- calibrated probability: `0.817505`
- base XGBoost probability: `0.613764`
- raw XGBoost margin: `0.463160`

Major positive source contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `discharge_disposition_id` | 22 | +1.765499 |
| `payer_code` | SP | +0.246370 |
| `medical_specialty` | Missing | +0.182430 |
| `diag_1` | 434 | +0.071881 |
| `number_diagnoses` | 4 | +0.048688 |

Major negative contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `number_inpatient` | 0 | -0.037153 |
| `num_medications` | 6 | -0.033741 |
| `time_in_hospital` | 3 | -0.029038 |

The strongly positive disposition-related contribution dominated the local
prediction.

---

## High-Confidence False Positive

The high-confidence false positive had:

- calibrated probability: `0.808658`
- base XGBoost probability: `0.606522`
- raw XGBoost margin: `0.432716`

Major positive contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `discharge_disposition_id` | 22 | +1.710795 |
| `payer_code` | SP | +0.244724 |
| `medical_specialty` | Missing | +0.185204 |
| `number_diagnoses` | 7 | +0.068018 |
| `diag_1` | 434 | +0.050664 |

Major negative contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `number_inpatient` | 0 | -0.046821 |
| `time_in_hospital` | 3 | -0.031258 |
| `num_lab_procedures` | 41 | -0.019075 |

The explanation profile was notably similar to the high-confidence true
positive.

This illustrates an important model limitation: patients with highly
similar model-recognized feature patterns can still experience different
real-world outcomes.

A high predicted probability should therefore not be interpreted as
certainty of readmission.

---

## Near-Threshold False Negative

The selected false negative had:

- calibrated probability: `0.104979`
- reference threshold: `0.105`

Its probability was only approximately:

`0.000021`

below the reference threshold.

Positive contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `discharge_disposition_id` | 3 | +0.357164 |
| `repaglinide` | Steady | +0.096834 |
| `number_emergency` | 1 | +0.070974 |
| `diabetesMed` | Yes | +0.034255 |
| `age` | [70-80) | +0.025666 |

Negative contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `number_inpatient` | 0 | -0.101778 |
| `payer_code` | MC | -0.078657 |
| `diag_1` | 715 | -0.076253 |
| `time_in_hospital` | 3 | -0.047248 |

This case demonstrates that some classification errors occur extremely
close to the selected operating threshold rather than because the model
assigned an extremely low probability.

---

## Low-Risk True Negative

The low-risk true negative had:

- calibrated probability: `0.043255`
- base XGBoost probability: `0.042144`
- raw XGBoost margin: `-3.123600`

Strong negative contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `discharge_disposition_id` | 1 | -0.300420 |
| `age` | [50-60) | -0.218937 |
| `payer_code` | CP | -0.112394 |
| `time_in_hospital` | 1 | -0.109649 |
| `num_medications` | 5 | -0.104827 |
| `number_inpatient` | 0 | -0.082425 |

The strongest positive contribution was comparatively small:

- `diabetesMed = Yes`: `+0.024952`

This represents a relatively clear low-risk prediction profile.

---

## Near-Threshold True Negative

The closest unused Validation case to the threshold had:

- calibrated probability: `0.104993`
- reference threshold: `0.105`

The probability was only approximately:

`0.000007`

below the decision threshold.

Positive contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `number_inpatient` | 1 | +0.290450 |
| `time_in_hospital` | 8 | +0.089893 |
| `number_emergency` | 2 | +0.084382 |
| `number_diagnoses` | 9 | +0.036758 |
| `num_medications` | 23 | +0.027261 |
| `num_lab_procedures` | 68 | +0.024469 |

Negative contributions included:

| Feature | Value | SHAP contribution |
|---|---|---:|
| `discharge_disposition_id` | 1 | -0.184935 |
| `A1Cresult` | Norm | -0.063618 |
| `diag_1` | 38 | -0.054581 |
| `payer_code` | MD | -0.026972 |

This case illustrates competing positive and negative model signals that
produce a probability almost exactly at the reference operating threshold.

---

## Local Waterfall Figures

Waterfall plots were generated for all five predefined cases:

- `phase9_validation_shap_waterfall_high_confidence_true_positive`
- `phase9_validation_shap_waterfall_high_confidence_false_positive`
- `phase9_validation_shap_waterfall_near_threshold_false_negative`
- `phase9_validation_shap_waterfall_low_risk_true_negative`
- `phase9_validation_shap_waterfall_closest_unused_to_threshold`

Each figure was saved in both PNG and SVG format.

The waterfall plots explain the raw XGBoost margin.

The calibrated probability shown in each figure is provided for clinical
operating-point context and is not the output directly decomposed by SHAP.

---

## Main Interpretability Findings

Phase 9 produced several important findings.

### 1. Discharge disposition is the dominant model feature

`discharge_disposition_id` had substantially higher global SHAP importance
than any other source variable.

The targeted cohort audit found no encounters containing the audited
non-readmittable disposition IDs.

Nevertheless, this feature confirms that the model's intended prediction
time is at or near discharge.

### 2. Prior healthcare utilization is strongly represented

Important utilization-related variables included:

- `number_inpatient`
- `number_emergency`
- `number_outpatient`
- `time_in_hospital`
- `number_diagnoses`
- `num_medications`
- `num_lab_procedures`

These variables repeatedly appeared in both global and local
explanations.

### 3. High-confidence false positives can resemble true positives

The highest-probability TP and FP had very similar local SHAP profiles.

This demonstrates that model explanations can clarify why an error
occurred even when the model itself behaved consistently with its learned
risk patterns.

### 4. Threshold errors can be extremely close to the operating boundary

The selected false negative and near-threshold true negative were both
within approximately `2.1e-05` of the frozen threshold.

This reinforces the interpretation of the threshold as an operational
decision boundary rather than a biological distinction between two
fundamentally different patient groups.

---

## Interpretation Limitations

Several limitations must be maintained when interpreting SHAP results.

### SHAP is not causal

A positive SHAP value means that a feature increased the model output for
that observation relative to the SHAP reference value.

It does not imply that modifying the feature would change the patient's
true readmission risk.

### Global importance does not imply clinical usefulness

A feature may be highly important because the model relies on it strongly.
This does not automatically make it an appropriate intervention target.

### Correlated features can redistribute attribution

Clinical variables can be correlated, and SHAP attribution may be shared
or redistributed among related predictors.

### One-hot encoded categories require careful interpretation

Category-level transformed SHAP values should not be interpreted in
isolation as universal risk effects.

Source-feature aggregation was therefore also used.

### The current model is not an admission-time model

Because discharge-related information is retained, predictions should be
interpreted as being generated at or near discharge.

---

## Phase 9 Artifacts

### Metrics

- `artifacts/metrics/phase9_shap_validation.json`
- `artifacts/metrics/phase9_discharge_disposition_audit.json`
- `artifacts/metrics/phase9_shap_figures.json`
- `artifacts/metrics/phase9_local_explanations.json`

### Tables

- `reports/tables/phase9_transformed_feature_metadata.csv`
- `reports/tables/phase9_transformed_shap_importance.csv`
- `reports/tables/phase9_source_shap_importance.csv`
- `reports/tables/phase9_discharge_disposition_audit.csv`
- `reports/tables/phase9_local_explanation_cases.csv`
- `reports/tables/phase9_local_source_contributions.csv`

### Figures

Global figures include:

- SHAP beeswarm
- grouped source-feature importance
- numeric dependence plots

Local figures include:

- five predefined SHAP waterfall explanations

All figures were exported in PNG and SVG formats.

---

## Phase 9 Conclusion

Phase 9 successfully added global and local explainability to the frozen
readmission model without reopening model selection.

The analysis demonstrated:

- reproducible reconstruction of the frozen XGBoost model
- accurate SHAP additivity
- interpretable global feature rankings
- aggregation from transformed features to clinical source variables
- explicit audit of the dominant discharge-disposition variable
- reproducible local explanation case selection
- clinically interpretable examples of TP, FP, FN, and TN predictions
- no use of the locked test set

At the end of Phase 9:

- model frozen: True
- calibration frozen: True
- threshold frozen: True
- test used: False

The next phase will evaluate subgroup performance and robustness while
preserving the frozen model, calibration strategy, and reference threshold.