# Clinical Cohort Definition

## Objective

This project predicts 30-day hospital readmission among eligible
hospital encounters in the UCI Diabetes 130-US Hospitals dataset.

Cohort construction is completed before model development so that
eligibility rules, target construction, and leakage controls are explicit
and reproducible.

## Raw Dataset

The validated raw snapshot contains:

| Item | Count |
|---|---:|
| Encounters | 101,766 |
| Unique patients | 71,518 |
| Predictive features | 47 |
| Positive `<30` outcomes | 11,357 |
| Positive rate | 11.16% |

## Outcome Definition

The original outcome contains three values:

- `NO`
- `>30`
- `<30`

The binary modeling target is:

- `readmitted_30d = 1` when `readmitted == "<30"`
- `readmitted_30d = 0` when `readmitted == "NO"` or `readmitted == ">30"`

The positive class therefore represents recorded hospital readmission
within 30 days.

## Identifier Policy

The UCI interface provides two identifiers separately from predictive
features:

- `encounter_id`
- `patient_nbr`

Neither identifier is permitted as a predictive model input.

`encounter_id` is retained for encounter-level integrity checks.

`patient_nbr` is retained for cohort construction and patient-grouped
validation whenever repeated encounters are included.

## Encounter Ordering

For all 71,518 patients in the current snapshot, the first observed row
for a patient corresponds to the minimum observed `encounter_id`.

However, `encounter_id` is not treated as a documented timestamp.

The full dataset contains four global inversions in encounter-ID order.

Therefore, numeric encounter-ID ordering is not interpreted as a
verified chronological time axis.

## Terminal and Hospice Dispositions

The official UCI mapping identifies the following terminal or hospice
discharge disposition IDs:

| ID | Description |
|---:|---|
| 11 | Expired |
| 13 | Hospice / home |
| 14 | Hospice / medical facility |
| 19 | Expired at home, Medicaid only, hospice |
| 20 | Expired in a medical facility, Medicaid only, hospice |
| 21 | Expired, place unknown, Medicaid only, hospice |

Disposition ID 21 is defined in the mapping but is not observed in the
current dataset snapshot.

Across all encounters, 2,423 terminal or hospice encounters are observed,
affecting 2,399 unique patients.

These encounters are excluded from eligible readmission cohorts.

## Primary Cohort

The primary cohort follows a reference-style patient-independent design.

Construction:

1. Start with all 101,766 encounters.
2. Retain the first observed encounter for each patient.
3. This leaves 71,518 encounters for 71,518 patients.
4. Exclude first encounters with terminal or hospice dispositions.
5. The final primary cohort contains 69,973 encounters.

Final primary cohort:

| Item | Value |
|---|---:|
| Encounters | 69,973 |
| Unique patients | 69,973 |
| Positive 30-day readmissions | 6,277 |
| Negative outcomes | 63,696 |
| Positive rate | 8.97% |

Each patient appears exactly once.

## Sensitivity Cohort

The sensitivity cohort changes the order of eligibility filtering and
patient selection.

Construction:

1. Exclude all terminal or hospice encounters.
2. Retain the first remaining eligible observed encounter for each patient.

Final sensitivity cohort:

| Item | Value |
|---|---:|
| Encounters | 69,990 |
| Unique patients | 69,990 |
| Positive 30-day readmissions | 6,285 |
| Negative outcomes | 63,705 |
| Positive rate | 8.98% |

This strategy recovers 17 patients whose first observed encounter had a
terminal or hospice disposition but who also have another eligible record.

Because a verified encounter timestamp is unavailable, this cohort is
treated as a sensitivity analysis rather than the primary cohort.

## All-Eligible Robustness Cohort

The third cohort retains every non-terminal eligible encounter.

Final robustness cohort:

| Item | Value |
|---|---:|
| Encounters | 99,343 |
| Unique patients | 69,990 |
| Positive 30-day readmissions | 11,314 |
| Negative outcomes | 88,029 |
| Positive rate | 11.39% |

There are 16,341 patients with repeated encounters in this cohort.

Any train, validation, and test split using this cohort must therefore be
grouped by `patient_nbr`.

## Reference Reproduction Audit

A reproducibility audit was performed against values reported for the
reference-style cohort.

Reference values used for comparison:

- cohort size: 69,984
- positive `<30` outcomes: 6,459

The current reproducible pipeline produces:

- cohort size: 69,973
- positive `<30` outcomes: 6,277

Differences:

- cohort size: -11
- positive outcomes: -182

All combinations of the terminal and hospice disposition IDs were audited.

No combination reproduced the reported cohort size.

No combination reproduced both the reported cohort size and positive
outcome count.

Eligibility rules are therefore not modified merely to force agreement
with reference numbers.

The discrepancy is retained as a reproducibility limitation.

## Data-Consistency Audit

Some patients have higher-valued encounter identifiers associated with
records appearing after encounters coded as hospice or expired when
sorted numerically by `encounter_id`.

Because encounter IDs are not documented timestamps, this finding is
treated as a data-consistency limitation rather than proof of
chronological post-terminal encounters.

Row-level records generated for this audit are stored only inside ignored
interim directories.

Only aggregate summaries are version controlled.

## Leakage Controls

The cohort pipeline enforces:

- unique `encounter_id`
- non-missing `patient_nbr`
- exclusion of terminal and hospice encounters
- one encounter per patient in the primary cohort
- one encounter per patient in the sensitivity cohort
- exact binary-target mapping
- patient grouping when repeated encounters are retained
- exclusion of `encounter_id` and `patient_nbr` from predictive features

## Generated Cohorts

Local cohort files are written to:

`data/interim/cohorts/`

The generated files are:

- `primary.csv`
- `sensitivity_first_eligible.csv`
- `all_eligible_encounters.csv`

These files contain row-level patient data and are excluded from version
control.

## Reproducibility

Build all cohorts with:

    python scripts/build_cohorts.py

Run cohort and pipeline tests with:

    pytest

## Intended Primary Analysis

The primary modeling workflow will use:

1. primary cohort
2. leakage-safe train/validation/test split
3. preprocessing learned from training data only
4. logistic-regression baseline
5. XGBoost model
6. class-imbalance analysis
7. discrimination evaluation
8. calibration analysis
9. validation-only threshold selection
10. test-set evaluation
11. SHAP explainability
12. subgroup and robustness analysis

The sensitivity and all-eligible cohorts are reserved for later
robustness analyses.