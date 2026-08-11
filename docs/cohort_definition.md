# Clinical Cohort Definition

## Objective

This project predicts 30-day hospital readmission among eligible
hospital encounters in the UCI Diabetes 130-US Hospitals dataset.

Cohort construction is performed before model development so that
clinical eligibility decisions, target definition, and leakage controls
are explicit and reproducible.

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

The original outcome contains:

- `NO`
- `>30`
- `<30`

The binary modeling target is:

```text
readmitted_30d = 1 if readmitted == "<30"
readmitted_30d = 0 otherwise