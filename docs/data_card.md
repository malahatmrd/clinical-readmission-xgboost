# Data Card

## Dataset

**Diabetes 130-US Hospitals for Years 1999-2008**

UCI Machine Learning Repository — Dataset ID 296.

This project uses de-identified inpatient encounter data for patients
with diabetes across multiple US hospitals.

## Prediction Question

Can information available from a hospital encounter be used to estimate
the risk of readmission within 30 days?

## Raw Dataset Snapshot

| Item | Value |
|---|---:|
| Encounters | 101,766 |
| Unique patients | 71,518 |
| Predictive features | 47 |
| Identifier columns | 2 |
| Target columns | 1 |

## Identifier Policy

The UCI interface exposes the following identifiers separately from the
predictive features:

- `encounter_id`
- `patient_nbr`

These identifiers are stored separately from model features.

`encounter_id` is used only for encounter integrity checks.

`patient_nbr` is used for patient-level cohort construction and
group-aware validation.

Neither identifier is permitted as a predictive model input.

## Target

Original target:

- `NO`
- `>30`
- `<30`

The primary binary prediction target is defined as:

- Positive: `<30`
- Negative: `NO` or `>30`

Observed distribution:

| Label | Count | Percentage |
|---|---:|---:|
| NO | 54,864 | 53.91% |
| >30 | 35,545 | 34.93% |
| <30 | 11,357 | 11.16% |

The positive class is therefore substantially imbalanced.

Accuracy alone will not be used as the primary model-selection metric.

## Repeated Encounters

There are:

- 101,766 encounters
- 71,518 unique patients
- 16,773 patients with more than one encounter

23.45% of patients have repeated encounters.

The maximum number of encounters observed for a single patient is 40.

This creates a potential patient-leakage problem if ordinary row-level
random splitting is used.

The modeling pipeline will therefore use patient-aware cohort and
validation strategies.

## Missingness Audit

Five features have particularly high missingness:

| Feature | Missing |
|---|---:|
| weight | 96.86% |
| max_glu_serum | 94.75% |
| A1Cresult | 83.28% |
| medical_specialty | 49.08% |
| payer_code | 39.56% |

Missingness is not treated automatically as random noise.

For laboratory variables such as `A1Cresult` and `max_glu_serum`,
absence may represent that a test was not performed and may itself carry
information about clinical workflow.

Feature-specific preprocessing decisions will therefore be documented
explicitly.

## Data Integrity Checks

The raw-data validation pipeline checks:

- expected row count
- expected feature count
- identifier schema
- target schema
- target labels
- missing identifiers
- duplicate encounter IDs
- SHA-256 hashes

## Dataset Snapshot Hashes

### Identifiers

`a0dd26ab6d8ab54170f7c2a02b07c809a7d38986bbf2bc6ca7d7dfa3a0744b2e`

### Features

`63e2811bc212729887e6b1ab88f96af5e5b33aced8c1790ec61c1efd38990b04`

### Target

`823567cc53b56e91867fe90bee00a1b64ea3712561b383fd870da019c6ca14bd`

## Raw Data Policy

Raw patient-level data are intentionally excluded from version control.

The repository stores only:

- acquisition code
- validation code
- aggregate quality reports
- metadata
- hashes
- reproducibility information

The dataset must be acquired using the provided data-acquisition script.

## Current Limitations

This dataset is historical and observational.

The project must not be interpreted as evidence of prospective clinical
utility.

Performance measured in this repository does not establish transportability
to other hospitals, populations, time periods, or healthcare systems.

## Intended Use

Research, education, reproducible machine-learning experimentation, and
portfolio demonstration.

## Prohibited Interpretation

The resulting models are not clinical decision-support systems and must not
be used for patient-care decisions without appropriate external validation,
prospective evaluation, governance, and regulatory review.