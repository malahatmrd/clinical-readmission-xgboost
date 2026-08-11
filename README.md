# Clinical Readmission Prediction with XGBoost

Reliable, leakage-aware, and reproducible prediction of 30-day hospital
readmission using real-world clinical encounter data.

## Project Goal

This project develops a research-oriented medical machine-learning
pipeline for predicting hospital readmission within 30 days.

The emphasis is not only on predictive performance, but also on:

- reproducibility
- clinical cohort definition
- leakage prevention
- data-quality auditing
- class-imbalance handling
- probability calibration
- threshold analysis
- explainability
- subgroup evaluation
- robustness analysis
- automated testing

## Dataset

Diabetes 130-US Hospitals for Years 1999-2008 from the UCI Machine
Learning Repository.

Validated raw snapshot:

| Property | Value |
|---|---:|
| Encounters | 101,766 |
| Unique patients | 71,518 |
| Predictive features | 47 |
| Positive `<30` outcomes | 11,357 |
| Raw positive rate | 11.16% |

Detailed dataset documentation:

`docs/data_card.md`

## Target

Positive class:

`readmitted == "<30"`

Negative class:

`readmitted == "NO"` or `readmitted == ">30"`

The modeling target is stored as:

`readmitted_30d`

## Clinical Cohort

The primary cohort is constructed before model development.

Primary cohort:

| Property | Value |
|---|---:|
| Encounters | 69,973 |
| Unique patients | 69,973 |
| Positive readmissions | 6,277 |
| Positive rate | 8.97% |

Construction:

1. start with 101,766 raw encounters
2. retain first observed encounter per patient
3. obtain 71,518 patient-independent encounters
4. exclude terminal and hospice dispositions
5. obtain 69,973 primary encounters

Additional robustness cohorts:

- first eligible encounter per patient: 69,990
- all eligible encounters: 99,343 encounters across 69,990 patients

Detailed methodology:

`docs/cohort_definition.md`

## Leakage Prevention

The UCI dataset exposes `encounter_id` and `patient_nbr` separately from
predictive features.

Neither identifier is allowed as a predictive model input.

For analyses retaining repeated encounters, data splitting must be grouped
by `patient_nbr` so that a patient cannot appear in more than one split.

All preprocessing, feature transformation, model tuning, calibration,
and threshold selection will be learned without using the held-out test set.

## Primary Data Split

The primary cohort is partitioned into stratified train, validation, and
locked test sets.

| Split | Encounters | Positive Rate |
|---|---:|---:|
| Train | 48,981 | 8.971% |
| Validation | 10,496 | 8.975% |
| Test | 10,496 | 8.965% |

The test partition is locked after creation and is not used for
preprocessing decisions, feature selection, hyperparameter tuning,
calibration fitting, or threshold selection.

Build the reproducible split with:

    python scripts/build_primary_split.py

Detailed protocol:

docs/split_protocol.md

## Data Acquisition

Download the dataset with:

    python -m clinical_readmission.data.download

Raw data are stored under:

`data/raw/`

Patient-level raw files are excluded from Git.

## Data Validation

Run:

    python scripts/validate_data.py

Validation includes:

- expected row count
- schema validation
- identifier integrity
- target validation
- duplicate encounter detection
- SHA-256 provenance

## Data Audits

Raw dataset audit:

    python scripts/audit_raw_data.py

Feature-quality audit:

    python scripts/audit_features.py

Discharge-disposition audit:

    python scripts/audit_discharge_dispositions.py

Cohort-rule audit:

    python scripts/audit_cohort_rules.py

Cohort reconciliation audit:

    python scripts/audit_cohort_reconciliation.py

Temporal-consistency audit:

    python scripts/audit_temporal_consistency.py

## Build Cohorts

Run:

    python scripts/build_cohorts.py

Generated patient-level cohort files are stored under:

`data/interim/cohorts/`

These files are excluded from version control.

## Repository Structure

    clinical-readmission-xgboost/
    ├── artifacts/
    │   └── metrics/
    ├── configs/
    ├── data/
    │   ├── raw/
    │   ├── reference/
    │   ├── interim/
    │   └── processed/
    ├── docs/
    ├── reports/
    │   ├── figures/
    │   └── tables/
    ├── scripts/
    ├── src/
    │   └── clinical_readmission/
    ├── tests/
    ├── pyproject.toml
    ├── requirements.txt
    └── requirements-lock.txt

## Environment

The project currently targets Python 3.12.

Create and activate the environment:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

## Testing

Run:

    pytest

Current test suite:

`14 passed`

## Code Quality

Run:

    ruff check .

Current status:

`All checks passed!`

## Project Roadmap

- [x] Phase 0 — reproducible project bootstrap
- [x] Phase 1 — data acquisition, provenance, and quality audit
- [x] Phase 2 — clinical cohort definition and leakage controls
- [x] Phase 3 - leakage-safe train/validation/test splitting
- [x] Phase 4 - preprocessing and logistic-regression baseline
- [ ] Phase 5 — XGBoost training
- [ ] Phase 6 — class-imbalance and hyperparameter optimization
- [ ] Phase 7 — discrimination and calibration analysis
- [ ] Phase 8 — validation-only threshold selection
- [ ] Phase 9 — SHAP explainability
- [ ] Phase 10 — subgroup and robustness analysis
- [ ] Phase 11 — final model card and reproducibility release

## Clinical and Reproducibility Limitations

This dataset is historical and observational.

The repository does not establish prospective clinical utility or
transportability to other hospitals, populations, time periods, or
healthcare systems.

A reproducibility discrepancy was identified between the current UCI
snapshot and values reported for a reference-style cohort. The project
documents this discrepancy rather than altering eligibility rules to
force numerical agreement.

## Disclaimer

This repository is intended for research, education, and portfolio
demonstration.

It is not a clinical decision-support system and must not be used for
patient-care decisions without appropriate external validation,
prospective evaluation, governance, and regulatory review.