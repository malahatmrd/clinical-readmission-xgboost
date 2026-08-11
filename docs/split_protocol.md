# Train / Validation / Test Split Protocol

## Purpose

The primary clinical cohort is partitioned before preprocessing or model
development.

The split protocol is designed to provide reproducible model development
while preventing information from the held-out test set from influencing
training, model selection, calibration, or threshold selection.

## Source Cohort

The split is generated from the primary clinical cohort.

| Property | Value |
|---|---:|
| Encounters | 69,973 |
| Unique patients | 69,973 |
| Positive 30-day readmissions | 6,277 |
| Positive rate | 8.97% |

Each patient appears exactly once in the primary cohort.

## Split Fractions

The configured split fractions are:

| Split | Fraction |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

The split configuration is stored in:

`configs/data.yaml`

## Stratification

Splitting is stratified using:

`readmitted_30d`

This preserves the low positive-event prevalence across train,
validation, and test partitions.

## Random Seeds

The first-stage split uses:

`random_seed = 42`

The second-stage validation/test split uses:

`random_seed = 43`

Both values are recorded in the split manifest.

## Final Split

| Split | Encounters | Positive | Negative | Positive Rate |
|---|---:|---:|---:|---:|
| Train | 48,981 | 4,394 | 44,587 | 8.971% |
| Validation | 10,496 | 942 | 9,554 | 8.975% |
| Test | 10,496 | 941 | 9,555 | 8.965% |

Total:

- 69,973 encounters
- 6,277 positive outcomes
- 63,696 negative outcomes

## Patient Leakage

The primary cohort contains one encounter per patient.

Therefore:

- train/validation patient overlap = 0
- train/test patient overlap = 0
- validation/test patient overlap = 0

Repeated-patient robustness cohorts will require explicit patient-grouped
splitting in later analyses.

## Test Set Policy

The test set is locked after creation.

The test partition must not be used for:

- preprocessing decisions
- missing-value strategy selection
- feature engineering decisions
- feature selection
- hyperparameter tuning
- model selection
- class-weight selection
- probability-calibration fitting
- decision-threshold selection

The test set is reserved for final evaluation after the full modeling
strategy has been selected using training and validation data.

## Train Set Role

The training partition may be used to fit:

- preprocessing transformations
- imputation parameters
- categorical encoders
- numerical transformations
- logistic-regression models
- XGBoost models
- class weighting
- candidate model parameters

Any data-dependent transformation must be fitted on training data rather
than on the complete cohort.

## Validation Set Role

The validation partition may be used for:

- model comparison
- hyperparameter selection
- early stopping
- class-imbalance strategy comparison
- calibration strategy selection
- operating-threshold selection
- model-development diagnostics

Repeated inspection of validation performance is treated as part of model
development and is therefore separate from final test evaluation.

## Test Set Role

The locked test set is used only after model-development decisions have
been finalized.

Final test evaluation will include appropriate discrimination,
calibration, and clinically relevant classification metrics.

## Reproducibility

The row-level split assignment is generated with:

    python scripts/build_primary_split.py

The assignment file is stored locally at:

`data/processed/splits/primary_split_assignments.csv`

Because it contains patient and encounter identifiers, this file is
excluded from version control.

## Reproducibility Manifest

Aggregate split metadata are stored in:

`artifacts/metrics/primary_split_manifest.json`

The manifest records:

- source cohort SHA-256
- assignment SHA-256
- random seeds
- split fractions
- scikit-learn version
- target column
- patient grouping column
- final split counts
- locked-test policy

Current assignment SHA-256:

`e8201f2f411995f47fb5b525061762dca8cd65831578f12fa9ce2eff05bc9482`

Current source-cohort SHA-256:

`f1de76f15beed99c154e81c078b99b3840b95aae01e18354b570363d8cc8ddd8`

## Determinism Audit

The split builder was executed repeatedly using the same code,
configuration, dataset snapshot, and software environment.

The resulting assignment-file SHA-256 was identical across runs.

This confirms deterministic split reconstruction under the recorded
environment.

## Software Version

The split manifest records:

`scikit-learn 1.9.0`

Software versions are recorded because implementation details may change
between library versions.

## Privacy Policy

Row-level split assignments are not committed to Git.

Only aggregate summaries, reproducibility metadata, and cryptographic
hashes are version controlled.

## Next Modeling Stage

After this split is frozen, the next stage is:

1. define preprocessing policy
2. fit preprocessing using training data only
3. establish a logistic-regression baseline
4. evaluate development performance on validation data
5. proceed to XGBoost development