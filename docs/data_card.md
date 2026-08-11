## Modeling Cohorts

The raw dataset is not used directly for primary model training.

The primary cohort is constructed by retaining the first observed
encounter for each patient and then excluding terminal or hospice
dispositions.

Primary cohort:

- 69,973 encounters
- 69,973 unique patients
- 6,277 positive 30-day readmissions
- 8.97% positive rate

Sensitivity cohort:

- 69,990 encounters
- 69,990 unique patients
- 6,285 positive 30-day readmissions

All-eligible robustness cohort:

- 99,343 encounters
- 69,990 unique patients
- 11,314 positive 30-day readmissions

See `docs/cohort_definition.md` for the complete eligibility,
reproducibility, and leakage-control rationale.