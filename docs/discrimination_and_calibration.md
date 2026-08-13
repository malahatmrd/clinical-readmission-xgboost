# Discrimination and Calibration Analysis

## Purpose

Phase 7 evaluates the discrimination, uncertainty, and probability
calibration of the development finalists before any locked-test
evaluation.

The test partition remains completely unused.

## Validation Finalists

Three primary development models were retained for detailed analysis:

- Logistic Regression
- Early-Stopped XGBoost
- Tuned XGBoost

The tuned XGBoost was selected in Phase 6 using Average Precision as
the prespecified primary development metric.

## Reproducibility Audit

Validation probabilities were regenerated from the recorded model
configurations.

All three finalist models reproduced their previously recorded
ROC-AUC, Average Precision, Brier Score, and Log Loss within numerical
tolerance.

The regenerated validation prediction artifact contains no patient or
encounter identifiers and is protected by a recorded SHA256 digest.

## Bootstrap Uncertainty

A stratified paired bootstrap with 2,000 resamples was applied to the
validation predictions.

The same bootstrap resamples were used across models to support paired
comparisons.

The tuned XGBoost achieved a higher point estimate for both ROC-AUC and
Average Precision than the other finalists.

Compared with Logistic Regression, the tuned XGBoost showed an Average
Precision improvement whose 95% paired bootstrap confidence interval
excluded zero.

Compared with the early-stopped XGBoost, however, the confidence
intervals for the ROC-AUC and Average Precision differences included
zero.

The uncalibrated tuned XGBoost also showed consistently worse Brier
Score and Log Loss than the simpler finalists.

## Calibration Diagnostics

The observed validation prevalence was approximately 0.08975.

The uncalibrated tuned XGBoost produced a mean predicted probability of
approximately 0.12710, indicating systematic risk overestimation.

Its quantile expected calibration error was approximately 0.03736.

This motivated evaluation of post-hoc calibration.

## Train-Only Post-Hoc Calibration

Calibration was performed without using the validation or test
partitions for fitting.

Five-fold stratified cross-validation within the original Train
partition was used to generate out-of-fold predictions for calibration.

Two methods were evaluated:

- Sigmoid calibration
- Isotonic calibration

Both methods substantially improved the probability-quality metrics of
the tuned XGBoost.

## Calibration Selection

Sigmoid calibration was selected for the tuned XGBoost.

Compared with the uncalibrated tuned model, sigmoid calibration:

- preserved ROC-AUC exactly
- preserved Average Precision exactly
- improved Brier Score
- improved Log Loss
- corrected the substantial mean-risk overestimation

The paired bootstrap confidence intervals for both the Brier Score and
Log Loss improvements excluded zero.

Isotonic calibration produced slightly lower point estimates for Brier
Score and Log Loss than sigmoid calibration, but the paired confidence
intervals for those differences included zero.

In contrast, isotonic calibration caused a reproducible reduction in
Average Precision relative to sigmoid calibration.

Therefore, the small and uncertain probability-quality advantage of
isotonic calibration did not justify its loss of discrimination.

## Selected Phase 7 Variant

The frozen development variant is:

**Tuned XGBoost with sigmoid calibration**

Validation performance:

| Metric | Estimate |
|---|---:|
| ROC-AUC | 0.650820 |
| Average Precision | 0.171639 |
| Brier Score | 0.079418 |
| Log Loss | 0.289742 |
| Calibration Intercept | -0.118232 |
| Calibration Slope | 0.948403 |
| Quantile ECE | 0.008876 |

The observed validation prevalence was approximately 0.089748 and the
mean predicted probability after sigmoid calibration was approximately
0.090009.

## Validation Confidence Intervals

For the selected tuned XGBoost with sigmoid calibration:

| Metric | Estimate | 95% Bootstrap CI |
|---|---:|---:|
| ROC-AUC | 0.650820 | 0.631752–0.669115 |
| Average Precision | 0.171639 | 0.156007–0.190405 |
| Brier Score | 0.079418 | 0.078523–0.080337 |
| Log Loss | 0.289742 | 0.286550–0.293015 |

## Interpretation

The model demonstrates moderate discrimination rather than extremely
high predictive performance.

The main methodological result of Phase 7 is that improved ranking from
the tuned XGBoost initially came with systematic probability
overestimation.

Train-only sigmoid calibration corrected this problem without changing
ROC-AUC or Average Precision.

This separation between discrimination and calibration is important for
clinical risk prediction because rank ordering alone does not guarantee
accurate absolute risk estimates.

## Leakage Control

The locked test partition has not been used for:

- finalist model selection
- bootstrap model comparison
- calibration fitting
- calibration-method selection
- threshold selection
- figure construction

The selected model family and calibration method are frozen before
locked-test evaluation.

## Figures

Phase 7 produces:

- Validation ROC curve
- Validation Precision-Recall curve
- Validation calibration curve

Each figure is exported as both a 300-DPI PNG and a vector SVG.

## Next Stage

The next phase will define clinically meaningful operating thresholds
using development data only.

The locked test set will remain untouched until all model, calibration,
and threshold decisions have been finalized.