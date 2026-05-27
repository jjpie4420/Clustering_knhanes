# Working Title

Comparative Predictive Utility of FFQ-Derived Habitual Dietary Composition and 24-Hour Recall-Derived Intake for Metabolic Phenotypes in Korean Adults: A Machine Learning Analysis of KNHANES 2012–2016

---

# Core Hypothesis

FFQ-derived habitual dietary composition provides more stable and informative predictive signals for chronic metabolic phenotypes than single-day 24-hour recall-derived intake.

---

# Study Design

- Data source: KNHANES 2012–2016
- Population: Korean adults aged 20–64 years
- Exclusion:
  - pregnancy
  - extreme energy intake
  - missing FFQ or recall data
  - missing BMI
- Final analytic sample: n = 16,082

---

# Outcomes

## Main metabolic phenotypes

1. Diabetes
2. Hypertension
3. Dyslipidemia
4. Obesity

## Sensitivity / clean phenotypes

1. Untreated diabetes
2. Untreated hypertension
3. Untreated dyslipidemia
4. Clean prediabetes
5. Clean prehypertension

---

# Dietary Exposure Sets

## FFQ-derived variables

- Energy-adjusted nutrient composition using residual method
- Habitual dietary exposure

## 24-hour recall-derived variables

- Energy-adjusted nutrient composition using residual method
- Single-day intake exposure

---

# Machine Learning Models

## Main model

- Logistic regression with L2 regularization
- 5-fold stratified cross-validation
- Class-weight balancing

## Secondary model

- Random forest

---

# Main Comparisons

## Diet-only prediction

- FFQ-adjusted nutrients
- Recall-adjusted nutrients
- FFQ + recall combined

## Clinical + diet prediction

- Clinical baseline
- Clinical + FFQ
- Clinical + recall
- Clinical + FFQ + recall

---

# Primary Evaluation Metrics

- AUROC
- AUPRC
- Balanced accuracy
- F1 score

---

# Key Results

## 1. FFQ and recall showed moderate agreement

FFQ and recall-derived nutrient variables showed moderate correlations, suggesting that the two dietary assessment methods capture overlapping but non-identical dietary signals.

## 2. FFQ consistently outperformed recall

Energy-adjusted FFQ-derived dietary composition outperformed recall-derived intake across all metabolic phenotypes.

AUROC difference range:

- Diabetes: +0.071
- Untreated diabetes: +0.065
- Prediabetes: +0.047
- Hypertension: +0.056
- Untreated hypertension: +0.040
- Prehypertension: +0.050
- Dyslipidemia: +0.047
- Untreated dyslipidemia: +0.048
- Obesity: +0.052

## 3. Bootstrap confidence intervals supported robustness

All bootstrap 95% confidence intervals for AUROC difference were above zero.

## 4. Combined FFQ + recall models provided limited additional gain

Adding recall-derived variables to FFQ-derived variables produced minimal improvement, suggesting limited independent predictive information from single-day recall beyond habitual intake.

## 5. FFQ-derived feature importance was more stable

Energy-adjusted FFQ-derived nutrients showed stronger and more stable coefficient patterns across cross-validation folds than recall-derived nutrients.

## 6. Subgroup analyses supported consistency

FFQ superiority was observed across:

- sex groups
- age groups
- BMI categories

## 7. Sensitivity analyses supported robustness

FFQ superiority remained after:

- stricter energy filtering
- removing diagnosed participants
- removing medication users
- balanced case-control sampling

---

# Proposed Figures

## Figure 1

FFQ vs recall nutrient agreement

- correlation summary
- Bland-Altman plots for energy, fat, sodium

## Figure 2

Energy-adjusted AUROC comparison

- FFQ vs recall across outcomes

## Figure 3

Bootstrap AUROC difference

- ΔAUROC with 95% CI

## Figure 4

Feature importance heatmap

- energy-adjusted FFQ coefficients across outcomes

## Figure 5

Subgroup robustness

- sex, age, BMI subgroup ΔAUROC

---

# Proposed Tables

## Table 1

Baseline characteristics of analytic cohort

## Table 2

Prediction performance of FFQ vs recall

## Table 3

Bootstrap AUROC and AUPRC differences

## Table 4

Sensitivity analysis results

## Supplementary Table 1

Feature importance ranking

## Supplementary Table 2

Subgroup analysis

---

# Discussion Points

## Main interpretation

FFQ-derived habitual dietary composition may better approximate long-term dietary exposure relevant to chronic metabolic phenotypes than single-day recall-derived intake.

## Precision nutrition implication

For chronic metabolic risk stratification, habitual dietary assessment may provide more stable and useful signals than single-day dietary recall alone.

## Why recall performed worse

- high within-person variability
- day-specific intake fluctuation
- single-day recall limitation
- episodic food consumption
- higher measurement noise

## Why FFQ performed better

- captures habitual intake
- better aligned with chronic disease biology
- lower day-to-day variability
- more stable feature contribution

---

# Limitations

1. Cross-sectional design prevents causal inference.
2. Dietary data were self-reported.
3. Only single-day 24-hour recall was available.
4. Residual confounding remains possible.
5. Survey weights were not incorporated into the primary ML framework.
6. FFQ and recall may differ in measurement error structure.
7. External validation was not performed.

---

# SCI-Level Additional Analyses Needed

## Essential

1. DeLong test or bootstrap test for AUROC difference  
   - already done with bootstrap CI

2. Energy-adjusted nutrient analysis  
   - already done

3. Sensitivity analysis excluding medication/diagnosed participants  
   - already done

4. Subgroup robustness  
   - already done

## Strongly recommended

1. Survey-weighted descriptive Table 1
2. Clinical + diet incremental value summary
3. Net reclassification improvement or decision curve analysis
4. Calibration analysis
5. External or temporal validation

---

# Recommended Next Analysis

## 20_calibration_analysis.py

Purpose:

- compare calibration of FFQ vs recall models
- generate calibration curves
- calculate Brier score

Rationale:

AUROC alone measures discrimination, but precision nutrition applications also require reliable risk probability estimation.

---

# Candidate Abstract Result Sentence

Energy-adjusted FFQ-derived dietary composition significantly outperformed single-day recall-derived intake in predicting metabolic phenotypes, with AUROC improvements ranging from 0.040 to 0.071 across cardiometabolic outcomes. This superiority remained consistent across subgroup and sensitivity analyses.

---

# Candidate Conclusion

In Korean adults, FFQ-derived habitual dietary composition provided more stable and informative predictive signals for chronic metabolic phenotypes than single-day 24-hour recall-derived intake. These findings suggest that habitual dietary assessment remains highly relevant for precision nutrition models targeting chronic metabolic risk.