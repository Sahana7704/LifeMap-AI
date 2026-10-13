# LifeMap AI Diabetes Module - Model Card

## Model Overview
- **Model Name:** LifeMap AI Two-Tier Calibrated Diabetes Screening System
- **Architecture:** XGBoost (Extreme Gradient Boosting) with Post-Hoc Isotonic Probability Calibration
- **Version:** 1.0.0 (October 2026)
- **Developed by:** LifeMap AI Research & Engineering Team
- **Model Types:**
  - **Tier 1 (Non-Invasive):** Calibrated XGBoost using 10 survey and physical examination features.
  - **Tier 2 (Biomarker-Augmented):** Calibrated XGBoost incorporating 10 tabular features plus Glycated Hemoglobin (HbA1c, %).

---

## Intended Use & Clinical Scope
- **Intended Use:** Population-level preventive screening and non-diagnostic risk stratification for adult type 2 diabetes. The model serves as an educational and preventive triage tool to guide follow-up confirmatory laboratory evaluations.
- **Target Population:** Non-institutionalized US adults aged 18 to 80 years.
- **Out-of-Scope Uses:**
  - Definitive clinical diagnosis of diabetes mellitus.
  - Triage in acute diabetic emergencies (e.g., Diabetic Ketoacidosis, Hyperosmolar Hyperglycemic State).
  - Insulin dosing or pharmacological treatment planning.
  - Pediatric screening (<18 years of age).

---

## Dynamic Tier Routing Logic
At prediction runtime (`/predict` endpoint):
- **Tier 2 Trigger:** If `hba1c` is provided and falls within the biologically plausible physiological range ($3.0\% \le \text{HbA1c} \le 18.0\%$), the payload is routed to the Tier 2 model (`tier2_final_calibrated_xgb.joblib`).
- **Tier 1 Fallback:** If `hba1c` is null or omitted, the payload is routed to Tier 1 (`final_calibrated_xgb_full.joblib`).
- **Input Validation:** Values outside $[3.0\%, 18.0\%]$ return a `422 Unprocessable Entity` error alerting the user to recheck the clinical report.

---

## Training & Validation Cohorts
- **Development Dataset:** National Health and Nutrition Examination Survey (NHANES) pooled cycles 2013-2018 (H, I, J).
  - **Total Cohort Size:** 17,468 adult respondents.
  - **Prevalence:** 2,449 diabetic (14.02%), 15,019 non-diabetic (85.98%).
  - **Class Imbalance Ratio:** 1:6.13.
- **Split Partitions (Seed 42):**
  - **Training Set (70%):** 12,227 participants (1,714 diabetic, 10,513 non-diabetic).
  - **Validation Set (15%):** 2,620 participants (367 diabetic, 2,253 non-diabetic) — used strictly for hyperparameter tuning, probability calibrator fitting, and dual-operating-point threshold selection.
  - **Test Set (15%):** 2,621 participants (368 diabetic, 2,253 non-diabetic) — held out untouched.
- **External Validation:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 cohort:
  - 249,049 adult participants evaluated zero-shot with shared-7 features without retraining or recalibration.

---

## Model Features
1. `age`: Participant chronological age (18.0 - 80.0 years)
2. `sex`: Biological sex (1 = Male, 0 = Female)
3. `bmi`: Body Mass Index (12.0 - 67.3 kg/m²)
4. `mean_sbp`: Mean systolic blood pressure from examination (mmHg)
5. `mean_dbp`: Mean diastolic blood pressure from examination (mmHg)
6. `high_bp`: Self-reported diagnosis of hypertension (1 = Yes, 0 = No)
7. `high_chol`: Self-reported diagnosis of hypercholesterolemia (1 = Yes, 0 = No)
8. `smoker`: Cigarette smoking history (1 = >=100 lifetime cigarettes, 0 = No)
9. `phys_activity`: Moderate/vigorous weekly physical activity (1 = Active, 0 = Inactive)
10. `family_history`: First-degree family history of diabetes (1 = Yes, 0 = No)
11. `hba1c` *(Tier 2 only)*: Automated laboratory glycated hemoglobin assay (%)

---

## Empirical Performance Metrics (Held-Out Test Set)

### Tier 1 (Non-Invasive, n=2,621)
Operating at Validation-Selected Screening Operating Point ($\theta = 0.144$):
- **Accuracy:** 0.763 [95% CI: 0.747 - 0.778]
- **Precision:** 0.353 [95% CI: 0.320 - 0.385]
- **Recall (Sensitivity):** 0.823 [95% CI: 0.782 - 0.860]
- **Specificity:** 0.753 [95% CI: 0.734 - 0.771]
- **F1 Score:** 0.494 [95% CI: 0.458 - 0.529]
- **ROC-AUC:** 0.856 [95% CI: 0.835 - 0.876]
- **PR-AUC:** 0.457 [95% CI: 0.412 - 0.505]
- **Brier Score:** 0.092 (Raw XGB: 0.155)
- **Expected Calibration Error (ECE):** 0.0163

Operating at High-Accuracy Operating Point ($\theta = 0.402$):
- **Accuracy:** 0.864 [95% CI: 0.850 - 0.877]
- **Precision:** 0.521 [95% CI: 0.467 - 0.579]
- **Recall:** 0.443 [95% CI: 0.391 - 0.495]
- **Specificity:** 0.933 [95% CI: 0.922 - 0.943]
- **F1 Score:** 0.479 [95% CI: 0.431 - 0.526]

### Tier 2 (Biomarker-Augmented, n=2,392)
Operating at Validation-Selected Screening Operating Point ($\theta = 0.290$):
- **Accuracy:** 0.913 [95% CI: 0.902 - 0.924]
- **Precision:** 0.649 [95% CI: 0.605 - 0.693]
- **Recall (Sensitivity):** 0.854 [95% CI: 0.816 - 0.893]
- **Specificity:** 0.923 [95% CI: 0.911 - 0.934]
- **F1 Score:** 0.737 [95% CI: 0.701 - 0.772]
- **ROC-AUC:** 0.946 [95% CI: 0.933 - 0.958]
- **PR-AUC:** 0.801 [95% CI: 0.763 - 0.837]
- **Brier Score:** 0.050 (Raw XGB: 0.077)
- **Expected Calibration Error (ECE):** 0.0118

---

## Subgroup Evaluation & Fairness (Tier 1 Screening Mode)
Evaluated on $n=2,621$ test cases:
- **Sex:**
  - *Male* ($N=1,281, n_{diab}=187$): ROC-AUC = 0.849, Recall = 0.824, Specificity = 0.748 [Reliable]
  - *Female* ($N=1,340, n_{diab}=181$): ROC-AUC = 0.862, Recall = 0.823, Specificity = 0.758 [Reliable]
- **Age Groups:**
  - *18-39 Years* ($N=882, n_{diab}=25$): ROC-AUC = 0.834, Recall = 0.800, Specificity = 0.884 [Flagged: <30 cases, exploratory]
  - *40-59 Years* ($N=858, n_{diab}=124$): ROC-AUC = 0.821, Recall = 0.815, Specificity = 0.722 [Reliable]
  - *60+ Years* ($N=881, n_{diab}=219$): ROC-AUC = 0.781, Recall = 0.831, Specificity = 0.613 [Reliable]
- **BMI Categories:**
  - *<25 kg/m²* ($N=709, n_{diab}=35$): ROC-AUC = 0.832, Recall = 0.800, Specificity = 0.846 [Reliable]
  - *25-29.9 kg/m²* ($N=840, n_{diab}=98$): ROC-AUC = 0.838, Recall = 0.816, Specificity = 0.763 [Reliable]
  - *>=30 kg/m²* ($N=1,072, n_{diab}=235$): ROC-AUC = 0.814, Recall = 0.830, Specificity = 0.671 [Reliable]

---

## Key Limitations & Safety Notices
1. **Self-Reported Outcome Bias:** Ground-truth diabetes status in NHANES is defined by self-reported physician diagnosis (`DIQ010`), not active fasting blood glucose challenge. Undiagnosed diabetic cases may introduce false negative label noise in the raw data.
2. **Missing Biomarker Rate:** 8.77% of NHANES adult respondents lacked HbA1c lab samples due to non-attendance at the Mobile Examination Center.
3. **Clinical Confirmation Requirement:** All predictions generated by the system include the explicit clinician escalation notice:
   > *"Suggestive of diabetes range; confirm with a clinician."*
