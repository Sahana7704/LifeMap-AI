# LifeMap AI: Diabetes Risk Module Comprehensive Technical Report

**Project**: LifeMap AI (Preventive Healthcare Capstone)  
**Module**: Non-Invasive Explainable Diabetes Risk Stratification  
**Author**: Antigravity Autonomous Agent  
**Date**: October 2026  

---

## 1. Executive Summary & Clinical Rationale

Type 2 Diabetes Mellitus (T2DM) frequently develops over a prolonged asymptomatic prediabetic phase lasting 5 to 10 years. Conventional diagnostic procedures rely heavily on invasive biochemical markers such as Glycated Hemoglobin (HbA1c) and Fasting Plasma Glucose (FPG). While diagnostic laboratory tests are essential for clinical confirmation, they are inaccessible for routine, low-cost population-wide screening in underserved or remote populations.

This module delivers a **hybrid, explainable, non-invasive risk prediction system** engineered strictly from non-laboratory demographic, physiological, and behavioral survey indicators. By combining calibrated gradient-boosted decision trees, out-of-fold stacking, native game-theoretic TreeSHAP feature attributions, and American Diabetes Association (ADA) rule-based lab escalation, this module bridges the gap between machine learning screening and actionable preventive care.

---

## 2. Cohorts & Data Architecture

### 2.1 Primary Training & Testing Cohort: CDC NHANES (2013–2018)
- **Source**: National Health and Nutrition Examination Survey (NHANES), Cycles H (2013-2014), I (2015-2016), and J (2017-2018).
- **Target Population**: Non-institutionalized US civilian adults aged 18 years and older ($N = 17,468$).
- **Label Definition**: Ground-truth diabetes status derived from Questionnaire file `DIQ010`:
  - Positive ($Y = 1$): Diagnosed diabetes (`DIQ010 == 1`).
  - Negative ($Y = 0$): No diabetes (`DIQ010 == 2`).
  - Exclusions: Borderline / prediabetes (`DIQ010 == 3`), refused (`7`), don't know (`9`), and missing values were dropped to ensure label clarity.
  - Overall NHANES diabetes prevalence: **14.02%** ($n = 2,449 / 17,468$).
- **Feature Set (10 Non-Lab Features)**:
  1. `age`: Chronological age in years (18–80+).
  2. `age_bucket`: Categorical age grouped into 13 standardized BRFSS intervals (18-24 up to 80+).
  3. `sex`: Biological sex (1 = Male, 0 = Female).
  4. `bmi`: Body Mass Index ($\text{kg/m}^2$, examination data).
  5. `mean_sbp`: Mean systolic blood pressure across up to 3 valid oscillatory readings.
  6. `mean_dbp`: Mean diastolic blood pressure across up to 3 valid oscillatory readings.
  7. `high_bp`: Ever told had high blood pressure (`BPQ020 == 1`).
  8. `high_chol`: Ever told had high cholesterol (`BPQ080 == 1`).
  9. `smoker`: Lifetime smoking history of $\ge 100$ cigarettes (`SMQ020 == 1`).
  10. `phys_activity`: Weekly moderate-to-vigorous physical activity (`PAQ605/620/635/650/665`).
  11. `family_history`: Biological family history of diabetes (`MCQ300c == 1`).

### 2.2 External Validation Cohort: CDC BRFSS 2015
- **Source**: Behavioral Risk Factor Surveillance System (BRFSS 2015).
- **Sample Size**: $N = 249,049$ adults.
- **Diabetes Prevalence**: **14.19%** ($n = 35,346 / 249,049$).
- **Harmonized Shared-7 Feature Subspace**:
  - `age_bucket`, `sex`, `bmi`, `high_bp`, `high_chol`, `smoker`, `phys_activity`.

---

## 3. Leakage Prevention & Methodological Rigor

To guarantee honest, generalizable findings and avoid common pitfalls of healthcare ML:
1. **Zero Lab Predictors**: Blood glucose, HbA1c, oral glucose tolerance, and insulin levels were strictly excluded from model feature sets.
2. **Stratified Pre-Split Partitioning**: The NHANES dataset was partitioned into 70% Train ($n=12,227$), 15% Validation ($n=2,620$), and 15% Test ($n=2,621$) prior to any statistical imputation, scaling, or transformation.
3. **Imputer & Scaler Fitting**: `SimpleImputer` (median for continuous, mode for binary) and `StandardScaler` were fit strictly on the training split and applied unchanged to validation, test, and external sets.
4. **Resampling Isolation**: Synthetic Minority Over-sampling Technique (SMOTE) was applied strictly within training folds, never on validation or test sets.
5. **Threshold & Calibration Integrity**: Probability calibration (Isotonic Regression) and the classification decision threshold ($0.140$) were optimized strictly on the validation set.
6. **Leakage Audit**: All models were systematically evaluated against an automated threshold ($AUC \le 0.90$). No models breached this limit (best full test AUC was 0.859), confirming realistic clinical discrimination without data contamination.

---

## 4. Model Evaluation & Benchmark Results

All metrics on the held-out NHANES test set ($n = 2,621$) and BRFSS external validation set ($n = 249,049$) were computed alongside empirical **95% bootstrap confidence intervals** across 1,000 resamples.

### 4.1 NHANES Held-Out Test Set (Full 10 Features)

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline (All 0)** | 0.860 [0.847, 0.873] | 0.500 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 0.500 [0.468, 0.534] | 0.140 [0.127, 0.153] | 0.140 |
| **Logistic Regression** | 0.754 [0.738, 0.771] | 0.783 | 0.344 | 0.823 [0.788, 0.861] | 0.485 | 0.858 [0.838, 0.876] | 0.477 [0.430, 0.532] | 0.162 |
| **Random Forest** | 0.811 [0.797, 0.825] | 0.771 | 0.403 | 0.715 [0.668, 0.759] | 0.515 | 0.853 [0.833, 0.870] | 0.459 [0.411, 0.513] | 0.126 |
| **XGBoost (Default)** | 0.762 [0.746, 0.778] | 0.789 | 0.352 | 0.826 [0.786, 0.864] | 0.494 | 0.858 [0.838, 0.875] | 0.480 [0.432, 0.534] | 0.155 |
| **Linear SVM (Calibrated)** | 0.865 [0.852, 0.878] | 0.584 | 0.555 | 0.193 [0.156, 0.232] | 0.286 | 0.858 [0.838, 0.876] | 0.477 [0.428, 0.531] | 0.092 |
| **PyTorch MLP** | 0.739 [0.723, 0.755] | 0.786 | 0.332 | 0.851 [0.815, 0.886] | 0.478 | 0.857 [0.838, 0.876] | 0.489 [0.439, 0.544] | 0.166 |
| **TabNet** | 0.861 [0.848, 0.874] | 0.532 | 0.540 | 0.073 [0.048, 0.101] | 0.129 | 0.849 [0.830, 0.868] | 0.458 [0.410, 0.510] | 0.094 |
| **Stacking Ensemble** | 0.868 [0.855, 0.881] | 0.582 | 0.591 | 0.185 [0.149, 0.224] | 0.282 | 0.859 [0.840, 0.877] | 0.486 [0.436, 0.543] | 0.091 |
| **XGBoost (SMOTE)** | 0.799 [0.783, 0.814] | 0.767 | 0.386 | 0.723 [0.674, 0.765] | 0.503 | 0.848 [0.828, 0.866] | 0.453 [0.407, 0.505] | 0.132 |
| **Final Calibrated XGBoost** | 0.761 [0.744, 0.778] | 0.788 | 0.351 | 0.826 [0.786, 0.864] | 0.493 | 0.856 [0.836, 0.874] | 0.457 [0.412, 0.505] | **0.092** |

> [!NOTE]
> **Statistical Significance & Model Parity**: Across the primary non-lab models (Logistic Regression, Random Forest, XGBoost, PyTorch MLP, and Stacking Ensemble), 95% bootstrap confidence intervals overlap substantially (e.g., ROC-AUC spans 0.838–0.877). The 5-fold stacked hybrid ensemble matches but does not statistically or significantly outperform tuned XGBoost. Given this parity, calibrated XGBoost is selected as the primary deployable model due to its deterministic inference latency (<5ms) and direct TreeSHAP integration.

---

### 4.2 NHANES Held-Out Test Set (Shared 7 Features)

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Shared)** | 0.735 [0.718, 0.752] | 0.766 | 0.323 | 0.810 [0.767, 0.849] | 0.462 | 0.830 [0.809, 0.849] | 0.407 [0.359, 0.461] | 0.177 |
| **Random Forest (Shared)** | 0.722 [0.705, 0.739] | 0.761 | 0.312 | 0.815 [0.772, 0.853] | 0.452 | 0.829 [0.808, 0.848] | 0.412 [0.364, 0.468] | 0.169 |
| **XGBoost (Shared)** | 0.707 [0.689, 0.723] | 0.769 | 0.306 | 0.856 [0.819, 0.890] | 0.450 | 0.838 [0.816, 0.857] | 0.447 [0.395, 0.503] | 0.174 |
| **SVM (Linear, Shared)** | 0.860 [0.846, 0.873] | 0.501 | 1.000 | 0.003 [0.000, 0.009] | 0.005 | 0.829 [0.808, 0.848] | 0.382 [0.341, 0.427] | 0.099 |
| **MLP (PyTorch, Shared)** | 0.699 [0.682, 0.716] | 0.771 | 0.302 | 0.870 [0.833, 0.901] | 0.448 | 0.839 [0.818, 0.857] | 0.430 [0.381, 0.487] | 0.181 |
| **TabNet (Shared)** | 0.865 [0.852, 0.877] | 0.539 | 0.627 | 0.087 [0.060, 0.118] | 0.153 | 0.821 [0.799, 0.841] | 0.418 [0.368, 0.469] | 0.099 |
| **Stacking Ensemble (Shared)** | 0.864 [0.850, 0.877] | 0.522 | 0.773 | 0.046 [0.026, 0.069] | 0.087 | 0.838 [0.817, 0.856] | 0.434 [0.381, 0.489] | 0.097 |
| **Final Calibrated XGB (Shared)** | 0.675 [0.657, 0.693] | 0.763 | 0.287 | 0.886 [0.852, 0.917] | 0.433 | 0.838 [0.816, 0.856] | 0.425 [0.377, 0.475] | **0.096** |

---

### 4.3 CDC BRFSS 2015 External Validation ($N = 249,049$)

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Shared)** | 0.630 [0.623, 0.637] | 0.714 | 0.254 | 0.832 [0.819, 0.846] | 0.390 | 0.784 [0.776, 0.793] | 0.347 [0.330, 0.366] | 0.231 |
| **Random Forest (Shared)** | 0.632 [0.625, 0.638] | 0.711 | 0.254 | 0.821 [0.808, 0.835] | 0.388 | 0.784 [0.776, 0.792] | 0.362 [0.344, 0.380] | 0.218 |
| **XGBoost (Shared)** | 0.599 [0.593, 0.606] | 0.709 | 0.243 | 0.861 [0.849, 0.873] | 0.379 | 0.790 [0.782, 0.798] | 0.374 [0.355, 0.394] | 0.231 |
| **SVM (Linear, Shared)** | 0.858 [0.853, 0.863] | 0.502 | 0.390 | 0.004 [0.002, 0.007] | 0.008 | 0.778 [0.769, 0.786] | 0.317 [0.302, 0.331] | 0.109 |
| **MLP (PyTorch, Shared)** | 0.587 [0.580, 0.594] | 0.707 | 0.239 | 0.876 [0.864, 0.887] | 0.376 | 0.792 [0.784, 0.800] | 0.373 [0.356, 0.393] | 0.239 |
| **TabNet (Shared)** | 0.858 [0.853, 0.863] | 0.536 | 0.503 | 0.087 [0.076, 0.098] | 0.148 | 0.780 [0.772, 0.789] | 0.353 [0.336, 0.371] | 0.109 |
| **Stacking Ensemble (Shared)** | 0.860 [0.855, 0.865] | 0.520 | 0.587 | 0.045 [0.037, 0.052] | 0.084 | 0.792 [0.784, 0.800] | 0.373 [0.355, 0.393] | 0.106 |
| **Final Calibrated XGB (Shared)** | 0.546 [0.539, 0.553] | 0.694 | 0.225 | 0.900 [0.889, 0.911] | 0.360 | 0.788 [0.780, 0.796] | 0.355 [0.339, 0.373] | **0.106** |

---

### 4.4 Common Operating Point Comparison (Validation Recall = 80%)

To ensure fair clinical comparison across heterogeneous probability scales, a common operating point was established: decision thresholds were optimized on the validation partition to target **80% sensitivity (recall)**, and fixed thresholds were then evaluated on the NHANES held-out test partition and BRFSS external cohort.

| Model | NHANES $\tau_{80}$ | NHANES Precision | NHANES Specificity | NHANES Recall | NHANES F1 | BRFSS $\tau_{80}$ | BRFSS Precision | BRFSS Specificity | BRFSS Recall | BRFSS F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.473 | 0.332 [0.302, 0.364] | 0.725 [0.706, 0.744] | 0.829 [0.788, 0.864] | 0.475 [0.443, 0.510] | 0.516 | 0.241 [0.232, 0.249] | 0.548 [0.541, 0.555] | 0.801 [0.785, 0.816] | 0.370 [0.360, 0.380] |
| **Random Forest** | 0.352 | 0.333 [0.303, 0.363] | 0.728 [0.710, 0.746] | 0.826 [0.788, 0.864] | 0.475 [0.442, 0.508] | 0.521 | 0.236 [0.228, 0.245] | 0.532 [0.524, 0.539] | 0.803 [0.788, 0.817] | 0.365 [0.355, 0.375] |
| **XGBoost** | 0.499 | 0.351 [0.318, 0.383] | 0.751 [0.732, 0.768] | 0.826 [0.788, 0.864] | 0.493 [0.456, 0.528] | 0.570 | 0.239 [0.230, 0.248] | 0.541 [0.533, 0.548] | 0.804 [0.789, 0.818] | 0.368 [0.358, 0.379] |
| **Linear SVM** | 0.155 | 0.375 [0.341, 0.408] | 0.783 [0.765, 0.799] | 0.796 [0.753, 0.837] | 0.510 [0.474, 0.544] | 0.129 | 0.243 [0.235, 0.252] | 0.558 [0.551, 0.565] | 0.796 [0.781, 0.811] | 0.373 [0.363, 0.383] |
| **PyTorch MLP** | 0.527 | 0.349 [0.317, 0.380] | 0.742 [0.724, 0.761] | 0.842 [0.804, 0.878] | 0.495 [0.459, 0.527] | 0.609 | 0.244 [0.236, 0.253] | 0.558 [0.550, 0.565] | 0.798 [0.783, 0.812] | 0.374 [0.364, 0.384] |
| **TabNet** | 0.159 | 0.328 [0.297, 0.360] | 0.728 [0.710, 0.747] | 0.818 [0.777, 0.856] | 0.468 [0.434, 0.501] | 0.141 | 0.226 [0.218, 0.234] | 0.500 [0.492, 0.507] | 0.812 [0.797, 0.826] | 0.354 [0.344, 0.363] |
| **Stacking Ensemble** | 0.138 | 0.344 [0.311, 0.375] | 0.742 [0.723, 0.760] | 0.826 [0.788, 0.864] | 0.485 [0.449, 0.520] | 0.136 | 0.242 [0.234, 0.251] | 0.550 [0.542, 0.557] | 0.803 [0.788, 0.817] | 0.372 [0.362, 0.382] |
| **Final Calibrated XGBoost** | 0.144 | 0.353 [0.320, 0.385] | 0.753 [0.734, 0.771] | 0.823 [0.785, 0.861] | 0.494 [0.458, 0.529] | 0.168 | 0.238 [0.229, 0.246] | 0.536 [0.529, 0.544] | 0.806 [0.791, 0.820] | 0.367 [0.357, 0.377] |

---

### 4.5 Dual Operating-Point Analysis: High-Accuracy Mode vs. Screening Mode

In preventive healthcare screening with an underlying disease prevalence of ~14%, raw classification accuracy is fundamentally misleading. A trivial zero-information rule assigning "no diabetes" to every patient achieves an ostensibly high 86.0% accuracy while failing to detect a single diabetic individual (0% sensitivity, 0% recall). When models are tuned to **High-Accuracy Mode** on the validation partition ($\tau \approx 0.40–0.48$ to maximize percentage correct), they achieve 85.8%–87.1% accuracy primarily by predicting the negative majority class. However, clinical sensitivity collapses to 8.4%–35.6% on test sets and BRFSS, missing between 64% and 92% of all diabetic cases (e.g., leaving over 29,200 to 32,300 cases undetected on the nationwide BRFSS cohort). 

In contrast, **Screening Mode** intentionally calibrates the operating threshold lower on the validation set ($\tau \approx 0.13–0.17$) to prioritize clinical sensitivity ($\ge 80\%$ validation recall). Under Screening Mode, models successfully capture 82.3%–86.7% of true positive cases on NHANES and 86.7%–87.4% on BRFSS (identifying over 30,800 cases and reducing missed cases to under 4,700). While this sensitivity prioritization lowers overall accuracy to 58.4%–76.3% due to higher false-positive referrals for confirmatory blood draws, Balanced Accuracy increases substantially from 0.537–0.655 up to 0.705–0.789. For population health screening, a false positive results in an inexpensive, non-harmful follow-up A1c test, whereas a false negative allows progressive, irreversible microvascular damage to proceed unmanaged.

| Cohort | Model | Operating Point | $\tau$ | Accuracy (95% CI) | Balanced Acc (95% CI) | Precision (95% CI) | Recall (95% CI) | Specificity (95% CI) | F1 Score (95% CI) | TP (95% CI) | FP (95% CI) | TN (95% CI) | FN (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NHANES Full** | Calibrated XGBoost | High-Accuracy Mode | 0.402 | 0.869 [0.855–0.881] | 0.619 [0.597–0.641] | 0.568 [0.494–0.637] | 0.272 [0.227–0.316] | 0.966 [0.958–0.973] | 0.368 [0.316–0.419] | 100 [82–121] | 76 [60–94] | 2,177 [2,139–2,215] | 268 [237–301] |
| **NHANES Full** | Calibrated XGBoost | Screening (80% Rec) | 0.166 | 0.763 [0.747–0.780] | 0.789 [0.768–0.811] | 0.353 [0.322–0.385] | 0.823 [0.786–0.861] | 0.754 [0.736–0.773] | 0.494 [0.459–0.529] | 303 [270–334] | 555 [511–596] | 1,698 [1,650–1,746] | 65 [49–80] |
| **NHANES Full** | Stacking Ensemble | High-Accuracy Mode | 0.449 | 0.871 [0.858–0.884] | 0.655 [0.630–0.682] | 0.562 [0.496–0.626] | 0.356 [0.306–0.408] | 0.955 [0.946–0.963] | 0.436 [0.384–0.485] | 131 [110–153] | 102 [84–122] | 2,151 [2,112–2,188] | 237 [205–266] |
| **NHANES Full** | Stacking Ensemble | Screening (80% Rec) | 0.140 | 0.757 [0.741–0.772] | 0.786 [0.765–0.807] | 0.346 [0.315–0.377] | 0.826 [0.789–0.864] | 0.745 [0.728–0.763] | 0.488 [0.452–0.521] | 304 [272–337] | 574 [534–616] | 1,679 [1,631–1,729] | 64 [50–80] |
| **NHANES Shared** | Calibrated XGBoost | High-Accuracy Mode | 0.466 | 0.865 [0.852–0.878] | 0.577 [0.558–0.597] | 0.560 [0.468–0.646] | 0.177 [0.138–0.214] | 0.977 [0.971–0.983] | 0.269 [0.216–0.319] | 65 [50–80] | 51 [38–66] | 2,202 [2,166–2,238] | 303 [271–336] |
| **NHANES Shared** | Calibrated XGBoost | Screening (80% Rec) | 0.170 | 0.699 [0.682–0.716] | 0.769 [0.749–0.789] | 0.301 [0.272–0.329] | 0.867 [0.831–0.902] | 0.672 [0.653–0.691] | 0.447 [0.414–0.479] | 319 [285–352] | 740 [695–784] | 1,513 [1,462–1,564] | 49 [35–63] |
| **NHANES Shared** | Stacking Ensemble | High-Accuracy Mode | 0.475 | 0.865 [0.851–0.878] | 0.538 [0.525–0.553] | 0.646 [0.500–0.783] | 0.084 [0.058–0.114] | 0.992 [0.989–0.996] | 0.149 [0.104–0.198] | 31 [20–42] | 17 [9–25] | 2,236 [2,201–2,272] | 337 [303–371] |
| **NHANES Shared** | Stacking Ensemble | Screening (80% Rec) | 0.130 | 0.716 [0.698–0.732] | 0.776 [0.756–0.795] | 0.313 [0.285–0.342] | 0.859 [0.825–0.894] | 0.692 [0.672–0.710] | 0.459 [0.427–0.491] | 316 [284–351] | 693 [651–736] | 1,560 [1,507–1,607] | 52 [38–66] |
| **BRFSS Shared** | Calibrated XGBoost | High-Accuracy Mode | 0.466 | 0.858 [0.857–0.860] | 0.572 [0.570–0.574] | 0.503 [0.494–0.512] | 0.172 [0.168–0.176] | 0.972 [0.971–0.973] | 0.257 [0.252–0.262] | 6,085 [5,940–6,239] | 6,013 [5,872–6,162] | 207,690 [207,320–208,039] | 29,261 [28,960–29,590] |
| **BRFSS Shared** | Calibrated XGBoost | Screening (80% Rec) | 0.170 | 0.584 [0.582–0.586] | 0.705 [0.703–0.707] | 0.238 [0.235–0.240] | 0.874 [0.870–0.877] | 0.536 [0.534–0.538] | 0.374 [0.370–0.376] | 30,875 [30,538–31,179] | 99,105 [98,642–99,589] | 114,598 [114,139–115,062] | 4,471 [4,349–4,605] |
| **BRFSS Shared** | Stacking Ensemble | High-Accuracy Mode | 0.475 | 0.860 [0.859–0.862] | 0.537 [0.536–0.538] | 0.550 [0.538–0.563] | 0.085 [0.083–0.088] | 0.988 [0.988–0.989] | 0.148 [0.143–0.153] | 3,022 [2,919–3,127] | 2,468 [2,373–2,565] | 211,235 [210,872–211,587] | 32,324 [31,970–32,657] |
| **BRFSS Shared** | Stacking Ensemble | Screening (80% Rec) | 0.130 | 0.599 [0.597–0.601] | 0.711 [0.709–0.713] | 0.244 [0.241–0.246] | 0.867 [0.864–0.871] | 0.555 [0.553–0.557] | 0.380 [0.377–0.383] | 30,649 [30,326–30,965] | 95,112 [94,633–95,614] | 118,591 [118,093–119,102] | 4,697 [4,559–4,829] |

---

## 5. Probability Calibration & Clinical Risk Stratification

### 5.1 Isotonic Probability Calibration
Raw tree ensemble probabilities frequently suffer from overconfidence in positive classifications. An isotonic calibration model fit on the validation partition substantially compressed predicted error:
- **Raw XGBoost Test Brier Score**: `0.1553`
- **Calibrated XGBoost Test Brier Score**: `0.0921` (a **40.7% relative improvement** in probabilistic reliability).

### 5.2 Decision Threshold Optimization
Because the test set exhibits class imbalance (prevalence 14.02%), the standard default threshold of 0.50 produces severe under-diagnosis. The optimal screening threshold determined on the validation ROC/PR curve is **$\tau = 0.140$**, balancing high screening sensitivity ($\approx 82.6\%$) with clinical feasibility.

### 5.3 Risk Stratification Levels
Three risk tiers were established based on predicted probability:
- **Low Risk** ($P < 0.15$): Captures **67.2%** of the test cohort; observed diabetes rate is **3.69%**.
- **Moderate Risk** ($0.15 \le P < 0.40$): Captures **21.4%** of the test cohort; observed diabetes rate is **26.56%**.
- **High Risk** ($P \ge 0.40$): Captures **11.4%** of the test cohort; observed diabetes rate is **51.68%**.

This stratification demonstrates a strong monotonic risk gradient: high-risk individuals have a **14-fold higher observed diabetes rate** than low-risk individuals.

---

## 6. Interpretability & Explainability (Native TreeSHAP)

### 6.1 Native XGBoost TreeSHAP Feature Importance (`pred_contribs=True`)

Global feature importance was computed directly from the final XGBoost booster using native TreeSHAP contribution prediction on the held-out test cohort:

| Rank | Feature | Mean Absolute SHAP Value | Clinical Role |
| :---: | :--- | :---: | :--- |
| **1** | Age (years) | 1.0149 | Exponential risk accumulation with chronological senescence. |
| **2** | Family History of Diabetes | 0.6762 | Polygenic inheritance and shared familial lifestyle predispositions. |
| **3** | BMI ($\text{kg/m}^2$) | 0.3512 | Adiposity-driven peripheral insulin resistance. |
| **4** | Told High Blood Pressure | 0.3385 | Manifestation of metabolic syndrome and vascular stiffness. |
| **5** | Told High Cholesterol | 0.2576 | Atherogenic dyslipidemia and lipid-mediated lipotoxicity. |
| **6** | Diastolic BP (mmHg) | 0.1486 | Peripheral vascular resistance. |
| **7** | Physically Active | 0.1402 | Muscular glucose uptake via insulin-independent GLUT4 translocation. |
| **8** | Systolic BP (mmHg) | 0.1099 | Hemodynamic arterial stiffening. |
| **9** | Sex (Male=1, Female=0) | 0.1044 | Slight baseline male predisposition in early adulthood. |
| **10** | Smoker ($\ge 100$ Cigs) | 0.0099 | Endothelial dysfunction and systemic low-grade inflammation. |

---

## 7. Rule-Based Lab Escalation & Recommendation Engine

To support clinical safety when optional diagnostic laboratory markers are provided, a rule-based engine operates downstream of machine learning inference:
1. **HbA1c Assessment**:
   - $< 5.7\%$: Normal
   - $5.7\% - 6.4\%$: Prediabetes
   - $\ge 6.5\%$: Suggestive of diabetes range; confirm with a clinician
2. **Fasting Plasma Glucose Assessment**:
   - $< 100\text{ mg/dL}$: Normal
   - $100 - 125\text{ mg/dL}$: Prediabetes
   - $\ge 126\text{ mg/dL}$: Suggestive of diabetes range; confirm with a clinician
3. **Escalation Logic**:
   - If either lab reaches the diagnostic range, the individual is escalated to **High Risk** regardless of survey probability, accompanied by the clinical advisory: *"Diagnostic threshold met (HbA1c >= 6.5% or Fasting Glucose >= 126 mg/dL); suggestive of diabetes range; confirm with a clinician."* Lab values raise only the risk level and never change the model's predicted probability (`predicted_probability` always strictly equals the calibrated statistical model output).
   - If prediabetic markers are detected and non-lab survey probability was Low, the rating is escalated to **Moderate Risk**: *"Elevated glycemic marker detected (HbA1c 5.7-6.4% or Glucose 100-125 mg/dL); confirm with a clinician."* The predicted probability remains unmodified.
   - If lab fields are omitted or missing, the system gracefully reports `"not provided"` and preserves pure non-lab ML inference without fabrication.
4. **Lifestyle Recommendations**: Automated clinical translation provides evidence-based guidance mapped to modifiable factors (e.g., CDC Diabetes Prevention Program guidelines recommending 5-7% weight reduction and $\ge 150\text{ min/week}$ physical activity).

---

## 8. Limitations, Domain Discrepancies, and Assumptions

1. **Cross-Survey Domain Shift**:
   - In NHANES test evaluation, the Shared-7 XGBoost model achieved ROC-AUC **0.838**. When evaluated on BRFSS ($N=249,049$), ROC-AUC dropped to **0.790**.
   - This performance drop reflects genuine real-world domain shift: NHANES features derive from standardized in-person mobile examination center interviews and direct physical measurements, whereas BRFSS relies on self-reported state telephone interviews.
2. **Class Imbalance & Trivial Baseline Reality**:
   - A trivial model predicting "no diabetes" for every patient achieves **86.0% accuracy** due to 14% base prevalence.
   - This illustrates why reporting raw accuracy alone in clinical machine learning is deceptive; balanced accuracy (0.788) and PR-AUC (0.457 vs baseline 0.140) provide the true clinical measure of utility.
3. **Historical Benchmark Framing (Pima Indians vs. NHANES)**:
   - The historical Pima Indians benchmark dataset relies on invasive biochemical markers (fasting plasma glucose, 2-hour serum insulin, triceps skinfold thickness) and achieves an ROC-AUC of **0.822**.
   - In comparison, our NHANES survey models demonstrate comparable discrimination (**0.838–0.858**) using non-invasive survey parameters alone, eliminating the barrier of immediate blood draws for population-level screening without overstating algorithmic superiority.
4. **Cross-Sectional vs. Longitudinal Scope**:
   - NHANES is a cross-sectional surveillance cohort. Models predict *current undiagnosed or diagnosed prevalence*, not 10-year prospective incident risk. Prospective longitudinal risk modeling requires Framingham or UK Biobank follow-up datasets.

---

## 9. Tier 2 Lab-Augmented Model (NHANES + HbA1c)

### 9.1 Data Harmonization & Merge
To support clinical escalation when biochemical laboratory markers are available, a Tier 2 model was developed incorporating **Glycated Hemoglobin (HbA1c, %)** from NHANES laboratory files (`GHB_H.xpt`, `GHB_I.xpt`, `GHB_J.xpt`) merged on participant identifier `SEQN`.
- **Cohort Coverage**: Out of 17,468 adult participants, **15,936 rows (91.23%)** possess complete, valid laboratory HbA1c records.
- **Class Balance**: 13,658 non-diabetic controls (85.71%) and 2,278 diagnosed diabetics (14.29%).
- **Preserved Split Integrity**: The identical train/validation/test partitioning from Tier 1 was applied (filtering missing lab records):
  - **Train Split**: 11,153 participants (1,589 cases, 9,564 controls; prevalence = 14.25%)
  - **Validation Split**: 2,391 participants (347 cases, 2,044 controls; prevalence = 14.51%)
  - **Test Split**: 2,392 participants (342 cases, 2,050 controls; prevalence = 14.30%)

### 9.2 Label Definition & Label-Noise Analysis
Crucially, the prediction target remains strictly the **self-reported clinical diagnosis (`DIQ010 == 1`)**. The label is **never** constructed or modified from HbA1c or blood glucose. Evaluating the concordance between self-reported diagnosis and biochemical threshold ($\text{HbA1c} \ge 6.5\%$) reveals meaningful epidemiological discrepancy:
- **Concordant Negative** (Self-Report No, $\text{HbA1c} < 6.5\%$): **13,266 (83.25%)**
- **Concordant Positive** (Self-Report Yes, $\text{HbA1c} \ge 6.5\%$): **1,550 (9.73%)**
- **Undiagnosed Diabetes** (Self-Report No, $\text{HbA1c} \ge 6.5\%$): **392 (2.46%)** — Individuals with biochemical diabetes who are unaware of their condition.
- **Treated / Controlled Diabetes** (Self-Report Yes, $\text{HbA1c} < 6.5\%$): **728 (4.57%)** — Diagnosed diabetics successfully managed below clinical threshold via pharmacotherapy or lifestyle (representing **31.96% of all diagnosed diabetics**).
- **Total Discordance**: **1,120 participants (7.03%)**. This physiological mismatch mathematically establishes an upper ceiling on classification accuracy (~93–94%) when predicting self-report from biomarkers, confirming why test AUC cannot exceed 0.98 without target leakage.

### 9.3 Model Performance Benchmark (Tier 2 Held-Out Test Set, $N = 2,392$)

All models evaluated with 95% bootstrap confidence intervals (1,000 resamples):

| Model | Accuracy (95% CI) | Balanced Acc | Precision | Recall (95% CI) | Specificity | F1 Score | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline (All No)** | 0.857 [0.842–0.872] | 0.500 | 0.000 | 0.000 [0.000–0.000] | 1.000 | 0.000 | 0.500 [0.470–0.530] | 0.143 [0.128–0.158] | 0.143 |
| **Ablation: HbA1c ONLY (LogReg)** | 0.903 [0.891–0.915] | 0.871 | 0.623 | 0.825 [0.783–0.864] | 0.917 | 0.709 | 0.929 [0.909–0.946] | 0.770 [0.721–0.816] | 0.087 |
| **Ablation: Tier 1 ONLY (XGB)** | 0.763 [0.745–0.780] | 0.785 | 0.356 | 0.816 [0.774–0.858] | 0.754 | 0.496 | 0.855 [0.834–0.876] | 0.472 [0.419–0.534] | 0.158 |
| **Logistic Regression** | 0.893 [0.880–0.904] | 0.890 | 0.582 | 0.886 [0.851–0.920] | 0.894 | 0.702 | 0.944 [0.929–0.957] | 0.802 [0.757–0.847] | 0.085 |
| **Random Forest** | 0.913 [0.902–0.924] | 0.888 | 0.649 | 0.854 [0.816–0.893] | 0.923 | 0.737 | 0.949 [0.936–0.961] | 0.827 [0.790–0.861] | 0.076 |
| **XGBoost** | 0.906 [0.894–0.916] | 0.886 | 0.623 | 0.860 [0.821–0.893] | 0.913 | 0.722 | 0.949 [0.934–0.960] | 0.833 [0.795–0.863] | 0.077 |
| **PyTorch MLP** | 0.885 [0.871–0.897] | 0.884 | 0.561 | 0.883 [0.844–0.918] | 0.885 | 0.686 | 0.950 [0.938–0.963] | 0.824 [0.780–0.865] | 0.087 |
| **Stacking Ensemble** | 0.936 [0.926–0.945] | 0.860 | 0.789 | 0.754 [0.708–0.797] | 0.966 | 0.771 | **0.951 [0.938–0.963]** | **0.836 [0.798–0.872]** | **0.050** |
| **Final Calibrated XGBoost** | 0.935 [0.925–0.945] | 0.846 | 0.805 | 0.722 [0.676–0.771] | 0.971 | 0.761 | 0.946 [0.933–0.958] | 0.801 [0.763–0.837] | **0.050** |

> [!NOTE]
> **Statistical Significance & Model Parity in Tier 2**: The 95% bootstrap confidence intervals for ROC-AUC overlap closely across Random Forest (0.936–0.961), XGBoost (0.934–0.960), PyTorch MLP (0.938–0.963), and the Stacking Ensemble (0.938–0.963). While Stacking marginally optimizes PR-AUC, calibrated XGBoost matches its Brier score (0.050) while offering sub-millisecond inference and direct TreeSHAP tree traversal.

### 9.4 Tier 1 vs. Tier 2 Fair Comparison on Identical Test Rows ($N = 2,392$)

Evaluating both tiers on the **exact same 2,392 test individuals** isolates the true clinical value of laboratory augmentation:

| Metric / Setting | Tier 1 (No HbA1c, Same Rows) | Tier 2 (With HbA1c, Same Rows) | Absolute Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC (Overall Discrimination)** | 0.852 | **0.946** | **+0.094** |
| **PR-AUC (Precision-Recall Area)** | 0.452 | **0.801** | **+0.349** |
| **Brier Score (Lower is Better)** | 0.094 | **0.050** | **-0.044 (46.8% relative error reduction)** |
| **High-Accuracy Mode: Threshold ($\tau$)** | 0.402 | 0.376 | - |
| **High-Accuracy Mode: Accuracy** | 0.865 | **0.935** | **+0.070** |
| **High-Accuracy Mode: Recall (Sensitivity)** | 0.272 | **0.722** | **+0.450 (+165% increase in captured cases)** |
| **Screening Mode: Threshold ($\tau_{80}$)** | 0.142 | 0.290 | - |
| **Screening Mode: Accuracy** | 0.758 | **0.923** | **+0.165 (68% reduction in false positives)** |
| **Screening Mode: Recall (Sensitivity)** | 0.827 | **0.827** | **+0.000 (Parity at Target Sensitivity)** |

### 9.5 Interpretability: Global TreeSHAP Ranking
Native XGBoost TreeSHAP contribution analysis (`pred_contribs=True`) reveals that `hba1c` ranks as the **#1 dominant predictor** with a Mean |SHAP| attribution of **1.9867**, more than quadruple the next strongest feature:
1. `hba1c` (1.9867) — Glycated hemoglobin level
2. `age` (0.4538) — Chronological age
3. `family_history` (0.4470) — Genetic predisposition
4. `high_bp` (0.3859) — Diagnosed hypertension
5. `high_chol` (0.2179) — Dyslipidemia
6. `bmi` (0.1845) — Adiposity
7. `mean_dbp` (0.1452) — Diastolic blood pressure
8. `phys_activity` (0.1108) — Exercise engagement
9. `sex` (0.0755) — Biological sex
10. `mean_sbp` (0.0430) — Systolic blood pressure
11. `smoker` (0.0239) — Smoking status

### 9.6 One System Architecture: The Unified `/predict` Endpoint
The service provides a single production endpoint (`POST /predict`):
- **Missing `hba1c`**: Automatically routes to Tier 1 non-lab models, returning survey-based calibrated risk and `"model_used": "tier1_no_labs"`.
- **Valid `hba1c` (3.0–18.0%)**: Automatically activates Tier 2 lab-augmented models, returning updated risk probability, tier 2 SHAP attributions, and `"model_used": "tier2_with_hba1c"`.
- **Invalid / Out-of-Range `hba1c`**: Returns clear HTTP 422 validation error without inventing or imputing missing lab data.

### 9.7 Tier 2 Limitations & Methodological Constraints
1. **Lack of External Validation**: Unlike Tier 1 (which was externally validated on 249,049 BRFSS participants), Tier 2 cannot be evaluated on BRFSS because nationwide telephone surveys do not collect biochemical blood draws.
2. **Missing Lab Subsetting**: Tier 2 models are trained strictly on individuals who attended mobile examination centers and provided blood samples (8.8% of NHANES participants lacked valid lab data).
3. **Self-Report Label Noise**: Because ground truth is survey-reported diagnosis (`DIQ010`), treated diabetics maintaining normoglycemia via medication appear as "false negatives" to pure glycemic rules, capping theoretical accuracy.
4. **Different Clinical Questions**: Tier 1 answers *"Should this person receive a screening blood test?"* whereas Tier 2 answers *"Given non-diagnostic survey factors and an available A1c, what is the calibrated overall risk?"* Their metrics reflect distinct operational contexts and must not be conflated.

