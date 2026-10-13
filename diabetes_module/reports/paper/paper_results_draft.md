# Methods, Empirical Results, and Limitations Draft

## 1. Methods

### 1.1 Study Cohorts and Harmonization
We analyzed data from the National Health and Nutrition Examination Survey (NHANES) pooled across three biennial cycles: 2013-2014, 2015-2016, and 2017-2018 [CITE]. From an initial sample of 29,400 individuals, 17,961 adults aged $\ge 18$ years completed the diabetes questionnaire (`DIQ010`). After excluding indeterminate and borderline diagnoses ($n=482$) and missing responses ($n=11$), the final harmonized NHANES cohort comprised 17,468 participants (2,449 self-reported diabetic cases [14.02%] and 15,019 non-diabetic controls [85.98%]). For external validation, we utilized the 2015 Behavioral Risk Factor Surveillance System (BRFSS) [CITE], comprising 249,049 adult respondents (35,346 diabetic cases [14.19%] and 213,703 controls) after dropping 4,631 prediabetes cases.

### 1.2 Two-Tier Model Architecture and Preprocessing
The framework is partitioned into two tiers depending on laboratory biomarker availability:
- **Tier 1 (Non-Invasive Screening):** Ten features spanning chronological age, biological sex, body mass index (BMI), mean examination systolic blood pressure, mean diastolic blood pressure, told high blood pressure, told high cholesterol, cigarette smoking history, physical activity status, and first-degree family history of diabetes.
- **Tier 2 (Biomarker-Augmented Screening):** The ten Tier 1 features augmented by laboratory-assayed glycated hemoglobin (HbA1c, %).

Missing values in continuous and binary predictors were imputed using median and mode strategies, respectively. Crucially, all preprocessing transformations and scalers were fitted strictly on the training partition ($n=12,227$) and applied to the validation ($n=2,620$) and test ($n=2,621$) sets without leakage [CITE].

### 1.3 Model Development, Stacking, and Calibration
We trained and evaluated five core model architectures: Logistic Regression (L-BFGS), Random Forest (300 estimators), XGBoost (learning rate 0.04, maximum depth 4, 180 estimators), a 3-layer PyTorch Multilayer Perceptron (MLP; 64-32-16 architecture with batch normalization and dropout), and a Stacking Ensemble utilizing a logistic regression meta-learner [CITE]. Class imbalance ($1:6.13$) was addressed via loss-function class weighting and positive-instance scaling (`scale_pos_weight = 6.13`). 

Because raw boosting probabilities tend to be uncalibrated, post-hoc isotonic probability calibration was fitted on out-of-fold validation predictions to produce the Final Calibrated XGBoost model [CITE]. Operating points were selected exclusively on the validation set at two clinical settings: a High-Accuracy mode (maximizing balanced accuracy) and a Screening mode targeting $\ge 80\%$ recall [CITE].

---

## 2. Empirical Results

### 2.1 Tier 1 Performance on NHANES Test Split (n=2,621)
At the common validation-selected 80% target recall operating points, model discrimination was consistent across architectures:
- **Logistic Regression ($\theta = 0.473$):** Accuracy 0.741 [95% CI: 0.724 - 0.758], Recall 0.837 [95% CI: 0.798 - 0.873], Specificity 0.725 [95% CI: 0.706 - 0.744], ROC-AUC 0.855 [95% CI: 0.833 - 0.875], PR-AUC 0.477 [95% CI: 0.430 - 0.532].
- **Random Forest ($\theta = 0.352$):** Accuracy 0.742 [95% CI: 0.726 - 0.760], Recall 0.829 [95% CI: 0.791 - 0.866], Specificity 0.728 [95% CI: 0.710 - 0.746], ROC-AUC 0.849 [95% CI: 0.827 - 0.869], PR-AUC 0.459 [95% CI: 0.411 - 0.513].
- **XGBoost ($\theta = 0.499$):** Accuracy 0.762 [95% CI: 0.744 - 0.776], Recall 0.826 [95% CI: 0.785 - 0.861], Specificity 0.751 [95% CI: 0.732 - 0.768], ROC-AUC 0.858 [95% CI: 0.838 - 0.878], PR-AUC 0.480 [95% CI: 0.432 - 0.534].
- **PyTorch MLP ($\theta = 0.527$):** Accuracy 0.757 [95% CI: 0.740 - 0.774], Recall 0.848 [95% CI: 0.811 - 0.883], Specificity 0.742 [95% CI: 0.724 - 0.761], ROC-AUC 0.853 [95% CI: 0.832 - 0.873], PR-AUC 0.489 [95% CI: 0.439 - 0.544].
- **Stacking Ensemble ($\theta = 0.138$):** Accuracy 0.754 [95% CI: 0.737 - 0.770], Recall 0.826 [95% CI: 0.785 - 0.863], Specificity 0.742 [95% CI: 0.723 - 0.760], ROC-AUC 0.857 [95% CI: 0.836 - 0.878], PR-AUC 0.486 [95% CI: 0.436 - 0.543].

Paired bootstrap significance testing (1,000 resamples) revealed that performance differences between Stacking and XGBoost ($\Delta \text{AUC} = -0.001$, 95% CI: $[-0.009, +0.007]$, $p=0.844$) and between MLP and XGBoost ($\Delta \text{AUC} = -0.005$, 95% CI: $[-0.019, +0.008]$, $p=0.452$) were not statistically significant, as all confidence intervals spanned zero. XGBoost exhibited a minor, non-significant advantage over Logistic Regression ($\Delta \text{AUC} = +0.003$, 95% CI: $[-0.005, +0.011]$, $p=0.458$).

### 2.2 External Validation on CDC BRFSS (n=249,049)
Evaluated zero-shot using the Shared-7 subset, discrimination remained stable across the entire US population:
- **Logistic Regression:** ROC-AUC 0.804 [95% CI: 0.801 - 0.806], Recall 0.866, Accuracy 0.593.
- **XGBoost:** ROC-AUC 0.804 [95% CI: 0.802 - 0.807], Recall 0.871, Accuracy 0.588.
- **Stacking Ensemble:** ROC-AUC 0.806 [95% CI: 0.804 - 0.809], Recall 0.870, Accuracy 0.595.

### 2.3 Tier 2 Biomarker-Augmented Performance (n=2,392)
Incorporating laboratory HbA1c substantially improved predictive discrimination:
- **Baseline Healthy Rule (Predict all healthy):** Accuracy 0.857, Recall 0.000, ROC-AUC 0.500.
- **Baseline HbA1c $\ge 6.5\%$ Rule:** Accuracy 0.903, Recall 0.708, Specificity 0.936, F1 0.677.
- **Final Calibrated XGBoost ($\theta = 0.290$, Screening Mode):** Accuracy 0.913 [95% CI: 0.902 - 0.924], Recall 0.854 [95% CI: 0.816 - 0.893], Specificity 0.923 [95% CI: 0.911 - 0.934], ROC-AUC 0.946 [95% CI: 0.933 - 0.958], PR-AUC 0.801 [95% CI: 0.763 - 0.837].
- **High-Accuracy Mode ($\theta = 0.376$):** Accuracy 0.935 [95% CI: 0.925 - 0.945], Recall 0.722 [95% CI: 0.676 - 0.771], Specificity 0.971 [95% CI: 0.963 - 0.978].

Direct comparison on the identical 2,392 test individuals demonstrated that Tier 2 improved ROC-AUC by $+0.091$ over Tier 1 (0.946 vs. 0.856; paired bootstrap 95% CI: $[+0.076, +0.106]$, $p < 0.001$). Importantly, the simple clinical heuristic of $\text{HbA1c} \ge 6.5\%$ achieved an overall accuracy of 0.903, closely approaching the machine learning models, but missed 29.2% of self-reported diabetic cases (recall = 0.708), whereas Tier 2 captured 85.4% of cases.

### 2.4 Calibration and Robustness
Isotonic probability calibration substantially reduced Brier scores (Tier 1: 0.155 to 0.092; Tier 2: 0.077 to 0.050) and reduced Expected Calibration Error to 0.016 (Tier 1) and 0.012 (Tier 2). Robustness evaluations confirmed stability:
- **5-Fold Stratified CV on Training Split:** Mean ROC-AUC $0.848 \pm 0.009$ (XGBoost) and $0.846 \pm 0.010$ (Logistic Regression).
- **Temporal Holdout (Trained on 2013-2016, Tested on 2017-2018):** ROC-AUC was 0.842 for XGBoost and 0.839 for Logistic Regression.
- **Complete-Case Analysis:** Evaluating on complete cases ($n=2,209$) yielded ROC-AUC 0.851 (XGBoost) versus 0.858 on the full imputed test pipeline, indicating that median/mode imputation introduced no artificial inflation of model performance.

---

## 3. Limitations

1. **Self-Reported Diagnosis Label:** NHANES and BRFSS ground truth relies on self-reported physician diagnoses rather than oral glucose tolerance testing [CITE]. Consequently, asymptomatic undiagnosed individuals with elevated glucose are misclassified as controls in the raw data.
2. **Confidence Interval Overlap:** In Tier 1, 95% confidence intervals across classical ML, deep learning, and ensemble methods overlap heavily. Complex architectures (MLP, TabNet, Stacking) do not offer statistically significant gains over tuned XGBoost or regularized Logistic Regression on these tabular data.
3. **Equivalence of Simple Clinical Heuristics:** In Tier 2, while calibrated XGBoost achieves higher sensitivity (0.854 vs 0.708), the raw classification accuracy of the simple threshold rule ($\text{HbA1c} \ge 6.5\%$, accuracy 0.903) is within 1% of the model in screening mode (0.913) [CITE].
4. **Generalizability to Clinical Cohorts:** Both development datasets reflect community-dwelling survey participants. Application in acute ambulatory or high-risk clinical populations warrants prospectively collected local clinical validation.
