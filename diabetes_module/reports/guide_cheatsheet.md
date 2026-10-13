# LifeMap AI: Diabetes Risk Module — Executive Cheatsheet

### Key Performance Numbers

* **Baseline Accuracy (Predict All Healthy)**: **86.0%** (Tier 1) / **85.7%** (Tier 2) at 14.0% disease prevalence.
* **Tier 1 (Non-Lab, 10 Features)**:
  * *High-Accuracy Mode* ($\tau = 0.402$): **Accuracy = 86.9%** | **Recall = 27.2%** | **Specificity = 96.6%** | **ROC-AUC = 0.856**
  * *Screening Mode* ($\tau = 0.166$): **Accuracy = 76.3%** | **Recall = 82.3%** | **Specificity = 75.4%** | **ROC-AUC = 0.856**
* **Tier 2 (Lab-Augmented with HbA1c, 11 Features)**:
  * *High-Accuracy Mode* ($\tau = 0.376$): **Accuracy = 93.5%** | **Recall = 72.2%** | **Specificity = 97.1%** | **ROC-AUC = 0.946**
  * *Screening Mode* ($\tau = 0.290$): **Accuracy = 92.3%** | **Recall = 82.7%** | **Specificity = 93.9%** | **ROC-AUC = 0.946**
* **Biochemical Rule Baseline ($\text{HbA1c} \ge 6.5\%$)**: **Accuracy = 93.2%** | **Recall = 69.3%** | **Specificity = 97.2%** (misses 30.7% of diagnosed diabetics).

---

### 8 Critical Questions & Honest Answers

1. **Why isn't accuracy 90% for Tier 1?**  
   Non-invasive demographic and survey features lack biochemical certainty; at 14% disease prevalence, pushing accuracy to 90% without blood markers would force the model to predict everyone healthy, missing nearly all diabetic individuals.

2. **Why use ML when an HbA1c cutoff gets 93% accuracy?**  
   The rigid 6.5% laboratory cutoff fails to detect the 32% of diagnosed diabetics whose glucose is pharmacologically controlled below 6.5%, whereas machine learning combines lifestyle and demographic context to correctly identify managed patients and borderline risks.

3. **Why is recall more important than accuracy for screening?**  
   Missing an undiagnosed diabetic patient (false negative) delays preventive treatment and risks irreversible renal or vascular damage, whereas a false positive only leads to an inexpensive, harmless confirmatory blood test.

4. **Why did accuracy and ROC-AUC drop on BRFSS?**  
   The ~0.046 ROC-AUC drop reflects survey modality shift between in-person calibrated clinical examinations (NHANES) and telephone self-reports (BRFSS with coarser activity definitions), rather than algorithmic overfitting.

5. **What is the deep learning component?**  
   The system trains a 4-layer PyTorch Multi-Layer Perceptron (MLP) with batch normalization and dropout, alongside a sparse attentive TabNet model, which serve as neural benchmarks and contribute meta-features to the stacking ensemble.

6. **How do you know there is no data leakage?**  
   Data splits were stratified before any imputation or scaling, all laboratory tests were strictly excluded from Tier 1, and an automated audit verified that test ROC-AUC remained strictly below realistic clinical upper bounds (0.90 for Tier 1; 0.98 for Tier 2).

7. **What is SHAP?**  
   SHAP (SHapley Additive exPlanations) is a game-theoretic framework that computes the exact marginal contribution of each clinical feature to an individual patient's predicted risk relative to the population base rate.

8. **Is this a clinical diagnosis tool?**  
   No, LifeMap AI is an epidemiological risk stratification and screening tool designed to flag high-risk individuals for confirmatory medical evaluation, not a diagnostic medical device.
