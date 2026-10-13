# Tier 2 Label-Noise Analysis: Self-Report vs. Biochemical Glycohemoglobin (HbA1c)

Among $N = 15,936$ NHANES adult participants with valid laboratory HbA1c measurements:

| Category | Definition | Count ($N$) | Percentage of Cohort | Clinical & Epidemiological Meaning |
| :--- | :--- | :---: | :---: | :--- |
| **Concordant Negative** | Self-Report No ($Y=0$) & $\text{HbA1c} < 6.5\%$ | 13,266 | 83.25% | Healthy normoglycemic or prediabetic controls. |
| **Concordant Positive** | Self-Report Yes ($Y=1$) & $\text{HbA1c} \ge 6.5\%$ | 1,550 | 9.73% | Diagnosed diabetics with uncontrolled or active hyperglycemia. |
| **Undiagnosed Diabetes** | Self-Report No ($Y=0$) & $\text{HbA1c} \ge 6.5\%$ | 392 | 2.46% | **Undiagnosed diabetes**: Asymptomatic individuals meeting biochemical diagnostic criteria without formal medical diagnosis. Represents **2.87%** of self-reported non-diabetics. |
| **Controlled/Treated Diabetes** | Self-Report Yes ($Y=1$) & $\text{HbA1c} < 6.5\%$ | 728 | 4.57% | **Treated/Managed diabetes**: Patients with known diagnosis maintaining glycemia in non-diabetic range via metformin, insulin, lifestyle, or GLP-1 receptor agonists. Represents **31.96%** of diagnosed diabetics. |
| **Total Discordant (Label Noise)** | Mismatch between Self-Report and $\text{HbA1c} \ge 6.5\%$ | 1,120 | 7.03% | **Structural ceiling on classification accuracy**: Because models are trained to predict medical self-report (`DIQ010`), perfect biochemical agreement is impossible. |

### Key Takeaway for Machine Learning Modeling
1. The ground truth label remains strictly the self-reported clinical diagnosis (`DIQ010`). We do **NOT** modify or sanitize labels based on HbA1c, preserving honest epidemiological evaluation.
2. The existence of 728 treated diabetics (32.0% of all diabetics) and 392 undiagnosed individuals mathematically caps the maximum achievable accuracy when predicting self-report from biochemical markers alone, proving why accuracy in the high 90s is impossible without severe label leakage.
