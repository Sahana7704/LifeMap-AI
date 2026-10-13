"""
Append Part B to results_report.ipynb cleanly starting from results_report_BACKUP.ipynb.
"""

import shutil
import nbformat as nbf
from pathlib import Path

nb_bak = Path("diabetes_module/notebooks/results_report_BACKUP.ipynb")
nb_path = Path("diabetes_module/notebooks/results_report.ipynb")

# Reset from backup
shutil.copy2(nb_bak, nb_path)
nb = nbf.read(nb_path, as_version=4)

orig_cell_count = len(nb.cells)
print(f"Original cell count from backup: {orig_cell_count}")

# 1. Part B Main Header
cell_b_header = nbf.v4.new_markdown_cell("""---
# Part B: Tier 2 Lab-Augmented Model (NHANES + HbA1c), used automatically when an HbA1c value is provided
This section presents the **Tier 2 Lab-Augmented Diabetes Risk Architecture**, engineered to incorporate biochemical Glycated Hemoglobin (HbA1c) when laboratory markers are available.""")

# 2. Section 12: How the Two Tiers Work Together
cell_12_md = nbf.v4.new_markdown_cell("""## 12. How the Two Tiers Work Together in a Unified System
LifeMap AI deploys both screening paradigms through a single production endpoint (`POST /predict`):
- **Tier 1 (Non-Invasive Screening)**: When no laboratory tests have been performed, the system executes the Tier 1 model trained exclusively on non-lab survey features (`age`, `sex`, `bmi`, `mean_sbp`, `mean_dbp`, `high_bp`, `high_chol`, `smoker`, `phys_activity`, `family_history`). Response metadata designates `"model_used": "tier1_no_labs"`.
- **Tier 2 (Lab-Augmented Stratification)**: When an HbA1c value (3.0%–18.0%) is provided, the API automatically routes inference to Tier 2, integrating biochemical glycemia into the feature space. Response metadata designates `"model_used": "tier2_with_hba1c"`.

```
                    ┌─────────────────────────────────────────────────────┐
                    │               Client POST /predict                   │
                    │   Demographics, Vitals, History + Optional HbA1c   │
                    └──────────────────────────┬──────────────────────────┘
                                               │
                                      Is HbA1c Provided?
                                      (Range: 3.0 - 18.0%)
                                       /               \\
                                      /                 \\
                                  [ NO ]              [ YES ]
                                    /                     \\
                                   v                       v
               ┌───────────────────────────────┐ ┌───────────────────────────────┐
               │    TIER 1: Non-Lab Model      │ │    TIER 2: Lab-Augmented      │
               │  10 Survey / Vitals Features  │ │   10 Features + HbA1c Lab     │
               │  Calibrated XGBoost (AUROC=.86)│ │  Calibrated XGBoost (AUROC=.95)│
               │  model_used: tier1_no_labs    │ │  model_used: tier2_with_hba1c │
               └───────────────┬───────────────┘ └───────────────┬───────────────┘
                               │                                 │
                               └────────────────┬────────────────┘
                                                v
                               ┌─────────────────────────────────┐
                               │  ADA Rule-Based Safety Checker  │
                               │  & Lab Escalation Verification   │
                               └────────────────┬────────────────┘
                                                v
                               ┌─────────────────────────────────┐
                               │     Unified JSON Response:      │
                               │ Risk %, Tiers, SHAP Drivers,Recs│
                               └─────────────────────────────────┘
```""")

# 3. Section 13: Tier 2 Dataset Architecture & Summary
cell_13_md = nbf.v4.new_markdown_cell("""## 13. Tier 2 Dataset Architecture & Summary
Summary of the NHANES laboratory-augmented adult cohort ($N = 15,936$ with valid HbA1c).
**Leakage Note**: The ground-truth label remains strictly the self-reported clinical diagnosis (`DIQ010`), preserving honest evaluation. HbA1c was measured in an independent venipuncture blood draw and was never part of questionnaire label construction.""")

cell_13_code = nbf.v4.new_code_cell(r"""t2_data_summary = pd.DataFrame({
    'Cohort Partition': ['Full NHANES Adult Cohort', 'Tier 2 Cohort (Valid HbA1c)', 'Tier 2 Training Split', 'Tier 2 Validation Split', 'Tier 2 Held-Out Test Split'],
    'Total Participants (N)': ['17,468', '15,936 (91.23%)', '11,153 (70.0%)', '2,391 (15.0%)', '2,392 (15.0%)'],
    'Diabetes Cases': ['2,449', '2,278', '1,589', '347', '342'],
    'Prevalence': ['14.02%', '14.29%', '14.25%', '14.51%', '14.30%'],
    'Feature Count': ['10 Features', '11 Features (+HbA1c)', '11 Features', '11 Features', '11 Features'],
    'Label Source': ['DIQ010 (Self-Report)', 'DIQ010 (Self-Report)', 'DIQ010 (Self-Report)', 'DIQ010 (Self-Report)', 'DIQ010 (Self-Report)']
})
display(HTML(t2_data_summary.to_html(index=False, classes='table table-bordered table-striped table-hover')))
print('Tier 2 Feature Space (11 Total Features):')
print('  Continuous (5): age, bmi, mean_sbp, mean_dbp, hba1c')
print('  Binary (6):     sex, high_bp, high_chol, smoker, phys_activity, family_history')""")

# 4. Section 14: Label-Noise Analysis
cell_14_md = nbf.v4.new_markdown_cell("""## 14. Label-Noise Analysis & Theoretical Accuracy Bounds
Discrepancy between self-reported diagnosis (`DIQ010`) and biochemical threshold ($\text{HbA1c} \ge 6.5\%$).
**Structural Accuracy Ceiling**: Because 31.96% of diagnosed diabetics have controlled glycemia ($\text{HbA1c} < 6.5\%$) via pharmacotherapy, and 2.46% of non-diagnosed adults exhibit undiagnosed hyperglycemia, overall accuracy is bounded around 93%–94%.""")

cell_14_code = nbf.v4.new_code_cell(r"""with open('../reports/tables/tier2_label_noise.md', 'r', encoding='utf-8') as f:
    display(HTML(f"<div style='background-color:#F8FAFC; padding:15px; border-left:4px solid #3B82F6;'>{f.read()}</div>"))

display(Image(filename='../reports/figures/tier2_label_noise_bar.png'))""")

# 5. Section 15: Tier 2 Model Comparison Table
cell_15_md = nbf.v4.new_markdown_cell("""## 15. Tier 2 Model Comparison Table (Held-Out Test Set, $N = 2,392$)
Evaluation of candidate architectures with **95% bootstrap confidence intervals** (1,000 resamples).
**Best value per column highlighted in green.**""")

cell_15_code = nbf.v4.new_code_cell(r"""t2_table_df = pd.read_csv('../reports/tables/tier2_test_comparison.csv')

def parse_num(val):
    try:
        return float(str(val).split()[0])
    except:
        return 0.0

num_t2 = t2_table_df.copy()
for m in ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'Brier']:
    num_t2[f'{m}_val'] = num_t2[m].apply(parse_num)

def highlight_best_t2(s):
    style_res = pd.DataFrame('', index=s.index, columns=s.columns)
    mask = s['Model'] != 'Trivial Baseline (All No)'
    green_style = 'background-color: #D1FAE5; font-weight: bold; color: #065F46;'
    for c in ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']:
        best_idx = num_t2.loc[mask, f'{c}_val'].idxmax()
        style_res.loc[best_idx, c] = green_style
    best_brier = num_t2.loc[mask, 'Brier_val'].idxmin()
    style_res.loc[best_brier, 'Brier'] = green_style
    return style_res

styled_t2 = t2_table_df.style.apply(highlight_best_t2, axis=None).set_caption(
    '<b>Comprehensive Tier 2 Model Performance (NHANES + HbA1c Held-Out Test Set, N=2,392)</b>'
)
display(styled_t2)""")

# 6. Section 16: Key Metrics Bar Chart, Confusion Matrices & ROC/PR Curves
cell_16_md = nbf.v4.new_markdown_cell("""## 16. Key Metrics Bar Chart, Confusion Matrices & ROC/PR Curves
Visual benchmark across all evaluated Tier 2 architectures.""")

cell_16_code = nbf.v4.new_code_cell(r"""display(HTML('<h3>16.1 Key Performance Metrics Bar Chart</h3>'))
display(Image(filename='../reports/figures/tier2_model_metrics_bar.png'))

display(HTML('<h3>16.2 Confusion Matrices Across Evaluated Architectures</h3>'))
display(Image(filename='../reports/figures/tier2_confusion_matrices.png'))

display(HTML('<h3>16.3 Overlaid ROC and Precision-Recall Curves</h3>'))
display(Image(filename='../reports/figures/tier2_roc_pr_curves.png'))""")

# 7. Section 17: Probability Calibration & Deep MLP Training Dynamics
cell_17_md = nbf.v4.new_markdown_cell("""## 17. Probability Calibration & Deep MLP Training Dynamics
Reliability curves before vs. after isotonic calibration and PyTorch MLP training convergence.""")

cell_17_code = nbf.v4.new_code_cell(r"""print('=== PyTorch Tier 2 Deep MLP Architecture Specification ===')
print('Input Dimension: 11 tabular features (10 survey/vitals + HbA1c)')
print('Layer 1: Linear(11 -> 64) -> BatchNorm1d(64) -> ReLU() -> Dropout(0.25)')
print('Layer 2: Linear(64 -> 32) -> BatchNorm1d(32) -> ReLU() -> Dropout(0.20)')
print('Layer 3: Linear(32 -> 16) -> ReLU()')
print('Output Layer: Linear(16 -> 1) [BCEWithLogitsLoss with positive weighting = 6.02]')
print('Optimization: Adam (lr=1e-3, weight_decay=1e-4) with validation early stopping\n')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
im1 = PILImage.open('../reports/figures/tier2_calibration_curves.png')
ax1.imshow(im1)
ax1.axis('off')
ax1.set_title('Isotonic Probability Calibration', fontsize=12, fontweight='bold')

im2 = PILImage.open('../reports/figures/tier2_mlp_training_history.png')
ax2.imshow(im2)
ax2.axis('off')
ax2.set_title('PyTorch MLP Training Loss Dynamics', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""")

# 8. Section 18: Tier 2 Dual Operating-Point Analysis
cell_18_md = nbf.v4.new_markdown_cell("""## 18. Tier 2 Dual Operating-Point Analysis: High-Accuracy Mode vs. Screening Mode
Evaluating **Calibrated XGBoost** and **Stacking Ensemble** under:
- **(a) High-Accuracy Mode ($\tau_{\text{acc}}$)**: Threshold chosen on validation to maximize overall accuracy.
- **(b) Screening Mode ($\tau_{80}$)**: Threshold chosen on validation to guarantee $\ge 80\%$ recall.""")

cell_18_code = nbf.v4.new_code_cell(r"""t2_dop_table = pd.read_csv('../reports/tables/tier2_dual_operating_points.csv')
display(HTML('<h4>Table 18: Tier 2 Dual Operating-Point Metrics & 95% Bootstrap CIs</h4>'))
display(HTML(t2_dop_table.to_html(index=False, classes='table table-bordered table-striped table-hover')))

display(Image(filename='../reports/figures/tier2_dual_operating_points_cm.png'))""")

# 9. Section 19: Tier 1 vs Tier 2 Fair Comparison on Identical Rows
cell_19_md = nbf.v4.new_markdown_cell("""## 19. Tier 1 vs. Tier 2 Fair Comparison on Identical Rows ($N = 2,392$)
Direct comparison isolating the clinical contribution of laboratory HbA1c on the exact same held-out test cohort.""")

cell_19_code = nbf.v4.new_code_cell(r"""t1_t2_comp = pd.read_csv('../reports/tables/tier2_tier1_vs_tier2_comparison.csv')
display(HTML('<h4>Table 19: Tier 1 vs. Tier 2 Metrics on Identical Test Rows</h4>'))
display(HTML(t1_t2_comp.to_html(index=False, classes='table table-bordered table-hover table-striped')))

display(Image(filename='../reports/figures/tier2_tier1_vs_tier2_comparison.png'))""")

# 10. Section 20: Tier 2 Interpretability & Feature Attribution
cell_20_md = nbf.v4.new_markdown_cell("""## 20. Tier 2 Interpretability: Native TreeSHAP & LIME Explanations
Global feature attribution rankings and patient-level waterfall decompositions.""")

cell_20_code = nbf.v4.new_code_cell(r"""t2_shap_rank = pd.read_csv('../reports/tables/tier2_shap_global_rankings.csv')
print('=== Top 5 Tier 2 Global Feature Drivers ===')
for _, r in t2_shap_rank.head(5).iterrows():
    print(f"  Rank {int(r['Rank'])}: {r['Feature']:<16} (Mean |SHAP| = {r['Mean_Abs_SHAP']:.4f})")
print(f"\nHbA1c Rank: #{int(t2_shap_rank[t2_shap_rank['Feature']=='hba1c']['Rank'].values[0])} (Overwhelmingly the primary driver)\n")

display(Image(filename='../reports/figures/tier2_shap_summary.png'))

display(HTML('<h3>Patient-Level SHAP Waterfall Explanations (Low, Moderate, High Risk)</h3>'))
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
for idx, (f_name, tit) in enumerate([
    ('tier2_shap_waterfall_patient_1.png', 'Patient 1: Low Risk (< 5%)'),
    ('tier2_shap_waterfall_patient_2.png', 'Patient 2: Moderate Risk (25-35%)'),
    ('tier2_shap_waterfall_patient_3.png', 'Patient 3: High Risk (> 85%)')
]):
    im = PILImage.open(f'../reports/figures/{f_name}')
    axes[idx].imshow(im)
    axes[idx].axis('off')
    axes[idx].set_title(tit, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

display(HTML('<h3>Patient-Level LIME Surrogate Explanations</h3>'))
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
for idx, (f_name, tit) in enumerate([
    ('tier2_lime_patient_1.png', 'LIME: Patient 1 (Low Risk)'),
    ('tier2_lime_patient_2.png', 'LIME: Patient 2 (Moderate Risk)'),
    ('tier2_lime_patient_3.png', 'LIME: Patient 3 (High Risk)')
]):
    im = PILImage.open(f'../reports/figures/{f_name}')
    axes[idx].imshow(im)
    axes[idx].axis('off')
    axes[idx].set_title(tit, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""")

# 11. Section 21: Tier 2 Risk Level Stratification
cell_21_md = nbf.v4.new_markdown_cell("""## 21. Tier 2 Risk Level Stratification & Observed Diabetes Rates
Distribution of test individuals across Low, Moderate, and High clinical tiers and observed diabetes incidence.""")

cell_21_code = nbf.v4.new_code_cell(r"""with open('../models/tier2_risk_thresholds.json', 'r') as f:
    t2_risk_meta = json.load(f)

print(f"Tier 2 Optimal Screening Threshold: {t2_risk_meta['decision_threshold']:.3f}")
print(f"Tier 2 High-Accuracy Threshold:    {t2_risk_meta['high_accuracy_threshold']:.3f}")
print(f"Low Risk Tier:      < {t2_risk_meta['threshold_low']:.2f}")
print(f"Moderate Risk Tier: {t2_risk_meta['threshold_low']:.2f} to {t2_risk_meta['threshold_high']:.2f}")
print(f"High Risk Tier:     >= {t2_risk_meta['threshold_high']:.2f}\n")

r_t2_df = pd.DataFrame(t2_risk_meta['risk_summary'])
r_t2_df['share_of_test'] = r_t2_df['share_of_test'].apply(lambda x: f'{x*100:.2f}%')
r_t2_df['observed_diabetes_rate'] = r_t2_df['observed_diabetes_rate'].apply(lambda x: f'{x*100:.2f}%')
r_t2_df.columns = ['Risk Tier Level', 'Score Range', 'Test Patients (N)', 'Share of Test Cohort', 'Observed Diabetes Rate']
display(HTML(r_t2_df.to_html(index=False, classes='table table-bordered table-hover')))

display(Image(filename='../reports/figures/tier2_risk_level_distribution.png'))""")

# 12. Section 22: API Demonstration
cell_22_md = nbf.v4.new_markdown_cell("""## 22. Production API Demonstration: The Unified `/predict` Endpoint
Verification of single-endpoint dispatch across 5 clinical scenarios, demonstrating automatic routing to `tier1_no_labs` or `tier2_with_hba1c`.""")

cell_22_code = nbf.v4.new_code_cell(r"""with open('../reports/tables/tier2_api_example_outputs.json', 'r') as f:
    api_demo = json.load(f)

demo_rows = []
for item in api_demo:
    sc = item['scenario']
    code = item['status_code']
    resp = item['response']
    model_u = resp.get('model_used', 'N/A (Error)')
    pred_p = resp.get('percentage_risk', 'N/A')
    fin_risk = resp.get('final_risk_level', 'N/A')
    notes = resp.get('escalation_notes', resp.get('detail', 'N/A'))
    if isinstance(notes, list):
        notes = notes[0].get('msg', str(notes))
    demo_rows.append({
        'Scenario': sc,
        'HTTP Status': code,
        'Model Routed': model_u,
        'Predicted Risk': pred_p,
        'Final Risk Tier': fin_risk,
        'Escalation / Error Notes': notes
    })

demo_df = pd.DataFrame(demo_rows)
display(HTML(demo_df.to_html(index=False, classes='table table-bordered table-striped table-hover')))""")

# 13. Section 23: Combined Clinical Limitations
cell_23_md = nbf.v4.new_markdown_cell("""## 23. Combined Clinical Limitations & Future Directions
1. **Scope and Objectives**: Tier 1 is an accessible non-invasive population screening tool. Tier 2 is a lab-augmented clinical risk stratification tool. Because they answer different clinical questions, their metrics must not be conflated.
2. **External Validation Asymmetry**: Tier 1 is validated on CDC BRFSS ($N = 249,049$). Tier 2 cannot be evaluated on BRFSS due to absence of laboratory blood draws in telephone surveys.
3. **Cross-Sectional Prevalence**: Both models estimate current prevalence, not 10-year incident risk. Prospective validation on longitudinal cohorts (e.g. UK Biobank) is recommended for prospective clinical adoption.
4. **Label Noise Integrity**: Ground truth is medical self-report (`DIQ010`), creating structural divergence for pharmacologically treated individuals.""")

# Append all cells
new_cells = [
    cell_b_header,
    cell_12_md,
    cell_13_md, cell_13_code,
    cell_14_md, cell_14_code,
    cell_15_md, cell_15_code,
    cell_16_md, cell_16_code,
    cell_17_md, cell_17_code,
    cell_18_md, cell_18_code,
    cell_19_md, cell_19_code,
    cell_20_md, cell_20_code,
    cell_21_md, cell_21_code,
    cell_22_md, cell_22_code,
    cell_23_md
]

for c in new_cells:
    nb.cells.append(c)

new_cell_count = len(nb.cells)
nbf.write(nb, nb_path)
print(f"Successfully appended Part B! Cell count: {orig_cell_count} -> {new_cell_count} (+{new_cell_count - orig_cell_count} new cells)")
