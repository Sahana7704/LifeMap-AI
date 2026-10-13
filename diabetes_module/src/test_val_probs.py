"""
Compute validation probabilities and thresholds for dual operating points.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, recall_score
from train_and_evaluate import TabularPreprocessor, ProbabilityCalibrator, DiabetesMLP

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODULE_DIR = ROOT_DIR / "diabetes_module"
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = MODULE_DIR / "models"
REPORTS_DIR = MODULE_DIR / "reports"

nhanes_df = pd.read_csv(DATA_DIR / "nhanes_diabetes_processed.csv")
with open(MODELS_DIR / "split_indices.json") as f:
    splits = json.load(f)

val_idx = splits.get("val_idx", splits.get("val_indices"))
y_val = nhanes_df.loc[val_idx, "diabetes"].values

prep_full = joblib.load(MODELS_DIR / "preprocessor_full.joblib")
prep_shared = joblib.load(MODELS_DIR / "preprocessor_shared.joblib")

full_cols = [c for c in nhanes_df.columns if c not in ["SEQN", "diabetes"]]
shared_cols = ["age_bucket", "sex", "bmi", "high_bp", "high_chol", "smoker", "phys_activity"]

X_val_full = prep_full.transform(nhanes_df.loc[val_idx, full_cols])
X_val_shared = prep_shared.transform(nhanes_df.loc[val_idx, shared_cols])

# 1. Final Calibrated XGBoost
cal_xgb_full = joblib.load(MODELS_DIR / "final_calibrated_xgb_full.joblib")
cal_xgb_shared = joblib.load(MODELS_DIR / "final_calibrated_xgb_shared.joblib")

prob_cal_xgb_val_full = cal_xgb_full.predict_proba(X_val_full)[:, 1]
prob_cal_xgb_val_shared = cal_xgb_shared.predict_proba(X_val_shared)[:, 1]

# 2. Base models for Stacking Ensemble
lr_full = joblib.load(MODELS_DIR / "lr_full.joblib")
rf_full = joblib.load(MODELS_DIR / "rf_full.joblib")
xgb_full = joblib.load(MODELS_DIR / "xgb_full.joblib")
svm_full_dict = joblib.load(MODELS_DIR / "svm_full.joblib")
prob_svm_val_full = np.clip(svm_full_dict["iso"].predict(svm_full_dict["svm"].decision_function(X_val_full)), 1e-4, 1-1e-4)
mlp_full = DiabetesMLP(input_dim=10)
mlp_full.load_state_dict(torch.load(MODELS_DIR / "mlp_full.pt"))
mlp_full.eval()

lr_shared = joblib.load(MODELS_DIR / "lr_shared.joblib")
rf_shared = joblib.load(MODELS_DIR / "rf_shared.joblib")
xgb_shared = joblib.load(MODELS_DIR / "xgb_shared.joblib")
svm_shared_dict = joblib.load(MODELS_DIR / "svm_shared.joblib")
prob_svm_val_shared = np.clip(svm_shared_dict["iso"].predict(svm_shared_dict["svm"].decision_function(X_val_shared)), 1e-4, 1-1e-4)
mlp_shared = DiabetesMLP(input_dim=7)
mlp_shared.load_state_dict(torch.load(MODELS_DIR / "mlp_shared.pt"))
mlp_shared.eval()

def predict_mlp(model, X_mat):
    with torch.no_grad():
        logits = model(torch.tensor(X_mat, dtype=torch.float32)).squeeze().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

# Stacking models
stack_full = joblib.load(MODELS_DIR / "stacking_full.joblib")
stack_shared = joblib.load(MODELS_DIR / "stacking_shared.joblib")

# Meta features for full validation
val_meta_full = np.column_stack([
    lr_full.predict_proba(X_val_full)[:, 1],
    rf_full.predict_proba(X_val_full)[:, 1],
    xgb_full.predict_proba(X_val_full)[:, 1],
    prob_svm_val_full,
    predict_mlp(mlp_full, X_val_full)
])
prob_stack_val_full = stack_full.predict_proba(val_meta_full)[:, 1]

# Meta features for shared validation
val_meta_shared = np.column_stack([
    lr_shared.predict_proba(X_val_shared)[:, 1],
    rf_shared.predict_proba(X_val_shared)[:, 1],
    xgb_shared.predict_proba(X_val_shared)[:, 1],
    prob_svm_val_shared,
    predict_mlp(mlp_shared, X_val_shared)
])
prob_stack_val_shared = stack_shared.predict_proba(val_meta_shared)[:, 1]

print("Validation evaluation shapes:")
print("y_val:", y_val.shape, "Prevalence:", y_val.mean())
print("prob_cal_xgb_val_full:", prob_cal_xgb_val_full.shape, "mean:", prob_cal_xgb_val_full.mean())
print("prob_stack_val_full:", prob_stack_val_full.shape, "mean:", prob_stack_val_full.mean())
print("prob_cal_xgb_val_shared:", prob_cal_xgb_val_shared.shape, "mean:", prob_cal_xgb_val_shared.mean())
print("prob_stack_val_shared:", prob_stack_val_shared.shape, "mean:", prob_stack_val_shared.mean())

def find_thresholds(y_true, y_prob):
    # Grid of candidate thresholds
    thresholds = np.linspace(0.01, 0.99, 981)
    
    # (a) Max Accuracy
    best_acc = -1
    best_th_acc = 0.5
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_th_acc = th
            
    # (b) Screening: 80% recall
    # Find highest threshold with recall >= 0.80
    valid_th_recall = []
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= 0.80:
            valid_th_recall.append((th, rec))
            
    if valid_th_recall:
        best_th_rec = max(valid_th_recall, key=lambda x: x[0])[0]
        rec_at_th = [r for t, r in valid_th_recall if t == best_th_rec][0]
    else:
        best_th_rec = 0.5
        rec_at_th = 0.0
        
    return best_th_acc, best_acc, best_th_rec, rec_at_th

th_acc_full_xgb, acc_full_xgb, th_rec_full_xgb, rec_full_xgb = find_thresholds(y_val, prob_cal_xgb_val_full)
th_acc_full_stk, acc_full_stk, th_rec_full_stk, rec_full_stk = find_thresholds(y_val, prob_stack_val_full)

th_acc_sh_xgb, acc_sh_xgb, th_rec_sh_xgb, rec_sh_xgb = find_thresholds(y_val, prob_cal_xgb_val_shared)
th_acc_sh_stk, acc_sh_stk, th_rec_sh_stk, rec_sh_stk = find_thresholds(y_val, prob_stack_val_shared)

print("\n--- Thresholds Computed on Validation Set ---")
print(f"Full Features - Calibrated XGBoost:")
print(f"  Max Acc Threshold: {th_acc_full_xgb:.3f} (Val Acc: {acc_full_xgb:.4f})")
print(f"  80% Recall Threshold: {th_rec_full_xgb:.3f} (Val Recall: {rec_full_xgb:.4f})")
print(f"Full Features - Stacking Ensemble:")
print(f"  Max Acc Threshold: {th_acc_full_stk:.3f} (Val Acc: {acc_full_stk:.4f})")
print(f"  80% Recall Threshold: {th_rec_full_stk:.3f} (Val Recall: {rec_full_stk:.4f})")

print(f"\nShared-7 Features - Calibrated XGBoost:")
print(f"  Max Acc Threshold: {th_acc_sh_xgb:.3f} (Val Acc: {acc_sh_xgb:.4f})")
print(f"  80% Recall Threshold: {th_rec_sh_xgb:.3f} (Val Recall: {rec_sh_xgb:.4f})")
print(f"Shared-7 Features - Stacking Ensemble:")
print(f"  Max Acc Threshold: {th_acc_sh_stk:.3f} (Val Acc: {acc_sh_stk:.4f})")
print(f"  80% Recall Threshold: {th_rec_sh_stk:.3f} (Val Recall: {rec_sh_stk:.4f})")
