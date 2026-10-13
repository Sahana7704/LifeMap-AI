"""
Dual Operating-Point Analysis: High-Accuracy Mode vs Screening Mode.
Models: Final Calibrated XGBoost and Stacking Ensemble.
Cohorts: NHANES test (full features), NHANES test (shared-7), and CDC BRFSS (shared-7).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from train_and_evaluate import TabularPreprocessor, ProbabilityCalibrator, DiabetesMLP

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODULE_DIR = ROOT_DIR / "diabetes_module"
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = MODULE_DIR / "models"
REPORTS_DIR = MODULE_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.RandomState(SEED)

def get_validation_probabilities():
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

    # Calibrated XGBoost
    cal_xgb_full = joblib.load(MODELS_DIR / "final_calibrated_xgb_full.joblib")
    cal_xgb_shared = joblib.load(MODELS_DIR / "final_calibrated_xgb_shared.joblib")
    prob_cal_xgb_val_full = cal_xgb_full.predict_proba(X_val_full)[:, 1]
    prob_cal_xgb_val_shared = cal_xgb_shared.predict_proba(X_val_shared)[:, 1]

    # Stacking full meta-features
    lr_full = joblib.load(MODELS_DIR / "lr_full.joblib")
    rf_full = joblib.load(MODELS_DIR / "rf_full.joblib")
    xgb_full = joblib.load(MODELS_DIR / "xgb_full.joblib")
    svm_full_dict = joblib.load(MODELS_DIR / "svm_full.joblib")
    prob_svm_val_full = np.clip(svm_full_dict["iso"].predict(svm_full_dict["svm"].decision_function(X_val_full)), 1e-4, 1-1e-4)
    mlp_full = DiabetesMLP(input_dim=10)
    mlp_full.load_state_dict(torch.load(MODELS_DIR / "mlp_full.pt"))
    mlp_full.eval()

    with torch.no_grad():
        mlp_full_logits = mlp_full(torch.tensor(X_val_full, dtype=torch.float32)).squeeze().numpy()
        prob_mlp_val_full = 1.0 / (1.0 + np.exp(-mlp_full_logits))

    stack_full = joblib.load(MODELS_DIR / "stacking_full.joblib")
    val_meta_full = np.column_stack([
        lr_full.predict_proba(X_val_full)[:, 1],
        rf_full.predict_proba(X_val_full)[:, 1],
        xgb_full.predict_proba(X_val_full)[:, 1],
        prob_svm_val_full,
        prob_mlp_val_full
    ])
    prob_stack_val_full = stack_full.predict_proba(val_meta_full)[:, 1]

    # Stacking shared meta-features
    lr_shared = joblib.load(MODELS_DIR / "lr_shared.joblib")
    rf_shared = joblib.load(MODELS_DIR / "rf_shared.joblib")
    xgb_shared = joblib.load(MODELS_DIR / "xgb_shared.joblib")
    svm_shared_dict = joblib.load(MODELS_DIR / "svm_shared.joblib")
    prob_svm_val_shared = np.clip(svm_shared_dict["iso"].predict(svm_shared_dict["svm"].decision_function(X_val_shared)), 1e-4, 1-1e-4)
    mlp_shared = DiabetesMLP(input_dim=7)
    mlp_shared.load_state_dict(torch.load(MODELS_DIR / "mlp_shared.pt"))
    mlp_shared.eval()

    with torch.no_grad():
        mlp_shared_logits = mlp_shared(torch.tensor(X_val_shared, dtype=torch.float32)).squeeze().numpy()
        prob_mlp_val_shared = 1.0 / (1.0 + np.exp(-mlp_shared_logits))

    stack_shared = joblib.load(MODELS_DIR / "stacking_shared.joblib")
    val_meta_shared = np.column_stack([
        lr_shared.predict_proba(X_val_shared)[:, 1],
        rf_shared.predict_proba(X_val_shared)[:, 1],
        xgb_shared.predict_proba(X_val_shared)[:, 1],
        prob_svm_val_shared,
        prob_mlp_val_shared
    ])
    prob_stack_val_shared = stack_shared.predict_proba(val_meta_shared)[:, 1]

    return {
        "y_val": y_val,
        "cal_xgb_full": prob_cal_xgb_val_full,
        "stack_full": prob_stack_val_full,
        "cal_xgb_shared": prob_cal_xgb_val_shared,
        "stack_shared": prob_stack_val_shared
    }

def find_thresholds(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 981)
    
    # (a) High-Accuracy Mode: maximize validation accuracy
    best_acc = -1.0
    best_th_acc = 0.5
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_th_acc = th

    # (b) Screening Mode: highest threshold reaching at least 80% recall
    valid_th_recall = []
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= 0.80:
            valid_th_recall.append((th, rec))

    if valid_th_recall:
        best_th_rec = max(valid_th_recall, key=lambda x: x[0])[0]
    else:
        best_th_rec = 0.5

    return best_th_acc, best_th_rec

def evaluate_with_bootstrap(y_true, y_prob, threshold, n_bootstraps=1000, rng=RNG):
    y_true = np.asarray(y_true, dtype=int)
    pred = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    n = len(y_true)

    point_acc = (tp + tn) / n
    point_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    point_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    point_bacc = 0.5 * (point_rec + point_spec)
    point_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    point_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Fast, exact non-parametric bootstrap via Multinomial distribution
    p = np.array([tp, fp, tn, fn], dtype=float) / n
    draws = rng.multinomial(n, p, size=n_bootstraps) # shape (n_bootstraps, 4)

    boot_tp = draws[:, 0]
    boot_fp = draws[:, 1]
    boot_tn = draws[:, 2]
    boot_fn = draws[:, 3]

    boot_acc = (boot_tp + boot_tn) / n
    boot_rec = np.where((boot_tp + boot_fn) > 0, boot_tp / (boot_tp + boot_fn), 0.0)
    boot_spec = np.where((boot_tn + boot_fp) > 0, boot_tn / (boot_tn + boot_fp), 0.0)
    boot_bacc = 0.5 * (boot_rec + boot_spec)
    boot_prec = np.where((boot_tp + boot_fp) > 0, boot_tp / (boot_tp + boot_fp), 0.0)
    boot_f1 = np.where((2 * boot_tp + boot_fp + boot_fn) > 0, (2 * boot_tp) / (2 * boot_tp + boot_fp + boot_fn), 0.0)

    def ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    def ci_int(arr):
        return (int(np.round(np.percentile(arr, 2.5))), int(np.round(np.percentile(arr, 97.5))))

    return {
        "point": {
            "accuracy": point_acc,
            "balanced_acc": point_bacc,
            "precision": point_prec,
            "recall": point_rec,
            "specificity": point_spec,
            "f1": point_f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        },
        "ci": {
            "accuracy": ci(boot_acc),
            "balanced_acc": ci(boot_bacc),
            "precision": ci(boot_prec),
            "recall": ci(boot_rec),
            "specificity": ci(boot_spec),
            "f1": ci(boot_f1),
            "tp": ci_int(boot_tp),
            "fp": ci_int(boot_fp),
            "tn": ci_int(boot_tn),
            "fn": ci_int(boot_fn)
        }
    }

def format_val_ci(val, ci_tup, is_int=False):
    if is_int:
        return f"{val:,} [{ci_tup[0]:,}-{ci_tup[1]:,}]"
    else:
        return f"{val:.3f} [{ci_tup[0]:.3f}-{ci_tup[1]:.3f}]"

def main():
    print("=" * 70)
    print("STEP 1: Computing Validation Set Probabilities & Thresholds")
    print("=" * 70)
    val_data = get_validation_probabilities()
    y_val = val_data["y_val"]

    # Thresholds
    th_acc_full_xgb, th_rec_full_xgb = find_thresholds(y_val, val_data["cal_xgb_full"])
    th_acc_full_stk, th_rec_full_stk = find_thresholds(y_val, val_data["stack_full"])
    th_acc_sh_xgb, th_rec_sh_xgb = find_thresholds(y_val, val_data["cal_xgb_shared"])
    th_acc_sh_stk, th_rec_sh_stk = find_thresholds(y_val, val_data["stack_shared"])

    print(f"NHANES Full - Calibrated XGB: High-Acc={th_acc_full_xgb:.3f}, Screening(80% Rec)={th_rec_full_xgb:.3f}")
    print(f"NHANES Full - Stacking:       High-Acc={th_acc_full_stk:.3f}, Screening(80% Rec)={th_rec_full_stk:.3f}")
    print(f"NHANES Shared - Calibrated XGB: High-Acc={th_acc_sh_xgb:.3f}, Screening(80% Rec)={th_rec_sh_xgb:.3f}")
    print(f"NHANES Shared - Stacking:       High-Acc={th_acc_sh_stk:.3f}, Screening(80% Rec)={th_rec_sh_stk:.3f}")
    print("BRFSS Shared will reuse the NHANES Shared validation thresholds directly.")

    print("\n" + "=" * 70)
    print("STEP 2: Loading Saved Test & External Predictions")
    print("=" * 70)
    df_test_full = pd.read_csv(PREDICTIONS_DIR / "nhanes_full_test_probabilities.csv")
    df_test_shared = pd.read_csv(PREDICTIONS_DIR / "nhanes_shared_test_probabilities.csv")
    df_brfss = pd.read_csv(PREDICTIONS_DIR / "brfss_probabilities.csv")

    configs = [
        # NHANES Full Test
        {
            "cohort": "NHANES Test (Full Features)",
            "model_name": "Final Calibrated XGBoost",
            "mode": "High-Accuracy Mode",
            "threshold": th_acc_full_xgb,
            "y_true": df_test_full["y_true"].values,
            "y_prob": df_test_full["prob_final_calibrated_xgb"].values
        },
        {
            "cohort": "NHANES Test (Full Features)",
            "model_name": "Final Calibrated XGBoost",
            "mode": "Screening Mode (80% Recall)",
            "threshold": th_rec_full_xgb,
            "y_true": df_test_full["y_true"].values,
            "y_prob": df_test_full["prob_final_calibrated_xgb"].values
        },
        {
            "cohort": "NHANES Test (Full Features)",
            "model_name": "Stacking Ensemble",
            "mode": "High-Accuracy Mode",
            "threshold": th_acc_full_stk,
            "y_true": df_test_full["y_true"].values,
            "y_prob": df_test_full["prob_stacking"].values
        },
        {
            "cohort": "NHANES Test (Full Features)",
            "model_name": "Stacking Ensemble",
            "mode": "Screening Mode (80% Recall)",
            "threshold": th_rec_full_stk,
            "y_true": df_test_full["y_true"].values,
            "y_prob": df_test_full["prob_stacking"].values
        },
        # NHANES Shared-7 Test
        {
            "cohort": "NHANES Test (Shared-7)",
            "model_name": "Final Calibrated XGBoost",
            "mode": "High-Accuracy Mode",
            "threshold": th_acc_sh_xgb,
            "y_true": df_test_shared["y_true"].values,
            "y_prob": df_test_shared["prob_final_calibrated_xgb"].values
        },
        {
            "cohort": "NHANES Test (Shared-7)",
            "model_name": "Final Calibrated XGBoost",
            "mode": "Screening Mode (80% Recall)",
            "threshold": th_rec_sh_xgb,
            "y_true": df_test_shared["y_true"].values,
            "y_prob": df_test_shared["prob_final_calibrated_xgb"].values
        },
        {
            "cohort": "NHANES Test (Shared-7)",
            "model_name": "Stacking Ensemble",
            "mode": "High-Accuracy Mode",
            "threshold": th_acc_sh_stk,
            "y_true": df_test_shared["y_true"].values,
            "y_prob": df_test_shared["prob_stacking"].values
        },
        {
            "cohort": "NHANES Test (Shared-7)",
            "model_name": "Stacking Ensemble",
            "mode": "Screening Mode (80% Recall)",
            "threshold": th_rec_sh_stk,
            "y_true": df_test_shared["y_true"].values,
            "y_prob": df_test_shared["prob_stacking"].values
        },
        # CDC BRFSS 2015 External Validation (Reusing NHANES Shared validation thresholds)
        {
            "cohort": "CDC BRFSS (Shared-7)",
            "model_name": "Final Calibrated XGBoost",
            "mode": "High-Accuracy Mode",
            "threshold": th_acc_sh_xgb,
            "y_true": df_brfss["y_true"].values,
            "y_prob": df_brfss["prob_final_calibrated_xgb"].values
        },
        {
            "cohort": "CDC BRFSS (Shared-7)",
            "model_name": "Final Calibrated XGBoost",
            "mode": "Screening Mode (80% Recall)",
            "threshold": th_rec_sh_xgb,
            "y_true": df_brfss["y_true"].values,
            "y_prob": df_brfss["prob_final_calibrated_xgb"].values
        },
        {
            "cohort": "CDC BRFSS (Shared-7)",
            "model_name": "Stacking Ensemble",
            "mode": "High-Accuracy Mode",
            "threshold": th_acc_sh_stk,
            "y_true": df_brfss["y_true"].values,
            "y_prob": df_brfss["prob_stacking"].values
        },
        {
            "cohort": "CDC BRFSS (Shared-7)",
            "model_name": "Stacking Ensemble",
            "mode": "Screening Mode (80% Recall)",
            "threshold": th_rec_sh_stk,
            "y_true": df_brfss["y_true"].values,
            "y_prob": df_brfss["prob_stacking"].values
        }
    ]

    print("\n" + "=" * 70)
    print("STEP 3: Computing Metrics & Bootstrap CIs")
    print("=" * 70)
    rows = []
    cm_results = []
    for cfg in configs:
        res = evaluate_with_bootstrap(cfg["y_true"], cfg["y_prob"], cfg["threshold"], n_bootstraps=1000)
        pt = res["point"]
        ci = res["ci"]
        
        row = {
            "Cohort": cfg["cohort"],
            "Model": cfg["model_name"],
            "Operating Point": cfg["mode"],
            "Threshold": f"{cfg['threshold']:.3f}",
            "Accuracy": format_val_ci(pt["accuracy"], ci["accuracy"]),
            "Balanced Acc": format_val_ci(pt["balanced_acc"], ci["balanced_acc"]),
            "Precision": format_val_ci(pt["precision"], ci["precision"]),
            "Recall": format_val_ci(pt["recall"], ci["recall"]),
            "Specificity": format_val_ci(pt["specificity"], ci["specificity"]),
            "F1": format_val_ci(pt["f1"], ci["f1"]),
            "TP": format_val_ci(pt["tp"], ci["tp"], is_int=True),
            "FP": format_val_ci(pt["fp"], ci["fp"], is_int=True),
            "TN": format_val_ci(pt["tn"], ci["tn"], is_int=True),
            "FN": format_val_ci(pt["fn"], ci["fn"], is_int=True)
        }
        rows.append(row)
        cm_results.append({
            "cohort": cfg["cohort"],
            "model": cfg["model_name"],
            "mode": cfg["mode"],
            "threshold": cfg["threshold"],
            "cm": np.array([[pt["tn"], pt["fp"]], [pt["fn"], pt["tp"]]]),
            "pt": pt
        })

    out_df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "dual_operating_points.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved dual operating point table to {out_csv}")
    print("\nTable Preview:")
    print(out_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 4: Generating Confusion Matrices Figure")
    print("=" * 70)
    # Plot a 3x4 grid: 3 cohorts (rows) x 4 model-modes (cols)
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    cohort_names = ["NHANES Test (Full Features)", "NHANES Test (Shared-7)", "CDC BRFSS (Shared-7)"]

    for row_idx, c_name in enumerate(cohort_names):
        c_items = [item for item in cm_results if item["cohort"] == c_name]
        for col_idx, item in enumerate(c_items):
            ax = axes[row_idx, col_idx]
            cm = item["cm"]
            total = cm.sum()
            labels = np.array([
                [f"TN: {cm[0,0]:,}\n({cm[0,0]/total:.1%})", f"FP: {cm[0,1]:,}\n({cm[0,1]/total:.1%})"],
                [f"FN: {cm[1,0]:,}\n({cm[1,0]/total:.1%})", f"TP: {cm[1,1]:,}\n({cm[1,1]/total:.1%})"]
            ])
            sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 10, "weight": "bold"})
            
            short_model = "XGBoost" if "XGB" in item["model"] else "Stacking"
            short_mode = "High-Acc" if "High" in item["mode"] else "Screening (80% Rec)"
            ax.set_title(f"{short_model} | {short_mode}\n$\\tau={item['threshold']:.3f}$ | Rec: {item['pt']['recall']:.1%} | Acc: {item['pt']['accuracy']:.1%}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label", fontsize=10)
            ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=9)
            ax.set_yticklabels(["Non-Diabetic", "Diabetic"], fontsize=9)

        # Row label on left
        axes[row_idx, 0].annotate(
            c_name, xy=(0, 0.5), xytext=(-axes[row_idx, 0].yaxis.labelpad - 35, 0),
            xycoords=axes[row_idx, 0].yaxis.label, textcoords="offset points",
            size=12, ha="right", va="center", rotation=90, weight="bold", color="#1E3A8A"
        )

    plt.suptitle("Confusion Matrices: Dual Operating-Point Analysis Across Evaluation Cohorts", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    out_fig = FIGURES_DIR / "dual_operating_points_cm.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to {out_fig}")

if __name__ == "__main__":
    main()
