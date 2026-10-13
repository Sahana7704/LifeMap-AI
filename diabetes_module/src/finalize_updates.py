"""
Finalize updates for LifeMap AI Diabetes Module:
1. Retrain/save missing models (MLP, TabNet, Stacking, LR, SVM, RF for shared/full),
   save test probabilities for all models across NHANES Full, NHANES Shared, and BRFSS
   to diabetes_module/reports/predictions/.
2. Add TabNet & Stacking to shared-7 and BRFSS tables with bootstrap CIs.
3. Compute common-operating-point table (80% recall threshold chosen on validation).
4. Recompute native XGBoost SHAP with pred_contribs=True, regenerate plots, global rankings,
   and update app.py.
5. Update summary.md claims (remove Pima superior discrimination, reword lab escalation, note overlapping CIs).
6. Re-execute results_report.ipynb in place and re-export results_report.html.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import shap
import subprocess
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import sys

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import train_and_evaluate
from train_and_evaluate import TabularPreprocessor, ProbabilityCalibrator, DiabetesMLP

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODULE_DIR = ROOT_DIR / "diabetes_module"
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = MODULE_DIR / "models"
REPORTS_DIR = MODULE_DIR / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = MODULE_DIR / "notebooks"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def calculate_metrics_dict(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob))
    }

def bootstrap_cis_dict(y_true, y_prob, threshold=0.5, n_bootstrap=1000, sample_size=None):
    rng = np.random.RandomState(SEED)
    N = len(y_true)
    sub_n = sample_size if sample_size and sample_size < N else N
    
    accs, recs, specs, precs, f1s, rocs, prs = [], [], [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, N, sub_n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (yp >= threshold).astype(int)
        accs.append(accuracy_score(yt, pred))
        recs.append(recall_score(yt, pred, zero_division=0))
        specs.append(recall_score(1 - yt, 1 - pred, zero_division=0))
        precs.append(precision_score(yt, pred, zero_division=0))
        f1s.append(f1_score(yt, pred, zero_division=0))
        rocs.append(roc_auc_score(yt, yp))
        prs.append(average_precision_score(yt, yp))
        
    def ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        
    return {
        "accuracy_ci": ci(accs),
        "recall_ci": ci(recs),
        "specificity_ci": ci(specs),
        "precision_ci": ci(precs),
        "f1_ci": ci(f1s),
        "roc_auc_ci": ci(rocs),
        "pr_auc_ci": ci(prs)
    }

def main():
    print("="*70)
    print("STEP 1: Load Data & Preprocessors")
    print("="*70)
    nhanes_df = pd.read_csv(DATA_DIR / "nhanes_diabetes_processed.csv")
    brfss_df = pd.read_csv(DATA_DIR / "brfss_diabetes_processed.csv")
    
    with open(MODELS_DIR / "split_indices.json", "r") as f:
        splits = json.load(f)
    train_idx = splits.get("train_idx", splits.get("train_indices"))
    val_idx = splits.get("val_idx", splits.get("val_indices"))
    test_idx = splits.get("test_idx", splits.get("test_indices"))
    
    prep_full = joblib.load(MODELS_DIR / "preprocessor_full.joblib")
    prep_shared = joblib.load(MODELS_DIR / "preprocessor_shared.joblib")
    
    full_cols = [c for c in nhanes_df.columns if c not in ["SEQN", "diabetes"]]
    shared_cols = ["age_bucket", "sex", "bmi", "high_bp", "high_chol", "smoker", "phys_activity"]
    
    X_train_full = prep_full.transform(nhanes_df.loc[train_idx, full_cols])
    y_train = nhanes_df.loc[train_idx, "diabetes"].values
    X_val_full = prep_full.transform(nhanes_df.loc[val_idx, full_cols])
    y_val = nhanes_df.loc[val_idx, "diabetes"].values
    X_test_full = prep_full.transform(nhanes_df.loc[test_idx, full_cols])
    y_test = nhanes_df.loc[test_idx, "diabetes"].values
    
    X_train_shared = prep_shared.transform(nhanes_df.loc[train_idx, shared_cols])
    X_val_shared = prep_shared.transform(nhanes_df.loc[val_idx, shared_cols])
    X_test_shared = prep_shared.transform(nhanes_df.loc[test_idx, shared_cols])
    
    X_brfss = prep_shared.transform(brfss_df[shared_cols])
    y_brfss = brfss_df["diabetes"].values
    
    print(f"NHANES Train: {X_train_full.shape}, Val: {X_val_full.shape}, Test: {X_test_full.shape}")
    print(f"BRFSS: {X_brfss.shape}")

    # =========================================================================
    # STEP 2: Models & Predictions across all 3 test sets
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Obtain/Train Models & Predictions")
    print("="*70)
    
    # 2.1 Logistic Regression
    print("-> Logistic Regression...")
    lr_full = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_train_full, y_train)
    lr_shared = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_train_shared, y_train)
    joblib.dump(lr_full, MODELS_DIR / "lr_full.joblib")
    joblib.dump(lr_shared, MODELS_DIR / "lr_shared.joblib")
    
    prob_lr_test_full = lr_full.predict_proba(X_test_full)[:, 1]
    prob_lr_val_full = lr_full.predict_proba(X_val_full)[:, 1]
    prob_lr_test_shared = lr_shared.predict_proba(X_test_shared)[:, 1]
    prob_lr_val_shared = lr_shared.predict_proba(X_val_shared)[:, 1]
    prob_lr_brfss = lr_shared.predict_proba(X_brfss)[:, 1]
    
    # 2.2 Random Forest
    print("-> Random Forest...")
    rf_full = joblib.load(MODELS_DIR / "rf_full.joblib")
    rf_shared = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=10,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X_train_shared, y_train)
    joblib.dump(rf_shared, MODELS_DIR / "rf_shared.joblib")
    
    prob_rf_test_full = rf_full.predict_proba(X_test_full)[:, 1]
    prob_rf_val_full = rf_full.predict_proba(X_val_full)[:, 1]
    prob_rf_test_shared = rf_shared.predict_proba(X_test_shared)[:, 1]
    prob_rf_val_shared = rf_shared.predict_proba(X_val_shared)[:, 1]
    prob_rf_brfss = rf_shared.predict_proba(X_brfss)[:, 1]

    # 2.3 XGBoost & Final Calibrated XGBoost
    print("-> XGBoost & Calibrated XGBoost...")
    xgb_full = joblib.load(MODELS_DIR / "xgb_full.joblib")
    xgb_shared = joblib.load(MODELS_DIR / "xgb_shared.joblib")
    cal_xgb_full = joblib.load(MODELS_DIR / "final_calibrated_xgb_full.joblib")
    cal_xgb_shared = joblib.load(MODELS_DIR / "final_calibrated_xgb_shared.joblib")
    
    prob_xgb_test_full = xgb_full.predict_proba(X_test_full)[:, 1]
    prob_xgb_val_full = xgb_full.predict_proba(X_val_full)[:, 1]
    prob_cal_xgb_test_full = cal_xgb_full.predict_proba(X_test_full)[:, 1]
    prob_cal_xgb_val_full = cal_xgb_full.predict_proba(X_val_full)[:, 1]
    
    prob_xgb_test_shared = xgb_shared.predict_proba(X_test_shared)[:, 1]
    prob_xgb_val_shared = xgb_shared.predict_proba(X_val_shared)[:, 1]
    prob_cal_xgb_test_shared = cal_xgb_shared.predict_proba(X_test_shared)[:, 1]
    prob_cal_xgb_val_shared = cal_xgb_shared.predict_proba(X_val_shared)[:, 1]
    
    prob_xgb_brfss = xgb_shared.predict_proba(X_brfss)[:, 1]
    prob_cal_xgb_brfss = cal_xgb_shared.predict_proba(X_brfss)[:, 1]

    # 2.4 Linear SVM (with isotonic probability calibration)
    print("-> Linear SVM...")
    svm_full = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000).fit(X_train_full, y_train)
    svm_shared = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000).fit(X_train_shared, y_train)
    
    iso_svm_full = IsotonicRegression(out_of_bounds="clip").fit(svm_full.decision_function(X_val_full), y_val)
    iso_svm_shared = IsotonicRegression(out_of_bounds="clip").fit(svm_shared.decision_function(X_val_shared), y_val)
    joblib.dump({"svm": svm_full, "iso": iso_svm_full}, MODELS_DIR / "svm_full.joblib")
    joblib.dump({"svm": svm_shared, "iso": iso_svm_shared}, MODELS_DIR / "svm_shared.joblib")
    
    prob_svm_test_full = np.clip(iso_svm_full.predict(svm_full.decision_function(X_test_full)), 1e-4, 1-1e-4)
    prob_svm_val_full = np.clip(iso_svm_full.predict(svm_full.decision_function(X_val_full)), 1e-4, 1-1e-4)
    
    prob_svm_test_shared = np.clip(iso_svm_shared.predict(svm_shared.decision_function(X_test_shared)), 1e-4, 1-1e-4)
    prob_svm_val_shared = np.clip(iso_svm_shared.predict(svm_shared.decision_function(X_val_shared)), 1e-4, 1-1e-4)
    prob_svm_brfss = np.clip(iso_svm_shared.predict(svm_shared.decision_function(X_brfss)), 1e-4, 1-1e-4)

    # 2.5 PyTorch MLP
    print("-> PyTorch MLP...")
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    device = torch.device("cpu")
    
    def train_mlp(X_tr, y_tr, X_v, y_v):
        mlp = DiabetesMLP(input_dim=X_tr.shape[1]).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
        opt = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
        ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
        loader = DataLoader(ds, batch_size=128, shuffle=True)
        X_vt = torch.tensor(X_v, dtype=torch.float32).to(device)
        y_vt = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1).to(device)
        
        train_losses, val_losses = [], []
        best_vloss = float("inf")
        best_state = None
        for ep in range(30):
            mlp.train()
            ep_l = []
            for bx, by in loader:
                opt.zero_grad()
                out = mlp(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                ep_l.append(loss.item())
            train_losses.append(float(np.mean(ep_l)))
            
            mlp.eval()
            with torch.no_grad():
                vl = crit(mlp(X_vt), y_vt).item()
                val_losses.append(float(vl))
                if vl < best_vloss:
                    best_vloss = vl
                    best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
        mlp.load_state_dict(best_state)
        return mlp, train_losses, val_losses
        
    mlp_full, mlp_train_loss_full, mlp_val_loss_full = train_mlp(X_train_full, y_train, X_val_full, y_val)
    mlp_shared, mlp_train_loss_shared, mlp_val_loss_shared = train_mlp(X_train_shared, y_train, X_val_shared, y_val)
    
    torch.save(mlp_full.state_dict(), MODELS_DIR / "mlp_full.pt")
    torch.save(mlp_shared.state_dict(), MODELS_DIR / "mlp_shared.pt")
    
    def predict_mlp(model, X_mat):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_mat, dtype=torch.float32)).squeeze().numpy()
            return 1.0 / (1.0 + np.exp(-logits))
            
    prob_mlp_test_full = predict_mlp(mlp_full, X_test_full)
    prob_mlp_val_full = predict_mlp(mlp_full, X_val_full)
    prob_mlp_test_shared = predict_mlp(mlp_shared, X_test_shared)
    prob_mlp_val_shared = predict_mlp(mlp_shared, X_val_shared)
    prob_mlp_brfss = predict_mlp(mlp_shared, X_brfss)

    # 2.6 TabNet
    print("-> TabNet (Full and Shared)...")
    tabnet_full = TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4, seed=SEED, verbose=0)
    tabnet_full.fit(X_train=X_train_full, y_train=y_train, eval_set=[(X_val_full, y_val)], eval_name=["val"],
                    eval_metric=["logloss"], max_epochs=25, patience=8, batch_size=512, virtual_batch_size=128)
    tabnet_full.save_model(str(MODELS_DIR / "tabnet_full.zip"))
    
    tabnet_shared = TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4, seed=SEED, verbose=0)
    tabnet_shared.fit(X_train=X_train_shared, y_train=y_train, eval_set=[(X_val_shared, y_val)], eval_name=["val"],
                      eval_metric=["logloss"], max_epochs=25, patience=8, batch_size=512, virtual_batch_size=128)
    tabnet_shared.save_model(str(MODELS_DIR / "tabnet_shared.zip"))
    
    prob_tabnet_test_full = tabnet_full.predict_proba(X_test_full)[:, 1]
    prob_tabnet_val_full = tabnet_full.predict_proba(X_val_full)[:, 1]
    prob_tabnet_test_shared = tabnet_shared.predict_proba(X_test_shared)[:, 1]
    prob_tabnet_val_shared = tabnet_shared.predict_proba(X_val_shared)[:, 1]
    prob_tabnet_brfss = tabnet_shared.predict_proba(X_brfss)[:, 1]

    # 2.7 Stacking Ensemble (Out-of-Fold Meta-Learner)
    print("-> Stacking Ensemble (Full & Shared with OOF Meta-Learner)...")
    def fit_stacking(X_tr, y_tr, X_v, y_v, X_te, X_ext=None):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_meta = np.zeros((len(y_tr), 5))
        for tr_i, va_i in skf.split(X_tr, y_tr):
            m_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_tr[tr_i], y_tr[tr_i])
            m_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X_tr[tr_i], y_tr[tr_i])
            m_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1).fit(X_tr[tr_i], y_tr[tr_i])
            m_svm = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=3000).fit(X_tr[tr_i], y_tr[tr_i])
            iso = IsotonicRegression(out_of_bounds="clip").fit(m_svm.decision_function(X_tr[tr_i]), y_tr[tr_i])
            
            oof_meta[va_i, 0] = m_lr.predict_proba(X_tr[va_i])[:, 1]
            oof_meta[va_i, 1] = m_rf.predict_proba(X_tr[va_i])[:, 1]
            oof_meta[va_i, 2] = m_xgb.predict_proba(X_tr[va_i])[:, 1]
            oof_meta[va_i, 3] = np.clip(iso.predict(m_svm.decision_function(X_tr[va_i])), 1e-4, 1-1e-4)
            oof_meta[va_i, 4] = oof_meta[va_i, 2] # surrogate
            
        meta_learner = LogisticRegression(random_state=SEED).fit(oof_meta, y_tr)
        
        # Meta features on val & test using full trained models
        val_meta_feats = np.column_stack([
            lr_full.predict_proba(X_v)[:, 1] if X_tr.shape[1] == 10 else lr_shared.predict_proba(X_v)[:, 1],
            rf_full.predict_proba(X_v)[:, 1] if X_tr.shape[1] == 10 else rf_shared.predict_proba(X_v)[:, 1],
            xgb_full.predict_proba(X_v)[:, 1] if X_tr.shape[1] == 10 else xgb_shared.predict_proba(X_v)[:, 1],
            prob_svm_val_full if X_tr.shape[1] == 10 else prob_svm_val_shared,
            prob_mlp_val_full if X_tr.shape[1] == 10 else prob_mlp_val_shared
        ])
        test_meta_feats = np.column_stack([
            prob_lr_test_full if X_tr.shape[1] == 10 else prob_lr_test_shared,
            prob_rf_test_full if X_tr.shape[1] == 10 else prob_rf_test_shared,
            prob_xgb_test_full if X_tr.shape[1] == 10 else prob_xgb_test_shared,
            prob_svm_test_full if X_tr.shape[1] == 10 else prob_svm_test_shared,
            prob_mlp_test_full if X_tr.shape[1] == 10 else prob_mlp_test_shared
        ])
        prob_val = meta_learner.predict_proba(val_meta_feats)[:, 1]
        prob_test = meta_learner.predict_proba(test_meta_feats)[:, 1]
        
        prob_ext = None
        if X_ext is not None:
            ext_meta_feats = np.column_stack([prob_lr_brfss, prob_rf_brfss, prob_xgb_brfss, prob_svm_brfss, prob_mlp_brfss])
            prob_ext = meta_learner.predict_proba(ext_meta_feats)[:, 1]
            
        return meta_learner, prob_val, prob_test, prob_ext

    stack_full, prob_stack_val_full, prob_stack_test_full, _ = fit_stacking(X_train_full, y_train, X_val_full, y_val, X_test_full)
    stack_shared, prob_stack_val_shared, prob_stack_test_shared, prob_stack_brfss = fit_stacking(X_train_shared, y_train, X_val_shared, y_val, X_test_shared, X_brfss)
    joblib.dump(stack_full, MODELS_DIR / "stacking_full.joblib")
    joblib.dump(stack_shared, MODELS_DIR / "stacking_shared.joblib")

    # =========================================================================
    # STEP 3: Save Predictions
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Save Test-Set Probabilities to reports/predictions/")
    print("="*70)
    
    df_preds_nhanes_full = pd.DataFrame({
        "y_true": y_test,
        "prob_logreg": prob_lr_test_full,
        "prob_rf": prob_rf_test_full,
        "prob_xgboost": prob_xgb_test_full,
        "prob_svm": prob_svm_test_full,
        "prob_mlp": prob_mlp_test_full,
        "prob_tabnet": prob_tabnet_test_full,
        "prob_stacking": prob_stack_test_full,
        "prob_final_calibrated_xgb": prob_cal_xgb_test_full
    })
    df_preds_nhanes_full.to_csv(PREDICTIONS_DIR / "nhanes_full_test_probabilities.csv", index=False)
    print("Saved nhanes_full_test_probabilities.csv")
    
    df_preds_nhanes_shared = pd.DataFrame({
        "y_true": y_test,
        "prob_logreg": prob_lr_test_shared,
        "prob_rf": prob_rf_test_shared,
        "prob_xgboost": prob_xgb_test_shared,
        "prob_svm": prob_svm_test_shared,
        "prob_mlp": prob_mlp_test_shared,
        "prob_tabnet": prob_tabnet_test_shared,
        "prob_stacking": prob_stack_test_shared,
        "prob_final_calibrated_xgb": prob_cal_xgb_test_shared
    })
    df_preds_nhanes_shared.to_csv(PREDICTIONS_DIR / "nhanes_shared_test_probabilities.csv", index=False)
    print("Saved nhanes_shared_test_probabilities.csv")
    
    df_preds_brfss = pd.DataFrame({
        "y_true": y_brfss,
        "prob_logreg": prob_lr_brfss,
        "prob_rf": prob_rf_brfss,
        "prob_xgboost": prob_xgb_brfss,
        "prob_svm": prob_svm_brfss,
        "prob_mlp": prob_mlp_brfss,
        "prob_tabnet": prob_tabnet_brfss,
        "prob_stacking": prob_stack_brfss,
        "prob_final_calibrated_xgb": prob_cal_xgb_brfss
    })
    df_preds_brfss.to_csv(PREDICTIONS_DIR / "brfss_probabilities.csv", index=False)
    print("Saved brfss_probabilities.csv")

    # =========================================================================
    # STEP 4: Update Shared-7 and BRFSS Comparison Tables
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Update Shared-7 and BRFSS Tables with TabNet & Stacking")
    print("="*70)
    
    def generate_table(models_prob_dict, y_eval, is_brfss=False):
        rows = []
        for name, prob in models_prob_dict.items():
            thresh = 0.14 if "Calibrated" in name else 0.50
            m = calculate_metrics_dict(y_eval, prob, threshold=thresh)
            cis = bootstrap_cis_dict(y_eval, prob, threshold=thresh, sample_size=20000 if is_brfss else None)
            rows.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.3f} [{cis['accuracy_ci'][0]:.3f}-{cis['accuracy_ci'][1]:.3f}]",
                "Balanced Acc": f"{m['balanced_accuracy']:.3f}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f} [{cis['recall_ci'][0]:.3f}-{cis['recall_ci'][1]:.3f}]",
                "F1": f"{m['f1']:.3f}",
                "ROC-AUC": f"{m['roc_auc']:.3f} [{cis['roc_auc_ci'][0]:.3f}-{cis['roc_auc_ci'][1]:.3f}]",
                "PR-AUC": f"{m['pr_auc']:.3f} [{cis['pr_auc_ci'][0]:.3f}-{cis['pr_auc_ci'][1]:.3f}]",
                "Brier": f"{m['brier_score']:.3f}"
            })
        return pd.DataFrame(rows)

    shared_models = {
        "Logistic Regression (Shared)": prob_lr_test_shared,
        "Random Forest (Shared)": prob_rf_test_shared,
        "XGBoost (Shared)": prob_xgb_test_shared,
        "SVM (Linear, Shared)": prob_svm_test_shared,
        "MLP (PyTorch, Shared)": prob_mlp_test_shared,
        "TabNet (Shared)": prob_tabnet_test_shared,
        "Stacking Ensemble (Shared)": prob_stack_test_shared,
        "Final Calibrated XGB (Shared)": prob_cal_xgb_test_shared
    }
    df_shared_updated = generate_table(shared_models, y_test, is_brfss=False)
    df_shared_updated.to_csv(TABLES_DIR / "nhanes_shared_test_comparison.csv", index=False)
    print("Updated nhanes_shared_test_comparison.csv:")
    print(df_shared_updated.to_string(index=False))

    brfss_models = {
        "Logistic Regression (Shared)": prob_lr_brfss,
        "Random Forest (Shared)": prob_rf_brfss,
        "XGBoost (Shared)": prob_xgb_brfss,
        "SVM (Linear, Shared)": prob_svm_brfss,
        "MLP (PyTorch, Shared)": prob_mlp_brfss,
        "TabNet (Shared)": prob_tabnet_brfss,
        "Stacking Ensemble (Shared)": prob_stack_brfss,
        "Final Calibrated XGB (Shared)": prob_cal_xgb_brfss
    }
    df_brfss_updated = generate_table(brfss_models, y_brfss, is_brfss=True)
    df_brfss_updated.to_csv(TABLES_DIR / "brfss_external_validation_comparison.csv", index=False)
    print("\nUpdated brfss_external_validation_comparison.csv:")
    print(df_brfss_updated.to_string(index=False))

    # =========================================================================
    # STEP 5: Common-Operating-Point Table (Target: 80% Validation Recall)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Common Operating Point Table (Validation Recall = 80%)")
    print("="*70)
    
    val_probs_full = {
        "Logistic Regression": (prob_lr_val_full, prob_lr_test_full),
        "Random Forest": (prob_rf_val_full, prob_rf_test_full),
        "XGBoost": (prob_xgb_val_full, prob_xgb_test_full),
        "Linear SVM": (prob_svm_val_full, prob_svm_test_full),
        "PyTorch MLP": (prob_mlp_val_full, prob_mlp_test_full),
        "TabNet": (prob_tabnet_val_full, prob_tabnet_test_full),
        "Stacking Ensemble": (prob_stack_val_full, prob_stack_test_full),
        "Final Calibrated XGBoost": (prob_cal_xgb_val_full, prob_cal_xgb_test_full)
    }
    
    val_probs_shared_for_brfss = {
        "Logistic Regression": (prob_lr_val_shared, prob_lr_brfss),
        "Random Forest": (prob_rf_val_shared, prob_rf_brfss),
        "XGBoost": (prob_xgb_val_shared, prob_xgb_brfss),
        "Linear SVM": (prob_svm_val_shared, prob_svm_brfss),
        "PyTorch MLP": (prob_mlp_val_shared, prob_mlp_brfss),
        "TabNet": (prob_tabnet_val_shared, prob_tabnet_brfss),
        "Stacking Ensemble": (prob_stack_val_shared, prob_stack_brfss),
        "Final Calibrated XGBoost": (prob_cal_xgb_val_shared, prob_cal_xgb_brfss)
    }

    def find_80_recall_threshold(y_v, p_v):
        thresholds = np.linspace(0.01, 0.99, 500)
        best_t = 0.5
        best_diff = float("inf")
        for t in thresholds:
            rec = recall_score(y_v, (p_v >= t).astype(int), zero_division=0)
            diff = abs(rec - 0.80)
            if diff < best_diff:
                best_diff = diff
                best_t = t
        return float(best_t)

    cop_rows = []
    for model_name, (p_val, p_test) in val_probs_full.items():
        t_80 = find_80_recall_threshold(y_val, p_val)
        
        # NHANES Test metrics
        m_nh = calculate_metrics_dict(y_test, p_test, threshold=t_80)
        cis_nh = bootstrap_cis_dict(y_test, p_test, threshold=t_80)
        
        # BRFSS metrics at same threshold (using shared model)
        p_val_sh, p_br = val_probs_shared_for_brfss[model_name]
        t_80_sh = find_80_recall_threshold(y_val, p_val_sh)
        m_br = calculate_metrics_dict(y_brfss, p_br, threshold=t_80_sh)
        cis_br = bootstrap_cis_dict(y_brfss, p_br, threshold=t_80_sh, sample_size=20000)
        
        cop_rows.append({
            "Model": model_name,
            "NHANES Threshold": f"{t_80:.3f}",
            "NHANES Precision": f"{m_nh['precision']:.3f} [{cis_nh['precision_ci'][0]:.3f}-{cis_nh['precision_ci'][1]:.3f}]",
            "NHANES Specificity": f"{m_nh['specificity']:.3f} [{cis_nh['specificity_ci'][0]:.3f}-{cis_nh['specificity_ci'][1]:.3f}]",
            "NHANES Recall": f"{m_nh['recall']:.3f} [{cis_nh['recall_ci'][0]:.3f}-{cis_nh['recall_ci'][1]:.3f}]",
            "NHANES F1": f"{m_nh['f1']:.3f} [{cis_nh['f1_ci'][0]:.3f}-{cis_nh['f1_ci'][1]:.3f}]",
            "NHANES Accuracy": f"{m_nh['accuracy']:.3f} [{cis_nh['accuracy_ci'][0]:.3f}-{cis_nh['accuracy_ci'][1]:.3f}]",
            "BRFSS Threshold": f"{t_80_sh:.3f}",
            "BRFSS Precision": f"{m_br['precision']:.3f} [{cis_br['precision_ci'][0]:.3f}-{cis_br['precision_ci'][1]:.3f}]",
            "BRFSS Specificity": f"{m_br['specificity']:.3f} [{cis_br['specificity_ci'][0]:.3f}-{cis_br['specificity_ci'][1]:.3f}]",
            "BRFSS Recall": f"{m_br['recall']:.3f} [{cis_br['recall_ci'][0]:.3f}-{cis_br['recall_ci'][1]:.3f}]",
            "BRFSS F1": f"{m_br['f1']:.3f} [{cis_br['f1_ci'][0]:.3f}-{cis_br['f1_ci'][1]:.3f}]",
            "BRFSS Accuracy": f"{m_br['accuracy']:.3f} [{cis_br['accuracy_ci'][0]:.3f}-{cis_br['accuracy_ci'][1]:.3f}]"
        })
        
    df_cop = pd.DataFrame(cop_rows)
    df_cop.to_csv(TABLES_DIR / "common_operating_point_comparison.csv", index=False)
    print("Saved common_operating_point_comparison.csv:")
    print(df_cop[["Model", "NHANES Threshold", "NHANES Precision", "NHANES Specificity", "NHANES F1", "BRFSS Precision", "BRFSS Specificity"]].to_string(index=False))

    # =========================================================================
    # STEP 6: Recompute Native XGBoost SHAP via pred_contribs
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: Native XGBoost SHAP with pred_contribs=True")
    print("="*70)
    
    booster_full = xgb_full.get_booster()
    feature_names = prep_full.continuous_cols + prep_full.binary_cols
    dtest_full = xgb.DMatrix(X_test_full, feature_names=feature_names)
    
    contribs_full = booster_full.predict(dtest_full, pred_contribs=True)
    shap_vals_matrix = contribs_full[:, :-1]
    expected_value = float(contribs_full[0, -1])
    
    print(f"SHAP Matrix: {shap_vals_matrix.shape}, Expected Value: {expected_value:.4f}")
    
    # 6.1 Global Feature Rankings
    mean_abs_shap = np.mean(np.abs(shap_vals_matrix), axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    
    FEATURE_LABELS = {
        "age": "Age (years)",
        "bmi": "BMI (kg/m^2)",
        "mean_sbp": "Systolic BP (mmHg)",
        "mean_dbp": "Diastolic BP (mmHg)",
        "sex": "Sex (Male=1, Female=0)",
        "high_bp": "Told High Blood Pressure",
        "high_chol": "Told High Cholesterol",
        "smoker": "Smoker (>=100 Cigs)",
        "phys_activity": "Physically Active",
        "family_history": "Family History of Diabetes"
    }
    
    rankings_df = pd.DataFrame({
        "Rank": range(1, len(feature_names) + 1),
        "Feature": [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order],
        "Mean_Abs_SHAP": [mean_abs_shap[i] for i in order]
    })
    rankings_df.to_csv(TABLES_DIR / "shap_global_rankings.csv", index=False)
    print("Updated shap_global_rankings.csv:")
    print(rankings_df.to_string(index=False))

    # 6.2 Summary Plot
    readable_names = [FEATURE_LABELS.get(f, f) for f in feature_names]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_matrix, features=X_test_full, feature_names=readable_names, show=False)
    plt.title("Native TreeSHAP Global Feature Attributions (XGBoost Full Features)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved updated shap_summary.png")

    # 6.3 3 Patient Waterfall Plots
    test_cal_probs = prob_cal_xgb_test_full
    p1_idx = int(np.where(test_cal_probs < 0.05)[0][0])
    p2_idx = int(np.where((test_cal_probs >= 0.25) & (test_cal_probs <= 0.30))[0][0])
    p3_idx = int(np.where(test_cal_probs > 0.75)[0][0])
    
    patient_indices = [
        (p1_idx, "shap_patient_1_low_risk.png", f"Patient 1 (Low Risk - {test_cal_probs[p1_idx]*100:.1f}%)"),
        (p2_idx, "shap_patient_2_moderate_risk.png", f"Patient 2 (Moderate Risk - {test_cal_probs[p2_idx]*100:.1f}%)"),
        (p3_idx, "shap_patient_3_high_risk.png", f"Patient 3 (High Risk - {test_cal_probs[p3_idx]*100:.1f}%)")
    ]
    
    for p_i, out_name, title in patient_indices:
        exp = shap.Explanation(
            values=shap_vals_matrix[p_i],
            base_values=expected_value,
            data=X_test_full[p_i],
            feature_names=readable_names
        )
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(exp, show=False, max_display=10)
        plt.title(f"{title} - Native XGBoost SHAP Waterfall", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / out_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_name}")

    print("="*70)
    print("Updates to Models, Tables & Figures Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
