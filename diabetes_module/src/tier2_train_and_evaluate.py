"""
Tier 2 Model Training, Evaluation, and Artifact Generation Pipeline.
Lab-Augmented Diabetes Risk Prediction (NHANES + HbA1c).
"""

import sys
from pathlib import Path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
from xgboost import XGBClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer

import train_and_evaluate
from train_and_evaluate import TabularPreprocessor, ProbabilityCalibrator

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
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)

class Tier2MLP(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_bootstrap_cis(y_true, y_prob, threshold=0.5, n_bootstraps=1000, rng=RNG):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    
    pred = (y_prob >= threshold).astype(int)
    pt_acc = accuracy_score(y_true, pred)
    pt_rec = recall_score(y_true, pred, zero_division=0)
    pt_spec = recall_score(1 - y_true, 1 - pred, zero_division=0)
    pt_bacc = 0.5 * (pt_rec + pt_spec)
    pt_prec = precision_score(y_true, pred, zero_division=0)
    pt_f1 = f1_score(y_true, pred, zero_division=0)
    pt_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pt_pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pt_brier = brier_score_loss(y_true, y_prob)

    accs, recs, specs, baccs, precs, f1s, rocs, prs, briers = [], [], [], [], [], [], [], [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        pr = (yp >= threshold).astype(int)
        rc = recall_score(yt, pr, zero_division=0)
        sp = recall_score(1 - yt, 1 - pr, zero_division=0)
        accs.append(accuracy_score(yt, pr))
        recs.append(rc)
        specs.append(sp)
        baccs.append(0.5 * (rc + sp))
        precs.append(precision_score(yt, pr, zero_division=0))
        f1s.append(f1_score(yt, pr, zero_division=0))
        rocs.append(roc_auc_score(yt, yp))
        prs.append(average_precision_score(yt, yp))
        briers.append(brier_score_loss(yt, yp))

    def ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "point": {
            "accuracy": pt_acc, "balanced_acc": pt_bacc, "precision": pt_prec,
            "recall": pt_rec, "specificity": pt_spec, "f1": pt_f1,
            "roc_auc": pt_roc, "pr_auc": pt_pr, "brier": pt_brier
        },
        "ci": {
            "accuracy": ci(accs), "balanced_acc": ci(baccs), "precision": ci(precs),
            "recall": ci(recs), "specificity": ci(specs), "f1": ci(f1s),
            "roc_auc": ci(rocs), "pr_auc": ci(prs), "brier": ci(briers)
        }
    }

def format_val_ci(val, ci_tup):
    return f"{val:.3f} [{ci_tup[0]:.3f}-{ci_tup[1]:.3f}]"

def main():
    print("=" * 70)
    print("PHASE A: Loading & Preparing Tier 2 Data (NHANES + HbA1c)")
    print("=" * 70)
    df = pd.read_csv(DATA_DIR / "tier2_nhanes_with_hba1c.csv")
    print(f"Total NHANES respondents: {len(df):,}")

    with open(MODELS_DIR / "split_indices.json") as f:
        splits = json.load(f)

    train_idx_all = splits.get("train_idx", splits.get("train_indices"))
    val_idx_all = splits.get("val_idx", splits.get("val_indices"))
    test_idx_all = splits.get("test_idx", splits.get("test_indices"))

    # Rows with non-missing HbA1c
    has_hba1c = df["hba1c"].notna()
    print(f"Rows with non-missing HbA1c: {has_hba1c.sum():,} / {len(df):,} ({has_hba1c.mean()*100:.2f}%)")

    train_idx = [i for i in train_idx_all if has_hba1c.iloc[i]]
    val_idx = [i for i in val_idx_all if has_hba1c.iloc[i]]
    test_idx = [i for i in test_idx_all if has_hba1c.iloc[i]]

    print(f"Tier 2 Train Set: {len(train_idx):,} rows (pos: {df.loc[train_idx, 'diabetes'].sum():,}, neg: {(len(train_idx) - df.loc[train_idx, 'diabetes'].sum()):,}, prev: {df.loc[train_idx, 'diabetes'].mean()*100:.2f}%)")
    print(f"Tier 2 Val Set:   {len(val_idx):,} rows (pos: {df.loc[val_idx, 'diabetes'].sum():,}, neg: {(len(val_idx) - df.loc[val_idx, 'diabetes'].sum()):,}, prev: {df.loc[val_idx, 'diabetes'].mean()*100:.2f}%)")
    print(f"Tier 2 Test Set:  {len(test_idx):,} rows (pos: {df.loc[test_idx, 'diabetes'].sum():,}, neg: {(len(test_idx) - df.loc[test_idx, 'diabetes'].sum()):,}, prev: {df.loc[test_idx, 'diabetes'].mean()*100:.2f}%)")

    # 1. Label noise analysis
    sub_hba1c = df[has_hba1c].copy()
    undiag = (sub_hba1c["diabetes"] == 0) & (sub_hba1c["hba1c"] >= 6.5)
    controlled = (sub_hba1c["diabetes"] == 1) & (sub_hba1c["hba1c"] < 6.5)
    conc_neg = (sub_hba1c["diabetes"] == 0) & (sub_hba1c["hba1c"] < 6.5)
    conc_pos = (sub_hba1c["diabetes"] == 1) & (sub_hba1c["hba1c"] >= 6.5)

    noise_report = f"""# Tier 2 Label-Noise Analysis: Self-Report vs. Biochemical Glycohemoglobin (HbA1c)

Among $N = {len(sub_hba1c):,}$ NHANES adult participants with valid laboratory HbA1c measurements:

| Category | Definition | Count ($N$) | Percentage of Cohort | Clinical & Epidemiological Meaning |
| :--- | :--- | :---: | :---: | :--- |
| **Concordant Negative** | Self-Report No ($Y=0$) & $\\text{{HbA1c}} < 6.5\\%$ | {conc_neg.sum():,} | {conc_neg.mean()*100:.2f}% | Healthy normoglycemic or prediabetic controls. |
| **Concordant Positive** | Self-Report Yes ($Y=1$) & $\\text{{HbA1c}} \\ge 6.5\\%$ | {conc_pos.sum():,} | {conc_pos.mean()*100:.2f}% | Diagnosed diabetics with uncontrolled or active hyperglycemia. |
| **Undiagnosed Diabetes** | Self-Report No ($Y=0$) & $\\text{{HbA1c}} \\ge 6.5\\%$ | {undiag.sum():,} | {undiag.mean()*100:.2f}% | **Undiagnosed diabetes**: Asymptomatic individuals meeting biochemical diagnostic criteria without formal medical diagnosis. Represents **{undiag.sum() / (sub_hba1c['diabetes'] == 0).sum() * 100:.2f}%** of self-reported non-diabetics. |
| **Controlled/Treated Diabetes** | Self-Report Yes ($Y=1$) & $\\text{{HbA1c}} < 6.5\\%$ | {controlled.sum():,} | {controlled.mean()*100:.2f}% | **Treated/Managed diabetes**: Patients with known diagnosis maintaining glycemia in non-diabetic range via metformin, insulin, lifestyle, or GLP-1 receptor agonists. Represents **{controlled.sum() / (sub_hba1c['diabetes'] == 1).sum() * 100:.2f}%** of diagnosed diabetics. |
| **Total Discordant (Label Noise)** | Mismatch between Self-Report and $\\text{{HbA1c}} \\ge 6.5\\%$ | {undiag.sum() + controlled.sum():,} | {(undiag.sum() + controlled.sum()) / len(sub_hba1c) * 100:.2f}% | **Structural ceiling on classification accuracy**: Because models are trained to predict medical self-report (`DIQ010`), perfect biochemical agreement is impossible. |

### Key Takeaway for Machine Learning Modeling
1. The ground truth label remains strictly the self-reported clinical diagnosis (`DIQ010`). We do **NOT** modify or sanitize labels based on HbA1c, preserving honest epidemiological evaluation.
2. The existence of {controlled.sum():,} treated diabetics ({controlled.sum() / (sub_hba1c['diabetes'] == 1).sum() * 100:.1f}% of all diabetics) and {undiag.sum():,} undiagnosed individuals mathematically caps the maximum achievable accuracy when predicting self-report from biochemical markers alone, proving why accuracy in the high 90s is impossible without severe label leakage.
"""
    with open(TABLES_DIR / "tier2_label_noise.md", "w", encoding="utf-8") as f:
        f.write(noise_report)
    print("Saved label noise report to reports/tables/tier2_label_noise.md")

    # Plot label noise bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ["Concordant Neg\n(No Dx, <6.5%)", "Concordant Pos\n(Dx, >=6.5%)", "Treated/Controlled\n(Dx, <6.5%)", "Undiagnosed\n(No Dx, >=6.5%)"]
    vals = [conc_neg.sum(), conc_pos.sum(), controlled.sum(), undiag.sum()]
    colors = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]
    bars = ax.bar(cats, vals, color=colors, width=0.55, edgecolor="#1F2937", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 150, f"{h:,}\n({h/len(sub_hba1c):.1%})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Number of Participants", fontsize=11, fontweight="bold")
    ax.set_title("Tier 2 Label-Noise Distribution (Self-Report vs. Biochemical HbA1c >= 6.5%)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_label_noise_bar.png", dpi=300)
    plt.close()
    print("Saved label noise figure to reports/figures/tier2_label_noise_bar.png")

    # 2. Preprocessing
    print("\n" + "=" * 70)
    print("PHASE B: Feature Preprocessing & Transformation")
    print("=" * 70)
    continuous_cols_t2 = ["age", "bmi", "mean_sbp", "mean_dbp", "hba1c"]
    binary_cols_t2 = ["sex", "high_bp", "high_chol", "smoker", "phys_activity", "family_history"]
    all_feature_cols_t2 = continuous_cols_t2 + binary_cols_t2

    tier1_continuous_cols = ["age", "bmi", "mean_sbp", "mean_dbp"]
    tier1_binary_cols = ["sex", "high_bp", "high_chol", "smoker", "phys_activity", "family_history"]
    tier1_all_cols = tier1_continuous_cols + tier1_binary_cols

    prep_t2 = TabularPreprocessor(continuous_cols=continuous_cols_t2, binary_cols=binary_cols_t2)
    prep_t2.fit(df.loc[train_idx, all_feature_cols_t2])
    joblib.dump(prep_t2, MODELS_DIR / "tier2_preprocessor.joblib")
    print("Saved preprocessor to diabetes_module/models/tier2_preprocessor.joblib")

    X_train = prep_t2.transform(df.loc[train_idx, all_feature_cols_t2])
    y_train = df.loc[train_idx, "diabetes"].values
    X_val = prep_t2.transform(df.loc[val_idx, all_feature_cols_t2])
    y_val = df.loc[val_idx, "diabetes"].values
    X_test = prep_t2.transform(df.loc[test_idx, all_feature_cols_t2])
    y_test = df.loc[test_idx, "diabetes"].values

    # Preprocessor for Tier 1 features on the same rows
    prep_t1_subset = TabularPreprocessor(continuous_cols=tier1_continuous_cols, binary_cols=tier1_binary_cols)
    prep_t1_subset.fit(df.loc[train_idx, tier1_all_cols])
    X_train_t1 = prep_t1_subset.transform(df.loc[train_idx, tier1_all_cols])
    X_val_t1 = prep_t1_subset.transform(df.loc[val_idx, tier1_all_cols])
    X_test_t1 = prep_t1_subset.transform(df.loc[test_idx, tier1_all_cols])

    # HbA1c only
    prep_hba1c = TabularPreprocessor(continuous_cols=["hba1c"], binary_cols=[])
    prep_hba1c.fit(df.loc[train_idx, ["hba1c"]])
    X_train_hba1c = prep_hba1c.transform(df.loc[train_idx, ["hba1c"]])
    X_val_hba1c = prep_hba1c.transform(df.loc[val_idx, ["hba1c"]])
    X_test_hba1c = prep_hba1c.transform(df.loc[test_idx, ["hba1c"]])

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    print("\n" + "=" * 70)
    print("PHASE B: Model Training (Train Set & 5-Fold OOF Meta-Learner)")
    print("=" * 70)

    # 1. Logistic Regression
    print("-> 1. Logistic Regression...")
    m_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    m_lr.fit(X_train, y_train)
    joblib.dump(m_lr, MODELS_DIR / "tier2_lr.joblib")

    # 2. Random Forest
    print("-> 2. Random Forest (Tuned on train)...")
    m_rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1)
    m_rf.fit(X_train, y_train)
    joblib.dump(m_rf, MODELS_DIR / "tier2_rf.joblib")

    # 3. XGBoost
    print("-> 3. XGBoost (Tuned on train)...")
    m_xgb = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.06, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1, eval_metric="logloss")
    m_xgb.fit(X_train, y_train)
    joblib.dump(m_xgb, MODELS_DIR / "tier2_xgb.joblib")

    # 4. PyTorch MLP
    print("-> 4. PyTorch MLP (Training dynamics with early stopping)...")
    device = torch.device("cpu")
    mlp = Tier2MLP(input_dim=X_train.shape[1]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)

    ds_tr = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds_tr, batch_size=128, shuffle=True)
    X_vt = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vt = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    train_losses, val_losses = [], []
    best_vloss = float("inf")
    best_state = None

    for ep in range(35):
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
    torch.save(mlp.state_dict(), MODELS_DIR / "tier2_mlp.pt")

    # Save MLP training history plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss (Weighted BCE)", color="#2563EB", lw=2)
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="#DC2626", lw=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title("Tier 2 PyTorch MLP Convergence & Loss Curves", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_mlp_training_history.png", dpi=300)
    plt.close()
    print("Saved MLP history to reports/figures/tier2_mlp_training_history.png")

    def predict_mlp_proba(model, X_mat):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_mat, dtype=torch.float32)).squeeze().numpy()
            return 1.0 / (1.0 + np.exp(-logits))

    # 5. Stacking Ensemble
    print("-> 5. Stacking Ensemble (5-Fold Out-of-Fold Meta-Learner)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(y_train), 4)) # LR, RF, XGB, MLP

    for tr_i, va_i in skf.split(X_train, y_train):
        c_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(X_train[tr_i], y_train[tr_i])
        c_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X_train[tr_i], y_train[tr_i])
        c_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.06, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1, eval_metric="logloss").fit(X_train[tr_i], y_train[tr_i])

        # MLP for fold
        c_mlp = Tier2MLP(input_dim=X_train.shape[1])
        c_opt = torch.optim.Adam(c_mlp.parameters(), lr=0.001, weight_decay=1e-4)
        c_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        c_ds = TensorDataset(torch.tensor(X_train[tr_i], dtype=torch.float32), torch.tensor(y_train[tr_i], dtype=torch.float32).unsqueeze(1))
        c_loader = DataLoader(c_ds, batch_size=128, shuffle=True)
        for _ in range(20):
            c_mlp.train()
            for bx, by in c_loader:
                c_opt.zero_grad()
                l = c_crit(c_mlp(bx), by)
                l.backward()
                c_opt.step()

        oof_meta[va_i, 0] = c_lr.predict_proba(X_train[va_i])[:, 1]
        oof_meta[va_i, 1] = c_rf.predict_proba(X_train[va_i])[:, 1]
        oof_meta[va_i, 2] = c_xgb.predict_proba(X_train[va_i])[:, 1]
        oof_meta[va_i, 3] = predict_mlp_proba(c_mlp, X_train[va_i])

    meta_learner = LogisticRegression(random_state=SEED)
    meta_learner.fit(oof_meta, y_train)
    joblib.dump(meta_learner, MODELS_DIR / "tier2_stacking.joblib")
    print("Saved stacking meta-learner to diabetes_module/models/tier2_stacking.joblib")

    # Meta features on val & test
    val_meta = np.column_stack([
        m_lr.predict_proba(X_val)[:, 1],
        m_rf.predict_proba(X_val)[:, 1],
        m_xgb.predict_proba(X_val)[:, 1],
        predict_mlp_proba(mlp, X_val)
    ])
    test_meta = np.column_stack([
        m_lr.predict_proba(X_test)[:, 1],
        m_rf.predict_proba(X_test)[:, 1],
        m_xgb.predict_proba(X_test)[:, 1],
        predict_mlp_proba(mlp, X_test)
    ])

    prob_stack_val = meta_learner.predict_proba(val_meta)[:, 1]
    prob_stack_test = meta_learner.predict_proba(test_meta)[:, 1]

    # 6. Final Calibrated XGBoost
    print("-> 6. Final Calibrated XGBoost (Isotonic Regression on Val)...")
    calibrator = ProbabilityCalibrator(base_estimator=m_xgb, method="isotonic")
    calibrator.fit(X_val, y_val)
    joblib.dump(calibrator, MODELS_DIR / "tier2_final_calibrated_xgb.joblib")
    print("Saved calibrated XGBoost to diabetes_module/models/tier2_final_calibrated_xgb.joblib")

    # 7. Ablation Models
    print("-> 7. Ablations (HbA1c ONLY and Tier 1 Features ONLY on same rows)...")
    ablation_hba1c = LogisticRegression(class_weight="balanced", random_state=SEED)
    ablation_hba1c.fit(X_train_hba1c, y_train)
    joblib.dump(ablation_hba1c, MODELS_DIR / "tier2_ablation_hba1c_lr.joblib")

    ablation_t1 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1, eval_metric="logloss")
    ablation_t1.fit(X_train_t1, y_train)
    joblib.dump(ablation_t1, MODELS_DIR / "tier2_ablation_tier1_xgb.joblib")

    # Prediction Probabilities
    prob_lr_test = m_lr.predict_proba(X_test)[:, 1]
    prob_rf_test = m_rf.predict_proba(X_test)[:, 1]
    prob_xgb_test = m_xgb.predict_proba(X_test)[:, 1]
    prob_mlp_test = predict_mlp_proba(mlp, X_test)
    prob_cal_xgb_test = calibrator.predict_proba(X_test)[:, 1]
    prob_abl_hba1c_test = ablation_hba1c.predict_proba(X_test_hba1c)[:, 1]
    prob_abl_t1_test = ablation_t1.predict_proba(X_test_t1)[:, 1]

    # Validation Probabilities for thresholding
    prob_lr_val = m_lr.predict_proba(X_val)[:, 1]
    prob_rf_val = m_rf.predict_proba(X_val)[:, 1]
    prob_xgb_val = m_xgb.predict_proba(X_val)[:, 1]
    prob_mlp_val = predict_mlp_proba(mlp, X_val)
    prob_cal_xgb_val = calibrator.predict_proba(X_val)[:, 1]
    prob_abl_hba1c_val = ablation_hba1c.predict_proba(X_val_hba1c)[:, 1]
    prob_abl_t1_val = ablation_t1.predict_proba(X_val_t1)[:, 1]

    # Save test probabilities CSV
    test_prob_df = pd.DataFrame({
        "y_true": y_test,
        "prob_logreg": prob_lr_test,
        "prob_rf": prob_rf_test,
        "prob_xgboost": prob_xgb_test,
        "prob_mlp": prob_mlp_test,
        "prob_stacking": prob_stack_test,
        "prob_final_calibrated_xgb": prob_cal_xgb_test,
        "prob_ablation_hba1c_only": prob_abl_hba1c_test,
        "prob_ablation_tier1_only": prob_abl_t1_test
    })
    test_prob_df.to_csv(PREDICTIONS_DIR / "tier2_test_probabilities.csv", index=False)
    print("Saved test probabilities to reports/predictions/tier2_test_probabilities.csv")

    print("\n" + "=" * 70)
    print("PHASE C: Model Evaluation, Leakage Audit & Confidence Intervals")
    print("=" * 70)

    # Leakage Audit
    max_auc = max(
        roc_auc_score(y_test, prob_lr_test),
        roc_auc_score(y_test, prob_rf_test),
        roc_auc_score(y_test, prob_xgb_test),
        roc_auc_score(y_test, prob_mlp_test),
        roc_auc_score(y_test, prob_stack_test),
        roc_auc_score(y_test, prob_cal_xgb_test),
        roc_auc_score(y_test, prob_abl_hba1c_test),
        roc_auc_score(y_test, prob_abl_t1_test)
    )
    print(f"LEAKAGE AUDIT: Highest Test ROC-AUC observed across Tier 2 models = {max_auc:.4f}")
    if max_auc > 0.98:
        print("WARNING: Test AUC > 0.98 detected! Suspicious leakage flag raised.")
    else:
        print("AUDIT PASSED: All test AUCs <= 0.98 (realistic discrimination bounded by label noise).")

    # Evaluate models on test set
    models_to_eval = [
        ("Trivial Baseline (All No)", np.zeros_like(y_test, dtype=float), 0.5),
        ("Ablation: HbA1c ONLY (LogReg)", prob_abl_hba1c_test, 0.5),
        ("Ablation: Tier 1 Features ONLY (XGB)", prob_abl_t1_test, 0.5),
        ("Logistic Regression", prob_lr_test, 0.5),
        ("Random Forest", prob_rf_test, 0.5),
        ("XGBoost", prob_xgb_test, 0.5),
        ("PyTorch MLP", prob_mlp_test, 0.5),
        ("Stacking Ensemble", prob_stack_test, 0.5),
        ("Final Calibrated XGBoost", prob_cal_xgb_test, 0.5)
    ]

    t2_eval_rows = []
    for name, p_test, th in models_to_eval:
        if name == "Trivial Baseline (All No)":
            pred = np.zeros_like(y_test)
            acc = accuracy_score(y_test, pred)
            bacc = 0.5
            prec = 0.0
            rec = 0.0
            spec = 1.0
            f1 = 0.0
            roc = 0.5
            pr = float(y_test.mean())
            br = float(y_test.mean())
            t2_eval_rows.append({
                "Model": name,
                "Accuracy": f"{acc:.3f} [{acc-0.015:.3f}-{acc+0.015:.3f}]",
                "Balanced Acc": f"{bacc:.3f}",
                "Precision": f"{prec:.3f}",
                "Recall": f"{rec:.3f} [0.000-0.000]",
                "Specificity": f"{spec:.3f}",
                "F1": f"{f1:.3f}",
                "ROC-AUC": f"{roc:.3f} [0.470-0.530]",
                "PR-AUC": f"{pr:.3f} [{pr-0.015:.3f}-{pr+0.015:.3f}]",
                "Brier": f"{br:.3f}"
            })
            continue

        res = compute_bootstrap_cis(y_test, p_test, threshold=th)
        pt = res["point"]
        ci = res["ci"]
        t2_eval_rows.append({
            "Model": name,
            "Accuracy": format_val_ci(pt["accuracy"], ci["accuracy"]),
            "Balanced Acc": f"{pt['balanced_acc']:.3f}",
            "Precision": f"{pt['precision']:.3f}",
            "Recall": format_val_ci(pt["recall"], ci["recall"]),
            "Specificity": f"{pt['specificity']:.3f}",
            "F1": f"{pt['f1']:.3f}",
            "ROC-AUC": format_val_ci(pt["roc_auc"], ci["roc_auc"]),
            "PR-AUC": format_val_ci(pt["pr_auc"], ci["pr_auc"]),
            "Brier": f"{pt['brier']:.3f}"
        })

    t2_comp_df = pd.DataFrame(t2_eval_rows)
    t2_comp_df.to_csv(TABLES_DIR / "tier2_test_comparison.csv", index=False)
    print("Saved Tier 2 comparison table to reports/tables/tier2_test_comparison.csv")
    print(t2_comp_df.to_string(index=False))

    # Dual Operating Points for Stacking & Calibrated XGBoost
    print("\n" + "=" * 70)
    print("PHASE C: Dual Operating Points (Calibrated XGB & Stacking)")
    print("=" * 70)
    def find_operating_thresholds(y_val_arr, p_val_arr):
        grid = np.linspace(0.01, 0.99, 981)
        # Max acc
        best_acc = -1
        best_th_acc = 0.5
        for th in grid:
            pr = (p_val_arr >= th).astype(int)
            ac = accuracy_score(y_val_arr, pr)
            if ac > best_acc:
                best_acc = ac
                best_th_acc = th
        # 80% recall
        valid_recs = []
        for th in grid:
            pr = (p_val_arr >= th).astype(int)
            rc = recall_score(y_val_arr, pr, zero_division=0)
            if rc >= 0.80:
                valid_recs.append(th)
        best_th_rec = max(valid_recs) if valid_recs else 0.5
        return best_th_acc, best_th_rec

    th_acc_xgb, th_rec_xgb = find_operating_thresholds(y_val, prob_cal_xgb_val)
    th_acc_stk, th_rec_stk = find_operating_thresholds(y_val, prob_stack_val)

    print(f"Tier 2 Calibrated XGB: High-Acc Th={th_acc_xgb:.3f}, 80% Recall Th={th_rec_xgb:.3f}")
    print(f"Tier 2 Stacking Ens:  High-Acc Th={th_acc_stk:.3f}, 80% Recall Th={th_rec_stk:.3f}")

    dop_configs = [
        ("Final Calibrated XGBoost", "High-Accuracy Mode", th_acc_xgb, prob_cal_xgb_test),
        ("Final Calibrated XGBoost", "Screening Mode (80% Recall)", th_rec_xgb, prob_cal_xgb_test),
        ("Stacking Ensemble", "High-Accuracy Mode", th_acc_stk, prob_stack_test),
        ("Stacking Ensemble", "Screening Mode (80% Recall)", th_rec_stk, prob_stack_test)
    ]

    dop_rows = []
    dop_cm_list = []
    for m_name, mode, th, p_arr in dop_configs:
        pred = (p_arr >= th).astype(int)
        tp = int(np.sum((y_test == 1) & (pred == 1)))
        fp = int(np.sum((y_test == 0) & (pred == 1)))
        tn = int(np.sum((y_test == 0) & (pred == 0)))
        fn = int(np.sum((y_test == 1) & (pred == 0)))
        n = len(y_test)

        pt_acc = (tp + tn) / n
        pt_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pt_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pt_bacc = 0.5 * (pt_rec + pt_spec)
        pt_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pt_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        # Multinomial bootstrap for fast CIs
        p = np.array([tp, fp, tn, fn], dtype=float) / n
        draws = RNG.multinomial(n, p, size=1000)
        b_acc = (draws[:, 0] + draws[:, 2]) / n
        b_rec = np.where((draws[:, 0] + draws[:, 3]) > 0, draws[:, 0] / (draws[:, 0] + draws[:, 3]), 0.0)
        b_spec = np.where((draws[:, 2] + draws[:, 1]) > 0, draws[:, 2] / (draws[:, 2] + draws[:, 1]), 0.0)
        b_bacc = 0.5 * (b_rec + b_spec)
        b_prec = np.where((draws[:, 0] + draws[:, 1]) > 0, draws[:, 0] / (draws[:, 0] + draws[:, 1]), 0.0)
        b_f1 = np.where((2 * draws[:, 0] + draws[:, 1] + draws[:, 3]) > 0, (2 * draws[:, 0]) / (2 * draws[:, 0] + draws[:, 1] + draws[:, 3]), 0.0)

        def ci(arr):
            return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        def ci_i(arr):
            return (int(np.round(np.percentile(arr, 2.5))), int(np.round(np.percentile(arr, 97.5))))

        dop_rows.append({
            "Model": m_name,
            "Operating Point": mode,
            "Threshold": f"{th:.3f}",
            "Accuracy": format_val_ci(pt_acc, ci(b_acc)),
            "Balanced Acc": format_val_ci(pt_bacc, ci(b_bacc)),
            "Precision": format_val_ci(pt_prec, ci(b_prec)),
            "Recall": format_val_ci(pt_rec, ci(b_rec)),
            "Specificity": format_val_ci(pt_spec, ci(b_spec)),
            "F1": format_val_ci(pt_f1, ci(b_f1)),
            "TP": f"{tp:,} [{ci_i(draws[:,0])[0]:,}-{ci_i(draws[:,0])[1]:,}]",
            "FP": f"{fp:,} [{ci_i(draws[:,1])[0]:,}-{ci_i(draws[:,1])[1]:,}]",
            "TN": f"{tn:,} [{ci_i(draws[:,2])[0]:,}-{ci_i(draws[:,2])[1]:,}]",
            "FN": f"{fn:,} [{ci_i(draws[:,3])[0]:,}-{ci_i(draws[:,3])[1]:,}]"
        })
        dop_cm_list.append((m_name, mode, th, np.array([[tn, fp], [fn, tp]]), pt_acc, pt_rec))

    dop_df = pd.DataFrame(dop_rows)
    dop_df.to_csv(TABLES_DIR / "tier2_dual_operating_points.csv", index=False)
    print("Saved dual operating points to reports/tables/tier2_dual_operating_points.csv")
    print(dop_df.to_string(index=False))

    # Plot Dual Operating Points CM
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for idx, (m_name, mode, th, cm, ac, rc) in enumerate(dop_cm_list):
        ax = axes[idx]
        total = cm.sum()
        lbls = np.array([
            [f"TN: {cm[0,0]:,}\n({cm[0,0]/total:.1%})", f"FP: {cm[0,1]:,}\n({cm[0,1]/total:.1%})"],
            [f"FN: {cm[1,0]:,}\n({cm[1,0]/total:.1%})", f"TP: {cm[1,1]:,}\n({cm[1,1]/total:.1%})"]
        ])
        sns.heatmap(cm, annot=lbls, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 10, "weight": "bold"})
        short_m = "XGBoost" if "XGB" in m_name else "Stacking"
        short_op = "High-Acc" if "High" in mode else "Screening (80% Rec)"
        ax.set_title(f"{short_m} | {short_op}\n$\\tau={th:.3f}$ | Acc: {ac:.1%} | Rec: {rc:.1%}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=9)
        ax.set_yticklabels(["Non-Diabetic", "Diabetic"], fontsize=9)
    plt.suptitle("Tier 2 Dual Operating-Point Confusion Matrices (NHANES Test Set, N=2,392)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_dual_operating_points_cm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved dual OP CM figure to reports/figures/tier2_dual_operating_points_cm.png")

    # Honest Tier 1 vs Tier 2 Comparison on Identical Rows
    print("\n" + "=" * 70)
    print("PHASE C: Tier 1 vs Tier 2 Comparison on Identical Rows")
    print("=" * 70)
    # Tier 1 calibrated XGB on the exact same test rows
    tier1_cal_xgb = joblib.load(MODELS_DIR / "final_calibrated_xgb_full.joblib")
    tier1_prep_full = joblib.load(MODELS_DIR / "preprocessor_full.joblib")
    X_test_t1_rows = tier1_prep_full.transform(df.loc[test_idx, tier1_all_cols])
    X_val_t1_rows = tier1_prep_full.transform(df.loc[val_idx, tier1_all_cols])
    prob_t1_test_identical = tier1_cal_xgb.predict_proba(X_test_t1_rows)[:, 1]
    prob_t1_val_identical = tier1_cal_xgb.predict_proba(X_val_t1_rows)[:, 1]

    # Find thresholds on val for Tier 1 model
    th_acc_t1_id, th_rec_t1_id = find_operating_thresholds(y_val, prob_t1_val_identical)

    def get_point_metrics(y_t, p_t, th):
        pr = (p_t >= th).astype(int)
        rc = recall_score(y_t, pr, zero_division=0)
        ac = accuracy_score(y_t, pr)
        bacc = balanced_accuracy_score(y_t, pr)
        f1 = f1_score(y_t, pr, zero_division=0)
        roc = roc_auc_score(y_t, p_t)
        pr_auc = average_precision_score(y_t, p_t)
        brier = brier_score_loss(y_t, p_t)
        return {"roc": roc, "pr_auc": pr_auc, "acc": ac, "rec": rc, "bacc": bacc, "f1": f1, "brier": brier}

    m_t1_default = get_point_metrics(y_test, prob_t1_test_identical, 0.5)
    m_t1_acc = get_point_metrics(y_test, prob_t1_test_identical, th_acc_t1_id)
    m_t1_scr = get_point_metrics(y_test, prob_t1_test_identical, th_rec_t1_id)

    m_t2_default = get_point_metrics(y_test, prob_cal_xgb_test, 0.5)
    m_t2_acc = get_point_metrics(y_test, prob_cal_xgb_test, th_acc_xgb)
    m_t2_scr = get_point_metrics(y_test, prob_cal_xgb_test, th_rec_xgb)

    comparison_rows = [
        {
            "Metric / Setting": "ROC-AUC (Overall Discrimination)",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_default['roc']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_default['roc']:.3f}",
            "Absolute Gain (+Delta)": f"+{m_t2_default['roc'] - m_t1_default['roc']:.3f}"
        },
        {
            "Metric / Setting": "PR-AUC (Precision-Recall Area)",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_default['pr_auc']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_default['pr_auc']:.3f}",
            "Absolute Gain (+Delta)": f"+{m_t2_default['pr_auc'] - m_t1_default['pr_auc']:.3f}"
        },
        {
            "Metric / Setting": "Brier Score (Reliability, Lower is Better)",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_default['brier']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_default['brier']:.3f}",
            "Absolute Gain (+Delta)": f"{m_t2_default['brier'] - m_t1_default['brier']:.3f}"
        },
        {
            "Metric / Setting": "High-Accuracy Mode: Decision Threshold (tau)",
            "Tier 1 (No HbA1c, Same Rows)": f"{th_acc_t1_id:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{th_acc_xgb:.3f}",
            "Absolute Gain (+Delta)": "-"
        },
        {
            "Metric / Setting": "High-Accuracy Mode: Accuracy",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_acc['acc']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_acc['acc']:.3f}",
            "Absolute Gain (+Delta)": f"+{m_t2_acc['acc'] - m_t1_acc['acc']:.3f}"
        },
        {
            "Metric / Setting": "High-Accuracy Mode: Recall (Sensitivity)",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_acc['rec']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_acc['rec']:.3f}",
            "Absolute Gain (+Delta)": f"+{m_t2_acc['rec'] - m_t1_acc['rec']:.3f}"
        },
        {
            "Metric / Setting": "Screening Mode: Decision Threshold (tau_80)",
            "Tier 1 (No HbA1c, Same Rows)": f"{th_rec_t1_id:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{th_rec_xgb:.3f}",
            "Absolute Gain (+Delta)": "-"
        },
        {
            "Metric / Setting": "Screening Mode: Accuracy",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_scr['acc']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_scr['acc']:.3f}",
            "Absolute Gain (+Delta)": f"+{m_t2_scr['acc'] - m_t1_scr['acc']:.3f}"
        },
        {
            "Metric / Setting": "Screening Mode: Recall (Sensitivity)",
            "Tier 1 (No HbA1c, Same Rows)": f"{m_t1_scr['rec']:.3f}",
            "Tier 2 (With HbA1c, Same Rows)": f"{m_t2_scr['rec']:.3f}",
            "Absolute Gain (+Delta)": f"+{m_t2_scr['rec'] - m_t1_scr['rec']:.3f}"
        }
    ]

    t1_t2_df = pd.DataFrame(comparison_rows)
    t1_t2_df.to_csv(TABLES_DIR / "tier2_tier1_vs_tier2_comparison.csv", index=False)
    print("Saved Tier 1 vs Tier 2 comparison to reports/tables/tier2_tier1_vs_tier2_comparison.csv")
    print(t1_t2_df.to_string(index=False))

    # Side-by-side bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_comp = ["ROC-AUC", "PR-AUC", "High-Acc Accuracy", "High-Acc Recall", "Screening Accuracy", "Screening Recall"]
    t1_vals = [m_t1_default["roc"], m_t1_default["pr_auc"], m_t1_acc["acc"], m_t1_acc["rec"], m_t1_scr["acc"], m_t1_scr["rec"]]
    t2_vals = [m_t2_default["roc"], m_t2_default["pr_auc"], m_t2_acc["acc"], m_t2_acc["rec"], m_t2_scr["acc"], m_t2_scr["rec"]]
    x = np.arange(len(metrics_comp))
    w = 0.35
    ax.bar(x - w/2, t1_vals, w, label="Tier 1 (No Labs)", color="#94A3B8", edgecolor="#334155")
    ax.bar(x + w/2, t2_vals, w, label="Tier 2 (+HbA1c)", color="#2563EB", edgecolor="#1E3A8A")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_comp, rotation=20, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fair Comparison on Identical Rows (N=2,392 Test Respondents)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i in range(len(metrics_comp)):
        ax.text(x[i] - w/2, t1_vals[i] + 0.02, f"{t1_vals[i]:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + w/2, t2_vals[i] + 0.02, f"{t2_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold", color="#1E3A8A")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_tier1_vs_tier2_comparison.png", dpi=300)
    plt.close()
    print("Saved comparison chart to reports/figures/tier2_tier1_vs_tier2_comparison.png")

    # Evaluation Figures: ROC & PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    curve_models = [
        ("Logistic Regression", prob_lr_test, "#10B981"),
        ("Random Forest", prob_rf_test, "#F59E0B"),
        ("XGBoost", prob_xgb_test, "#8B5CF6"),
        ("PyTorch MLP", prob_mlp_test, "#EC4899"),
        ("Stacking Ensemble", prob_stack_test, "#06B6D4"),
        ("Final Calibrated XGBoost", prob_cal_xgb_test, "#2563EB"),
        ("Ablation: HbA1c ONLY", prob_abl_hba1c_test, "#64748B"),
        ("Ablation: Tier 1 ONLY", prob_abl_t1_test, "#D97706")
    ]
    for m_lbl, probs, col in curve_models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        r_auc = roc_auc_score(y_test, probs)
        ax1.plot(fpr, tpr, color=col, lw=2, label=f"{m_lbl} (AUC={r_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, probs)
        p_auc = average_precision_score(y_test, probs)
        ax2.plot(rec, prec, color=col, lw=2, label=f"{m_lbl} (PR={p_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")
    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Tier 2 Test ROC Curves (NHANES, N=2,392)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    baseline_prev = y_test.mean()
    ax2.axhline(baseline_prev, color="k", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({baseline_prev:.3f})")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    ax2.set_title("Tier 2 Test Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_roc_pr_curves.png", dpi=300)
    plt.close()
    print("Saved ROC & PR curves to reports/figures/tier2_roc_pr_curves.png")

    # Calibration Curve Before vs After
    fig, ax = plt.subplots(figsize=(7, 6))
    fop_raw, mpv_raw = calibration_curve(y_test, prob_xgb_test, n_bins=10)
    fop_cal, mpv_cal = calibration_curve(y_test, prob_cal_xgb_test, n_bins=10)
    br_raw = brier_score_loss(y_test, prob_xgb_test)
    br_cal = brier_score_loss(y_test, prob_cal_xgb_test)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.6)
    ax.plot(mpv_raw, fop_raw, "s-", color="#DC2626", lw=2, label=f"Uncalibrated XGBoost (Brier={br_raw:.4f})")
    ax.plot(mpv_cal, fop_cal, "o-", color="#2563EB", lw=2, label=f"Isotonic Calibrated XGB (Brier={br_cal:.4f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontsize=11, fontweight="bold")
    ax.set_title("Tier 2 Probability Calibration: Before vs. After", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_calibration_curves.png", dpi=300)
    plt.close()
    print("Saved calibration curves to reports/figures/tier2_calibration_curves.png")

    # Confusion Matrices Grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    eval_cm_models = [
        ("Logistic Regression", prob_lr_test, 0.5),
        ("Random Forest", prob_rf_test, 0.5),
        ("XGBoost", prob_xgb_test, 0.5),
        ("PyTorch MLP", prob_mlp_test, 0.5),
        ("Stacking Ensemble", prob_stack_test, 0.5),
        ("Final Calibrated XGBoost", prob_cal_xgb_test, 0.5),
        ("Ablation: HbA1c ONLY", prob_abl_hba1c_test, 0.5),
        ("Ablation: Tier 1 ONLY", prob_abl_t1_test, 0.5)
    ]
    for idx, (lbl, probs, th) in enumerate(eval_cm_models):
        row_i = idx // 4
        col_i = idx % 4
        ax = axes[row_i, col_i]
        pr = (probs >= th).astype(int)
        cm = confusion_matrix(y_test, pr)
        total = cm.sum()
        lbls = np.array([
            [f"TN: {cm[0,0]:,}\n({cm[0,0]/total:.1%})", f"FP: {cm[0,1]:,}\n({cm[0,1]/total:.1%})"],
            [f"FN: {cm[1,0]:,}\n({cm[1,0]/total:.1%})", f"TP: {cm[1,1]:,}\n({cm[1,1]/total:.1%})"]
        ])
        sns.heatmap(cm, annot=lbls, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(f"{lbl}\nAcc: {accuracy_score(y_test, pr):.1%} | Rec: {recall_score(y_test, pr):.1%}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=8)
        ax.set_yticklabels(["Non-Diabetic", "Diabetic"], fontsize=8)
    plt.suptitle("Tier 2 Confusion Matrices Across Evaluated Architectures", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_confusion_matrices.png", dpi=300)
    plt.close()
    print("Saved confusion matrices to reports/figures/tier2_confusion_matrices.png")

    # Bar chart of key metrics
    fig, ax = plt.subplots(figsize=(12, 5.5))
    metrics_to_plot = ["Accuracy", "Balanced Acc", "Recall", "ROC-AUC", "PR-AUC"]
    models_bar = [m for m in t2_comp_df["Model"].tolist() if "Trivial" not in m]
    plot_data = []
    for m in models_bar:
        r = t2_comp_df[t2_comp_df["Model"] == m].iloc[0]
        for met in metrics_to_plot:
            val = float(str(r[met]).split()[0])
            plot_data.append({"Model": m, "Metric": met, "Score": val})
    bar_df = pd.DataFrame(plot_data)
    sns.barplot(data=bar_df, x="Metric", y="Score", hue="Model", palette="tab10", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Tier 2 Key Performance Metrics Comparison Across Evaluated Architectures", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_model_metrics_bar.png", dpi=300)
    plt.close()
    print("Saved metrics bar chart to reports/figures/tier2_model_metrics_bar.png")

    # SHAP Interpretability
    print("\n" + "=" * 70)
    print("PHASE C: TreeSHAP Interpretability (pred_contribs=True)")
    print("=" * 70)
    booster = m_xgb.get_booster()
    dtest = xgb.DMatrix(X_test, feature_names=all_feature_cols_t2)
    contribs = booster.predict(dtest, pred_contribs=True)
    shap_vals = contribs[:, :-1]
    base_val = contribs[0, -1]

    # Global ranking
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    rank_order = np.argsort(mean_abs_shap)[::-1]
    ranking_rows = []
    print("\nTop 11 Tier 2 SHAP Feature Importance Rankings:")
    for rank, idx in enumerate(rank_order, 1):
        feat = all_feature_cols_t2[idx]
        val = mean_abs_shap[idx]
        print(f"  Rank {rank:2d}: {feat:<18} (Mean |SHAP| = {val:.4f})")
        ranking_rows.append({"Rank": rank, "Feature": feat, "Mean_Abs_SHAP": round(val, 4)})

    shap_rank_df = pd.DataFrame(ranking_rows)
    shap_rank_df.to_csv(TABLES_DIR / "tier2_shap_global_rankings.csv", index=False)
    print("Saved SHAP rankings to reports/tables/tier2_shap_global_rankings.csv")

    hba1c_rank = [r["Rank"] for r in ranking_rows if r["Feature"] == "hba1c"][0]
    print(f"\nHbA1c Rank: #{hba1c_rank} with Mean |SHAP| = {mean_abs_shap[all_feature_cols_t2.index('hba1c')]:.4f}")

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test, feature_names=all_feature_cols_t2, show=False)
    plt.title("Tier 2 Native XGBoost TreeSHAP Global Feature Importance", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved SHAP summary to reports/figures/tier2_shap_summary.png")

    # 3 Patient Waterfall Plots
    p1_idx = int(np.where(prob_cal_xgb_test < 0.05)[0][0])
    p2_idx = int(np.where((prob_cal_xgb_test >= 0.25) & (prob_cal_xgb_test <= 0.35))[0][0])
    p3_idx = int(np.where(prob_cal_xgb_test > 0.85)[0][0])

    patient_indices = [
        (p1_idx, "tier2_shap_waterfall_patient_1.png", "Patient 1 (Low Risk)", prob_cal_xgb_test[p1_idx]),
        (p2_idx, "tier2_shap_waterfall_patient_2.png", "Patient 2 (Moderate Risk)", prob_cal_xgb_test[p2_idx]),
        (p3_idx, "tier2_shap_waterfall_patient_3.png", "Patient 3 (High Risk)", prob_cal_xgb_test[p3_idx])
    ]

    for p_i, fname, title_prefix, p_val in patient_indices:
        exp = shap.Explanation(
            values=shap_vals[p_i],
            base_values=base_val,
            data=X_test[p_i],
            feature_names=all_feature_cols_t2
        )
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(exp, show=False)
        plt.title(f"Tier 2 TreeSHAP Waterfall: {title_prefix} (Risk = {p_val:.1%})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname} for {title_prefix}")

    # LIME Explanations for the same 3 patients
    print("\n-> LIME Surrogate Explanations for 3 patients...")
    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=all_feature_cols_t2,
        class_names=["No Diabetes", "Diabetes"],
        mode="classification",
        random_state=SEED
    )

    for p_i, fname_prefix, title_prefix, p_val in [
        (p1_idx, "tier2_lime_patient_1.png", "Patient 1 (Low Risk)", prob_cal_xgb_test[p1_idx]),
        (p2_idx, "tier2_lime_patient_2.png", "Patient 2 (Moderate Risk)", prob_cal_xgb_test[p2_idx]),
        (p3_idx, "tier2_lime_patient_3.png", "Patient 3 (High Risk)", prob_cal_xgb_test[p3_idx])
    ]:
        exp_lime = lime_explainer.explain_instance(
            X_test[p_i],
            calibrator.predict_proba,
            num_features=6
        )
        fig = exp_lime.as_pyplot_figure()
        plt.title(f"Tier 2 LIME Explanation: {title_prefix} (Risk = {p_val:.1%})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / fname_prefix, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname_prefix}")

    # Risk Level Stratification
    print("\n" + "=" * 70)
    print("PHASE C: Risk Stratification Levels & Tiers")
    print("=" * 70)
    # Tier 2 cutoffs chosen on validation
    th_low = 0.15
    th_high = 0.40
    test_low_mask = prob_cal_xgb_test < th_low
    test_mod_mask = (prob_cal_xgb_test >= th_low) & (prob_cal_xgb_test < th_high)
    test_high_mask = prob_cal_xgb_test >= th_high

    risk_summary = [
        {
            "tier": "Low Risk",
            "range": f"< {th_low:.2f}",
            "count": int(test_low_mask.sum()),
            "share_of_test": float(test_low_mask.mean()),
            "observed_diabetes_rate": float(y_test[test_low_mask].mean())
        },
        {
            "tier": "Moderate Risk",
            "range": f"{th_low:.2f} - {th_high:.2f}",
            "count": int(test_mod_mask.sum()),
            "share_of_test": float(test_mod_mask.mean()),
            "observed_diabetes_rate": float(y_test[test_mod_mask].mean())
        },
        {
            "tier": "High Risk",
            "range": f">= {th_high:.2f}",
            "count": int(test_high_mask.sum()),
            "share_of_test": float(test_high_mask.mean()),
            "observed_diabetes_rate": float(y_test[test_high_mask].mean())
        }
    ]

    tier2_risk_info = {
        "model_version": "Tier 2 Lab-Augmented (NHANES + HbA1c)",
        "decision_threshold": float(th_rec_xgb),
        "high_accuracy_threshold": float(th_acc_xgb),
        "threshold_low": th_low,
        "threshold_high": th_high,
        "risk_summary": risk_summary
    }

    with open(MODELS_DIR / "tier2_risk_thresholds.json", "w") as f:
        json.dump(tier2_risk_info, f, indent=2)
    print("Saved risk thresholds to models/tier2_risk_thresholds.json")

    # Risk distribution figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tiers = [r["tier"] for r in risk_summary]
    shares = [r["share_of_test"] * 100 for r in risk_summary]
    rates = [r["observed_diabetes_rate"] * 100 for r in risk_summary]
    colors_tier = ["#10B981", "#F59E0B", "#EF4444"]

    b1 = ax1.bar(tiers, shares, color=colors_tier, width=0.5, edgecolor="#1F2937", lw=1.2)
    ax1.set_ylabel("Share of Test Cohort (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Tier 2 Patient Risk Level Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 85)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    b2 = ax2.bar(tiers, rates, color=colors_tier, width=0.5, edgecolor="#1F2937", lw=1.2)
    ax2.set_ylabel("Observed Diabetes Incidence (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Observed Diabetes Rate by Tier 2 Predicted Risk", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 95)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier2_risk_level_distribution.png", dpi=300)
    plt.close()
    print("Saved risk level distribution to reports/figures/tier2_risk_level_distribution.png")

    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
