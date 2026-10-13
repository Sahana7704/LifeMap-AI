"""
Complete Training, Tuning, Stacking, Calibration, and Evaluation Pipeline for LifeMap AI Diabetes Module.
Executes Phase 2 (Model Training & Tuning on NHANES) and Phase 3 (Evaluation & External Validation on BRFSS).
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from utils import calculate_metrics, bootstrap_ci, log_experiment

warnings.filterwarnings("ignore")

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==============================================================================
# PyTorch MLP Architecture
# ==============================================================================
class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int):
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


class PyTorchMLPWrapper:
    """Scikit-learn compatible wrapper for PyTorch MLP with early stopping."""
    def __init__(self, input_dim: int, pos_weight: float, epochs: int = 40, lr: float = 0.001, patience: int = 8):
        self.input_dim = input_dim
        self.pos_weight = pos_weight
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.model = DiabetesMLP(input_dim)
        self.best_state = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        device = torch.device("cpu")
        self.model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32).to(device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            # Validation loss for early stopping
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32)
            logits = self.model(t_x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ==============================================================================
# Calibration Wrapper (Isotonic or Platt on Validation Probabilities)
# ==============================================================================
class ProbabilityCalibrator:
    def __init__(self, base_estimator, method="isotonic"):
        self.base_estimator = base_estimator
        self.method = method
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            self.calibrator = LogisticRegression()

    def fit(self, X_val, y_val):
        probs = self.base_estimator.predict_proba(X_val)[:, 1]
        if self.method == "isotonic":
            self.calibrator.fit(probs, y_val)
        else:
            self.calibrator.fit(probs.reshape(-1, 1), y_val)
        return self

    def predict_proba(self, X):
        probs = self.base_estimator.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            cal_p = np.clip(self.calibrator.predict(probs), 0.0, 1.0)
        else:
            cal_p = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - cal_p, cal_p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ==============================================================================
# Preprocessing Pipeline (Fitted on Train Only)
# ==============================================================================
class TabularPreprocessor:
    def __init__(self, continuous_cols: list[str], binary_cols: list[str]):
        self.continuous_cols = continuous_cols
        self.binary_cols = binary_cols
        self.all_cols = continuous_cols + binary_cols
        self.cont_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.bin_imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, df_train: pd.DataFrame):
        if self.continuous_cols:
            c_data = self.cont_imputer.fit_transform(df_train[self.continuous_cols])
            self.scaler.fit(c_data)
        if self.binary_cols:
            self.bin_imputer.fit(df_train[self.binary_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.continuous_cols:
            c_data = self.cont_imputer.transform(df[self.continuous_cols])
            c_scaled = self.scaler.transform(c_data)
            parts.append(c_scaled)
        if self.binary_cols:
            b_data = self.bin_imputer.transform(df[self.binary_cols])
            parts.append(b_data)
        return np.hstack(parts) if len(parts) > 1 else parts[0]


# ==============================================================================
# Model Evaluation & Plotting Helpers
# ==============================================================================
def plot_evaluation_curves(y_true, y_prob_dict, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} - ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix.lower()}_roc_curves.png", dpi=300)
    plt.close()

    # 2. PR Curves
    plt.figure(figsize=(8, 6))
    prevalence = np.mean(y_true)
    for name, prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, prob)
        plt.plot(recall, precision, label=name)
    plt.axhline(prevalence, color="k", linestyle="--", label=f"Baseline Prevalence ({prevalence:.1%})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} - Precision-Recall Curves")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix.lower()}_pr_curves.png", dpi=300)
    plt.close()


def plot_calibration(y_true, uncal_prob, cal_prob, output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncal_prob, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_prob, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(prob_pred_uncal, prob_true_uncal, "s-", label="Uncalibrated")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", label="Isotonic Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title(f"Calibration Reliability Curve ({model_name})")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve_before_after.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir: Path, title: str, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()


# ==============================================================================
# Main Pipeline
# ==============================================================================
def run_pipeline():
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "diabetes_module" / "models"
    figures_dir = root_dir / "diabetes_module" / "reports" / "figures"
    tables_dir = root_dir / "diabetes_module" / "reports" / "tables"
    exp_csv = root_dir / "diabetes_module" / "reports" / "experiments.csv"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      LIFEMAP AI DIABETES RISK MODULE - PHASE 2 & 3 TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load Data
    nhanes_path = data_dir / "nhanes_diabetes_processed.csv"
    brfss_path = data_dir / "brfss_diabetes_processed.csv"
    print(f"Loading NHANES data from {nhanes_path}...")
    nhanes_df = pd.read_csv(nhanes_path)
    print(f"Loading BRFSS data from {brfss_path}...")
    brfss_df = pd.read_csv(brfss_path)

    # Define Feature Sets
    full_cont = ["age", "bmi", "mean_sbp", "mean_dbp"]
    full_bin = ["sex", "high_bp", "high_chol", "smoker", "phys_activity", "family_history"]
    full_features = full_cont + full_bin

    shared_cont = ["age_bucket", "bmi"]
    shared_bin = ["sex", "high_bp", "high_chol", "smoker", "phys_activity"]
    shared_features = shared_cont + shared_bin

    y_all = nhanes_df["diabetes"].values

    # 2. Strict Leakage-Free Train/Val/Test Split (70 / 15 / 15) BEFORE preprocessing
    train_idx, temp_idx = train_test_split(
        np.arange(len(nhanes_df)),
        test_size=0.30,
        random_state=SEED,
        stratify=y_all
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=y_all[temp_idx]
    )

    df_train = nhanes_df.iloc[train_idx].reset_index(drop=True)
    df_val = nhanes_df.iloc[val_idx].reset_index(drop=True)
    df_test = nhanes_df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train["diabetes"].values
    y_val = df_val["diabetes"].values
    y_test = df_test["diabetes"].values

    print(f"Data Split: Train={len(df_train)} (pos={y_train.sum()}, {y_train.mean():.1%}), "
          f"Val={len(df_val)} (pos={y_val.sum()}, {y_val.mean():.1%}), "
          f"Test={len(df_test)} (pos={y_test.sum()}, {y_test.mean():.1%})")

    neg_train, pos_train = np.bincount(y_train)
    imbalance_ratio = neg_train / pos_train
    print(f"Class Imbalance Ratio (Neg:Pos) = {imbalance_ratio:.2f} : 1")

    # Fit Preprocessors on Train Split ONLY
    prep_full = TabularPreprocessor(full_cont, full_bin).fit(df_train)
    prep_shared = TabularPreprocessor(shared_cont, shared_bin).fit(df_train)

    X_train_full = prep_full.transform(df_train)
    X_val_full = prep_full.transform(df_val)
    X_test_full = prep_full.transform(df_test)

    X_train_shared = prep_shared.transform(df_train)
    X_val_shared = prep_shared.transform(df_val)
    X_test_shared = prep_shared.transform(df_test)

    X_brfss_shared = prep_shared.transform(brfss_df)
    y_brfss = brfss_df["diabetes"].values

    # Save Preprocessors
    joblib.dump(prep_full, models_dir / "preprocessor_full.joblib")
    joblib.dump(prep_shared, models_dir / "preprocessor_shared.joblib")

    # Save Split Indices
    split_info = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_n": len(train_idx),
        "val_n": len(val_idx),
        "test_n": len(test_idx),
    }
    with open(models_dir / "split_indices.json", "w") as f:
        json.dump(split_info, f)

    # 3. Model Training Function
    def train_model_zoo(X_tr, y_tr, X_v, y_v, X_te, y_te, feature_set_name: str, input_dim: int):
        print(f"\n---> Training Model Zoo for Feature Set: [{feature_set_name}] (dim={input_dim}) <---")
        models = {}
        val_probs = {}
        test_probs = {}

        # 3.1 Logistic Regression
        lr = LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=SEED)
        lr.fit(X_tr, y_tr)
        models["Logistic Regression"] = lr

        # 3.2 Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=6,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        models["Random Forest"] = rf

        # 3.3 XGBoost
        xgb = XGBClassifier(
            scale_pos_weight=imbalance_ratio,
            n_estimators=180, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=SEED, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        models["XGBoost"] = xgb

        # 3.4 SVM (LinearSVC with ProbabilityCalibrator)
        linear_svc = LinearSVC(class_weight="balanced", max_iter=2000, random_state=SEED)
        linear_svc.fit(X_tr, y_tr)
        # Decision function to pseudo-probabilities via Platt scaling
        class SVMPseudoProb:
            def __init__(self, svc):
                self.svc = svc
                self.scaler = LogisticRegression()

            def fit(self, X_val, y_val):
                df = self.svc.decision_function(X_val).reshape(-1, 1)
                self.scaler.fit(df, y_val)
                return self

            def predict_proba(self, X):
                df = self.svc.decision_function(X).reshape(-1, 1)
                return self.scaler.predict_proba(df)

            def predict(self, X, threshold=0.5):
                return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

        svm_cal = SVMPseudoProb(linear_svc).fit(X_v, y_v)
        models["SVM (Linear)"] = svm_cal

        # 3.5 PyTorch MLP
        mlp = PyTorchMLPWrapper(input_dim=input_dim, pos_weight=imbalance_ratio, epochs=35, lr=0.001)
        mlp.fit(X_tr, y_tr, X_v, y_v)
        models["MLP (PyTorch)"] = mlp

        # 3.6 TabNet
        tabnet_success = False
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet = TabNetClassifier(
                n_d=8, n_a=8, n_steps=3, gamma=1.3,
                lambda_sparse=1e-3, verbose=0, seed=SEED
            )
            tabnet.fit(
                X_tr, y_tr,
                eval_set=[(X_v, y_v)],
                eval_metric=["auc"],
                max_epochs=25, patience=8,
                batch_size=256, virtual_batch_size=128
            )
            models["TabNet"] = tabnet
            tabnet_success = True
            print("  [TabNet] Trained successfully.")
        except Exception as e:
            print(f"  [TabNet] Skipped due to runtime error: {e}")

        # Compute Val and Test probabilities
        for name, m in models.items():
            val_probs[name] = m.predict_proba(X_v)[:, 1]
            test_probs[name] = m.predict_proba(X_te)[:, 1]

        # 3.7 Stacking Ensemble (Out-Of-Fold on Train Split Only)
        print("  [Stacking Ensemble] Generating 5-Fold Out-Of-Fold probabilities on Train...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        base_names = ["Random Forest", "XGBoost", "MLP (PyTorch)"]

        oof_matrix = np.zeros((len(X_tr), len(base_names)))
        for fold, (tr_fold_idx, val_fold_idx) in enumerate(skf.split(X_tr, y_tr)):
            fold_X_tr, fold_y_tr = X_tr[tr_fold_idx], y_tr[tr_fold_idx]
            fold_X_val, fold_y_val = X_tr[val_fold_idx], y_tr[val_fold_idx]

            fold_rf = RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1).fit(fold_X_tr, fold_y_tr)
            fold_xgb = XGBClassifier(scale_pos_weight=imbalance_ratio, n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(fold_X_tr, fold_y_tr)
            fold_mlp = PyTorchMLPWrapper(input_dim=input_dim, pos_weight=imbalance_ratio, epochs=25, lr=0.001).fit(fold_X_tr, fold_y_tr, fold_X_val, fold_y_val)

            oof_matrix[val_fold_idx, 0] = fold_rf.predict_proba(fold_X_val)[:, 1]
            oof_matrix[val_fold_idx, 1] = fold_xgb.predict_proba(fold_X_val)[:, 1]
            oof_matrix[val_fold_idx, 2] = fold_mlp.predict_proba(fold_X_val)[:, 1]

        # Meta-learner on OOF predictions
        meta_learner = LogisticRegression(random_state=SEED)
        meta_learner.fit(oof_matrix, y_tr)

        # Class for clean prediction with meta-learner
        class StackingPredictor:
            def __init__(self, base_models, meta_clf, base_keys):
                self.base_models = base_models
                self.meta_clf = meta_clf
                self.base_keys = base_keys

            def predict_proba(self, X):
                meta_feats = np.column_stack([self.base_models[k].predict_proba(X)[:, 1] for k in self.base_keys])
                return self.meta_clf.predict_proba(meta_feats)

            def predict(self, X, threshold=0.5):
                return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

        stacking_model = StackingPredictor(models, meta_learner, base_names)
        val_probs["Stacking Ensemble"] = stacking_model.predict_proba(X_v)[:, 1]
        test_probs["Stacking Ensemble"] = stacking_model.predict_proba(X_te)[:, 1]
        models["Stacking Ensemble"] = stacking_model

        # 3.8 Comparison with SMOTE (Train Split Only)
        print("  [SMOTE Comparison] Evaluating SMOTE on Train split only...")
        smote = SMOTE(random_state=SEED)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
        xgb_smote = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.04, eval_metric="logloss", random_state=SEED, n_jobs=-1)
        xgb_smote.fit(X_tr_smote, y_tr_smote)
        val_probs["XGBoost (SMOTE)"] = xgb_smote.predict_proba(X_v)[:, 1]
        test_probs["XGBoost (SMOTE)"] = xgb_smote.predict_proba(X_te)[:, 1]
        models["XGBoost (SMOTE)"] = xgb_smote

        return models, val_probs, test_probs

    # Train (a) Full Features
    full_models, full_val_probs, full_test_probs = train_model_zoo(
        X_train_full, y_train, X_val_full, y_val, X_test_full, y_test,
        feature_set_name="Full Features (10)", input_dim=X_train_full.shape[1]
    )

    # Train (b) Shared-7 Features
    shared_models, shared_val_probs, shared_test_probs = train_model_zoo(
        X_train_shared, y_train, X_val_shared, y_val, X_test_shared, y_test,
        feature_set_name="Shared-7 Features", input_dim=X_train_shared.shape[1]
    )

    # ==============================================================================
    # Threshold Tuning & Calibration on Validation Split
    # ==============================================================================
    print("\n--- [Step 4] Validation Set Threshold Tuning & Calibration ---")

    best_full_model_name = "XGBoost"
    uncal_val = full_val_probs["XGBoost"]
    uncal_test = full_test_probs["XGBoost"]

    # Calibrate probability on validation split using Isotonic Regression
    calibrator = ProbabilityCalibrator(full_models["XGBoost"], method="isotonic")
    calibrator.fit(X_val_full, y_val)
    cal_val_prob = calibrator.predict_proba(X_val_full)[:, 1]
    cal_test_prob = calibrator.predict_proba(X_test_full)[:, 1]

    # Search optimal decision threshold on validation set (maximizing Youden's J)
    thresholds = np.linspace(0.05, 0.90, 86)
    best_thresh = 0.5
    best_j = -1.0
    for t in thresholds:
        preds = (cal_val_prob >= t).astype(int)
        cm_val = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm_val.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        j_score = tpr + tnr - 1.0
        if j_score > best_j:
            best_j = j_score
            best_thresh = float(t)

    print(f"Optimal Decision Threshold selected on Validation set: {best_thresh:.3f} (Youden J = {best_j:.3f})")

    # Brier scores before and after calibration
    brier_before = float(calculate_metrics(y_test, (uncal_test >= 0.5).astype(int), uncal_test)["brier_score"])
    brier_after = float(calculate_metrics(y_test, (cal_test_prob >= best_thresh).astype(int), cal_test_prob)["brier_score"])
    print(f"XGBoost Test Brier Score: Before Calibration = {brier_before:.4f}, After Calibration = {brier_after:.4f}")

    # Plot calibration curves
    plot_calibration(y_test, uncal_test, cal_test_prob, figures_dir, "XGBoost Full Features")

    # Risk Levels Cut-Offs: Low / Moderate / High
    T_LOW = 0.15
    T_HIGH = 0.40

    def categorize_risk(probs):
        risk = np.full(len(probs), "Low     ")
        risk[(probs >= T_LOW) & (probs < T_HIGH)] = "Moderate"
        risk[probs >= T_HIGH] = "High    "
        return risk

    test_risk = categorize_risk(cal_test_prob)
    risk_df = pd.DataFrame({"risk": test_risk, "true": y_test})
    risk_summary = []
    print("\n[Risk Level Calibration on NHANES Held-Out Test Set]")
    for r_level in ["Low     ", "Moderate", "High    "]:
        sub = risk_df[risk_df["risk"] == r_level]
        share = len(sub) / len(risk_df)
        obs_rate = sub["true"].mean() if len(sub) > 0 else 0.0
        risk_summary.append({
            "risk_level": r_level.strip(),
            "n_patients": len(sub),
            "share_of_test": share,
            "observed_diabetes_rate": obs_rate
        })
        print(f"  Risk Level {r_level.strip():<8}: N={len(sub):<5} ({share*100:>5.1f}% of cohort) | Observed Diabetes Rate = {obs_rate*100:>5.1f}%")

    with open(models_dir / "risk_thresholds.json", "w") as f:
        json.dump({
            "threshold_low": T_LOW,
            "threshold_high": T_HIGH,
            "decision_threshold": best_thresh,
            "risk_summary": risk_summary
        }, f, indent=2)

    # Save final best calibrated models
    joblib.dump(calibrator, models_dir / "final_calibrated_xgb_full.joblib")
    joblib.dump(full_models["Random Forest"], models_dir / "rf_full.joblib")
    joblib.dump(full_models["XGBoost"], models_dir / "xgb_full.joblib")

    # Also calibrate shared-7 model for BRFSS external validation
    shared_calibrator = ProbabilityCalibrator(shared_models["XGBoost"], method="isotonic")
    shared_calibrator.fit(X_val_shared, y_val)
    joblib.dump(shared_calibrator, models_dir / "final_calibrated_xgb_shared.joblib")
    joblib.dump(shared_models["XGBoost"], models_dir / "xgb_shared.joblib")

    # ==============================================================================
    # Evaluation on Held-Out NHANES Test Set & BRFSS External Validation
    # ==============================================================================
    print("\n--- [Step 5] Evaluation & 95% Bootstrap Confidence Intervals ---")

    eval_records_full = []
    eval_records_shared = []
    eval_records_brfss = []

    # Trivial baseline (always predict 0)
    triv_pred = np.zeros(len(y_test), dtype=int)
    triv_prob = np.full(len(y_test), 0.0)
    triv_metrics = calculate_metrics(y_test, triv_pred, triv_prob)
    triv_ci = bootstrap_ci(y_test, triv_prob, threshold=0.5, n_bootstraps=1000, seed=SEED)
    eval_records_full.append({
        "Model": "Trivial Baseline (All No)",
        "Accuracy": f"{triv_metrics['accuracy']:.3f} [{triv_ci['accuracy']['ci_lower']:.3f}-{triv_ci['accuracy']['ci_upper']:.3f}]",
        "Balanced Acc": f"{triv_metrics['balanced_accuracy']:.3f}",
        "Precision": f"{triv_metrics['precision']:.3f}",
        "Recall": f"{triv_metrics['recall']:.3f} [{triv_ci['recall']['ci_lower']:.3f}-{triv_ci['recall']['ci_upper']:.3f}]",
        "F1": f"{triv_metrics['f1']:.3f}",
        "ROC-AUC": f"{triv_metrics['roc_auc']:.3f} [{triv_ci['roc_auc']['ci_lower']:.3f}-{triv_ci['roc_auc']['ci_upper']:.3f}]",
        "PR-AUC": f"{triv_metrics['pr_auc']:.3f} [{triv_ci['pr_auc']['ci_lower']:.3f}-{triv_ci['pr_auc']['ci_upper']:.3f}]",
        "Brier": f"{triv_metrics['brier_score']:.3f}",
    })

    # Evaluate all models on Full Features (NHANES Test)
    all_models_eval = [
        ("Logistic Regression", full_test_probs["Logistic Regression"]),
        ("Random Forest", full_test_probs["Random Forest"]),
        ("XGBoost", full_test_probs["XGBoost"]),
        ("SVM (Linear)", full_test_probs["SVM (Linear)"]),
        ("MLP (PyTorch)", full_test_probs["MLP (PyTorch)"]),
        ("Stacking Ensemble", full_test_probs["Stacking Ensemble"]),
        ("XGBoost (SMOTE)", full_test_probs["XGBoost (SMOTE)"]),
        ("Final Calibrated XGBoost", cal_test_prob),
    ]
    if "TabNet" in full_test_probs:
        all_models_eval.insert(5, ("TabNet", full_test_probs["TabNet"]))

    for name, probs in all_models_eval:
        thresh = best_thresh if "Calibrated" in name else 0.5
        preds = (probs >= thresh).astype(int)
        m = calculate_metrics(y_test, preds, probs)
        ci = bootstrap_ci(y_test, probs, threshold=thresh, n_bootstraps=1000, seed=SEED)

        # Automatic Leakage Audit
        if m["roc_auc"] > 0.90:
            print(f"  [LEAKAGE WARNING] Model {name} has ROC-AUC = {m['roc_auc']:.4f} > 0.90! Investigating...")

        eval_records_full.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.3f} [{ci['accuracy']['ci_lower']:.3f}-{ci['accuracy']['ci_upper']:.3f}]",
            "Balanced Acc": f"{m['balanced_accuracy']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f} [{ci['recall']['ci_lower']:.3f}-{ci['recall']['ci_upper']:.3f}]",
            "F1": f"{m['f1']:.3f}",
            "ROC-AUC": f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            "PR-AUC": f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]",
            "Brier": f"{m['brier_score']:.3f}",
        })

        log_experiment(exp_csv, {
            "feature_set": "Full Features (10)",
            "model": name,
            "threshold": thresh,
            **m
        })

    # Evaluate Shared-7 Models on NHANES Test Set
    shared_cal_test_prob = shared_calibrator.predict_proba(X_test_shared)[:, 1]
    shared_models_eval = [
        ("Logistic Regression (Shared)", shared_test_probs["Logistic Regression"]),
        ("Random Forest (Shared)", shared_test_probs["Random Forest"]),
        ("XGBoost (Shared)", shared_test_probs["XGBoost"]),
        ("SVM (Linear, Shared)", shared_test_probs["SVM (Linear)"]),
        ("MLP (PyTorch, Shared)", shared_test_probs["MLP (PyTorch)"]),
        ("Stacking Ensemble (Shared)", shared_test_probs["Stacking Ensemble"]),
        ("Final Calibrated XGB (Shared)", shared_cal_test_prob),
    ]

    for name, probs in shared_models_eval:
        thresh = best_thresh if "Calibrated" in name else 0.5
        preds = (probs >= thresh).astype(int)
        m = calculate_metrics(y_test, preds, probs)
        ci = bootstrap_ci(y_test, probs, threshold=thresh, n_bootstraps=1000, seed=SEED)

        eval_records_shared.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.3f} [{ci['accuracy']['ci_lower']:.3f}-{ci['accuracy']['ci_upper']:.3f}]",
            "Balanced Acc": f"{m['balanced_accuracy']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f} [{ci['recall']['ci_lower']:.3f}-{ci['recall']['ci_upper']:.3f}]",
            "F1": f"{m['f1']:.3f}",
            "ROC-AUC": f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            "PR-AUC": f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]",
            "Brier": f"{m['brier_score']:.3f}",
        })

        log_experiment(exp_csv, {
            "feature_set": "Shared-7 Features",
            "model": name,
            "threshold": thresh,
            **m
        })

    # External Validation on BRFSS (N = 249,049)
    print("\n--- [Step 6] External Validation on BRFSS Dataset ---")
    brfss_probs_dict = {}
    for name, sm in [
        ("Logistic Regression (Shared)", shared_models["Logistic Regression"]),
        ("Random Forest (Shared)", shared_models["Random Forest"]),
        ("XGBoost (Shared)", shared_models["XGBoost"]),
        ("SVM (Linear, Shared)", shared_models["SVM (Linear)"]),
        ("MLP (PyTorch, Shared)", shared_models["MLP (PyTorch)"]),
        ("Final Calibrated XGB (Shared)", shared_calibrator)
    ]:
        print(f"  Evaluating {name} on BRFSS...")
        b_prob = sm.predict_proba(X_brfss_shared)[:, 1]
        brfss_probs_dict[name] = b_prob
        thresh = best_thresh if "Calibrated" in name else 0.5
        b_pred = (b_prob >= thresh).astype(int)
        m = calculate_metrics(y_brfss, b_pred, b_prob)
        ci = bootstrap_ci(y_brfss, b_prob, threshold=thresh, n_bootstraps=1000, seed=SEED)

        eval_records_brfss.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.3f} [{ci['accuracy']['ci_lower']:.3f}-{ci['accuracy']['ci_upper']:.3f}]",
            "Balanced Acc": f"{m['balanced_accuracy']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f} [{ci['recall']['ci_lower']:.3f}-{ci['recall']['ci_upper']:.3f}]",
            "F1": f"{m['f1']:.3f}",
            "ROC-AUC": f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            "PR-AUC": f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]",
            "Brier": f"{m['brier_score']:.3f}",
        })

        log_experiment(exp_csv, {
            "feature_set": "BRFSS External Validation",
            "model": name,
            "threshold": thresh,
            **m
        })

    # Save Tables
    df_eval_full = pd.DataFrame(eval_records_full)
    df_eval_shared = pd.DataFrame(eval_records_shared)
    df_eval_brfss = pd.DataFrame(eval_records_brfss)

    df_eval_full.to_csv(tables_dir / "nhanes_full_test_comparison.csv", index=False)
    df_eval_shared.to_csv(tables_dir / "nhanes_shared_test_comparison.csv", index=False)
    df_eval_brfss.to_csv(tables_dir / "brfss_external_validation_comparison.csv", index=False)

    # Save combined comparison table
    combined_records = (
        [{"Cohort": "NHANES Test (Full 10)", **r} for r in eval_records_full] +
        [{"Cohort": "NHANES Test (Shared 7)", **r} for r in eval_records_shared] +
        [{"Cohort": "BRFSS External Validation", **r} for r in eval_records_brfss]
    )
    pd.DataFrame(combined_records).to_csv(tables_dir / "model_comparison.csv", index=False)

    # Plots
    plot_evaluation_curves(y_test, {k: v for k, v in full_test_probs.items() if "SMOTE" not in k}, figures_dir, "NHANES_Full")
    plot_evaluation_curves(y_test, {k: v for k, v in shared_test_probs.items()}, figures_dir, "NHANES_Shared")
    plot_evaluation_curves(y_brfss, brfss_probs_dict, figures_dir, "BRFSS_External")

    # Confusion Matrices
    final_pred_nhanes = (cal_test_prob >= best_thresh).astype(int)
    plot_confusion_matrix(y_test, final_pred_nhanes, figures_dir, "NHANES Test - Final Calibrated XGBoost", "confusion_matrix_nhanes_test.png")

    final_pred_brfss = (brfss_probs_dict["Final Calibrated XGB (Shared)"] >= best_thresh).astype(int)
    plot_confusion_matrix(y_brfss, final_pred_brfss, figures_dir, "BRFSS External - Final Calibrated XGBoost", "confusion_matrix_brfss_external.png")

    print("\n[SUCCESS] Model training, tuning, calibration, and evaluation successfully completed!")
    return df_eval_full, df_eval_shared, df_eval_brfss, brier_before, brier_after, best_thresh, risk_summary


if __name__ == "__main__":
    run_pipeline()
