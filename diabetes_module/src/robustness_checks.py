"""
Robustness checks for LifeMap AI Diabetes Module (Evaluation Part E8).
Computes:
1. 5-Fold Stratified Cross-Validation on the NHANES training split.
2. Temporal Split: Training on 2013-2016 (Cycles H & I), testing on 2017-2018 (Cycle J).
3. Complete-Case Analysis vs. Standard Imputed Pipeline on the test split.

Outputs saved to diabetes_module/reports/paper/tables/robustness_checks.csv and .tex.
Seed 42 everywhere.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

import sys
src_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(src_dir))

from train_and_evaluate import TabularPreprocessor

SEED = 42
np.random.seed(SEED)


def get_models():
    """Instantiate Logistic Regression and XGBoost with saved hyperparameters."""
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs"
    )
    xgb = XGBClassifier(
        colsample_bytree=0.8,
        eval_metric="logloss",
        learning_rate=0.04,
        max_depth=4,
        n_estimators=180,
        n_jobs=-1,
        random_state=SEED,
        scale_pos_weight=6.133605600933489,
        subsample=0.8
    )
    return {"Logistic Regression": lr, "XGBoost": xgb}


def run_robustness_checks():
    base_dir = src_dir.parent
    data_path = base_dir.parent / "data" / "processed" / "nhanes_diabetes_processed.csv"
    splits_path = base_dir / "models" / "split_indices.json"
    out_dir = base_dir / "reports" / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    with open(splits_path, "r") as f:
        splits = json.load(f)

    cont_cols = ["age", "bmi", "mean_sbp", "mean_dbp"]
    bin_cols = ["sex", "high_bp", "high_chol", "smoker", "phys_activity", "family_history"]
    feat_cols = cont_cols + bin_cols

    train_df = df.iloc[splits["train_indices"]].reset_index(drop=True)
    val_df = df.iloc[splits["val_indices"]].reset_index(drop=True)
    test_df = df.iloc[splits["test_indices"]].reset_index(drop=True)

    results = []

    # --------------------------------------------------------------------------
    # Check A: 5-Fold Stratified CV on Training Split
    # --------------------------------------------------------------------------
    print("Running 5-fold Stratified CV on training split...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = get_models()

    for m_name, clf in models.items():
        roc_list, rec_list, acc_list = [], [], []
        for f_tr_idx, f_va_idx in skf.split(train_df, train_df["diabetes"]):
            f_tr = train_df.iloc[f_tr_idx]
            f_va = train_df.iloc[f_va_idx]
            prep = TabularPreprocessor(cont_cols, bin_cols).fit(f_tr)
            X_tr, y_tr = prep.transform(f_tr), f_tr["diabetes"].values
            X_va, y_va = prep.transform(f_va), f_va["diabetes"].values

            # Fit freshly instantiated model
            model_inst = get_models()[m_name]
            model_inst.fit(X_tr, y_tr)

            p_va = model_inst.predict_proba(X_va)[:, 1]
            roc_list.append(roc_auc_score(y_va, p_va))

            # 80% recall operating point from fold train
            p_tr_pos = model_inst.predict_proba(X_tr)[:, 1][y_tr == 1]
            fold_thresh = np.sort(p_tr_pos)[int(0.20 * len(p_tr_pos))]
            y_pred = (p_va >= fold_thresh).astype(int)
            rec_list.append(recall_score(y_va, y_pred))
            acc_list.append(accuracy_score(y_va, y_pred))

        roc_str = f"{np.mean(roc_list):.3f} +/- {np.std(roc_list):.3f}"
        rec_str = f"{np.mean(rec_list):.3f} +/- {np.std(rec_list):.3f}"
        acc_str = f"{np.mean(acc_list):.3f} +/- {np.std(acc_list):.3f}"

        results.append({
            "Setting": "5-Fold CV (Train split)",
            "Model": m_name,
            "ROC-AUC": roc_str,
            "Recall": rec_str,
            "Accuracy": acc_str,
            "Details": "Stratified 5-fold on n=12,227"
        })

    # --------------------------------------------------------------------------
    # Check B: Temporal Split (Cycles H+I 2013-16 -> Cycle J 2017-18)
    # --------------------------------------------------------------------------
    print("Running Temporal Split evaluation...")
    df_hi = df[df["cycle"].isin(["H", "I"])].reset_index(drop=True)
    df_j = df[df["cycle"] == "J"].reset_index(drop=True)

    prep_temp = TabularPreprocessor(cont_cols, bin_cols).fit(df_hi)
    X_hi, y_hi = prep_temp.transform(df_hi), df_hi["diabetes"].values
    X_j, y_j = prep_temp.transform(df_j), df_j["diabetes"].values

    for m_name, _ in get_models().items():
        model_inst = get_models()[m_name]
        model_inst.fit(X_hi, y_hi)

        p_j = model_inst.predict_proba(X_j)[:, 1]
        roc_temp = roc_auc_score(y_j, p_j)

        # 80% recall threshold from training cycles H+I
        p_hi_pos = model_inst.predict_proba(X_hi)[:, 1][y_hi == 1]
        t_thresh = np.sort(p_hi_pos)[int(0.20 * len(p_hi_pos))]
        y_pred_j = (p_j >= t_thresh).astype(int)

        rec_temp = recall_score(y_j, y_pred_j)
        acc_temp = accuracy_score(y_j, y_pred_j)

        results.append({
            "Setting": "Temporal Split (2013-16 -> 2017-18)",
            "Model": m_name,
            "ROC-AUC": f"{roc_temp:.3f}",
            "Recall": f"{rec_temp:.3f}",
            "Accuracy": f"{acc_temp:.3f}",
            "Details": f"Train H+I (n={len(df_hi)}) -> Test J (n={len(df_j)})"
        })

    # --------------------------------------------------------------------------
    # Check C: Complete-Case Analysis vs Imputed Pipeline
    # --------------------------------------------------------------------------
    print("Running Complete-Case Analysis...")
    train_cc = train_df.dropna(subset=feat_cols).reset_index(drop=True)
    test_cc = test_df.dropna(subset=feat_cols).reset_index(drop=True)

    for m_name, _ in get_models().items():
        # Standard Imputed Pipeline on Test Set
        prep_imp = TabularPreprocessor(cont_cols, bin_cols).fit(train_df)
        X_tr_imp, y_tr_imp = prep_imp.transform(train_df), train_df["diabetes"].values
        X_te_imp, y_te_imp = prep_imp.transform(test_df), test_df["diabetes"].values
        model_imp = get_models()[m_name].fit(X_tr_imp, y_tr_imp)
        roc_imp = roc_auc_score(y_te_imp, model_imp.predict_proba(X_te_imp)[:, 1])

        # Complete-Case Pipeline
        prep_cc = TabularPreprocessor(cont_cols, bin_cols).fit(train_cc)
        X_tr_cc, y_tr_cc = prep_cc.transform(train_cc), train_cc["diabetes"].values
        X_te_cc, y_te_cc = prep_cc.transform(test_cc), test_cc["diabetes"].values
        model_cc = get_models()[m_name].fit(X_tr_cc, y_tr_cc)
        roc_cc = roc_auc_score(y_te_cc, model_cc.predict_proba(X_te_cc)[:, 1])

        results.append({
            "Setting": "Imputed Pipeline (Standard)",
            "Model": m_name,
            "ROC-AUC": f"{roc_imp:.3f}",
            "Recall": "-",
            "Accuracy": "-",
            "Details": f"Imputed test (n={len(test_df)})"
        })
        results.append({
            "Setting": "Complete-Case Analysis",
            "Model": m_name,
            "ROC-AUC": f"{roc_cc:.3f}",
            "Recall": "-",
            "Accuracy": "-",
            "Details": f"Complete cases only (n={len(test_cc)} of {len(test_df)})"
        })

    res_df = pd.DataFrame(results)
    csv_path = out_dir / "robustness_checks.csv"
    tex_path = out_dir / "robustness_checks.tex"

    res_df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(res_df.to_latex(index=False, escape=False))

    print(f"Robustness checks completed successfully. Saved to {csv_path} and {tex_path}.")
    print(res_df)
    return res_df


if __name__ == "__main__":
    run_robustness_checks()
