"""
Prepares evaluation cache and builds diabetes_module/notebooks/results_report.ipynb.
Then executes the notebook and exports to diabetes_module/reports/results_report.html.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess
import sys

# Ensure src is in sys.path
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
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = MODULE_DIR / "notebooks"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def run_cache_and_plots():
    print("1. Loading processed data and artifacts...")
    nhanes_df = pd.read_csv(DATA_DIR / "nhanes_diabetes_processed.csv")
    brfss_df = pd.read_csv(DATA_DIR / "brfss_diabetes_processed.csv")
    
    with open(MODELS_DIR / "split_indices.json", "r") as f:
        splits = json.load(f)
    train_idx = splits.get("train_idx", splits.get("train_indices"))
    val_idx = splits.get("val_idx", splits.get("val_indices"))
    test_idx = splits.get("test_idx", splits.get("test_indices"))
    
    prep_full = joblib.load(MODELS_DIR / "preprocessor_full.joblib")
    
    full_cols = [c for c in nhanes_df.columns if c not in ["SEQN", "diabetes"]]
    X_train_raw = nhanes_df.loc[train_idx, full_cols]
    y_train = nhanes_df.loc[train_idx, "diabetes"].values
    X_val_raw = nhanes_df.loc[val_idx, full_cols]
    y_val = nhanes_df.loc[val_idx, "diabetes"].values
    X_test_raw = nhanes_df.loc[test_idx, full_cols]
    y_test = nhanes_df.loc[test_idx, "diabetes"].values
    
    X_train = prep_full.transform(X_train_raw)
    X_val = prep_full.transform(X_val_raw)
    X_test = prep_full.transform(X_test_raw)
    
    feature_names = prep_full.continuous_cols + prep_full.binary_cols
    
    print("2. Re-evaluating models for predictions and curves...")
    # Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_test)[:, 1]
    
    # Random Forest
    rf = joblib.load(MODELS_DIR / "rf_full.joblib")
    prob_rf = rf.predict_proba(X_test)[:, 1]
    
    # XGBoost
    xgb = joblib.load(MODELS_DIR / "xgb_full.joblib")
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # Calibrated XGBoost
    cal_xgb = joblib.load(MODELS_DIR / "final_calibrated_xgb_full.joblib")
    prob_cal_xgb = cal_xgb.predict_proba(X_test)[:, 1]
    
    # Linear SVM
    svm = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000)
    svm.fit(X_train, y_train)
    val_svm_scores = svm.decision_function(X_val)
    test_svm_scores = svm.decision_function(X_test)
    from sklearn.isotonic import IsotonicRegression
    svm_iso = IsotonicRegression(out_of_bounds="clip").fit(val_svm_scores, y_val)
    prob_svm = np.clip(svm_iso.predict(test_svm_scores), 1e-4, 1 - 1e-4)
    
    # MLP with loss tracking
    print("   Training MLP to capture loss curve...")
    device = torch.device("cpu")
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    mlp = DiabetesMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    
    mlp_train_losses = []
    mlp_val_losses = []
    
    for epoch in range(35):
        mlp.train()
        epoch_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mlp_train_losses.append(float(np.mean(epoch_losses)))
        
        mlp.eval()
        with torch.no_grad():
            v_loss = criterion(mlp(X_val_t), y_val_t).item()
            mlp_val_losses.append(float(v_loss))
            
    mlp.eval()
    with torch.no_grad():
        test_logits = mlp(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze().numpy()
        prob_mlp = 1.0 / (1.0 + np.exp(-test_logits))
        
    # TabNet
    print("   Training TabNet to capture history and feature importances...")
    tabnet = TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4, seed=SEED, verbose=0)
    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["logloss"],
        max_epochs=25,
        patience=8,
        batch_size=512,
        virtual_batch_size=128
    )
    prob_tabnet = tabnet.predict_proba(X_test)[:, 1]
    tab_hist_dict = tabnet.history.history
    tabnet_train_loss = [float(x) for x in tab_hist_dict.get("loss", [0.4])]
    tabnet_val_loss = [float(x) for x in tab_hist_dict.get("val_logloss", tab_hist_dict.get("val_0_logloss", tabnet_train_loss))]
    tabnet_feat_imp = tabnet.feature_importances_
    
    # Stacking meta-predictions
    # Approximate OOF meta learner for test set
    meta_test_features = np.column_stack([prob_lr, prob_rf, prob_xgb, prob_svm, prob_mlp])
    meta_clf = LogisticRegression(random_state=SEED)
    # Fit meta on validation predictions
    val_meta = np.column_stack([
        lr.predict_proba(X_val)[:, 1],
        rf.predict_proba(X_val)[:, 1],
        xgb.predict_proba(X_val)[:, 1],
        np.clip(svm_iso.predict(val_svm_scores), 1e-4, 1-1e-4),
        1.0 / (1.0 + np.exp(-mlp(X_val_t).squeeze().detach().numpy()))
    ])
    meta_clf.fit(val_meta, y_val)
    prob_stack = meta_clf.predict_proba(meta_test_features)[:, 1]

    models_probs = {
        "Logistic Regression": prob_lr,
        "Random Forest": prob_rf,
        "XGBoost": prob_xgb,
        "Linear SVM": prob_svm,
        "PyTorch MLP": prob_mlp,
        "TabNet": prob_tabnet,
        "Stacking Ensemble": prob_stack,
        "Final Calibrated XGBoost": prob_cal_xgb
    }
    
    # Save cache
    cache = {
        "y_test": y_test,
        "models_probs": models_probs,
        "feature_names": feature_names,
        "mlp_train_losses": mlp_train_losses,
        "mlp_val_losses": mlp_val_losses,
        "tabnet_train_loss": tabnet_train_loss,
        "tabnet_val_loss": tabnet_val_loss,
        "tabnet_feat_imp": tabnet_feat_imp.tolist()
    }
    joblib.dump(cache, MODELS_DIR / "test_eval_cache.joblib")
    print("Saved evaluation cache.")

    # Generate and save PNG figures
    print("3. Generating required PNG figures...")
    
    # 3.1 Model metrics bar chart
    df_full = pd.read_csv(TABLES_DIR / "nhanes_full_test_comparison.csv")
    # Clean numeric metrics for plotting
    plot_df = []
    for _, row in df_full.iterrows():
        name = row["Model"]
        if name == "Trivial Baseline (All No)":
            continue
        acc = float(str(row["Accuracy"]).split()[0])
        rec = float(str(row["Recall"]).split()[0])
        roc = float(str(row["ROC-AUC"]).split()[0])
        pr = float(str(row["PR-AUC"]).split()[0])
        f1 = float(row["F1"])
        plot_df.append({"Model": name, "Accuracy": acc, "Recall": rec, "ROC-AUC": roc, "PR-AUC": pr, "F1": f1})
    pdf = pd.DataFrame(plot_df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pdf.set_index("Model")[["Accuracy", "Recall", "ROC-AUC", "PR-AUC", "F1"]].plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
    ax.set_title("Performance Comparison Across Evaluated Models (NHANES Full Features Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_metrics_bar_chart.png", dpi=300)
    plt.close()

    # 3.2 Confusion matrices for all models
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    model_names_list = list(models_probs.keys())
    for idx, m_name in enumerate(model_names_list):
        ax = axes[idx]
        p = models_probs[m_name]
        thresh = 0.14 if "Calibrated" in m_name else 0.50
        y_pred = (p >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"{m_name}\n(Threshold: {thresh})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["No Diabetes", "Diabetes"])
        ax.set_yticklabels(["No Diabetes", "Diabetes"])
    plt.suptitle("Confusion Matrices Across Models on NHANES Held-Out Test Set", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_all_models.png", dpi=300)
    plt.close()

    # 3.3 Overlaid ROC & PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = sns.color_palette("tab10", len(models_probs))
    for (m_name, p), col in zip(models_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{m_name} (AUC = {roc_val:.3f})", color=col, linewidth=2)
        
        pre, rec, _ = precision_recall_curve(y_test, p)
        pr_val = auc(rec, pre)
        ax2.plot(rec, pre, label=f"{m_name} (PR = {pr_val:.3f})", color=col, linewidth=2)
        
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.500)")
    ax1.set_title("Receiver Operating Characteristic (ROC) Overlay", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5)
    
    baseline_prev = y_test.sum() / len(y_test)
    ax2.axhline(baseline_prev, color="k", linestyle="--", alpha=0.6, label=f"Baseline Prevalence ({baseline_prev:.3f})")
    ax2.set_title("Precision-Recall (PR) Overlay", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nhanes_all_models_roc_pr.png", dpi=300)
    plt.savefig(FIGURES_DIR / "nhanes_all_models_roc_curves.png", dpi=300)
    plt.savefig(FIGURES_DIR / "nhanes_all_models_pr_curves.png", dpi=300)
    plt.close()

    # 3.4 Deep Learning: MLP Training History
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(mlp_train_losses) + 1), mlp_train_losses, label="Training Loss (BCE)", color="#2563EB", linewidth=2.5)
    ax.plot(range(1, len(mlp_val_losses) + 1), mlp_val_losses, label="Validation Loss", color="#DC2626", linewidth=2.5, linestyle="--")
    ax.set_title("PyTorch Deep MLP Training & Validation Loss Dynamics", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_training_history.png", dpi=300)
    plt.close()

    # 3.5 TabNet Training History & Feature Importances
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tabnet_train_loss) + 1), tabnet_train_loss, label="TabNet Training Loss", color="#059669", linewidth=2.5)
    ax.plot(range(1, len(tabnet_val_loss) + 1), tabnet_val_loss, label="Validation LogLoss", color="#D97706", linewidth=2.5, linestyle="--")
    ax.set_title("TabNet Attention-Based Training Convergence", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("LogLoss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tabnet_training_history.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    s_idx = np.argsort(tabnet_feat_imp)
    ax.barh(np.array(feature_names)[s_idx], tabnet_feat_imp[s_idx], color="#4F46E5", edgecolor="black", alpha=0.85)
    ax.set_title("TabNet Built-in Sparse Attention Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mask Weight (Feature Importance)", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tabnet_feature_importances.png", dpi=300)
    plt.close()

    # 3.6 Side-by-side NHANES vs BRFSS comparison
    nhanes_shared = pd.read_csv(TABLES_DIR / "nhanes_shared_test_comparison.csv")
    brfss_shared = pd.read_csv(TABLES_DIR / "brfss_external_validation_comparison.csv")
    
    comp_data = []
    for _, row in nhanes_shared.iterrows():
        m = row["Model"]
        roc = float(str(row["ROC-AUC"]).split()[0])
        rec = float(str(row["Recall"]).split()[0])
        comp_data.append({"Model": m, "Cohort": "NHANES Test (Shared-7)", "ROC-AUC": roc, "Recall": rec})
    for _, row in brfss_shared.iterrows():
        m = row["Model"]
        roc = float(str(row["ROC-AUC"]).split()[0])
        rec = float(str(row["Recall"]).split()[0])
        comp_data.append({"Model": m, "Cohort": "BRFSS External (N=249k)", "ROC-AUC": roc, "Recall": rec})
    comp_df = pd.DataFrame(comp_data)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=comp_df, x="Model", y="ROC-AUC", hue="Cohort", palette=["#2563EB", "#F97316"], ax=ax)
    ax.set_title("External Generalizability: NHANES Test vs. BRFSS External Validation (ROC-AUC)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_ylim(0.65, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nhanes_vs_brfss_comparison.png", dpi=300)
    plt.close()

    # 3.7 Risk level distribution
    with open(MODELS_DIR / "risk_thresholds.json", "r") as f:
        risk_meta = json.load(f)
    r_summary = risk_meta["risk_summary"]
    tiers = [r["risk_level"] for r in r_summary]
    shares = [r["share_of_test"] * 100 for r in r_summary]
    rates = [r["observed_diabetes_rate"] * 100 for r in r_summary]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.pie(shares, labels=[f"{t}\n({s:.1f}%)" for t, s in zip(tiers, shares)], autopct="%1.1f%%",
            startangle=140, colors=["#10B981", "#F59E0B", "#EF4444"], explode=(0.02, 0.04, 0.08),
            textprops={'fontsize': 11, 'weight': 'bold'})
    ax1.set_title("Distribution of Test Population Across Risk Tiers", fontsize=12, fontweight="bold")
    
    bars = ax2.bar(tiers, rates, color=["#10B981", "#F59E0B", "#EF4444"], edgecolor="black", width=0.55)
    ax2.set_title("Observed Diabetes Rate in Each Risk Tier", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=11)
    ax2.set_ylim(0, 60)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "risk_level_distribution.png", dpi=300)
    plt.close()
    
    print("All PNG figures successfully generated and saved to reports/figures/.")


def build_and_export_notebook():
    print("\n4. Building Jupyter Notebook: diabetes_module/notebooks/results_report.ipynb...")
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell(
        "# LifeMap AI: Comprehensive Diabetes Risk Assessment & Model Evaluation Report\n"
        "**Capstone Project — Non-Invasive Explainable Diabetes Risk Stratification**  \n"
        "*Primary Development Cohort: CDC NHANES (2013–2018) | External Validation: CDC BRFSS (2015, N=249,049)*"
    ))

    # Cell 1: Setup & Styling
    cells.append(nbf.v4.new_code_cell(
        "# Setup environment & display options\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import json\n"
        "from pathlib import Path\n"
        "from IPython.display import display, HTML, Image\n"
        "from PIL import Image as PILImage\n"
        "\n"
        "pd.set_option('display.max_columns', 20)\n"
        "pd.set_option('display.precision', 4)\n"
        "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n"
        "print('Environment initialized successfully.')"
    ))

    # Cell 2: Dataset Summary Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Dataset Architecture & Summary\n"
        "Summary of the primary CDC NHANES development cohort and CDC BRFSS 2015 external validation cohort."
    ))

    # Cell 3: Dataset Summary Code
    cells.append(nbf.v4.new_code_cell(
        "nhanes_df = pd.read_csv('../../data/processed/nhanes_diabetes_processed.csv')\n"
        "brfss_df = pd.read_csv('../../data/processed/brfss_diabetes_processed.csv')\n"
        "\n"
        "n_nhanes = len(nhanes_df)\n"
        "nhanes_pos = int(nhanes_df['diabetes'].sum())\n"
        "nhanes_rate = nhanes_pos / n_nhanes * 100\n"
        "\n"
        "n_brfss = len(brfss_df)\n"
        "brfss_pos = int(brfss_df['diabetes'].sum())\n"
        "brfss_rate = brfss_pos / n_brfss * 100\n"
        "\n"
        "full_features = [c for c in nhanes_df.columns if c not in ['SEQN', 'diabetes']]\n"
        "shared_features = ['age_bucket', 'sex', 'bmi', 'high_bp', 'high_chol', 'smoker', 'phys_activity']\n"
        "\n"
        "summary_table = pd.DataFrame({\n"
        "    'Dataset Cohort': ['CDC NHANES (Primary Development)', 'CDC BRFSS 2015 (External Validation)'],\n"
        "    'Total Participants (N)': [f'{n_nhanes:,}', f'{n_brfss:,}'],\n"
        "    'Diabetes Cases': [f'{nhanes_pos:,}', f'{brfss_pos:,}'],\n"
        "    'Class Prevalence': [f'{nhanes_rate:.2f}%', f'{brfss_rate:.2f}%'],\n"
        "    'Feature Subspace': [f'{len(full_features)} Non-Lab Features', f'{len(shared_features)} Shared Survey Features'],\n"
        "    'Target Definition': ['DIQ010 (1=Yes, 2=No)', 'DIABETE3 (1=Yes, 0=No)']\n"
        "})\n"
        "display(HTML(summary_table.to_html(index=False, classes='table table-striped table-hover')))\n"
        "print(f'\\nNHANES Full Feature Set ({len(full_features)}):\\n  {full_features}')\n"
        "print(f'\\nShared-7 Feature Space ({len(shared_features)}):\\n  {shared_features}')"
    ))

    # Cell 4: Model Comparison Table Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Model Comparison Table (NHANES Held-Out Test Set — Full Features)\n"
        "Performance of all candidate architectures on the held-out test partition ($n = 2,621$).\n"
        "**The top-performing model for each column is highlighted in green.**"
    ))

    # Cell 5: Model Comparison Code
    cells.append(nbf.v4.new_code_cell(
        "table_df = pd.read_csv('../reports/tables/nhanes_full_test_comparison.csv')\n"
        "\n"
        "def parse_metric(val):\n"
        "    try:\n"
        "        return float(str(val).split()[0])\n"
        "    except:\n"
        "        return 0.0\n"
        "\n"
        "num_df = table_df.copy()\n"
        "metrics = ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'Brier']\n"
        "for m in metrics:\n"
        "    num_df[f'{m}_val'] = num_df[m].apply(parse_metric)\n"
        "\n"
        "def highlight_max_cols(s):\n"
        "    style_res = pd.DataFrame('', index=s.index, columns=s.columns)\n"
        "    mask = s['Model'] != 'Trivial Baseline (All No)'\n"
        "    green_style = 'background-color: #D1FAE5; font-weight: bold; color: #065F46;'\n"
        "    \n"
        "    for c in ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']:\n"
        "        best_idx = num_df.loc[mask, f'{c}_val'].idxmax()\n"
        "        style_res.loc[best_idx, c] = green_style\n"
        "        \n"
        "    # Brier score: lower is better\n"
        "    best_brier_idx = num_df.loc[mask, 'Brier_val'].idxmin()\n"
        "    style_res.loc[best_brier_idx, 'Brier'] = green_style\n"
        "    return style_res\n"
        "\n"
        "styled = table_df.style.apply(highlight_max_cols, axis=None).set_caption(\n"
        "    '<b>Comprehensive Model Performance (NHANES Full Features Test Set, N=2,621)</b>'\n"
        ")\n"
        "display(styled)"
    ))

    # Cell 6: Bar Chart Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Key Performance Metrics Comparison (Bar Chart)\n"
        "Side-by-side comparison across accuracy, recall, ROC-AUC, PR-AUC, and F1 score."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='../reports/figures/model_metrics_bar_chart.png'))"
    ))

    # Cell 7: Confusion Matrices Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Confusion Matrices for All Evaluated Models\n"
        "Matrix breakdown across all candidate architectures on the held-out test set."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='../reports/figures/confusion_matrices_all_models.png'))"
    ))

    # Cell 8: Overlaid ROC & PR Curves Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. ROC and Precision-Recall Curves (All Models Overlaid)\n"
        "Simultaneous comparison of discriminative curves across models."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='../reports/figures/nhanes_all_models_roc_pr.png'))"
    ))

    # Cell 9: Calibration Curves Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Probability Calibration: Before vs. After Calibration\n"
        "Reliability diagram demonstrating reduction of overconfidence via Isotonic Regression."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename='../reports/figures/calibration_curve_before_after.png'))"
    ))

    # Cell 10: Deep Learning Section Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Deep Learning Section: PyTorch Deep MLP & TabNet Sparse Attention\n"
        "Neural network architectures, training dynamics, loss convergence, and attention-based feature selection."
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('=== PyTorch Deep MLP Architecture Specification ===')\n"
        "print('Input Dimension: 10 non-lab tabular features')\n"
        "print('Layer 1: Linear(10 -> 64) -> BatchNorm1d(64) -> ReLU() -> Dropout(0.25)')\n"
        "print('Layer 2: Linear(64 -> 32) -> BatchNorm1d(32) -> ReLU() -> Dropout(0.20)')\n"
        "print('Layer 3: Linear(32 -> 16) -> ReLU()')\n"
        "print('Output Layer: Linear(16 -> 1) [BCEWithLogitsLoss with positive weighting = 6.13]')\n"
        "print('Optimization: Adam (lr=1e-3, weight_decay=1e-4) with validation early stopping\\n')\n"
        "\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))\n"
        "img_mlp = PILImage.open('../reports/figures/mlp_training_history.png')\n"
        "ax1.imshow(img_mlp)\n"
        "ax1.axis('off')\n"
        "ax1.set_title('PyTorch MLP Loss Dynamics', fontsize=12, fontweight='bold')\n"
        "\n"
        "img_tab = PILImage.open('../reports/figures/tabnet_training_history.png')\n"
        "ax2.imshow(img_tab)\n"
        "ax2.axis('off')\n"
        "ax2.set_title('TabNet Training LogLoss Convergence', fontsize=12, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print('\\n=== TabNet Sparse Attention Feature Importances ===')\n"
        "display(Image(filename='../reports/figures/tabnet_feature_importances.png'))"
    ))

    # Cell 11: BRFSS External Validation Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. External Validation: CDC BRFSS 2015 ($N = 249,049$)\n"
        "Assessing generalizability and domain shift on an independent nationwide telephone survey."
    ))
    cells.append(nbf.v4.new_code_cell(
        "brfss_table = pd.read_csv('../reports/tables/brfss_external_validation_comparison.csv')\n"
        "display(HTML('<h4>Table: External Validation Metrics (CDC BRFSS 2015, N=249,049)</h4>'))\n"
        "display(HTML(brfss_table.to_html(index=False, classes='table table-bordered table-striped')))\n"
        "\n"
        "display(Image(filename='../reports/figures/nhanes_vs_brfss_comparison.png'))"
    ))

    # Cell 12: Explainability Section Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Model Interpretability & Explainability (TreeSHAP & LIME)\n"
        "Global feature attribution ranking, local patient waterfall explanations, and surrogate LIME comparisons."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(HTML('<h3>9.1 Global TreeSHAP Feature Attributions</h3>'))\n"
        "display(Image(filename='../reports/figures/shap_summary.png'))\n"
        "\n"
        "display(HTML('<h3>9.2 Patient-Level SHAP Waterfall Explanations (Low, Moderate, High Risk)</h3>'))\n"
        "fig, axes = plt.subplots(1, 3, figsize=(22, 6))\n"
        "for idx, (p_name, title) in enumerate([\n"
        "    ('shap_patient_1_low_risk.png', 'Patient 1: Low Risk (1.8%)'),\n"
        "    ('shap_patient_2_moderate_risk.png', 'Patient 2: Moderate Risk (27.3%)'),\n"
        "    ('shap_patient_3_high_risk.png', 'Patient 3: High Risk (90.9%)')\n"
        "]):\n"
        "    im = PILImage.open(f'../reports/figures/{p_name}')\n"
        "    axes[idx].imshow(im)\n"
        "    axes[idx].axis('off')\n"
        "    axes[idx].set_title(title, fontsize=12, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "display(HTML('<h3>9.3 Patient-Level LIME Surrogate Explanations</h3>'))\n"
        "fig, axes = plt.subplots(1, 3, figsize=(22, 6))\n"
        "for idx, (l_name, title) in enumerate([\n"
        "    ('lime_patient_1.png', 'LIME: Patient 1 (Low Risk)'),\n"
        "    ('lime_patient_2.png', 'LIME: Patient 2 (Moderate Risk)'),\n"
        "    ('lime_patient_3.png', 'LIME: Patient 3 (High Risk)')\n"
        "]):\n"
        "    im = PILImage.open(f'../reports/figures/{l_name}')\n"
        "    axes[idx].imshow(im)\n"
        "    axes[idx].axis('off')\n"
        "    axes[idx].set_title(title, fontsize=12, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "with open('../reports/tables/shap_lime_comparison.md', 'r') as f:\n"
        "    display(HTML(f\"<div style='background-color:#F8FAFC; padding:15px; border-left:4px solid #3B82F6;'>{f.read()}</div>\"))"
    ))

    # Cell 13: Clinical Risk Stratification Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Clinical Decision Thresholds & Risk Level Stratification\n"
        "Distribution across Low, Moderate, and High clinical tiers and observed diabetes incidence."
    ))
    cells.append(nbf.v4.new_code_cell(
        "with open('../models/risk_thresholds.json', 'r') as f:\n"
        "    risk_info = json.load(f)\n"
        "\n"
        "print(f'Optimal Decision Screening Threshold: {risk_info[\"decision_threshold\"]:.3f}')\n"
        "print(f'Low Risk Tier Cut-off:      < {risk_info[\"threshold_low\"]:.2f}')\n"
        "print(f'Moderate Risk Tier Cut-off: {risk_info[\"threshold_low\"]:.2f} to {risk_info[\"threshold_high\"]:.2f}')\n"
        "print(f'High Risk Tier Cut-off:     >= {risk_info[\"threshold_high\"]:.2f}\\n')\n"
        "\n"
        "r_df = pd.DataFrame(risk_info['risk_summary'])\n"
        "r_df['share_of_test'] = r_df['share_of_test'].apply(lambda x: f'{x*100:.2f}%')\n"
        "r_df['observed_diabetes_rate'] = r_df['observed_diabetes_rate'].apply(lambda x: f'{x*100:.2f}%')\n"
        "r_df.columns = ['Risk Tier Level', 'Test Patients (N)', 'Share of Test Cohort', 'Observed Diabetes Rate']\n"
        "display(HTML(r_df.to_html(index=False, classes='table table-bordered table-hover')))\n"
        "\n"
        "display(Image(filename='../reports/figures/risk_level_distribution.png'))"
    ))

    nb["cells"] = cells

    nb_path = NOTEBOOKS_DIR / "results_report.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved notebook structure to {nb_path}.")

    print("5. Executing notebook in-place...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_to_run = nbf.read(f, as_version=4)
    ep.preprocess(nb_to_run, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb_to_run, f)
    print("Notebook successfully executed with all outputs embedded.")

    print("6. Exporting self-contained HTML report (embedded images)...")
    html_out = REPORTS_DIR / "results_report.html"
    cmd = [
        str(MODULE_DIR / ".venv-diabetes" / "Scripts" / "jupyter-nbconvert.exe"),
        "--to", "html",
        "--embed-images",
        str(nb_path),
        "--output", "results_report.html",
        "--output-dir", str(REPORTS_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully generated HTML report at {html_out} (Size: {html_out.stat().st_size:,} bytes).")
    else:
        print(f"Warning: HTML export returned code {res.returncode}:\n{res.stderr}")


if __name__ == "__main__":
    run_cache_and_plots()
    build_and_export_notebook()

