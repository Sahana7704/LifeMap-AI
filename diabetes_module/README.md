# LifeMap AI: Diabetes Risk & Explainability Module

This module provides a hybrid, non-invasive, explainable diabetes risk prediction system developed for **LifeMap AI** (preventive healthcare capstone). The system trains on CDC NHANES (2013-2018), validates externally on CDC BRFSS (2015, $N = 249,049$), applies TreeSHAP and LIME interpretability, and integrates an American Diabetes Association (ADA) rule-based clinical lab checker and lifestyle recommendation engine.

---

## 1. Directory Structure

```
diabetes_module/
├── INTEGRATION.md                     # Microservice integration guide for Node.js & Next.js
├── README.md                          # Module setup and execution documentation
├── requirements.txt                   # Dedicated Python dependencies
├── .venv-diabetes/                    # Dedicated virtual environment (Python 3.11)
├── src/
│   ├── preprocess.py                  # Raw NHANES 24 XPT merge & BRFSS preprocessing
│   ├── train_and_evaluate.py          # 8+ ML models, bootstrap CIs, calibration, thresholding
│   ├── explainability.py              # TreeSHAP summary/waterfalls & LIME patient explanations
│   ├── pima_baseline.py               # Historical Pima Indians diabetes baseline benchmark
│   ├── app.py                         # Standalone FastAPI service with /predict endpoint
│   ├── test_app.py                    # API verification suite (3 clinical test cases)
│   └── utils.py                       # Common metrics, bootstrap CIs, and seed utilities
├── models/
│   ├── preprocessor_full.joblib       # Train-fitted imputer/scaler (10 features)
│   ├── preprocessor_shared.joblib     # Train-fitted imputer/scaler (7 shared features)
│   ├── xgb_full.joblib                # Raw XGBoost model (Full features)
│   ├── xgb_shared.joblib              # Raw XGBoost model (Shared features)
│   ├── final_calibrated_xgb_full.joblib   # Isotonic-calibrated XGBoost (Full features)
│   ├── final_calibrated_xgb_shared.joblib # Isotonic-calibrated XGBoost (Shared features)
│   ├── rf_full.joblib                 # Random Forest model (Used for TreeSHAP)
│   ├── risk_thresholds.json           # Optimal decision and risk stratification cutoffs
│   └── split_indices.json             # Stratified train/val/test split indices
├── reports/
│   ├── summary.md                     # Comprehensive technical capstone report
│   ├── experiments.csv                # Complete experiment tracking log
│   ├── figures/                       # Generated ROC, PR, calibration, SHAP & LIME plots
│   └── tables/                        # Model comparison CSVs, data dictionary & API JSON
└── experiments/
    └── pima_baseline/                 # Pima baseline results and serialized model
```

---

## 2. Installation & Environment Setup

This module runs in a dedicated virtual environment isolated from the rest of LifeMap AI:

```powershell
# From repo root
cd diabetes_module
python -m venv .venv-diabetes
.\.venv-diabetes\Scripts\pip.exe install -r requirements.txt
```

---

## 3. Execution Pipeline (Order of Execution)

Every phase can be reproduced end-to-end using the following commands:

### Step 1: Preprocessing & Data Extraction
Merges 24 NHANES XPT files across 3 cycles (H, I, J) and prepares BRFSS 2015:
```powershell
diabetes_module\.venv-diabetes\Scripts\python.exe diabetes_module\src\preprocess.py
```
*Outputs: `data/processed/nhanes_diabetes_processed.csv` ($N = 17,468$) and `data/processed/brfss_diabetes_processed.csv` ($N = 249,049$).*

### Step 2: Model Training, Evaluation & External Validation
Trains 8+ models on Full & Shared features, computes 1,000 bootstrap 95% CIs, optimizes thresholds, and calibrates probabilities:
```powershell
diabetes_module\.venv-diabetes\Scripts\python.exe diabetes_module\src\train_and_evaluate.py
```
*Outputs: Serialized models in `diabetes_module/models/`, metric tables in `diabetes_module/reports/tables/`, and performance curves in `diabetes_module/reports/figures/`.*

### Step 3: Interpretability & Explainability
Generates global TreeSHAP rankings, patient waterfall plots, and LIME surrogate models:
```powershell
diabetes_module\.venv-diabetes\Scripts\python.exe diabetes_module\src\explainability.py
```
*Outputs: `shap_summary.png`, patient waterfalls, LIME plots, and `shap_lime_comparison.md`.*

### Step 4: Historical Pima Baseline Experiment
Evaluates baseline random forest on the historical Pima Indians Diabetes dataset:
```powershell
diabetes_module\.venv-diabetes\Scripts\python.exe diabetes_module\src\pima_baseline.py
```

### Step 5: Verification & API Testing
Executes test cases against the FastAPI microservice:
```powershell
diabetes_module\.venv-diabetes\Scripts\python.exe diabetes_module\src\test_app.py
```

---

## 4. Launching the FastAPI Microservice

To run the standalone API server:
```powershell
diabetes_module\.venv-diabetes\Scripts\python.exe -m uvicorn app:app --app-dir diabetes_module/src --host 0.0.0.0 --port 8005 --reload
```
Interactive Swagger documentation is available at `http://localhost:8005/docs`.
See `INTEGRATION.md` for complete Node.js backend and Next.js frontend integration details.
