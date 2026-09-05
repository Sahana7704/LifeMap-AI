import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def train_diabetes_models():
    print("\n==========================================")
    print("1. TRAINING DIABETES RISK PREDICTION MODELS")
    print("==========================================")
    csv_path = os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Clean gender: filter out 'Other' (< 0.02% of data) to maintain clean binary/standard categories
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    # Features and target
    X = df.drop(columns=["diabetes"])
    y = df["diabetes"].astype(int)

    categorical_cols = ["gender", "smoking_history"]
    numeric_cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ]
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Get transformed feature names
    cat_feature_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + cat_feature_names

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=2.5, random_state=42, eval_metric="logloss")
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        print(f"\nTraining {name} on Diabetes dataset...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc)
        }
        print(f"Results for {name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n--> Selected Winning Diabetes Model: {best_model_name} (F1: {best_f1:.4f})")
    winning_model = models[best_model_name]

    # Initialize SHAP explainer
    print("Fitting SHAP TreeExplainer for Diabetes Model...")
    sample_background = X_train[:500]
    if hasattr(winning_model, "get_booster") or isinstance(winning_model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(winning_model)
    else:
        explainer = shap.LinearExplainer(winning_model, sample_background)

    # Save artifacts
    diabetes_bundle = {
        "model": winning_model,
        "model_name": best_model_name,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "metrics": results[best_model_name],
        "all_metrics": results,
        "expected_value": float(np.atleast_1d(explainer.expected_value)[0]) if isinstance(explainer.expected_value, (np.ndarray, list)) else float(explainer.expected_value)
    }

    joblib.dump(diabetes_bundle, os.path.join(MODELS_DIR, "diabetes_model.joblib"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "diabetes_shap_explainer.joblib"))
    print(f"Saved diabetes model bundle to {os.path.join(MODELS_DIR, 'diabetes_model.joblib')}")
    return diabetes_bundle

def train_heart_disease_models():
    print("\n================================================")
    print("2. TRAINING CARDIOVASCULAR RISK PREDICTION MODELS")
    print("================================================")
    csv_path = os.path.join(DATA_DIR, "heart_disease_uci.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Clean target: num > 0 indicates presence of heart disease
    df["target"] = (df["num"] > 0).astype(int)

    # Select relevant features
    features_to_use = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
    df_clean = df[features_to_use + ["target"]].copy()

    # Convert boolean/string representations
    df_clean["sex"] = df_clean["sex"].astype(str)
    df_clean["fbs"] = df_clean["fbs"].fillna(False).astype(str)
    df_clean["exang"] = df_clean["exang"].fillna(False).astype(str)
    df_clean["cp"] = df_clean["cp"].fillna("asymptomatic").astype(str)
    df_clean["restecg"] = df_clean["restecg"].fillna("normal").astype(str)
    df_clean["slope"] = df_clean["slope"].fillna("flat").astype(str)
    df_clean["thal"] = df_clean["thal"].fillna("normal").astype(str)

    # Impute numeric 0s in chol and trestbps where clinically impossible
    df_clean.loc[df_clean["chol"] == 0, "chol"] = np.nan
    df_clean.loc[df_clean["trestbps"] == 0, "trestbps"] = np.nan

    X = df_clean[features_to_use]
    y = df_clean["target"]

    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ]
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    cat_feature_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + cat_feature_names

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=1.2, random_state=42, eval_metric="logloss")
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        print(f"\nTraining {name} on Heart Disease dataset...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc)
        }
        print(f"Results for {name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n--> Selected Winning Heart Disease Model: {best_model_name} (F1: {best_f1:.4f})")
    winning_model = models[best_model_name]

    print("Fitting SHAP TreeExplainer for Heart Disease Model...")
    sample_background = X_train[:300]
    if hasattr(winning_model, "get_booster") or isinstance(winning_model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(winning_model)
    else:
        explainer = shap.LinearExplainer(winning_model, sample_background)

    heart_bundle = {
        "model": winning_model,
        "model_name": best_model_name,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "metrics": results[best_model_name],
        "all_metrics": results,
        "expected_value": float(np.atleast_1d(explainer.expected_value)[0]) if isinstance(explainer.expected_value, (np.ndarray, list)) else float(explainer.expected_value)
    }

    joblib.dump(heart_bundle, os.path.join(MODELS_DIR, "heart_disease_model.joblib"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "heart_shap_explainer.joblib"))
    print(f"Saved heart disease model bundle to {os.path.join(MODELS_DIR, 'heart_disease_model.joblib')}")
    return heart_bundle

if __name__ == "__main__":
    d_res = train_diabetes_models()
    h_res = train_heart_disease_models()

    summary = {
        "diabetes": {
            "selected_model": d_res["model_name"],
            "metrics": d_res["metrics"],
            "comparison": d_res["all_metrics"]
        },
        "cardiovascular": {
            "selected_model": h_res["model_name"],
            "metrics": h_res["metrics"],
            "comparison": h_res["all_metrics"]
        }
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining summary written to models/training_summary.json successfully!")
