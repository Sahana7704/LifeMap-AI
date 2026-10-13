"""
Pima Indians Diabetes Baseline Experiment.
Evaluates a Random Forest on the traditional Pima dataset to document historical baseline performance.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import joblib


def run_pima_baseline():
    root_dir = Path(__file__).resolve().parent.parent.parent
    pima_path = root_dir / "data" / "pima_indians_diabetes.csv"
    exp_dir = root_dir / "diabetes_module" / "experiments" / "pima_baseline"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not pima_path.exists():
        print(f"Pima dataset not found at {pima_path}. Skipping.")
        return None

    print(f"Loading Pima dataset from {pima_path}...")
    df = pd.read_csv(pima_path)

    # In Pima, 0s in Glucose, BloodPressure, SkinThickness, Insulin, BMI represent missing values
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "Pima Indians Diabetes",
        "n_samples": len(df),
        "n_features": X.shape[1],
        "features": list(X.columns),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
    }

    results_file = exp_dir / "pima_results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    model_file = exp_dir / "pima_rf_model.joblib"
    joblib.dump(pipeline, model_file)

    print(f"[Pima Baseline Complete] Results saved to {results_file}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    run_pima_baseline()
