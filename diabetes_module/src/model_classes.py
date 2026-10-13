"""
Core Model and Preprocessing Classes for LifeMap AI Diabetes Serving.
Extracted into a standalone lightweight module to allow production inference
without requiring heavyweight training dependencies (such as PyTorch, imbalanced-learn, matplotlib, or seaborn).
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """
    Calibration wrapper (Isotonic or Platt/Logistic) on validation probabilities.
    """
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


class TabularPreprocessor:
    """
    Standard preprocessing pipeline for continuous and binary clinical survey features.
    """
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


# Expose classes to __main__ and train_and_evaluate namespaces for seamless joblib unpickling
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "TabularPreprocessor", TabularPreprocessor)
    setattr(sys.modules["__main__"], "ProbabilityCalibrator", ProbabilityCalibrator)

# Register alias in sys.modules so pickles serialized under 'train_and_evaluate' load seamlessly
if "train_and_evaluate" not in sys.modules:
    sys.modules["train_and_evaluate"] = sys.modules[__name__]
else:
    setattr(sys.modules["train_and_evaluate"], "TabularPreprocessor", TabularPreprocessor)
    setattr(sys.modules["train_and_evaluate"], "ProbabilityCalibrator", ProbabilityCalibrator)
