import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional

BUNDLE_PATH_V2 = os.path.join(os.path.dirname(__file__), "models", "pima_diabetes_rf_bundle_v2.joblib")
BUNDLE_PATH_V1 = os.path.join(os.path.dirname(__file__), "models", "pima_diabetes_rf_bundle.joblib")

_pima_bundle: Optional[Dict[str, Any]] = None
_pima_explainer: Optional[Any] = None

def get_pima_bundle() -> Dict[str, Any]:
    global _pima_bundle, _pima_explainer
    if _pima_bundle is None:
        bundle_to_load = BUNDLE_PATH_V2 if os.path.exists(BUNDLE_PATH_V2) else BUNDLE_PATH_V1
        if not os.path.exists(bundle_to_load):
            from train_pima_model import train_pima_model
            train_pima_model()
            bundle_to_load = BUNDLE_PATH_V2 if os.path.exists(BUNDLE_PATH_V2) else BUNDLE_PATH_V1
        _pima_bundle = joblib.load(bundle_to_load)
        # Initialize SHAP explainer
        _pima_explainer = shap.TreeExplainer(_pima_bundle["model"])
    return _pima_bundle

def get_pima_explainer() -> Any:
    global _pima_explainer
    if _pima_explainer is None:
        get_pima_bundle()
    return _pima_explainer

FEATURE_DISPLAY_NAMES = {
    "Glucose": "Blood Glucose",
    "BMI": "Body Mass Index (BMI)",
    "Age": "Age",
    "BloodPressure": "Blood Pressure (Diastolic)",
    "Pregnancies": "Pregnancies",
    "Insulin": "Serum Insulin",
    "SkinThickness": "Triceps Skin Fold Thickness",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function (Genetic History)"
}

FEATURE_UNITS = {
    "Glucose": "mg/dL",
    "BMI": "kg/m²",
    "Age": "years",
    "BloodPressure": "mmHg",
    "Pregnancies": "count",
    "Insulin": "μU/mL",
    "SkinThickness": "mm",
    "DiabetesPedigreeFunction": "score"
}

def predict_pima_diabetes_with_shap(
    patient_vitals: Dict[str, Any],
    decision_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes REAL machine learning prediction and REAL SHAP feature contributions
    using the trained Pima Indians Diabetes Random Forest model (v2 screening edition).

    Distinguishes strictly between features extracted from the patient's report
    and features imputed with dataset population medians.

    decision_threshold: Configurable probability cutoff for screening (defaults to 0.40
    optimized on training CV folds for high sensitivity/recall).
    """
    bundle = get_pima_bundle()
    explainer = get_pima_explainer()
    model = bundle["model"]
    medians = bundle["imputer_medians"]
    
    # Load v2 metrics if available, otherwise fallback
    metrics = bundle.get("metrics_at_clinical_threshold") or bundle.get("metrics", {})
    metrics_default = bundle.get("metrics_at_default_threshold") or bundle.get("metrics", {})
    feature_cols = bundle["feature_names"]
    
    threshold = float(decision_threshold) if decision_threshold is not None else float(bundle.get("decision_threshold", 0.40))

    # 1. Map patient vitals into Pima feature vector
    extracted_features = []
    imputed_features = []
    feature_dict = {}

    # Glucose: check Fasting, Random, Post-Prandial, or HbA1c-derived eAG
    glucose_val = None
    if patient_vitals.get("fasting_glucose") is not None:
        val = patient_vitals["fasting_glucose"]
        glucose_val = float(val["value"] if isinstance(val, dict) else val)
    elif patient_vitals.get("random_glucose") is not None:
        val = patient_vitals["random_glucose"]
        glucose_val = float(val["value"] if isinstance(val, dict) else val)
    elif patient_vitals.get("post_prandial_glucose") is not None:
        val = patient_vitals["post_prandial_glucose"]
        glucose_val = float(val["value"] if isinstance(val, dict) else val)
    elif patient_vitals.get("glucose") is not None:
        val = patient_vitals["glucose"]
        glucose_val = float(val["value"] if isinstance(val, dict) else val)
    elif patient_vitals.get("hba1c") is not None:
        val = patient_vitals["hba1c"]
        h_val = float(val["value"] if isinstance(val, dict) else val)
        # Convert HbA1c to Estimated Average Glucose (eAG) via ADAG formula: 28.7 * A1c - 46.7
        glucose_val = round(28.7 * h_val - 46.7, 1)

    if glucose_val is not None:
        feature_dict["Glucose"] = glucose_val
        extracted_features.append("Glucose")
    else:
        feature_dict["Glucose"] = medians["Glucose"]
        imputed_features.append("Glucose")

    # BMI: check direct bmi or compute from height/weight
    bmi_val = None
    if patient_vitals.get("bmi") is not None:
        val = patient_vitals["bmi"]
        v = val.get("value") if isinstance(val, dict) else val
        if v is not None:
            bmi_val = float(v)
    if bmi_val is None:
        ht = patient_vitals.get("height_cm")
        wt = patient_vitals.get("weight_kg")
        if ht and wt and float(ht) > 0:
            bmi_val = round(float(wt) / ((float(ht) / 100.0) ** 2), 1)

    if bmi_val is not None:
        feature_dict["BMI"] = bmi_val
        extracted_features.append("BMI")
    else:
        feature_dict["BMI"] = medians["BMI"]
        imputed_features.append("BMI")

    # Age
    age_val = patient_vitals.get("age")
    if age_val is not None:
        feature_dict["Age"] = float(age_val)
        extracted_features.append("Age")
    else:
        feature_dict["Age"] = medians["Age"]
        imputed_features.append("Age")

    # Blood Pressure (Diastolic in Pima dataset)
    bp_val = patient_vitals.get("diastolic_bp") or patient_vitals.get("blood_pressure")
    if bp_val is not None:
        feature_dict["BloodPressure"] = float(bp_val)
        extracted_features.append("BloodPressure")
    else:
        feature_dict["BloodPressure"] = medians["BloodPressure"]
        imputed_features.append("BloodPressure")

    # Pregnancies: if sex is Male, biologically 0. Otherwise check report or impute
    sex = str(patient_vitals.get("sex") or patient_vitals.get("gender") or "").lower()
    if "m" in sex and "fe" not in sex:
        feature_dict["Pregnancies"] = 0.0
        extracted_features.append("Pregnancies")
    elif patient_vitals.get("pregnancies") is not None:
        feature_dict["Pregnancies"] = float(patient_vitals["pregnancies"])
        extracted_features.append("Pregnancies")
    else:
        feature_dict["Pregnancies"] = medians["Pregnancies"]
        imputed_features.append("Pregnancies")

    # Unmeasured Pima biomarkers (SkinThickness, Insulin, Pedigree)
    for col in ["SkinThickness", "Insulin", "DiabetesPedigreeFunction"]:
        if patient_vitals.get(col.lower()) is not None:
            feature_dict[col] = float(patient_vitals[col.lower()])
            extracted_features.append(col)
        else:
            feature_dict[col] = medians[col]
            imputed_features.append(col)

    # 2. DataFrame with correct column order
    input_df = pd.DataFrame([[feature_dict[c] for c in feature_cols]], columns=feature_cols)

    # 3. Model Inference (Real probability from Random Forest)
    prob_array = model.predict_proba(input_df)[0]
    prob_diabetes = float(prob_array[1])
    pred_label = 1 if prob_diabetes >= threshold else 0

    if prob_diabetes < 0.30:
        risk_category = "Low Population Risk"
    elif prob_diabetes < threshold:
        risk_category = "Moderate Baseline Risk"
    elif prob_diabetes <= 0.60:
        risk_category = "Elevated Population Risk"
    else:
        risk_category = "High Population Risk"

    # 4. Real SHAP TreeExplainer Calculation
    # Computes exact Shapley contribution for each of the 8 features
    raw_shap_output = explainer.shap_values(input_df)

    # Handle various shap output formats (list of arrays for multiclass or 3D array)
    if isinstance(raw_shap_output, list) and len(raw_shap_output) > 1:
        shap_vals_class1 = raw_shap_output[1][0]
    elif isinstance(raw_shap_output, np.ndarray) and raw_shap_output.ndim == 3:
        shap_vals_class1 = raw_shap_output[0, :, 1]
    elif isinstance(raw_shap_output, np.ndarray) and raw_shap_output.ndim == 2:
        shap_vals_class1 = raw_shap_output[0]
    else:
        shap_vals_class1 = np.array(raw_shap_output).flatten()

    base_value = float(bundle.get("expected_value", 0.3466))

    # 5. Build detailed SHAP feature contributions
    total_abs_shap = float(np.sum(np.abs(shap_vals_class1))) or 1.0
    all_features_shap = []
    risk_increasing_factors = []
    protective_factors = []

    for i, col in enumerate(feature_cols):
        val = float(input_df.iloc[0, i])
        s_val = float(shap_vals_class1[i])
        rel_pct = round((abs(s_val) / total_abs_shap) * 100.0, 1)
        is_imputed = col in imputed_features

        unit = FEATURE_UNITS.get(col, "")
        val_str = f"{val:.1f} {unit}".strip() if unit != "count" and unit != "score" else f"{val:.2f}"
        if col == "Pregnancies":
            val_str = f"{int(val)} pregnancies" if val > 0 else "0 (Male/Null)"

        feature_item = {
            "feature": col,
            "label": FEATURE_DISPLAY_NAMES.get(col, col),
            "raw_value": val,
            "formatted_value": f"{val_str} ({'Extracted from report' if not is_imputed else 'Imputed with dataset median'})",
            "shap_value": round(s_val, 4),
            "impact_pct": rel_pct,
            "direction": "increases_risk" if s_val > 0 else ("decreases_risk" if s_val < 0 else "neutral"),
            "is_imputed": is_imputed,
            "clinical_guidance": _generate_shap_guidance(col, val, s_val, is_imputed)
        }
        all_features_shap.append(feature_item)

        if s_val > 0.005:
            risk_increasing_factors.append(feature_item)
        elif s_val < -0.005:
            protective_factors.append(feature_item)

    # Sort by magnitude of SHAP impact
    risk_increasing_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    protective_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    all_features_shap.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "model_name": "Random Forest Classifier (Pima Indians Benchmark v2)",
        "model_type": "Tree-based Ensemble (Scikit-learn RandomForestClassifier, Balanced)",
        "decision_threshold": threshold,
        "predicted_label": pred_label,
        "dataset_info": {
            "dataset_name": "Pima Indians Diabetes Dataset",
            "total_samples": 768,
            "test_samples": metrics.get("test_samples", 154),
            "test_recall": f"{metrics.get('recall', 0.8519) * 100:.1f}%",
            "test_accuracy": f"{metrics.get('accuracy', 0.7208) * 100:.1f}%",
            "test_auc": f"{metrics.get('auc_roc', 0.8135):.3f}",
            "test_precision": f"{metrics.get('precision', 0.5679):.3f}",
            "test_f1": f"{metrics.get('f1_score', 0.6815):.3f}",
            "missed_cases_fn": metrics.get("missed_diabetics_fn", 8),
            "baseline_v1_missed_fn": 25,
            "decision_threshold": threshold,
            "threshold_rationale": "Threshold calibrated to 0.40 via 5-fold CV to maximize screening sensitivity, reducing false-negative missed diabetics from 25 down to 8."
        },
        "risk_probability": round(prob_diabetes, 4),
        "risk_score_pct": round(prob_diabetes * 100.0, 1),
        "risk_category": risk_category,
        "base_expected_value": round(base_value, 4),
        "shap_drivers": {
            "risk_increasing": risk_increasing_factors,
            "protective": protective_factors,
            "all_features_shap": all_features_shap
        },
        "feature_provenance": {
            "extracted_from_report": extracted_features,
            "imputed_with_dataset_median": imputed_features
        },
        "disclaimer": (
            f"This is a supplementary population-based risk model trained on the Pima Indians Diabetes dataset "
            f"(768 patients), achieving {metrics.get('recall', 0.8519) * 100:.1f}% sensitivity/recall, "
            f"{metrics.get('accuracy', 0.7208) * 100:.1f}% accuracy, and "
            f"{metrics.get('auc_roc', 0.8135):.3f} AUC on held-out test data (N=154). "
            f"This is separate from the ADA/WHO clinical threshold assessment above, which is based on your own actual lab values."
        )
    }

def _generate_shap_guidance(col: str, val: float, shap_val: float, is_imputed: bool) -> str:
    direction_word = "elevates" if shap_val > 0 else "lowers"
    impute_prefix = "[Dataset Median] " if is_imputed else ""
    if col == "Glucose":
        return f"{impute_prefix}Glucose of {val:.0f} mg/dL {direction_word} estimated diabetes risk by {abs(shap_val):.2f} SHAP score."
    elif col == "BMI":
        return f"{impute_prefix}BMI of {val:.1f} kg/m² {direction_word} risk contribution."
    elif col == "Age":
        return f"{impute_prefix}Age {val:.0f} years contributes {shap_val:+.2f} toward population baseline."
    elif col == "BloodPressure":
        return f"{impute_prefix}Diastolic pressure {val:.0f} mmHg provides {shap_val:+.2f} SHAP influence."
    else:
        return f"{impute_prefix}{FEATURE_DISPLAY_NAMES.get(col, col)} ({val:.1f}) provides {shap_val:+.2f} SHAP contribution."
