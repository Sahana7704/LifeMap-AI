"""
FastAPI Service for LifeMap AI Diabetes Risk Module.
Provides POST /predict endpoint with strict input validation, calibrated ML probability,
top 5 SHAP attributions, rule-based clinical lab checker, and personalized lifestyle recommendations.
"""

import os
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd
import joblib

# Ensure src directory is in sys.path
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import model_classes
from model_classes import TabularPreprocessor, ProbabilityCalibrator

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(
    title="LifeMap AI - Diabetes Risk & Explainability API",
    description="Preventive non-invasive diabetes risk prediction with SHAP explainability and clinical lab escalation.",
    version="1.0.0"
)

frontend_origins_env = os.environ.get("FRONTEND_ORIGIN", "*")
allowed_origins = [o.strip() for o in frontend_origins_env.split(",")] if frontend_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "LifeMap AI Diabetes Microservice",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "lifemap-diabetes-service",
        "models_loaded": {
            "tier1": prep_full is not None and calibrated_model is not None,
            "tier2": prep_tier2 is not None and calibrated_model_tier2 is not None
        }
    }

# Load Artifacts
_models_env = os.environ.get("DIABETES_MODELS_DIR")
if _models_env:
    MODELS_DIR = Path(_models_env)
else:
    # 1. Local container / module layout: diabetes_module/models
    _candidate1 = Path(__file__).resolve().parent.parent / "models"
    # 2. Repo layout: LifeMap AI/diabetes_module/models
    _candidate2 = Path(__file__).resolve().parent.parent.parent / "diabetes_module" / "models"
    MODELS_DIR = _candidate1 if _candidate1.exists() else _candidate2

try:
    # Tier 1 models (No Labs)
    prep_full = joblib.load(MODELS_DIR / "preprocessor_full.joblib")
    raw_xgb = joblib.load(MODELS_DIR / "xgb_full.joblib")
    calibrated_model = joblib.load(MODELS_DIR / "final_calibrated_xgb_full.joblib")
    xgb_booster = raw_xgb.get_booster()

    # Tier 2 models (Lab-Augmented: +HbA1c)
    prep_tier2 = joblib.load(MODELS_DIR / "tier2_preprocessor.joblib")
    raw_xgb_tier2 = joblib.load(MODELS_DIR / "tier2_xgb.joblib")
    calibrated_model_tier2 = joblib.load(MODELS_DIR / "tier2_final_calibrated_xgb.joblib")
    xgb_booster_tier2 = raw_xgb_tier2.get_booster()
except Exception as e:
    print(f"Warning: Models could not be pre-loaded at startup: {e}")
    prep_full, raw_xgb, calibrated_model, xgb_booster = None, None, None, None
    prep_tier2, raw_xgb_tier2, calibrated_model_tier2, xgb_booster_tier2 = None, None, None, None

FEATURE_LABELS = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m^2)",
    "mean_sbp": "Systolic Blood Pressure (mmHg)",
    "mean_dbp": "Diastolic Blood Pressure (mmHg)",
    "hba1c": "Glycated Hemoglobin (HbA1c, %)",
    "sex": "Sex (Male/Female)",
    "high_bp": "History of High Blood Pressure",
    "high_chol": "History of High Cholesterol",
    "smoker": "Smoking History",
    "phys_activity": "Physical Activity",
    "family_history": "Family History of Diabetes"
}


# ==============================================================================
# Pydantic Schemas & Strict Validation
# ==============================================================================
class NonLabFeatures(BaseModel):
    age: float = Field(..., description="Age in years (18-120)")
    sex: int = Field(..., description="Biological sex (1=Male, 0=Female)")
    bmi: float = Field(..., description="Body Mass Index in kg/m^2 (10-80)")
    mean_sbp: float = Field(..., description="Systolic blood pressure in mmHg (60-260)")
    mean_dbp: float = Field(..., description="Diastolic blood pressure in mmHg (30-160)")
    high_bp: int = Field(..., description="Ever told had high blood pressure (1=Yes, 0=No)")
    high_chol: int = Field(..., description="Ever told had high cholesterol (1=Yes, 0=No)")
    smoker: int = Field(..., description="Smoked at least 100 cigarettes in lifetime (1=Yes, 0=No)")
    phys_activity: int = Field(..., description="Engages in physical activity (1=Yes, 0=No)")
    family_history: int = Field(..., description="Close relative had diabetes (1=Yes, 0=No)")

    @field_validator("age")
    def validate_age(cls, v):
        if v < 18 or v > 120:
            raise ValueError("age must be between 18 and 120 years")
        return v

    @field_validator("sex", "high_bp", "high_chol", "smoker", "phys_activity", "family_history")
    def validate_binary(cls, v, info):
        if v not in (0, 1):
            raise ValueError(f"{info.field_name} must be either 0 or 1")
        return v

    @field_validator("bmi")
    def validate_bmi(cls, v):
        if v < 10.0 or v > 80.0:
            raise ValueError("bmi must be between 10.0 and 80.0 kg/m^2")
        return v

    @field_validator("mean_sbp")
    def validate_sbp(cls, v):
        if v < 60.0 or v > 260.0:
            raise ValueError("mean_sbp must be between 60.0 and 260.0 mmHg")
        return v

    @field_validator("mean_dbp")
    def validate_dbp(cls, v):
        if v < 30.0 or v > 160.0:
            raise ValueError("mean_dbp must be between 30.0 and 160.0 mmHg")
        return v


class LabValues(BaseModel):
    hba1c: Optional[float] = Field(None, description="Glycated hemoglobin percentage (3.0 - 18.0)")
    fasting_glucose: Optional[float] = Field(None, description="Fasting plasma glucose in mg/dL (e.g. 100.0)")

    @field_validator("hba1c")
    def validate_hba1c(cls, v):
        if v is not None and (v < 3.0 or v > 18.0):
            raise ValueError("hba1c if provided must be between 3.0% and 18.0%")
        return v

    @field_validator("fasting_glucose")
    def validate_glucose(cls, v):
        if v is not None and (v < 30.0 or v > 600.0):
            raise ValueError("fasting_glucose if provided must be between 30.0 and 600.0 mg/dL")
        return v


class PredictRequest(BaseModel):
    features: NonLabFeatures
    labs: Optional[LabValues] = None
    hba1c: Optional[float] = Field(None, description="Optional direct HbA1c percentage (3.0 - 18.0)")

    @field_validator("hba1c")
    def validate_top_hba1c(cls, v):
        if v is not None and (v < 3.0 or v > 18.0):
            raise ValueError("hba1c if provided must be between 3.0% and 18.0%")
        return v


# ==============================================================================
# Rule-Based Clinical Engines
# ==============================================================================
def evaluate_labs(labs: Optional[LabValues]) -> Dict[str, Any]:
    """Rule-based clinical lab checker according to ADA diagnostic guidelines."""
    res = {
        "hba1c_value": "not provided",
        "hba1c_category": "not provided",
        "fasting_glucose_value": "not provided",
        "fasting_glucose_category": "not provided",
        "overall_lab_status": "not provided",
        "escalation_recommended": False,
        "escalation_reason": "No laboratory values provided."
    }

    if labs is None:
        return res

    hba1c_cat = None
    if labs.hba1c is not None:
        res["hba1c_value"] = f"{labs.hba1c:.1f}%"
        if labs.hba1c < 5.7:
            hba1c_cat = "Normal (< 5.7%)"
        elif 5.7 <= labs.hba1c <= 6.4:
            hba1c_cat = "Prediabetes (5.7 - 6.4%)"
        else:
            hba1c_cat = "suggestive of diabetes range; confirm with a clinician"
        res["hba1c_category"] = hba1c_cat

    glucose_cat = None
    if labs.fasting_glucose is not None:
        res["fasting_glucose_value"] = f"{labs.fasting_glucose:.1f} mg/dL"
        if labs.fasting_glucose < 100.0:
            glucose_cat = "Normal (< 100 mg/dL)"
        elif 100.0 <= labs.fasting_glucose <= 125.0:
            glucose_cat = "Prediabetes (100 - 125 mg/dL)"
        else:
            glucose_cat = "suggestive of diabetes range; confirm with a clinician"
        res["fasting_glucose_category"] = glucose_cat

    # Determine overall status
    if "suggestive of diabetes range" in str(hba1c_cat) or "suggestive of diabetes range" in str(glucose_cat) or "Diabetes Range" in str(hba1c_cat) or "Diabetes Range" in str(glucose_cat):
        res["overall_lab_status"] = "suggestive of diabetes range; confirm with a clinician"
        res["escalation_recommended"] = True
        res["escalation_reason"] = "Diagnostic threshold met (HbA1c >= 6.5% or Fasting Glucose >= 126 mg/dL); suggestive of diabetes range; confirm with a clinician."
    elif "Prediabetes" in str(hba1c_cat) or "Prediabetes" in str(glucose_cat):
        res["overall_lab_status"] = "prediabetes range"
        res["escalation_recommended"] = True
        res["escalation_reason"] = "Elevated glycemic marker detected (HbA1c 5.7-6.4% or Glucose 100-125 mg/dL); confirm with a clinician."
    elif hba1c_cat or glucose_cat:
        res["overall_lab_status"] = "normal"
        res["escalation_recommended"] = False
        res["escalation_reason"] = "All provided lab values are within standard physiological reference ranges."

    return res


def generate_recommendations(features: NonLabFeatures) -> List[Dict[str, str]]:
    """Rule-based lifestyle recommendation engine mapping key risk factors to general guidance."""
    recs = []

    if features.bmi >= 25.0:
        recs.append({
            "trigger": f"Elevated BMI ({features.bmi:.1f} kg/m^2)",
            "advice": "A structured weight management plan targeting a 5-7% reduction in body weight has been shown to reduce type 2 diabetes incidence by up to 58% in clinical trials."
        })

    if features.phys_activity == 0:
        recs.append({
            "trigger": "Sedentary / Low Physical Activity",
            "advice": "Incorporate at least 150 minutes of moderate-intensity aerobic physical activity per week (e.g., brisk walking, swimming, cycling), coupled with resistance exercise 2-3 times weekly."
        })

    if features.high_bp == 1 or features.mean_sbp >= 130.0 or features.mean_dbp >= 80.0:
        recs.append({
            "trigger": f"Elevated Blood Pressure ({features.mean_sbp:.0f}/{features.mean_dbp:.0f} mmHg)",
            "advice": "Adopt dietary patterns rich in whole grains, potassium, and magnesium (such as the DASH diet) and reduce dietary sodium intake below 2,300 mg/day."
        })

    if features.smoker == 1:
        recs.append({
            "trigger": "Active / Lifetime Smoking History",
            "advice": "Tobacco use exacerbates systemic insulin resistance and endothelial inflammation. Engaging with evidence-based smoking cessation resources offers rapid cardiovascular and metabolic benefits."
        })

    if features.high_chol == 1:
        recs.append({
            "trigger": "History of Dyslipidemia / High Cholesterol",
            "advice": "Prioritize soluble dietary fiber (oats, legumes, flaxseeds) and reduce saturated fats to improve lipid profiles and insulin sensitivity."
        })

    if not recs:
        recs.append({
            "trigger": "Optimal Baseline Non-Lab Profile",
            "advice": "Continue maintaining current balanced nutrition, regular exercise, and routine annual metabolic health screening."
        })

    return recs


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "LifeMap AI Diabetes Risk Prediction Service",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict_diabetes_risk(request: PredictRequest):
    # Determine if HbA1c is provided (either via request.hba1c or request.labs.hba1c)
    hba1c_val = None
    if request.hba1c is not None:
        hba1c_val = request.hba1c
    elif request.labs is not None and request.labs.hba1c is not None:
        hba1c_val = request.labs.hba1c

    # Validate range 3.0 - 18.0 when provided
    if hba1c_val is not None:
        try:
            hba1c_val = float(hba1c_val)
            if hba1c_val < 3.0 or hba1c_val > 18.0 or np.isnan(hba1c_val):
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="hba1c if provided must be a valid number between 3.0% and 18.0%."
            )

    feats = request.features
    import xgboost as xgb_mod

    if hba1c_val is not None:
        # TIER 2: Lab-Augmented Model (NHANES + HbA1c)
        if prep_tier2 is None or calibrated_model_tier2 is None or xgb_booster_tier2 is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tier 2 models are not initialized or failed to load."
            )
        model_used = "tier2_with_hba1c"
        row_dict = {
            "age": feats.age,
            "bmi": feats.bmi,
            "mean_sbp": feats.mean_sbp,
            "mean_dbp": feats.mean_dbp,
            "hba1c": hba1c_val,
            "sex": feats.sex,
            "high_bp": feats.high_bp,
            "high_chol": feats.high_chol,
            "smoker": feats.smoker,
            "phys_activity": feats.phys_activity,
            "family_history": feats.family_history,
        }
        df_single = pd.DataFrame([row_dict])
        X_trans = prep_tier2.transform(df_single)
        raw_prob = float(calibrated_model_tier2.predict_proba(X_trans)[0, 1])

        feature_order = prep_tier2.continuous_cols + prep_tier2.binary_cols
        dmatrix = xgb_mod.DMatrix(X_trans, feature_names=feature_order)
        contribs = xgb_booster_tier2.predict(dmatrix, pred_contribs=True)[0]
        shap_vals = contribs[:-1]
    else:
        # TIER 1: Non-Lab Model (Demographics + Vitals + Behavioral)
        if prep_full is None or calibrated_model is None or xgb_booster is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tier 1 models are not initialized or failed to load."
            )
        model_used = "tier1_no_labs"
        row_dict = {
            "age": feats.age,
            "bmi": feats.bmi,
            "mean_sbp": feats.mean_sbp,
            "mean_dbp": feats.mean_dbp,
            "sex": feats.sex,
            "high_bp": feats.high_bp,
            "high_chol": feats.high_chol,
            "smoker": feats.smoker,
            "phys_activity": feats.phys_activity,
            "family_history": feats.family_history,
        }
        df_single = pd.DataFrame([row_dict])
        X_trans = prep_full.transform(df_single)
        raw_prob = float(calibrated_model.predict_proba(X_trans)[0, 1])

        feature_order = prep_full.continuous_cols + prep_full.binary_cols
        dmatrix = xgb_mod.DMatrix(X_trans, feature_names=feature_order)
        contribs = xgb_booster.predict(dmatrix, pred_contribs=True)[0]
        shap_vals = contribs[:-1]

    # 1. Model predicted calibrated probability (unmodified output of model used)
    model_prob = float(round(raw_prob, 4))

    # 2. Model Risk Stratification (Low < 0.15, Moderate 0.15 - 0.40, High >= 0.40)
    if model_prob < 0.15:
        model_risk_level = "Low"
    elif model_prob < 0.40:
        model_risk_level = "Moderate"
    else:
        model_risk_level = "High"

    # Top 5 SHAP Feature Contributions via native XGBoost pred_contribs
    shap_items = []
    for f_name, s_val in zip(feature_order, shap_vals):
        readable = FEATURE_LABELS.get(f_name, f_name)
        shap_items.append({
            "feature": readable,
            "shap_value": float(round(s_val, 4)),
            "impact": "Increases Risk" if s_val > 0 else "Protective (Decreases Risk)",
            "magnitude": float(abs(round(s_val, 4)))
        })

    shap_items.sort(key=lambda x: x["magnitude"], reverse=True)
    top_5_shap = shap_items[:5]
    if model_used == "tier2_with_hba1c":
        hba1c_item = next((item for item in shap_items if "hba1c" in item["feature"].lower() or "glycated hemoglobin" in item["feature"].lower()), None)
        if hba1c_item and not any("hba1c" in x["feature"].lower() or "glycated hemoglobin" in x["feature"].lower() for x in top_5_shap):
            top_5_shap = top_5_shap[:4] + [hba1c_item]

    # 3. Clinical Lab Checker & Transparent Level Escalation ONLY
    effective_labs = request.labs
    if effective_labs is None and hba1c_val is not None:
        effective_labs = LabValues(hba1c=hba1c_val)
    elif effective_labs is not None and effective_labs.hba1c is None and hba1c_val is not None:
        effective_labs = LabValues(hba1c=hba1c_val)

    lab_report = evaluate_labs(effective_labs)
    lab_status = lab_report.get("overall_lab_status", "not provided")

    # Lab rule minimum risk levels:
    # prediabetes range -> at least Moderate; diabetes range -> High
    RISK_ORDER = {"Low": 1, "Moderate": 2, "High": 3}
    if lab_status == "suggestive of diabetes range; confirm with a clinician":
        lab_min_level = "High"
    elif lab_status == "prediabetes range":
        lab_min_level = "Moderate"
    else:
        lab_min_level = "Low"

    # final_risk_level = higher of model's risk level and lab-rule minimum level
    if RISK_ORDER.get(lab_min_level, 1) > RISK_ORDER.get(model_risk_level, 1):
        final_risk_level = lab_min_level
        escalation_notes = (
            f"Risk level raised from {model_risk_level} to {final_risk_level} by clinical lab rule "
            f"({lab_status}); model probability ({model_prob:.4f}) was not changed."
        )
    else:
        final_risk_level = model_risk_level
        escalation_notes = "ML-predicted risk level maintained; probability was not changed by lab values."

    # Lifestyle Recommendations
    recommendations = generate_recommendations(feats)

    return {
        "status": "success",
        "model_used": model_used,
        "predicted_probability": model_prob,
        "model_probability": model_prob,
        "percentage_risk": f"{model_prob * 100:.1f}%",
        "base_risk_level": model_risk_level,
        "model_risk_level": model_risk_level,
        "lab_status": lab_status,
        "final_risk_level": final_risk_level,
        "escalation_notes": escalation_notes,
        "top_5_shap_contributions": top_5_shap,
        "all_shap_contributions": shap_items,
        "laboratory_assessment": lab_report,
        "lifestyle_recommendations": recommendations,
        "disclaimer": (
            "DISCLAIMER: LifeMap AI provides probabilistic risk assessment and lifestyle recommendations "
            "for educational and preventive wellness purposes only. It is not intended as medical diagnosis, "
            "treatment, or clinical advice. Consult a qualified healthcare professional for formal diagnosis."
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8005, reload=False)
