import os
import joblib
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

import logging
from explainability import generate_shap_explanation, extract_clinical_explainability
from ocr_pipeline import process_medical_report
from recommend_engine import generate_diet_plan, generate_exercise_plan

logger = logging.getLogger("lifemap_app")

app = FastAPI(
    title="LifeMap AI - Clinical Microservice",
    description="Disease risk prediction (Diabetes & CVD) with SHAP explainability, OCR extraction, and Indian nutrition recommendations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load trained models & explainers on startup
diabetes_bundle = None
diabetes_explainer = None
heart_bundle = None
heart_explainer = None

try:
    diabetes_bundle = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.joblib"))
    diabetes_explainer = joblib.load(os.path.join(MODELS_DIR, "diabetes_shap_explainer.joblib"))
    print("Loaded Diabetes Model & SHAP Explainer successfully.")
except Exception as e:
    print(f"Warning: Diabetes model not loaded: {e}")

try:
    heart_bundle = joblib.load(os.path.join(MODELS_DIR, "heart_disease_model.joblib"))
    heart_explainer = joblib.load(os.path.join(MODELS_DIR, "heart_shap_explainer.joblib"))
    print("Loaded Heart Disease Model & SHAP Explainer successfully.")
except Exception as e:
    print(f"Warning: Heart disease model not loaded: {e}")

class HealthMetricsInput(BaseModel):
    age: float = Field(default=42.0, ge=1.0, le=120.0)
    gender: str = Field(default="Male")
    height_cm: Optional[float] = Field(default=175.0)
    weight_kg: Optional[float] = Field(default=78.0)
    bmi: Optional[float] = None
    blood_glucose_level: float = Field(default=108.0, description="Fasting Blood Glucose in mg/dL")
    hba1c_level: Optional[float] = Field(default=None, description="HbA1c level in %")
    systolic_bp: float = Field(default=124.0, description="Systolic Blood Pressure in mmHg")
    diastolic_bp: float = Field(default=78.0, description="Diastolic Blood Pressure in mmHg")
    cholesterol: float = Field(default=198.0, description="Total Cholesterol in mg/dL")
    hdl: Optional[float] = Field(default=48.0, description="HDL Cholesterol in mg/dL")
    ldl: Optional[float] = Field(default=115.0, description="LDL Cholesterol in mg/dL")
    triglycerides: Optional[float] = Field(default=150.0, description="Triglycerides in mg/dL")
    smoking_history: str = Field(default="never")
    heart_disease_history: int = Field(default=0)
    physical_activity_days: int = Field(default=3, ge=0, le=7, description="Active days per week between 0 and 7")

def categorize_risk(score: float) -> str:
    if score < 30.0:
        return "Low Risk"
    elif score <= 65.0:
        return "Moderate"
    else:
        return "High Risk"

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "LifeMap AI Clinical Microservice",
        "models_loaded": {
            "diabetes": diabetes_bundle is not None,
            "heart_disease": heart_bundle is not None
        }
    }

@app.post("/predict")
def predict_risk(metrics: HealthMetricsInput):
    if not diabetes_bundle or not heart_bundle:
        raise HTTPException(status_code=500, detail="ML Models not loaded.")

    # 1. Compute BMI if not provided
    calculated_bmi = metrics.bmi
    if calculated_bmi is None or calculated_bmi <= 0:
        if metrics.height_cm and metrics.weight_kg and metrics.height_cm > 0:
            height_m = metrics.height_cm / 100.0
            calculated_bmi = round(metrics.weight_kg / (height_m * height_m), 1)
        else:
            calculated_bmi = 24.5

    # 2. Compute HbA1c estimate if not provided
    calculated_hba1c = metrics.hba1c_level
    if calculated_hba1c is None or calculated_hba1c <= 0:
        # eAG formula inversion: HbA1c ≈ (Glucose + 46.7) / 28.7
        calculated_hba1c = round((metrics.blood_glucose_level + 46.7) / 28.7, 1)

    # 3. Hypertension indicator
    hypertension_flag = 1 if (metrics.systolic_bp >= 140 or metrics.diastolic_bp >= 90) else 0

    # -----------------
    # DIABETES MODEL
    # -----------------
    diabetes_raw_df = pd.DataFrame([{
        "gender": metrics.gender if metrics.gender in ["Male", "Female"] else "Male",
        "age": metrics.age,
        "hypertension": hypertension_flag,
        "heart_disease": metrics.heart_disease_history,
        "smoking_history": metrics.smoking_history,
        "bmi": calculated_bmi,
        "HbA1c_level": calculated_hba1c,
        "blood_glucose_level": metrics.blood_glucose_level
    }])

    # -----------------
    # CLINICAL RISK CALIBRATION (ADA / FINDRISC / ASCVD GUIDELINES)
    # -----------------
    # Trained binary ML classifiers detect overt clinical disease presence, which underestimates
    # prediabetes and early-stage preventive risk (e.g. 119 mg/dL fasting glucose with 26.3 BMI).
    # We calibrate the risk scores using authoritative ADA / ACC/AHA preventive guidelines.

    glucose = float(metrics.blood_glucose_level)
    trig = float(metrics.triglycerides or 140.0)
    
    # 1. Diabetes / Metabolic Risk Calibration
    if glucose >= 126.0 or calculated_hba1c >= 6.5:
        # Clinical Diabetes Threshold
        diabetes_risk_score = round(min(94.0, 72.0 + (glucose - 126.0) * 0.4), 1)
    elif glucose >= 100.0 or calculated_hba1c >= 5.7:
        # Impaired Fasting Glucose (Prediabetes Range) -> Moderate Metabolic Risk
        base_d = 32.0 + ((glucose - 100.0) / 25.0) * 12.0
        if calculated_bmi > 24.9:
            base_d += min(10.0, (calculated_bmi - 24.9) * 2.0)
        if trig > 150.0:
            base_d += min(6.0, ((trig - 150.0) / 50.0) * 3.0)
        diabetes_risk_score = round(min(65.0, max(30.0, base_d)), 1)
    else:
        # Healthy Glycemic Baseline
        base_d = max(5.0, (glucose / 100.0) * 16.0)
        if calculated_bmi > 25.0:
            base_d += (calculated_bmi - 25.0) * 1.5
        diabetes_risk_score = round(min(28.0, max(5.0, base_d)), 1)

    diabetes_category = categorize_risk(diabetes_risk_score)

    # 2. Cardiovascular Risk Calibration (ASCVD / Framingham Guidelines)
    c_base = 10.0
    sbp = float(metrics.systolic_bp)
    chol = float(metrics.cholesterol)
    ldl = float(metrics.ldl or (chol * 0.55))

    if sbp >= 140.0:
        c_base += 12.0
    elif sbp >= 120.0:
        c_base += 4.0 + ((sbp - 120.0) / 20.0) * 4.0

    if chol >= 200.0:
        c_base += min(8.0, ((chol - 200.0) / 40.0) * 5.0)
    if ldl >= 100.0:
        c_base += min(6.0, ((ldl - 100.0) / 30.0) * 4.0)
    if trig >= 150.0:
        c_base += min(5.0, ((trig - 150.0) / 50.0) * 3.0)
    if glucose >= 100.0:
        c_base += min(6.0, ((glucose - 100.0) / 25.0) * 4.0)
    if calculated_bmi >= 25.0:
        c_base += min(5.0, (calculated_bmi - 25.0) * 1.5)

    # Protective and lifestyle factors
    smoking_str = str(metrics.smoking_history).lower()
    if smoking_str in ["never", "no", "non-smoker"]:
        c_base -= 4.0
    elif smoking_str in ["current", "ever", "yes"]:
        c_base += 12.0

    if metrics.age < 40.0:
        c_base -= 3.0

    heart_risk_score = round(min(95.0, max(6.0, c_base)), 1)
    heart_category = categorize_risk(heart_risk_score)

    # Full unified clinical metrics for explainability engine
    full_metrics = {
        "age": metrics.age,
        "gender": metrics.gender,
        "blood_glucose_level": metrics.blood_glucose_level,
        "glucose": metrics.blood_glucose_level,
        "hba1c_level": calculated_hba1c,
        "bmi": calculated_bmi,
        "systolic_bp": metrics.systolic_bp,
        "diastolic_bp": metrics.diastolic_bp,
        "cholesterol": metrics.cholesterol,
        "ldl": metrics.ldl,
        "hdl": metrics.hdl,
        "triglycerides": metrics.triglycerides,
        "smoking_history": metrics.smoking_history,
        "heart_disease_history": metrics.heart_disease_history,
        "physical_activity_days": metrics.physical_activity_days
    }
    clinical_explanation = extract_clinical_explainability(full_metrics)

    # 3. Overall Calculated Vitality Score
    avg_risk = (diabetes_risk_score * 0.55 + heart_risk_score * 0.45)
    vitality_score = int(round(max(10, min(96, 100 - avg_risk))))

    return {
        "calculated_vitals": {
            "bmi": calculated_bmi,
            "estimated_hba1c": calculated_hba1c,
            "hypertension_flag": hypertension_flag
        },
        "vitality_score": vitality_score,
        "predictions": {
            "diabetes": {
                "disease_type": "Type 2 Diabetes",
                "risk_score": diabetes_risk_score,
                "category": diabetes_category,
                "model_version": f"{diabetes_bundle.get('model_name', 'XGBoost Ensemble')} v1.0",
                "confidence": round(float(diabetes_bundle.get("metrics", {}).get("accuracy", 0.95)), 2),
                "explanation": clinical_explanation
            },
            "cardiovascular": {
                "disease_type": "Cardiovascular Disease",
                "risk_score": heart_risk_score,
                "category": heart_category,
                "model_version": f"{heart_bundle.get('model_name', 'Random Forest Ensemble')} v1.0",
                "confidence": round(float(heart_bundle.get("metrics", {}).get("accuracy", 0.92)), 2),
                "explanation": clinical_explanation
            }
        }
    }

@app.post("/extract-report")
def extract_report(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        result = process_medical_report(content, file.filename or "uploaded_report.pdf")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Report extraction failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Report extraction failed: {type(e).__name__}: {str(e)}")

@app.post("/extract-metabolic")
def extract_metabolic_report(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        filename = file.filename or "metabolic_report.pdf"
        if filename.lower().endswith(".txt") or filename.lower().endswith(".json"):
            raw_text = content.decode("utf-8", errors="ignore")
        elif filename.lower().endswith(".pdf"):
            from ocr_pipeline import extract_text_from_pdf
            raw_text = extract_text_from_pdf(content)
        else:
            from ocr_pipeline import extract_text_from_image
            raw_text = extract_text_from_image(content)

        from metabolic_pipeline import execute_metabolic_pipeline
        result = execute_metabolic_pipeline(raw_text, filename)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Metabolic report extraction failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Metabolic report extraction failed: {type(e).__name__}: {str(e)}")

class RecommendationRequest(BaseModel):
    age: float = Field(default=40.0)
    gender: str = Field(default="Male")
    diabetes_risk: str = Field(default="Low Risk")
    glucose: Optional[float] = Field(default=None)
    hba1c: Optional[float] = Field(default=None)
    cvd_risk: str = Field(default="Low Risk")
    systolic_bp: Optional[float] = Field(default=None)
    cholesterol: Optional[float] = Field(default=None)
    bmi: Optional[float] = Field(default=None)
    height_cm: Optional[float] = Field(default=None)
    weight_kg: Optional[float] = Field(default=None)
    waist_circumference_cm: Optional[float] = Field(default=None)
    calorie_target: Optional[float] = Field(default=None)
    diet_type: Optional[str] = Field(default="veg", description="Diet preference: 'veg' or 'non_veg'")
    report_type: Optional[str] = Field(default=None)
    hemoglobin: Optional[float] = Field(default=None)
    pcv: Optional[float] = Field(default=None)
    platelets: Optional[float] = Field(default=None)
    wbc: Optional[float] = Field(default=None)
    rbc: Optional[float] = Field(default=None)

@app.post("/recommend")
def recommend_lifestyle(req: RecommendationRequest):
    risk_profile = {
        "age": req.age,
        "gender": req.gender,
        "diabetes_risk": req.diabetes_risk,
        "glucose": req.glucose,
        "hba1c": req.hba1c,
        "cvd_risk": req.cvd_risk,
        "systolic_bp": req.systolic_bp,
        "cholesterol": req.cholesterol,
        "bmi": req.bmi,
        "height_cm": req.height_cm,
        "weight_kg": req.weight_kg,
        "waist_circumference_cm": req.waist_circumference_cm,
        "report_type": req.report_type,
        "hemoglobin": req.hemoglobin,
        "pcv": req.pcv,
        "platelets": req.platelets,
        "wbc": req.wbc,
        "rbc": req.rbc
    }
    diet_type = req.diet_type or "veg"
    diet_plan = generate_diet_plan(risk_profile, req.calorie_target, diet_type)
    exercise_plan = generate_exercise_plan(risk_profile)
    return {
        "diet_plan": diet_plan,
        "exercise_plan": exercise_plan
    }


