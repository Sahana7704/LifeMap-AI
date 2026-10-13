# LifeMap AI Diabetes Module - Integration Guide

This guide outlines how to integrate the standalone explainable Diabetes Risk Module into the existing LifeMap AI architecture (**Node.js backend** and **Next.js frontend**) without altering the existing `ml_service/` codebase.

---

## 1. Architectural Overview

```
[ Frontend: Next.js (Port 3000) ]
              |
              v (REST API)
[ Backend: Node.js / Express (Port 5000) ]
              |
      +-------+-------+
      |               |
      v               v
[ ml_service (8000) ]  [ diabetes_module (8005) ]
(Cardio, Stroke, etc.) (Calibrated XGBoost + TreeSHAP + Lab Escalation)
```

- **Port Isolation**: Runs as an independent FastAPI microservice on **port 8005**.
- **Zero Disruption**: The existing `ml_service/app.py` on port 8000 is left untouched.
- **Explainability Native**: Computes exact TreeSHAP attributions per request in <15ms.

---

## 2. Running the Diabetes Microservice

Activate the dedicated virtual environment and launch Uvicorn:

```powershell
# From repo root (Windows PowerShell)
diabetes_module\.venv-diabetes\Scripts\python.exe -m uvicorn app:app --app-dir diabetes_module/src --host 0.0.0.0 --port 8005 --reload
```

Health check endpoint:
```http
GET http://localhost:8005/
```
Response:
```json
{
  "status": "online",
  "service": "LifeMap AI Diabetes Risk Prediction Service",
  "version": "1.0.0"
}
```

---

## 3. API Contract: `POST /predict`

### Request Payload (`PredictRequest`)
```json
{
  "features": {
    "age": 52.0,
    "sex": 1,
    "bmi": 29.4,
    "mean_sbp": 134.0,
    "mean_dbp": 84.0,
    "high_bp": 1,
    "high_chol": 1,
    "smoker": 0,
    "phys_activity": 0,
    "family_history": 1
  },
  "labs": {
    "hba1c": 5.9,
    "fasting_glucose": 110.0
  }
}
```

*Note: The `labs` object is optional. If omitted or partially omitted (e.g. `"fasting_glucose": null`), the service still predicts risk from non-lab features and marks missing labs as `"not provided"`.*

### Response Payload
```json
{
  "status": "success",
  "predicted_probability": 0.4017,
  "model_probability": 0.4017,
  "percentage_risk": "40.2%",
  "base_risk_level": "High",
  "model_risk_level": "High",
  "lab_status": "not provided",
  "final_risk_level": "High",
  "escalation_notes": "ML-predicted risk level maintained; model probability (0.4017) was not changed.",
  "top_5_shap_contributions": [
    {
      "feature": "Family History of Diabetes",
      "shap_value": 0.133,
      "impact": "Increases Risk",
      "magnitude": 0.133
    },
    {
      "feature": "History of High Blood Pressure",
      "shap_value": 0.1027,
      "impact": "Increases Risk",
      "magnitude": 0.1027
    },
    {
      "feature": "Diastolic Blood Pressure (mmHg)",
      "shap_value": -0.066,
      "impact": "Protective (Decreases Risk)",
      "magnitude": 0.066
    },
    {
      "feature": "History of High Cholesterol",
      "shap_value": 0.0594,
      "impact": "Increases Risk",
      "magnitude": 0.0594
    },
    {
      "feature": "Age (years)",
      "shap_value": 0.0232,
      "impact": "Increases Risk",
      "magnitude": 0.0232
    }
  ],
  "laboratory_assessment": {
    "hba1c_value": "5.9%",
    "hba1c_category": "Prediabetes (5.7 - 6.4%)",
    "fasting_glucose_value": "110.0 mg/dL",
    "fasting_glucose_category": "Prediabetes (100 - 125 mg/dL)",
    "overall_lab_status": "Prediabetes",
    "escalation_recommended": true,
    "escalation_reason": "Elevated glycemic marker detected (HbA1c 5.7-6.4% or Glucose 100-125 mg/dL)."
  },
  "lifestyle_recommendations": [
    {
      "trigger": "Elevated BMI (29.4 kg/m^2)",
      "advice": "A structured weight management plan targeting a 5-7% reduction in body weight has been shown to reduce type 2 diabetes incidence by up to 58% in clinical trials."
    },
    {
      "trigger": "Sedentary / Low Physical Activity",
      "advice": "Incorporate at least 150 minutes of moderate-intensity aerobic physical activity per week (e.g., brisk walking, swimming, cycling), coupled with resistance exercise 2-3 times weekly."
    },
    {
      "trigger": "Elevated Blood Pressure (134/84 mmHg)",
      "advice": "Adopt dietary patterns rich in whole grains, potassium, and magnesium (such as the DASH diet) and reduce dietary sodium intake below 2,300 mg/day."
    },
    {
      "trigger": "History of Dyslipidemia / High Cholesterol",
      "advice": "Prioritize soluble dietary fiber (oats, legumes, flaxseeds) and reduce saturated fats to improve lipid profiles and insulin sensitivity."
    }
  ],
  "disclaimer": "DISCLAIMER: LifeMap AI provides probabilistic risk assessment and lifestyle recommendations for educational and preventive wellness purposes only. It is not intended as medical diagnosis, treatment, or clinical advice. Consult a qualified healthcare professional for formal diagnosis."
}
```

---

## 4. Backend Proxy Implementation (Node.js Express)

To connect the Express backend (`backend/src/routes/reportRoutes.js` or a new `diabetesRoutes.js`):

```javascript
const axios = require('axios');
const DIABETES_SERVICE_URL = process.env.DIABETES_SERVICE_URL || 'http://localhost:8005';

router.post('/api/diabetes/predict', async (req, res) => {
  try {
    const { features, labs } = req.body;
    const response = await axios.post(`${DIABETES_SERVICE_URL}/predict`, {
      features,
      labs
    });
    return res.status(200).json(response.data);
  } catch (error) {
    const status = error.response ? error.response.status : 500;
    const message = error.response ? error.response.data : error.message;
    return res.status(status).json({ error: message });
  }
});
```

---

## 5. Frontend UI Components (Next.js)

The response payload is optimized for frontend display:
1. **Risk Score Gauge**: Render `percentage_risk` and color-code `final_risk_level`:
   - `Low`: Green (`#10B981`)
   - `Moderate`: Amber (`#F59E0B`)
   - `High`: Rose (`#EF4444`)
2. **Clinical Escalation Banner**: If `final_risk_level != base_risk_level`, render an alert banner highlighting `escalation_notes`.
3. **SHAP Waterfall / Feature Impact Card**: Render the top 5 attributions from `top_5_shap_contributions`. Features with `impact == "Increases Risk"` colored red; protective features colored green.
4. **Laboratory Diagnostic Card**: Display `hba1c_category` and `fasting_glucose_category` alongside ADA clinical thresholds.
5. **Actionable Lifestyle Cards**: Map `lifestyle_recommendations` to accordion items for patient empowerment.
