import os
import requests
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Fallback: parse .env if variables not yet in os.environ
if not os.environ.get("BACKEND_API_URL") and os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

BASE_URL = os.environ.get("BACKEND_API_URL") or os.environ.get("NEXT_PUBLIC_API_URL")
if not BASE_URL:
    raise ValueError("BACKEND_API_URL environment variable is required.")

TEST_EMAIL = os.environ.get("TEST_EMAIL")
TEST_PASSWORD = os.environ.get("TEST_PASSWORD")
if not TEST_EMAIL or not TEST_PASSWORD:
    raise ValueError("TEST_EMAIL and TEST_PASSWORD environment variables are required.")

def run_tests():
    # Login / register test user
    login_resp = requests.post(f"{BASE_URL}/auth/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }, timeout=5)
    if login_resp.status_code == 200:
        token = login_resp.json()["token"]
    else:
        reg_resp = requests.post(f"{BASE_URL}/auth/register", json={
            "name": "Test Patient",
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
            "role": "patient"
        }, timeout=5)
        token = reg_resp.json()["token"]

    headers = {"Authorization": f"Bearer {token}"}

    def process_pdf(pdf_path):
        with open(pdf_path, "rb") as f:
            files = {"report": (os.path.basename(pdf_path), f, "application/pdf")}
            resp = requests.post(f"{BASE_URL}/reports/metabolic-upload", headers=headers, files=files, timeout=60)
        if resp.status_code != 200:
            return None
        return resp.json()

    res_without = process_pdf("C:/Users/rsaha/Downloads/test_report_WITHOUT_HbA1c.pdf")
    res_with = process_pdf("C:/Users/rsaha/Downloads/test_report_WITH_HbA1c.pdf")
    res_empty = process_pdf("C:/Users/rsaha/Downloads/test_report_EMPTY.pdf")

    def format_row(field_name, exp_without, act_without, exp_with, act_with, exp_empty, act_empty):
        return f"| {field_name:<26} | {str(exp_without):<12} | {str(act_without):<12} | {str(exp_with):<12} | {str(act_with):<12} | {str(exp_empty):<14} | {str(act_empty):<14} |"

    print("\n" + "="*120)
    print("AUTOMATED TEST RESULTS: END-TO-END CLINICAL REPORT EXTRACTION & MODEL PIPELINE")
    print("="*120)
    print("| Field / Property           | Exp (WITHOUT) | Act (WITHOUT) | Exp (WITH)   | Act (WITH)   | Exp (EMPTY)    | Act (EMPTY)    |")
    print("|" + "-"*28 + "|" + "-"*15 + "|" + "-"*15 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*16 + "|" + "-"*16 + "|")

    def get_val(data, key):
        if not data:
            return "Not provided"
        v = data.get("calculated_vitals", {}).get(key)
        if v is None:
            return "Not provided"
        return v

    def get_yn(data, key):
        if not data:
            return "Not provided"
        v = data.get("calculated_vitals", {}).get(key)
        if v is None:
            return "Not provided"
        return "yes" if v == 1 else "no"

    # Compute expected WITH probability directly from saved Tier 2 model
    import joblib
    import pandas as pd
    import sys
    from pathlib import Path

    dm_dir = Path(__file__).resolve().parent.parent / "diabetes_module" / "src"
    if str(dm_dir) not in sys.path:
        sys.path.insert(0, str(dm_dir))
    import train_and_evaluate
    import __main__
    __main__.TabularPreprocessor = train_and_evaluate.TabularPreprocessor
    __main__.ProbabilityCalibrator = train_and_evaluate.ProbabilityCalibrator

    models_dir = Path(__file__).resolve().parent.parent / "diabetes_module" / "models"
    t2_prep = joblib.load(models_dir / "tier2_preprocessor.joblib")
    t2_cal = joblib.load(models_dir / "tier2_final_calibrated_xgb.joblib")

    with_features = {
        "age": float(get_val(res_with, "age")),
        "bmi": float(get_val(res_with, "bmi")),
        "mean_sbp": float(get_val(res_with, "mean_sbp")),
        "mean_dbp": float(get_val(res_with, "mean_dbp")),
        "hba1c": float(get_val(res_with, "hba1c")),
        "sex": 1 if str(get_val(res_with, "gender") or get_val(res_with, "sex")).lower() == "male" else 0,
        "high_bp": 1 if get_yn(res_with, "high_bp") == "yes" else 0,
        "high_chol": 1 if get_yn(res_with, "high_chol") == "yes" else 0,
        "smoker": 1 if get_yn(res_with, "smoker") == "yes" else 0,
        "phys_activity": 1 if get_yn(res_with, "phys_activity") == "yes" else 0,
        "family_history": 1 if get_yn(res_with, "family_history") == "yes" else 0,
    }
    df_with = pd.DataFrame([with_features])
    X_with = t2_prep.transform(df_with)
    exp_with_prob = float(t2_cal.predict_proba(X_with)[0, 1])
    exp_without_prob = 0.4017

    act_without_prob = float(res_without.get("diabetes_tier_prediction", {}).get("predicted_probability"))
    act_with_prob = float(res_with.get("diabetes_tier_prediction", {}).get("predicted_probability"))

    fields = [
        ("Age", 52, get_val(res_without, "age"), 52, get_val(res_with, "age"), "Not provided", get_val(res_empty, "age")),
        ("Sex", "Male", get_val(res_without, "gender") or get_val(res_without, "sex"), "Male", get_val(res_with, "gender") or get_val(res_with, "sex"), "Not provided", get_val(res_empty, "gender") or get_val(res_empty, "sex")),
        ("Height (cm)", 170.0, get_val(res_without, "height_cm"), 170.0, get_val(res_with, "height_cm"), "Not provided", get_val(res_empty, "height_cm")),
        ("Weight (kg)", 85.0, get_val(res_without, "weight_kg"), 85.0, get_val(res_with, "weight_kg"), "Not provided", get_val(res_empty, "weight_kg")),
        ("BMI", 29.4, get_val(res_without, "bmi"), 29.4, get_val(res_with, "bmi"), "Not provided", get_val(res_empty, "bmi")),
        ("Systolic BP (SBP)", 134, get_val(res_without, "mean_sbp"), 134, get_val(res_with, "mean_sbp"), "Not provided", get_val(res_empty, "mean_sbp")),
        ("Diastolic BP (DBP)", 84, get_val(res_without, "mean_dbp"), 84, get_val(res_with, "mean_dbp"), "Not provided", get_val(res_empty, "mean_dbp")),
        ("Fasting Glucose", 112, get_val(res_without, "fasting_glucose"), 112, get_val(res_with, "fasting_glucose"), "Not provided", get_val(res_empty, "fasting_glucose")),
        ("HbA1c", "Not provided", get_val(res_without, "hba1c"), 6.1, get_val(res_with, "hba1c"), "Not provided", get_val(res_empty, "hba1c")),
        ("Total Cholesterol", 224, get_val(res_without, "total_cholesterol"), 224, get_val(res_with, "total_cholesterol"), "Not provided", get_val(res_empty, "total_cholesterol")),
        ("HDL Cholesterol", 41, get_val(res_without, "hdl"), 41, get_val(res_with, "hdl"), "Not provided", get_val(res_empty, "hdl")),
        ("LDL Cholesterol", 148, get_val(res_without, "ldl"), 148, get_val(res_with, "ldl"), "Not provided", get_val(res_empty, "ldl")),
        ("Triglycerides", 175, get_val(res_without, "triglycerides"), 175, get_val(res_with, "triglycerides"), "Not provided", get_val(res_empty, "triglycerides")),
        ("Told High BP", "yes", get_yn(res_without, "high_bp"), "yes", get_yn(res_with, "high_bp"), "Not provided", get_yn(res_empty, "high_bp")),
        ("Told High Cholesterol", "yes", get_yn(res_without, "high_chol"), "yes", get_yn(res_with, "high_chol"), "Not provided", get_yn(res_empty, "high_chol")),
        ("Smoker (100+ cigs)", "no", get_yn(res_without, "smoker"), "no", get_yn(res_with, "smoker"), "Not provided", get_yn(res_empty, "smoker")),
        ("Physically Active", "no", get_yn(res_without, "phys_activity"), "no", get_yn(res_with, "phys_activity"), "Not provided", get_yn(res_empty, "phys_activity")),
        ("Family History Diabetes", "yes", get_yn(res_without, "family_history"), "yes", get_yn(res_with, "family_history"), "Not provided", get_yn(res_empty, "family_history")),
        ("Model Used", "tier1_no_labs", res_without.get("diabetes_tier_prediction", {}).get("model_used") if res_without.get("diabetes_tier_prediction") else "None", "tier2_with_hba1c", res_with.get("diabetes_tier_prediction", {}).get("model_used") if res_with.get("diabetes_tier_prediction") else "None", "None", res_empty.get("diabetes_tier_prediction", {}).get("model_used") if res_empty.get("diabetes_tier_prediction") else "None"),
        ("Risk Probability", f"{exp_without_prob:.4f}", f"{act_without_prob:.4f}", f"{exp_with_prob:.4f}", f"{act_with_prob:.4f}", "None", "None" if not res_empty.get("diabetes_tier_prediction") else "Present"),
        ("Risk Level Badge", "High", res_without.get("diabetes_tier_prediction", {}).get("final_risk_level") if res_without.get("diabetes_tier_prediction") else "None", "Moderate", res_with.get("diabetes_tier_prediction", {}).get("final_risk_level") if res_with.get("diabetes_tier_prediction") else "None", "None", "None" if not res_empty.get("diabetes_tier_prediction") else "Present"),
        ("Audit Gate Status", "Passed", "Passed" if res_without.get("step4_audit", {}).get("audit_passed") else "Failed", "Passed", "Passed" if res_with.get("step4_audit", {}).get("audit_passed") else "Failed", "Passed", "Passed" if res_empty.get("step4_audit", {}).get("audit_passed") else "Failed"),
    ]

    for f in fields:
        print(format_row(f[0], f[1], f[2], f[3], f[4], f[5], f[6]))

    print("-" * 120)

    # Detailed SHAP Factors
    print("\n[TOP SHAP RISK FACTORS - WITHOUT HbA1c]")
    for factor in res_without.get("predictions", {}).get("diabetes", {}).get("explanation", {}).get("risk_increasing_factors", []):
        print(f"  + {factor['label']}: impact {factor['impact_pct']}% (raw value: {factor['raw_value']}) [Source: {factor['source']}]")

    print("\n[TOP SHAP RISK FACTORS - WITH HbA1c]")
    for factor in res_with.get("predictions", {}).get("diabetes", {}).get("explanation", {}).get("risk_increasing_factors", []):
        print(f"  + {factor['label']}: impact {factor['impact_pct']}% (raw value: {factor['raw_value']}) [Source: {factor['source']}]")

    print("\n[PROTECTIVE FACTORS - WITH HbA1c]")
    prot_factors = res_with.get("predictions", {}).get("diabetes", {}).get("explanation", {}).get("protective_factors", [])
    if not prot_factors:
        print("  (None - all abnormal/elevated markers such as glucose 112, BMI 29.4, SBP 134 strictly excluded from protective)")
    else:
        for factor in prot_factors:
            print(f"  - {factor['label']}: impact {factor['impact_pct']}% (raw value: {factor['raw_value']})")

    # Audit for forbidden legacy features
    all_json_str = json.dumps([res_without, res_with, res_empty])
    has_pregnancies = "pregnancies" in all_json_str.lower()
    has_insulin = "serum insulin" in all_json_str.lower()
    print(f"\nLegacy Pima 'Pregnancies' present: {has_pregnancies}")
    print(f"Legacy Pima 'Serum Insulin' present: {has_insulin}")
    assert not has_pregnancies, "Pregnancies feature detected!"
    assert not has_insulin, "Serum insulin feature detected!"

    # Requirement 5 Assertions:
    # 1. WITHOUT file expected probability 0.4017 (tolerance 0.005)
    print(f"\nAsserting WITHOUT probability: actual={act_without_prob:.4f}, expected={exp_without_prob:.4f} (tolerance 0.005)...")
    assert abs(act_without_prob - exp_without_prob) <= 0.005, f"WITHOUT probability {act_without_prob} out of tolerance 0.005 from {exp_without_prob}"

    # 2. WITH file expected probability equal to Tier 2 model output (tolerance 0.005)
    print(f"Asserting WITH probability: actual={act_with_prob:.4f}, expected={exp_with_prob:.4f} (tolerance 0.005)...")
    assert abs(act_with_prob - exp_with_prob) <= 0.005, f"WITH probability {act_with_prob} does not match computed Tier 2 model output {exp_with_prob}"

    # 3. Assert no probability-adjustment code remains in source
    print("Asserting no probability-adjustment code remains in source...")
    app_src = (dm_dir / "app.py").read_text(encoding="utf-8")
    ramp_patterns = [
        "(6.5 - 5.7)",
        "(hba1c - 5.7)",
        "(hba1c_val - 5.7)",
        "0.4017 +",
        "max(raw_prob, 0.68)",
        "max(raw_prob, 0.6800)",
    ]
    for pattern in ramp_patterns:
        assert pattern not in app_src, f"Forbidden ramp/adjustment pattern '{pattern}' found in diabetes_module/src/app.py!"

    print("\n[SUCCESS] All assertions passed!")

if __name__ == "__main__":
    run_tests()
