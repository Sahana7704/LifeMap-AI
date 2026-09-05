import json
import urllib.request
import urllib.parse
import mimetypes
import uuid
import sys

BASE_URL = "http://localhost:5000/api"

def create_multipart_form(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()

    for k, v in fields.items():
        body.extend(f'--{boundary}\r\n'.encode('utf-8'))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode('utf-8'))
        body.extend(f'{v}\r\n'.encode('utf-8'))

    for field_name, (filename, file_content, mimetype) in files.items():
        body.extend(f'--{boundary}\r\n'.encode('utf-8'))
        body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode('utf-8'))
        body.extend(f'Content-Type: {mimetype}\r\n\r\n'.encode('utf-8'))
        if isinstance(file_content, str):
            body.extend(file_content.encode('utf-8'))
        else:
            body.extend(file_content)
        body.extend(b'\r\n')

    body.extend(f'--{boundary}--\r\n'.encode('utf-8'))
    content_type = f'multipart/form-data; boundary={boundary}'
    return body, content_type

def make_request(url, method="GET", headers=None, data=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"error": err_body}

def get_auth_token():
    test_user = f"testuser_{uuid.uuid4().hex[:6]}"
    signup_data = json.dumps({
        "identifier": test_user,
        "name": "Clinical Test Subject",
        "email": f"{test_user}@lifemap.ai",
        "password": "Password123!",
        "age": 42,
        "gender": "Male"
    }).encode('utf-8')
    status, res = make_request(f"{BASE_URL}/auth/register", method="POST",
                               headers={"Content-Type": "application/json"}, data=signup_data)
    if status == 201 and "token" in res:
        return res["token"], test_user
    # Fallback to login
    login_data = json.dumps({"identifier": test_user, "password": "Password123!"}).encode('utf-8')
    status, res = make_request(f"{BASE_URL}/auth/login", method="POST",
                               headers={"Content-Type": "application/json"}, data=login_data)
    return res["token"], test_user

def upload_metabolic_report(token, report_text, filename):
    body, ctype = create_multipart_form({}, {
        "report": (filename, report_text, "text/plain")
    })
    status, res = make_request(
        f"{BASE_URL}/reports/metabolic-upload",
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
        data=body
    )
    return status, res

def get_recommendations(token, body_json=None):
    data = json.dumps(body_json or {}).encode('utf-8')
    status, res = make_request(
        f"{BASE_URL}/recommendations/generate",
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=data
    )
    return status, res

def test_bug_h():
    print("\n=======================================================")
    print("TESTING BUG H: Cross-Report Contamination (3 Pairs)")
    print("=======================================================")
    token, username = get_auth_token()
    print(f"Logged in user: {username}")

    # PAIR 1: Report A1 has full anthropometry & diabetic glucose; Report B1 has only normal glucose & HbA1c
    rep_a1 = """METABOLIC LAB REPORT
Patient: Anita Roy
Age: 45 | Female
Fasting Blood Glucose: 154 mg/dL  *HIGH*
HbA1c: 7.8 %  *HIGH*
Height: 168 cm
Weight: 84 kg
Waist Circumference: 92 cm
"""
    rep_b1 = """METABOLIC LAB REPORT
Patient: Anita Roy
Age: 45 | Female
Fasting Blood Glucose: 86 mg/dL
HbA1c: 5.1 %
"""
    print("\n--- Testing Pair 1 ---")
    st_a, res_a = upload_metabolic_report(token, rep_a1, "pair1_report_a.txt")
    assert st_a in [200, 201], f"Upload A1 failed: {res_a}"
    metrics_a = res_a.get("extracted_metrics", {})
    assert metrics_a.get("fasting_glucose") == 154
    assert metrics_a.get("height_cm") == 168
    assert metrics_a.get("weight_kg") == 84
    assert metrics_a.get("waist_circumference_cm") == 92
    print("Report A1 uploaded successfully: FG=154, Ht=168, Wt=84, Waist=92")

    # Upload Report B1 WITHOUT logging out
    st_b, res_b = upload_metabolic_report(token, rep_b1, "pair1_report_b.txt")
    assert st_b in [200, 201], f"Upload B1 failed: {res_b}"
    metrics_b = res_b.get("extracted_metrics", {})
    vitals_b = res_b.get("verified_vitals", {})

    print(f"Report B1 extracted: FG={metrics_b.get('fasting_glucose')}, HbA1c={metrics_b.get('hba1c')}, Ht={metrics_b.get('height_cm')}, Wt={metrics_b.get('weight_kg')}, Waist={metrics_b.get('waist_circumference_cm')}")

    # Assert ZERO carry-over from Report A1
    assert metrics_b.get("fasting_glucose") == 86, f"Expected FG 86, got {metrics_b.get('fasting_glucose')}"
    assert metrics_b.get("hba1c") == 5.1, f"Expected HbA1c 5.1, got {metrics_b.get('hba1c')}"
    assert metrics_b.get("height_cm") is None, f"Contamination: Height {metrics_b.get('height_cm')} carried over!"
    assert metrics_b.get("weight_kg") is None, f"Contamination: Weight {metrics_b.get('weight_kg')} carried over!"
    assert metrics_b.get("waist_circumference_cm") is None, f"Contamination: Waist {metrics_b.get('waist_circumference_cm')} carried over!"
    assert vitals_b.get("height_cm") is None, f"Contamination: Vitals Height carried over!"
    assert vitals_b.get("weight_kg") is None, f"Contamination: Vitals Weight carried over!"
    print("PASS: Pair 1 Report B1 contains ZERO fields from Report A1!")

    # PAIR 2: Report A2 has height/weight/waist only; Report B2 has Random Glucose 165 only
    rep_a2 = """ANTHROPOMETRY REPORT
Patient: Sameer Verma
Age: 35 | Male
Height: 178 cm
Weight: 92 kg
Waist Circumference: 96 cm
BMI: 29.0 kg/m²
"""
    rep_b2 = """BLOOD GLUCOSE REPORT
Patient: Sameer Verma
Age: 35 | Male
Random Blood Glucose: 165 mg/dL  *HIGH*
"""
    print("\n--- Testing Pair 2 ---")
    st_a2, res_a2 = upload_metabolic_report(token, rep_a2, "pair2_report_a.txt")
    assert st_a2 in [200, 201], f"Upload A2 failed: {res_a2}"
    print("Report A2 uploaded: Ht=178, Wt=92, Waist=96, BMI=29.0")

    st_b2, res_b2 = upload_metabolic_report(token, rep_b2, "pair2_report_b.txt")
    assert st_b2 in [200, 201], f"Upload B2 failed: {res_b2}"
    metrics_b2 = res_b2.get("extracted_metrics", {})
    print(f"Report B2 extracted: RG={metrics_b2.get('random_glucose')}, FG={metrics_b2.get('fasting_glucose')}, Ht={metrics_b2.get('height_cm')}, Wt={metrics_b2.get('weight_kg')}")
    assert metrics_b2.get("random_glucose") == 165
    assert metrics_b2.get("fasting_glucose") is None
    assert metrics_b2.get("height_cm") is None, f"Contamination: Height carried over in Pair 2!"
    assert metrics_b2.get("weight_kg") is None, f"Contamination: Weight carried over in Pair 2!"
    assert metrics_b2.get("waist_circumference_cm") is None, f"Contamination: Waist carried over in Pair 2!"
    print("PASS: Pair 2 Report B2 contains ZERO fields from Report A2!")

    # PAIR 3: Report A3 has HbA1c 8.4%; Report B3 has Fasting Glucose 92 & Height 172 & Weight 64
    rep_a3 = """DIABETES MONITORING
Patient: Priya Sen
Age: 50 | Female
HbA1c: 8.4 %  *HIGH*
"""
    rep_b3 = """ANNUAL HEALTH REPORT
Patient: Priya Sen
Age: 50 | Female
Fasting Blood Sugar: 92 mg/dL
Height: 172 cm
Weight: 64 kg
"""
    print("\n--- Testing Pair 3 ---")
    st_a3, res_a3 = upload_metabolic_report(token, rep_a3, "pair3_report_a.txt")
    assert st_a3 in [200, 201], f"Upload A3 failed: {res_a3}"
    print("Report A3 uploaded: HbA1c=8.4")

    st_b3, res_b3 = upload_metabolic_report(token, rep_b3, "pair3_report_b.txt")
    assert st_b3 in [200, 201], f"Upload B3 failed: {res_b3}"
    metrics_b3 = res_b3.get("extracted_metrics", {})
    print(f"Report B3 extracted: FG={metrics_b3.get('fasting_glucose')}, HbA1c={metrics_b3.get('hba1c')}, Ht={metrics_b3.get('height_cm')}, Wt={metrics_b3.get('weight_kg')}")
    assert metrics_b3.get("fasting_glucose") == 92
    assert metrics_b3.get("height_cm") == 172
    assert metrics_b3.get("weight_kg") == 64
    assert metrics_b3.get("hba1c") is None, f"Contamination: HbA1c 8.4 carried over into Report B3!"
    print("PASS: Pair 3 Report B3 contains ZERO fields from Report A3!")
    print("\nBUG H: ALL 3 VERIFICATION PAIRS PASSED ZERO-CONTAMINATION CHECK!")

def test_bug_i():
    print("\n=======================================================")
    print("TESTING BUG I: Multi-Section Extraction Across Headers")
    print("=======================================================")
    token, username = get_auth_token()

    multi_section_report = """==================================================
DR. LAL PATHLABS CLINICAL REFERENCE LABORATORY
Patient: Rajesh Gupta
Age: 52 Years / Male
Date of Collection: 05-Sep-2026
==================================================
DEPARTMENT OF BIOCHEMISTRY
TEST NAME: HbA1c (GLYCATED HEMOGLOBIN)
Method: High Performance Liquid Chromatography (HPLC)
Result: 7.4 %
Biological Reference Interval:
< 5.7 % : Normal
5.7 - 6.4 % : Prediabetes
>= 6.5 % : Diabetes
Status: High

--------------------------------------------------
DEPARTMENT OF ENDOCRINOLOGY
TEST NAME: FASTING BLOOD SUGAR (GLUCOSE)
Method: Hexokinase / UV
Result: 138 mg/dL
Biological Reference Interval:
70 - 100 mg/dL : Normal
101 - 125 mg/dL : Impaired Fasting Glucose
>= 126 mg/dL : Provisional Diabetes
Status: High
==================================================
"""
    status, res = upload_metabolic_report(token, multi_section_report, "dr_lal_multisection.txt")
    assert status in [200, 201], f"Upload failed: {res}"
    metrics = res.get("extracted_metrics", {})
    verified = res.get("verified_vitals", {})
    diabetes = res.get("diabetes_assessment", {})

    print(f"Extracted Metrics: {metrics}")
    print(f"Verified Vitals: {verified}")
    print(f"Diabetes Classifications: {[c.get('marker') + ' -> ' + c.get('classification') for c in diabetes.get('classifications', [])]}")

    # Confirm BOTH values were extracted
    assert metrics.get("hba1c") == 7.4, f"Failed: HbA1c not extracted (got {metrics.get('hba1c')})"
    assert metrics.get("fasting_glucose") == 138, f"Failed: Fasting Glucose not extracted (got {metrics.get('fasting_glucose')})"

    # Confirm BOTH values appear in the classifications
    markers = [c.get("marker", "").lower() for c in diabetes.get("classifications", [])]
    has_fbg = any("glucose" in m or "fasting" in m for m in markers)
    has_a1c = any("hba1c" in m or "a1c" in m or "glycated" in m for m in markers)
    assert has_fbg, "Failed: Fasting Glucose missing from diabetes classifications"
    assert has_a1c, "Failed: HbA1c missing from diabetes classifications"

    print("PASS: Both Fasting Blood Sugar (138) and HbA1c (7.4) extracted from multi-section report!")
    print("BUG I: PASSED!")

def test_bug_j():
    print("\n=======================================================")
    print("TESTING BUG J: Nutrition Default Height/Weight/BMI Fabrication")
    print("=======================================================")
    token, username = get_auth_token()

    # Upload report with NO height, NO weight, NO BMI
    glucose_only_report = """DIABETIC PROFILE LAB REPORT
Patient: Vikram Mehta
Age: 48 Years
Gender: Male
TEST PARAMETERS:
Fasting Blood Sugar: 128 mg/dL
HbA1c: 6.9 %
"""
    status, rep_res = upload_metabolic_report(token, glucose_only_report, "glucose_only.txt")
    assert status in [200, 201], f"Upload failed: {rep_res}"
    metrics = rep_res.get("extracted_metrics", {})
    assert metrics.get("height_cm") is None
    assert metrics.get("weight_kg") is None
    assert metrics.get("bmi") is None

    # Request nutrition recommendations
    rec_status, rec_res = get_recommendations(token)
    assert rec_status == 200, f"Recommendations failed: {rec_res}"

    diet_plan = rec_res.get("diet_plan", {})
    breakdown = diet_plan.get("calorie_breakdown", {})
    rationale = diet_plan.get("clinical_rationale", "")

    print(f"Nutrition Breakdown formula: {breakdown.get('formula')}")
    print(f"Anthropometrics Available: {breakdown.get('anthropometrics_available')}")
    print(f"Reported height_cm: {breakdown.get('height_cm')}")
    print(f"Reported weight_kg: {breakdown.get('weight_kg')}")
    print(f"Reported bmr: {breakdown.get('bmr')}")
    print(f"Calorie Target: {diet_plan.get('calorie_target')}")
    print(f"Calculation Steps: {breakdown.get('calculation_steps')}")
    print(f"Clinical Rationale: {rationale}")

    # 1. Height, Weight, and BMR must NOT be fabricated
    assert breakdown.get("height_cm") is None, f"Fabrication detected! height_cm={breakdown.get('height_cm')}"
    assert breakdown.get("weight_kg") is None, f"Fabrication detected! weight_kg={breakdown.get('weight_kg')}"
    assert breakdown.get("bmr") is None, f"Fabrication detected! bmr={breakdown.get('bmr')}"
    assert breakdown.get("anthropometrics_available") is False, "anthropometrics_available should be False"

    # 2. Never assert '172 cm' or '71.0 kg' in calculation steps or rationale
    full_text = " ".join(breakdown.get("calculation_steps", [])) + " " + rationale
    assert "172" not in full_text, f"Fabricated '172' found in nutrition text: {full_text}"
    assert "71.0" not in full_text and "71 kg" not in full_text, f"Fabricated '71' found in nutrition text: {full_text}"

    # 3. Must clearly state unpersonalized population default baseline
    assert "no personalization" in full_text.lower() or "not found" in full_text.lower() or "standard" in full_text.lower()
    print("PASS: Calorie target is clearly labeled as non-personalized standard baseline with NO fabricated height/weight!")
    print("BUG J: PASSED!")

if __name__ == "__main__":
    try:
        test_bug_h()
        test_bug_i()
        test_bug_j()
        print("\n=======================================================")
        print("ALL TESTS FOR BUGS H, I, J COMPLETED AND PASSED!")
        print("=======================================================")
    except Exception as e:
        print(f"\nTEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
