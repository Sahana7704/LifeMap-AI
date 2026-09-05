import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api"
PDF_DIR = "diabetes_obesity_test_reports_PDF/diabetes_obesity_test_reports_pdf"

# Register single account
email = f"same_account_test_{int(time.time())}@lifemap.ai"
res = requests.post(f"{BASE_URL}/auth/register", json={
    "name": "Single User Account",
    "email": email,
    "password": "Password123!"
})
token = res.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"Registered single account: {email}")

# Step 1: Upload File 01 (Divya Krishnan: Female, 34, FBG 142, HbA1c 7.1, BMI 30.5, Ht 160, Wt 78)
with open(f"{PDF_DIR}/01_easy_clean_full_panel.pdf", "rb") as f:
    r1 = requests.post(f"{BASE_URL}/reports/upload", headers=headers, files={"report": ("01_easy_clean_full_panel.pdf", f.read(), "application/pdf")})
assert r1.status_code in [200, 201], f"Upload 1 failed: {r1.text}"
m1 = r1.json()["extracted_metrics"]
print("\n[Upload 1 - File 01]:")
print(f"  Patient: {m1.get('gender')}, Age: {m1.get('age')}, FBG: {m1.get('fasting_glucose')}, HbA1c: {m1.get('hba1c')}, BMI: {m1.get('bmi')}, Ht: {m1.get('height_cm')}, Wt: {m1.get('weight_kg')}")

# Step 2: On the SAME account, without logging out, upload File 02 (Rohit Sharma: Male, 29, FBG 91, HbA1c 5.4, NO BMI/Ht/Wt)
with open(f"{PDF_DIR}/02_easy_lab_chain_style.pdf", "rb") as f:
    r2 = requests.post(f"{BASE_URL}/reports/upload", headers=headers, files={"report": ("02_easy_lab_chain_style.pdf", f.read(), "application/pdf")})
assert r2.status_code in [200, 201], f"Upload 2 failed: {r2.text}"
m2 = r2.json()["extracted_metrics"]
print("\n[Upload 2 - File 02 (SAME ACCOUNT, NO LOGOUT)]:")
print(f"  Patient: {m2.get('gender')}, Age: {m2.get('age')}, FBG: {m2.get('fasting_glucose')}, HbA1c: {m2.get('hba1c')}, BMI: {m2.get('bmi')}, Ht: {m2.get('height_cm')}, Wt: {m2.get('weight_kg')}")

# Contamination Assertions for Upload 2
assert m2.get('gender') == 'Male', f"Contamination! Expected Male, got {m2.get('gender')}"
assert m2.get('age') == 29, f"Contamination! Expected 29, got {m2.get('age')}"
assert m2.get('fasting_glucose') == 91, f"Contamination! Expected 91, got {m2.get('fasting_glucose')}"
assert m2.get('hba1c') == 5.4, f"Contamination! Expected 5.4, got {m2.get('hba1c')}"
assert m2.get('bmi') is None, f"Contamination! BMI from File 01 leaked: {m2.get('bmi')}"
assert m2.get('height_cm') is None, f"Contamination! Height from File 01 leaked: {m2.get('height_cm')}"
assert m2.get('weight_kg') is None, f"Contamination! Weight from File 01 leaked: {m2.get('weight_kg')}"

# Step 3: Check recommendations for File 02 (must NOT fabricate BMR from File 01's 160cm/78kg)
rec2 = requests.post(f"{BASE_URL}/recommendations/generate", headers=headers, json={}).json()
bd2 = rec2.get("diet_plan", {}).get("calorie_breakdown", {})
print(f"  Recommendations after File 02 -> BMR: {bd2.get('bmr')}, Gender: {bd2.get('gender')}")
assert bd2.get('bmr') is None, f"Contamination! BMR was fabricated from File 01: {bd2.get('bmr')}"

# Step 4: Upload File 09 on SAME account (Naveen Kumar: Male, 31, NO glucose, BMI 29.1, Ht 175, Wt 89, Waist 98)
with open(f"{PDF_DIR}/09_hard_partial_data_bmi_only.pdf", "rb") as f:
    r3 = requests.post(f"{BASE_URL}/reports/upload", headers=headers, files={"report": ("09_hard_partial_data_bmi_only.pdf", f.read(), "application/pdf")})
assert r3.status_code in [200, 201], f"Upload 3 failed: {r3.text}"
m3 = r3.json()["extracted_metrics"]
print("\n[Upload 3 - File 09 (SAME ACCOUNT, NO LOGOUT)]:")
print(f"  Patient: {m3.get('gender')}, Age: {m3.get('age')}, FBG: {m3.get('fasting_glucose')}, HbA1c: {m3.get('hba1c')}, BMI: {m3.get('bmi')}, Ht: {m3.get('height_cm')}, Wt: {m3.get('weight_kg')}, Waist: {m3.get('waist_circumference_cm')}")

assert m3.get('fasting_glucose') is None, f"Contamination! Glucose leaked into File 09: {m3.get('fasting_glucose')}"
assert m3.get('hba1c') is None, f"Contamination! HbA1c leaked into File 09: {m3.get('hba1c')}"
assert m3.get('bmi') == 29.1, f"Expected BMI 29.1, got {m3.get('bmi')}"
assert m3.get('waist_circumference_cm') == 98, f"Expected Waist 98, got {m3.get('waist_circumference_cm')}"

print("\n>>> ALL CONTAMINATION ASSERTIONS PASSED! ZERO CROSS-REPORT CONTAMINATION DETECTED! <<<")
