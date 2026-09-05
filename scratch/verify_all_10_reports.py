import os
import sys
import json
import time
import requests

BASE_URL = "http://127.0.0.1:5000/api"
FASTAPI_URL = "http://127.0.0.1:8000"
PDF_DIR = "diabetes_obesity_test_reports_PDF/diabetes_obesity_test_reports_pdf"

def register_user(email, password="Password123!", name="Test User"):
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "name": name,
        "email": email,
        "password": password
    })
    if res.status_code in [200, 201]:
        return res.json().get("token")
    login_res = requests.post(f"{BASE_URL}/auth/login", json={"email": email, "password": password})
    return login_res.json().get("token")

def run_test():
    print("=" * 80)
    print("RUNNING ALL 10 TEST REPORTS FRESH AGAINST ANSWER KEY")
    print("=" * 80)
    email = f"physician_verify_{int(time.time())}@lifemap.ai"
    token = register_user(email)
    assert token, "Failed to get auth token!"
    headers = {"Authorization": f"Bearer {token}"}

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    summary_results = []

    for pdf_name in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        resp = requests.post(
            f"{BASE_URL}/reports/upload",
            headers=headers,
            files={"report": (pdf_name, file_bytes, "application/pdf")}
        )

        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}

        # Check recommendations
        rec_data = None
        if status_code in [200, 201]:
            rec_resp = requests.post(f"{BASE_URL}/recommendations/generate", headers=headers, json={})
            if rec_resp.status_code == 200:
                rec_data = rec_resp.json()

        summary_results.append({
            "file": pdf_name,
            "status_code": status_code,
            "body": body,
            "recommendations": rec_data
        })

    with open("scratch/results_10_files.json", "w") as out_f:
        json.dump(summary_results, out_f, indent=2)
    print("Finished running all 10 reports. Results written to scratch/results_10_files.json")

if __name__ == "__main__":
    run_test()
