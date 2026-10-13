"""
Test script to run the 3 required example inputs against the FastAPI service
using starlette TestClient, verifying:
1. Low risk individual with normal profile (no lab values provided).
2. Moderate risk individual with prediabetic labs (HbA1c = 5.9%, fasting glucose = 110 mg/dL).
3. High risk individual with missing lab fields (e.g. fasting_glucose=None / only hba1c or no labs at all)
   verifying graceful handling and zero lab inference.
"""

from fastapi.testclient import TestClient
import json
from pathlib import Path
import sys

# Ensure src is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from app import app

client = TestClient(app)

def test_endpoints():
    print("Testing GET / ...")
    r0 = client.get("/")
    print("GET / Response:", r0.json())
    assert r0.status_code == 200

    # Example 1: Healthy 25yo, active, normal BMI, no labs
    case1 = {
        "features": {
            "age": 25.0,
            "sex": 0,
            "bmi": 21.5,
            "mean_sbp": 115.0,
            "mean_dbp": 72.0,
            "high_bp": 0,
            "high_chol": 0,
            "smoker": 0,
            "phys_activity": 1,
            "family_history": 0
        },
        "labs": None
    }

    # Example 2: 52yo male, overweight, sedentary, prediabetes labs
    case2 = {
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

    # Example 3: 65yo female, high BP, high BMI, missing fasting_glucose (missing value test)
    case3 = {
        "features": {
            "age": 65.0,
            "sex": 0,
            "bmi": 36.2,
            "mean_sbp": 145.0,
            "mean_dbp": 90.0,
            "high_bp": 1,
            "high_chol": 1,
            "smoker": 1,
            "phys_activity": 0,
            "family_history": 1
        },
        "labs": {
            "hba1c": 7.1,
            "fasting_glucose": None
        }
    }

    results = {}
    for idx, (name, payload) in enumerate([("Case 1 (Low Risk, No Labs)", case1),
                                          ("Case 2 (Moderate Risk, Prediabetic Labs)", case2),
                                          ("Case 3 (High Risk, Missing Glucose Lab)", case3)], 1):
        print(f"\n==================== Running {name} ====================")
        res = client.post("/predict", json=payload)
        assert res.status_code == 200, f"Failed with {res.status_code}: {res.text}"
        data = res.json()
        print(json.dumps(data, indent=2))
        results[f"case_{idx}"] = {
            "input": payload,
            "output": data
        }

    out_file = src_dir.parent / "reports" / "tables" / "api_example_outputs.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved example outputs to {out_file}")

if __name__ == "__main__":
    test_endpoints()
