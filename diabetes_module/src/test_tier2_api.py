"""
Test updated app.py /predict endpoint with 5 scenario calls:
1. No HbA1c (Tier 1)
2. Normal HbA1c (Tier 2, 5.2%)
3. Prediabetic HbA1c (Tier 2, 5.9%)
4. Diabetic-range HbA1c (Tier 2, 8.5%)
5. Invalid HbA1c (25.0% - out of range)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

base_patient = {
    "age": 52.0,
    "sex": 1,
    "bmi": 29.4,
    "mean_sbp": 134.0,
    "mean_dbp": 86.0,
    "high_bp": 1,
    "high_chol": 1,
    "smoker": 0,
    "phys_activity": 1,
    "family_history": 1
}

test_scenarios = [
    {
        "scenario_name": "Call 1: No HbA1c (Pure Tier 1 Request)",
        "payload": {
            "features": base_patient
        }
    },
    {
        "scenario_name": "Call 2: Normal HbA1c (Tier 2, 5.2%)",
        "payload": {
            "features": base_patient,
            "hba1c": 5.2
        }
    },
    {
        "scenario_name": "Call 3: Prediabetic HbA1c (Tier 2, 5.9%)",
        "payload": {
            "features": base_patient,
            "hba1c": 5.9
        }
    },
    {
        "scenario_name": "Call 4: Diabetic-range HbA1c (Tier 2, 8.5%)",
        "payload": {
            "features": base_patient,
            "hba1c": 8.5
        }
    },
    {
        "scenario_name": "Call 5: Invalid HbA1c (Tier 2, 25.0% - Out of Range)",
        "payload": {
            "features": base_patient,
            "hba1c": 25.0
        }
    }
]

results = []
print("=" * 70)
print("TESTING /predict ENDPOINT ACROSS 5 SCENARIOS")
print("=" * 70)

for sc in test_scenarios:
    print(f"\n--- {sc['scenario_name']} ---")
    resp = client.post("/predict", json=sc["payload"])
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print("Response JSON:")
    print(json.dumps(data, indent=2))
    
    results.append({
        "scenario": sc["scenario_name"],
        "status_code": resp.status_code,
        "payload": sc["payload"],
        "response": data
    })

out_json = Path(__file__).resolve().parent.parent / "reports" / "tables" / "tier2_api_example_outputs.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved test outputs to {out_json}")
