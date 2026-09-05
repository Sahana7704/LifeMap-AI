import sys
import os
import json
import requests

# Add ml_service to path
ml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_service"))
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

from metabolic_pipeline import execute_metabolic_pipeline, step4_metabolic_post_generation_audit
from ocr_pipeline import process_medical_report, parse_metrics_from_text
from recommend_engine import calculate_recommended_calories, generate_diet_plan

def test_all():
    print("==================================================================")
    print("RUNNING LIFEMAP AI COMPREHENSIVE REGRESSION & BUG FIX TEST SUITE")
    print("==================================================================")

    passed_count = 0
    total_count = 0

    # -----------------------------------------------------------------
    # TEST 1: BUG A — Demographics / Height / Weight Nutrition Alignment
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 1: BUG A (Demographics in Nutrition Module) ---")
    female_report = """METABOLIC REPORT
Patient Name: Anita Sen
Age: 36
Gender: Female
Height: 160 cm
Weight: 65 kg
Fasting Blood Sugar: 104 mg/dL
"""
    male_report = """METABOLIC REPORT
Patient Name: Vikram Roy
Age: 45
Gender: Male
Height: 175 cm
Weight: 84 kg
Fasting Blood Sugar: 112 mg/dL
"""
    res_f = execute_metabolic_pipeline(female_report, "anita.txt")
    v_f = res_f["verified_vitals"]
    cal_f, rat_f, bd_f = calculate_recommended_calories({
        "age": v_f.get("age"),
        "gender": v_f.get("sex"),
        "height_cm": v_f.get("height_cm"),
        "weight_kg": v_f.get("weight_kg"),
        "bmi": v_f.get("bmi", {}).get("value")
    })

    res_m = execute_metabolic_pipeline(male_report, "vikram.txt")
    v_m = res_m["verified_vitals"]
    cal_m, rat_m, bd_m = calculate_recommended_calories({
        "age": v_m.get("age"),
        "gender": v_m.get("sex"),
        "height_cm": v_m.get("height_cm"),
        "weight_kg": v_m.get("weight_kg"),
        "bmi": v_m.get("bmi", {}).get("value")
    })

    print(f"Female report: Height={bd_f['height_cm']}cm, Weight={bd_f['weight_kg']}kg, Gender={bd_f['gender']}, BMR={bd_f['bmr']}")
    print(f"Male report: Height={bd_m['height_cm']}cm, Weight={bd_m['weight_kg']}kg, Gender={bd_m['gender']}, BMR={bd_m['bmr']}")

    assert bd_f['height_cm'] == 160.0, f"Expected female height 160.0, got {bd_f['height_cm']}"
    assert bd_f['weight_kg'] == 65.0, f"Expected female weight 65.0, got {bd_f['weight_kg']}"
    assert bd_f['gender'] == "Female", f"Expected female gender, got {bd_f['gender']}"

    assert bd_m['height_cm'] == 175.0, f"Expected male height 175.0, got {bd_m['height_cm']}"
    assert bd_m['weight_kg'] == 84.0, f"Expected male weight 84.0, got {bd_m['weight_kg']}"
    assert bd_m['gender'] == "Male", f"Expected male gender, got {bd_m['gender']}"
    assert bd_f['height_cm'] != 158.0 or bd_m['height_cm'] != 158.0, "Bug A still present: both defaulted to 158cm!"

    # Case 3: Only BMI / Height / Weight (No Glucose) - Tests Bug A specific checklist requirement
    anthropometric_only = """ANTHROPOMETRIC ASSESSMENT
Patient Name: Manisha Patel
Age: 38
Gender: Female
Height: 162 cm
Weight: 70 kg
"""
    res_a = execute_metabolic_pipeline(anthropometric_only, "manisha.txt")
    v_a = res_a["verified_vitals"]
    assert v_a.get("height_cm") == 162.0
    assert v_a.get("weight_kg") == 70.0
    assert res_a["obesity_assessment"]["bmi_value"] == 26.7

    cal_a, rat_a, bd_a = calculate_recommended_calories({
        "age": v_a.get("age"),
        "gender": v_a.get("sex"),
        "height_cm": v_a.get("height_cm"),
        "weight_kg": v_a.get("weight_kg"),
        "bmi": v_a.get("bmi", {}).get("value")
    })
    print(f"Anthropometrics-only report (No Glucose): Obesity BMI={res_a['obesity_assessment']['bmi_value']}, Nutrition Ht={bd_a['height_cm']}cm, Wt={bd_a['weight_kg']}kg")
    assert bd_a['height_cm'] == v_a['height_cm'] == 162.0, "Height mismatch between obesity module and nutrition module!"
    assert bd_a['weight_kg'] == v_a['weight_kg'] == 70.0, "Weight mismatch between obesity module and nutrition module!"
    print(">>> TEST 1 PASSED: Nutrition module reads exact extracted demographics and vitals (including no-glucose anthropometric reports).")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 2: BUG B — HbA1c Treated as First-Class Extracted Field
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 2: BUG B (First-Class HbA1c Field) ---")
    hba1c_report = """DIABETIC HEALTH PROFILE
Patient Name: Rahul Verma
Age: 52
Gender: Male
Fasting Glucose: 138 mg/dL *HIGH*
HbA1c: 7.1 % (Reference: < 5.7 %) *HIGH*
Height: 172 cm
Weight: 78 kg
"""
    res_b = execute_metabolic_pipeline(hba1c_report, "rahul_hba1c.txt")
    v_b = res_b["verified_vitals"]
    da_b = res_b["diabetes_assessment"]

    print("Verified vitals:", json.dumps(v_b, indent=2))
    print("Classifications:", json.dumps(da_b["classifications"], indent=2))

    assert v_b.get("hba1c") is not None, "HbA1c missing from verified vitals!"
    assert v_b["hba1c"]["value"] == 7.1, f"Expected HbA1c 7.1, got {v_b['hba1c']['value']}"
    assert v_b["hba1c"]["unit"] == "%"

    hba1c_class = next((c for c in da_b["classifications"] if "HbA1c" in c["marker"]), None)
    assert hba1c_class is not None, "HbA1c classification missing from diabetes_assessment!"
    assert "Diagnostic" in hba1c_class["classification"], f"Expected Diagnostic classification, got {hba1c_class['classification']}"
    assert hba1c_class["source_flagged"] is True, "HbA1c should be source_flagged=True!"

    # Verify both Fasting Glucose AND HbA1c are present in classifications
    fbg_class = next((c for c in da_b["classifications"] if "Glucose" in c["marker"]), None)
    assert fbg_class is not None, "Fasting glucose classification missing!"
    print(">>> TEST 2 PASSED: HbA1c extracted, classified, and tagged as dedicated first-class field.")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 3: BUG C — 4 Valid Fields Do NOT Trigger Data-Limited Warning
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 3: BUG C (Field Count & Data-Limited Logic) ---")
    partial_4field_report = """ANTHROPOMETRIC & BODY COMPOSITION REPORT
Patient Name: Devendra K
Age: 31
Gender: Male
Height: 175 cm
Weight: 89 kg
BMI: 29.1 kg/m2 (Printed) *HIGH*
Waist Circumference: 98 cm *HIGH*
"""
    res_c = execute_metabolic_pipeline(partial_4field_report, "devendra.txt")
    print("Data limited flag:", res_c.get("data_limited"))
    print("Combined risk note:", res_c.get("combined_risk_note"))

    assert res_c.get("data_limited") is False, "Bug C still active: 4 valid fields triggered data_limited=True!"
    assert "Assessment is data-limited as fewer than two metabolic fields" not in res_c.get("combined_risk_note", ""), "Data-limited message wrongly triggered!"
    assert "Obese" in res_c.get("obesity_assessment", {}).get("classification", "")
    assert res_c.get("obesity_assessment", {}).get("central_obesity") is True
    print(">>> TEST 3 PASSED: 4 valid fields correctly count as sufficient data, no data-limited warning.")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 4: BUG D — Diabetes Report NOT Classified as CBC Hematology
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 4: BUG D (Report Classification) ---")
    diabetes_doc = """METABOLIC PANEL & DIABETIC SCREEN
Patient Name: Sunita Rao
Age: 49
Gender: Female
Fasting Blood Sugar: 128 mg/dL
Glycated Hemoglobin (HbA1c): 7.1 % *HIGH*
Height: 156 cm
Weight: 72 kg
"""
    # Test through general process_medical_report
    parsed_d = process_medical_report(diabetes_doc.encode('utf-8'), "sunita_diabetes.txt")
    rep_type_d = parsed_d.get("report_type")
    extracted_d = parsed_d.get("extracted_metrics", {})
    print("Report Type:", rep_type_d)
    print("Extracted Hemoglobin:", extracted_d.get("hemoglobin"))
    print("Extracted Fasting Glucose:", extracted_d.get("fasting_glucose"))
    print("Extracted HbA1c:", extracted_d.get("hba1c"))

    assert "CBC" not in rep_type_d and "Complete Blood Count" not in rep_type_d, f"Bug D reproduced: misclassified as {rep_type_d}!"
    assert rep_type_d == "Metabolic Panel (Diabetes & Obesity)" or "Diabetic" in rep_type_d, f"Unexpected report type: {rep_type_d}"
    assert extracted_d.get("hemoglobin") is None, f"HbA1c 7.1 was falsely captured as Hemoglobin: {extracted_d.get('hemoglobin')}!"
    assert extracted_d.get("hba1c") == 7.1 or extracted_d.get("hba1c") == "7.1%", f"HbA1c missing or wrong: {extracted_d.get('hba1c')}"
    print(">>> TEST 4 PASSED: Diabetes report correctly dispatched as Metabolic, never CBC, zero fake hemoglobin.")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 5: BUG E — Audit Gate Trustworthiness
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 5: BUG E (Audit Gate Groundedness) ---")
    # Ground truth Step 2 JSON
    verified_step2 = {
        "fasting_glucose": {"value": 115.0, "source_flagged": True},
        "bmi": {"value": 26.2, "source": "calculated", "source_flagged": True},
        "required_flagged_checklist": ["Fasting Blood Glucose", "BMI"]
    }
    # Case 1: Grounded Step 3 output
    grounded_step3 = {
        "combined_risk_note": "Adiposity evaluation (Obese: BMI 26.2 kg/m²) and glycemic assessment (Prediabetes range) indicate elevated risk.",
        "diabetes_assessment": {
            "summary": "Fasting glucose indicates Prediabetes range.",
            "classifications": [{
                "marker": "Fasting Blood Glucose",
                "value": "115.0 mg/dL",
                "classification": "Prediabetes range",
                "diagnostic_threshold": "ADA Standard: >=126 mg/dL Diabetes, 100-125 mg/dL Prediabetes",
                "source_flagged": True
            }]
        },
        "obesity_assessment": {
            "bmi_value": 26.2,
            "classification": "Obese (Asian cutoff; standard WHO cutoff is >=30)",
            "source_flagged": True
        }
    }
    audit_grounded = step4_metabolic_post_generation_audit(verified_step2, grounded_step3)
    print("Grounded audit passed:", audit_grounded["audit_passed"])
    assert audit_grounded["audit_passed"] is True, f"Grounded output failed audit: {audit_grounded['audit_issues']}"

    # Case 2: Hallucinated / Ungrounded Number (e.g. 185 mg/dL never in source)
    ungrounded_step3 = {
        "combined_risk_note": "Fasting glucose was dangerously elevated at 185 mg/dL with BMI 26.2 kg/m².",
        "diabetes_assessment": {
            "summary": "Fasting glucose was 185 mg/dL.",
            "classifications": []
        },
        "obesity_assessment": {"bmi_value": 26.2}
    }
    audit_ungrounded = step4_metabolic_post_generation_audit(verified_step2, ungrounded_step3)
    print("Ungrounded audit passed (should be False):", audit_ungrounded["audit_passed"])
    print("Ungrounded audit issues detected:", audit_ungrounded["audit_issues"])
    assert audit_ungrounded["audit_passed"] is False, "Bug E present: ungrounded 185 mg/dL passed audit!"
    assert any("185" in iss for iss in audit_ungrounded["audit_issues"]), "Audit failed to identify ungrounded 185!"
    print(">>> TEST 5 PASSED: Audit gate strictly catches fabricated numbers and confirms grounded output.")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 6: BUG F — All-Normal Report Consistency & Non-Contradiction
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 6: BUG F (Normal Report Language Consistency) ---")
    normal_report = """METABOLIC HEALTH REPORT
Patient Name: Neha Gupta
Age: 29
Gender: Female
Fasting Blood Sugar: 92 mg/dL (Reference: 70 - 99 mg/dL)
Height: 165 cm
Weight: 58 kg
BMI: 21.3 kg/m2 (Reference: 18.5 - 22.9 kg/m2)
"""
    res_f = execute_metabolic_pipeline(normal_report, "neha_normal.txt")
    note_f = res_f.get("combined_risk_note", "")
    print("Normal Patient Combined Risk Note:")
    print("  ->", note_f)

    # Must contain reassuring language
    assert "healthy normal reference ranges" in note_f or "standard reference ranges" in note_f, "Missing normal baseline confirmation!"
    # Strictly forbid contradictory risk-escalating language
    contradictory_words = ["compounds", "elevated concern", "requiring intervention", "insulin resistance", "elevated metabolic concern"]
    for w in contradictory_words:
        assert w not in note_f.lower(), f"Bug F reproduced: found risk-escalating word '{w}' in all-Normal patient output!"
    print(">>> TEST 6 PASSED: All-Normal report contains reassuring language and zero risk-escalating contradictions.")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 7: BUG G — Identical Output on Consecutive Runs
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 7: BUG G (Deterministic Consistency Across Runs) ---")
    test_report = """METABOLIC LAB REPORT
Patient Name: Rakesh Mehta
Age: 54
Gender: Male
Fasting Blood Glucose: 160 mg/dL *HIGH*
Height: 168 cm
Weight: 96 kg
BMI: 34.0 kg/m2 *HIGH*
"""
    runs = []
    for i in range(3):
        res = execute_metabolic_pipeline(test_report, f"run_{i}.txt")
        # Extract core metrics and classifications
        run_snapshot = {
            "fg": res["verified_vitals"]["fasting_glucose"]["value"],
            "bmi": res["verified_vitals"]["bmi"]["value"],
            "diabetes_status": res["diabetes_assessment"]["fasting_glucose_status"],
            "obesity_class": res["obesity_assessment"]["classification"],
            "combined_note": res["combined_risk_note"]
        }
        runs.append(run_snapshot)

    print("Run 1:", runs[0])
    print("Run 2:", runs[1])
    print("Run 3:", runs[2])

    assert runs[0] == runs[1] == runs[2], "Bug G reproduced: consecutive runs produced different results!"
    print(">>> TEST 7 PASSED: Consecutive uploads of identical reports produce 100% deterministic output.")
    passed_count += 1

    # -----------------------------------------------------------------
    # TEST 8: CONFIRMED WORKING BEHAVIORS (Verify No Regression)
    # -----------------------------------------------------------------
    total_count += 1
    print("\n--- TEST 8: CONFIRMED WORKING BEHAVIORS (Non-relevant Hard-Stop Gate) ---")
    liver_report = """LIVER FUNCTION TEST (LFT)
Patient Name: Sowmya Reddy
Age: 41
Gender: Female
Bilirubin Total: 1.2 mg/dL
SGOT / AST: 28 U/L
SGPT / ALT: 32 U/L
Alkaline Phosphatase: 85 U/L
Total Protein: 7.2 g/dL
Albumin: 4.1 g/dL
"""
    res_lft = execute_metabolic_pipeline(liver_report, "lft.txt")
    print("LFT pipeline success:", res_lft.get("success"))
    print("LFT error message:", res_lft.get("error"))

    assert res_lft.get("success") is False, "Regression! Zero-relevant report did not trigger hard stop!"
    assert "Could not extract diabetes or obesity-relevant values" in res_lft.get("error", ""), f"Unexpected error: {res_lft.get('error')}"
    print(">>> TEST 8 PASSED: Confirmed working hard-stop gate did not regress.")
    passed_count += 1

    print("\n==================================================================")
    print(f"ALL {passed_count}/{total_count} TESTS PASSED PERFECTLY!")
    print("==================================================================")

if __name__ == "__main__":
    test_all()
