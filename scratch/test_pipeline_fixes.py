import sys
import os

# Add ml_service to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_service")))

from metabolic_pipeline import (
    step1_metabolic_extraction,
    step2_metabolic_pre_generation_gate,
    step3_metabolic_risk_scoring,
    step4_metabolic_post_generation_audit,
    execute_metabolic_pipeline
)

print("--- TEST 1: Bug L (Sex extraction on inline all-caps 'AGE: 60 SEX: M') ---")
test_text_1 = """
DR. LAL PATHLABS
PATIENT REPORT
AGE: 60 SEX: M
FASTING BLOOD GLUCOSE: 95 mg/dL
WAIST CIRCUMFERENCE: 85 cm
HEIGHT: 170 cm
WEIGHT: 70 kg
"""
res1 = execute_metabolic_pipeline(test_text_1, "test_bug_l.txt")
assert res1["verified_vitals"]["sex"] == "Male", f"Expected Male, got {res1['verified_vitals']['sex']}"
# Male with waist 85 cm should NOT have central obesity (cutoff 90 cm)
assert not res1["obesity_assessment"].get("central_obesity", False), "Male waist 85 cm should NOT be flagged as central obesity!"
print("Test 1 PASS: Sex correctly extracted as Male, waist cutoff 90 applied (85cm is normal).")

print("\n--- TEST 2: Bug L test with Female ---")
test_text_2 = """
DR. LAL PATHLABS
PATIENT REPORT
AGE: 60 SEX: F
FASTING BLOOD GLUCOSE: 95 mg/dL
WAIST CIRCUMFERENCE: 85 cm
"""
res2 = execute_metabolic_pipeline(test_text_2, "test_bug_l_female.txt")
assert res2["verified_vitals"]["sex"] == "Female", f"Expected Female, got {res2['verified_vitals']['sex']}"
# Female with waist 85 cm SHOULD have central obesity (cutoff 80 cm)
assert res2["obesity_assessment"].get("central_obesity", True), "Female waist 85 cm SHOULD be flagged as central obesity!"
print("Test 2 PASS: Sex correctly extracted as Female, waist cutoff 80 applied (85cm is flagged).")

print("\n--- TEST 3: Bug N (Underweight + Normal Glucose) ---")
test_text_3 = """
CITY CLINICAL LABORATORY
Patient: Test Patient
Age: 25   Sex: Female
Fasting Glucose: 88 mg/dL
* Reference range 70-99 mg/dL
Height: 165 cm
Weight: 45 kg
BMI: 16.5 kg/m2
"""
res3 = execute_metabolic_pipeline(test_text_3, "test_bug_n.txt")
# Verify Fasting glucose is NOT source_flagged
fg_class = [c for c in res3["diabetes_assessment"]["classifications"] if "Fasting" in c["marker"]][0]
assert not fg_class["source_flagged"], "Normal glucose (88 mg/dL) must NOT have source_flagged=True!"
# Verify synthesis note mentions Underweight and nutritional support, NOT diabetes reduction
comb_note = res3["combined_risk_note"]
assert "underweight" in comb_note.lower(), f"Combined note missing 'underweight': {comb_note}"
assert "nutritional support" in comb_note.lower(), f"Combined note missing 'nutritional support': {comb_note}"
assert "borderline or moderate metabolic risk" not in comb_note.lower(), f"Combined note wrongly mentions moderate risk: {comb_note}"
print("Test 3 PASS: Normal glucose has source_flagged=False. Underweight synthesis note has correct nutritional support phrasing.")

print("\n--- TEST 4: Bug O (HbA1c field detected but unreadable value) ---")
test_text_4 = """
DIAGNOSTIC PATH LAB
Patient Name: John Doe
Age: 50  Sex: Male
Fasting Blood Sugar: 92 mg/dL
HbA1c (Glycated Hemoglobin): ??? %
"""
res4 = execute_metabolic_pipeline(test_text_4, "test_bug_o.txt")
assert res4.get("hba1c_ocr_note") is not None, f"Expected hba1c_ocr_note, got {res4.get('hba1c_ocr_note')}"
print(f"Test 4 PASS: Detected unreadable HbA1c note: '{res4.get('hba1c_ocr_note')}'")

print("\nALL PIPELINE UNIT TESTS PASSED SUCCESSFULLY!")
