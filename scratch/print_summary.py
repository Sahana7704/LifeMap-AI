import json

with open('scratch/results_10_files.json', 'r') as f:
    data = json.load(f)

for item in data:
    fname = item['file']
    code = item['status_code']
    body = item['body']
    rec = item['recommendations']
    print(f"=== {fname} (HTTP {code}) ===")
    if code == 422:
        print("  Hard-stop gate triggered as expected!")
        print("  Error:", body.get('error'))
        print("  Risk score:", body.get('risk_score'))
        print()
        continue

    metrics = body.get('extracted_metrics', {})
    fg = metrics.get('fasting_glucose')
    hba1c = metrics.get('hba1c')
    bmi = metrics.get('bmi')
    ht = metrics.get('height_cm')
    wt = metrics.get('weight_kg')
    wc = metrics.get('waist_circumference_cm')
    age = metrics.get('age')
    gender = metrics.get('gender')
    note = metrics.get('combined_risk_note')
    audit = body.get('audit_passed')

    print(f"  Fasting Glucose: {fg}")
    print(f"  HbA1c: {hba1c}")
    print(f"  BMI: {bmi}")
    print(f"  Height: {ht}, Weight: {wt}, Waist: {wc}")
    print(f"  Age: {age}, Gender: {gender}")
    print(f"  Synthesis: {note}")
    print(f"  Audit Passed: {audit}")

    if rec:
        diet = rec.get('diet_plan', {})
        bd = diet.get('calorie_breakdown', {})
        print(f"  Nutrition Module -> Gender: {bd.get('gender')}, BMR: {bd.get('bmr')}, TDEE: {bd.get('tdee')}")
        print(f"  Caloric Rationale: {diet.get('clinical_rationale')}")
    print()
