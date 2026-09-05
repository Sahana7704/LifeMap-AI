from ml_service.recommend_engine import generate_diet_plan, generate_exercise_plan

risk_profile = {
    'diabetes_risk': 'Moderate',
    'glucose': 118,
    'cvd_risk': 'High Risk',
    'systolic_bp': 142,
    'bmi': 27.2
}

diet = generate_diet_plan(risk_profile, 1800)
ex = generate_exercise_plan(risk_profile)

print('=== DIET PLAN ===')
print('Restrictions:', diet['dietary_restrictions'])
print('Meals count:', len(diet['meals']))
for m in diet['meals']:
    print(f"- {m['slot']}: {m['dish_name']} ({m['calories']} kcal, {m['macronutrients']['fibre_g']}g fibre)")

print('\n=== EXERCISE PLAN ===')
print('Intensity:', ex['intensity'])
print('Summary:', ex['summary_note'])
for r in ex['routines']:
    print(f"- {r['title']} ({r['duration_minutes']} min, {r['frequency']})")
