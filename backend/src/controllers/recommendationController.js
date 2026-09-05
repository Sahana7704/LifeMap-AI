const crypto = require('crypto');
const db = require('../db');
const fastapiClient = require('../services/fastapiClient');

async function generateRecommendations(req, res) {
  try {
    const userId = req.user.user_id;
    const body = req.body || {};

    // Get user's latest prediction, medical report, and health profile
    const [latestPreds, latestReportRes, profileRes] = await Promise.all([
      db.query('SELECT * FROM risk_predictions WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5', [userId]),
      db.query('SELECT * FROM medical_reports WHERE user_id = $1 ORDER BY upload_date DESC LIMIT 1', [userId]),
      db.query('SELECT * FROM health_profiles WHERE user_id = $1', [userId])
    ]);

    const profile = profileRes.rows[0] || {};
    const vitals = typeof profile.vitals === 'string' ? JSON.parse(profile.vitals || '{}') : (profile.vitals || {});

    let reportMetrics = {};
    if (latestReportRes.rows.length > 0) {
      const rep = latestReportRes.rows[0];
      reportMetrics = typeof rep.extracted_metrics === 'string'
        ? JSON.parse(rep.extracted_metrics || '{}')
        : (rep.extracted_metrics || {});
    }

    const verifiedHeight = reportMetrics.height_cm || reportMetrics.verified_vitals?.height_cm || null;
    const verifiedWeight = reportMetrics.weight_kg || reportMetrics.verified_vitals?.weight_kg || null;
    const verifiedAge = reportMetrics.age || reportMetrics.verified_vitals?.age || null;
    const verifiedGender = reportMetrics.gender || reportMetrics.sex || reportMetrics.verified_vitals?.sex || reportMetrics.patient?.sex || profile.gender || 'Male';
    const verifiedBmi = reportMetrics.bmi || reportMetrics.verified_vitals?.bmi?.value || null;

    const latestPred = latestPreds.rows[0];
    const isCBC = latestPred && (
      (latestPred.disease_type && latestPred.disease_type.toLowerCase().includes('cbc')) ||
      (latestPred.disease_type && latestPred.disease_type.toLowerCase().includes('hematology'))
    );

    let diabetesRisk = 'Low Risk';
    let cvdRisk = 'Low Risk';

    latestPreds.rows.forEach(p => {
      if (p.disease_type && p.disease_type.includes('Diabetes')) diabetesRisk = p.risk_category;
      if (p.disease_type && p.disease_type.includes('Cardio')) cvdRisk = p.risk_category;
    });

    const calorieTarget = body.calorie_target ? Number(body.calorie_target) : null;
    const lifestyle = typeof profile.lifestyle_factors === 'string' ? JSON.parse(profile.lifestyle_factors || '{}') : (profile.lifestyle_factors || {});
    const dietType = body.diet_type || body.diet_preference || lifestyle.diet_preference || 'veg';

    const payload = {
      age: Number(body.age || verifiedAge || 40),
      gender: body.gender || verifiedGender || 'Male',
      calorie_target: calorieTarget,
      diet_type: dietType,
      bmi: verifiedBmi ? Number(verifiedBmi) : null,
      height_cm: verifiedHeight ? Number(verifiedHeight) : null,
      weight_kg: verifiedWeight ? Number(verifiedWeight) : null,
      waist_circumference_cm: (reportMetrics.waist_circumference_cm || reportMetrics.verified_vitals?.waist_circumference_cm) ? Number(reportMetrics.waist_circumference_cm || reportMetrics.verified_vitals?.waist_circumference_cm) : null
    };

    if (isCBC) {
      const cbcVitals = typeof latestPred.input_vitals === 'string' ? JSON.parse(latestPred.input_vitals || '{}') : (latestPred.input_vitals || {});
      payload.report_type = 'Complete Blood Count (CBC)';
      payload.hemoglobin = cbcVitals.hemoglobin || cbcVitals.hb;
      payload.pcv = cbcVitals.pcv || cbcVitals.packed_cell_volume;
      payload.platelets = cbcVitals.platelets || cbcVitals.platelet_count;
      payload.wbc = cbcVitals.wbc || cbcVitals.total_wbc;
      payload.rbc = cbcVitals.rbc || cbcVitals.total_rbc;
    } else {
      payload.diabetes_risk = body.diabetes_risk || diabetesRisk;
      payload.glucose = Number(body.glucose || reportMetrics.fasting_glucose || reportMetrics.glucose || (reportMetrics.verified_vitals?.fasting_glucose?.value) || 0) || null;
      payload.hba1c = Number(body.hba1c || reportMetrics.hba1c || (reportMetrics.verified_vitals?.hba1c?.value) || 0) || null;
      payload.cvd_risk = body.cvd_risk || cvdRisk;
      payload.systolic_bp = Number(body.systolic_bp || reportMetrics.systolic_bp || 0) || null;
      payload.cholesterol = Number(body.cholesterol || reportMetrics.cholesterol || 0) || null;
    }

    const mlRecommendations = await fastapiClient.getRecommendations(payload);
    const diet = mlRecommendations.diet_plan;
    const exercise = mlRecommendations.exercise_plan;

    const planIdDiet = diet.plan_id || `diet_${crypto.randomUUID().slice(0, 10)}`;
    const planIdExercise = exercise.plan_id || `ex_${crypto.randomUUID().slice(0, 10)}`;

    // Store in DB - save the full days array, clinical findings alignment, and calorie breakdown
    const storedMealsData = {
      days: diet.days,
      single_day_meals: diet.meals,
      diet_type: diet.diet_type || dietType,
      clinical_rationale: diet.clinical_rationale,
      findings_alignment: diet.findings_alignment,
      recommended_baseline_calories: diet.recommended_baseline_calories,
      calorie_breakdown: diet.calorie_breakdown
    };

    await db.query(
      `INSERT INTO diet_plans 
       (plan_id, user_id, calorie_target, total_estimated_calories, total_macronutrients, dietary_restrictions, meals, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [
        planIdDiet,
        userId,
        diet.calorie_target,
        diet.total_estimated_calories,
        JSON.stringify(diet.total_macronutrients),
        JSON.stringify(diet.dietary_restrictions),
        JSON.stringify(storedMealsData),
        new Date().toISOString()
      ]
    );

    await db.query(
      `INSERT INTO exercise_plans 
       (plan_id, user_id, intensity, summary_note, weekly_target_minutes, routines, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)`,
      [
        planIdExercise,
        userId,
        exercise.intensity,
        exercise.summary_note,
        exercise.weekly_target_minutes,
        JSON.stringify(exercise.routines),
        new Date().toISOString()
      ]
    );

    return res.status(200).json({
      message: 'Personalized 7-day Indian nutrition & exercise recommendations generated',
      diet_plan: diet,
      exercise_plan: exercise
    });
  } catch (err) {
    console.error('Recommendation generation error:', err);
    return res.status(500).json({ error: 'Failed to generate recommendations', details: err.message });
  }
}

async function getCurrentPlan(req, res) {
  try {
    const userId = req.user.user_id;
    const dietRes = await db.query(
      'SELECT * FROM diet_plans WHERE user_id = $1 ORDER BY created_at DESC LIMIT 1',
      [userId]
    );
    const exRes = await db.query(
      'SELECT * FROM exercise_plans WHERE user_id = $1 ORDER BY created_at DESC LIMIT 1',
      [userId]
    );

    let diet = dietRes.rows[0] || null;
    let exercise = exRes.rows[0] || null;

    if (diet) {
      const parsedMeals = typeof diet.meals === 'string' ? JSON.parse(diet.meals || '[]') : diet.meals;
      const parsedMacros = typeof diet.total_macronutrients === 'string' ? JSON.parse(diet.total_macronutrients || '{}') : diet.total_macronutrients;
      const parsedRestrictions = typeof diet.dietary_restrictions === 'string' ? JSON.parse(diet.dietary_restrictions || '[]') : diet.dietary_restrictions;

      if (parsedMeals && parsedMeals.days) {
        diet = {
          ...diet,
          diet_type: parsedMeals.diet_type || diet.diet_type || 'veg',
          days: parsedMeals.days,
          meals: parsedMeals.single_day_meals || (parsedMeals.days[0]?.meals || []),
          total_macronutrients: parsedMacros,
          dietary_restrictions: parsedRestrictions,
          clinical_rationale: parsedMeals.clinical_rationale,
          findings_alignment: parsedMeals.findings_alignment,
          recommended_baseline_calories: parsedMeals.recommended_baseline_calories,
          calorie_breakdown: parsedMeals.calorie_breakdown
        };
      } else {
        diet = {
          ...diet,
          diet_type: parsedMeals?.diet_type || diet.diet_type || 'veg',
          meals: Array.isArray(parsedMeals) ? parsedMeals : [],
          total_macronutrients: parsedMacros,
          dietary_restrictions: parsedRestrictions,
          clinical_rationale: parsedMeals?.clinical_rationale,
          findings_alignment: parsedMeals?.findings_alignment,
          recommended_baseline_calories: parsedMeals?.recommended_baseline_calories,
          calorie_breakdown: parsedMeals?.calorie_breakdown
        };
      }
    }

    if (exercise) {
      exercise = {
        ...exercise,
        routines: typeof exercise.routines === 'string' ? JSON.parse(exercise.routines || '[]') : exercise.routines
      };
    }

    return res.json({
      diet_plan: diet,
      exercise_plan: exercise
    });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to retrieve active plans', details: err.message });
  }
}

module.exports = {
  generateRecommendations,
  getCurrentPlan
};
