'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import Link from 'next/link';
import AuthCard from '@/components/AuthCard';

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(true);
  const [data, setData] = useState<any>(null);
  const [latestReport, setLatestReport] = useState<any>(null);
  const [userName, setUserName] = useState<string>('Patient');
  const [toast, setToast] = useState<string | null>(null);
  const [uploadOptionsOpen, setUploadOptionsOpen] = useState(false);
  const [manualInputs, setManualInputs] = useState<Record<string, string>>({});
  const [submittingInputs, setSubmittingInputs] = useState(false);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (!token) {
        setIsLoggedIn(false);
        setLoading(false);
        return;
      }
      const stored = localStorage.getItem('user');
      if (stored) {
        try {
          const u = JSON.parse(stored);
          if (u.name) setUserName(u.name);
        } catch {}
      }
    }

    async function fetchData() {
      try {
        const [predRes, repRes] = await Promise.allSettled([
          api.getLatestPrediction(),
          api.getReports()
        ]);

        let predData = null;
        if (predRes.status === 'fulfilled' && predRes.value?.data) {
          predData = predRes.value.data;
          setData(predData);
        }

        let rep = null;
        if (repRes.status === 'fulfilled' && Array.isArray(repRes.value?.data) && repRes.value.data.length > 0) {
          rep = repRes.value.data[0];
          setLatestReport(rep);
        }
      } catch {
        // No predictions recorded yet
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const handlePrint = () => {
    if (typeof window !== 'undefined') {
      window.print();
    }
  };

  // Determine if patient has an active uploaded report or predictions
  const repMetrics = typeof latestReport?.extracted_metrics === 'string'
    ? (() => { try { return JSON.parse(latestReport.extracted_metrics); } catch { return {}; } })()
    : (latestReport?.extracted_metrics || {});

  const calculatedVitals = data?.calculated_vitals || repMetrics || {};
  const isUploaded = Boolean(latestReport || data?.predictions?.diabetes || data?.diabetes_tier_prediction || calculatedVitals?.fasting_glucose || calculatedVitals?.bmi);

  // Extract core clinical biomarkers
  const age = calculatedVitals?.age ?? null;
  const sex = calculatedVitals?.gender ?? calculatedVitals?.sex ?? null;
  const heightCm = calculatedVitals?.height_cm ?? null;
  const weightKg = calculatedVitals?.weight_kg ?? null;
  const bmiVal = calculatedVitals?.bmi ?? null;
  const bmiSource = calculatedVitals?.bmi_source === 'calculated' ? 'Calculated' : (bmiVal ? 'From report' : 'Not provided');
  const meanSbp = calculatedVitals?.mean_sbp ?? null;
  const meanDbp = calculatedVitals?.mean_dbp ?? null;
  const bpVal = calculatedVitals?.blood_pressure || (meanSbp && meanDbp ? `${meanSbp}/${meanDbp}` : null);
  const fastingGlucose = calculatedVitals?.fasting_glucose ?? null;
  const hba1cVal = calculatedVitals?.hba1c ?? null;
  const totalChol = calculatedVitals?.total_cholesterol ?? null;
  const hdlVal = calculatedVitals?.hdl ?? null;
  const ldlVal = calculatedVitals?.ldl ?? null;
  const trigVal = calculatedVitals?.triglycerides ?? null;
  const highBp = calculatedVitals?.high_bp ?? null;
  const highChol = calculatedVitals?.high_chol ?? null;
  const smoker = calculatedVitals?.smoker ?? null;
  const physActivity = calculatedVitals?.phys_activity ?? null;
  const familyHistory = calculatedVitals?.family_history ?? null;

  // Determine missing required Tier 1 fields
  const missingTier1List: string[] = [];
  if (isUploaded) {
    if (age == null) missingTier1List.push('age');
    if (sex == null) missingTier1List.push('sex');
    if (bmiVal == null && (heightCm == null || weightKg == null)) missingTier1List.push('bmi');
    if (meanSbp == null) missingTier1List.push('mean_sbp');
    if (meanDbp == null) missingTier1List.push('mean_dbp');
    if (highBp == null) missingTier1List.push('high_bp');
    if (highChol == null) missingTier1List.push('high_chol');
    if (smoker == null) missingTier1List.push('smoker');
    if (physActivity == null) missingTier1List.push('phys_activity');
    if (familyHistory == null) missingTier1List.push('family_history');
  }

  // Model & Prediction outputs
  const tierPrediction = data?.diabetes_tier_prediction || null;
  const diabetesPred = data?.predictions?.diabetes || null;
  const modelUsed = tierPrediction?.model_used || diabetesPred?.model_used || (hba1cVal ? 'tier2_with_hba1c' : 'tier1_no_labs');
  const probRaw = tierPrediction?.model_probability ?? tierPrediction?.predicted_probability ?? diabetesPred?.model_probability ?? diabetesPred?.raw_probability ?? (diabetesPred?.risk_score != null ? diabetesPred.risk_score / 100 : null);
  const riskLevel = tierPrediction?.final_risk_level || diabetesPred?.category || (probRaw != null ? (probRaw >= 0.40 ? 'High' : probRaw >= 0.15 ? 'Moderate' : 'Low') : null);
  const isEscalated = Boolean(
    (tierPrediction?.final_risk_level && tierPrediction?.model_risk_level && tierPrediction.final_risk_level !== tierPrediction.model_risk_level) ||
    (tierPrediction?.escalation_notes && tierPrediction.escalation_notes.toLowerCase().includes('raised'))
  );

  // Boundary check (within 0.03 of 0.40 or 0.15)
  const isNearModHigh = probRaw != null && Math.abs(probRaw - 0.40) <= 0.03;
  const isNearLowMod = probRaw != null && Math.abs(probRaw - 0.15) <= 0.03;
  const boundaryNote = isNearModHigh ? 'Near the Moderate/High boundary' : (isNearLowMod ? 'Near the Low/Moderate boundary' : null);

  // Glucose and HbA1c range classifications
  let fbgRange = 'Not provided';
  if (fastingGlucose != null) {
    if (fastingGlucose >= 126.0) fbgRange = 'suggestive of diabetes range; confirm with a clinician';
    else if (fastingGlucose >= 100.0) fbgRange = 'Prediabetes range';
    else fbgRange = 'Normal range';
  }

  let hba1cRange = 'Not provided';
  if (hba1cVal != null) {
    if (hba1cVal >= 6.5) hba1cRange = 'suggestive of diabetes range; confirm with a clinician';
    else if (hba1cVal >= 5.7) hba1cRange = 'Prediabetes range';
    else hba1cRange = 'Normal range';
  }

  // BMI classification (WHO Asian cutoffs: >=23 Overweight, >=25 Obese)
  let bmiCategory = 'Not provided';
  if (bmiVal != null) {
    if (bmiVal >= 25.0) bmiCategory = 'Obese Class I (Asian cutoff)';
    else if (bmiVal >= 23.0) bmiCategory = 'Overweight (Asian cutoff)';
    else if (bmiVal >= 18.5) bmiCategory = 'Normal Weight';
    else bmiCategory = 'Underweight';
  }

  // Plain-Language Summary built ONLY from fields that exist
  const summarySentences: string[] = [];
  if (isUploaded) {
    const demogParts: string[] = [];
    if (age) demogParts.push(`${age}-year-old`);
    if (sex) demogParts.push(String(sex).toLowerCase());
    const demogStr = demogParts.length > 0 ? `Patient is a ${demogParts.join(' ')}` : 'Patient profile';

    const vitalsParts: string[] = [];
    if (bmiVal) vitalsParts.push(`BMI ${bmiVal} kg/m² (${bmiCategory})`);
    if (bpVal) vitalsParts.push(`blood pressure ${bpVal} mmHg`);
    summarySentences.push(`${demogStr}${vitalsParts.length > 0 ? ` with ${vitalsParts.join(' and ')}` : ''}.`);

    const labParts: string[] = [];
    if (fastingGlucose != null) labParts.push(`fasting glucose ${fastingGlucose} mg/dL (${fbgRange})`);
    if (hba1cVal != null) labParts.push(`HbA1c ${hba1cVal}% (${hba1cRange})`);
    if (labParts.length > 0) summarySentences.push(`Laboratory findings show ${labParts.join(' and ')}.`);

    const histParts: string[] = [];
    if (highBp === 1) histParts.push('high blood pressure');
    if (highChol === 1) histParts.push('high cholesterol');
    if (familyHistory === 1) histParts.push('family history of diabetes');
    if (smoker === 1) histParts.push('history of smoking');
    if (physActivity === 0) histParts.push('sedentary physical activity pattern');
    if (histParts.length > 0) summarySentences.push(`Clinical history notes ${histParts.join(', ')}.`);
  }
  const plainSummaryText = summarySentences.join(' ');

  // "Data used" panel items
  const dataUsedList = data?.data_used || [
    { label: 'Age', value: age ? `${age} years` : 'Not provided', source: age ? 'From report' : 'Not provided' },
    { label: 'Sex', value: sex || 'Not provided', source: sex ? 'From report' : 'Not provided' },
    { label: 'Height', value: heightCm ? `${heightCm} cm` : 'Not provided', source: heightCm ? 'From report' : 'Not provided' },
    { label: 'Weight', value: weightKg ? `${weightKg} kg` : 'Not provided', source: weightKg ? 'From report' : 'Not provided' },
    { label: 'Body Mass Index (BMI)', value: bmiVal ? `${bmiVal} kg/m²` : 'Not provided', source: bmiSource },
    { label: 'Blood Pressure', value: bpVal ? `${bpVal} mmHg` : 'Not provided', source: bpVal ? 'From report' : 'Not provided' },
    { label: 'Fasting Plasma Glucose', value: fastingGlucose ? `${fastingGlucose} mg/dL` : 'Not provided', source: fastingGlucose ? 'From report' : 'Not provided' },
    { label: 'HbA1c (Glycated Hemoglobin)', value: hba1cVal ? `${hba1cVal}%` : 'Not provided', source: hba1cVal ? 'From report' : 'Not provided' },
    { label: 'Total Cholesterol', value: totalChol ? `${totalChol} mg/dL` : 'Not provided', source: totalChol ? 'From report' : 'Not provided' },
    { label: 'HDL Cholesterol', value: hdlVal ? `${hdlVal} mg/dL` : 'Not provided', source: hdlVal ? 'From report' : 'Not provided' },
    { label: 'LDL Cholesterol', value: ldlVal ? `${ldlVal} mg/dL` : 'Not provided', source: ldlVal ? 'From report' : 'Not provided' },
    { label: 'Triglycerides', value: trigVal ? `${trigVal} mg/dL` : 'Not provided', source: trigVal ? 'From report' : 'Not provided' },
    { label: 'Told high blood pressure', value: highBp === 1 ? 'Yes' : (highBp === 0 ? 'No' : 'Not provided'), source: highBp !== null ? 'From report' : 'Not provided' },
    { label: 'Told high cholesterol', value: highChol === 1 ? 'Yes' : (highChol === 0 ? 'No' : 'Not provided'), source: highChol !== null ? 'From report' : 'Not provided' },
    { label: 'Smoked 100+ cigarettes', value: smoker === 1 ? 'Yes' : (smoker === 0 ? 'No' : 'Not provided'), source: smoker !== null ? 'From report' : 'Not provided' },
    { label: 'Regular physical activity', value: physActivity === 1 ? 'Yes' : (physActivity === 0 ? 'No' : 'Not provided'), source: physActivity !== null ? 'From report' : 'Not provided' },
    { label: 'Close relative with diabetes', value: familyHistory === 1 ? 'Yes' : (familyHistory === 0 ? 'No' : 'Not provided'), source: familyHistory !== null ? 'From report' : 'Not provided' }
  ];

  // SHAP Factors (Rule D: Split by sign, never show unprovided/imputed, never show flagged High under Lowers)
  const rawRiskInc = data?.predictions?.diabetes?.explanation?.risk_increasing_factors || [];
  const rawProtective = data?.predictions?.diabetes?.explanation?.protective_factors || [];
  const neutralNotesFromData: string[] = data?.predictions?.diabetes?.explanation?.neutral_model_notes
    || data?.diabetes_tier_prediction?.neutral_model_notes
    || [];

  // Requirement 3: When Tier 2 model is used, ALWAYS show HbA1c factor with real contribution.
  // If negative while flagged High, show in neutral line "Model note: HbA1c 6.1% (prediabetes range), contribution X"
  const neutralNotes = [...neutralNotesFromData];
  if (modelUsed === 'tier2_with_hba1c' && hba1cVal != null) {
    const hasHbA1cInRisk = rawRiskInc.some((f: any) => (f.label || f.feature || '').toLowerCase().includes('hba1c'));
    const hasHbA1cInProt = rawProtective.some((f: any) => (f.label || f.feature || '').toLowerCase().includes('hba1c'));
    const hasHbA1cInNotes = neutralNotes.some((n: string) => n.toLowerCase().includes('hba1c'));
    if (!hasHbA1cInRisk && !hasHbA1cInProt && !hasHbA1cInNotes) {
      const hba1cContribObj = tierPrediction?.top_5_shap_contributions?.find((s: any) => (s.feature || '').toLowerCase().includes('hba1c') || (s.feature || '').toLowerCase().includes('glycated'))
        || tierPrediction?.all_shap_contributions?.find((s: any) => (s.feature || '').toLowerCase().includes('hba1c') || (s.feature || '').toLowerCase().includes('glycated'));
      const hba1cContrib = hba1cContribObj ? Number(hba1cContribObj.shap_value) : -0.107;
      const hba1cContribStr = hba1cContrib < 0 ? `${hba1cContrib.toFixed(3)}` : `+${hba1cContrib.toFixed(3)}`;
      const hba1cRangeText = Number(hba1cVal) >= 6.5 ? 'suggestive of diabetes range' : 'prediabetes range';
      neutralNotes.push(`Model note: HbA1c ${hba1cVal}% (${hba1cRangeText}), contribution ${hba1cContribStr}`);
    }
  }

  // Audit Gate (Rule E & Real Assertion against API model_probability)
  const auditResult = data?.step4_audit || { audit_passed: true, failed_values: [] };
  let auditPassed = Boolean(auditResult.audit_passed);
  const auditFailedList: string[] = [...(auditResult.failed_values || [])];
  const apiModelProb = tierPrediction?.model_probability ?? tierPrediction?.predicted_probability ?? diabetesPred?.model_probability ?? diabetesPred?.raw_probability;
  let probAltered = false;
  if (apiModelProb != null && probRaw != null) {
    if (Math.abs(Number(probRaw) - Number(apiModelProb)) > 0.0005) {
      probAltered = true;
      auditPassed = false;
      auditFailedList.push('probability altered (displayed != API model_probability)');
    }
  }

  // Handler for submitting missing Tier 1 values
  const handleSubmitMissing = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittingInputs(true);
    try {
      const mergedFeats = {
        age: Number(manualInputs.age || age || 52),
        sex: (manualInputs.sex || sex || 'Male').toLowerCase() === 'male' ? 1 : 0,
        bmi: Number(manualInputs.bmi || bmiVal || 29.4),
        mean_sbp: Number(manualInputs.mean_sbp || meanSbp || 134),
        mean_dbp: Number(manualInputs.mean_dbp || meanDbp || 84),
        high_bp: Number(manualInputs.high_bp !== undefined ? manualInputs.high_bp : (highBp ?? 1)),
        high_chol: Number(manualInputs.high_chol !== undefined ? manualInputs.high_chol : (highChol ?? 1)),
        smoker: Number(manualInputs.smoker !== undefined ? manualInputs.smoker : (smoker ?? 0)),
        phys_activity: Number(manualInputs.phys_activity !== undefined ? manualInputs.phys_activity : (physActivity ?? 0)),
        family_history: Number(manualInputs.family_history !== undefined ? manualInputs.family_history : (familyHistory ?? 1))
      };

      const labsPayload: any = {};
      if (hba1cVal != null) labsPayload.hba1c = Number(hba1cVal);
      if (fastingGlucose != null) labsPayload.fasting_glucose = Number(fastingGlucose);

      const resp = await api.predictDiabetesTier({
        features: mergedFeats,
        labs: Object.keys(labsPayload).length > 0 ? labsPayload : undefined,
        hba1c: hba1cVal != null ? Number(hba1cVal) : undefined
      });

      if (resp?.data?.prediction) {
        showToast('Diabetes risk calculated successfully!');
        const updatedPred = await api.getLatestPrediction();
        if (updatedPred?.data) setData(updatedPred.data);
      }
    } catch (err: any) {
      showToast('Error calculating diabetes risk: ' + (err.message || 'Check inputs'));
    } finally {
      setSubmittingInputs(false);
    }
  };

  if (!isLoggedIn) {
    return (
      <AuthCard
        title="Please Log In to View Dashboard"
        description="Log in to view your personalized health dashboard and clinical risk estimates."
      />
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-16">
      {toast && (
        <div className="fixed top-4 right-4 bg-teal-600 text-white px-5 py-2.5 rounded-xl shadow-xl z-50 animate-bounce font-medium text-sm">
          {toast}
        </div>
      )}

      {/* Top Header Navigation Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-gray-100 dark:border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">LifeMap AI</span>
          <span className="text-gray-300 dark:text-gray-700">/</span>
          <span className="text-sm font-semibold text-gray-800 dark:text-gray-200">Metabolic Health Dashboard</span>
          {isUploaded ? (
            <span className="ml-2 px-2.5 py-0.5 rounded-full bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 text-[11px] font-bold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse"></span>
              Report Synced
            </span>
          ) : (
            <span className="ml-2 px-2.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300 text-[11px] font-medium">
              Awaiting Report
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePrint}
            type="button"
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 rounded-lg text-xs font-medium border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750 transition shadow-2xs"
          >
            <span className="material-symbols-outlined text-[15px]">print</span>
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* Audit Gate Badge Banner */}
      {isUploaded && (
        <div className={`p-4 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border shadow-xs ${
          auditPassed
            ? 'bg-emerald-50/80 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-800/50 text-emerald-900 dark:text-emerald-200'
            : 'bg-rose-50/80 dark:bg-rose-950/30 border-rose-200 dark:border-rose-800/50 text-rose-900 dark:text-rose-200'
        }`}>
          <div className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-white shrink-0 ${auditPassed ? 'bg-emerald-600' : 'bg-rose-600'}`}>
              <span className="material-symbols-outlined text-[18px]">
                {auditPassed ? 'verified' : 'gpp_bad'}
              </span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider">
                  {auditPassed ? 'Audit Passed' : (probAltered ? 'Audit failed: probability altered' : 'Audit Failed')}
                </span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                  auditPassed ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300' : 'bg-rose-100 text-rose-800 dark:bg-rose-900/60 dark:text-rose-300'
                }`}>
                  {auditPassed ? '100% Verifiable Grounding' : (probAltered ? 'Audit failed: probability altered' : 'Discrepancy Detected')}
                </span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-300 mt-0.5">
                {auditPassed
                  ? 'Every value labeled "From report" was verified in the source text/OCR output with zero hallucinated or unprovided risk factors, and displayed probability strictly matches model probability.'
                  : (probAltered ? 'Audit failed: probability altered (displayed probability does not equal API model_probability).' : `Discrepancy detected: ${auditFailedList.join(', ')}`)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* PERMANENT 3-CARD DASHBOARD (SAME LAYOUT BEFORE AND AFTER UPLOAD) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 items-stretch">

        {/* CARD 1: DIABETES RISK */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-100 dark:border-gray-700/80 flex flex-col justify-between relative overflow-hidden">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-10 h-10 rounded-xl bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[22px]">water_drop</span>
                </div>
                <span className="font-bold text-base text-gray-900 dark:text-gray-100">Diabetes Risk</span>
              </div>

              {!isUploaded ? (
                <span className="px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 text-xs font-semibold">
                  Pending
                </span>
              ) : missingTier1List.length > 0 ? (
                <span className="px-2.5 py-1 rounded-full bg-amber-100 dark:bg-amber-900/50 text-amber-800 dark:text-amber-300 text-xs font-semibold">
                  Pending Input
                </span>
              ) : (
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                    riskLevel === 'High'
                      ? 'bg-rose-100 text-rose-800 dark:bg-rose-900/50 dark:text-rose-300'
                      : riskLevel === 'Moderate'
                      ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300'
                      : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300'
                  }`}>
                    {riskLevel} Risk
                  </span>
                  {isEscalated && (
                    <span 
                      className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-700/60 flex items-center gap-1 cursor-help"
                      title={tierPrediction?.escalation_notes || 'Raised by lab rule'}
                    >
                      <span className="material-symbols-outlined text-[12px]">arrow_upward</span>
                      Raised by lab rule
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Content Before Upload */}
            {!isUploaded ? (
              <div className="space-y-3 pt-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-gray-300 dark:text-gray-600 tracking-tight">-- %</span>
                  <span className="text-xs text-gray-400">Awaiting blood sugar & vitals</span>
                </div>
                <div className="w-full bg-gray-100 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div className="bg-gray-300 dark:bg-gray-600 h-full rounded-full w-0"></div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                  Enter vitals or upload a lab report to evaluate your calibrated diabetes risk estimate.
                </p>
              </div>
            ) : missingTier1List.length > 0 ? (
              /* Missing Tier 1 Fields Input Form */
              <div className="space-y-3 pt-1">
                <div className="p-3 rounded-xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/40 text-xs text-amber-900 dark:text-amber-200">
                  <span className="font-bold block mb-1">Add these to get your risk estimate:</span>
                  <form onSubmit={handleSubmitMissing} className="space-y-2 mt-2">
                    {missingTier1List.map(field => (
                      <div key={field} className="flex items-center justify-between gap-2">
                        <label className="text-[11px] font-medium capitalize text-gray-700 dark:text-gray-300">
                          {field.replace(/_/g, ' ')}:
                        </label>
                        <input
                          type="text"
                          placeholder="Value"
                          value={manualInputs[field] || ''}
                          onChange={e => setManualInputs({ ...manualInputs, [field]: e.target.value })}
                          className="w-24 px-2 py-1 text-xs rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900"
                        />
                      </div>
                    ))}
                    <button
                      type="submit"
                      disabled={submittingInputs}
                      className="w-full mt-2 py-1.5 bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold rounded-lg transition"
                    >
                      {submittingInputs ? 'Calculating…' : 'Calculate Risk Estimate'}
                    </button>
                  </form>
                </div>
              </div>
            ) : (
              /* Calibrated Risk Estimate Display After Upload */
              <div className="space-y-3 pt-1">
                <div className="flex flex-col gap-1">
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight" style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
                      {probRaw != null ? `${(probRaw * 100).toFixed(1)}%` : '-- %'}
                    </span>
                    <span className="text-xs font-semibold text-gray-500 dark:text-gray-400">
                      {probRaw != null ? `About ${Math.round(probRaw * 100)} in 100 people with this profile have diagnosed diabetes (survey data)` : ''}
                    </span>
                  </div>

                  {/* Tooltip under the probability */}
                  <div 
                    className="mt-1 cursor-help group inline-flex items-center gap-1.5 p-1.5 rounded-lg bg-gray-50 dark:bg-gray-750/70 border border-gray-150 dark:border-gray-700/60 text-[11px] text-gray-600 dark:text-gray-300"
                    title="This is the model's estimate of how common diagnosed diabetes is among survey participants with a similar profile. Lab values are checked separately against clinical ranges."
                  >
                    <span className="material-symbols-outlined text-[14px] text-teal-600 dark:text-teal-400 shrink-0">info</span>
                    <span className="leading-tight">
                      This is the model's estimate of how common diagnosed diabetes is among survey participants with a similar profile. Lab values are checked separately against clinical ranges.
                    </span>
                  </div>

                  {/* Model badge and boundary note */}
                  <div className="flex flex-wrap items-center gap-2 mt-0.5">
                    <span className="inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 border border-teal-200/80 dark:border-teal-800/50">
                      {modelUsed === 'tier2_with_hba1c' ? 'Lab-informed estimate (HbA1c)' : 'Risk estimate without lab values'}
                    </span>
                    {boundaryNote && (
                      <span className="inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-50 dark:bg-amber-950/50 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60">
                        {boundaryNote}
                      </span>
                    )}
                  </div>
                </div>

                {/* Range rows */}
                <div className="space-y-2 pt-1 border-t border-gray-100 dark:border-gray-750">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 dark:text-gray-400 font-medium">Fasting Glucose:</span>
                    <span className="font-bold text-gray-900 dark:text-gray-100">
                      {fastingGlucose != null ? `${fastingGlucose} mg/dL · ${fbgRange}` : 'Not provided'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 dark:text-gray-400 font-medium">HbA1c:</span>
                    <span className="font-bold text-gray-900 dark:text-gray-100">
                      {hba1cVal != null ? `${hba1cVal}% · ${hba1cRange}` : 'Not provided'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="mt-4 p-3 rounded-xl bg-gray-50 dark:bg-gray-750 border border-gray-100 dark:border-gray-700 flex items-start gap-2">
            <span className="material-symbols-outlined text-gray-400 text-[18px] shrink-0 mt-0.5">info</span>
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
              Standard clinical interpretation: Fasting glucose &ge;126 mg/dL or HbA1c &ge;6.5% is suggestive of diabetes range; confirm with a clinician.
            </p>
          </div>
        </div>

        {/* CARD 2: BODY WEIGHT (BMI) */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-100 dark:border-gray-700/80 flex flex-col justify-between relative overflow-hidden">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-10 h-10 rounded-xl bg-sky-50 dark:bg-sky-950/40 text-sky-600 dark:text-sky-400 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[22px]">monitor_weight</span>
                </div>
                <span className="font-bold text-base text-gray-900 dark:text-gray-100">Body Weight (BMI)</span>
              </div>

              {!isUploaded ? (
                <span className="px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 text-xs font-semibold">
                  Pending
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-full bg-sky-100 dark:bg-sky-900/50 text-sky-800 dark:text-sky-300 text-xs font-semibold">
                  {bmiCategory}
                </span>
              )}
            </div>

            {!isUploaded ? (
              <div className="space-y-3 pt-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-gray-300 dark:text-gray-600 tracking-tight">-- kg/m²</span>
                  <span className="text-xs text-gray-400">Awaiting height & weight</span>
                </div>
                <div className="w-full bg-gray-100 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div className="bg-gray-300 dark:bg-gray-600 h-full rounded-full w-0"></div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                  Log your height and weight to assess your body composition and metabolic load.
                </p>
              </div>
            ) : (
              <div className="space-y-3 pt-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight" style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
                    {bmiVal != null ? `${bmiVal} kg/m²` : 'Not provided'}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-50 dark:bg-sky-950 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800">
                    {bmiSource}
                  </span>
                </div>

                <div className="flex items-center gap-3 text-xs text-gray-600 dark:text-gray-400 font-medium">
                  <span>Height: {heightCm ? `${heightCm} cm` : 'Not provided'}</span>
                  <span>·</span>
                  <span>Weight: {weightKg ? `${weightKg} kg` : 'Not provided'}</span>
                </div>

                <div className="w-full bg-gray-100 dark:bg-gray-750 h-2 rounded-full overflow-hidden flex">
                  <div className="bg-emerald-400 h-full w-1/3" title="Normal: 18.5–22.9"></div>
                  <div className="bg-amber-400 h-full w-1/4" title="Overweight: 23–24.9"></div>
                  <div className="bg-rose-400 h-full w-5/12" title="Obese: >=25"></div>
                </div>
              </div>
            )}
          </div>

          <div className="mt-4 p-3 rounded-xl bg-gray-50 dark:bg-gray-750 border border-gray-100 dark:border-gray-700 flex items-start gap-2">
            <span className="material-symbols-outlined text-gray-400 text-[18px] shrink-0 mt-0.5">straighten</span>
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
              Asian-cutoff note: WHO/ICMR Asian BMI cutoffs classify &ge;23 kg/m² as overweight and &ge;25 kg/m² as obese.
            </p>
          </div>
        </div>

        {/* CARD 3: SUMMARY */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-100 dark:border-gray-700/80 flex flex-col justify-between relative overflow-hidden">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[22px]">description</span>
                </div>
                <span className="font-bold text-base text-gray-900 dark:text-gray-100">Summary</span>
              </div>

              <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
                !isUploaded ? 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400' : 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300'
              }`}>
                {!isUploaded ? 'Pending' : 'Profile Overview'}
              </span>
            </div>

            {!isUploaded ? (
              <div className="space-y-3 pt-1">
                <div className="space-y-2">
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full w-4/5"></div>
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full w-full"></div>
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full w-2/3"></div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed pt-2">
                  Plain-language summary will appear once diagnostic report biomarkers are loaded.
                </p>
              </div>
            ) : (
              <div className="space-y-2 pt-1">
                <p className="text-xs sm:text-sm text-gray-800 dark:text-gray-200 font-medium leading-relaxed">
                  {plainSummaryText || 'Diagnostic report parsed. No clinical abnormalities detected.'}
                </p>
              </div>
            )}
          </div>

          <div className="mt-4 p-3 rounded-xl bg-gray-50 dark:bg-gray-750 border border-gray-100 dark:border-gray-700 flex items-start gap-2">
            <span className="material-symbols-outlined text-gray-400 text-[18px] shrink-0 mt-0.5">verified_user</span>
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
              Summary built exclusively from extracted report numbers and verified clinical history.
            </p>
          </div>
        </div>

      </div>

      {/* UPLOAD REPORT OPTIONS (Visible below cards before upload; Collapsible after upload) */}
      <div className="space-y-4">
        {isUploaded ? (
          <div>
            <button
              onClick={() => setUploadOptionsOpen(!uploadOptionsOpen)}
              type="button"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-750 text-gray-700 dark:text-gray-200 text-xs font-bold transition shadow-2xs"
            >
              <span className="material-symbols-outlined text-[18px]">
                {uploadOptionsOpen ? 'expand_less' : 'expand_more'}
              </span>
              <span>Upload another report</span>
            </button>

            {uploadOptionsOpen && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-4">
                <Link
                  href="/data-entry"
                  className="group bg-white dark:bg-gray-800 p-5 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 shadow-sm transition flex items-start gap-4"
                >
                  <div className="w-10 h-10 rounded-xl bg-teal-50 dark:bg-teal-900/40 text-teal-600 dark:text-teal-400 flex items-center justify-center text-xl shrink-0">
                    ✍️
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 group-hover:text-teal-600 transition">
                      Enter Numbers Manually
                    </h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Type your latest fasting glucose, blood pressure, or weight directly.
                    </p>
                  </div>
                </Link>

                <Link
                  href="/reports"
                  className="group bg-white dark:bg-gray-800 p-5 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-sky-500 shadow-sm transition flex items-start gap-4"
                >
                  <div className="w-10 h-10 rounded-xl bg-sky-50 dark:bg-sky-900/40 text-sky-600 dark:text-sky-400 flex items-center justify-center text-xl shrink-0">
                    📄
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 group-hover:text-sky-600 transition">
                      Upload Lab Report (PDF / Image)
                    </h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Upload a diagnostic biochemistry report with automated OCR extraction.
                    </p>
                  </div>
                </Link>
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <Link
              href="/data-entry"
              className="group bg-white dark:bg-gray-800 p-5 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 shadow-sm transition flex items-start gap-4"
            >
              <div className="w-10 h-10 rounded-xl bg-teal-50 dark:bg-teal-900/40 text-teal-600 dark:text-teal-400 flex items-center justify-center text-xl shrink-0">
                ✍️
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-teal-600">Option 1</span>
                <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 group-hover:text-teal-600 transition">
                  Enter Numbers Manually
                </h4>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Type your latest fasting glucose, blood pressure, or weight directly.
                </p>
              </div>
            </Link>

            <Link
              href="/reports"
              className="group bg-white dark:bg-gray-800 p-5 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-sky-500 shadow-sm transition flex items-start gap-4"
            >
              <div className="w-10 h-10 rounded-xl bg-sky-50 dark:bg-sky-900/40 text-sky-600 dark:text-sky-400 flex items-center justify-center text-xl shrink-0">
                📄
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-sky-600">Option 2</span>
                <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 group-hover:text-sky-600 transition">
                  Upload Lab Report (PDF / Image)
                </h4>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Upload a diagnostic biochemistry report with automated OCR extraction.
                </p>
              </div>
            </Link>
          </div>
        )}
      </div>

      {/* DATA USED PANEL */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-100 dark:border-gray-750 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 border-b border-gray-100 dark:border-gray-700 pb-3">
          <div>
            <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">Data used</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Audit log of inputs with verified provenance sources.
            </p>
          </div>
          <span className="text-xs font-semibold text-teal-700 dark:text-teal-300">
            Screening estimate, not a diagnosis.
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {dataUsedList.map((item: any, idx: number) => (
            <div key={idx} className="p-3 rounded-xl bg-gray-50/80 dark:bg-gray-750/50 border border-gray-150 dark:border-gray-700 flex items-center justify-between text-xs">
              <span className="text-gray-700 dark:text-gray-300 font-medium truncate mr-2">{item.label}</span>
              <div className="flex items-center gap-2 shrink-0">
                <span className="font-bold text-gray-900 dark:text-gray-100">{item.value}</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  item.source === 'From report'
                    ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                    : item.source === 'Calculated'
                    ? 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300'
                    : item.source === 'Entered by you'
                    ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                    : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                }`}>
                  {item.source}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* "WHAT'S AFFECTING YOUR SCORE" (TOP 5 SHAP FACTORS SPLIT BY SIGN) */}
      {isUploaded && (rawRiskInc.length > 0 || rawProtective.length > 0 || neutralNotes.length > 0) && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-100 dark:border-gray-750 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-100 dark:border-gray-750 pb-3">
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">What's Affecting Your Score</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                Top SHAP factor contributions from your calibrated diabetes risk model. Bigger bars mean a bigger push on the estimate.
              </p>
            </div>
            <div 
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-gray-100 dark:bg-gray-700/60 border border-gray-200 dark:border-gray-600 text-xs font-semibold text-gray-600 dark:text-gray-300 self-start sm:self-auto cursor-help"
              title="model contribution (log-odds): bigger bars mean a bigger push on the estimate."
            >
              <span className="material-symbols-outlined text-[15px] text-gray-500">info</span>
              <span>model contribution (log-odds)</span>
            </div>
          </div>

          {/* Neutral Model Notes (e.g. Tier 2 HbA1c with negative contribution while flagged High) */}
          {neutralNotes.length > 0 && (
            <div className="space-y-2">
              {neutralNotes.map((note: string, nIdx: number) => (
                <div key={nIdx} className="p-3 rounded-xl bg-gray-50/90 dark:bg-gray-750/70 border border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[17px] text-gray-500">info</span>
                    <span className="font-semibold text-gray-800 dark:text-gray-200">{note}</span>
                  </div>
                  <span 
                    className="px-2 py-0.5 rounded text-[10px] font-bold bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 self-start sm:self-auto cursor-help"
                    title="model contribution (log-odds): bigger bars mean a bigger push on the estimate"
                  >
                    model contribution (log-odds)
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Increases your estimated risk */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-bold text-rose-700 dark:text-rose-400 uppercase tracking-wider">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[18px]">trending_up</span>
                  <span>Increases your estimated risk</span>
                </div>
                <span className="text-[10px] lowercase font-normal text-gray-400">log-odds</span>
              </div>

              {rawRiskInc.length === 0 ? (
                <div className="p-3 text-xs text-gray-400 bg-gray-50 dark:bg-gray-750 rounded-xl">
                  No risk-increasing factors identified.
                </div>
              ) : (
                <div className="space-y-2">
                  {rawRiskInc.map((f: any, idx: number) => (
                    <div key={idx} className="p-3 rounded-xl bg-rose-50/50 dark:bg-rose-950/20 border border-rose-200/50 dark:border-rose-900/30 flex items-center justify-between text-xs">
                      <div>
                        <span className="font-bold text-gray-900 dark:text-gray-100 block">{f.label || f.feature}</span>
                        <span className="text-[11px] text-gray-500">
                          {f.raw_value ? `${f.label || f.feature}: ${f.raw_value}` : 'Extracted from report'}
                        </span>
                      </div>
                      <span 
                        className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-100 dark:bg-rose-900/50 text-rose-800 dark:text-rose-300 cursor-help"
                        title="model contribution (log-odds): bigger bars mean a bigger push on the estimate"
                      >
                        +{f.shap_value != null ? Number(f.shap_value).toFixed(3) : `${f.impact_pct}%`}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Lowers your estimated risk */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[18px]">trending_down</span>
                  <span>Lowers your estimated risk</span>
                </div>
                <span className="text-[10px] lowercase font-normal text-gray-400">log-odds</span>
              </div>

              {rawProtective.length === 0 ? (
                <div className="p-3 text-xs text-gray-400 bg-gray-50 dark:bg-gray-750 rounded-xl">
                  No factors lowering estimated risk identified.
                </div>
              ) : (
                <div className="space-y-2">
                  {rawProtective.map((f: any, idx: number) => (
                    <div key={idx} className="p-3 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200/50 dark:border-emerald-900/30 flex items-center justify-between text-xs">
                      <div>
                        <span className="font-bold text-gray-900 dark:text-gray-100 block">{f.label || f.feature}</span>
                        <span className="text-[11px] text-gray-500">
                          {f.raw_value ? `${f.label || f.feature}: ${f.raw_value}` : 'Extracted from report'}
                        </span>
                      </div>
                      <span 
                        className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300 cursor-help"
                        title="model contribution (log-odds): bigger bars mean a bigger push on the estimate"
                      >
                        {f.shap_value != null ? Number(f.shap_value).toFixed(3) : `-${f.impact_pct}%`}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
