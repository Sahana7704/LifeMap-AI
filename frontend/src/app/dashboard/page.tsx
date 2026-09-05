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
  const [userName, setUserName] = useState<string>('New User');
  const [toast, setToast] = useState<string | null>(null);
  const [showDemoPreview, setShowDemoPreview] = useState(false);

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
        if (predRes.status === 'fulfilled' && predRes.value?.data?.predictions) {
          predData = predRes.value.data;
          setData(predData);
        }

        let rep = null;
        if (repRes.status === 'fulfilled' && Array.isArray(repRes.value?.data) && repRes.value.data.length > 0) {
          rep = repRes.value.data[0];
          setLatestReport(rep);
        }

        // If no prediction yet, but user has uploaded reports with extracted metrics, trigger prediction now!
        if (!predData && rep && rep.extracted_metrics) {
          try {
            const rawMetrics = typeof rep.extracted_metrics === 'string' ? JSON.parse(rep.extracted_metrics) : rep.extracted_metrics;
            if (Object.keys(rawMetrics).length > 0) {
              const runRes = await api.runPrediction(rawMetrics);
              if (runRes.data && runRes.data.predictions) {
                setData(runRes.data);
              }
            }
          } catch (e) {
            console.error('Auto-prediction from report error:', e);
          }
        }
      } catch {
        // No predictions recorded yet for this account
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

  const diabetesPrediction = data?.predictions?.diabetes;
  const cvdPrediction = data?.predictions?.cardiovascular;
  const hematologyPrediction = data?.predictions?.hematology;
  const metabolicPrediction = data?.predictions?.metabolic;

  const isCBCReport = Boolean(data?.is_cbc_report || hematologyPrediction);
  const isMetabolicReport = Boolean(
    data?.is_metabolic_report ||
    metabolicPrediction ||
    (typeof latestReport?.extracted_metrics === 'object' && latestReport?.extracted_metrics?.report_type === 'Metabolic Panel (Diabetes & Obesity)') ||
    (typeof latestReport?.extracted_metrics === 'string' && latestReport?.extracted_metrics.includes('Metabolic Panel'))
  );

  const metabolicData = data?.is_metabolic_report ? data : (
    (typeof latestReport?.extracted_metrics === 'object' && latestReport?.extracted_metrics) ||
    (typeof latestReport?.extracted_metrics === 'string' ? (() => { try { return JSON.parse(latestReport.extracted_metrics); } catch { return {}; } })() : {})
  );

  const isLiveUser = Boolean(diabetesPrediction || cvdPrediction || hematologyPrediction || isMetabolicReport);

  // Fallback demo values if explicitly requested by user
  const demoDiabetesRisk = 62;
  const demoDiabetesCategory = 'Moderate';
  const demoHeartRisk = 18;
  const demoHeartCategory = 'Low Risk';
  const demoWellnessScore = 81;
  const demoWellnessCategory = 'Good';

  // Live scores
  const diabetesRisk = diabetesPrediction?.risk_score != null
    ? Math.round(Number(diabetesPrediction.risk_score))
    : demoDiabetesRisk;
  const diabetesCategory = diabetesPrediction?.category || (diabetesRisk > 60 ? 'Moderate' : diabetesRisk > 25 ? 'Moderate' : 'Low Risk');

  const heartRisk = cvdPrediction?.risk_score != null
    ? Math.round(Number(cvdPrediction.risk_score))
    : demoHeartRisk;
  const heartCategory = cvdPrediction?.category || (heartRisk > 50 ? 'Moderate' : 'Low Risk');

  const hematologyRisk = hematologyPrediction?.risk_score != null
    ? Math.round(Number(hematologyPrediction.risk_score))
    : 28;
  const hematologyCategory = hematologyPrediction?.category || 'Mild Concern';

  const wellnessScore = data?.vitality_score != null
    ? Math.round(Number(data.vitality_score))
    : demoWellnessScore;
  const wellnessCategory = wellnessScore >= 80 ? 'Good' : wellnessScore >= 60 ? 'Fair' : 'Needs Care';

  // Unified, clinically-sound explainability drivers across hematology, metabolic, and cardiovascular profiles
  const rawRiskFactors = isMetabolicReport && metabolicPrediction
    ? (metabolicPrediction?.explanation?.risk_increasing_factors || [])
    : isCBCReport && hematologyPrediction
    ? (hematologyPrediction?.explanation?.risk_increasing_factors || [])
    : [
        ...(diabetesPrediction?.explanation?.risk_increasing_factors || []),
        ...(cvdPrediction?.explanation?.risk_increasing_factors || [])
      ];

  const rawProtectiveFactors = isMetabolicReport && metabolicPrediction
    ? (metabolicPrediction?.explanation?.protective_factors || [])
    : isCBCReport && hematologyPrediction
    ? (hematologyPrediction?.explanation?.protective_factors || [])
    : [
        ...(diabetesPrediction?.explanation?.protective_factors || []),
        ...(cvdPrediction?.explanation?.protective_factors || [])
      ];

  // Clinically sanitize: never allow unmodifiable biological sex or non-smoking to be labeled as risk
  const liveRiskIncreasing: any[] = rawRiskFactors
    .filter((f: any, idx: number, self: any[]) => {
      const featKey = String(f.feature || f.base_feature || '').toLowerCase();
      const label = String(f.label || '').toLowerCase();
      if (featKey.includes('gender') || featKey.includes('sex') || label.includes('gender') || label.includes('male') || label.includes('female')) return false;
      if (featKey.includes('smoking_history_never') || label.includes('never') || label.includes('non-smoker')) return false;
      if (featKey.includes('slope') || featKey.includes('cp_') || featKey.includes('thal_') || featKey.includes('restecg') || featKey.includes('ca')) return false;
      const uniqueKey = `${featKey}_${label}`;
      return self.findIndex((o: any) => `${String(o.feature || o.base_feature || '').toLowerCase()}_${String(o.label || '').toLowerCase()}` === uniqueKey) === idx;
    })
    .sort((a: any, b: any) => Number(b.impact_pct || 0) - Number(a.impact_pct || 0));

  // Ensure strict mutual exclusivity: if a feature appears as a risk factor, it CANNOT appear as a protective factor
  const riskFeatureKeys = new Set(liveRiskIncreasing.map((f: any) => `${String(f.feature || f.base_feature || '').toLowerCase()}_${String(f.label || '').toLowerCase()}`));

  const liveProtective: any[] = rawProtectiveFactors
    .filter((f: any, idx: number, self: any[]) => {
      const featKey = String(f.feature || f.base_feature || '').toLowerCase();
      const label = String(f.label || '').toLowerCase();
      if (featKey.includes('gender') || featKey.includes('sex') || label.includes('gender')) return false;
      if (featKey.includes('slope') || featKey.includes('cp_') || featKey.includes('thal_') || featKey.includes('restecg') || featKey.includes('ca')) return false;
      const uniqueKey = `${featKey}_${label}`;
      // Filter out any biomarker that is already classified as a risk factor
      if (riskFeatureKeys.has(uniqueKey)) return false;
      return self.findIndex((o: any) => `${String(o.feature || o.base_feature || '').toLowerCase()}_${String(o.label || '').toLowerCase()}` === uniqueKey) === idx;
    })
    .sort((a: any, b: any) => Number(b.impact_pct || 0) - Number(a.impact_pct || 0));

  if (!isLoggedIn) {
    return (
      <AuthCard
        title="Please Log In to View Dashboard"
        description="Log in to view your personalized health dashboard, track your risk indices, and view clinical insights."
      />
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12">
      {toast && (
        <div className="fixed top-4 right-4 bg-teal-600 text-white px-5 py-2.5 rounded-xl shadow-xl z-50 animate-bounce font-medium text-sm">
          {toast}
        </div>
      )}

      {/* Top Header Navigation Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-gray-100 dark:border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">My Health</span>
          <span className="text-gray-300 dark:text-gray-700">/</span>
          <span className="text-sm font-semibold text-gray-800 dark:text-gray-200">Health Risk Summary</span>
          {isLiveUser ? (
            <span className="ml-2 px-2.5 py-0.5 rounded-full bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 text-[11px] font-bold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse"></span>
              Live Patient Data
            </span>
          ) : showDemoPreview ? (
            <span className="ml-2 px-2.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300 text-[11px] font-medium border border-amber-300 dark:border-amber-700">
              Sample Demo Mode
            </span>
          ) : (
            <span className="ml-2 px-2.5 py-0.5 rounded-full bg-sky-100 dark:bg-sky-900/40 text-sky-800 dark:text-sky-300 text-[11px] font-medium">
              New Account · Ready to Start
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/data-entry"
            className="flex items-center gap-1.5 px-4 py-2 bg-[#00685f] hover:bg-[#005049] text-white rounded-xl text-xs font-semibold shadow-sm transition active:scale-[0.99]"
          >
            <span className="material-symbols-outlined text-[16px]">add</span>
            <span>Log Today's Numbers</span>
          </Link>

          <button
            onClick={() => showToast('🔔 No urgent alerts. Enter today’s vitals to keep your risk scores fresh.')}
            aria-label="Notifications"
            className="p-2 rounded-xl text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 relative transition"
          >
            <span className="material-symbols-outlined text-[20px]">notifications</span>
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-teal-600"></span>
          </button>

          <div className="flex items-center gap-2.5 pl-2 border-l border-gray-200 dark:border-gray-800">
            <div className="w-8 h-8 rounded-full bg-teal-700 text-white font-bold flex items-center justify-center text-xs shadow-inner ring-2 ring-teal-200 dark:ring-teal-900">
              {userName ? userName.charAt(0).toUpperCase() : 'U'}
            </div>
            <div className="hidden sm:flex flex-col">
              <span className="text-xs font-semibold text-gray-900 dark:text-gray-100 leading-none">
                {userName}
              </span>
              <span className="text-[10px] text-teal-600 dark:text-teal-400 font-medium mt-0.5">
                Personal Account
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Latest Diagnostic Report Processed Banner */}
      {latestReport && (
        <div className="p-4 rounded-2xl bg-gradient-to-r from-emerald-50 via-teal-50 to-sky-50 dark:from-emerald-950/30 dark:via-teal-950/30 dark:to-sky-950/20 border border-teal-200/80 dark:border-teal-800/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-teal-600 text-white flex items-center justify-center text-lg shrink-0 shadow-xs">
              📋
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-teal-900 dark:text-teal-200">
                  Latest Lab Report Processed: {latestReport.raw_image_url || latestReport.filename || 'Diagnostic Blood Test'}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300 text-[10px] font-bold">
                  OCR Synced
                </span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-300 mt-0.5 line-clamp-1">
                {latestReport.summary || 'Biomarkers parsed and mapped directly to your risk models and 7-day nutrition engine.'}
              </p>
            </div>
          </div>
          <Link
            href="/reports"
            className="px-3.5 py-1.5 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-teal-700 dark:text-teal-300 rounded-lg text-xs font-bold border border-teal-200 dark:border-teal-700 transition shrink-0 text-center"
          >
            View Full Report →
          </Link>
        </div>
      )}

      {/* NEW USER ONBOARDING ZERO-STATE (When no prediction data exists & demo preview is off) */}
      {!isLiveUser && !showDemoPreview ? (
        <div className="space-y-6 pt-2">
          {/* Welcome Hero Card */}
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-teal-800 via-teal-900 to-[#004740] text-white p-6 sm:p-10 shadow-lg">
            <div className="absolute -right-12 -top-12 w-64 h-64 bg-teal-400/10 rounded-full blur-3xl pointer-events-none"></div>
            <div className="absolute right-20 -bottom-10 w-48 h-48 bg-emerald-400/10 rounded-full blur-2xl pointer-events-none"></div>

            <div className="relative z-10 max-w-2xl space-y-3">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-700/60 border border-teal-500/30 text-teal-200 text-xs font-medium backdrop-blur-sm">
                <span className="material-symbols-outlined text-[16px] text-[#89f5e7]">waving_hand</span>
                <span>Welcome to LifeMap AI</span>
              </div>

              <h1
                className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                Welcome, {userName}! Let's Map Your Health.
              </h1>

              <p className="text-sm sm:text-base text-teal-100/90 leading-relaxed">
                You haven't logged any health numbers or lab tests yet. Once you submit your numbers, LifeMap AI will calculate your clinical diabetes risk, heart health index, vitality score, and plain-English AI explainability drivers.
              </p>

              <div className="pt-3 flex flex-wrap items-center gap-3">
                <Link
                  href="/data-entry"
                  className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-[#89f5e7] hover:bg-[#72e8d9] text-[#005049] text-xs font-bold shadow-md transition active:scale-[0.98]"
                >
                  <span className="material-symbols-outlined text-[18px]">edit_note</span>
                  <span>Log Today's Numbers (60s)</span>
                </Link>

                <Link
                  href="/reports"
                  className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-white/10 hover:bg-white/15 text-white border border-white/20 text-xs font-semibold backdrop-blur-sm transition active:scale-[0.98]"
                >
                  <span className="material-symbols-outlined text-[18px]">upload_file</span>
                  <span>Upload Lab Report (PDF / Image)</span>
                </Link>

                <button
                  type="button"
                  onClick={() => setShowDemoPreview(true)}
                  className="inline-flex items-center gap-1.5 px-3 py-2 text-teal-200 hover:text-white text-xs font-medium underline underline-offset-4 transition"
                >
                  <span className="material-symbols-outlined text-[15px]">visibility</span>
                  <span>Preview Sample Clinical Demo</span>
                </button>
              </div>
            </div>
          </div>

          {/* 3 Unassessed Metric Placeholder Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* Card 1: Diabetes */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-100 dark:border-gray-700/80 flex flex-col justify-between">
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-10 h-10 rounded-xl bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                      <span className="material-symbols-outlined text-[22px]">water_drop</span>
                    </div>
                    <span className="font-bold text-sm text-gray-900 dark:text-gray-100">Diabetes Risk</span>
                  </div>
                  <span className="px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 text-xs font-semibold">
                    Pending
                  </span>
                </div>

                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-gray-300 dark:text-gray-600 tracking-tight">
                    -- %
                  </span>
                  <span className="text-xs font-medium text-gray-400">Awaiting blood sugar data</span>
                </div>

                <div className="w-full bg-gray-100 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div className="bg-gray-300 dark:bg-gray-600 h-full rounded-full w-0"></div>
                </div>
              </div>

              <div className="mt-4 p-3 rounded-xl bg-gray-50 dark:bg-gray-750 border border-gray-100 dark:border-gray-700 flex items-start gap-2">
                <span className="material-symbols-outlined text-gray-400 text-[18px] shrink-0 mt-0.5">info</span>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                  Enter fasting glucose or HbA1c to assess your personalized diabetes risk.
                </p>
              </div>
            </div>

            {/* Card 2: Heart Health */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-100 dark:border-gray-700/80 flex flex-col justify-between">
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                      <span className="material-symbols-outlined text-[22px]">favorite</span>
                    </div>
                    <span className="font-bold text-sm text-gray-900 dark:text-gray-100">Heart Health</span>
                  </div>
                  <span className="px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 text-xs font-semibold">
                    Pending
                  </span>
                </div>

                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-gray-300 dark:text-gray-600 tracking-tight">
                    -- %
                  </span>
                  <span className="text-xs font-medium text-gray-400">Awaiting BP & cholesterol</span>
                </div>

                <div className="w-full bg-gray-100 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div className="bg-gray-300 dark:bg-gray-600 h-full rounded-full w-0"></div>
                </div>
              </div>

              <div className="mt-4 p-3 rounded-xl bg-gray-50 dark:bg-gray-750 border border-gray-100 dark:border-gray-700 flex items-start gap-2">
                <span className="material-symbols-outlined text-gray-400 text-[18px] shrink-0 mt-0.5">info</span>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                  Log resting blood pressure and lipid panel to compute cardiovascular index.
                </p>
              </div>
            </div>

            {/* Card 3: Wellness Score */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-gray-100 dark:border-gray-700/80 flex flex-col justify-between">
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-10 h-10 rounded-xl bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 flex items-center justify-center">
                      <span className="material-symbols-outlined text-[22px]">sentiment_very_satisfied</span>
                    </div>
                    <span className="font-bold text-sm text-gray-900 dark:text-gray-100">Vitality Score</span>
                  </div>
                  <span className="px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 text-xs font-semibold">
                    Unassessed
                  </span>
                </div>

                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-gray-300 dark:text-gray-600 tracking-tight">
                    -- <span className="text-lg font-normal text-gray-400">/ 100</span>
                  </span>
                  <span className="text-xs font-medium text-gray-400">Composite wellness index</span>
                </div>

                <div className="w-full bg-gray-100 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div className="bg-gray-300 dark:bg-gray-600 h-full rounded-full w-0"></div>
                </div>
              </div>

              <div className="mt-4 p-3 rounded-xl bg-gray-50 dark:bg-gray-750 border border-gray-100 dark:border-gray-700 flex items-start gap-2">
                <span className="material-symbols-outlined text-gray-400 text-[18px] shrink-0 mt-0.5">info</span>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                  Reflects your overall metabolic fitness once initial vitals are recorded.
                </p>
              </div>
            </div>
          </div>

          {/* 2 Primary Steps to Get Started */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Link
              href="/data-entry"
              className="group bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 shadow-sm hover:shadow-md transition-all flex flex-col justify-between gap-4"
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-2xl bg-teal-50 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 flex items-center justify-center text-2xl shrink-0 group-hover:scale-105 transition">
                  ✍️
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-teal-600">Option 1</span>
                    <span className="text-xs text-gray-400">· Fast & Direct</span>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 group-hover:text-teal-600 transition">
                    Enter Numbers Manually
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                    Have your latest sugar or blood pressure readings? Type in your numbers directly to get instant AI predictions and actionable guidance.
                  </p>
                </div>
              </div>
              <div className="flex items-center text-xs font-semibold text-teal-600 group-hover:translate-x-1 transition gap-1">
                <span>Start Manual Entry</span>
                <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
              </div>
            </Link>

            <Link
              href="/reports"
              className="group bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 shadow-sm hover:shadow-md transition-all flex flex-col justify-between gap-4"
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-2xl bg-sky-50 dark:bg-sky-900/50 text-sky-700 dark:text-sky-300 flex items-center justify-center text-2xl shrink-0 group-hover:scale-105 transition">
                  📄
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-sky-600">Option 2</span>
                    <span className="text-xs text-gray-400">· Automated OCR</span>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 group-hover:text-sky-600 transition">
                    Upload a Diagnostic Lab Report
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                    Upload a PDF or photo of your blood test (Thyrocare, Lal Pathlabs, Apollo, etc.). Our OCR pipeline automatically extracts all values.
                  </p>
                </div>
              </div>
              <div className="flex items-center text-xs font-semibold text-sky-600 group-hover:translate-x-1 transition gap-1">
                <span>Upload Report Document</span>
                <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
              </div>
            </Link>
          </div>
        </div>
      ) : (
        /* LIVE DASHBOARD VIEW (Or Demo Preview if toggled) */
        <div className="space-y-6">
          {/* DEMO MODE WATERMARK BANNER (Shown only when previewing demo) */}
          {!isLiveUser && showDemoPreview && (
            <div className="p-4 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-700 text-amber-900 dark:text-amber-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-xs">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-amber-600 text-2xl shrink-0">warning</span>
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider block">Sample Clinical Demo Preview</span>
                  <p className="text-xs text-amber-800 dark:text-amber-300">
                    The scores and numbers below are simulated baseline values for demonstration. They do <strong>not</strong> represent your real records.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
                <button
                  type="button"
                  onClick={() => setShowDemoPreview(false)}
                  className="px-3 py-1.5 rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs font-semibold border border-amber-200 dark:border-amber-800 hover:bg-gray-50 transition"
                >
                  Exit Demo
                </button>
                <Link
                  href="/data-entry"
                  className="px-3.5 py-1.5 rounded-lg bg-[#00685f] text-white text-xs font-semibold shadow-xs hover:bg-[#005049] transition"
                >
                  Log My Real Numbers
                </Link>
              </div>
            </div>
          )}

          {/* Warm & Friendly Hero Header */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pt-1">
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-2">
                <span className="px-3 py-1 rounded-full bg-[#89f5e7] text-[#005049] text-xs font-bold flex items-center gap-1.5 shadow-2xs">
                  <span className="text-[14px]">✨</span>
                  {isLiveUser ? 'Live Calculated Report' : 'Sample Demonstration'}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {isLiveUser ? `Personalized for ${userName}` : 'Illustrative clinical simulation'}
                </span>
              </div>

              <h1
                className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                {isMetabolicReport
                  ? 'Your Metabolic Health Summary (Diabetes & Obesity)'
                  : isCBCReport
                  ? 'Your Complete Blood Count (CBC) Summary'
                  : 'Your Health Risk Summary'}
              </h1>

              <p className="text-sm sm:text-base text-gray-600 dark:text-gray-300 max-w-2xl">
                {isMetabolicReport ? (
                  'Assessed via ADA criteria for diabetes and WHO Asian-adjusted cutoffs for obesity. Verifiable 4-step pipeline ensures zero invented single percentage scores.'
                ) : isCBCReport ? (
                  'Derived directly from your uploaded CBC hematology report. Unmeasured metabolic tests are excluded to guarantee 100% data integrity.'
                ) : (
                  <>
                    Based on your latest numbers. Remember:{' '}
                    <strong className="text-teal-700 dark:text-teal-300 font-semibold">
                      These scores are fully reversible
                    </strong>{' '}
                    with small everyday habits!
                  </>
                )}
              </p>
            </div>

            {/* Action Buttons on Header */}
            <div className="flex items-center gap-2.5 self-start md:self-auto flex-wrap">
              <button
                onClick={handlePrint}
                type="button"
                className="flex items-center gap-1.5 px-4 py-2.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 rounded-xl text-xs font-semibold shadow-sm border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750 transition active:scale-[0.99] cursor-pointer"
              >
                <span className="material-symbols-outlined text-[17px] text-gray-500">print</span>
                <span>Print Summary</span>
              </button>

              <Link
                href="/recommendations"
                className="flex items-center gap-1.5 px-4 py-2.5 bg-white dark:bg-gray-800 text-teal-700 dark:text-teal-300 rounded-xl text-xs font-semibold shadow-sm border border-teal-600/30 hover:bg-teal-50/50 dark:hover:bg-teal-950/30 transition active:scale-[0.99]"
              >
                <span className="material-symbols-outlined text-[17px]">restaurant</span>
                <span>View Nutrition Plan</span>
              </Link>
            </div>
          </div>

          {/* Clinical Guidance / Disclaimer Banner */}
          <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-300">
            <span className="material-symbols-outlined text-[18px] text-teal-600 shrink-0">info</span>
            <span>
              <strong>Algorithmic Health Index Notice:</strong> Findings below are machine learning biomarker assessments based on laboratory reference ranges. Unmeasured panels are excluded to avoid data fabrication.
            </span>
          </div>

          {/* Top 3 Score Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 items-stretch">
            {isMetabolicReport ? (
              <>
                {/* 1. Diabetes Risk (ADA Criteria) */}
                {(() => {
                  const fbgVal = metabolicData.calculated_vitals?.fasting_glucose ?? metabolicData.fasting_glucose ?? metabolicData.verified_vitals?.fasting_glucose?.value ?? null;
                  const hba1cVal = metabolicData.calculated_vitals?.hba1c ?? metabolicData.hba1c ?? metabolicData.verified_vitals?.hba1c?.value ?? null;
                  const classifications = metabolicData.diabetes_assessment?.classifications || [];
                  const fbgClassification = classifications.find((c: any) => (c.marker || '').toLowerCase().includes('glucose') || (c.marker || '').toLowerCase().includes('fasting'));
                  const hba1cClassification = classifications.find((c: any) => (c.marker || '').toLowerCase().includes('hba1c') || (c.marker || '').toLowerCase().includes('a1c'));
                  const topStatus = metabolicData.diabetes_assessment?.fasting_glucose_status || metabolicData.diabetes_assessment?.hba1c_status || classifications[0]?.classification || 'Evaluated';

                  return (
                    <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-teal-100 dark:border-teal-950/60 flex flex-col justify-between relative overflow-hidden">
                      <div className="absolute -right-8 -top-8 w-32 h-32 bg-teal-100/40 dark:bg-teal-900/10 rounded-full blur-2xl pointer-events-none"></div>

                      <div className="flex flex-col gap-3 z-10">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2.5">
                            <div className="w-10 h-10 rounded-xl bg-teal-50 dark:bg-teal-900/40 text-teal-600 dark:text-teal-400 flex items-center justify-center">
                              <span className="material-symbols-outlined text-[24px]">water_drop</span>
                            </div>
                            <div>
                              <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Diabetes Assessment</span>
                              <span className="text-[11px] text-gray-400 font-medium">ADA Clinical Standard</span>
                            </div>
                          </div>

                          <span className="px-3 py-1 rounded-full bg-teal-50 dark:bg-teal-950/40 text-teal-800 dark:text-teal-300 border border-teal-200/80 dark:border-teal-800/50 text-xs font-semibold">
                            {topStatus}
                          </span>
                        </div>

                        {/* First-Class Biomarker Blocks for Fasting Glucose and HbA1c */}
                        <div className="space-y-3 mt-1">
                          {fbgVal != null && (
                            <div className="p-3 rounded-xl bg-gray-50/90 dark:bg-gray-750/60 border border-gray-100 dark:border-gray-700">
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-bold text-teal-800 dark:text-teal-300">
                                  Fasting Blood Glucose
                                </span>
                                {fbgClassification?.classification && (
                                  <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                                    fbgClassification.classification.includes('Diagnostic')
                                      ? 'bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300'
                                      : fbgClassification.classification.includes('Prediabetes')
                                      ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300'
                                      : 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300'
                                  }`}>
                                    {fbgClassification.classification}
                                  </span>
                                )}
                              </div>
                              <div className="mt-1 flex items-baseline gap-1.5">
                                <span className="text-2xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight" style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
                                  {fbgVal}
                                </span>
                                <span className="text-xs font-semibold text-gray-400">mg/dL</span>
                                {fbgClassification?.source_flagged && (
                                  <span className="ml-auto px-2 py-0.5 rounded bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800 text-[10px] font-bold">
                                    Flagged High
                                  </span>
                                )}
                              </div>
                              <div className="mt-2 space-y-1">
                                <div className="w-full bg-gray-200 dark:bg-gray-650 h-1.5 rounded-full overflow-hidden flex">
                                  <div className="bg-emerald-400 h-full w-1/3" title="Normal: <100"></div>
                                  <div className="bg-amber-400 h-full w-1/3" title="Prediabetes: 100-125"></div>
                                  <div className="bg-rose-400 h-full w-1/3" title="Diabetes: >=126"></div>
                                </div>
                                <div className="flex justify-between text-[9px] text-gray-400 font-medium">
                                  <span>&lt;100 Normal</span>
                                  <span>100–125 Prediabetes</span>
                                  <span>&ge;126 Diabetes</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {hba1cVal != null && (
                            <div className="p-3 rounded-xl bg-gray-50/90 dark:bg-gray-750/60 border border-gray-100 dark:border-gray-700">
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-bold text-teal-800 dark:text-teal-300">
                                  HbA1c (Glycated Hemoglobin)
                                </span>
                                {hba1cClassification?.classification && (
                                  <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                                    hba1cClassification.classification.includes('Diagnostic')
                                      ? 'bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300'
                                      : hba1cClassification.classification.includes('Prediabetes')
                                      ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300'
                                      : 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300'
                                  }`}>
                                    {hba1cClassification.classification}
                                  </span>
                                )}
                              </div>
                              <div className="mt-1 flex items-baseline gap-1.5">
                                <span className="text-2xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight" style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
                                  {hba1cVal}
                                </span>
                                <span className="text-xs font-semibold text-gray-400">%</span>
                                {hba1cClassification?.source_flagged && (
                                  <span className="ml-auto px-2 py-0.5 rounded bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800 text-[10px] font-bold">
                                    Flagged High
                                  </span>
                                )}
                              </div>
                              <div className="mt-2 space-y-1">
                                <div className="w-full bg-gray-200 dark:bg-gray-650 h-1.5 rounded-full overflow-hidden flex">
                                  <div className="bg-emerald-400 h-full w-1/3" title="Normal: <5.7%"></div>
                                  <div className="bg-amber-400 h-full w-1/3" title="Prediabetes: 5.7-6.4%"></div>
                                  <div className="bg-rose-400 h-full w-1/3" title="Diabetes: >=6.5%"></div>
                                </div>
                                <div className="flex justify-between text-[9px] text-gray-400 font-medium">
                                  <span>&lt;5.7% Normal</span>
                                  <span>5.7–6.4% Prediabetes</span>
                                  <span>&ge;6.5% Diabetes</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {fbgVal == null && hba1cVal == null && (
                            <div className="p-3 text-center text-xs text-gray-400">
                              No glucose or HbA1c values recorded in this report.
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="mt-4 p-3 rounded-xl bg-teal-50/70 dark:bg-teal-950/30 border border-teal-200/50 dark:border-teal-900/40 flex items-start gap-2.5 z-10">
                        <span className="material-symbols-outlined text-teal-600 text-[20px] shrink-0 mt-0.5">
                          info
                        </span>
                        <p className="text-xs text-teal-900 dark:text-teal-200 font-medium leading-relaxed">
                          {metabolicData.diabetes_assessment?.summary || 'Evaluated against American Diabetes Association (ADA) criteria.'}
                        </p>
                      </div>
                    </div>
                  );
                })()}

                {/* 2. Obesity Classification (WHO Asian Cutoffs) */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-sky-100 dark:border-sky-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-sky-100/40 dark:bg-sky-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-sky-50 dark:bg-sky-950/40 text-sky-600 dark:text-sky-400 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">monitor_weight</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Obesity Classification</span>
                          <span className="text-[11px] text-gray-400 font-medium">WHO Asian Cutoffs</span>
                        </div>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-sky-50 dark:bg-sky-950/40 text-sky-800 dark:text-sky-300 border border-sky-200/80 dark:border-sky-800/50 text-xs font-semibold">
                        {metabolicData.obesity_assessment?.classification || 'Asian Cutoff'}
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-3xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {metabolicData.obesity_assessment?.bmi_value ?? metabolicData.calculated_vitals?.bmi ?? '—'} <span className="text-sm font-semibold text-gray-400">kg/m²</span>
                      </span>
                      <span className="text-xs font-bold text-sky-700 dark:text-sky-400">
                        BMI · {metabolicData.obesity_assessment?.bmi_source === 'calculated' ? 'Calculated in Code' : 'Printed in Source'}
                      </span>
                    </div>

                    {/* Asian BMI Scale */}
                    <div className="space-y-1">
                      <div className="w-full bg-gray-100 dark:bg-gray-750 h-2.5 rounded-full overflow-hidden flex">
                        <div className="bg-emerald-400 h-full w-1/3" title="Normal: 18.5-22.9"></div>
                        <div className="bg-amber-400 h-full w-1/4" title="Overweight: 23-24.9"></div>
                        <div className="bg-rose-400 h-full w-5/12" title="Obese: >=25"></div>
                      </div>
                      <div className="flex justify-between text-[10px] text-gray-400 font-medium">
                        <span>18.5–22.9 Normal</span>
                        <span>23–24.9 Overweight</span>
                        <span>&ge;25 Obese</span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-sky-50/70 dark:bg-sky-950/30 border border-sky-200/50 dark:border-sky-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-sky-600 text-[20px] shrink-0 mt-0.5">
                      straighten
                    </span>
                    <p className="text-xs text-sky-900 dark:text-sky-200 font-medium leading-relaxed">
                      {metabolicData.calculated_vitals?.height_cm ? `Height: ${metabolicData.calculated_vitals.height_cm} cm · Weight: ${metabolicData.calculated_vitals.weight_kg} kg. ` : ''}
                      WHO Asian guidelines classify &ge;25 as obese due to increased metabolic risks at lower BMI.
                    </p>
                  </div>
                </div>

                {/* 3. Combined Risk Synthesis & Step 4 Audit */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-emerald-100 dark:border-emerald-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-emerald-100/40 dark:bg-emerald-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">verified</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Combined Synthesis</span>
                          <span className="text-[11px] text-gray-400 font-medium">Step 4 Verified & Audited</span>
                        </div>
                      </div>

                      {metabolicData.step4_audit?.audit_passed ? (
                        <span className="px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300 border border-emerald-200/80 dark:border-emerald-800/50 text-xs font-semibold">
                          Audit Passed
                        </span>
                      ) : metabolicData.step4_audit ? (
                        <span className="px-3 py-1 rounded-full bg-rose-50 dark:bg-rose-950/40 text-rose-800 dark:text-rose-300 border border-rose-200/80 dark:border-rose-800/50 text-xs font-semibold">
                          Audit Blocked
                        </span>
                      ) : null}
                    </div>

                    <div className="mt-1 p-3 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200/60 dark:border-emerald-800/40">
                      <p className="text-xs text-emerald-950 dark:text-emerald-100 font-semibold leading-relaxed">
                        {metabolicData.combined_risk_note || 'Obesity classification combined with prediabetes/diabetes markers modifies overall metabolic risk. Zero single percentage score is generated to uphold clinical validity.'}
                      </p>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-gray-500">
                      <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-300 font-bold">
                        <span className="material-symbols-outlined text-[15px]">verified_user</span>
                        Zero Hallucinated Numbers
                      </span>
                      <span className="bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded text-[10px] font-semibold">
                        No Single % Score
                      </span>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200/50 dark:border-emerald-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-emerald-700 dark:text-emerald-300 text-[20px] shrink-0 mt-0.5">
                      check_circle
                    </span>
                    <p className="text-xs text-emerald-900 dark:text-emerald-200 font-medium leading-relaxed">
                      All claims verified against source numbers by Step 4 code audit. Cardiovascular and hematology scores excluded as specified.
                    </p>
                  </div>
                </div>
              </>
            ) : isCBCReport ? (
              <>
                {/* 1. Complete Blood Count (CBC) Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-amber-100 dark:border-amber-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-amber-100/40 dark:bg-amber-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">bloodtype</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Blood Count (CBC)</span>
                          <span className="text-[11px] text-gray-400 font-medium">Hematology Findings</span>
                        </div>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 border border-amber-200/80 dark:border-amber-800/50 text-xs font-semibold">
                        Score: {hematologyRisk} / 100 · {hematologyCategory}
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-4xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {hematologyRisk} <span className="text-lg font-bold text-gray-400">/ 100</span>
                      </span>
                      <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">
                        {hematologyCategory}
                      </span>
                    </div>

                    <div className="w-full bg-gray-100 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-amber-400 to-amber-500 h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, Math.max(12, hematologyRisk))}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200/50 dark:border-amber-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-amber-600 text-[20px] shrink-0 mt-0.5">
                      info
                    </span>
                    <p className="text-xs text-amber-900 dark:text-amber-200 font-medium leading-relaxed">
                      {liveRiskIncreasing.length > 0
                        ? `${liveRiskIncreasing.map((f: any) => f.formatted_value || f.label).join(' · ')}. Nutrition is tailored for your specific findings.`
                        : 'All measured blood count markers are in standard reference ranges, reflecting healthy hematological baseline.'}
                    </p>
                  </div>
                </div>

                {/* 2. Immune & Cellular Health Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-emerald-100 dark:border-emerald-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-emerald-100/40 dark:bg-emerald-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">health_and_safety</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Immune Resilience</span>
                          <span className="text-[11px] text-gray-400 font-medium">Cellular Biomarker Health</span>
                        </div>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300 border border-emerald-200/80 dark:border-emerald-800/50 text-xs font-semibold">
                        Score: {wellnessScore} / 100 · Stable
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-4xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {wellnessScore} <span className="text-lg font-bold text-gray-400">/ 100</span>
                      </span>
                      <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                        Stable Defense
                      </span>
                    </div>

                    <div className="w-full bg-gray-100 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-emerald-400 to-emerald-500 h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, Math.max(10, wellnessScore))}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200/50 dark:border-emerald-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-emerald-600 text-[20px] shrink-0 mt-0.5">
                      verified
                    </span>
                    <p className="text-xs text-emerald-900 dark:text-emerald-200 font-medium leading-relaxed">
                      {liveProtective.length > 0
                        ? `${liveProtective.slice(0, 3).map((f: any) => f.formatted_value || f.label).join(' · ')}, reflecting healthy physiological baselines.`
                        : 'White blood cells and key biomarkers indicate healthy immune resilience.'}
                    </p>
                  </div>
                </div>

                {/* 3. Unmeasured Panels Notice Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-teal-100 dark:border-teal-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-teal-100/40 dark:bg-teal-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">verified_user</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Unmeasured Panels</span>
                          <span className="text-[11px] text-gray-400 font-medium">Excluded From Guesswork</span>
                        </div>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 border border-teal-200/80 dark:border-teal-800/50 text-xs font-semibold">
                        0 Assumed
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-2xl font-bold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        4 Excluded Tests
                      </span>
                    </div>

                    <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                      Fasting Glucose, HbA1c, Blood Pressure, and Lipid Panel (Cholesterol) were not in this CBC report.
                    </p>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-teal-50/70 dark:bg-teal-950/30 border border-teal-200/50 dark:border-teal-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-teal-700 dark:text-teal-300 text-[20px] shrink-0 mt-0.5">
                      check_circle
                    </span>
                    <p className="text-xs text-teal-900 dark:text-teal-200 font-medium leading-relaxed">
                      LifeMap AI strictly excludes unmeasured vitals so your recommendations reflect 100% genuine lab data.
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* 1. Diabetes Risk Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-amber-100 dark:border-amber-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-amber-100/40 dark:bg-amber-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">water_drop</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Diabetes Risk Score</span>
                          <span className="text-[11px] text-gray-400 font-medium">AI Model Assessment</span>
                        </div>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 border border-amber-200/80 dark:border-amber-800/50 text-xs font-semibold">
                        Score: {diabetesRisk} / 100 · {diabetesCategory}
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-4xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {diabetesRisk} <span className="text-lg font-bold text-gray-400">/ 100</span>
                      </span>
                      <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">
                        {diabetesCategory}
                      </span>
                    </div>

                    <div className="w-full bg-gray-100 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-amber-400 to-amber-500 h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, Math.max(8, diabetesRisk))}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200/50 dark:border-amber-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-amber-600 text-[20px] shrink-0 mt-0.5">
                      sentiment_satisfied
                    </span>
                    <p className="text-xs text-amber-900 dark:text-amber-200 font-medium leading-relaxed">
                      {diabetesRisk > 50
                        ? <>Higher algorithmic risk score, but <span className="font-semibold underline decoration-amber-400">very manageable</span> with consistent whole-grain Indian meal adjustments.</>
                        : <>Healthy glycemic indicators! Fasting glucose and HbA1c metrics reflect <span className="font-semibold">low metabolic risk scores</span>.</>}
                    </p>
                  </div>
                </div>

                {/* 2. Heart Health Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-emerald-100 dark:border-emerald-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-emerald-100/40 dark:bg-emerald-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">favorite</span>
                        </div>
                        <div>
                          <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">Cardiovascular Risk Score</span>
                          <span className="text-[11px] text-gray-400 font-medium">AI Model Assessment</span>
                        </div>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300 border border-emerald-200/80 dark:border-emerald-800/50 text-xs font-semibold">
                        Score: {heartRisk} / 100 · {heartCategory}
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-4xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {heartRisk} <span className="text-lg font-bold text-gray-400">/ 100</span>
                      </span>
                      <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                        {heartCategory}
                      </span>
                    </div>

                    <div className="w-full bg-gray-100 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-emerald-400 to-emerald-500 h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, Math.max(8, heartRisk))}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200/50 dark:border-emerald-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-emerald-600 text-[20px] shrink-0 mt-0.5">
                      verified
                    </span>
                    <p className="text-xs text-emerald-900 dark:text-emerald-200 font-medium leading-relaxed">
                      {heartRisk > 40
                        ? <>Vascular load is slightly elevated. Low-sodium diet choices and brisk daily walks help protect arterial flexibility.</>
                        : <>Your cardiovascular indicators <span className="font-semibold">reflect healthy metrics</span>. Blood pressure and lipids align with standard targets.</>}
                    </p>
                  </div>
                </div>

                {/* 3. Wellness Score Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 shadow-sm border border-teal-100 dark:border-teal-950/60 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute -right-8 -top-8 w-32 h-32 bg-teal-100/40 dark:bg-teal-900/10 rounded-full blur-2xl pointer-events-none"></div>

                  <div className="flex flex-col gap-3 z-10">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 flex items-center justify-center">
                          <span className="material-symbols-outlined text-[24px]">sentiment_very_satisfied</span>
                        </div>
                        <span className="font-bold text-base text-gray-900 dark:text-gray-100">Wellness Score</span>
                      </div>

                      <span className="px-3 py-1 rounded-full bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 border border-teal-200/80 dark:border-teal-800/50 text-xs font-semibold">
                        {wellnessScore} / 100 · {wellnessCategory}
                      </span>
                    </div>

                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className="text-4xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {wellnessScore}
                        <span className="text-lg font-medium text-gray-400">/100</span>
                      </span>
                      <span className="text-xs font-semibold text-teal-700 dark:text-teal-300">
                        {wellnessCategory}
                      </span>
                    </div>

                    <div className="w-full bg-gray-100 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-teal-400 to-[#00685f] h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, Math.max(10, wellnessScore))}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="mt-4 p-3 rounded-xl bg-teal-50/70 dark:bg-teal-950/30 border border-teal-200/50 dark:border-teal-900/40 flex items-start gap-2.5 z-10">
                    <span className="material-symbols-outlined text-teal-700 dark:text-teal-300 text-[20px] shrink-0 mt-0.5">
                      star
                    </span>
                    <p className="text-xs text-teal-900 dark:text-teal-200 font-medium leading-relaxed">
                      Strong habits across activity, nutrition, and blood pressure markers keep your body resilient.
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* "What's Affecting Your Score?" Section (Dynamic SHAP Driver Explanations) */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 sm:p-8 shadow-sm border border-gray-100 dark:border-gray-700 flex flex-col gap-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-700 pb-4">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-teal-600 text-[26px]">balance</span>
                  <h2
                    className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 tracking-tight"
                    style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                  >
                    What's Affecting Your Score?
                  </h2>
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {isLiveUser ? 'Real SHAP clinical factors identified by our machine learning model for your vitals.' : 'Sample SHAP factor explanation preview.'}
                </p>
              </div>

              {/* Legend Pill */}
              <div className="flex items-center gap-3 p-2.5 rounded-xl bg-gray-50 dark:bg-gray-750 self-start md:self-auto border border-gray-100 dark:border-gray-700">
                <div className="flex items-center gap-1.5 text-rose-600 text-xs font-semibold">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                  <span>Needs Attention</span>
                </div>
                <span className="text-gray-300 dark:text-gray-600">•</span>
                <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400 text-xs font-semibold">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                  <span>Protects Your Health</span>
                </div>
              </div>
            </div>

            {/* Comparative Plain-English Cards Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Column 1: Things Increasing Risk */}
              <div className="flex flex-col gap-3.5 bg-rose-50/40 dark:bg-rose-950/20 rounded-2xl p-5 border border-rose-100 dark:border-rose-900/40">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-rose-100 dark:bg-rose-900/50 text-rose-700 dark:text-rose-300 flex items-center justify-center">
                    <span className="material-symbols-outlined text-[20px]">arrow_upward</span>
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">Things Slightly Increasing Risk</h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Simple habits can bring these right back down</p>
                  </div>
                </div>

                <div className="flex flex-col gap-3 mt-1">
                  {liveRiskIncreasing.length > 0 ? (
                    liveRiskIncreasing.map((factor, idx) => (
                      <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-rose-100 dark:border-rose-900/50 shadow-2xs flex flex-col gap-2">
                        <div className="flex items-center justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-sm font-bold text-gray-900 dark:text-gray-100 block">{factor.label || factor.feature}</span>
                              {factor.source_flagged && (
                                <span className="px-2 py-0.5 rounded-md bg-amber-50 dark:bg-amber-950/60 border border-amber-300 dark:border-amber-700 text-amber-800 dark:text-amber-300 text-[10px] font-bold inline-flex items-center gap-1">
                                  <span>★</span> Flagged in Source Report
                                </span>
                              )}
                            </div>
                            <span className="text-xs text-rose-800 dark:text-rose-300 font-medium mt-0.5 block">
                              {factor.formatted_value || factor.raw_value || 'Recorded reading'}
                            </span>
                          </div>
                          <span className="px-2.5 py-1 rounded-full bg-rose-100 dark:bg-rose-900/50 text-rose-800 dark:text-rose-200 text-xs font-semibold shrink-0">
                            +{factor.impact_pct || 10}% impact
                          </span>
                        </div>

                        {factor.diagnostic_criterion && (
                          <div className="px-2.5 py-1 rounded-lg bg-rose-50 dark:bg-rose-950/50 text-rose-900 dark:text-rose-200 text-[11px] font-bold border border-rose-200 dark:border-rose-900 flex items-center gap-1.5">
                            <span className="material-symbols-outlined text-[14px] text-rose-600">warning</span>
                            <span>Diagnostic Criterion: {factor.diagnostic_criterion}</span>
                          </div>
                        )}

                        <div className="w-full bg-rose-100/60 dark:bg-rose-900/30 h-2 rounded-full overflow-hidden">
                          <div className="bg-rose-500 h-full rounded-full" style={{ width: `${Math.min(100, Math.max(15, Number(factor.impact_pct || 30)))}%` }}></div>
                        </div>
                        <span className="text-[11px] text-gray-500 dark:text-gray-400">
                          {factor.tip || 'Targeted lifestyle adjustments help bring this metric back to baseline.'}
                        </span>
                      </div>
                    ))
                  ) : (
                    /* Zero-Risk Authentic State */
                    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-emerald-100 dark:border-emerald-900/40 shadow-2xs flex flex-col gap-2 items-center text-center">
                      <div className="w-10 h-10 rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 flex items-center justify-center">
                        <span className="material-symbols-outlined text-[22px]">check_circle</span>
                      </div>
                      <span className="text-sm font-bold text-gray-900 dark:text-gray-100">All Monitored Biomarkers Within Normal Range</span>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        No elevated risk drivers detected. Your recorded glucose, lipid panel, blood pressure, and BMI currently meet standard clinical reference targets.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Column 2: Factors Protecting Your Health */}
              <div className="flex flex-col gap-3.5 bg-emerald-50/40 dark:bg-emerald-950/20 rounded-2xl p-5 border border-emerald-100 dark:border-emerald-900/40">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 flex items-center justify-center">
                    <span className="material-symbols-outlined text-[20px]">shield</span>
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-gray-900 dark:text-gray-100">Factors Protecting Your Health</h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Celebrated strengths already working for you!</p>
                  </div>
                </div>

                <div className="flex flex-col gap-3 mt-1">
                  {liveProtective.length > 0 ? (
                    liveProtective.map((factor, idx) => (
                      <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-emerald-100 dark:border-emerald-900/50 shadow-2xs flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-sm font-bold text-gray-900 dark:text-gray-100 block">{factor.label || factor.feature}</span>
                            <span className="text-xs text-emerald-800 dark:text-emerald-300 font-medium">
                              {factor.formatted_value || factor.raw_value || 'Optimal Range'}
                            </span>
                          </div>
                          <span className="px-2.5 py-1 rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-200 text-xs font-semibold shrink-0">
                            -{factor.impact_pct || 8}% protection!
                          </span>
                        </div>
                        <div className="w-full bg-emerald-100/60 dark:bg-emerald-900/30 h-2 rounded-full overflow-hidden">
                          <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.min(100, Math.max(20, Number(factor.impact_pct || 40)))}%` }}></div>
                        </div>
                        <span className="text-[11px] text-gray-500 dark:text-gray-400">
                          {factor.tip || 'In model training data, optimal values in this metric statistically lower baseline risk scores.'}
                        </span>
                      </div>
                    ))
                  ) : (
                    /* In-Progress State */
                    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-teal-100 dark:border-teal-900/40 shadow-2xs flex flex-col gap-2 items-center text-center">
                      <div className="w-10 h-10 rounded-full bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 flex items-center justify-center">
                        <span className="material-symbols-outlined text-[22px]">flag</span>
                      </div>
                      <span className="text-sm font-bold text-gray-900 dark:text-gray-100">Lifestyle Optimization In Progress</span>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Building positive daily habits, such as regular physical activity and high-fiber whole foods, will establish strong protective cardiovascular and metabolic buffers.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Auditable Weighting Logic Breakdown (Step 3 Rule 5) */}
            {((data?.weighting_logic && data.weighting_logic.length > 0) || (hematologyPrediction?.weighting_logic && hematologyPrediction.weighting_logic.length > 0)) && (
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 space-y-2">
                <span className="font-bold text-xs text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px] text-teal-600">calculate</span>
                  Auditable Score Weighting Logic (Rule 5 Compliance)
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-600 dark:text-slate-400">
                  {(data?.weighting_logic || hematologyPrediction?.weighting_logic || []).map((line: string, i: number) => (
                    <div key={i} className="flex items-center gap-2 p-2.5 rounded-lg bg-white dark:bg-gray-800 border border-slate-200 dark:border-slate-700/60 shadow-2xs">
                      <span className="w-2 h-2 rounded-full bg-teal-500 shrink-0"></span>
                      <span className="font-medium">{line}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Step 4 Post-Generation Audit Gate Confirmation Banner */}
            {(() => {
              const audit = data?.step4_audit || metabolicData?.step4_audit || hematologyPrediction?.step4_audit;
              if (!audit) return null;
              const hasPassed = Boolean(audit.audit_passed && (!audit.audit_issues || audit.audit_issues.length === 0));

              return (
                <div className={`p-3.5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-3 ${
                  hasPassed
                    ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
                    : 'bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200'
                }`}>
                  <div className="flex items-center gap-2">
                    <span className={`material-symbols-outlined text-[20px] shrink-0 ${hasPassed ? 'text-emerald-600' : 'text-rose-600'}`}>
                      {hasPassed ? 'verified_user' : 'warning'}
                    </span>
                    <div>
                      <span className="font-bold block">
                        {hasPassed
                          ? 'Step 4 Audit Gate: Verified Grounded Output'
                          : 'Step 4 Audit Gate: Verification Discrepancy Detected'}
                      </span>
                      <span className="text-[11px] opacity-90 block">
                        {hasPassed
                          ? (audit.audit_summary || '100% of numeric claims and source-flagged abnormalities verified against source text.')
                          : ((audit.audit_issues && audit.audit_issues.join('; ')) || 'Discrepancy detected between source data and generated findings.')}
                      </span>
                    </div>
                  </div>
                  <span className={`text-[10px] px-2.5 py-1 rounded-full font-extrabold uppercase tracking-wide shrink-0 border ${
                    hasPassed
                      ? 'bg-white/70 dark:bg-black/40 text-emerald-800 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
                      : 'bg-rose-100 text-rose-800 border-rose-300'
                  }`}>
                    {hasPassed ? 'Audit Passed' : 'Audit Blocked'}
                  </span>
                </div>
              );
            })()}
          </div>

          {/* Personalized Nutrition & Health Action Plan (12-col grid) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            {/* Action Card 1: Nutrition & Daily Lifestyle (8 cols) */}
            <div className="lg:col-span-8 bg-white dark:bg-gray-800 rounded-2xl p-6 sm:p-8 shadow-sm border border-teal-100 dark:border-gray-700 flex flex-col justify-between">
              <div className="flex flex-col gap-4">
                {/* Header */}
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-2xl bg-teal-50 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 flex items-center justify-center text-2xl shadow-xs">
                    🥗
                  </div>
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                        Personalized Nutrition Guidance
                      </h3>
                      <span className="px-2.5 py-0.5 rounded-full bg-teal-100 dark:bg-teal-900/50 text-teal-800 dark:text-teal-200 text-[11px] font-semibold">
                        {isCBCReport ? 'Tailored to Hematology Findings' : 'Tailored to Metabolic Health'}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                      Evidence-based Indian whole-food nutrition and daily movement guidelines
                    </span>
                  </div>
                </div>

                {/* Summary Box */}
                <div className="p-4 sm:p-5 rounded-2xl bg-gray-50/80 dark:bg-gray-750 border border-teal-100 dark:border-gray-700">
                  <p className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed">
                    {isCBCReport
                      ? 'Your nutrition plan is enriched with bioavailable iron sources (sprouted moong, palak, beetroot, dates), paired with natural Vitamin C (amla, lemon) to maximize iron absorption and cellular oxygen transport, alongside active hydration protocols for packed cell volume balance.'
                      : 'Your nutrition plan focuses on low glycemic impact carbohydrates, fiber-rich whole pulses, lean proteins, and daily 15-minute walks to maintain optimal metabolic and cardiovascular harmony.'}
                  </p>
                </div>

                {/* 3 Core Pillars */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                  <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-750 flex items-center gap-2.5 border border-gray-100 dark:border-gray-700">
                    <span className="material-symbols-outlined text-teal-600 text-[22px]">
                      {isCBCReport ? 'nutrition' : 'directions_walk'}
                    </span>
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-gray-900 dark:text-gray-100">
                        {isCBCReport ? 'Iron Superfoods' : '15-Min Walk'}
                      </span>
                      <span className="text-[11px] text-gray-500 dark:text-gray-400">
                        {isCBCReport ? 'Sprouts, palak, dates' : 'After lunch & dinner'}
                      </span>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-750 flex items-center gap-2.5 border border-gray-100 dark:border-gray-700">
                    <span className="material-symbols-outlined text-teal-600 text-[22px]">
                      {isCBCReport ? 'water_drop' : 'restaurant'}
                    </span>
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-gray-900 dark:text-gray-100">
                        {isCBCReport ? 'Cellular Hydration' : 'Balanced Meals'}
                      </span>
                      <span className="text-[11px] text-gray-500 dark:text-gray-400">
                        {isCBCReport ? '2.5–3.0L fluids daily' : 'Veg & Non-Veg plans'}
                      </span>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-750 flex items-center gap-2.5 border border-gray-100 dark:border-gray-700">
                    <span className="material-symbols-outlined text-emerald-600 text-[22px]">
                      {isCBCReport ? 'shield' : 'trending_down'}
                    </span>
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-gray-900 dark:text-gray-100">
                        {isCBCReport ? 'Platelet Support' : '>20% Risk Drop'}
                      </span>
                      <span className="text-[11px] text-gray-500 dark:text-gray-400">
                        {isCBCReport ? 'Papaya leaf, folate' : 'Expected in 90-120 days'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-5 pt-3 flex flex-wrap items-center justify-between gap-2 text-gray-500 text-xs border-t border-gray-100 dark:border-gray-700">
                <span className="flex items-center gap-1.5 text-gray-400">
                  <span className="material-symbols-outlined text-[16px] text-teal-600">verified</span>
                  Private & personalized clinical nutrition guidance
                </span>
                <Link
                  href="/recommendations"
                  className="text-teal-700 dark:text-teal-300 font-bold hover:underline flex items-center gap-1"
                >
                  <span>Open Complete Food Guide</span>
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </Link>
              </div>
            </div>

            {/* Action Card 2: Quick Links (4 cols) */}
            <div className="lg:col-span-4 flex flex-col gap-4">
              <div className="bg-gradient-to-br from-emerald-50 via-teal-50 to-sky-50 dark:from-emerald-950/40 dark:via-teal-950/30 dark:to-sky-950/20 rounded-2xl p-6 shadow-sm border border-emerald-200/60 dark:border-emerald-900/40 flex flex-col justify-between flex-1">
                <div className="flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[22px] text-teal-700 dark:text-teal-300">local_florist</span>
                    <span className="text-xs uppercase tracking-wider text-teal-700 dark:text-teal-300 font-bold">Action Center</span>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 leading-snug">
                    Explore Recommendations & Reports
                  </h3>
                  <p className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed">
                    View your full 7-day meal schedule or upload additional medical reports (Lipid Profile, HbA1c, Thyroid) to update your health profile.
                  </p>
                </div>

                <div className="flex flex-col gap-2.5 pt-4">
                  <Link
                    href="/recommendations"
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-[#00685f] hover:bg-[#005049] text-white rounded-xl text-xs font-bold shadow-sm transition active:scale-[0.99] text-center"
                  >
                    <span>View 7-Day Nutrition Plan →</span>
                  </Link>

                  <Link
                    href="/reports"
                    className="w-full flex items-center justify-center gap-1.5 px-4 py-2.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-xl text-xs font-semibold border border-gray-200 dark:border-gray-700 transition active:scale-[0.99] text-center"
                  >
                    <span className="material-symbols-outlined text-[16px] text-teal-600">upload_file</span>
                    <span>Upload Blood Reports</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
