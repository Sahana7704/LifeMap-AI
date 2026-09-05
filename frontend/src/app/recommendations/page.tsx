'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import Link from 'next/link';
import AuthCard from '@/components/AuthCard';

export default function RecommendationsPage() {
  const [dietPlan, setDietPlan] = useState<any>(null);
  const [exercisePlan, setExercisePlan] = useState<any>(null);
  const [activeDayIdx, setActiveDayIdx] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [generating, setGenerating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [targetCalories, setTargetCalories] = useState<number>(1800);
  const [dietPreference, setDietPreference] = useState<'veg' | 'non_veg'>('veg');
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(true);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  const fetchPlans = async () => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (!token) {
        setIsLoggedIn(false);
        setLoading(false);
        return;
      }
    }

    try {
      const res = await api.getCurrentPlan();
      if (res.data) {
        if (res.data.diet_plan) {
          setDietPlan(res.data.diet_plan);
          if (res.data.diet_plan.calorie_target) {
            setTargetCalories(Number(res.data.diet_plan.calorie_target));
          }
          if (res.data.diet_plan.diet_type) {
            setDietPreference(res.data.diet_plan.diet_type === 'non_veg' ? 'non_veg' : 'veg');
          }
        }
        if (res.data.exercise_plan) {
          setExercisePlan(res.data.exercise_plan);
        }
      }
    } catch {
      // Clean initial state if no plans generated yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPlans();
  }, []);

  const handleGenerate = async (customCals?: number, customDiet?: 'veg' | 'non_veg') => {
    const calsToUse = customCals || targetCalories || 1800;
    const dietToUse = customDiet || dietPreference || 'veg';
    setGenerating(true);
    setError(null);
    try {
      const res = await api.generateRecommendations({
        calorie_target: calsToUse,
        diet_type: dietToUse,
        diet_preference: dietToUse
      });
      setDietPlan(res.data.diet_plan);
      setExercisePlan(res.data.exercise_plan);
      setActiveDayIdx(0);
      showToast(`✅ Generated personalized 7-day ${dietToUse === 'non_veg' ? 'Non-Vegetarian' : 'Vegetarian'} Indian meal plan!`);
    } catch (e: any) {
      setError(e.response?.data?.error || e.message || 'Failed to generate recommendations');
    } finally {
      setGenerating(false);
    }
  };

  const isDishNonVeg = (item: any) => {
    if (item?.diet_type === 'non_veg') return true;
    if (item?.diet_type === 'veg') return false;
    const name = String(item?.dish_name || item?.substitute || item || '').toLowerCase();
    return (
      name.includes('chicken') ||
      name.includes('egg') ||
      name.includes('fish') ||
      name.includes('murgh') ||
      name.includes('bhurji') ||
      name.includes('omelette') ||
      name.includes('macher') ||
      name.includes('keema') ||
      name.includes('salmon') ||
      name.includes('pomfret') ||
      name.includes('prawn') ||
      name.includes('meat')
    );
  };

  const renderDietBadge = (item: any) => {
    const nonVeg = isDishNonVeg(item);
    return (
      <span
        title={nonVeg ? 'Non-Vegetarian (Contains Egg / Poultry / Fish)' : '100% Pure Vegetarian'}
        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-bold border transition ${
          nonVeg
            ? 'border-rose-200 dark:border-rose-800 bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300'
            : 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300'
        }`}
      >
        <span
          className={`w-2.5 h-2.5 rounded-[2px] border flex items-center justify-center shrink-0 ${
            nonVeg ? 'border-rose-600 dark:border-rose-400' : 'border-emerald-600 dark:border-emerald-400'
          }`}
        >
          <span
            className={`w-1 h-1 rounded-full ${
              nonVeg ? 'bg-rose-600 dark:bg-rose-400' : 'bg-emerald-600 dark:bg-emerald-400'
            }`}
          ></span>
        </span>
        <span>{nonVeg ? 'Non-Veg' : 'Veg'}</span>
      </span>
    );
  };

  if (!isLoggedIn) {
    return (
      <AuthCard
        title="Please Log In to View Recommendations"
        description="Log in to view and generate your personalized Indian diet and exercise recommendations tailored to your health risk profile."
      />
    );
  }

  // Days array (with fallback for legacy single-day objects)
  const days = dietPlan?.days && dietPlan.days.length > 0 ? dietPlan.days : (
    dietPlan?.meals ? [{
      day_number: 1,
      day_name: 'Day 1',
      theme: 'Daily Nutritional Balance',
      target_calories: dietPlan.calorie_target || targetCalories,
      total_estimated_calories: dietPlan.total_estimated_calories || targetCalories,
      attainment_percentage: 100,
      macronutrients: dietPlan.total_macronutrients || {},
      meals: dietPlan.meals
    }] : []
  );

  const activeDay = days[activeDayIdx] || days[0];

  const getSlotIcon = (slot: string) => {
    const s = (slot || '').toLowerCase();
    if (s.includes('breakfast')) return '🌅';
    if (s.includes('lunch')) return '☀️';
    if (s.includes('snack')) return '☕';
    if (s.includes('dinner')) return '🌙';
    return '🍽️';
  };

  return (
    <div className="max-w-5xl mx-auto p-4 space-y-6">
      {toast && (
        <div className="fixed top-4 right-4 bg-teal-600 text-white px-5 py-2.5 rounded-lg shadow-xl z-50 animate-bounce font-medium text-sm">
          {toast}
        </div>
      )}

      {/* Header & Controls Panel */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-2xl">🥗</span>
              <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100" style={{ fontFamily: 'Inter, sans-serif' }}>
                7-Day Indian Nutrition & Wellness Plan
              </h1>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Authentic Indian regional whole-food menus clinically balanced to reach your exact daily caloric targets.
            </p>
          </div>

          <button
            onClick={() => handleGenerate()}
            disabled={generating}
            className="px-5 py-2.5 bg-[#00685f] hover:bg-[#005049] text-white font-semibold rounded-xl shadow-sm disabled:opacity-50 transition flex items-center justify-center space-x-2 whitespace-nowrap active:scale-[0.98]"
          >
            <span>{generating ? '⏳ Generating Plan…' : '✨ Regenerate 7-Day Plan'}</span>
          </button>
        </div>

        {/* Dietary Preference & Calories Control Bar */}
        <div className="pt-4 border-t border-gray-100 dark:border-gray-700 flex flex-col xl:flex-row xl:items-center justify-between gap-4">
          {/* Dietary Preference Selector */}
          <div className="flex items-center gap-3 shrink-0">
            <span className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 shrink-0">
              Diet Option:
            </span>
            <div className="inline-flex rounded-xl bg-gray-100 dark:bg-gray-750 p-1 border border-gray-200/80 dark:border-gray-700 shrink-0">
              {/* Veg Button */}
              <button
                type="button"
                onClick={() => {
                  setDietPreference('veg');
                  handleGenerate(targetCalories, 'veg');
                }}
                disabled={generating}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap transition ${
                  dietPreference === 'veg'
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-emerald-600 dark:hover:text-emerald-400'
                }`}
              >
                <span className="w-3 h-3 rounded-[2px] border-2 border-current flex items-center justify-center shrink-0">
                  <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                </span>
                <span>Vegetarian</span>
              </button>

              {/* Non-Veg Button */}
              <button
                type="button"
                onClick={() => {
                  setDietPreference('non_veg');
                  handleGenerate(targetCalories, 'non_veg');
                }}
                disabled={generating}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap transition ${
                  dietPreference === 'non_veg'
                    ? 'bg-rose-600 text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-rose-600 dark:hover:text-rose-400'
                }`}
              >
                <span className="w-3 h-3 rounded-[2px] border-2 border-current flex items-center justify-center shrink-0">
                  <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                </span>
                <span>Non-Vegetarian</span>
              </button>
            </div>
          </div>

          {/* Caloric Target Control: Exactly 1 Clinically Tailored Recommendation with Step Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 shrink-0">
              Calorie Target:
            </span>

            {(() => {
              const baseCalories = Number(dietPlan?.recommended_baseline_calories || dietPlan?.calorie_target || 1750);
              const isModified = targetCalories !== baseCalories;

              return (
                <div className="flex flex-wrap items-center gap-2.5">
                  {/* Single Tailored Target Pill */}
                  <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-teal-50 dark:bg-teal-950/50 border border-teal-200 dark:border-teal-800 text-teal-900 dark:text-teal-200 text-xs font-bold shadow-2xs">
                    <span className="material-symbols-outlined text-[16px] text-teal-600">verified</span>
                    <span>
                      Recommended:{' '}
                      <span className="text-teal-700 dark:text-teal-300 font-extrabold text-sm">{baseCalories}</span> kcal/day
                    </span>
                    <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-teal-200/60 dark:bg-teal-800/60 text-teal-800 dark:text-teal-200">
                      Tailored for Your Health
                    </span>
                  </div>

                  {/* Interactive Modifier Controls ([-] input [+]) */}
                  <div className="flex items-center gap-1 bg-gray-50 dark:bg-gray-800/80 p-1 rounded-xl border border-gray-200 dark:border-gray-700">
                    <button
                      type="button"
                      title="Decrease by 50 kcal"
                      onClick={() => {
                        const newCals = Math.max(1200, targetCalories - 50);
                        setTargetCalories(newCals);
                      }}
                      disabled={generating || targetCalories <= 1200}
                      className="w-7 h-7 rounded-lg bg-white dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 font-bold text-sm flex items-center justify-center transition border border-gray-200 dark:border-gray-600 disabled:opacity-40"
                    >
                      −
                    </button>

                    <div className="relative flex items-center">
                      <input
                        type="number"
                        min={1200}
                        max={3500}
                        step={50}
                        value={targetCalories}
                        onChange={e => setTargetCalories(Number(e.target.value) || 1200)}
                        className="w-16 py-1 text-center text-xs font-bold bg-transparent text-gray-900 dark:text-gray-100 focus:outline-none"
                      />
                      <span className="text-[11px] text-gray-400 font-medium pr-1.5 select-none">kcal</span>
                    </div>

                    <button
                      type="button"
                      title="Increase by 50 kcal"
                      onClick={() => {
                        const newCals = Math.min(3500, targetCalories + 50);
                        setTargetCalories(newCals);
                      }}
                      disabled={generating || targetCalories >= 3500}
                      className="w-7 h-7 rounded-lg bg-white dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 font-bold text-sm flex items-center justify-center transition border border-gray-200 dark:border-gray-600 disabled:opacity-40"
                    >
                      +
                    </button>
                  </div>

                  {/* Apply Custom Calories Button */}
                  <button
                    onClick={() => handleGenerate(targetCalories, dietPreference)}
                    disabled={generating}
                    className="px-3.5 py-1.5 text-xs bg-teal-600 hover:bg-teal-700 text-white rounded-xl font-bold transition shadow-sm flex items-center gap-1.5 disabled:opacity-50"
                  >
                    {generating ? (
                      <span className="material-symbols-outlined text-[14px] animate-spin">refresh</span>
                    ) : (
                      <span className="material-symbols-outlined text-[14px]">tune</span>
                    )}
                    <span>{isModified ? 'Update Plan' : 'Recalculate'}</span>
                  </button>

                  {/* Reset link if user modified it */}
                  {isModified && (
                    <button
                      type="button"
                      onClick={() => {
                        setTargetCalories(baseCalories);
                        handleGenerate(baseCalories, dietPreference);
                      }}
                      disabled={generating}
                      className="text-xs text-gray-500 hover:text-teal-600 dark:text-gray-400 dark:hover:text-teal-400 underline font-medium ml-1 transition"
                    >
                      Reset to {baseCalories} kcal
                    </button>
                  )}
                </div>
              );
            })()}
          </div>
        </div>
      </div>



      {/* Scientific Caloric Target Justification Card */}
      {dietPlan?.calorie_breakdown && (
        <div className="p-5 rounded-2xl bg-white dark:bg-gray-800 border border-teal-200/80 dark:border-teal-900/60 shadow-xs space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-gray-100 dark:border-gray-700">
            <div className="flex items-center gap-2 text-teal-900 dark:text-teal-100 font-bold text-sm">
              <span className="material-symbols-outlined text-[20px] text-teal-600 dark:text-teal-400">calculate</span>
              <span>
                {dietPlan.calorie_breakdown.anthropometrics_available === false
                  ? 'Calorie Target Estimation (Standard Population Baseline)'
                  : 'Scientific Calorie Target Justification'}
              </span>
            </div>
            <span className="text-[11px] text-gray-500 dark:text-gray-400">
              {dietPlan.calorie_breakdown.anthropometrics_available === false
                ? 'Standard Baseline (~1,800 kcal) • Unpersonalized'
                : 'Mifflin-St Jeor Clinical Standard'}
            </span>
          </div>

          {dietPlan.calorie_breakdown.anthropometrics_available === false ? (
            <div className="p-4 rounded-xl bg-amber-50/80 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 space-y-3">
              <div className="flex items-start gap-3">
                <span className="material-symbols-outlined text-amber-600 dark:text-amber-400 text-2xl shrink-0 mt-0.5">
                  info
                </span>
                <div className="space-y-1 text-xs text-amber-900 dark:text-amber-200">
                  <div className="font-bold text-sm text-amber-950 dark:text-amber-100">
                    Non-Personalized Caloric Baseline Applied (1,800 kcal/day)
                  </div>
                  <p>
                    Height, weight, and BMI were not documented in your uploaded lab report. Caloric targets and meal portions are currently estimated using standard population reference defaults with <strong>no report-derived personalization</strong>.
                  </p>
                  <p className="text-[11px] text-amber-800/80 dark:text-amber-300/80">
                    To receive an individualized Mifflin-St Jeor BMR calculation and tailored weight management deficit, please add your height and weight.
                  </p>
                </div>
              </div>
              <div className="pt-2 flex items-center justify-end">
                <a
                  href="/profile"
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold shadow-xs transition"
                >
                  <span className="material-symbols-outlined text-[16px]">edit_note</span>
                  <span>Enter Height & Weight in Profile</span>
                </a>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-750/70 border border-gray-100 dark:border-gray-700/60">
                <span className="text-[10px] uppercase font-bold text-gray-400 tracking-wider block">1. Basal Metabolic Rate (BMR)</span>
                <p className="text-base font-extrabold text-gray-900 dark:text-gray-100 mt-0.5">
                  {dietPlan.calorie_breakdown.bmr || dietPlan.calorie_breakdown.bmr_kcal} <span className="text-xs font-medium text-gray-400">kcal/d</span>
                </p>
                <span className="text-[10px] text-gray-500 dark:text-gray-400">Base resting burn</span>
              </div>

              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-750/70 border border-gray-100 dark:border-gray-700/60">
                <span className="text-[10px] uppercase font-bold text-gray-400 tracking-wider block">2. Maintenance (TDEE)</span>
                <p className="text-base font-extrabold text-gray-900 dark:text-gray-100 mt-0.5">
                  {dietPlan.calorie_breakdown.tdee || dietPlan.calorie_breakdown.tdee_kcal} <span className="text-xs font-medium text-gray-400">kcal/d</span>
                </p>
                <span className="text-[10px] text-gray-500 dark:text-gray-400">Light activity (1.30×)</span>
              </div>

              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-750/70 border border-gray-100 dark:border-gray-700/60">
                <span className="text-[10px] uppercase font-bold text-gray-400 tracking-wider block">3. Safe Clinical Deficit</span>
                <p className="text-base font-extrabold text-amber-600 dark:text-amber-400 mt-0.5">
                  -{dietPlan.calorie_breakdown.deficit != null ? dietPlan.calorie_breakdown.deficit : dietPlan.calorie_breakdown.clinical_deficit_kcal} <span className="text-xs font-medium text-gray-400">kcal/d</span>
                </p>
                <span className="text-[10px] text-gray-500 dark:text-gray-400">Weight management deficit</span>
              </div>

              <div className="p-3 rounded-xl bg-teal-50 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800/80">
                <span className="text-[10px] uppercase font-bold text-teal-700 dark:text-teal-300 tracking-wider block">4. Final Daily Target</span>
                <p className="text-base font-extrabold text-teal-800 dark:text-teal-200 mt-0.5">
                  {dietPlan.calorie_breakdown.recommended_calories || dietPlan.calorie_breakdown.recommended_target_kcal} <span className="text-xs font-medium text-teal-600 dark:text-teal-400">kcal/d</span>
                </p>
                <span className="text-[10px] text-teal-700 dark:text-teal-300">Same for Veg & Non-Veg</span>
              </div>
            </div>
          )}

          {dietPlan.calorie_breakdown.calculation_steps && dietPlan.calorie_breakdown.calculation_steps.length > 0 && (
            <div className="p-3 rounded-xl bg-gray-50/80 dark:bg-gray-750/40 text-xs text-gray-600 dark:text-gray-300 space-y-1">
              <span className="font-bold text-[11px] text-gray-700 dark:text-gray-200 block uppercase tracking-wider">
                Calculation Audit Trail:
              </span>
              {dietPlan.calorie_breakdown.calculation_steps.map((step: string, idx: number) => (
                <div key={idx} className="flex items-center gap-2 text-[11px]">
                  <span className="material-symbols-outlined text-[14px] text-teal-600 shrink-0">check_circle</span>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 rounded-xl text-red-600 text-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="text-center py-16 text-gray-500 animate-pulse">
          Loading your personalized Indian nutrition recommendations…
        </div>
      )}

      {/* Plan Display */}
      {!loading && days.length > 0 && activeDay && (
        <div className="space-y-6">
          {/* Day Navigation Tabs */}
          <div className="flex overflow-x-auto pb-1 gap-2 scrollbar-thin">
            {days.map((day: any, idx: number) => {
              const isActive = activeDayIdx === idx;
              return (
                <button
                  key={day.day_number || idx}
                  onClick={() => setActiveDayIdx(idx)}
                  className={`px-4 py-2.5 rounded-xl text-sm font-semibold transition whitespace-nowrap flex flex-col items-center flex-1 min-w-[110px] ${
                    isActive
                      ? 'bg-[#00685f] text-white shadow-md'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:border-teal-500'
                  }`}
                >
                  <span>Day {idx + 1}</span>
                  <span className={`text-[11px] font-normal ${isActive ? 'text-teal-100' : 'text-gray-400'}`}>
                    {day.day_name ? day.day_name.split(' ')[1]?.replace('(', '').replace(')', '') : `Day ${idx + 1}`}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Active Day Overview Card */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400">
                    Daily Nutrition Focus
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                    dietPreference === 'non_veg'
                      ? 'bg-rose-100 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300'
                      : 'bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300'
                  }`}>
                    {dietPreference === 'non_veg' ? '🍗 Non-Vegetarian Plan' : '🥗 Pure Vegetarian Plan'}
                  </span>
                </div>

                <h2 className="text-xl font-bold mt-1 text-gray-900 dark:text-gray-100" style={{ fontFamily: 'Inter, sans-serif' }}>
                  {activeDay.day_name || `Day ${activeDayIdx + 1}`}
                </h2>
                {activeDay.theme && (
                  <p className="text-xs text-gray-500 italic mt-0.5">Focus: {activeDay.theme}</p>
                )}
              </div>

              {/* Attainment progress pill */}
              <div className="text-right">
                <div className="text-2xl font-black text-teal-600 dark:text-teal-400">
                  {Number(activeDay.total_estimated_calories || targetCalories).toFixed(0)}
                  <span className="text-xs font-normal text-gray-400 ml-1">/ {targetCalories} kcal</span>
                </div>
                <div className="flex items-center space-x-1.5 mt-0.5 justify-end">
                  <div className="w-24 bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-teal-500 h-full rounded-full"
                      style={{
                        width: `${Math.min(
                          100,
                          ((activeDay.total_estimated_calories || targetCalories) / targetCalories) * 100
                        )}%`
                      }}
                    />
                  </div>
                  <span className="text-xs font-semibold text-teal-600">
                    {(((activeDay.total_estimated_calories || targetCalories) / targetCalories) * 100).toFixed(0)}%
                    attained
                  </span>
                </div>
              </div>
            </div>

            {/* Daily Macronutrients */}
            {activeDay.macronutrients && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                <div className="bg-teal-50/70 dark:bg-teal-950/30 border border-teal-100 dark:border-teal-900/50 rounded-xl p-3 text-center">
                  <p className="text-xs text-gray-500 font-medium">Carbohydrates</p>
                  <p className="text-lg font-bold text-teal-800 dark:text-teal-200 mt-0.5">
                    {Number(activeDay.macronutrients.carbohydrates_g || 0).toFixed(1)}g
                  </p>
                  <p className="text-[10px] text-gray-400">Complex whole grains</p>
                </div>
                <div className="bg-blue-50/70 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900/50 rounded-xl p-3 text-center">
                  <p className="text-xs text-gray-500 font-medium">Lean Protein</p>
                  <p className="text-lg font-bold text-blue-800 dark:text-blue-200 mt-0.5">
                    {Number(activeDay.macronutrients.protein_g || 0).toFixed(1)}g
                  </p>
                  <p className="text-[10px] text-gray-400">{dietPreference === 'non_veg' ? 'Poultry, fish & eggs' : 'Pulses & paneer'}</p>
                </div>
                <div className="bg-amber-50/70 dark:bg-amber-950/30 border border-amber-100 dark:border-amber-900/50 rounded-xl p-3 text-center">
                  <p className="text-xs text-gray-500 font-medium">Healthy Fats</p>
                  <p className="text-lg font-bold text-amber-800 dark:text-amber-200 mt-0.5">
                    {Number(activeDay.macronutrients.fats_g || 0).toFixed(1)}g
                  </p>
                  <p className="text-[10px] text-gray-400">MUFA & Omega-3</p>
                </div>
                <div className="bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-100 dark:border-emerald-900/50 rounded-xl p-3 text-center">
                  <p className="text-xs text-gray-500 font-medium">Dietary Fibre</p>
                  <p className="text-lg font-bold text-emerald-800 dark:text-emerald-200 mt-0.5">
                    {Number(activeDay.macronutrients.fibre_g || 0).toFixed(1)}g
                  </p>
                  <p className="text-[10px] text-gray-400">High satiety & low GI</p>
                </div>
              </div>
            )}

            {/* Clinical Restrictions & Tags */}
            {dietPlan.dietary_restrictions && dietPlan.dietary_restrictions.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {dietPlan.dietary_restrictions.map((r: string, i: number) => (
                  <span
                    key={i}
                    className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                  >
                    ✓ {r}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* 4 Daily Meal Cards */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold tracking-tight text-gray-900 dark:text-gray-100">
                Four Daily Meals — {activeDay.day_name || `Day ${activeDayIdx + 1}`}
              </h3>
              <span className="text-xs text-gray-500 font-medium">
                Showing {dietPreference === 'non_veg' ? 'Non-Vegetarian' : 'Vegetarian'} selections
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(activeDay.meals || []).map((meal: any, mIdx: number) => {
                const macros = meal.macronutrients || {};
                return (
                  <div
                    key={mIdx}
                    className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-5 flex flex-col justify-between space-y-3 hover:border-teal-400 transition"
                  >
                    <div>
                      {/* Slot Header */}
                      <div className="flex justify-between items-start gap-2">
                        <div className="flex items-start space-x-2.5">
                          <span className="text-2xl mt-0.5">{getSlotIcon(meal.slot || meal.meal_type)}</span>
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400">
                                {meal.slot || meal.meal_type || 'Meal'}
                              </span>
                              {renderDietBadge(meal)}
                            </div>
                            <h4 className="font-bold text-base leading-snug text-gray-900 dark:text-gray-100">
                              {meal.dish_name || 'Nutritious Meal'}
                            </h4>
                          </div>
                        </div>

                        <span className="px-2.5 py-1 bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 text-xs font-bold rounded-lg whitespace-nowrap shrink-0">
                          {Number(meal.calories || meal.total_calories || 0).toFixed(0)} kcal
                        </span>
                      </div>

                      {/* Portion recommendation */}
                      {meal.portion && (
                        <p className="text-xs text-gray-600 dark:text-gray-300 mt-2 bg-gray-50 dark:bg-gray-750 p-2.5 rounded-xl border border-gray-100 dark:border-gray-700">
                          <span className="font-semibold text-gray-900 dark:text-gray-100">Portion:</span>{' '}
                          {meal.portion}
                        </p>
                      )}

                      {/* Macros pill strip */}
                      <div className="flex flex-wrap gap-1.5 mt-2.5">
                        {macros.carbohydrates_g != null && (
                          <span className="text-[10px] px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-gray-600 dark:text-gray-300">
                            Carbs: {Number(macros.carbohydrates_g).toFixed(0)}g
                          </span>
                        )}
                        {macros.protein_g != null && (
                          <span className="text-[10px] px-2 py-0.5 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded font-medium">
                            Protein: {Number(macros.protein_g).toFixed(0)}g
                          </span>
                        )}
                        {macros.fats_g != null && (
                          <span className="text-[10px] px-2 py-0.5 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded">
                            Fats: {Number(macros.fats_g).toFixed(0)}g
                          </span>
                        )}
                        {macros.fibre_g != null && (
                          <span className="text-[10px] px-2 py-0.5 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded font-medium">
                            Fibre: {Number(macros.fibre_g).toFixed(0)}g
                          </span>
                        )}
                      </div>

                      {/* Clinical Benefit Tag */}
                      {meal.clinical_benefit && (
                        <p className="text-xs text-teal-800 dark:text-teal-300 mt-2.5 leading-relaxed bg-teal-50/60 dark:bg-teal-950/30 p-2.5 rounded-xl border border-teal-100 dark:border-teal-900/40">
                          🌱 <span className="font-medium">{meal.clinical_benefit}</span>
                        </p>
                      )}
                    </div>

                    {/* Healthy Indian Substitutions */}
                    {meal.substitutions && meal.substitutions.length > 0 && (
                      <div className="pt-2.5 border-t border-gray-100 dark:border-gray-700">
                        <p className="text-[11px] font-bold text-gray-500 uppercase tracking-wide mb-1.5">
                          💡 Alternative Swaps:
                        </p>
                        <div className="space-y-1.5">
                          {meal.substitutions.map((sub: any, sIdx: number) => (
                            <div
                              key={sIdx}
                              className="text-xs flex justify-between items-center text-gray-700 dark:text-gray-300 bg-gray-50/80 dark:bg-gray-700/50 px-2.5 py-1.5 rounded-lg gap-2"
                            >
                              <div className="flex items-center gap-1.5 overflow-hidden">
                                {renderDietBadge(sub)}
                                <span className="font-medium truncate">
                                  {sub.dish_name || sub.substitute || sub}
                                </span>
                              </div>
                              {sub.calories && (
                                <span className="text-[10px] text-gray-400 shrink-0">
                                  {Number(sub.calories).toFixed(0)} kcal
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && days.length === 0 && (
        <div className="text-center py-16 bg-white dark:bg-gray-800 rounded-2xl p-8 border border-gray-100 dark:border-gray-700 space-y-4">
          <div className="text-6xl">🍛</div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">No Diet Plan Generated Yet</h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm max-w-md mx-auto">
            Choose your dietary preference and click below to generate an authentic 7-day Indian meal plan customized to your target calories and health profile.
          </p>

          <div className="flex justify-center pt-2">
            <div className="inline-flex rounded-xl bg-gray-100 dark:bg-gray-700 p-1">
              <button
                type="button"
                onClick={() => setDietPreference('veg')}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition ${
                  dietPreference === 'veg' ? 'bg-emerald-600 text-white shadow-sm' : 'text-gray-600 dark:text-gray-300'
                }`}
              >
                🥗 Pure Vegetarian
              </button>
              <button
                type="button"
                onClick={() => setDietPreference('non_veg')}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition ${
                  dietPreference === 'non_veg' ? 'bg-rose-600 text-white shadow-sm' : 'text-gray-600 dark:text-gray-300'
                }`}
              >
                🍗 Non-Vegetarian
              </button>
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={() => handleGenerate(targetCalories, dietPreference)}
              disabled={generating}
              className="px-6 py-3 bg-[#00685f] hover:bg-[#005049] text-white font-semibold rounded-xl shadow transition"
            >
              {generating ? 'Generating…' : `✨ Generate My 7-Day Indian ${dietPreference === 'non_veg' ? 'Non-Veg' : 'Veg'} Plan`}
            </button>
          </div>
        </div>
      )}

      {/* Exercise Plan Section */}
      {exercisePlan && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">🏃</span>
                <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100" style={{ fontFamily: 'Inter, sans-serif' }}>
                  Prescribed Exercise Regimen
                </h2>
              </div>
              {exercisePlan.summary_note && (
                <p className="text-xs text-gray-500 mt-1">{exercisePlan.summary_note}</p>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <span
                className={`px-3 py-1 rounded-full text-xs font-bold ${
                  exercisePlan.intensity === 'Low'
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300'
                    : exercisePlan.intensity === 'Moderate'
                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300'
                    : 'bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300'
                }`}
              >
                {exercisePlan.intensity} Intensity
              </span>
              <span className="text-xs text-gray-500 font-semibold bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
                Target: {exercisePlan.weekly_target_minutes || 150} min/week
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            {(exercisePlan.routines || []).map((routine: any, rIdx: number) => (
              <div
                key={rIdx}
                className="border-l-4 border-teal-500 bg-gray-50 dark:bg-gray-750 p-4 rounded-r-xl space-y-2"
              >
                <div className="flex justify-between items-start">
                  <span className="text-xs font-bold text-teal-600 dark:text-teal-400">
                    {routine.day || `Routine ${rIdx + 1}`}
                  </span>
                  <span className="text-xs text-gray-500 font-semibold">
                    {routine.duration_minutes || 30} mins
                  </span>
                </div>
                <h4 className="font-bold text-sm text-gray-900 dark:text-gray-100">
                  {routine.title || 'Conditioning Session'}
                </h4>
                {routine.exercises && (
                  <div className="text-xs text-gray-500 space-y-0.5">
                    {(Array.isArray(routine.exercises) ? routine.exercises : [routine.exercises]).map(
                      (ex: string, eIdx: number) => (
                        <p key={eIdx}>• {ex}</p>
                      )
                    )}
                  </div>
                )}
                {routine.target_benefit && (
                  <p className="text-[11px] text-teal-700 dark:text-teal-300 font-medium pt-1">
                    ✓ {routine.target_benefit}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
