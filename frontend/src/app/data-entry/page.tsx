'use client';

import { useState } from 'react';
import { api } from '@/lib/api';
import { useRouter } from 'next/navigation';

const defaultForm = {
  age: '', gender: 'Male', height_cm: '', weight_kg: '',
  blood_glucose_level: '', hba1c_level: '', systolic_bp: '', diastolic_bp: '',
  cholesterol: '', hdl: '', ldl: '', triglycerides: '',
  smoking_history: 'never', heart_disease_history: 0, physical_activity_days: 3,
};

export default function DataEntryPage() {
  const [form, setForm] = useState<any>(defaultForm);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const router = useRouter();

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(null), 3500); };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    if (name === 'physical_activity_days') {
      if (value === '') {
        setForm({ ...form, [name]: '' });
      } else {
        const num = Math.min(7, Math.max(0, parseInt(value, 10) || 0));
        setForm({ ...form, [name]: num });
      }
      return;
    }
    setForm({ ...form, [name]: value });
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploadLoading(true);
    try {
      const formData = new FormData();
      formData.append('report', file);
      const baseUrl = (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
        ? 'http://localhost:5000/api'
        : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api');
      const res = await fetch(`${baseUrl}/reports/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${localStorage.getItem('token') || ''}` },
        body: formData,
      });
      const responseText = await res.text();
      let data: any = {};
      try {
        data = JSON.parse(responseText);
      } catch (parseErr) {
        if (!res.ok) {
          throw new Error(`Server returned error ${res.status} (${res.statusText || 'Timeout'}). Please try again.`);
        }
      }
      if (!res.ok) throw new Error(data.error || data.details || 'Upload failed');
      if (data.extracted_metrics) {
        const m = data.extracted_metrics;
        setForm((prev: any) => ({
          ...prev,
          blood_glucose_level: m.glucose || prev.blood_glucose_level,
          systolic_bp: m.systolic_bp || prev.systolic_bp,
          diastolic_bp: m.diastolic_bp || prev.diastolic_bp,
          cholesterol: m.cholesterol || prev.cholesterol,
          hdl: m.hdl || prev.hdl,
          ldl: m.ldl || prev.ldl,
          triglycerides: m.triglycerides || prev.triglycerides,
          age: m.age || prev.age,
          gender: m.gender || prev.gender,
        }));
        showToast('✅ Lab report parsed! Fields auto-filled from OCR.');
      }
    } catch (e: any) {
      setError('Report upload failed: ' + e.message);
    } finally {
      setUploadLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload: any = { ...form };
      Object.keys(payload).forEach(k => { if (payload[k] === '') delete payload[k]; });
      const res = await api.runPrediction(payload);
      setResult(res.data);
      showToast('✅ Risk prediction complete!');
    } catch (e: any) {
      setError(e.response?.data?.error || e.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 p-4">
      {toast && (
        <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg z-50 transition-all">
          {toast}
        </div>
      )}
      <h1 className="text-2xl font-bold" style={{ fontFamily: 'Inter, sans-serif' }}>Enter Health Data</h1>

      {/* OCR Lab Report Upload */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-4 border border-dashed border-gray-300 dark:border-gray-600">
        <h2 className="text-lg font-semibold mb-2">📄 Upload Lab Report (PDF / Image)</h2>
        <p className="text-sm text-gray-500 mb-3">OCR will auto-fill the form fields below.</p>
        <div className="flex items-center space-x-3">
          <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={e => setFile(e.target.files?.[0] || null)}
            className="text-sm" id="report-file" />
          <button onClick={handleUpload} disabled={!file || uploadLoading}
            className="px-4 py-2 bg-teal-600 text-white rounded hover:bg-teal-700 transition disabled:opacity-50">
            {uploadLoading ? 'Parsing…' : 'Parse Report'}
          </button>
        </div>
      </div>

      {/* Vitals Form */}
      <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-4">
        <h2 className="text-lg font-semibold">🩺 Vitals & Lifestyle</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { label: 'Age', name: 'age', type: 'number', placeholder: 'e.g. 45' },
            { label: 'Height (cm)', name: 'height_cm', type: 'number', placeholder: 'e.g. 172' },
            { label: 'Weight (kg)', name: 'weight_kg', type: 'number', placeholder: 'e.g. 70' },
          ].map(f => (
            <div key={f.name}>
              <label className="block text-sm font-medium mb-1" htmlFor={f.name}>{f.label}</label>
              <input id={f.name} name={f.name} type={f.type} placeholder={f.placeholder}
                value={form[f.name]} onChange={handleChange}
                className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600" />
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1" htmlFor="gender">Gender</label>
            <select id="gender" name="gender" value={form.gender} onChange={handleChange}
              className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600">
              <option>Male</option><option>Female</option><option>Other</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1" htmlFor="smoking_history">Smoking History</label>
            <select id="smoking_history" name="smoking_history" value={form.smoking_history} onChange={handleChange}
              className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600">
              <option value="never">Never</option>
              <option value="former">Former</option>
              <option value="current">Current</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { label: 'Fasting Glucose (mg/dL)', name: 'blood_glucose_level', placeholder: 'e.g. 98' },
            { label: 'HbA1c (%)', name: 'hba1c_level', placeholder: 'e.g. 5.6' },
            { label: 'Systolic BP (mmHg)', name: 'systolic_bp', placeholder: 'e.g. 120' },
            { label: 'Diastolic BP (mmHg)', name: 'diastolic_bp', placeholder: 'e.g. 80' },
            { label: 'Cholesterol (mg/dL)', name: 'cholesterol', placeholder: 'e.g. 185' },
            { label: 'HDL (mg/dL)', name: 'hdl', placeholder: 'e.g. 55' },
            { label: 'LDL (mg/dL)', name: 'ldl', placeholder: 'e.g. 110' },
            { label: 'Triglycerides (mg/dL)', name: 'triglycerides', placeholder: 'e.g. 140' },
            { label: 'Active days/week (0–7)', name: 'physical_activity_days', placeholder: '0–7', min: 0, max: 7 },
          ].map((f: any) => (
            <div key={f.name}>
              <label className="block text-sm font-medium mb-1" htmlFor={f.name}>{f.label}</label>
              <input id={f.name} name={f.name} type="number" placeholder={f.placeholder}
                min={f.min} max={f.max}
                value={form[f.name]} onChange={handleChange}
                className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600" />
            </div>
          ))}
        </div>

        {error && <p className="text-red-500 text-sm">{error}</p>}

        <button type="submit" disabled={loading}
          className="w-full py-3 bg-teal-600 text-white font-semibold rounded hover:bg-teal-700 transition disabled:opacity-50">
          {loading ? 'Running AI Prediction…' : '🔍 Run Risk Prediction'}
        </button>
      </form>

      {/* Results */}
      {result && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">📊 AI Health Risk Assessment</h2>
            <span className="text-xs bg-teal-50 dark:bg-teal-950/40 text-teal-800 dark:text-teal-300 font-medium px-2.5 py-1 rounded-md border border-teal-200 dark:border-teal-800">
              Vitality Score: {result.vitality_score} / 100
            </span>
          </div>

          <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-xl text-xs text-amber-800 dark:text-amber-300">
            ℹ️ <strong>Notice:</strong> Risk scores are machine learning statistical indexes out of 100 based on population training data, not absolute medical disease probabilities or clinical diagnoses.
          </div>

          {Object.entries(result.predictions || {}).map(([key, pred]: any) => (
            <div key={key} className="border rounded-xl p-4 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold capitalize text-base">{pred.disease_type} Risk Score</h3>
                <span className="text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
                  {pred.category}
                </span>
              </div>
              <p className="text-sm mt-1 text-gray-600 dark:text-gray-400">
                Model Score: <span className="font-bold text-gray-900 dark:text-gray-100 text-lg">{pred.risk_score?.toFixed(0)}</span> <span className="text-xs">/ 100</span>
              </p>
              <div className="mt-3 space-y-1.5 pt-2 border-t border-gray-100 dark:border-gray-700">
                {(pred.explanation?.risk_increasing_factors || []).slice(0, 3).map((f: any, i: number) => (
                  <p key={i} className="text-xs text-rose-600 dark:text-rose-400 font-medium flex items-center gap-1.5">
                    <span className="shrink-0 font-bold">▲</span>
                    <span><strong>{f.label || f.feature}:</strong> {f.tip || f.plain_english || f.formatted_value}</span>
                  </p>
                ))}
                {(pred.explanation?.protective_factors || []).slice(0, 2).map((f: any, i: number) => (
                  <p key={i} className="text-xs text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1.5">
                    <span className="shrink-0 font-bold">▼</span>
                    <span><strong>{f.label || f.feature}:</strong> {f.tip || f.plain_english || f.formatted_value}</span>
                  </p>
                ))}
              </div>
            </div>
          ))}
          <div className="flex space-x-3 mt-2">
            <button onClick={() => router.push('/recommendations')}
              className="px-4 py-2 bg-teal-600 text-white rounded hover:bg-teal-700 transition">
              Generate Recommendations →
            </button>
            <button onClick={() => router.push('/dashboard')}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition">
              View Dashboard →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
