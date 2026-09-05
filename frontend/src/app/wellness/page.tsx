'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

export default function WellnessPage() {
  const [logs, setLogs] = useState<any[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [form, setForm] = useState({ glucose: '', systolic_bp: '', diastolic_bp: '', bmi: '', notes: '' });

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(null), 3500); };

  const fetchData = async () => {
    try {
      const [logsRes, trendsRes] = await Promise.all([api.getLogs(), api.getTrends()]);
      setLogs(logsRes.data);
      setTrends(trendsRes.data.map((t: any) => ({
        ...t,
        date: new Date(t.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }),
      })));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const metrics: any = {};
      if (form.glucose) metrics.glucose = Number(form.glucose);
      if (form.systolic_bp) metrics.systolic_bp = Number(form.systolic_bp);
      if (form.diastolic_bp) metrics.diastolic_bp = Number(form.diastolic_bp);
      if (form.bmi) metrics.bmi = Number(form.bmi);

      await api.addLog({
        date: selectedDate.toISOString().split('T')[0],
        metrics,
        notes: form.notes,
      });
      showToast('✅ Wellness log saved!');
      setForm({ glucose: '', systolic_bp: '', diastolic_bp: '', bmi: '', notes: '' });
      await fetchData();
    } catch (e: any) {
      setError(e.response?.data?.error || e.message || 'Failed to save log');
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 space-y-6">
      {toast && (
        <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg z-50">{toast}</div>
      )}
      <h1 className="text-2xl font-bold" style={{ fontFamily: 'Inter, sans-serif' }}>Wellness Log</h1>

      {/* Log Entry Form */}
      <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-4">
        <h2 className="text-lg font-semibold">📝 Log Today's Metrics</h2>

        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium">Date:</label>
          <DatePicker
            selected={selectedDate}
            onChange={(date: Date | null) => date && setSelectedDate(date)}
            maxDate={new Date()}
            className="p-2 border rounded dark:bg-gray-700 dark:border-gray-600 text-sm"
            dateFormat="dd MMM yyyy"
          />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'Glucose (mg/dL)', name: 'glucose', placeholder: 'e.g. 98' },
            { label: 'Systolic BP', name: 'systolic_bp', placeholder: 'e.g. 120' },
            { label: 'Diastolic BP', name: 'diastolic_bp', placeholder: 'e.g. 80' },
            { label: 'BMI', name: 'bmi', placeholder: 'e.g. 23.5' },
          ].map(f => (
            <div key={f.name}>
              <label className="block text-xs font-medium mb-1" htmlFor={f.name}>{f.label}</label>
              <input id={f.name} name={f.name} type="number" step="0.1" placeholder={f.placeholder}
                value={(form as any)[f.name]} onChange={handleChange}
                className="w-full p-2 border rounded text-sm dark:bg-gray-700 dark:border-gray-600" />
            </div>
          ))}
        </div>

        <div>
          <label className="block text-sm font-medium mb-1" htmlFor="notes">Notes (optional)</label>
          <textarea id="notes" name="notes" rows={2} placeholder="e.g. Felt tired today, skipped evening walk…"
            value={form.notes} onChange={handleChange}
            className="w-full p-2 border rounded text-sm dark:bg-gray-700 dark:border-gray-600" />
        </div>

        {error && <p className="text-red-500 text-sm">{error}</p>}

        <button type="submit" className="px-4 py-2 bg-teal-600 text-white rounded hover:bg-teal-700 transition">
          Save Log Entry
        </button>
      </form>

      {/* Trend Charts */}
      {trends.length > 1 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-6">
          <h2 className="text-lg font-semibold">📈 Health Trends</h2>

          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-500">Blood Glucose (mg/dL)</h3>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="glucose" stroke="#f59e0b" strokeWidth={2} dot={false} name="Glucose" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-500">Blood Pressure (mmHg)</h3>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="systolic_bp" stroke="#ef4444" strokeWidth={2} dot={false} name="Systolic" />
                <Line type="monotone" dataKey="diastolic_bp" stroke="#3b82f6" strokeWidth={2} dot={false} name="Diastolic" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-500">Risk Scores (%)</h3>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="diabetes_risk" stroke="#a855f7" strokeWidth={2} dot={false} name="Diabetes Risk" />
                <Line type="monotone" dataKey="cvd_risk" stroke="#ec4899" strokeWidth={2} dot={false} name="CVD Risk" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Log History */}
      {!loading && logs.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold mb-3">📋 Log History</h2>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {logs.map((log: any) => {
              const m = typeof log.metrics === 'string' ? JSON.parse(log.metrics || '{}') : (log.metrics || {});
              return (
                <div key={log.log_id} className="flex justify-between items-start border-b dark:border-gray-700 pb-2">
                  <div>
                    <p className="text-sm font-medium">{new Date(log.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}</p>
                    <p className="text-xs text-gray-500">
                      {m.glucose && `Glucose: ${m.glucose} · `}
                      {m.systolic_bp && `BP: ${m.systolic_bp}/${m.diastolic_bp} · `}
                      {m.bmi && `BMI: ${m.bmi}`}
                    </p>
                    {log.notes && <p className="text-xs italic text-gray-400">{log.notes}</p>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
