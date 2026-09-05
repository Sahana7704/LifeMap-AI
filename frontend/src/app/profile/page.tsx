'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import Link from 'next/link';
import AuthCard from '@/components/AuthCard';

export default function ProfilePage() {
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState<any>({});
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const fetchProfile = async () => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (!token) {
        setIsLoggedIn(false);
        setLoading(false);
        return;
      }
    }

    try {
      const res = await api.getProfile();
      setProfile(res.data);
      setFormData({
        name: res.data.name || '',
        age: res.data.age != null ? res.data.age : '',
        gender: res.data.gender || 'Male',
        height: res.data.height != null ? res.data.height : '',
        weight: res.data.weight != null ? res.data.weight : '',
        diet_preference: res.data.diet_preference || 'veg',
      });
    } catch (e: any) {
      if (e.response?.status === 401) {
        setIsLoggedIn(false);
      } else {
        setError(e.response?.data?.error || e.message || 'Failed to load profile');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await api.updateProfile(formData);
      setProfile(res.data.profile);
      setEditMode(false);
      showToast('✅ Profile updated successfully!');
    } catch (e: any) {
      if (e.response?.status === 401) {
        setIsLoggedIn(false);
      } else {
        setError(e.response?.data?.error || e.message || 'Failed to update profile');
      }
    }
  };

  if (!isLoggedIn) {
    return (
      <AuthCard
        title="Please Log In to View Your Profile"
        description="Log in to view and update your personal health profile, body measurements, and dietary preferences."
      />
    );
  }

  if (loading) return <div className="flex items-center justify-center h-64"><span className="text-gray-500 animate-pulse">Loading profile…</span></div>;

  if (error) {
    return (
      <div className="max-w-md mx-auto my-12 p-6 rounded-2xl bg-white dark:bg-gray-800 border border-red-200 dark:border-red-900/60 shadow-sm text-center space-y-4">
        <div className="text-3xl">⚠️</div>
        <p className="text-red-600 dark:text-red-400 text-sm font-medium">{error}</p>
        <button
          onClick={() => { setError(null); setLoading(true); fetchProfile(); }}
          className="px-4 py-2 bg-teal-600 text-white text-xs font-semibold rounded-xl hover:bg-teal-700 transition"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6 p-4">
      {toast && (
        <div className="fixed top-4 right-4 bg-teal-600 text-white px-4 py-2 rounded-xl shadow-lg z-50 animate-bounce text-sm font-medium">
          {toast}
        </div>
      )}

      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Inter, sans-serif' }}>My Profile</h1>
        {!editMode && (
          <button
            onClick={() => setEditMode(true)}
            className="px-4 py-2 bg-teal-600 text-white rounded-xl text-xs font-semibold hover:bg-teal-700 transition"
          >
            Edit Profile
          </button>
        )}
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
        {editMode ? (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300" htmlFor="name">Full Name</label>
              <input id="name" name="name" type="text" value={formData.name || ''} onChange={handleChange} className="w-full p-2.5 text-sm border rounded-xl dark:bg-gray-750 dark:border-gray-600" required />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300" htmlFor="age">Age</label>
                <input id="age" name="age" type="number" placeholder="e.g. 42" value={formData.age || ''} onChange={handleChange} className="w-full p-2.5 text-sm border rounded-xl dark:bg-gray-750 dark:border-gray-600" min={1} max={120} />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300" htmlFor="gender">Gender</label>
                <select id="gender" name="gender" value={formData.gender || 'Male'} onChange={handleChange} className="w-full p-2.5 text-sm border rounded-xl dark:bg-gray-750 dark:border-gray-600">
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300" htmlFor="height">Height (cm)</label>
                <input id="height" name="height" type="number" placeholder="e.g. 172" value={formData.height || ''} onChange={handleChange} className="w-full p-2.5 text-sm border rounded-xl dark:bg-gray-750 dark:border-gray-600" min={50} max={250} step="0.1" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300" htmlFor="weight">Weight (kg)</label>
                <input id="weight" name="weight" type="number" placeholder="e.g. 70" value={formData.weight || ''} onChange={handleChange} className="w-full p-2.5 text-sm border rounded-xl dark:bg-gray-750 dark:border-gray-600" min={20} max={300} step="0.1" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300" htmlFor="diet_preference">
                Dietary Preference (Default for Meal Plans)
              </label>
              <select id="diet_preference" name="diet_preference" value={formData.diet_preference || 'veg'} onChange={handleChange} className="w-full p-2.5 text-sm border rounded-xl dark:bg-gray-750 dark:border-gray-600">
                <option value="veg">🥗 Pure Vegetarian (Lacto-Vegetarian, Paneer, Lentils)</option>
                <option value="non_veg">🍗 Non-Vegetarian (Eggs, Lean Poultry & Omega-3 Fish)</option>
              </select>
            </div>

            <div className="flex space-x-3 pt-2">
              <button type="submit" className="px-5 py-2.5 bg-teal-600 text-white rounded-xl text-xs font-bold hover:bg-teal-700 transition">Save Changes</button>
              <button type="button" onClick={() => setEditMode(false)} className="px-5 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-xl text-xs font-semibold hover:bg-gray-300 transition">Cancel</button>
            </div>
          </form>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 pb-2 border-b border-gray-100 dark:border-gray-700">
              <div>
                <span className="text-xs text-gray-400 font-medium block">Full Name</span>
                <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{profile.name || 'Not provided'}</span>
              </div>
              <div>
                <span className="text-xs text-gray-400 font-medium block">Email</span>
                <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{profile.email || '—'}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pb-2 border-b border-gray-100 dark:border-gray-700">
              <div>
                <span className="text-xs text-gray-400 font-medium block">Age</span>
                <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{profile.age != null ? `${profile.age} yrs` : 'Not set'}</span>
              </div>
              <div>
                <span className="text-xs text-gray-400 font-medium block">Gender</span>
                <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{profile.gender || 'Not set'}</span>
              </div>
              <div>
                <span className="text-xs text-gray-400 font-medium block">Height</span>
                <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{profile.height != null ? `${profile.height} cm` : 'Not set'}</span>
              </div>
              <div>
                <span className="text-xs text-gray-400 font-medium block">Weight</span>
                <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{profile.weight != null ? `${profile.weight} kg` : 'Not set'}</span>
              </div>
            </div>

            <div>
              <span className="text-xs text-gray-400 font-medium block mb-1">Dietary Preference</span>
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                profile.diet_preference === 'non_veg'
                  ? 'bg-rose-100 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300'
                  : 'bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300'
              }`}>
                {profile.diet_preference === 'non_veg' ? '🍗 Non-Vegetarian (Eggs, Chicken, Fish)' : '🥗 Pure Vegetarian (Plant & Dairy)'}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-4">
        <Link href="/dashboard" className="text-xs text-teal-600 hover:underline flex items-center gap-1 font-semibold">
          <span>← Back to Dashboard</span>
        </Link>
        <Link href="/recommendations" className="text-xs text-teal-600 hover:underline flex items-center gap-1 font-semibold">
          <span>View Nutrition Plan →</span>
        </Link>
      </div>
    </div>
  );
}
