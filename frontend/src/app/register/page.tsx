'use client';

import { useState } from 'react';
import { api } from '@/lib/api';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function RegisterPage() {
  const [form, setForm] = useState({ name: '', email: '', password: '', confirmPassword: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [consentChecked, setConsentChecked] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // Compute live password strength
  const getPasswordStrength = (pwd: string) => {
    if (!pwd) return { score: 0, label: 'None', color: 'bg-gray-200' };
    let score = 0;
    if (pwd.length >= 8) score += 1;
    if (pwd.length >= 12) score += 1;
    if (/[A-Z]/.test(pwd) && /[a-z]/.test(pwd)) score += 1;
    if (/[0-9]/.test(pwd)) score += 1;
    if (/[^A-Za-z0-9]/.test(pwd)) score += 1;

    if (score <= 1) return { score: 20, label: 'Weak (min 8 chars needed)', color: 'bg-rose-500', text: 'text-rose-600' };
    if (score === 2) return { score: 50, label: 'Fair (add numbers & capitals)', color: 'bg-amber-500', text: 'text-amber-600' };
    if (score === 3 || score === 4) return { score: 80, label: 'Good (Healthcare grade)', color: 'bg-teal-500', text: 'text-teal-600' };
    return { score: 100, label: 'Very Strong (Maximum protection)', color: 'bg-emerald-600', text: 'text-emerald-600' };
  };

  const strength = getPasswordStrength(form.password);
  const passwordsMatch = form.password && form.confirmPassword && form.password === form.confirmPassword;
  const passwordMismatch = form.confirmPassword && form.password !== form.confirmPassword;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (form.password.length < 8) {
      setError('For patient health record security, your password must be at least 8 characters long.');
      return;
    }

    if (!/[0-9]/.test(form.password) || !/[A-Za-z]/.test(form.password)) {
      setError('Password must contain both letters and at least one number.');
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError('The passwords entered do not match. Please re-check.');
      return;
    }

    if (!consentChecked) {
      setError('Please acknowledge the health data encryption and privacy statement.');
      return;
    }

    setLoading(true);

    try {
      const res = await api.register({
        name: form.name.trim(),
        email: form.email.trim(),
        password: form.password
      });
      const { token, user } = res.data;
      if (typeof window !== 'undefined') {
        localStorage.setItem('token', token);
        if (user) localStorage.setItem('user', JSON.stringify(user));
        window.dispatchEvent(new Event('auth-change'));
      }
      router.push('/dashboard');
    } catch (e: any) {
      setError(e.response?.data?.error || e.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-teal-50 via-gray-50 to-emerald-50 dark:from-gray-950 dark:via-gray-900 dark:to-teal-950/40 relative overflow-hidden">
      {/* Background aesthetic blobs */}
      <div className="absolute -top-24 -left-24 w-96 h-96 bg-teal-300/20 dark:bg-teal-700/10 rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-emerald-300/20 dark:bg-emerald-700/10 rounded-full blur-3xl pointer-events-none"></div>

      <div className="w-full max-w-lg space-y-6 relative z-10">
        {/* Header Branding */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-100 dark:bg-teal-900/60 text-teal-800 dark:text-teal-200 text-xs font-semibold">
            <span className="material-symbols-outlined text-[14px] text-teal-600">verified_user</span>
            <span>Personalized Health Portal</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <span className="text-2xl font-black tracking-tight text-[#00685f] dark:text-[#89f5e7]">
              LifeMap
            </span>
            <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-teal-600 text-white uppercase">
              AI
            </span>
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight text-gray-900 dark:text-gray-100">
            Create Your Health Profile
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Accurate preventive health risk assessment and personalized nutrition planning.
          </p>
        </div>

        {/* Form Card */}
        <div className="bg-white dark:bg-gray-800/90 backdrop-blur-md p-8 rounded-3xl shadow-xl border border-gray-100 dark:border-gray-700 space-y-5">
          {error && (
            <div className="p-3.5 bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900/60 rounded-xl text-rose-700 dark:text-rose-300 text-xs flex items-start gap-2.5">
              <span className="material-symbols-outlined text-[18px] shrink-0 mt-0.5">error</span>
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Full Name */}
            <div>
              <label htmlFor="name" className="block text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 mb-1.5">
                Full Name / Patient Name
              </label>
              <div className="relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                  <span className="material-symbols-outlined text-[18px]">person</span>
                </div>
                <input
                  id="name"
                  name="name"
                  type="text"
                  required
                  placeholder="Dr. / Mr. / Ms. Jane Doe"
                  value={form.name}
                  onChange={handleChange}
                  className="block w-full pl-10 pr-3 py-2.5 text-sm rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-750 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition"
                />
              </div>
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 mb-1.5">
                Email Address (Used for Report Delivery)
              </label>
              <div className="relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                  <span className="material-symbols-outlined text-[18px]">mail</span>
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  required
                  placeholder="jane.doe@example.com"
                  value={form.email}
                  onChange={handleChange}
                  className="block w-full pl-10 pr-3 py-2.5 text-sm rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-750 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="password" className="block text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300">
                  Account Password (Min 8 Characters)
                </label>
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="text-xs text-teal-600 dark:text-teal-400 hover:underline font-medium flex items-center gap-1"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {showPassword ? 'visibility_off' : 'visibility'}
                  </span>
                  <span>{showPassword ? 'Hide' : 'Show'}</span>
                </button>
              </div>
              <div className="relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                  <span className="material-symbols-outlined text-[18px]">lock</span>
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  required
                  placeholder="••••••••"
                  value={form.password}
                  onChange={handleChange}
                  className="block w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-750 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition"
                />
              </div>

              {/* Password Strength Indicator */}
              {form.password && (
                <div className="mt-2 space-y-1">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-gray-500">Password Security:</span>
                    <span className={`font-bold ${strength.text}`}>{strength.label}</span>
                  </div>
                  <div className="w-full bg-gray-100 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${strength.color}`}
                      style={{ width: `${strength.score}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="confirmPassword" className="block text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300">
                  Confirm Password
                </label>
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="text-xs text-teal-600 dark:text-teal-400 hover:underline font-medium flex items-center gap-1"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {showConfirmPassword ? 'visibility_off' : 'visibility'}
                  </span>
                  <span>{showConfirmPassword ? 'Hide' : 'Show'}</span>
                </button>
              </div>
              <div className="relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                  <span className="material-symbols-outlined text-[18px]">verified</span>
                </div>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  required
                  placeholder="••••••••"
                  value={form.confirmPassword}
                  onChange={handleChange}
                  className={`block w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border bg-gray-50/50 dark:bg-gray-750 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 transition ${
                    passwordsMatch
                      ? 'border-emerald-500 focus:ring-emerald-500'
                      : passwordMismatch
                      ? 'border-rose-400 focus:ring-rose-500'
                      : 'border-gray-200 dark:border-gray-700 focus:ring-teal-500'
                  }`}
                />
                {passwordsMatch && (
                  <div className="absolute inset-y-0 right-0 pr-3.5 flex items-center pointer-events-none text-emerald-500">
                    <span className="material-symbols-outlined text-[18px]">check_circle</span>
                  </div>
                )}
              </div>
              {passwordMismatch && (
                <p className="text-[11px] text-rose-500 mt-1">Passwords do not match.</p>
              )}
            </div>

            {/* Health Data Privacy & Encryption Consent */}
            <div className="pt-2">
              <label className="flex items-start gap-2.5 cursor-pointer text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                <input
                  type="checkbox"
                  checked={consentChecked}
                  onChange={e => setConsentChecked(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-gray-300"
                />
                <span>
                  I agree to terms of service and private health wellness analysis.
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={Boolean(loading || passwordMismatch)}
              className="w-full py-3 px-4 rounded-xl bg-[#00685f] hover:bg-[#005049] text-white font-bold text-sm shadow-md hover:shadow-lg disabled:opacity-50 transition active:scale-[0.99] flex items-center justify-center space-x-2 cursor-pointer"
            >
              {loading ? (
                <>
                  <span className="animate-spin text-[16px]">⏳</span>
                  <span>Creating Account…</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-[18px]">person_add</span>
                  <span>Create Account</span>
                </>
              )}
            </button>
          </form>

          {/* Clean Subtext */}
          <div className="pt-3 border-t border-gray-100 dark:border-gray-700/80 text-center text-[11px] text-gray-400">
            <span>Private & confidential health tracking</span>
          </div>
        </div>

        {/* Link to Login */}
        <p className="text-center text-xs text-gray-500 dark:text-gray-400">
          Already have a LifeMap AI account?{' '}
          <Link href="/login" className="font-bold text-[#00685f] dark:text-[#89f5e7] hover:underline">
            Sign In →
          </Link>
        </p>
      </div>
    </div>
  );
}
