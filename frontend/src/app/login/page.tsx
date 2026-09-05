'use client';

import { useState } from 'react';
import { api } from '@/lib/api';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const performLogin = async (loginEmail: string, loginPassword: string) => {
    setError(null);
    setLoading(true);

    try {
      const res = await api.login({ email: loginEmail.trim(), password: loginPassword });
      const { token, user } = res.data;
      if (typeof window !== 'undefined') {
        localStorage.setItem('token', token);
        if (user) localStorage.setItem('user', JSON.stringify(user));
        window.dispatchEvent(new Event('auth-change'));
      }
      router.push('/dashboard');
    } catch (e: any) {
      setError(e.response?.data?.error || e.message || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      setError('Please enter your email or username');
      return;
    }
    if (!password) {
      setError('Please enter your password');
      return;
    }
    await performLogin(email, password);
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center py-10 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-teal-50 via-gray-50 to-emerald-50 dark:from-gray-950 dark:via-gray-900 dark:to-teal-950/40 relative overflow-hidden">
      {/* Background aesthetic blobs */}
      <div className="absolute -top-24 -left-24 w-96 h-96 bg-teal-300/20 dark:bg-teal-700/10 rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-emerald-300/20 dark:bg-emerald-700/10 rounded-full blur-3xl pointer-events-none"></div>

      <div className="w-full max-w-md space-y-6 relative z-10">
        {/* Header Branding */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-100 dark:bg-teal-900/60 text-teal-800 dark:text-teal-200 text-xs font-semibold">
            <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
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
            Sign In to Your Account
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Access your medical history, diagnostic reports, and tailored health analytics.
          </p>
        </div>

        {/* Card Container */}
        <div className="bg-white dark:bg-gray-800/90 backdrop-blur-md p-7 rounded-3xl shadow-xl border border-gray-100 dark:border-gray-700 space-y-5">
          {error && (
            <div className="p-3.5 bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900/60 rounded-xl text-rose-700 dark:text-rose-300 text-xs flex items-start gap-2.5 animate-shake">
              <span className="material-symbols-outlined text-[18px] shrink-0 mt-0.5">lock_clock</span>
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email or Username Field */}
            <div>
              <label htmlFor="email" className="block text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 mb-1.5">
                Email or Name
              </label>
              <div className="relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                  <span className="material-symbols-outlined text-[18px]">person</span>
                </div>
                <input
                  id="email"
                  name="email"
                  type="text"
                  required
                  placeholder="Enter your email or username"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 text-sm rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-750 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition"
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="password" className="block text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300">
                  Password
                </label>
              </div>
              <div className="relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                  <span className="material-symbols-outlined text-[18px]">lock</span>
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  required
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-750 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                >
                  <span className="material-symbols-outlined text-[18px]">
                    {showPassword ? 'visibility_off' : 'visibility'}
                  </span>
                </button>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 px-4 rounded-xl bg-[#00685f] hover:bg-[#005049] text-white font-bold text-sm shadow-md hover:shadow-lg disabled:opacity-50 transition active:scale-[0.99] flex items-center justify-center space-x-2 cursor-pointer mt-2"
            >
              {loading ? (
                <>
                  <span className="animate-spin text-[16px]">⏳</span>
                  <span>Signing In…</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-[18px]">login</span>
                  <span>Sign In</span>
                </>
              )}
            </button>
          </form>

          {/* Clean Subtext */}
          <div className="pt-2 border-t border-gray-100 dark:border-gray-700/80 text-center text-[11px] text-gray-400">
            <span>Private & confidential health tracking</span>
          </div>
        </div>

        {/* Link to Register */}
        <p className="text-center text-xs text-gray-500 dark:text-gray-400">
          New to LifeMap AI?{' '}
          <Link href="/register" className="font-bold text-[#00685f] dark:text-[#89f5e7] hover:underline">
            Create an Account →
          </Link>
        </p>
      </div>
    </div>
  );
}
