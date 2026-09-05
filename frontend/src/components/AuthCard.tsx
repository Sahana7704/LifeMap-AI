'use client';

import Link from 'next/link';

interface AuthCardProps {
  title?: string;
  description?: string;
}

export default function AuthCard({
  title = 'Please Log In',
  description = 'You need to be logged in to access this feature and view your personalized health data.',
}: AuthCardProps) {
  return (
    <div className="max-w-md mx-auto text-center py-16 px-4 space-y-5 animate-in fade-in zoom-in-95 duration-200">
      <div className="w-16 h-16 mx-auto rounded-2xl bg-teal-50 dark:bg-teal-950/50 border border-teal-200/80 dark:border-teal-800/80 flex items-center justify-center text-3xl shadow-sm">
        🔐
      </div>

      <div className="space-y-2">
        <h2
          className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          {title}
        </h2>
        <p className="text-gray-500 dark:text-gray-400 text-sm max-w-sm mx-auto leading-relaxed">
          {description}
        </p>
      </div>

      <div className="pt-2 flex items-center justify-center gap-3">
        <Link
          href="/login"
          className="px-6 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl font-semibold shadow-sm transition active:scale-[0.98] text-sm"
        >
          Go to Login
        </Link>
        <Link
          href="/register"
          className="px-6 py-2.5 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-750 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 rounded-xl font-semibold transition text-sm shadow-2xs"
        >
          Create Account
        </Link>
      </div>
    </div>
  );
}
