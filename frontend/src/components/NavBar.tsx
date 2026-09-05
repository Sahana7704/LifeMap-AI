'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useState, useEffect, useCallback } from 'react';

const navLinks = [
  { href: '/dashboard', label: 'Dashboard' },
  { href: '/data-entry', label: 'Data Entry' },
  { href: '/reports', label: 'Reports' },
  { href: '/recommendations', label: 'Recommendations' },
  { href: '/wellness', label: 'Wellness' },
  { href: '/profile', label: 'Profile' },
];

export default function NavBar() {
  const pathname = usePathname();
  const router = useRouter();
  const [loggedIn, setLoggedIn] = useState(false);
  const [userName, setUserName] = useState<string>('');
  const [dark, setDark] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const checkAuth = useCallback(() => {
    if (typeof window === 'undefined') return;
    const token = localStorage.getItem('token');
    setLoggedIn(!!token);
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const u = JSON.parse(storedUser);
        setUserName(u.name || '');
      } catch {
        setUserName('');
      }
    }
  }, []);

  useEffect(() => {
    checkAuth();

    // Dark mode state
    const saved = localStorage.getItem('darkMode');
    const isDark = saved === 'true';
    setDark(isDark);
    if (isDark) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');

    // Listen to storage events and custom auth changes
    window.addEventListener('storage', checkAuth);
    window.addEventListener('auth-change', checkAuth);

    return () => {
      window.removeEventListener('storage', checkAuth);
      window.removeEventListener('auth-change', checkAuth);
    };
  }, [pathname, checkAuth]);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    localStorage.setItem('darkMode', String(next));
    document.documentElement.classList.toggle('dark', next);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setLoggedIn(false);
    setUserName('');
    window.dispatchEvent(new Event('auth-change'));
    router.push('/login');
  };

  const isAuth = pathname === '/login' || pathname === '/register';
  if (isAuth) return null;

  return (
    <nav className="sticky top-0 z-30 backdrop-blur-md bg-white/80 dark:bg-gray-900/85 border-b border-gray-200 dark:border-gray-800 px-4 py-3">
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        {/* Logo */}
        <Link href="/dashboard" className="flex items-center space-x-2">
          <span className="text-2xl font-extrabold text-teal-600 tracking-tight">LifeMap</span>
          <span className="text-xs font-semibold bg-teal-100 dark:bg-teal-900 text-teal-700 dark:text-teal-300 px-2 py-0.5 rounded-full">
            AI
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center space-x-1">
          {navLinks.map(link => (
            <Link
              key={link.href}
              href={link.href}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                pathname === link.href
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* Right controls */}
        <div className="flex items-center space-x-3">
          {/* Dark mode toggle */}
          <button
            onClick={toggleDark}
            aria-label="Toggle dark mode"
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition text-base"
          >
            {dark ? '☀️' : '🌙'}
          </button>

          {/* Auth indicator */}
          {loggedIn ? (
            <div className="flex items-center space-x-2">
              {userName && (
                <Link
                  href="/profile"
                  className="hidden lg:inline-block text-xs font-semibold px-2.5 py-1 rounded-full bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-800"
                >
                  👤 {userName.split(' ')[0]}
                </Link>
              )}
              <button
                onClick={handleLogout}
                className="px-3 py-1.5 text-xs font-medium bg-red-500 hover:bg-red-600 text-white rounded-md transition shadow-sm"
              >
                Logout
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="px-3.5 py-1.5 text-xs font-semibold bg-teal-600 hover:bg-teal-700 text-white rounded-md transition shadow-sm"
            >
              Login
            </Link>
          )}

          {/* Mobile hamburger */}
          <button
            className="md:hidden p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Open menu"
          >
            <span className="block w-5 h-0.5 bg-gray-700 dark:bg-gray-300 mb-1" />
            <span className="block w-5 h-0.5 bg-gray-700 dark:bg-gray-300 mb-1" />
            <span className="block w-5 h-0.5 bg-gray-700 dark:bg-gray-300" />
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden mt-3 pt-2 border-t border-gray-100 dark:border-gray-800 space-y-1">
          {navLinks.map(link => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMenuOpen(false)}
              className={`block px-3 py-2 rounded-md text-sm font-medium ${
                pathname === link.href ? 'bg-teal-600 text-white' : 'hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              {link.label}
            </Link>
          ))}
          <div className="pt-2 border-t border-gray-100 dark:border-gray-800">
            {loggedIn ? (
              <button
                onClick={() => {
                  setMenuOpen(false);
                  handleLogout();
                }}
                className="w-full text-left px-3 py-2 text-sm text-red-500 font-medium hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md"
              >
                Logout {userName ? `(${userName})` : ''}
              </button>
            ) : (
              <Link
                href="/login"
                onClick={() => setMenuOpen(false)}
                className="block px-3 py-2 text-sm text-teal-600 font-medium"
              >
                Login
              </Link>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}
