import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import NavBar from '@/components/NavBar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'LifeMap AI – Preventive Health',
  description: 'AI-powered preventive healthcare: diabetes & cardiovascular risk prediction, SHAP explanations, Indian diet plans, and wellness tracking.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"
          rel="stylesheet"
        />
      </head>
      <body className={`${inter.className} bg-[#faf8ff] dark:bg-gray-950 text-gray-900 dark:text-gray-100 min-h-screen flex flex-col`}>
        <NavBar />
        <main className="flex-1 container mx-auto px-4 py-6">{children}</main>
        <footer className="text-center py-4 text-sm border-t border-gray-200 dark:border-gray-800 text-gray-500">
          © {new Date().getFullYear()} LifeMap AI – Preventive Health Intelligence
        </footer>
      </body>
    </html>
  );
}
