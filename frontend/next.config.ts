import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const isProduction = process.env.NODE_ENV === 'production';
    const backendEnv = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
    if (!backendEnv && isProduction) {
      return [];
    }
    const backend = (backendEnv || 'http://127.0.0.1:5000/api')
      .replace(/\/api\/?$/, '');
    return [
      {
        source: '/api/:path*',
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
