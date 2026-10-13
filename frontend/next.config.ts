import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const isProduction = process.env.NODE_ENV === 'production';
    const backendEnv = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
    if (!backendEnv) {
      return [];
    }
    const backend = backendEnv.replace(/\/api\/?$/, '');
    return [
      {
        source: '/api/:path*',
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
