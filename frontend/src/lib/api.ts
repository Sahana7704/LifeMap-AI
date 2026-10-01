import axios from 'axios';

export function getApiBaseUrl(): string {
  // 1. Explicit NEXT_PUBLIC_API_URL takes precedence if configured
  const envUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (envUrl && envUrl !== '') {
    let clean = envUrl.replace(/\/+$/, '');
    if (!clean.endsWith('/api')) {
      clean = `${clean}/api`;
    }
    return clean;
  }

  const isProduction = process.env.NODE_ENV === 'production';

  // In production, prohibit hardcoded localhost fallback
  if (isProduction) {
    if (typeof window !== 'undefined') {
      return '/api';
    }
    throw new Error('NEXT_PUBLIC_API_URL environment variable is required in production with no localhost fallback.');
  }

  // 2. Local development fallback
  if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    return 'http://localhost:5000/api';
  }

  // 3. Browser on local network
  if (typeof window !== 'undefined') {
    return '/api';
  }

  // 4. Local SSR development fallback
  return 'http://127.0.0.1:5000/api';
}

export const API_URL = getApiBaseUrl();

// Helper to include JWT token from localStorage (if used) or cookies
function getAuthHeaders() {
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// Global response interceptor to handle session expiry gracefully
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== 'undefined') {
      const hadToken = Boolean(localStorage.getItem('token'));
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      if (hadToken) {
        window.dispatchEvent(new Event('auth-change'));
      }
    }
    return Promise.reject(error);
  }
);

export const api = {
  // Auth
  register: (data: any) => axios.post(`${API_URL}/auth/register`, data),
  login: (data: any) => axios.post(`${API_URL}/auth/login`, data),
  getAccounts: () => axios.get(`${API_URL}/auth/accounts`),

  // Profile
  getProfile: () => axios.get(`${API_URL}/profile`, { headers: getAuthHeaders() }),
  updateProfile: (data: any) => axios.put(`${API_URL}/profile`, data, { headers: getAuthHeaders() }),

  // Predictions
  runPrediction: (payload: any) =>
    axios.post(`${API_URL}/predictions/run`, payload, { headers: getAuthHeaders() }),
  getLatestPrediction: () =>
    axios.get(`${API_URL}/predictions/latest`, { headers: getAuthHeaders() }),
  getPredictionHistory: () =>
    axios.get(`${API_URL}/predictions/history`, { headers: getAuthHeaders() }),

  // Reports
  uploadReport: (file: File) => {
    const form = new FormData();
    form.append('report', file);
    return axios.post(`${API_URL}/reports/upload`, form, {
      headers: { ...getAuthHeaders() },
    });
  },
  uploadMetabolicReport: (file: File) => {
    const form = new FormData();
    form.append('report', file);
    return axios.post(`${API_URL}/reports/metabolic-upload`, form, {
      headers: { ...getAuthHeaders() },
    });
  },
  getReports: () => axios.get(`${API_URL}/reports`, { headers: getAuthHeaders() }),
  deleteAllReports: () => axios.delete(`${API_URL}/reports/all`, { headers: getAuthHeaders() }),
  deleteReport: (reportId: string) => axios.delete(`${API_URL}/reports/${reportId}`, { headers: getAuthHeaders() }),

  // Recommendations
  generateRecommendations: (payload: any) =>
    axios.post(`${API_URL}/recommendations/generate`, payload, { headers: getAuthHeaders() }),
  getCurrentPlan: () =>
    axios.get(`${API_URL}/recommendations/current`, { headers: getAuthHeaders() }),

  // Wellness
  addLog: (payload: any) =>
    axios.post(`${API_URL}/wellness/log`, payload, { headers: getAuthHeaders() }),
  getLogs: () => axios.get(`${API_URL}/wellness/logs`, { headers: getAuthHeaders() }),
  getTrends: () => axios.get(`${API_URL}/wellness/trends`, { headers: getAuthHeaders() }),
};
