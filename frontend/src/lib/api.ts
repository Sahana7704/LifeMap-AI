import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api';

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
