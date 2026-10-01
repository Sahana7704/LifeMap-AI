require('dotenv').config();

const express = require('express');
const cors = require('cors');
const db = require('./db');

// Import route modules
const authRoutes = require('./routes/authRoutes');
const profileRoutes = require('./routes/profileRoutes');
const reportRoutes = require('./routes/reportRoutes');
const predictionRoutes = require('./routes/predictionRoutes');
const recommendationRoutes = require('./routes/recommendationRoutes');
const wellnessRoutes = require('./routes/wellnessRoutes');

const app = express();
const PORT = process.env.PORT || 5000;

// CORS — allow Vercel deployments, Render, and localhost
const corsOptions = {
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, curl, Postman)
    if (!origin) return callback(null, true);
    // Allow all Vercel subdomains, Render subdomains, localhost, and custom origin
    if (
      origin.includes('vercel.app') ||
      origin.includes('localhost') ||
      origin.includes('127.0.0.1') ||
      origin.includes('onrender.com') ||
      process.env.ALLOWED_ORIGIN === '*' ||
      (process.env.ALLOWED_ORIGIN && origin === process.env.ALLOWED_ORIGIN)
    ) {
      return callback(null, true);
    }
    // Fallback: allow request so Vercel preview URLs or custom domains don't get blocked
    return callback(null, true);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

// Middleware
app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Initialize DB (PostgreSQL or fallback)
(async () => {
  try {
    await db.initDB();
    console.log('Database initialization completed');
  } catch (e) {
    console.error('Database init error:', e);
  }
})();

// Route mounting under /api
app.use('/api/auth', authRoutes);
app.use('/api/profile', profileRoutes);
app.use('/api/reports', reportRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/recommendations', recommendationRoutes);
app.use('/api/wellness', wellnessRoutes);

// Health check endpoints with ML service pre-warming for Render free-tier cold-starts
async function checkMlHealth() {
  const axios = require('axios');
  const isProduction = process.env.NODE_ENV === 'production' || process.env.RENDER === 'true';
  const fastApiUrl = process.env.FASTAPI_URL ? process.env.FASTAPI_URL.trim() : (isProduction ? null : 'http://127.0.0.1:8000');
  if (!fastApiUrl) {
    return 'unconfigured';
  }
  const cleanUrl = fastApiUrl.replace(/\/+$/, '');
  try {
    const mlRes = await axios.get(`${cleanUrl}/health`, { timeout: 8000 });
    return mlRes.data?.status || 'healthy';
  } catch (err) {
    // Fire a non-blocking background wake-up request to help spin up the Render container
    axios.get(`${cleanUrl}/health`, { timeout: 60000 }).catch(() => {});
    return 'warming_up';
  }
}

app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'LifeMap AI Node.js Backend API',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// Lightweight GET /health endpoint for Render health checks and external pings
app.get('/health', (req, res) => {
  // Non-blocking trigger to pre-warm ML container in background
  checkMlHealth().catch(() => {});
  res.json({
    status: 'ok',
    service: 'lifemap-backend-qbei',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/health', async (req, res) => {
  const mlStatus = await checkMlHealth();
  res.json({
    status: 'ok',
    service: 'lifemap-backend-qbei',
    ml_service: mlStatus,
    timestamp: new Date().toISOString()
  });
});

// Global error handler fallback
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error', details: err.message });
});

app.listen(PORT, () => {
  console.log(`LifeMap AI backend listening on port ${PORT}`);
});
