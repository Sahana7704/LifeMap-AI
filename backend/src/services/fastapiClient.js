const axios = require('axios');
const FormData = require('form-data');

function getFastApiUrl() {
  const isProduction = process.env.NODE_ENV === 'production' || process.env.RENDER === 'true';
  let raw = process.env.FASTAPI_URL ? process.env.FASTAPI_URL.trim() : '';

  if (!raw) {
    if (isProduction) {
      throw new Error(
        'FASTAPI_URL environment variable is required in production with no localhost fallback. ' +
        'Please configure FASTAPI_URL in Render environment settings (e.g. https://lifemap-ai-hbmn.onrender.com).'
      );
    }
    raw = 'http://127.0.0.1:8000';
  }

  if (!raw.startsWith('http://') && !raw.startsWith('https://')) {
    raw = isProduction ? `https://${raw}` : `http://${raw}`;
  }

  return raw.replace(/\/+$/, '');
}

function isTransientError(err) {
  const status = err.response?.status;
  const msg = (err.message || '').toLowerCase();
  return (
    status === 429 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    err.code === 'ECONNREFUSED' ||
    err.code === 'ECONNRESET' ||
    err.code === 'ETIMEDOUT' ||
    err.code === 'ENOTFOUND' ||
    err.code === 'ERR_NETWORK' ||
    msg.includes('timeout') ||
    msg.includes('network error')
  );
}

async function retryOperation(fn, operationName, maxRetries = 5, delayMs = 3000) {
  let lastError;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const status = err.response?.status;
      if (isTransientError(err) && attempt < maxRetries) {
        // Exponential backoff: 3s, 6s, 12s, 18s...
        const waitTime = status === 429
          ? Math.min(25000, delayMs * 2 * attempt)
          : Math.min(20000, delayMs * Math.pow(1.8, attempt - 1));
        console.warn(`[fastapiClient] ${operationName} attempt ${attempt}/${maxRetries} failed with ${status ? `status ${status}` : (err.code || err.message)} (${status === 429 ? 'Rate limit cooling down' : 'microservice waking up / cold start'}). Retrying in ${Math.round(waitTime / 1000)}s...`);
        await new Promise(r => setTimeout(r, waitTime));
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

async function predictRisk(healthMetrics) {
  const url = `${getFastApiUrl()}/predict`;
  return retryOperation(async () => {
    const response = await axios.post(url, healthMetrics, {
      timeout: 45000,
      headers: { 'Content-Type': 'application/json' }
    });
    return response.data;
  }, 'predictRisk');
}

async function extractReport(fileBuffer, filename, mimetype = 'application/pdf') {
  const url = `${getFastApiUrl()}/extract-report`;
  return retryOperation(async () => {
    const form = new FormData();
    form.append('file', fileBuffer, {
      filename,
      contentType: mimetype
    });
    const response = await axios.post(url, form, {
      headers: form.getHeaders(),
      timeout: 240000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    return response.data;
  }, 'extractReport', 4, 5000);
}

async function extractMetabolicReport(fileBuffer, filename, mimetype = 'application/pdf') {
  const url = `${getFastApiUrl()}/extract-metabolic`;
  return retryOperation(async () => {
    const form = new FormData();
    form.append('file', fileBuffer, {
      filename,
      contentType: mimetype
    });
    const response = await axios.post(url, form, {
      headers: form.getHeaders(),
      timeout: 240000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    return response.data;
  }, 'extractMetabolicReport', 4, 5000);
}

async function getRecommendations(recommendationData) {
  const url = `${getFastApiUrl()}/recommend`;
  return retryOperation(async () => {
    const response = await axios.post(url, recommendationData, {
      timeout: 60000,
      headers: { 'Content-Type': 'application/json' }
    });
    return response.data;
  }, 'getRecommendations');
}

module.exports = {
  predictRisk,
  extractReport,
  extractMetabolicReport,
  getRecommendations,
  getFastApiUrl
};
