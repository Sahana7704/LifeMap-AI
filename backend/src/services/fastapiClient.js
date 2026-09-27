const axios = require('axios');
const FormData = require('form-data');

let rawFastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
if (!rawFastApiUrl.startsWith('http://') && !rawFastApiUrl.startsWith('https://')) {
  rawFastApiUrl = `http://${rawFastApiUrl}`;
}
try {
  const parsed = new URL(rawFastApiUrl);
  if (!parsed.port && !parsed.hostname.includes('.') && parsed.hostname !== 'localhost') {
    parsed.port = '8000';
    rawFastApiUrl = parsed.toString().replace(/\/$/, '');
  }
} catch (e) {
  // Use fallback URL if malformed
}
const FASTAPI_URL = rawFastApiUrl;

function isTransientError(err) {
  const status = err.response?.status;
  return (
    status === 429 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    err.code === 'ECONNREFUSED' ||
    err.code === 'ECONNRESET' ||
    err.code === 'ETIMEDOUT'
  );
}

async function retryOperation(fn, operationName, maxRetries = 3, delayMs = 4000) {
  let lastError;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const status = err.response?.status;
      if (isTransientError(err) && attempt < maxRetries) {
        const waitTime = status === 429 ? (delayMs * 1.5 * attempt) : (delayMs * attempt);
        console.warn(`[fastapiClient] ${operationName} attempt ${attempt} failed with ${status ? `status ${status}` : err.code} (${status === 429 ? 'Rate limit cooling down' : 'service waking up'}). Retrying in ${Math.round(waitTime / 1000)}s...`);
        await new Promise(r => setTimeout(r, waitTime));
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

async function predictRisk(healthMetrics) {
  const url = `${FASTAPI_URL}/predict`;
  return retryOperation(async () => {
    const response = await axios.post(url, healthMetrics, {
      timeout: 45000,
      headers: { 'Content-Type': 'application/json' }
    });
    return response.data;
  }, 'predictRisk');
}

async function extractReport(fileBuffer, filename, mimetype = 'application/pdf') {
  const url = `${FASTAPI_URL}/extract-report`;
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
  }, 'extractReport', 3, 5000);
}

async function extractMetabolicReport(fileBuffer, filename, mimetype = 'application/pdf') {
  const url = `${FASTAPI_URL}/extract-metabolic`;
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
  }, 'extractMetabolicReport', 3, 5000);
}

async function getRecommendations(recommendationData) {
  const url = `${FASTAPI_URL}/recommend`;
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
  getRecommendations
};
