const axios = require('axios');
const FormData = require('form-data');

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';

async function predictRisk(healthMetrics) {
  const url = `${FASTAPI_URL}/predict`;
  const response = await axios.post(url, healthMetrics, {
    timeout: 30000,
    headers: { 'Content-Type': 'application/json' }
  });
  return response.data;
}

async function extractReport(fileBuffer, filename, mimetype = 'application/pdf') {
  const url = `${FASTAPI_URL}/extract-report`;
  const form = new FormData();
  form.append('file', fileBuffer, {
    filename,
    contentType: mimetype
  });

  const response = await axios.post(url, form, {
    headers: form.getHeaders(),
    timeout: 90000,
    maxContentLength: Infinity,
    maxBodyLength: Infinity
  });
  return response.data;
}

async function extractMetabolicReport(fileBuffer, filename, mimetype = 'application/pdf') {
  const url = `${FASTAPI_URL}/extract-metabolic`;
  const form = new FormData();
  form.append('file', fileBuffer, {
    filename,
    contentType: mimetype
  });

  const response = await axios.post(url, form, {
    headers: form.getHeaders(),
    timeout: 90000,
    maxContentLength: Infinity,
    maxBodyLength: Infinity
  });
  return response.data;
}

async function getRecommendations(recommendationData) {
  const url = `${FASTAPI_URL}/recommend`;
  const response = await axios.post(url, recommendationData, {
    timeout: 60000,
    headers: { 'Content-Type': 'application/json' }
  });
  return response.data;
}

module.exports = {
  predictRisk,
  extractReport,
  extractMetabolicReport,
  getRecommendations
};
