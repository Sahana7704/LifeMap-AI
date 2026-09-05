const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

const connectionString = process.env.DATABASE_URL || 'postgresql://lifemap_user:lifemap_pass@localhost:5432/lifemap_db';

let pool = null;
let useMemoryFallback = false;

// In-memory / file-persisted store fallback for local host testing without external Postgres
const fallbackStoragePath = path.join(__dirname, 'local_storage.json');
let fallbackStore = {
  users: [],
  health_profiles: [],
  medical_reports: [],
  risk_predictions: [],
  shap_explanations: [],
  diet_plans: [],
  exercise_plans: [],
  wellness_logs: []
};

// Load existing fallback data if present
if (fs.existsSync(fallbackStoragePath)) {
  try {
    fallbackStore = JSON.parse(fs.readFileSync(fallbackStoragePath, 'utf8'));
  } catch (e) {
    console.log('Notice: Initializing fresh local fallback database store');
  }
}

function saveFallback() {
  try {
    fs.writeFileSync(fallbackStoragePath, JSON.stringify(fallbackStore, null, 2));
  } catch (e) {
    console.error('Failed to persist fallback store:', e);
  }
}

async function initDB() {
  try {
    pool = new Pool({
      connectionString,
      connectionTimeoutMillis: 2000
    });

    const client = await pool.connect();
    console.log('Connected to PostgreSQL database successfully.');
    
    const schemaSql = fs.readFileSync(path.join(__dirname, 'schema.sql'), 'utf8');
    await client.query(schemaSql);
    client.release();
    console.log('PostgreSQL schema verified and ready.');
  } catch (err) {
    console.log('PostgreSQL not available on host (' + err.message + '). Using local persistent store fallback.');
    useMemoryFallback = true;
  }
}

async function query(text, params = []) {
  if (!useMemoryFallback && pool) {
    try {
      return await pool.query(text, params);
    } catch (err) {
      // If pool connection dropped, fallback
      useMemoryFallback = true;
    }
  }

  // Generic fallback query runner for standard SELECT / INSERT / UPDATE queries
  const sql = text.trim();
  const lower = sql.toLowerCase();

  // 1. INSERT INTO table
  if (lower.startsWith('insert into')) {
    const tableMatch = lower.match(/insert into\s+([a-z_]+)/);
    if (tableMatch) {
      const table = tableMatch[1];
      if (!fallbackStore[table]) fallbackStore[table] = [];

      // Extract column names from query
      const colsMatch = sql.match(/insert into\s+[a-z_]+\s*\(([^\)]+)\)/i);
      if (colsMatch) {
        const cols = colsMatch[1].split(',').map(c => c.trim().toLowerCase());
        const row = {};
        cols.forEach((col, idx) => {
          row[col] = params[idx];
        });
        fallbackStore[table].push(row);
        saveFallback();
        return { rows: [row], rowCount: 1 };
      }
    }
  }

  // 2. SELECT FROM table
  if (lower.startsWith('select')) {
    const tableMatch = lower.match(/from\s+([a-z_]+)/);
    if (tableMatch) {
      const table = tableMatch[1];
      let rows = [...(fallbackStore[table] || [])];

      // Handle WHERE conditions
      if (lower.includes('where')) {
        if (lower.includes('email =') && params.length > 0) {
          const target = String(params[0]).trim().toLowerCase();
          rows = rows.filter(r => {
            const email = (r.email || '').toLowerCase();
            const name = (r.name || '').toLowerCase();
            const prefix = email.split('@')[0];
            return email === target || name === target || prefix === target;
          });
        } else if (lower.includes('user_id =') && params.length > 0) {
          rows = rows.filter(r => r.user_id === params[0]);
        } else if (lower.includes('prediction_id =') && params.length > 0) {
          rows = rows.filter(r => r.prediction_id === params[0]);
        }
      }

      // Handle ORDER BY created_at DESC
      if (lower.includes('order by created_at desc') || lower.includes('order by upload_date desc') || lower.includes('order by date desc')) {
        rows = [...rows].sort((a, b) => {
          const tA = new Date(a.created_at || a.upload_date || a.date || 0).getTime();
          const tB = new Date(b.created_at || b.upload_date || b.date || 0).getTime();
          return tB - tA;
        });
      }

      // Handle LIMIT with boundary
      const limitMatch = lower.match(/\blimit\s+(\d+)\b/);
      if (limitMatch) {
        const limitVal = parseInt(limitMatch[1], 10);
        rows = rows.slice(0, limitVal);
      } else if (lower.includes('limit') && params.length > 0) {
        const limitVal = parseInt(params[params.length - 1], 10) || 10;
        rows = rows.slice(0, limitVal);
      }

      return { rows, rowCount: rows.length };
    }
  }

  // 3. UPDATE table
  if (lower.startsWith('update')) {
    const tableMatch = lower.match(/update\s+([a-z_]+)/);
    if (tableMatch) {
      const table = tableMatch[1];
      let rows = fallbackStore[table] || [];
      if (lower.includes('user_id =') && params.length > 0) {
        const targetUserId = params[params.length - 1];
        const record = rows.find(r => r.user_id === targetUserId);
        if (record) {
          if (table === 'users') {
            if (lower.includes('name =')) {
              record.name = params[0] || record.name;
            }
            if (lower.includes('password_hash =')) {
              record.password_hash = params[0] || record.password_hash;
            }
          } else if (table === 'health_profiles') {
            if (lower.includes('set vitals =')) {
              record.vitals = params[0] !== undefined ? params[0] : record.vitals;
              if (params[1] !== null && params[1] !== undefined) record.age = params[1];
              if (params[2] !== null && params[2] !== undefined) record.gender = params[2];
              if (params[3] !== null && params[3] !== undefined) record.bmi = params[3];
              if (params[4] !== null && params[4] !== undefined) record.height = params[4];
              if (params[5] !== null && params[5] !== undefined) record.weight = params[5];
              record.updated_at = new Date().toISOString();
            } else {
              record.age = params[0] !== undefined ? params[0] : record.age;
              record.gender = params[1] !== undefined ? params[1] : record.gender;
              record.height = params[2] !== undefined ? params[2] : record.height;
              record.weight = params[3] !== undefined ? params[3] : record.weight;
              record.bmi = params[4] !== undefined ? params[4] : record.bmi;
              record.lifestyle_factors = params[5] !== undefined ? params[5] : record.lifestyle_factors;
              record.vitals = params[6] !== undefined ? params[6] : record.vitals;
              record.updated_at = new Date().toISOString();
            }
          }
          saveFallback();
          return { rows: [record], rowCount: 1 };
        }
      }
    }
  }

  // 4. DELETE FROM table
  if (lower.startsWith('delete from')) {
    const tableMatch = lower.match(/delete from\s+([a-z_]+)/);
    if (tableMatch) {
      const table = tableMatch[1];
      if (fallbackStore[table]) {
        const initialCount = fallbackStore[table].length;
        if (lower.includes('where')) {
          if (lower.includes('user_id =') && lower.includes('report_id =')) {
            const reportId = String(params.find(p => String(p).startsWith('rep_') || String(p).startsWith('report_')) || params[0]);
            const userId = String(params.find(p => String(p).startsWith('usr_')) || params[1]);
            fallbackStore[table] = fallbackStore[table].filter(
              r => !(r.report_id === reportId && r.user_id === userId)
            );
          } else if (lower.includes('report_id =') && params.length > 0) {
            const reportId = String(params[0]);
            fallbackStore[table] = fallbackStore[table].filter(r => r.report_id !== reportId);
          } else if (lower.includes('user_id =') && params.length > 0) {
            const userId = String(params[0]);
            fallbackStore[table] = fallbackStore[table].filter(r => r.user_id !== userId);
          }
        } else {
          fallbackStore[table] = [];
        }
        const deletedCount = initialCount - fallbackStore[table].length;
        saveFallback();
        return { rows: [], rowCount: deletedCount };
      }
    }
  }

  return { rows: [], rowCount: 0 };
}

module.exports = {
  initDB,
  query,
  get isFallback() {
    return useMemoryFallback;
  }
};
