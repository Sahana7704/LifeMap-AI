const crypto = require('crypto');
const db = require('../db');

async function addLog(req, res) {
  try {
    const userId = req.user.user_id;
    const { date, metrics, notes } = req.body;

    if (!metrics) {
      return res.status(400).json({ error: 'Metrics are required to add a wellness log' });
    }

    const logId = `log_${crypto.randomUUID().slice(0, 10)}`;
    const logDate = date || new Date().toISOString().split('T')[0];

    await db.query(
      `INSERT INTO wellness_logs (log_id, user_id, date, metrics, notes, created_at)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [
        logId,
        userId,
        logDate,
        typeof metrics === 'object' ? JSON.stringify(metrics) : metrics,
        notes || '',
        new Date().toISOString()
      ]
    );

    return res.status(201).json({
      message: 'Wellness log recorded successfully',
      log: {
        log_id: logId,
        user_id: userId,
        date: logDate,
        metrics,
        notes
      }
    });
  } catch (err) {
    console.error('Wellness log error:', err);
    return res.status(500).json({ error: 'Failed to record wellness log', details: err.message });
  }
}

async function getLogs(req, res) {
  try {
    const userId = req.user.user_id;
    const result = await db.query(
      'SELECT * FROM wellness_logs WHERE user_id = $1 ORDER BY date DESC, created_at DESC',
      [userId]
    );
    return res.json(result.rows);
  } catch (err) {
    return res.status(500).json({ error: 'Failed to fetch wellness logs', details: err.message });
  }
}

async function getTrends(req, res) {
  try {
    const userId = req.user.user_id;
    const result = await db.query(
      'SELECT * FROM wellness_logs WHERE user_id = $1 ORDER BY date ASC, created_at ASC',
      [userId]
    );

    // Format for Recharts line chart
    const trends = result.rows.map(row => {
      const m = typeof row.metrics === 'string' ? JSON.parse(row.metrics || '{}') : (row.metrics || {});
      return {
        date: row.date,
        glucose: m.glucose || null,
        systolic_bp: m.systolic_bp || null,
        diastolic_bp: m.diastolic_bp || null,
        bmi: m.bmi || null,
        diabetes_risk: m.diabetes_risk || null,
        cvd_risk: m.cvd_risk || null,
        vitality_score: m.vitality_score || null
      };
    });

    return res.json(trends);
  } catch (err) {
    return res.status(500).json({ error: 'Failed to fetch wellness trends', details: err.message });
  }
}

module.exports = {
  addLog,
  getLogs,
  getTrends
};
