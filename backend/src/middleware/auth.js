const jwt = require('jsonwebtoken');
const cacheService = require('../services/redisService');
const db = require('../db');

const JWT_SECRET = process.env.JWT_SECRET || 'lifemap_super_secret_jwt_key_2026';

async function requireAuth(req, res, next) {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Unauthorized: Missing or invalid token format' });
    }

    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET);

    // Check cached session if available
    const cachedUser = await cacheService.get(`session:${decoded.userId}`);
    if (cachedUser) {
      req.user = cachedUser;
      return next();
    }

    const result = await db.query('SELECT user_id, name, email, role FROM users WHERE user_id = $1', [decoded.userId]);
    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Unauthorized: User not found' });
    }

    const user = result.rows[0];
    await cacheService.set(`session:${user.user_id}`, user, 3600);
    req.user = user;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Unauthorized: Invalid or expired token', details: err.message });
  }
}

function generateToken(user) {
  return jwt.sign(
    { userId: user.user_id, email: user.email, role: user.role },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
}

module.exports = {
  requireAuth,
  generateToken,
  JWT_SECRET
};
