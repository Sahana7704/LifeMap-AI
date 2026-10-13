// In-memory token bucket / sliding window rate limiter
function createRateLimiter({ windowMs, maxRequests, message }) {
  const requests = new Map();
  setInterval(() => {
    const now = Date.now();
    for (const [key, record] of requests.entries()) {
      if (now > record.resetTime) requests.delete(key);
    }
  }, 60000).unref();

  return (req, res, next) => {
    // Key on user id if authenticated, else IP address
    const key = req.user?.user_id || req.ip || req.connection?.remoteAddress || '127.0.0.1';
    const now = Date.now();
    const record = requests.get(key) || { count: 0, resetTime: now + windowMs };

    if (now > record.resetTime) {
      record.count = 0;
      record.resetTime = now + windowMs;
    }

    record.count += 1;
    requests.set(key, record);

    if (record.count > maxRequests) {
      return res.status(429).json({ error: message || 'Too many requests. Please try again later.' });
    }
    next();
  };
}

const authRateLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000,
  maxRequests: parseInt(process.env.RATE_LIMIT_AUTH_MAX || '100', 10),
  message: 'Too many authentication attempts. Please try again in 15 minutes.'
});

const uploadRateLimiter = createRateLimiter({
  windowMs: 10 * 60 * 1000,
  maxRequests: parseInt(process.env.RATE_LIMIT_UPLOAD_MAX || '100', 10),
  message: 'Upload rate limit exceeded. Please wait a few minutes before uploading another report.'
});

module.exports = {
  createRateLimiter,
  authRateLimiter,
  uploadRateLimiter
};
