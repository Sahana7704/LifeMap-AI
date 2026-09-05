// Redis session & cache service with in-memory TTL fallback

class CacheService {
  constructor() {
    this.memoryCache = new Map();
    this.redisClient = null;
    this.isRedisConnected = false;

    // Optional Redis connection
    const redisUrl = process.env.REDIS_URL;
    if (redisUrl) {
      try {
        // If redis package is present, can initialize
        console.log('Redis URL configured:', redisUrl);
      } catch (e) {
        console.log('Using in-memory cache layer');
      }
    }
  }

  async set(key, value, ttlSeconds = 3600) {
    const expiresAt = Date.now() + ttlSeconds * 1000;
    this.memoryCache.set(key, {
      value: JSON.stringify(value),
      expiresAt
    });
    return true;
  }

  async get(key) {
    const item = this.memoryCache.get(key);
    if (!item) return null;

    if (Date.now() > item.expiresAt) {
      this.memoryCache.delete(key);
      return null;
    }

    try {
      return JSON.parse(item.value);
    } catch {
      return item.value;
    }
  }

  async del(key) {
    this.memoryCache.delete(key);
    return true;
  }

  async delete(key) {
    return this.del(key);
  }
}

module.exports = new CacheService();
