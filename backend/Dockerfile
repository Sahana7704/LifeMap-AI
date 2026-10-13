# Multi-stage/Slim production container for LifeMap AI Node.js Backend
FROM node:20-slim

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=5000

# Install production dependencies only
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source code and database migrations
COPY src/ ./src/

# Create uploads directory with appropriate permissions
RUN mkdir -p uploads && chown -R node:node /app

# Non-root user execution
USER node

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:5000/health', (r) => { if (r.statusCode !== 200) process.exit(1); })"

CMD ["node", "src/server.js"]
