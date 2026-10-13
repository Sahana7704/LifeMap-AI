# Multi-stage production Dockerfile for LifeMap AI Next.js Frontend
# Stage 1: Install dependencies
FROM node:20-slim AS deps
WORKDIR /app

COPY package*.json ./
RUN npm ci

# Stage 2: Build Next.js application
FROM node:20-slim AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build-time environment arguments (Next.js rewrites and public URLs)
ARG BACKEND_URL=http://backend:5000/api
ARG NEXT_PUBLIC_API_URL=http://localhost:5000/api

ENV BACKEND_URL=${BACKEND_URL}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

RUN npm run build

# Stage 3: Minimal runner image
FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV NEXT_TELEMETRY_DISABLED=1

# Non-root system user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/', (r) => { if (r.statusCode >= 500) process.exit(1); })"

CMD ["npm", "start"]
