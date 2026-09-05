FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies
RUN npm install --omit=dev

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 5000

CMD ["node", "src/server.js"]
