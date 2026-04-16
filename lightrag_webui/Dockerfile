# Build stage
FROM oven/bun:1 AS builder
WORKDIR /app

# Copy package files and install dependencies
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Copy source files
COPY . .

# Create the directory structure that vite expects for output
# vite.config.ts builds to ../lightrag/api/webui
RUN mkdir -p /lightrag/api

# Build the application
RUN bun run build

# Runtime stage
FROM nginx:alpine

# Copy built files from the vite output directory
COPY --from=builder /lightrag/api/webui /usr/share/nginx/html/webui

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
