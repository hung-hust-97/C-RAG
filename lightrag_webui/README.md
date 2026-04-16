# LightRAG WebUI

LightRAG WebUI is a React-based web interface for interacting with the LightRAG system. It provides a user-friendly interface for querying, managing, and exploring LightRAG's functionalities.

## Architecture Overview

The UI container runs as an independent Docker service with the following architecture:

- **Build Stage**: Uses `oven/bun:1` to compile the React application with Vite
- **Runtime Stage**: Uses `nginx:alpine` to serve static files and proxy API requests
- **Size**: Optimized multi-stage build produces images under 50MB
- **Communication**: Nginx proxies `/api/*`, `/docs`, `/redoc`, and `/openapi.json` to the FastAPI backend

### Container Architecture

```
┌──────────────────────────────────┐
│  UI Container (Port 80)          │
│                                  │
│  ┌────────────────────────────┐  │
│  │   Nginx Server             │  │
│  │                            │  │
│  │  • Serves /webui/*         │  │
│  │  • Proxies /api/* → API    │  │
│  │  • Proxies /docs → API     │  │
│  │  • Health checks           │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
           ↓ (proxy)
┌──────────────────────────────────┐
│  API Container (Port 9621)       │
│  FastAPI Server                  │
└──────────────────────────────────┘
```

## Docker Deployment

### Using Docker Compose (Recommended)

The UI container is defined in `docker-compose.yml` and can be deployed with the full LightRAG stack:

```bash
# Start all services including UI
docker-compose up -d

# Start only UI and its dependencies
docker-compose up -d ui

# View UI logs
docker-compose logs -f ui

# Rebuild and restart UI
docker-compose up -d --build ui

# Stop UI container
docker-compose stop ui
```

### Environment Variables

Configure the UI container via environment variables in `.env`:

```bash
# UI container port mapping (host:container)
UI_PORT=8080

# Backend API URL for Nginx proxy
VITE_BACKEND_URL=http://lightrag:9621
```

### Standalone Docker Build

Build and run the UI container independently:

```bash
# Build the UI image
docker build -t lightrag-ui:latest -f lightrag_webui/Dockerfile lightrag_webui/

# Run the container
docker run -d \
  --name lightrag-ui \
  -p 8080:80 \
  -e VITE_BACKEND_URL=http://lightrag:9621 \
  lightrag-ui:latest

# View logs
docker logs -f lightrag-ui

# Stop and remove
docker stop lightrag-ui && docker rm lightrag-ui
```

## Nginx Configuration

The UI container uses Nginx to serve static files and proxy API requests. Configuration is located at `lightrag_webui/nginx.conf`.

### Proxy Rules

| Location Pattern | Proxy Target | Purpose |
|-----------------|--------------|---------|
| `/webui/` | Static files | Serves React application |
| `/api/` | `http://lightrag:9621/` | Proxies API requests |
| `/docs`, `/redoc`, `/openapi.json` | `http://lightrag:9621$request_uri` | Proxies API documentation |
| `/static/` | `http://lightrag:9621/static/` | Proxies FastAPI static assets |

### Key Configuration Features

- **SPA Routing**: `try_files $uri $uri/ /webui/index.html` handles client-side routing
- **Proxy Headers**: Forwards `Host`, `X-Real-IP`, `X-Forwarded-For`, `X-Forwarded-Proto`
- **Health Checks**: Responds to `wget` requests at `/webui/` for Docker health monitoring

### Customizing Nginx Configuration

To modify the Nginx configuration:

1. Edit `lightrag_webui/nginx.conf`
2. Rebuild the container: `docker-compose up -d --build ui`
3. Verify configuration: `docker exec ui nginx -t`
4. View Nginx logs: `docker-compose logs ui`

## Local Development

### Using Bun (Recommended)

1. **Install Bun:**

    If you haven't already installed Bun, follow the official documentation: [https://bun.sh/docs/installation](https://bun.sh/docs/installation)

2. **Install Dependencies:**

    In the `lightrag_webui` directory, run the following command to install project dependencies:

    ```bash
    bun install --frozen-lockfile
    ```

3. **Start Development Server:**

    ```bash
    bun run dev
    ```

    The dev server will proxy API requests to the backend URL specified in `.env.development`.

4. **Build for Production:**

    ```bash
    bun run build
    ```

    This command will bundle the project and output the built files to the `lightrag/api/webui` directory.

### Using Node.js / npm (Alternative)

If Bun is unavailable or the Bun build fails in your environment (e.g., older Linux distributions, restricted environments, or Bun version incompatibilities), you can use Node.js instead:

```bash
npm install
npm run dev      # Development server
npm run build    # Production build
```

> **Note:** Tests (`bun test`) still require Bun. All other scripts (`dev`, `build`, `preview`, `lint`) work with both Bun and Node.js/npm.

## Configuration

### Backend URL Configuration

The UI can be configured to connect to different backend API servers using the `VITE_BACKEND_URL` environment variable:

- **Development:** Set in `.env.development` (defaults to `http://localhost:9621`)
- **Docker Container:** Set via `VITE_BACKEND_URL` environment variable in `docker-compose.yml`
- **Production:** Can be set at build time or configured via Nginx proxy

**Example configurations:**

```bash
# .env.development (for local development)
VITE_BACKEND_URL=http://localhost:9621
VITE_API_PROXY=true
VITE_API_ENDPOINTS=/api,/documents,/graphs,/graph,/health,/query,/docs,/redoc,/openapi.json,/login,/auth-status,/static

# .env.production (for production builds)
VITE_BACKEND_URL=https://api.example.com

# Docker Compose (in .env file)
UI_PORT=8080
VITE_BACKEND_URL=http://lightrag:9621
```

**Default behavior:**
- If `VITE_BACKEND_URL` is not set, the UI uses relative URLs (empty string)
- This allows Nginx to proxy API requests when serving the UI
- See `lightrag_webui/nginx.conf` for the Nginx proxy configuration

## Script Commands

The following are some commonly used script commands defined in `package.json`:

| Command | Description |
|---------|-------------|
| `bun run dev` / `npm run dev` | Starts the development server |
| `bun run build` / `npm run build` | Builds the project for production |
| `bun run lint` / `npm run lint` | Runs the linter |
| `bun run preview` / `npm run preview` | Previews the production build |
| `bun run build:bun` | Builds using Bun runtime explicitly |
| `bun test` | Runs tests (Bun only) |

## Health Checks

The UI container includes Docker health checks to ensure the service is running correctly:

```yaml
healthcheck:
  test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/webui/"]
  interval: 30s
  timeout: 10s
  retries: 3
```

**Check health status:**

```bash
# View health status
docker-compose ps ui

# Manual health check
docker exec ui wget --quiet --tries=1 --spider http://localhost/webui/
```

## Troubleshooting

### UI Container Issues

#### Container fails to start

**Symptoms:** Container exits immediately or restarts repeatedly

**Solutions:**
1. Check logs: `docker-compose logs ui`
2. Verify Dockerfile syntax: `docker build -t test -f lightrag_webui/Dockerfile lightrag_webui/`
3. Ensure build stage completed: Look for "Build stage" and "Runtime stage" in logs
4. Check port conflicts: Ensure `UI_PORT` (default 8080) is not in use

#### Cannot access UI at http://localhost:8080/webui/

**Symptoms:** Browser shows "Connection refused" or "Cannot connect"

**Solutions:**
1. Verify container is running: `docker-compose ps ui`
2. Check port mapping: `docker port $(docker-compose ps -q ui)`
3. Verify health check: `docker-compose ps ui` should show "healthy"
4. Check firewall rules: Ensure port 8080 is open
5. Try accessing from container: `docker exec ui wget -O- http://localhost/webui/`

#### 502 Bad Gateway when accessing /api/*

**Symptoms:** UI loads but API requests fail with 502 error

**Solutions:**
1. Verify API container is running: `docker-compose ps lightrag`
2. Check API health: `docker-compose exec lightrag curl http://localhost:9621/health`
3. Verify network connectivity: `docker-compose exec ui wget -O- http://lightrag:9621/health`
4. Check Nginx configuration: `docker exec ui cat /etc/nginx/conf.d/default.conf`
5. Review Nginx error logs: `docker-compose logs ui | grep error`

#### Static files return 404

**Symptoms:** UI shows blank page or missing assets

**Solutions:**
1. Verify build output: `docker exec ui ls -la /usr/share/nginx/html/webui/`
2. Check Nginx access logs: `docker-compose logs ui`
3. Rebuild container: `docker-compose up -d --build ui`
4. Verify build stage succeeded: `docker-compose build ui --no-cache`

### Build Issues

#### `bun run build` fails silently or with exit code 1

**Symptoms:** Build command exits without clear error message

**Solutions:**
1. Try Node.js alternative: `npm install && npm run build`
2. Check Bun version: `bun --version` (requires Bun 1.x)
3. Clear cache: `rm -rf node_modules bun.lockb && bun install`
4. Check disk space: `df -h`

#### `Cannot find package '@/lib'`

**Symptoms:** Import errors during build or development

**Solution:** This error occurred in older versions when the Vite config used a TypeScript path alias (`@/`) that only Bun could resolve at config load time. This has been fixed by using a relative import in `vite.config.ts`.

#### Docker build fails at build stage

**Symptoms:** `docker build` fails during `bun run build`

**Solutions:**
1. Check build logs: `docker-compose build ui 2>&1 | tee build.log`
2. Verify source files are copied: Add `RUN ls -la` after `COPY . .` in Dockerfile
3. Check memory limits: Increase Docker memory allocation
4. Try building without cache: `docker-compose build --no-cache ui`

### Development Server Issues

#### Dev server cannot connect to API

**Symptoms:** API requests fail during local development

**Solutions:**
1. Verify API is running: `curl http://localhost:9621/health`
2. Check `.env.development`: Ensure `VITE_BACKEND_URL=http://localhost:9621`
3. Verify proxy configuration in `vite.config.ts`
4. Check CORS settings in API server

#### Hot reload not working

**Symptoms:** Changes to source files don't trigger browser refresh

**Solutions:**
1. Restart dev server: `bun run dev`
2. Clear browser cache
3. Check file watcher limits: `echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p`

### Production Deployment Issues

#### UI works locally but fails in Docker

**Symptoms:** Local build works but Docker container fails

**Solutions:**
1. Verify build output directory: Check that Vite outputs to correct path
2. Compare local and Docker builds: `bun run build` vs `docker-compose build ui`
3. Check environment variables: Ensure `VITE_BACKEND_URL` is set correctly
4. Review Dockerfile COPY paths: Ensure build artifacts are copied correctly

#### Performance issues with large files

**Symptoms:** Slow page loads or timeouts

**Solutions:**
1. Enable gzip compression in Nginx (add to `nginx.conf`):
   ```nginx
   gzip on;
   gzip_types text/plain text/css application/json application/javascript;
   ```
2. Add caching headers for static assets
3. Optimize bundle size: `bun run build --analyze`
4. Consider CDN for static assets

### Getting Help

If you encounter issues not covered here:

1. Check container logs: `docker-compose logs ui`
2. Check API logs: `docker-compose logs lightrag`
3. Verify all services are healthy: `docker-compose ps`
4. Review Nginx configuration: `docker exec ui cat /etc/nginx/conf.d/default.conf`
5. Test Nginx configuration: `docker exec ui nginx -t`
6. Check network connectivity: `docker-compose exec ui ping lightrag`

For more information, see the main [LightRAG documentation](../README.md) and [Docker Compose configuration](../docker-compose.yml).
