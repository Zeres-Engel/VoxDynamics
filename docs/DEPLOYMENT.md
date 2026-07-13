# 🚀 Deployment Guide

> **Production deployment strategies and best practices for VoxDynamics.**

---

## 📋 Table of Contents

- [Docker Deployment (Production)](#-docker-deployment-production)
- [Cloud Deployment](#-cloud-deployment)
- [Environment Configuration](#-environment-configuration)
- [Security Considerations](#-security-considerations)
- [Monitoring & Maintenance](#-monitoring--maintenance)
- [Performance Tuning](#-performance-tuning)

---

## 🐳 Docker Deployment (Production)

### Basic Production Setup

```bash
# Build with production optimizations
docker-compose -f docker-compose.yml up -d --build
```

### Production Docker Compose

Create `docker-compose.prod.yml` for production-specific settings:

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:15-alpine
    container_name: voxdynamics-db
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./backups:/backups
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: voxdynamics-app
    restart: always
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - model_cache:/root/.cache
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Optional: backup service
  pgbackups:
    image: prodrigestivill/postgres-backup-local:15-alpine
    restart: always
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
      BACKUP_DIR: /backups
      BACKUP_SUFFIX: .sql.gz
      SCHEDULE: "@daily"
    volumes:
      - ./backups:/backups
    depends_on:
      - postgres

volumes:
  pgdata:
  model_cache:
```

Run with:

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

### Production Dockerfile

Create `Dockerfile.prod` for smaller, more secure images:

```dockerfile
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY models/ ./models/
COPY requirements.txt .

# Non-root user for security
RUN useradd -m -u 1000 voxdynamics && chown -R voxdynamics:voxdynamics /app
USER voxdynamics

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## ☁️ Cloud Deployment

### AWS Elastic Beanstalk

1. **Prepare `Dockerrun.aws.json`:**
```json
{
  "AWSEBDockerrunVersion": "2",
  "containerDefinitions": [
    {
      "name": "voxdynamics-app",
      "image": "your-registry/voxdynamics-app:latest",
      "essential": true,
      "memory": 6144,
      "portMappings": [
        { "hostPort": 80, "containerPort": 8000 }
      ],
      "environment": [
        { "name": "POSTGRES_HOST", "value": "your-rds-endpoint" }
      ]
    }
  ]
}
```

2. **Use RDS for PostgreSQL** — create a PostgreSQL 15 instance and point `DATABASE_URL` to it.

### Google Cloud Run

1. **Build and push:**
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/voxdynamics
```

2. **Deploy:**
```bash
gcloud run deploy voxdynamics \
  --image gcr.io/$PROJECT_ID/voxdynamics \
  --memory 6Gi \
  --cpu 4 \
  --timeout 300 \
  --set-env-vars="DATABASE_URL=postgresql+asyncpg://..." \
  --concurrency 1
```

> **Note:** Cloud Run concurrency should be 1 per container since each instance loads models into memory.

### DigitalOcean App Platform

1. Connect GitHub repository
2. Set resource: 4GB RAM / 2 vCPU minimum
3. Add PostgreSQL as a managed database
4. Set environment variables from `.env`
5. Deploy — platform handles HTTPS, auto-scaling

---

## 🔐 Security Considerations

### Environment Variables

```bash
# NEVER commit .env to git. Use a secrets manager for production:
# - AWS Secrets Manager
# - Google Secret Manager
# - HashiCorp Vault
# - Docker Secrets

# Production .env example (keep secure):
POSTGRES_USER=voxdynamics_prod
POSTGRES_PASSWORD=<strong-random-password-64-chars>
POSTGRES_DB=voxdynamics
POSTGRES_HOST=your-rds-endpoint.amazonaws.com
POSTGRES_PORT=5432
APP_HOST=0.0.0.0
APP_PORT=8000
```

### Database Security

```sql
-- Production database hardening
CREATE USER voxdynamics_admin WITH PASSWORD '<strong-password>';
GRANT ALL PRIVILEGES ON DATABASE voxdynamics TO voxdynamics_admin;

-- Use SSL for connections
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';
```

### API Security

```python
# In production, restrict CORS to your domain:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Add rate limiting (requires slowapi):
# pip install slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@limiter.limit("10/minute")
@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    ...
```

### File Upload Security

```python
# Validate file extension on backend
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac'}
file_ext = os.path.splitext(file.filename)[1].lower()
if file_ext not in ALLOWED_EXTENSIONS:
    raise HTTPException(status_code=400, detail="Unsupported file format")

# Validate file size
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
content = await file.read()
if len(content) > MAX_FILE_SIZE:
    raise HTTPException(status_code=413, detail="File too large")
```

---

## 📊 Monitoring & Maintenance

### Health Check Endpoint

```bash
# Monitor in production
curl https://yourdomain.com/health

# Expected response:
{"status":"ok","models_loaded":true,"timestamp":"...","version":"2.0.0"}
```

### Database Backups

```bash
# Manual backup
docker exec voxdynamics-db pg_dump -U voxdynamics voxdynamics > backup_$(date +%Y%m%d).sql

# Restore
cat backup.sql | docker exec -i voxdynamics-db psql -U voxdynamics voxdynamics
```

### Logging

```bash
# Monitor application logs in real-time
docker-compose logs -f --tail=100 app

# Search for errors
docker-compose logs app | grep -i error

# Export logs
docker-compose logs app > voxdynamics_$(date +%Y%m%d).log
```

### Resource Monitoring

```bash
# Container resource usage
docker stats voxdynamics-app voxdynamics-db

# Disk usage
docker system df
```

### Regular Maintenance Tasks

| Task | Frequency | Command |
|:-----|:----------|:--------|
| Database backup | Daily | `pg_dump` or pgbackups container |
| Log rotation | Weekly | Docker json-file driver handles this |
| Model validation | Monthly | Run validation audio through pipeline |
| Security updates | Monthly | `docker pull postgres:15-alpine` |
| SSL cert renewal | Every 90 days | certbot / Let's Encrypt |

---

## ⚡ Performance Tuning

### Uvicorn Workers

```bash
# Number of workers = 2 × CPU cores (for CPU-bound tasks)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Database Connection Pool

```python
# Adjust based on concurrent users
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,      # Default: 5
    max_overflow=20,   # Default: 10
    pool_pre_ping=True,
)
```

### Memory Optimization

```python
# Load models with memory growth enabled (reduces fragmentation)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Reverse Proxy (NGINX)

```nginx
# /etc/nginx/sites-available/voxdynamics
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;

    client_max_body_size 60M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    location /stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
    }
}
```

### Resource Requirements

| Environment | RAM | CPU | Storage |
|:------------|:---:|:---:|:--------|
| Development | 4 GB | 2 cores | 2 GB |
| Production (light) | 6 GB | 2 cores | 10 GB |
| Production (heavy) | 16 GB | 4+ cores | 50 GB |

---

## 📋 Deployment Checklist

- [ ] Change all default passwords
- [ ] Configure HTTPS (SSL/TLS)
- [ ] Set up database backups
- [ ] Configure monitoring & alerts
- [ ] Set resource limits in Docker Compose
- [ ] Enable production mode (workers > 1)
- [ ] Remove `--reload` flag (development only)
- [ ] Set up reverse proxy (NGINX/Caddy)
- [ ] Configure firewall (allow only 443, 80)
- [ ] Test health endpoint after deployment
- [ ] Verify WebSocket connectivity
- [ ] Run validation audio through pipeline
- [ ] Set up log rotation
- [ ] Document rollback procedure

---

*Last updated: July 2026 — VoxDynamics v2.0*
