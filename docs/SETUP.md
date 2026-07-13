# 🛠️ Setup Guide

> **Comprehensive setup instructions for VoxDynamics — from local development to production.**

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|:------------|:-------:|:------|
| Docker | 24+ | Required for containerized deployment |
| Docker Compose | 2.20+ | Comes with Docker Desktop |
| Python | 3.10+ | Only needed for local development |
| Git | 2.30+ | Version control |

---

## 🐳 Quick Start (Docker — Recommended)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VoxDynamics
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults work out of the box):

```env
# Database
POSTGRES_USER=voxdynamics
POSTGRES_PASSWORD=voxdynamics_secret
POSTGRES_DB=voxdynamics

# App
APP_HOST=0.0.0.0
APP_PORT=8000

# Model paths
MODEL_DIR=models
```

### 3. Build & Launch

```bash
docker-compose up -d --build
```

### 4. Verify

```bash
# Check service status
curl http://localhost:8000/health

# Check logs
docker-compose logs -f app
```

Open **http://localhost:8000** in your browser.

### 5. Stop & Clean

```bash
# Stop services
docker-compose down

# Stop + delete database volume (reset all data)
docker-compose down -v
```

---

## 💻 Local Development Setup

### 1. Clone & Enter

```bash
git clone <repository-url>
cd VoxDynamics
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow 2.10.1 requires Python 3.10. If using a different Python version, install a compatible TensorFlow version.

```bash
# Example for Python 3.11:
pip install tensorflow==2.15.0
```

### 4. Start PostgreSQL

Using Docker for just the database:

```bash
docker run -d \
  --name voxdynamics-db \
  -e POSTGRES_USER=voxdynamics \
  -e POSTGRES_PASSWORD=voxdynamics_secret \
  -e POSTGRES_DB=voxdynamics \
  -p 5432:5432 \
  postgres:15-alpine
```

### 5. Configure Environment

```bash
# Create .env file for local dev (overrides docker-compose DB host)
cat > .env << EOF
DATABASE_URL=postgresql+asyncpg://voxdynamics:voxdynamics_secret@localhost:5432/voxdynamics
APP_HOST=0.0.0.0
APP_PORT=8000
EOF
```

### 6. Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🎯 First-Time Startup Checklist

| Step | Action | Expected Result |
|:----:|:-------|:----------------|
| 1 | Start Docker containers | Both `voxdynamics-app` and `voxdynamics-db` running |
| 2 | Wait ~30s | Models download & load from cache |
| 3 | Check `/health` | `"models_loaded": true` |
| 4 | Open browser | UI loads at localhost:8000 |
| 5 | Upload test audio | Analysis report renders with charts |

### Download Times

| Component | Size | First Download | Subsequent |
|:----------|:----:|:--------------:|:----------:|
| Silero VAD | ~15 MB | ~30s | Instant (cached) |
| CNN Weights | ~25 MB | Included in repo | N/A |
| Docker images | ~1.5 GB | ~5 min | Cached locally |

---

## 🔧 Configuration Reference

### Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `POSTGRES_USER` | `voxdynamics` | Database user |
| `POSTGRES_PASSWORD` | `voxdynamics_secret` | Database password |
| `POSTGRES_DB` | `voxdynamics` | Database name |
| `POSTGRES_HOST` | `postgres` | Database hostname (Docker service name) |
| `POSTGRES_PORT` | `5432` | Database port |
| `DATABASE_URL` | *(computed)* | Full async database URL |
| `APP_HOST` | `0.0.0.0` | Server bind address |
| `APP_PORT` | `8000` | Server port |
| `SAMPLE_RATE` | `16000` | Audio sample rate for VAD |
| `BUFFER_DURATION_S` | `2.5` | CNN input window (seconds) |
| `EMA_ALPHA` | `0.3` | Smoothing factor for real-time |
| `VAD_THRESHOLD` | `0.5` | VAD confidence threshold |

### Configuration via `app/config.py`

```python
class Settings(BaseSettings):
    # Database
    postgres_user: str = "voxdynamics"
    postgres_password: str = "voxdynamics_secret"
    postgres_db: str = "voxdynamics"
    database_url: str = "postgresql+asyncpg://voxdynamics:voxdynamics_secret@postgres:5432/voxdynamics"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Audio
    sample_rate: int = 16000
    buffer_duration_s: float = 2.5
    ema_alpha: float = 0.3
    vad_threshold: float = 0.5
```

---

## 🧪 Testing with Sample Audio

### Built-in Validation File

The following validation file is referenced in the source code but not yet included in the repository:

```
data/emotions/mix/angry_happy_surprised_disgust_sad.wav
```

Contains 5 consecutive emotions with silence gaps:
> **Angry → Happy → Surprised → Disgust → Sad (low-intensity)**

Upload this file via the UI to verify the pipeline end-to-end.

### Creating Test Samples

```python
import librosa
import soundfile as sf

# Load and concatenate multiple emotion samples
audio1, sr1 = librosa.load('angry_sample.wav', sr=16000)
audio2, sr2 = librosa.load('happy_sample.wav', sr=16000)

# Add silence between
silence = np.zeros(int(0.5 * 16000))
combined = np.concatenate([audio1, silence, audio2])

sf.write('test_mix.wav', combined, 16000)
```

---

## 🐛 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|:--------|:-------------|:---------|
| `Connection refused` on DB | PostgreSQL not started | `docker-compose up -d postgres` |
| `No module named tensorflow` | Missing dependency | `pip install tensorflow==2.10.1` |
| Models won't load | Wrong model path | Set `MODEL_DIR` or run from project root |
| WebSocket disconnects | Client sample rate mismatch | Ensure 16kHz PCM float32 |
| High memory usage | Loading both PyTorch + TF | Expected (~2-3GB) for both frameworks |
| Slow first startup | Downloading Silero VAD | Normal on first run (~30s) |

### Logs

```bash
# Application logs
docker-compose logs -f app

# Database logs
docker-compose logs -f postgres

# Full restart with clean state
docker-compose down -v && docker-compose up -d --build
```

### Getting Help

If you encounter issues not covered here:

1. Check the full logs: `docker-compose logs -f`
2. Verify model files exist: `ls models/`
3. Test database connection: `docker-compose exec postgres pg_isready`
4. Open a GitHub issue with logs attached

---

## 📁 Directory Permissions

Ensure the application can write to these directories:

```bash
# When running locally, these should exist:
ls -la models/     # Should contain .h5 and .pickle files
ls -la data/       # Sample audio files (optional)
```
