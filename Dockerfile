# ============================================================
# VoxDynamics — Dockerfile
# ============================================================

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download models at build time (optional, speeds up startup)
# RUN python -c "from app.core.emotion_model import EmotionPredictor; EmotionPredictor()"

EXPOSE 8000 7860

CMD ["python", "-m", "app.main"]
