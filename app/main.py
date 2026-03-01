# ============================================================
# VoxDynamics — FastAPI Main Application
# ============================================================
"""
Entry point for the VoxDynamics backend.

Starts:
  - FastAPI server on port 8000 (REST + WebSocket)
  - Gradio UI on port 7860
  - PostgreSQL connection on startup
"""

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, desc

from app.config import settings
from app.core.processor import AudioProcessor
from app.api.websocket import websocket_stream, log_emotion_to_db
from app.db.database import init_db, close_db, get_session
from app.db.models import EmotionLog

# ── Shared processor instance (models loaded once) ──────────
processor = AudioProcessor(
    sample_rate=settings.sample_rate,
    buffer_duration_s=settings.buffer_duration_s,
    ema_alpha=settings.ema_alpha,
    vad_threshold=settings.vad_threshold,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    print("=" * 60)
    print("  VoxDynamics — Starting Up")
    print("=" * 60)

    # 1. Init database tables
    print("[STARTUP] Initializing database...")
    await init_db()
    print("[STARTUP] Database ready ✓")

    # 2. Load AI models
    print("[STARTUP] Loading AI models (this may take a minute)...")
    processor.load_models()
    print("[STARTUP] Silero VAD loaded ✓")
    print("[STARTUP] Wav2Vec2 Emotion model loaded ✓")

    print("=" * 60)
    print("  VoxDynamics — Ready!")
    print(f"  API:    http://localhost:{settings.app_port}")
    print(f"  Gradio: http://localhost:{settings.gradio_port}")
    print("=" * 60)

    yield

    # Shutdown
    print("[SHUTDOWN] Closing database...")
    await close_db()
    print("[SHUTDOWN] Done.")


# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="VoxDynamics",
    description="Real-Time Speech Emotion Recognition System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ───────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": processor.models_loaded,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/emotions/{session_id}")
async def get_emotion_history(
    session_id: str,
    limit: int = Query(default=100, le=1000),
):
    """Retrieve emotion prediction history for a session."""
    async with get_session() as session:
        stmt = (
            select(EmotionLog)
            .where(EmotionLog.session_id == session_id)
            .order_by(desc(EmotionLog.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        logs = result.scalars().all()

    return {
        "session_id": session_id,
        "count": len(logs),
        "data": [log.to_dict() for log in reversed(logs)],
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all available session IDs."""
    async with get_session() as session:
        stmt = (
            select(EmotionLog.session_id)
            .distinct()
            .order_by(desc(EmotionLog.session_id))
            .limit(50)
        )
        result = await session.execute(stmt)
        session_ids = [row[0] for row in result.all()]

    return {"sessions": session_ids}


# ── WebSocket Endpoint ───────────────────────────────────────

@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming."""
    # Each WS connection gets its own processor state
    ws_processor = AudioProcessor(
        sample_rate=settings.sample_rate,
        buffer_duration_s=settings.buffer_duration_s,
        ema_alpha=settings.ema_alpha,
        vad_threshold=settings.vad_threshold,
    )
    # Share the loaded models (thread-safe for inference)
    ws_processor._vad = processor._vad
    ws_processor._emotion = processor._emotion
    await websocket_stream(websocket, ws_processor)


# ── Main Entry Point ─────────────────────────────────────────

def start_gradio():
    """Launch Gradio UI in a background thread."""
    from app.ui.dashboard import create_dashboard
    demo = create_dashboard(processor)
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
        prevent_thread_lock=True,
    )


if __name__ == "__main__":
    # Start Gradio in background
    import threading
    gradio_thread = threading.Thread(target=start_gradio, daemon=True)
    gradio_thread.start()

    # Start FastAPI
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level="info",
    )
