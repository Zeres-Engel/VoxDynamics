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
from app.db.models import EmotionLog, Session

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
    print(f"  Gradio: http://localhost:{settings.app_port}/")
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


@app.get("/api/emotions/{session_uuid}")
async def get_emotion_history(
    session_uuid: str,
    limit: int = Query(default=100, le=1000),
):
    """Retrieve emotion prediction history for a session."""
    async with get_session() as session:
        # Join EmotionLog with Session to filter by uuid
        stmt = (
            select(EmotionLog)
            .join(EmotionLog.session)
            .where(Session.session_uuid == session_uuid)
            .order_by(desc(EmotionLog.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        logs = result.scalars().all()

    return {
        "session_uuid": session_uuid,
        "count": len(logs),
        "data": [log.to_dict() for log in reversed(logs)],
    }


from sqlalchemy.sql import func

@app.get("/api/sessions")
async def list_sessions():
    """List all sessions with aggregated metadata for the sidebar."""
    async with get_session() as session:
        # Complex query to join, aggregate, and format
        stmt = (
            select(
                Session.session_uuid,
                Session.start_time,
                Session.end_time,
                func.count(EmotionLog.id).label("count"),
                func.avg(EmotionLog.arousal).label("avg_a"),
                func.avg(EmotionLog.dominance).label("avg_d"),
                func.avg(EmotionLog.valence).label("avg_v")
            )
            .outerjoin(EmotionLog, Session.id == EmotionLog.session_id)
            .group_by(Session.id)
            .order_by(desc(Session.start_time))
            .limit(50)
        )
        result = await session.execute(stmt)
        
        sessions_data = []
        for row in result.all():
            uuid, start_t, end_t, count, avg_a, avg_d, avg_v = row
            
            # Format time
            time_str = start_t.strftime("%H:%M:%S") if start_t else "Unknown"
            date_str = start_t.strftime("%m/%d") if start_t else ""
            
            # Calculate duration
            if start_t and end_t:
                duration_s = (end_t - start_t).total_seconds()
                dur_str = f"{duration_s:.0f}s"
            else:
                dur_str = "Active..."
                
            sessions_data.append({
                "UUID": uuid,
                "Time": f"{date_str} {time_str}",
                "Dur.": dur_str,
                "Points": count or 0,
                "A": round(avg_a or 0.0, 2),
                "D": round(avg_d or 0.0, 2),
                "V": round(avg_v or 0.0, 2),
            })

    return {"sessions": sessions_data}


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


# ── Gradio Integration ───────────────────────────────────────

from app.ui.dashboard import create_dashboard
import gradio as gr

# Create the Gradio dashboard
dashboard = create_dashboard(processor)

# Mount Gradio to FastAPI at root
app = gr.mount_gradio_app(app, dashboard, path="/")


# ── Main Entry Point ─────────────────────────────────────────

if __name__ == "__main__":
    pass

    # Start FastAPI
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level="info",
    )
