# ============================================================
# VoxDynamics — FastAPI Main Application
# ============================================================
"""
Entry point for the VoxDynamics backend.

Starts:
  - FastAPI server on port 8000 (REST + WebSocket)
  - HTML5 UI served at /
  - PostgreSQL connection on startup
"""

import io
import uuid
import soundfile as sf
import json
import numpy as np
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import select, desc
from sqlalchemy.sql import func

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
    print("[STARTUP] High-Accuracy CNN Emotion model loaded ✓")

    print("=" * 60)
    print("  VoxDynamics — Ready!")
    print(f"  API:    http://localhost:{settings.app_port}")
    print(f"  UI:     http://localhost:{settings.app_port}/")
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
    version="2.0.0",
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
        "version": "2.0.0",
    }


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accept an audio file upload, run emotion analysis on all segments,
    log the results to DB, and return structured results.
    """
    if not processor.models_loaded:
        raise HTTPException(status_code=503, detail="AI models are still loading. Please wait.")

    # Read uploaded file
    content = await file.read()
    try:
        waveform, sr = sf.read(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")

    # Convert to mono float32
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    # Normalize if int-encoded
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    # Run analysis
    results = processor.process_file(waveform, sample_rate=sr, window_s=2.5, hop_s=0.5)
    speech_results = [r for r in results if r.get("is_speech")]

    if not speech_results:
        raise HTTPException(status_code=422, detail="No speech detected in audio file.")

    # ── Build Summary ──────────────────────────────────────
    labels = [r["emotion_label"] for r in speech_results]
    dominant = max(set(labels), key=labels.count)
    dominant_emoji = next(r["emoji"] for r in speech_results if r["emotion_label"] == dominant)

    # Calculate secondary dimension averages
    avg_a = float(np.mean([r["arousal"] for r in speech_results])) if speech_results else 0.5
    avg_d = float(np.mean([r["dominance"] for r in speech_results])) if speech_results else 0.5
    avg_v = float(np.mean([r["valence"] for r in speech_results])) if speech_results else 0.5

    # Calculate average emotion probabilities across the session (for Radar)
    avg_scores = {}
    if speech_results and "scores" in speech_results[0]:
        all_emotions = speech_results[0]["scores"].keys()
        for emo in all_emotions:
            avg_scores[emo] = float(np.mean([r["scores"][emo] for r in speech_results]))
    
    avg_conf = float(np.mean([r["confidence"] for r in speech_results])) if speech_results else 0.0
    audio_duration_s = len(waveform) / sr

    summary = {
        "dominant_emotion": dominant,
        "dominant_emoji": dominant_emoji,
        "avg_arousal": round(avg_a, 4),
        "avg_dominance": round(avg_d, 4),
        "avg_valence": round(avg_v, 4),
        "avg_scores": avg_scores,
        "avg_confidence": round(avg_conf, 4),
        "audio_duration_s": round(audio_duration_s, 2),
        "speech_segments": len(speech_results),
    }

    # ── Log to DB ──────────────────────────────────────────
    session_uuid = str(uuid.uuid4())
    try:
        async with get_session() as db_session:
            new_sess = Session(session_uuid=session_uuid, start_time=datetime.utcnow())
            db_session.add(new_sess)
            await db_session.flush()

            for r in speech_results:
                log = EmotionLog(
                    session_id=new_sess.id,
                    emotion_label=r["emotion_label"],
                    arousal=r["arousal"],
                    dominance=r["dominance"],
                    valence=r["valence"],
                    confidence=r.get("confidence", 0.0),
                    duration_s=r.get("duration_s", 0.5),
                    offset_s=r.get("time_s", 0.0),
                    scores_json=json.dumps(r.get("scores", {})),
                    latency_ms=0.0,
                )
                db_session.add(log)

            new_sess.end_time = datetime.utcnow()
            await db_session.commit()
    except Exception as e:
        print(f"[DB] Warning: could not save session: {e}")

    return {
        "session_uuid": session_uuid,
        "summary": summary,
        "segments": results,  # Include all (speech + non-speech so FE can mark silence)
    }


@app.get("/api/emotions/{session_uuid}")
async def get_emotion_history(
    session_uuid: str,
    limit: int = Query(default=100, le=1000),
):
    """Retrieve emotion prediction history for a session."""
    async with get_session() as session:
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


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions with aggregated metadata."""
    async with get_session() as session:
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
            uuid_val, start_t, end_t, count, avg_a, avg_d, avg_v = row

            time_str = start_t.strftime("%H:%M:%S") if start_t else "Unknown"
            date_str = start_t.strftime("%m/%d") if start_t else ""

            if start_t and end_t:
                duration_s = (end_t - start_t).total_seconds()
                dur_str = f"{duration_s:.0f}s"
            else:
                dur_str = "Active..."

            sessions_data.append({
                "UUID": uuid_val,
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
    ws_processor = AudioProcessor(
        sample_rate=settings.sample_rate,
        buffer_duration_s=settings.buffer_duration_s,
        ema_alpha=settings.ema_alpha,
        vad_threshold=settings.vad_threshold,
    )
    ws_processor._vad = processor._vad
    ws_processor._emotion = processor._emotion
    await websocket_stream(websocket, ws_processor)


# ── Frontend Serving ─────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend", "template")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_ui():
    """Serve the HTML5 UI."""
    return FileResponse(os.path.join(TEMPLATE_DIR, "index.html"))


# ── Main Entry Point ─────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level="info",
    )
