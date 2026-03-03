# ============================================================
# VoxDynamics — WebSocket Handler
# ============================================================
"""WebSocket endpoint for real-time audio streaming from external clients."""

import json
import asyncio
import numpy as np
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from app.core.processor import AudioProcessor
from app.db.database import get_session
from app.db.models import EmotionLog
from app.config import settings


from app.db.database import get_session, get_session_by_uuid

async def log_emotion_to_db(result: dict) -> None:
    """Async-safe database logging (fire-and-forget)."""
    try:
        session_uuid = result.get("session_id")
        if not session_uuid:
            return

        db_session_obj = await get_session_by_uuid(session_uuid)
        if not db_session_obj:
            print(f"[DB LOG MSG] Ignoring log: No active session found for UUID {session_uuid[:8]}")
            return
            
        async with get_session() as session:
            log_entry = EmotionLog(
                session_id=db_session_obj.id,
                timestamp=datetime.utcnow(),
                emotion_label=result["emotion_label"],
                arousal=result["arousal"],
                dominance=result["dominance"],
                valence=result["valence"],
                confidence=result["confidence"],
                latency_ms=result["latency_ms"],
            )
            session.add(log_entry)
    except Exception as e:
        # Don't let DB errors crash the streaming loop
        print(f"[DB LOG ERROR] {e}")


async def websocket_stream(websocket: WebSocket, processor: AudioProcessor):
    """
    Handle a WebSocket connection for audio streaming.

    Protocol:
        Client sends: binary audio frames (PCM float32, 16kHz, mono)
        Server sends: JSON emotion results per frame
    """
    await websocket.accept()

    try:
        while True:
            # Receive raw audio bytes
            data = await websocket.receive_bytes()

            # Convert bytes → numpy float32
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            if len(audio_chunk) == 0:
                continue

            # Run through the processing pipeline
            result = processor.process_chunk(audio_chunk)

            # Send result back to client
            await websocket.send_json(result)

            # Log to database asynchronously (non-blocking)
            if result.get("is_speech", False):
                asyncio.create_task(log_emotion_to_db(result))

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected (session: {processor.session_id})")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await websocket.close(code=1011, reason=str(e))
