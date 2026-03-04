# ============================================================
# VoxDynamics — Database Models
# ============================================================
"""SQLAlchemy ORM models for emotion logging."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Index, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Session(Base):
    """Stores session metadata."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_uuid = Column(String(64), unique=True, nullable=False, index=True) # The string UUID used by the frontend
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)

    emotion_logs = relationship("EmotionLog", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_uuid": self.session_uuid,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, uuid={self.session_uuid})>"


class EmotionLog(Base):
    """Stores each emotion prediction event."""

    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Discrete label (mapped from dimensions)
    emotion_label = Column(String(32), nullable=False)

    # Dimensional values (0.0 – 1.0)
    arousal = Column(Float, nullable=False)
    dominance = Column(Float, nullable=False)
    valence = Column(Float, nullable=False)

    # Prediction confidence (0.0 – 1.0)
    confidence = Column(Float, nullable=False, default=0.0)

    # Logical Timing
    duration_s = Column(Float, nullable=True, default=0.5)
    offset_s = Column(Float, nullable=True, default=0.0)

    # Probability distribution (JSON string)
    scores_json = Column(String(512), nullable=True)

    # Inference latency in milliseconds
    latency_ms = Column(Float, nullable=False, default=0.0)

    session = relationship("Session", back_populates="emotion_logs")

    __table_args__ = (
        Index("ix_emotion_logs_session_time", "session_id", "timestamp"),
    )

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "emotion_label": self.emotion_label,
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "valence": round(self.valence, 4),
            "confidence": round(self.confidence, 4),
            "time_s": round(self.offset_s or 0.0, 2),
            "duration": round(self.duration_s or 0.5, 2),
            "scores": json.loads(self.scores_json) if self.scores_json else None,
            "latency_ms": round(self.latency_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"<EmotionLog(id={self.id}, session_id={self.session_id}, "
            f"emotion={self.emotion_label}, confidence={self.confidence:.2f})>"
        )
