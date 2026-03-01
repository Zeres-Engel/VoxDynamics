# ============================================================
# VoxDynamics — Database Models
# ============================================================
"""SQLAlchemy ORM models for emotion logging."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Index
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class EmotionLog(Base):
    """Stores each emotion prediction event."""

    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Discrete label (mapped from dimensions)
    emotion_label = Column(String(32), nullable=False)

    # Dimensional values (0.0 – 1.0)
    arousal = Column(Float, nullable=False)
    dominance = Column(Float, nullable=False)
    valence = Column(Float, nullable=False)

    # Prediction confidence (0.0 – 1.0)
    confidence = Column(Float, nullable=False, default=0.0)

    # Inference latency in milliseconds
    latency_ms = Column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_emotion_logs_session_time", "session_id", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "emotion_label": self.emotion_label,
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "valence": round(self.valence, 4),
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"<EmotionLog(id={self.id}, session={self.session_id}, "
            f"emotion={self.emotion_label}, confidence={self.confidence:.2f})>"
        )
