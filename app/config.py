# ============================================================
# VoxDynamics — Application Settings
# ============================================================
"""Centralized configuration via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # Database
    postgres_user: str = "voxdynamics"
    postgres_password: str = "voxdynamics_secret"
    postgres_db: str = "voxdynamics"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    database_url: str = (
        "postgresql+asyncpg://voxdynamics:voxdynamics_secret@postgres:5432/voxdynamics"
    )

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    gradio_port: int = 7860

    # Audio
    sample_rate: int = 16000
    chunk_duration_ms: int = 300          # ms per chunk from client (reduced from 500ms)
    buffer_duration_s: float = 2.0        # sliding window size (reduced from 3.0s)
    ema_alpha: float = 0.3                # smoothing factor

    # VAD
    vad_threshold: float = 0.5

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
