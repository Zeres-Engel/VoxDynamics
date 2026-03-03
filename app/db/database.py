# ============================================================
# VoxDynamics — Database Connection
# ============================================================
"""Async database engine, session factory, and initialization."""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from sqlalchemy import select

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from dotenv import load_dotenv

from app.db.models import Base

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://voxdynamics:voxdynamics_secret@localhost:5432/voxdynamics",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the engine connection pool."""
    await engine.dispose()


from app.db.models import Base, Session, EmotionLog

@asynccontextmanager
async def get_session():
    """Provide an async session scope."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise




async def get_session_by_uuid(session_uuid: str) -> Session | None:
    """Get a session by its string UUID."""
    async with get_session() as session:
        result = await session.execute(select(Session).where(Session.session_uuid == session_uuid))
        return result.scalar_one_or_none()


async def start_session(session_uuid: str) -> int:
    """Start a new session. Returns the internal session ID."""
    async with get_session() as session:
        new_session = Session(session_uuid=session_uuid)
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        return new_session.id


async def end_session(session_uuid: str) -> None:
    """Mark a session as ended."""
    async with get_session() as session:
        result = await session.execute(select(Session).where(Session.session_uuid == session_uuid))
        db_session = result.scalar_one_or_none()
        if db_session and db_session.end_time is None:
            db_session.end_time = datetime.utcnow()
            await session.commit()
