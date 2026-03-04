import asyncio
from app.db.database import engine, init_db
from app.db.models import Base

async def reset_db():
    print("[RESET] Dropping all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("[RESET] All tables dropped ✓")
    
    print("[RESET] Recreating tables...")
    await init_db()
    print("[RESET] Tables recreated ✓")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(reset_db())
