# database.py

import asyncio
import time

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from models import Base  # декларативный Base с вашими моделями

# создаём асинхронный движок
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

# фабрика асинхронных сессий
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    future=True,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_models(engine: AsyncEngine, timeout: int = 30):
    """
    Создаёт все таблицы на основе Base.metadata,
    повторяя попытки подключения к БД в течение `timeout` секунд.
    """
    deadline = time.time() + timeout
    while True:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            break
        except Exception:
            if time.time() > deadline:
                raise
            await asyncio.sleep(1)