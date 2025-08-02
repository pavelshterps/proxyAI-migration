import asyncio
import time
import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from models import Base  # декларативный Base с вашими моделями

log = logging.getLogger(__name__)

# Асинхронный движок с pre_ping-проверкой, чтобы уменьшить влияние "протухших" соединений.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

# Фабрика асинхронных сессий
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    future=True,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def wait_for_db(host: str, port: int = 5432, timeout: int = 30):
    """
    Ждём, пока DNS разрешится и можно будет открыть TCP-соединение к БД.
    """
    deadline = time.time() + timeout
    loop = asyncio.get_running_loop()
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for DB at {host}:{port}")
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            if hasattr(writer, "wait_closed"):
                await writer.wait_closed()
            log.debug("Database socket reachable", host=host, port=port)
            return
        except Exception as e:
            log.warning("DB not ready yet, retrying", error=str(e), host=host, port=port)
            await asyncio.sleep(1)


async def init_models(engine: AsyncEngine, timeout: int = 30):
    """
    Создаёт все таблицы на основе Base.metadata, с повторными попытками при временных сбоях.
    """
    deadline = time.time() + timeout
    while True:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            log.info("Database models initialized")
            break
        except Exception as e:
            if time.time() > deadline:
                log.error("Failed to initialize models after retries", exc_info=True)
                raise
            log.warning("Retrying model initialization", error=str(e))
            await asyncio.sleep(1)