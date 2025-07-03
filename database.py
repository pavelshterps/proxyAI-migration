from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config.settings import settings

# ваш движок
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

# ——————————————————————————————————————————————————————————
# Добавляем функцию для инициализации моделей (создания таблиц)
# ——————————————————————————————————————————————————————————
from models import Base  # импорт декларативного Base из models.py

async def init_models(engine):
    """
    Создаёт все таблицы на основе Base.metadata.
    Вызывается при старте FastAPI (lifespan или @on_event("startup")).
    """
    async with engine.begin() as conn:
        # синхронно выполняем create_all через run_sync
        await conn.run_sync(Base.metadata.create_all)