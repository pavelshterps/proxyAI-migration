# crud.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Upload, User
from schemas import UserCreate, UploadCreate  # предположим, в проекте есть эти схемы

async def get_user_by_api_key(db: AsyncSession, api_key: str) -> User | None:
    """
    Находит пользователя по его API-ключу.
    """
    result = await db.execute(
        select(User).where(User.api_key == api_key)
    )
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """
    Создаёт нового пользователя.
    """
    user = User(**user_data.dict())
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def create_upload_record(
    db: AsyncSession,
    user_id: str,
    filename: str,
    **kwargs
) -> Upload:
    """
    Создаёт запись об загрузке: статус 'uploaded' по умолчанию.
    """
    upload = Upload(
        user_id=user_id,
        filename=filename,
        status='uploaded',
        created_at=kwargs.get('created_at'),
        expires_at=kwargs.get('expires_at'),
        **{k: v for k, v in kwargs.items() if k not in ('created_at', 'expires_at')}
    )
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    return upload

async def get_upload_for_user(
    db: AsyncSession,
    user_id: str,
    upload_id: str
) -> Upload | None:
    """
    Возвращает upload по его ID, принадлежащий заданному пользователю.
    """
    result = await db.execute(
        select(Upload)
        .where(Upload.id == upload_id, Upload.user_id == user_id)
    )
    return result.scalar_one_or_none()

async def update_upload_status(
    db: AsyncSession,
    upload_id: str,
    status: str
) -> Upload:
    """
    Обновляет поле status для существующего upload.
    """
    result = await db.execute(
        select(Upload).where(Upload.id == upload_id)
    )
    upload = result.scalar_one()
    upload.status = status
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    return upload