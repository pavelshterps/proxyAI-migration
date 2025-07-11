# crud.py

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models import User, Upload


# === User CRUD for admin_routes and authentication ===

async def create_user(
    db: AsyncSession,
    username: str,
    api_key: str
) -> User:
    """
    Создать нового пользователя с заданным username и api_key.
    """
    user = User(username=username, api_key=api_key)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def delete_user(
    db: AsyncSession,
    user_id: int
) -> None:
    """
    Удалить пользователя по его ID.
    """
    stmt = select(User).where(User.id == user_id)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if user:
        await db.delete(user)
        await db.commit()


async def list_users(
    db: AsyncSession
) -> List[User]:
    """
    Получить список всех пользователей.
    """
    stmt = select(User)
    res = await db.execute(stmt)
    return res.scalars().all()


async def get_user_by_api_key(
    db: AsyncSession,
    api_key: str
) -> Optional[User]:
    """
    Найти пользователя по его API-ключу (для аутентификации).
    """
    stmt = select(User).where(User.api_key == api_key)
    res = await db.execute(stmt)
    return res.scalars().first()


# === Upload CRUD as before ===

async def create_upload_record(
    db: AsyncSession,
    user_id: int,
    upload_id: str,
    external_id: Optional[str] = None,
    callbacks: Optional[List[str]] = None
) -> Upload:
    """
    Создать запись об аудиозагрузке.
    """
    rec = Upload(
        user_id=user_id,
        upload_id=upload_id,
        external_id=external_id,
        callback_urls=callbacks or []
    )
    db.add(rec)
    await db.commit()
    return rec


async def get_upload_for_user(
    db: AsyncSession,
    user_id: int,
    upload_id: Optional[str] = None,
    external_id: Optional[str] = None
) -> Optional[Upload]:
    """
    Получить запись Upload по внутреннему или внешнему ID для данного пользователя.
    """
    stmt = select(Upload).where(Upload.user_id == user_id)
    if upload_id:
        stmt = stmt.where(Upload.upload_id == upload_id)
    if external_id:
        stmt = stmt.where(Upload.external_id == external_id)
    res = await db.execute(stmt)
    return res.scalars().first()


async def update_label_mapping(
    db: AsyncSession,
    user_id: int,
    upload_id: str,
    mapping: dict
) -> Optional[Upload]:
    """
    Обновить пользовательское отображение спикеров (label_mapping) для Upload.
    """
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    if not rec:
        return None
    rec.label_mapping = mapping
    await db.commit()
    return rec


async def get_label_mapping(
    db: AsyncSession,
    user_id: int,
    upload_id: str
) -> dict:
    """
    Получить текущее пользовательское отображение спикеров (или пустой dict).
    """
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    return rec.label_mapping or {}